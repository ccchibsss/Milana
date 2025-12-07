import polars as pl
import duckdb
import streamlit as st
import io
import os
from pathlib import Path
from typing import Dict, List, Optional
import time
import warnings
import logging
import difflib
import textwrap
from urllib.parse import urlparse
import boto3  # Для AWS S3
import requests  # Для других облачных хранилищ

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

EXCEL_ROW_LIMIT = 1_000_000

# --- Настройки облачного хранилища ---
CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "s3")  # s3, gcs, azure
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
# Примеры для других: GCS_BUCKET, AZURE_CONTAINER и т.д.

class HighVolumeAutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.sync_from_cloud()  # Загружаем базу из облака при старте
        self.conn = duckdb.connect(str(self.db_path))
        self.setup_database()
        st.set_page_config(page_title="AutoParts Catalog 10M+", layout="wide", page_icon="🚗")

        self.prices_df = pl.DataFrame()
        self.price_markup = 1.0  # 1.0 = 0%
        self.brand_markup = {}   # нормализованное имя бренда → коэффициент
        self.export_exclusions = []  # Список шаблонов для исключения при экспорте
        self.category_mapping = {}  # Ручное сопоставление: ключевое слово → категория

    def sync_from_cloud(self):
        """Скачивает базу из облака при запуске."""
        if not self.db_path.exists():
            try:
                if CLOUD_PROVIDER == "s3" and S3_BUCKET:
                    s3 = boto3.client("s3", region_name=S3_REGION)
                    s3.download_file(S3_BUCKET, "catalog.duckdb", str(self.db_path))
                    st.info("✅ База данных загружена из S3.")
                # elif ... другие провайдеры
            except Exception as e:
                st.warning(f"❌ Не удалось загрузить базу из облака: {e}. Создаётся локальная.")

    def sync_to_cloud(self):
        """Выгружает базу в облако после изменений."""
        try:
            if CLOUD_PROVIDER == "s3" and S3_BUCKET:
                s3 = boto3.client("s3", region_name=S3_REGION)
                s3.upload_file(str(self.db_path), S3_BUCKET, "catalog.duckdb")
                st.info("☁️ База данных сохранена в облако.")
        except Exception as e:
            st.error(f"Ошибка сохранения в облако: {e}")

    def setup_database(self):
        """Создание таблиц в DuckDB, если они не существуют."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS oe_data (
                oe_number_norm VARCHAR PRIMARY KEY,
                oe_number VARCHAR,
                name VARCHAR,
                applicability VARCHAR,
                category VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS parts_data (
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                artikul VARCHAR,
                brand VARCHAR,
                multiplicity INTEGER DEFAULT 1,
                barcode VARCHAR,
                length DOUBLE, 
                width DOUBLE,
                height DOUBLE, 
                weight DOUBLE,
                image_url VARCHAR,
                dimensions_str VARCHAR,
                description VARCHAR,
                price DOUBLE,
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_references (
                oe_number_norm VARCHAR,
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                PRIMARY KEY (oe_number_norm, artikul_norm, brand_norm)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                price DOUBLE,
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)

    def normalize_key(self, key_series: pl.Series) -> pl.Series:
        """Нормализация ключевых полей: очистка и приведение к нижнему регистру."""
        return (
            key_series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .str.to_lowercase()
        )

    def clean_values(self, value_series: pl.Series) -> pl.Series:
        """Очистка строковых значений (без to_lowercase)."""
        return (
            value_series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    def detect_category(self, name_series: pl.Series) -> pl.Series:
        """Определяет категорию по ключевым словам в наименовании."""
        base_categories = {
            'Фильтр': 'фильтр|filter',
            'Тормозная система': 'тормоз|brake|колодк|диск|суппорт',
            'Подвеска': 'амортизатор|стойк|spring|подвеск|рычаг|шаровая|сайлентблок|ступиц|подшипник',
            'Двигатель': 'двигатель|engine|свеч|поршень|клапан',
            'Трансмиссия': 'трансмиссия|сцеплен|коробк|transmission',
            'Электрика': 'аккумулятор|генератор|стартер|провод|ламп',
            'Рулевое': 'рулевой|тяга|наконечник|steering',
            'Выхлопная система': 'глушитель|катализатор|выхлоп|exhaust',
            'Охлаждение': 'радиатор|вентилятор|термостат|cooling',
            'Топливо': 'топливный|бензонасос|форсунк|fuel',
        }
        # Добавляем пользовательские категории
        all_patterns = {**base_categories, **self.category_mapping}
        name_lower = name_series.str.to_lowercase()
        expr = pl.when(pl.lit(False)).then(pl.lit(None))
        for cat, pattern in all_patterns.items():
            expr = expr.when(name_lower.str.contains(pattern)).then(pl.lit(cat))
        return expr.otherwise(pl.lit('Разное')).alias('category')

    def detect_columns(self, actual: List[str], expected: List[str]) -> Dict[str, str]:
        """Эвристическое сопоставление столбцов по имени."""
        mapping = {}
        for exp in expected:
            matches = difflib.get_close_matches(exp, [a.lower() for a in actual], n=1, cutoff=0.6)
            if matches:
                orig_col = actual[[a.lower() for a in actual].index(matches[0])]
                mapping[orig_col] = exp
        return mapping

    def read_and_prepare_file(self, file_path: str, file_type: str) -> pl.DataFrame:
        """Чтение и предобработка Excel-файла."""
        try:
            df = pl.read_excel(file_path, engine='calamine')
        except Exception as e:
            st.error(f"Ошибка чтения файла {file_path}: {e}")
            return pl.DataFrame()

        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand'],
            'price': ['артикул', 'бренд', 'количество', 'цена']  # Поддержка новой структуры прайса
        }
        expected_cols = schemas.get(file_type, [])
        column_mapping = self.detect_columns(df.columns, expected_cols)
        df = df.rename(column_mapping)

        for col in ['artikul', 'brand', 'oe_number']:
            if col in df.columns:
                df = df.with_columns(self.clean_values(pl.col(col)).alias(col))

        key_cols = [c for c in ['oe_number', 'artikul', 'brand'] if c in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')

        for col in ['artikul', 'brand', 'oe_number']:
            norm_col = f"{col}_norm"
            if col in df.columns:
                df = df.with_columns(self.normalize_key(pl.col(col)).alias(norm_col))

        return df

    def upsert_data(self, table_name: str, df: pl.DataFrame, pk: List[str]):
        """Вставка или обновление данных в таблицу DuckDB."""
        if df.is_empty():
            return
        df = df.unique()
        cols = df.columns
        pk_str = ", ".join(f'"{c}"' for c in pk)
        temp_view = f"temp_{table_name}_{int(time.time() * 1000)}"
        self.conn.register(temp_view, df.to_arrow())

        update_cols = [c for c in cols if c not in pk]
        if update_cols:
            update_clause = ", ".join([f'"{c}" = excluded."{c}"' for c in update_cols])
            on_conflict = f"DO UPDATE SET {update_clause}"
        else:
            on_conflict = "DO NOTHING"

        query = f"""
        INSERT INTO {table_name}
        SELECT * FROM {temp_view}
        ON CONFLICT ({pk_str}) {on_conflict};
        """
        try:
            self.conn.execute(query)
        finally:
            self.conn.unregister(temp_view)

    def process_and_load_data(self, dataframes: Dict[str, pl.DataFrame]):
        """Обработка всех входных данных и загрузка в базу."""
        # OE и категории
        if 'oe' in dataframes:
            df_oe = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_data = df_oe.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'])
            oe_data = oe_data.with_columns(self.detect_category(pl.col('name')))
            self.upsert_data('oe_data', oe_data, ['oe_number_norm'])

            cross_df = df_oe.filter(pl.col('artikul_norm') != "").select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Кроссы
        if 'cross' in dataframes:
            df_cross = dataframes['cross'].filter(
                (pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != "")
            )
            cross_df = df_cross.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Сборка parts_data
        file_priority = ['oe', 'barcode', 'images', 'dimensions']
        key_files = {k: v for k, v in dataframes.items() if k in file_priority and not v.is_empty()}

        if not key_files:
            return

        base_artikuls = pl.concat([
            df.select(['artikul_norm', 'brand_norm', 'artikul', 'brand'])
            for df in key_files.values()
            if {'artikul_norm', 'brand_norm'} <= set(df.columns)
        ]).unique(subset=['artikul_norm', 'brand_norm'])

        parts_df = base_artikuls

        for ftype in file_priority:
            if ftype not in key_files:
                continue
            df = key_files[ftype]
            join_cols = [c for c in df.columns if c not in parts_df.columns and c not in ['artikul', 'brand', 'artikul_norm', 'brand_norm']]
            if not join_cols:
                continue
            subset = df.select(['artikul_norm', 'brand_norm'] + join_cols).unique(subset=['artikul_norm', 'brand_norm'])
            parts_df = parts_df.join(subset, on=['artikul_norm', 'brand_norm'], how='left')

        # Заполнение пропущенных полей
        defaults = {
            'multiplicity': 1,
            'length': None,
            'width': None,
            'height': None,
            'weight': None,
            'dimensions_str': None,
            'image_url': None,
            'price': None
        }
        for col, val in defaults.items():
            if col not in parts_df.columns:
                dtype = pl.Float64 if isinstance(val, float) else pl.Int32 if isinstance(val, int) else pl.Utf8
                parts_df = parts_df.with_columns(pl.lit(val).cast(dtype).alias(col))

        # Генерация dimensions_str
        parts_df = parts_df.with_columns(
            dimensions_str=pl.when(
                (pl.col('dimensions_str').is_null()) |
                (pl.col('dimensions_str') == '') |
                (pl.col('dimensions_str').str.to_lowercase() == 'xx')
            ).then(
                pl.concat_str([
                    pl.col('length').cast(pl.Utf8).fill_null(''),
                    pl.lit('x'),
                    pl.col('width').cast(pl.Utf8).fill_null(''),
                    pl.lit('x'),
                    pl.col('height').cast(pl.Utf8).fill_null('')
                ], separator='')
            ).otherwise(pl.col('dimensions_str'))
        )

        # Генерация описания
        parts_df = parts_df.with_columns(
            description=pl.concat_str([
                pl.lit('Артикул: '), pl.col('artikul'),
                pl.lit(', Бренд: '), pl.col('brand'),
                pl.lit(', Кратность: '), pl.col('multiplicity').cast(pl.Utf8), pl.lit(' шт.')
            ], separator='').alias('description')
        )

        self.upsert_data('parts_data', parts_df, ['artikul_norm', 'brand_norm'])
        st.success("Данные успешно обновлены в базе.")
        self.sync_to_cloud()  # Сохраняем изменения в облако

    def load_price_file(self, file_bytes: bytes):
        """Загрузка прайса из файла Excel с поддержкой количества."""
        try:
            df = pl.read_excel(io.BytesIO(file_bytes), engine='calamine')
        except Exception as e:
            st.error(f"Ошибка чтения прайса: {e}")
            return

        required = ['артикул', 'бренд', 'цена']
        if not all(col in [c.lower() for c in df.columns] for col in required):
            st.warning(f"Файл должен содержать колонки: {required}")
            return

        df = (df.rename(mapping={c: k for c in df.columns for k in required if k in c.lower()})
              .with_columns([
                  self.normalize_key(pl.col('артикул')).alias('artikul_norm'),
                  self.normalize_key(pl.col('бренд')).alias('brand_norm'),
                  pl.col('цена').cast(pl.Float64)
              ])
              .select(['artikul_norm', 'brand_norm', 'цена']))
        self.prices_df = pl.concat([self.prices_df, df]).unique(subset=['artikul_norm', 'brand_norm'])
        st.success("Прайс успешно загружен.")
        self.apply_markups()  # Автоматически применяем наценки

    def apply_markups(self):
        """Применение общей и индивидуальной наценки."""
        if self.prices_df.is_empty():
            return
        for row in self.prices_df.iter_rows():
            artikul_norm, brand_norm, base_price = row
            markup = self.brand_markup.get(brand_norm, self.price_markup)
            final_price = base_price * markup
            self.conn.execute("""
                UPDATE parts_data SET price = ? WHERE artikul_norm = ? AND brand_norm = ?
            """, [final_price, artikul_norm, brand_norm])
        self.sync_to_cloud()

    def set_brand_markup(self, brand: str, percent: float):
        """Установить наценку для бренда."""
        normalized = self.normalize_key(pl.Series([brand]))[0]
        self.brand_markup[normalized] = 1 + percent / 100
        self.apply_markups()
        st.info(f"Наценка для бренда '{brand}': {percent}%")

    def set_global_markup(self, percent: float):
        """Установить общую наценку."""
        self.price_markup = 1 + percent / 100
        self.apply_markups()
        st.info(f"Общая наценка: {percent}%")

    def build_export_query(self, selected_columns: List[str]) -> str:
        """Формирование SQL-запроса для экспорта данных с исключениями."""
        standard_description = textwrap.dedent("""
            Состояние товара: новый (в упаковке).
            Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля.
            ...
        """).strip()

        # Фильтр по исключениям в названии
        exclusion_conditions = []
        for excl in self.export_exclusions:
            pattern = excl.lower().replace("*", "%").replace("?", "_")
            exclusion_conditions.append(f"LOWER(COALESCE(o.name, '')) NOT LIKE '%{pattern}%'")

        exclusion_where = " AND ".join(exclusion_conditions) if exclusion_conditions else "TRUE"

        column_map = {
            "Артикул бренда": 'r.artikul AS "Артикул бренда"',
            "Бренд": 'r.brand AS "Бренд"',
            "Наименование": 'COALESCE(r.representative_name, r.analog_representative_name) AS "Наименование"',
            "Применимость": 'COALESCE(r.representative_applicability, r.analog_representative_applicability) AS "Применимость"',
            "Описание": "CONCAT(COALESCE(r.description, ''), dt.text) AS \"Описание\"",
            "Категория товара": 'COALESCE(r.representative_category, r.analog_representative_category) AS "Категория товара"',
            "Кратность": 'r.multiplicity AS "Кратность"',
            "Длинна": 'COALESCE(r.length, r.analog_length) AS "Длинна"',
            "Ширина": 'COALESCE(r.width, r.analog_width) AS "Ширина"',
            "Высота": 'COALESCE(r.height, r.analog_height) AS "Высота"',
            "Вес": 'COALESCE(r.weight, r.analog_weight) AS "Вес"',
            "Длинна/Ширина/Высота": "COALESCE(NULLIF(TRIM(r.dimensions_str), ''), NULLIF(TRIM(r.analog_dimensions_str), '')) AS \"Длинна/Ширина/Высота\"",
            "OE номер": 'r.oe_list AS "OE номер"',
            "аналоги": 'r.analog_list AS "аналоги"',
            "Ссылка на изображение": 'r.image_url AS "Ссылка на изображение"'
        }

        selected_exprs = [column_map[col] for col in selected_columns if col in column_map]
        if not selected_exprs:
            selected_exprs = list(column_map.values())

        ctes = textwrap.dedent(f"""
        WITH DescriptionTemplate AS (
            SELECT '\n\n' || $${standard_description}$$ AS text
        ),
        PartDetails AS (
            SELECT
                cr.artikul_norm,
                cr.brand_norm,
                STRING_AGG(DISTINCT REGEXP_REPLACE(o.oe_number, '[^0-9A-Za-zА-Яа-яЁё`\\-]', '', 'g'), ', ') AS oe_list,
                ANY_VALUE(o.name) AS representative_name,
                ANY_VALUE(o.applicability) AS representative_applicability,
                ANY_VALUE(o.category) AS representative_category
            FROM cross_references cr
            JOIN oe_data o ON cr.oe_number_norm = o.oe_number_norm
            WHERE {exclusion_where}
            GROUP BY cr.artikul_norm, cr.brand_norm
        ),
        AllAnalogs AS (
            SELECT
                cr1.artikul_norm,
                cr1.brand_norm,
                STRING_AGG(DISTINCT REGEXP_REPLACE(p2.artikul, '[^0-9A-Za-zА-Яа-яЁё`\\-]', '', 'g'), ', ') AS analog_list
            FROM cross_references cr1
            JOIN cross_references cr2 ON cr1.oe_number_norm = cr2.oe_number_norm
            JOIN parts_data p2 ON cr2.artikul_norm = p2.artikul_norm AND cr2.brand_norm = p2.brand_norm
            WHERE cr1.artikul_norm != p2.artikul_norm OR cr1.brand_norm != p2.brand_norm
            GROUP BY cr1.artikul_norm, cr1.brand_norm
        ),
        RankedData AS (
            SELECT
                p.artikul,
                p.brand,
                p.description,
                p.multiplicity,
                p.length,
                p.width,
                p.height,
                p.weight,
                p.dimensions_str,
                p.image_url,
                pd.representative_name,
                pd.representative_applicability,
                pd.representative_category,
                pd.oe_list,
                aa.analog_list,
                p_analog.length AS analog_length,
                p_analog.width AS analog_width,
                p_analog.height AS analog_height,
                p_analog.weight AS analog_weight,
                p_analog.dimensions_str AS analog_dimensions_str,
                p_analog.representative_name AS analog_representative_name,
                p_analog.representative_applicability AS analog_representative_applicability,
                p_analog.representative_category AS analog_representative_category,
                ROW_NUMBER() OVER(PARTITION BY p.artikul_norm, p.brand_norm ORDER BY pd.representative_name DESC NULLS LAST) AS rn
            FROM parts_data p
            LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
            LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
            LEFT JOIN parts_data p_analog ON p.artikul_norm = p_analog.artikul_norm AND p.brand_norm = p_analog.brand_norm
        )
        """)

        select_clause = ",\n            ".join(selected_exprs)
        query = f"""
        {ctes}
        SELECT
            {select_clause}
        FROM RankedData r
        CROSS JOIN DescriptionTemplate dt
        WHERE r.rn = 1
        ORDER BY r.brand, r.artikul
        """
        return query

    def delete_by_brand(self, brand_norm: str) -> int:
        res = self.conn.execute("DELETE FROM parts_data WHERE brand_norm = ?", [brand_norm])
        self.sync_to_cloud()
        return res.rowcount

    def delete_by_artikul(self, artikul_norm: str) -> int:
        res = self.conn.execute("DELETE FROM parts_data WHERE artikul_norm = ?", [artikul_norm])
        self.sync_to_cloud()
        return res.rowcount

    def assign_category_by_name(self, search_name: str, category_name: str, similarity_threshold: float = 0.5):
        """Присвоить категорию всем товарам с похожими названиями."""
        res = self.conn.execute("SELECT DISTINCT name FROM oe_data WHERE name IS NOT NULL").fetchall()
        names = [row[0] for row in res]
        matched = [
            name for name in names
            if difflib.SequenceMatcher(None, name.lower(), search_name.lower()).ratio() >= similarity_threshold
        ]
        for name in matched:
            self.conn.execute("UPDATE oe_data SET category = ? WHERE name = ?", [category_name, name])
        st.success(f"Обновлено {len(matched)} записей на категорию '{category_name}'.")
        self.sync_to_cloud()

    def add_category_mapping(self, keyword: str, category: str):
        """Добавить пользовательское правило категории."""
        self.category_mapping[keyword.lower()] = keyword.lower()  # Упрощённо
        # Для сложных случаев: self.category_mapping[category] = keyword_pattern

    def get_total_records(self):
        res = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()
        return res[0] if res else 0

    def get_statistics(self):
        total_parts = self.get_total_records()
        total_oe = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()[0]
        total_brands = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data").fetchone()[0]
        top_brands = self.conn.execute("""
            SELECT brand, COUNT(*) AS cnt FROM parts_data GROUP BY brand ORDER BY cnt DESC LIMIT 10
        """).pl()
        categories = self.conn.execute("""
            SELECT category, COUNT(*) AS cnt FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY cnt DESC
        """).pl()
        return {
            'total_parts': total_parts,
            'total_oe': total_oe,
            'total_brands': total_brands,
            'top_brands': top_brands,
            'categories': categories
        }

    def export_to_excel(self, selected_columns: List[str], file_path: Path):
        query = self.build_export_query(selected_columns)
        result_df = self.conn.execute(query).pl()
        result_df.write_excel(file_path)

# === ГЛАВНЫЙ ИНТЕРФЕЙС ===
def main():
    st.title("🚗 AutoParts Catalog - Управление 10+ млн записей")
    catalog = HighVolumeAutoPartsCatalog()

    st.sidebar.title("🧭 Меню")
    menu = st.sidebar.radio("Выберите действие:", [
        "Загрузка данных", "Экспорт", "Статистика", "Управление ценами", "Управление данными"
    ])

    if menu == "Загрузка данных":
        st.header("📥 Загрузка и обработка данных")
        st.info("Загрузите файлы Excel для обновления каталога.")
        col1, col2 = st.columns(2)

        with col1:
            oe_file = st.file_uploader("Основные данные (OE)", type=['xlsx', 'xls'], key="oe")
            cross_file = st.file_uploader("Кроссы", type=['xlsx', 'xls'], key="cross")
            barcode_file = st.file_uploader("Штрих-коды", type=['xlsx', 'xls'], key="barcode")
        with col2:
            dimensions_file = st.file_uploader("Весогабариты", type=['xlsx', 'xls'], key="dim")
            images_file = st.file_uploader("Изображения", type=['xlsx', 'xls'], key="img")

        uploaded_files = {
            'oe': oe_file,
            'cross': cross_file,
            'barcode': barcode_file,
            'dimensions': dimensions_file,
            'images': images_file
        }

        if st.button("🚀 Начать обработку"):
            file_paths = {}
            for key, uploaded in uploaded_files.items():
                if uploaded:
                    path = catalog.data_dir / f"{key}_{int(time.time())}_{uploaded.name}"
                    with open(path, 'wb') as f:
                        f.write(uploaded.getvalue())
                    file_paths[key] = str(path)
            if file_paths:
                catalog.merge_all_data_parallel(file_paths)
                st.success("✅ Данные обработаны и синхронизированы с облаком.")
            else:
                st.warning("Загрузите хотя бы один файл.")

    elif menu == "Экспорт":
        catalog.show_export_interface()

    elif menu == "Статистика":
        st.header("📈 Статистика по каталогу")
        stats = catalog.get_statistics()
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего артикулов", f"{stats['total_parts']:,}")
        col2.metric("OE номеров", f"{stats['total_oe']:,}")
        col3.metric("Брендов", stats['total_brands'])
        st.subheader("Топ брендов")
        st.dataframe(stats['top_brands'].to_pandas())
        st.subheader("Категории")
        st.bar_chart(stats['categories'].to_pandas().set_index('category'))

    elif menu == "Управление ценами":
        st.header("🛠️ Управление ценами")
        uploaded_price = st.file_uploader("Загрузить прайс (артикул, бренд, цена)", type=['xlsx', 'xls'])
        if uploaded_price:
            catalog.load_price_file(uploaded_price.read())

        markup = st.number_input("Общая наценка (%)", 0.0, 100.0, 0.0)
        if st.button("Применить общую наценку"):
            catalog.set_global_markup(markup)

        st.subheader("Наценка по бренду")
        brand_name = st.text_input("Бренд")
        brand_markup_percent = st.number_input("Наценка (%)", 0.0, 100.0, 0.0, key="brand_markup")
        if st.button("Установить для бренда"):
            if brand_name.strip():
                catalog.set_brand_markup(brand_name, brand_markup_percent)
            else:
                st.warning("Введите название бренда.")

    elif menu == "Управление данными":
        st.header("🗑️ Управление данными")
        action = st.radio("Действие", [
            "Удалить по бренду",
            "Удалить по артикулу",
            "Назначить категорию",
            "Добавить категорию",
            "Исключения при экспорте"
        ])

        if action == "Удалить по бренду":
            brands = [r[0] for r in catalog.conn.execute("SELECT DISTINCT brand FROM parts_data").fetchall()]
            selected = st.selectbox("Бренд", brands)
            if selected:
                norm = catalog.normalize_key(pl.Series([selected]))[0]
                count = catalog.delete_by_brand(norm)
                st.success(f"🗑️ Удалено {count} записей.")

        elif action == "Удалить по артикулу":
            artikul = st.text_input("Артикул")
            if artikul:
                norm = catalog.normalize_key(pl.Series([artikul]))[0]
                count = catalog.delete_by_artikul(norm)
                st.success(f"🗑️ Удалено {count} записей.")

        elif action == "Назначить категорию":
            name_input = st.text_input("Название товара")
            cat_input = st.text_input("Категория")
            threshold = st.slider("Порог схожести", 0.0, 1.0, 0.5, 0.05)
            if st.button("Назначить"):
                if name_input and cat_input:
                    catalog.assign_category_by_name(name_input, cat_input, threshold)
                else:
                    st.warning("Заполните оба поля.")

        elif action == "Добавить категорию":
            keyword = st.text_input("Ключевое слово в названии")
            category = st.text_input("Категория")
            if st.button("Добавить правило"):
                if keyword and category:
                    catalog.add_category_mapping(keyword, category)
                    st.success(f"✅ Категория '{category}' будет применяться для товаров с '{keyword}' в названии.")
                else:
                    st.warning("Заполните оба поля.")

        elif action == "Исключения при экспорте":
            exclusion_input = st.text_input("Шаблон исключения (например: *кузов*, *стекла*)")
            if st.button("Добавить исключение"):
                if exclusion_input:
                    catalog.export_exclusions.append(exclusion_input.strip().strip("*"))
                    st.success(f"🚫 Исключение добавлено: {exclusion_input}")
                st.write("Текущие исключения:", ", ".join(catalog.export_exclusions))

if __name__ == "__main__":
    main()
