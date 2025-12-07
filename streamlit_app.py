import platform
import sys
import polars as pl
import duckdb
import streamlit as st
import os
import time
import logging
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import json
import pandas as pd  # Нужен для xlsxwriter
import fastexcel  # Чтение Excel
from tqdm import tqdm  # Прогресс-бар
import psutil  # Мониторинг памяти

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
EXCEL_ROW_LIMIT = 1_000_000

class HighVolumeAutoPartsCatalog:
    
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Конфигурация облачного хранилища
        self.cloud_config = self.load_cloud_config()
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(database=str(self.db_path))
        self.setup_database()
        
        # Инициализация новых компонентов
        self.price_rules = self.load_price_rules()
        self.exclusion_rules = self.load_exclusion_rules()
        self.category_mapping = self.load_category_mapping()
        
        st.set_page_config(
            page_title="AutoParts Catalog 10M+", 
            layout="wide",
            page_icon="🚗"
        )
    
    def load_cloud_config(self) -> Dict[str, Any]:
        config_path = self.data_dir / "cloud_config.json"
        default_config = {
            "enabled": False, "provider": "s3", "bucket": "", "region": "", 
            "sync_interval": 3600, "last_sync": 0
        }
        if config_path.exists():
            try:
                return json.loads(config_path.read_text())
            except:
                return default_config
        else:
            config_path.write_text(json.dumps(default_config, indent=2))
            return default_config
    
    def save_cloud_config(self):
        config_path = self.data_dir / "cloud_config.json"
        self.cloud_config["last_sync"] = int(time.time())
        config_path.write_text(json.dumps(self.cloud_config, indent=2))
    
    def load_price_rules(self) -> Dict[str, Any]:
        price_rules_path = self.data_dir / "price_rules.json"
        default_rules = {
            "global_markup": 0.2, "brand_markups": {}, "min_price": 0.0, "max_price": 99999.0
        }
        if price_rules_path.exists():
            try:
                return json.loads(price_rules_path.read_text())
            except:
                return default_rules
        else:
            price_rules_path.write_text(json.dumps(default_rules, indent=2))
            return default_rules
    
    def save_price_rules(self):
        price_rules_path = self.data_dir / "price_rules.json"
        price_rules_path.write_text(json.dumps(self.price_rules, indent=2))
    
    def load_exclusion_rules(self) -> List[str]:
        exclusion_path = self.data_dir / "exclusion_rules.txt"
        if exclusion_path.exists():
            try:
                return [line.strip() for line in exclusion_path.read_text().splitlines() if line.strip()]
            except:
                return []
        else:
            exclusion_path.write_text("Кузов\nСтекла\nМасла")
            return ["Кузов", "Стекла", "Масла"]
    
    def save_exclusion_rules(self):
        exclusion_path = self.data_dir / "exclusion_rules.txt"
        exclusion_path.write_text("\n".join(self.exclusion_rules))
    
    def load_category_mapping(self) -> Dict[str, str]:
        category_path = self.data_dir / "category_mapping.txt"
        default_mapping = {
            "Радиатор": "Охлаждение", "Шаровая опора": "Подвеска",
            "Фильтр масляный": "Фильтры", "Тормозные колодки": "Тормоза"
        }
        if category_path.exists():
            try:
                mapping = {}
                for line in category_path.read_text().splitlines():
                    if line.strip() and "|" in line:
                        key, value = line.split("|", 1)
                        mapping[key.strip()] = value.strip()
                return mapping
            except:
                return default_mapping
        else:
            content = "\n".join([f"{k}|{v}" for k, v in default_mapping.items()])
            category_path.write_text(content)
            return default_mapping
    
    def save_category_mapping(self):
        category_path = self.data_dir / "category_mapping.txt"
        content = "\n".join([f"{k}|{v}" for k, v in self.category_mapping.items()])
        category_path.write_text(content)
    
    def setup_database(self):
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
                multiplicity INTEGER,
                barcode VARCHAR,
                length DOUBLE, width DOUBLE, height DOUBLE, weight DOUBLE,
                image_url VARCHAR,
                dimensions_str VARCHAR,
                description VARCHAR,
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
                currency VARCHAR DEFAULT 'RUB',
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)
    
    def create_indexes(self):
        st.info("Создание индексов...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_prices_keys ON prices(artikul_norm, brand_norm)"
        ]
        for index_sql in indexes:
            self.conn.execute(index_sql)
        st.success("Индексы созданы.")
    
    @staticmethod
    def normalize_key(key_series: pl.Series) -> pl.Series:
        return (key_series.fill_null("").cast(pl.Utf8)
                .str.replace_all("'", "")
                .str.replace_all(r"[^0-9A-Za-zA-ya-яЁё`\-\s]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .str.to_lowercase())
    
    @staticmethod
    def clean_values(value_series: pl.Series) -> pl.Series:
        return (value_series.fill_null("").cast(pl.Utf8)
                .str.replace_all("'", "")
                .str.replace_all(r"[^0-9A-Za-zA-ya-яЁё`\-\s]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars())
    
    def determine_category_vectorized(self, name_series: pl.Series) -> pl.Series:
        name_lower = name_series.str.to_lowercase()
        categorization_expr = pl.when(pl.lit(False)).then(pl.lit(None))
        
        for key, category in self.category_mapping.items():
            categorization_expr = categorization_expr.when(
                name_lower.str.contains(key.lower())
            ).then(pl.lit(category))
        
        categories_map = {
            'Фильтр': 'фильтр|filter', 'Тормоза': 'тормоз|brake|колодк|диск|суппорт',
            'Подвеска': 'амортизатор|стойк|spring|подвеск|рычаг', 'Двигатель': 'двигатель|engine|свеч|поршень|клапан',
            'Трансмиссия': 'трансмиссия|сцеплен|коробк|transmission', 'Электрика': 'аккумулятор|генератор|стартер|провод|ламп',
            'Рулевое': 'рулевой|тяга|наконечник|steering', 'Выпуск': 'глушитель|катализатор|выхлоп|exhaust',
            'Охлаждение': 'радиатор|вентилятор|термостат|cooling', 'Топливо': 'топливный|бензонасос|форсунк|fuel'
        }
        
        for category, pattern in categories_map.items():
            categorization_expr = categorization_expr.when(name_lower.str.contains(pattern)).then(pl.lit(category))
        
        return categorization_expr.otherwise(pl.lit('Разное')).alias('category')
    
    def detect_columns(self, actual_columns: List[str], expected_columns: List[str]) -> Dict[str, str]:
        mapping = {}
        column_variants = {
            'oe_number': ['oe номер', 'oe', 'оe', 'номер', 'code', 'OE'],
            'artikul': ['артикул', 'article', 'sku'],
            'brand': ['бренд', 'brand', 'производитель', 'manufacturer'],
            'name': ['наименование', 'название', 'name', 'описание', 'description'],
            'applicability': ['применимость', 'автомобиль', 'vehicle', 'applicability'],
            'barcode': ['штрих-код', 'barcode', 'штрихкод', 'ean', 'eac13'],
            'multiplicity': ['кратность шт', 'кратность', 'multiplicity'],
            'length': ['длина (см)', 'длина', 'length', 'длинна'],
            'width': ['ширина (см)', 'ширина', 'width'], 'height': ['высота (см)', 'высота', 'height'],
            'weight': ['вес (кг)', 'вес, кг', 'вес', 'weight'], 
            'image_url': ['ссылка', 'url', 'изображение', 'image', 'картинка'],
            'dimensions_str': ['весогабариты', 'размеры', 'dimensions', 'size'],
            'price': ['цена', 'price', 'рекомендованная цена', 'retail price'],
            'currency': ['валюта', 'currency']
        }
        actual_lower = {col.lower(): col for col in actual_columns}
        for expected in expected_columns:
            variants = [v.lower() for v in column_variants.get(expected, [expected])]
            for variant in variants:
                for actual_l, actual_orig in actual_lower.items():
                    if variant in actual_l:
                        mapping[actual_orig] = expected
                        break
                if expected in mapping.values():
                    break
        return mapping

    def read_and_prepare_file(self, file_path: str, file_type: str) -> pl.DataFrame:
        logger.info(f"Чтение файла: {file_type} ({file_path})")
        try:
            if not os.path.exists(file_path):
                logger.error(f"Файл не найден: {file_path}")
                return pl.DataFrame()
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.warning(f"Файл пуст: {file_path}")
                return pl.DataFrame()

            # Используем fastexcel
            workbook = fastexcel.read_workbook(file_path)
            sheet_names = workbook.sheet_names
            if not sheet_names:
                logger.warning(f"Нет листов: {file_path}")
                return pl.DataFrame()
            df = workbook.load_sheet(sheet_names[0]).to_pandas()
            df = pl.from_pandas(df)

            if df.is_empty():
                logger.warning("Файл прочитан, но данные пусты")
                return pl.DataFrame()
        except Exception as e:
            logger.exception(f"Ошибка fastexcel: {e}. Попытка через openpyxl...")
            try:
                df = pl.read_excel(file_path, engine='openpyxl')
            except Exception as e2:
                logger.exception(f"Ошибка openpyxl: {e2}")
                return pl.DataFrame()

        # Продолжение без изменений
        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand'],
            'prices': ['artikul', 'brand', 'price', 'currency']
        }
        expected_cols = schemas.get(file_type, [])
        column_mapping = self.detect_columns(df.columns, expected_cols)
        if not column_mapping:
            logger.warning(f"Не удалось определить колонки: {file_type}, колонки: {df.columns}")
            return pl.DataFrame()
        df = df.rename(column_mapping)

        for col in ['artikul', 'brand', 'oe_number']:
            if col in df.columns:
                df = df.with_columns(self.clean_values(pl.col(col)).alias(col))
        key_cols = [col for col in ['oe_number', 'artikul', 'brand'] if col in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')
        for col in ['artikul', 'brand', 'oe_number']:
            if col in df.columns:
                df = df.with_columns(self.normalize_key(pl.col(col)).alias(f"{col}_norm"))
        return df

    def upsert_data(self, table_name: str, df: pl.DataFrame, pk: List[str]):
        if df.is_empty(): return
        df = df.unique(keep='first')
        cols = df.columns
        pk_str = ", ".join(f'"{c}"' for c in pk)
        temp_view_name = f"temp_{table_name}_{int(time.time())}"
        self.conn.register(temp_view_name, df.to_arrow())
        update_cols = [col for col in cols if col not in pk]
        on_conflict_action = "DO NOTHING" if not update_cols else \
            f"DO UPDATE SET {', '.join([f'\"{col}\" = excluded.\"{col}\"' for col in update_cols])}"
        sql = f"INSERT INTO {table_name} SELECT * FROM {temp_view_name} ON CONFLICT ({pk_str}) {on_conflict_action};"
        try:
            self.conn.execute(sql)
            logger.info(f"Успешно обновлено {len(df)} записей в {table_name}.")
        except Exception as e:
            logger.error(f"Ошибка UPSERT в {table_name}: {e}")
            st.error(f"Ошибка при записи в {table_name}.")
        finally:
            self.conn.unregister(temp_view_name)

    def upsert_prices(self, price_df: pl.DataFrame):
        if price_df.is_empty(): return
        if 'artikul' in price_df.columns and 'brand' in price_df.columns:
            price_df = price_df.with_columns([
                self.normalize_key(pl.col('artikul')).alias('artikul_norm'),
                self.normalize_key(pl.col('brand')).alias('brand_norm')
            ])
        if 'currency' not in price_df.columns:
            price_df = price_df.with_columns(pl.lit('RUB').alias('currency'))
        price_df = price_df.filter(
            (pl.col('price') >= self.price_rules['min_price']) &
            (pl.col('price') <= self.price_rules['max_price'])
        )
        self.upsert_data('prices', price_df, ['artikul_norm', 'brand_norm'])

    def apply_markup(self, price: float, brand: str) -> float:
        markup = self.price_rules['brand_markups'].get(brand, self.price_rules['global_markup'])
        return price * (1 + markup)

    def process_and_load_data(self, dataframes: Dict[str, pl.DataFrame]):
        st.info("🔄 Начало загрузки данных...")
        steps = [s for s in ['oe', 'cross', 'parts'] if s in dataframes or s == 'parts']
        num_steps = len(steps)
        progress_bar = st.progress(0)
        step_counter = 0

        if 'oe' in dataframes:
            step_counter += 1
            progress_bar.progress(step_counter / (num_steps + 1))
            df = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'])
            oe_df = oe_df.with_columns(self.determine_category_vectorized(pl.col('name')))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])
            cross_df = df.filter(pl.col('artikul_norm') != "").select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        if 'cross' in dataframes:
            step_counter += 1
            progress_bar.progress(step_counter / (num_steps + 1))
            df = dataframes['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            cross_df = df.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        if 'prices' in dataframes:
            st.info("💰 Обработка цен...")
            self.upsert_prices(dataframes['prices'])
            st.success("✅ Цены обновлены")

        step_counter += 1
        progress_bar.progress(step_counter / (num_steps + 1))
        progress_bar.empty()
        st.success("💾 Загрузка данных завершена.")

    def export_to_excel(self, output_path: Path, selected_columns: List[str] | None = None) -> bool:
        total_records = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"Экспорт {total_records:,} записей в Excel...")
        try:
            query = self.build_export_query(selected_columns, include_prices=True, apply_markup=True)
            df = self.conn.execute(query).pl()
            dimension_cols = ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность", "Цена"]
            for col_name in dimension_cols:
                if col_name in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col_name).is_not_null())
                        .then(pl.col(col_name).cast(pl.Utf8))
                        .otherwise(pl.lit(""))
                        .alias(col_name)
                    )
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                df.to_pandas().to_excel(writer, index=False, sheet_name='Данные')
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            st.success(f"✅ Экспорт в Excel: {output_path.name} ({file_size:.1f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта в Excel")
            st.error(f"❌ Ошибка: {e}")
            return False

    def build_export_query(self, selected_columns: List[str] | None, include_prices: bool = True, apply_markup: bool = True) -> str:
        standard_description = """Состояние товара: новый. Высококачественные автозапчасти..."""
        price_column = ""
        if include_prices:
            if apply_markup:
                price_column = f"""
                    CASE WHEN pr.price IS NOT NULL THEN pr.price * (1 + COALESCE(brm.markup, {self.price_rules['global_markup']}))
                    ELSE pr.price END AS "Цена",
                    COALESCE(pr.currency, 'RUB') AS "Валюта","""
            else:
                price_column = """pr.price AS "Цена", COALESCE(pr.currency, 'RUB') AS "Валюта","""

        exclusion_conditions = " OR ".join([f"r.representative_name NOT ILIKE '%{ex}%' " for ex in self.exclusion_rules if ex.strip()])
        exclusion_where = f"AND ({exclusion_conditions})" if exclusion_conditions else ""

        columns_map = [
            ("Артикул бренда", 'r.artikul AS "Артикул бренда"'),
            ("Бренд", 'r.brand AS "Бренд"'),
            ("Наименование", 'COALESCE(r.representative_name, r.analog_representative_name) AS "Наименование"'),
            ("Применимость", 'COALESCE(r.representative_applicability, r.analog_representative_applicability) AS "Применимость"'),
            ("Описание", "CONCAT(COALESCE(r.description, ''), dt.text) AS \"Описание\""),
            ("Категория товара", 'COALESCE(r.representative_category, r.analog_representative_category) AS "Категория товара"'),
            ("Кратность", 'r.multiplicity AS "Кратность"'),
            ("Длинна", 'COALESCE(r.length, r.analog_length) AS "Длинна"'),
            ("Ширина", 'COALESCE(r.width, r.analog_width) AS "Ширина"'),
            ("Высота", 'COALESCE(r.height, r.analog_height) AS "Высота"'),
            ("Вес", 'COALESCE(r.weight, r.analog_weight) AS "Вес"'),
            ("Длинна/Ширина/Высота", "COALESCE(CASE WHEN r.dimensions_str IS NULL OR r.dimensions_str = '' OR UPPER(TRIM(r.dimensions_str)) = 'XX' THEN NULL ELSE r.dimensions_str END, r.analog_dimensions_str) AS \"Длинна/Ширина/Высота\""),
            ("OE номер", 'r.oe_list AS "OE номер"'),
            ("аналоги", 'r.analog_list AS "аналоги"'),
            ("Ссылка на изображение", 'r.image_url AS "Ссылка на изображение"')
        ]
        if include_prices:
            columns_map.append(("Цена", '"Цена"'))
            columns_map.append(("Валюта", '"Валюта"'))
        selected_exprs = [expr for name, expr in columns_map if name in selected_columns] if selected_columns else [expr for _, expr in columns_map]
        ctes = f"""
        WITH DescriptionTemplate AS (SELECT CHR(10) || CHR(10) || $${standard_description}$$ AS text),
        BrandMarkups AS (SELECT brand, markup FROM ({self._get_brand_markups_sql()}) AS tmp),
        PartDetails AS (SELECT cr.artikul_norm, cr.brand_norm, STRING_AGG(DISTINCT regexp_replace(regexp_replace(o.oe_number, '''', ''), '[^0-9A-Za-zА-Яа-яЁё`\\-\\s]', '', 'g'), ', ') AS oe_list,
            ANY_VALUE(o.name) AS representative_name, ANY_VALUE(o.applicability) AS representative_applicability, ANY_VALUE(o.category) AS representative_category
            FROM cross_references cr JOIN oe_data o ON cr.oe_number_norm = o.oe_number_norm GROUP BY cr.artikul_norm, cr.brand_norm),
        AllAnalogs AS (SELECT cr1.artikul_norm, cr1.brand_norm, STRING_AGG(DISTINCT regexp_replace(regexp_replace(p2.artikul, '''', ''), '[^0-9A-Za-zА-Яа-яЁё`\\-\\s]', '', 'g'), ', ') as analog_list
            FROM cross_references cr1 JOIN cross_references cr2 ON cr1.oe_number_norm = cr2.oe_number_norm
            JOIN parts_data p2 ON cr2.artikul_norm = p2.artikul_norm AND cr2.brand_norm = p2.brand_norm
            WHERE (cr1.artikul_norm != p2.artikul_norm OR cr1.brand_norm != p2.brand_norm)
            GROUP BY cr1.artikul_norm, cr1.brand_norm),
        RankedData AS (SELECT p.artikul, p.brand, p.description, p.multiplicity, p.length, p.width, p.height, p.weight, p.dimensions_str, p.image_url,
            pd.representative_name, pd.representative_applicability, pd.representative_category, pd.oe_list, aa.analog_list,
            ROW_NUMBER() OVER(PARTITION BY p.artikul_norm, p.brand_norm ORDER BY pd.representative_name DESC NULLS LAST, pd.oe_list DESC NULLS LAST) as rn
            FROM parts_data p LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
            LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm)
        """
        price_join = "LEFT JOIN prices pr ON r.artikul_norm = pr.artikul_norm AND r.brand_norm = pr.brand_norm LEFT JOIN BrandMarkups brm ON r.brand = brm.brand" if include_prices else ""
        query = ctes + f"""
        SELECT {price_column} {', '.join(selected_exprs)}
        FROM RankedData r CROSS JOIN DescriptionTemplate dt {price_join}
        WHERE r.rn = 1 {exclusion_where}
        ORDER BY r.brand, r.artikul
        """
        return query

    def _get_brand_markups_sql(self) -> str:
        rows = [f"SELECT '{brand}' AS brand, {markup} AS markup" for brand, markup in self.price_rules['brand_markups'].items()]
        return " UNION ALL ".join(rows) if rows else "SELECT NULL AS brand, NULL AS markup LIMIT 0"

    def export_to_csv_optimized(self, output_path: str, selected_columns: List[str] | None = None, include_prices: bool = True, apply_markup: bool = True) -> bool:
        total_records = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"Экспорт {total_records:,} записей в CSV...")
        try:
            query = self.build_export_query(selected_columns, include_prices, apply_markup)
            df = self.conn.execute(query).pl()
            dimension_cols = ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]
            for col_name in dimension_cols:
                if col_name in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col_name).is_not_null())
                        .then(pl.col(col_name).cast(pl.Utf8))
                        .otherwise(pl.lit(""))
                        .alias(col_name)
                    )
            buf = io.StringIO()
            df.write_csv(buf, separator=';')
            csv_text = buf.getvalue()
            with open(output_path, 'wb') as f:
                f.write(b'\xef\xbb\xbf')
                f.write(csv_text.encode('utf-8'))
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            st.success(f"✅ Экспорт в CSV: {output_path} ({file_size:.1f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта в CSV")
            st.error(f"❌ Ошибка: {e}")
            return False

    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total_records = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm) FROM parts_data)").fetchone()[0]
        st.info(f"Всего записей: {total_records:,}")
        if total_records == 0:
            st.warning("Нет данных. Сначала загрузите файлы.")
            return

        available_columns = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение"
        ]
        prices_count = self.conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        if prices_count > 0:
            available_columns.extend(["Цена", "Валюта"])

        selected_columns = st.multiselect("Выберите столбцы", available_columns, default=available_columns)
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.radio("Формат", ["CSV", "Excel (.xlsx)"])
        with col2:
            include_prices = st.checkbox("Включить цены", value=True)
            apply_markup = st.checkbox("Применить наценку", value=True, disabled=not include_prices)

        if export_format == "CSV":
            if st.button("🚀 Экспорт в CSV"):
                path = self.data_dir / "report.csv"
                with st.spinner("Экспорт..."):
                    success = self.export_to_csv_optimized(str(path), selected_columns, include_prices, apply_markup)
                if success:
                    with open(path, "rb") as f:
                        st.download_button("📥 Скачать CSV", f, "report.csv", "text/csv")

        elif export_format == "Excel (.xlsx)":
            if st.button("🚀 Экспорт в Excel"):
                path = self.data_dir / "report.xlsx"
                with st.spinner("Экспорт в Excel..."):
                    success = self.export_to_excel(path, selected_columns)
                if success:
                    with open(path, "rb") as f:
                        st.download_button("📥 Скачать Excel", f, "report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    def log_memory_usage(self):
        process = psutil.Process()
        mem = process.memory_info().rss / 1024 / 1024
        logger.info(f"Память: {mem:.1f} МБ")

def main():
    st.title("🚗 AutoParts Catalog")
    catalog = HighVolumeAutoPartsCatalog()
    menu = st.sidebar.radio("Меню", ["Загрузка данных", "Экспорт", "Управление данными"])
    
    if menu == "Загрузка данных":
        st.header("📥 Загрузка данных")
        files = {
            'oe': st.file_uploader("1. Основные данные", type=['xlsx']),
            'cross': st.file_uploader("2. Кроссы", type=['xlsx']),
            'barcode': st.file_uploader("3. Штрих-коды", type=['xlsx']),
            'dimensions': st.file_uploader("4. Весогабариты", type=['xlsx']),
            'images': st.file_uploader("5. Изображения", type=['xlsx']),
            'prices': st.file_uploader("6. Цены", type=['xlsx'])
        }
        if st.button("Загрузить и обработать"):
            with st.spinner("Чтение..."):
                dataframes = {}
                for key, file in files.items():
                    if file:
                        path = f"/tmp/{file.name}"
                        with open(path, "wb") as f:
                            f.write(file.getvalue())
                        dataframes[key] = catalog.read_and_prepare_file(path, key)
                catalog.process_and_load_data(dataframes)

    elif menu == "Экспорт":
        catalog.show_export_interface()

    elif menu == "Управление данными":
        catalog.show_data_management()

if __name__ == "__main__":
    main()
