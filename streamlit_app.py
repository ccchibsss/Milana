import polars as pl
import duckdb
import streamlit as st
import os
import time
import io
import zipfile
import json
from pathlib import Path
from typing import Dict, List

# Время лимит для Excel
EXCEL_ROW_LIMIT = 1_000_000

class HighVolumeAutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(database=str(self.db_path))
        self.setup_database()
        self.create_indexes()

        st.set_page_config(
            page_title="AutoParts Catalog 10M+",
            layout="wide",
            page_icon="🚗"
        )

    def setup_database(self):
        # Таблицы
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
                length DOUBLE,
                width DOUBLE,
                height DOUBLE,
                weight DOUBLE,
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
            CREATE TABLE IF NOT EXISTS recommended_prices (
                artikul_norm VARCHAR PRIMARY KEY,
                price DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS price_list (
                artikul VARCHAR,
                brand VARCHAR,
                quantity INTEGER,
                price DOUBLE,
                PRIMARY KEY (artikul, brand)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS markup_settings (
                id INTEGER PRIMARY KEY,
                total_markup DOUBLE,
                brand_markup JSON
            )
        """)
        # Инициализация настроек
        if not self.conn.execute("SELECT 1 FROM markup_settings").fetchone():
            self.conn.execute("INSERT INTO markup_settings (id, total_markup, brand_markup) VALUES (1, 0, '{}')")

    def create_indexes(self):
        # Индексы
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)"
        ]
        for index_sql in indexes:
            self.conn.execute(index_sql)

    # --- Методы нормализации и очистки данных ---
    @staticmethod
    def normalize_key(key_series: pl.Series) -> pl.Series:
        return (
            key_series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all("'", "")
            .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .str.to_lowercase()
        )

    @staticmethod
    def clean_values(value_series: pl.Series) -> pl.Series:
        return (
            value_series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all("'", "")
            .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

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
            'width': ['ширина (см)', 'ширина', 'width'],
            'height': ['высота (см)', 'высота', 'height'],
            'weight': ['вес (кг)', 'вес, кг', 'вес', 'weight'],
            'image_url': ['ссылка', 'url', 'изображение', 'image', 'картинка'],
            'dimensions_str': ['весогабариты', 'размеры', 'dimensions', 'size']
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

    # --- Обработка файлов ---
    def read_and_prepare_file(self, file_path: str, file_type: str) -> pl.DataFrame:
        logger.info(f"Обработка файла: {file_type} ({file_path})")
        try:
            if not os.path.exists(file_path):
                logger.error(f"Файл не найден: {file_path}")
                return pl.DataFrame()
            df = pl.read_excel(file_path, engine='calamine')
            if df.is_empty():
                logger.warning(f"Пустой файл: {file_path}")
                return pl.DataFrame()
        except Exception as e:
            logger.exception(f"Ошибка при чтении файла {file_path}: {e}")
            return pl.DataFrame()

        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand']
        }
        expected_cols = schemas.get(file_type, [])
        column_mapping = self.detect_columns(df.columns, expected_cols)

        if not column_mapping:
            logger.warning(f"Не удалось определить колонки для файла {file_type}. Доступные: {df.columns}")
            return pl.DataFrame()

        df = df.rename(column_mapping)

        # Очистка значений
        if 'artikul' in df.columns:
            df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand=self.clean_values(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))

        key_cols = [col for col in ['oe_number', 'artikul', 'brand'] if col in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')

        # Нормализация ключей
        if 'artikul' in df.columns:
            df = df.with_columns(artikul_norm=self.normalize_key(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand_norm=self.normalize_key(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number_norm=self.normalize_key(pl.col('oe_number')))
        return df

    def upsert_data(self, table_name: str, df: pl.DataFrame, pk: List[str]):
        if df.is_empty():
            return
        df = df.unique(keep='first')
        cols = df.columns
        pk_str = ", ".join(f'"{c}"' for c in pk)
        temp_view_name = f"temp_{table_name}_{int(time.time())}"
        self.conn.register(temp_view_name, df.to_arrow())

        update_cols = [col for col in cols if col not in pk]
        if not update_cols:
            on_conflict_action = "DO NOTHING"
        else:
            update_clause = ", ".join([f'"{col}" = excluded."{col}"' for col in update_cols])
            on_conflict_action = f"DO UPDATE SET {update_clause}"

        sql = f"""
        INSERT INTO {table_name}
        SELECT * FROM {temp_view_name}
        ON CONFLICT ({pk_str}) {on_conflict_action};
        """
        try:
            self.conn.execute(sql)
            logger.info(f"Обновлено/вставлено {len(df)} записей в {table_name}")
        except Exception as e:
            logger.exception(f"Ошибка при UPSERT {table_name}: {e}")
            st.error(f"Ошибка при записи в таблицу {table_name}")
        finally:
            self.conn.unregister(temp_view_name)

    def process_and_load_data(self, dataframes: Dict[str, pl.DataFrame]):
        st.info("🔄 Начинаю загрузку данных...")
        steps = ['oe', 'cross', 'parts']
        num_steps = len(steps)
        progress_bar = st.progress(0, text="Подготовка...")
        step_counter = 0

        # Обработка OE
        if 'oe' in dataframes:
            step_counter += 1
            progress_bar.progress(step_counter / (num_steps + 1), f"({step_counter}/{num_steps}) Обработка OE")
            df_oe = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df_oe.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'])
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self.determine_category_vectorized(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])

            cross_df_from_oe = df_oe.filter(pl.col('artikul_norm') != "").select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df_from_oe, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка cross
        if 'cross' in dataframes:
            step_counter += 1
            progress_bar.progress(step_counter / (num_steps + 1), f"({step_counter}/{num_steps}) Обработка кроссов")
            df_cross = dataframes['cross'].filter(
                (pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != "")
            )
            self.upsert_data('cross_references', df_cross, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка parts
        step_counter += 1
        progress_bar.progress(step_counter / (num_steps + 1), f"({step_counter}/{num_steps}) Обработка артикула")
        # Объединение всех данных артикула
        parts_df = None
        files_order = ['oe', 'barcode', 'images', 'dimensions']
        key_files = {ftype: dataframes[ftype] for ftype in files_order if ftype in dataframes}
        if key_files:
            all_parts = pl.concat([
                df.select(['artikul', 'artikul_norm', 'brand', 'brand_norm'])
                for df in key_files.values()
                if 'artikul_norm' in df.columns and 'brand_norm' in df.columns
            ]).filter(pl.col('artikul_norm') != "").unique(subset=['artikul_norm', 'brand_norm'])
            parts_df = all_parts

            # Объединение данных по файлам
            for ftype in files_order:
                if ftype not in key_files:
                    continue
                df = key_files[ftype]
                if df.is_empty() or 'artikul_norm' not in df.columns:
                    continue
                join_cols = [col for col in df.columns if col not in ['artikul', 'artikul_norm', 'brand', 'brand_norm']]
                if not join_cols:
                    continue
                existing_cols = set(parts_df.columns)
                join_cols = [col for col in join_cols if col not in existing_cols]
                if not join_cols:
                    continue
                df_subset = df.select(['artikul_norm', 'brand_norm'] + join_cols).unique(subset=['artikul_norm', 'brand_norm'])
                parts_df = parts_df.join(df_subset, on=['artikul_norm', 'brand_norm'], how='left', coalesce=True)

        # Обработка размеров, описание
        if parts_df is not None and not parts_df.is_empty():
            # multiplicity
            if 'multiplicity' not in parts_df.columns:
                parts_df = parts_df.with_columns(pl.lit(1).cast(pl.Int32).alias('multiplicity'))
            else:
                parts_df = parts_df.with_columns(pl.col('multiplicity').fill_null(1).cast(pl.Int32))
            # dimensions
            for col in ['length', 'width', 'height']:
                if col not in parts_df.columns:
                    parts_df = parts_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
            if 'dimensions_str' not in parts_df.columns:
                parts_df = parts_df.with_columns(dimensions_str=pl.lit(None).cast(pl.Utf8))
            # Создание dimensions_str
            parts_df = parts_df.with_columns([
                pl.col('length').cast(pl.Utf8).fill_null('').alias('_length_str'),
                pl.col('width').cast(pl.Utf8).fill_null('').alias('_width_str'),
                pl.col('height').cast(pl.Utf8).fill_null('').alias('_height_str')
            ])

            parts_df = parts_df.with_columns(
                pl.when(
                    (pl.col('dimensions_str').is_not_null()) &
                    (pl.col('dimensions_str') != '') &
                    (pl.col('dimensions_str').cast(pl.Utf8).str.to_upper().alias('dim_upper') != 'XX')
                )
                .then(pl.col('dimensions_str'))
                .otherwise(
                    pl.concat_str([pl.col('_length_str'), pl.lit('x'), pl.col('_width_str'), pl.lit('x'), pl.col('_height_str')], separator='')
                ).alias('dimensions_str')
            )
            parts_df = parts_df.drop(['_length_str', '_width_str', '_height_str'])

            # Описание
            if 'artikul' not in parts_df.columns:
                parts_df = parts_df.with_columns(artikul=pl.lit(''))
            if 'brand' not in parts_df.columns:
                parts_df = parts_df.with_columns(brand=pl.lit(''))

            parts_df = parts_df.with_columns([
                pl.col('artikul').cast(pl.Utf8).fill_null('').alias('_artikul'),
                pl.col('brand').cast(pl.Utf8).fill_null('').alias('_brand'),
                pl.col('multiplicity').cast(pl.Utf8).alias('_multiplicity')
            ])

            parts_df = parts_df.with_columns(
                pl.concat_str([
                    'Артикул: ', pl.col('_artikul'),
                    ', Бренд: ', pl.col('_brand'),
                    ', Кратность: ', pl.col('_multiplicity'), ' шт.'
                ], separator='').alias('description')
            )
            parts_df = parts_df.drop(['_artikul', '_brand', '_multiplicity'])

            # Выбор финальных колонок
            final_cols = [
                'artikul_norm', 'brand_norm', 'artikul', 'brand', 'multiplicity', 'barcode',
                'length', 'width', 'height', 'weight', 'image_url', 'dimensions_str', 'description'
            ]
            select_exprs = []
            for c in final_cols:
                if c in parts_df.columns:
                    select_exprs.append(pl.col(c))
                else:
                    select_exprs.append(pl.lit(None).alias(c))
            parts_df = parts_df.select(select_exprs)

            # Добавление цены с учетом наценки
            # В цикле при экспорте
            self.upsert_data('parts_data', parts_df, ['artikul_norm', 'brand_norm'])

        progress_bar.progress(1.0, text="Обновление базы завершено")
        time.sleep(1)
        progress_bar.empty()
        st.success("💾 Загрузка завершена.")

    def merge_all_data_parallel(self, file_paths: Dict[str, str]) -> Dict:
        start_time = time.time()
        stats = {}
        st.info("🚀 Начинаю параллельную обработку файлов")
        n_files = len(file_paths)
        file_progress = st.progress(0, text="Обработка файлов...")

        dataframes = {}
        processed_files = 0
        with st.runtime.scriptrunner.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.read_and_prepare_file, path, ftype): ftype
                for ftype, path in file_paths.items()
            }
            for future in futures:
                ftype = futures[future]
                try:
                    df = future.result()
                    if not df.is_empty():
                        dataframes[ftype] = df
                        st.success(f"Файл '{ftype}' обработан")
                    else:
                        st.warning(f"Файл '{ftype}' пуст или не удалось")
                except Exception as e:
                    st.error(f"Ошибка при обработке {ftype}: {e}")
                processed_files += 1
                file_progress.progress(processed_files / n_files)
        file_progress.empty()

        if not dataframes:
            st.error("Нет загруженных данных")
            return {}

        self.process_and_load_data(dataframes)

        stats['processing_time'] = time.time() - start_time
        stats['total_records'] = self.get_total_records()
        st.success(f"Обработка завершена за {stats['processing_time']:.2f} секунд")
        st.success(f"Всего артикулов: {stats['total_records']:,}")
        self.create_indexes()
        return stats

    def get_total_records(self) -> int:
        try:
            return self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]
        except:
            return 0

    def get_statistics(self) -> Dict:
        stats = {}
        try:
            stats['total_parts'] = self.get_total_records()
            if stats['total_parts'] == 0:
                return {
                    'total_parts': 0, 'total_oe': 0, 'total_brands': 0,
                    'top_brands': pl.DataFrame(), 'categories': pl.DataFrame()
                }
            stats['total_oe'] = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()[0]
            stats['total_brands'] = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data WHERE brand IS NOT NULL").fetchone()[0]
            # top brands
            br_res = self.conn.execute(
                "SELECT brand, COUNT(*) as count FROM parts_data WHERE brand IS NOT NULL GROUP BY brand ORDER BY count DESC LIMIT 10"
            ).fetchall()
            stats['top_brands'] = pl.DataFrame(br_res, schema=["brand", "count"])
            # категории
            cat_res = self.conn.execute(
                "SELECT category, COUNT(*) as count FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY count DESC"
            ).fetchall()
            stats['categories'] = pl.DataFrame(cat_res, schema=["category", "count"])
        except Exception as e:
            st.error(f"Ошибка при сборе статистики: {e}")
            stats = {
                'total_parts': 0,
                'total_oe': 0,
                'total_brands': 0,
                'top_brands': pl.DataFrame(),
                'categories': pl.DataFrame()
            }
        return stats

    # --- Методы загрузки цен и настроек ---
    def load_price_recommendation(self, file_bytes):
        df = pl.read_excel(io.BytesIO(file_bytes))
        if 'артикул' not in df.columns or 'цена' not in df.columns:
            st.error("Файл должен содержать колонки 'артикул' и 'цена'")
            return
        df = df.select([
            pl.col('артикул').alias('artikul'),
            pl.col('цена').cast(pl.Float64)
        ])
        for row in df.iter_rows():
            artikul = row[0]
            price = row[1]
            artikul_norm_series = self.normalize_key(pl.Series([artikul]))
            artikul_norm = artikul_norm_series[0] if len(artikul_norm_series) > 0 else ''
            self.conn.execute("""
                INSERT INTO recommended_prices (artikul_norm, price)
                VALUES (?, ?)
                ON CONFLICT (artikul_norm) DO UPDATE SET price=excluded.price
            """, [artikul_norm, price])
        st.success("Рекомендованные цены успешно загружены.")

    def load_price_list(self, file_bytes):
        df = pl.read_excel(io.BytesIO(file_bytes))
        required_cols = ['артикул', 'бренд', 'кол-во', 'цена']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Файл должен содержать колонку '{col}'")
                return
        df = df.select([
            pl.col('артикул'),
            pl.col('бренд'),
            pl.col('кол-во').cast(pl.Int32),
            pl.col('цена').cast(pl.Float64)
        ])
        for row in df.iter_rows():
            artikul = row[0]
            brand = row[1]
            qty = row[2]
            price = row[3]
            artikul_norm_series = self.normalize_key(pl.Series([artikul]))
            brand_norm_series = self.normalize_key(pl.Series([brand]))
            artikul_norm = artikul_norm_series[0] if len(artikul_norm_series) > 0 else ''
            brand_norm = brand_norm_series[0] if len(brand_norm_series) > 0 else ''
            self.conn.execute("""
                INSERT INTO price_list (artikul, brand, quantity, price)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (artikul, brand) DO UPDATE SET
                quantity=excluded.quantity,
                price=excluded.price
            """, [artikul, brand, qty, price])
        st.success("Прайс-лист успешно загружен.")

    def set_markups(self, total_markup: float, brand_markups: Dict[str, float]):
        self.conn.execute("""
            UPDATE markup_settings SET total_markup = ?, brand_markup = ?
            WHERE id = 1
        """, [total_markup, json.dumps(brand_markups)])
        st.success("Настройки наценки обновлены.")

    def get_markups(self):
        row = self.conn.execute("SELECT total_markup, brand_markup FROM markup_settings WHERE id=1").fetchone()
        if row:
            total_markup = row[0]
            brand_markup = json.loads(row[1]) if row[1] else {}
            return total_markup, brand_markup
        return 0, {}

    def apply_markup(self, price, brand_norm=''):
        total_markup, brand_markups = self.get_markups()
        markup = total_markup
        if brand_norm and brand_norm in brand_markups:
            markup += brand_markups[brand_norm]
        return price * (1 + markup / 100)

    # --- Экспорт с учетом цен и наценки ---
    def build_export_query(self, selected_columns: List[str] | None):
        standard_description = """Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""
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
            ("Ссылка на изображение", 'r.image_url AS "Ссылка на изображение"'),
            ("Цена с наценкой", 'r.price_with_markup AS "Цена с наценкой"')
        ]

        if not selected_columns:
            selected_exprs = [expr for _, expr in columns_map]
        else:
            selected_exprs = [expr for name, expr in columns_map if name in selected_columns]
            if not selected_exprs:
                selected_exprs = [expr for _, expr in columns_map]

        ctes = f"""
        WITH DescriptionTemplate AS (
            SELECT CHR(10) || CHR(10) || $${standard_description}$$ AS text
        ),
        PartDetails AS (
            SELECT
                cr.artikul_norm,
                cr.brand_norm,
                STRING_AGG(DISTINCT regexp_replace(regexp_replace(o.oe_number, '''', ''), '[^0-9A-Za-zА-Яа-яЁё`\\-\\s]', '', 'g'), ', ') AS oe_list,
                ANY_VALUE(o.name) AS representative_name,
                ANY_VALUE(o.applicability) AS representative_applicability,
                ANY_VALUE(o.category) AS representative_category
            FROM cross_references cr
            JOIN oe_data o ON cr.oe_number_norm = o.oe_number_norm
            GROUP BY cr.artikul_norm, cr.brand_norm
        ),
        AllAnalogs AS (
            SELECT
                cr1.artikul_norm,
                cr1.brand_norm,
                STRING_AGG(DISTINCT regexp_replace(regexp_replace(p2.artikul, '''', ''), '[^0-9A-Za-zА-Яа-яЁё`\\-\\s]', '', 'g'), ', ') as analog_list
            FROM cross_references cr1
            JOIN cross_references cr2 ON cr1.oe_number_norm = cr2.oe_number_norm
            JOIN parts_data p2 ON cr2.artikul_norm = p2.artikul_norm AND cr2.brand_norm = p2.brand_norm
            WHERE (cr1.artikul_norm != p2.artikul_norm OR cr1.brand_norm != p2.brand_norm)
            GROUP BY cr1.artikul_norm, cr1.brand_norm
        ),
        InitialOENumbers AS (
            SELECT DISTINCT
                p.artikul_norm,
                p.brand_norm,
                cr.oe_number_norm
            FROM parts_data p
            LEFT JOIN cross_references cr ON p.artikul_norm = cr.artikul_norm AND p.brand_norm = cr.brand_norm
            WHERE cr.oe_number_norm IS NOT NULL
        ),
        Level1Analogs AS (
            SELECT DISTINCT
                i.artikul_norm AS source_artikul_norm,
                i.brand_norm AS source_brand_norm,
                cr2.artikul_norm AS related_artikul_norm,
                cr2.brand_norm AS related_brand_norm
            FROM InitialOENumbers i
            JOIN cross_references cr2 ON i.oe_number_norm = cr2.oe_number_norm
            WHERE NOT (i.artikul_norm = cr2.artikul_norm AND i.brand_norm = cr2.brand_norm)
        ),
        Level1OENumbers AS (
            SELECT DISTINCT
                l1.source_artikul_norm,
                l1.source_brand_norm,
                cr3.oe_number_norm
            FROM Level1Analogs l1
            JOIN cross_references cr3 ON l1.related_artikul_norm = cr3.artikul_norm 
                                        AND l1.related_brand_norm = cr3.brand_norm
            WHERE NOT EXISTS (
                SELECT 1 FROM InitialOENumbers i 
                WHERE i.artikul_norm = l1.source_artikul_norm 
                AND i.brand_norm = l1.source_brand_norm 
                AND i.oe_number_norm = cr3.oe_number_norm
            )
        ),
        Level2Analogs AS (
            SELECT DISTINCT
                loe.source_artikul_norm,
                loe.source_brand_norm,
                cr4.artikul_norm AS related_artikul_norm,
                cr4.brand_norm AS related_brand_norm
            FROM Level1OENumbers loe
            JOIN cross_references cr4 ON loe.oe_number_norm = cr4.oe_number_norm
            WHERE NOT (loe.source_artikul_norm = cr4.artikul_norm AND loe.source_brand_norm = cr4.brand_norm)
        ),
        AllRelatedParts AS (
            SELECT DISTINCT source_artikul_norm, source_brand_norm, related_artikul_norm, related_brand_norm
            FROM Level1Analogs
            UNION
            SELECT DISTINCT source_artikul_norm, source_brand_norm, related_artikul_norm, related_brand_norm
            FROM Level2Analogs
        ),
        AggregatedAnalogData AS (
            SELECT
                arp.source_artikul_norm AS artikul_norm,
                arp.source_brand_norm AS brand_norm,
                MAX(CASE WHEN p2.length IS NOT NULL THEN p2.length ELSE NULL END) AS length,
                MAX(CASE WHEN p2.width IS NOT NULL THEN p2.width ELSE NULL END) AS width,
                MAX(CASE WHEN p2.height IS NOT NULL THEN p2.height ELSE NULL END) AS height,
                MAX(CASE WHEN p2.weight IS NOT NULL THEN p2.weight ELSE NULL END) AS weight,
                ANY_VALUE(CASE WHEN p2.dimensions_str IS NOT NULL 
                               AND p2.dimensions_str != '' 
                               AND UPPER(TRIM(p2.dimensions_str)) != 'XX' 
                          THEN p2.dimensions_str ELSE NULL END) AS dimensions_str,
                ANY_VALUE(CASE WHEN pd2.representative_name IS NOT NULL AND pd2.representative_name != '' THEN pd2.representative_name ELSE NULL END) AS representative_name,
                ANY_VALUE(CASE WHEN pd2.representative_applicability IS NOT NULL AND pd2.representative_applicability != '' THEN pd2.representative_applicability ELSE NULL END) AS representative_applicability,
                ANY_VALUE(CASE WHEN pd2.representative_category IS NOT NULL AND pd2.representative_category != '' THEN pd2.representative_category ELSE NULL END) AS representative_category
            FROM AllRelatedParts arp
            JOIN parts_data p2 ON arp.related_artikul_norm = p2.artikul_norm AND arp.related_brand_norm = p2.brand_norm
            LEFT JOIN PartDetails pd2 ON p2.artikul_norm = pd2.artikul_norm AND p2.brand_norm = pd2.brand_norm
            GROUP BY arp.source_artikul_norm, arp.source_brand_norm
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
                ROW_NUMBER() OVER(PARTITION BY p.artikul_norm, p.brand_norm ORDER BY pd.representative_name DESC NULLS LAST, pd.oe_list DESC NULLS LAST) as rn
            FROM parts_data p
            LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
            LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
            LEFT JOIN AggregatedAnalogData p_analog ON p.artikul_norm = p_analog.artikul_norm AND p.brand_norm = p_analog.brand_norm
        )
        """

        select_clause = ",\n            ".join(selected_exprs)

        # Итоговый запрос
        query = ctes + r"""
        SELECT
            """ + select_clause + r"""
        FROM RankedData r
        CROSS JOIN DescriptionTemplate dt
        WHERE r.rn = 1
        ORDER BY r.brand, r.artikul
        """

        return query

    def export_to_csv_optimized(self, output_path: str, selected_columns: List[str] | None = None):
        total_records = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📤 Экспорт {total_records:,} записей в CSV...")
        try:
            query = self.build_export_query(selected_columns)
            df = self.conn.execute(query).pl()

            # Выровнять цены с учетом наценки
            if 'price' in df.columns:
                df = df.with_columns(
                    pl.col('price').apply(lambda p: self.apply_markup(p, brand_norm='')).alias('Цена с наценкой')
                )

            # В случае, если есть колонка 'Цена с наценкой', переименовать
            if 'Цена с наценкой' in df.columns:
                df = df.rename({'Цена с наценкой': 'price_with_markup'})

            # Обработка числовых колонок
            for col_name in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if col_name in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col_name).is_not_null())
                        .then(pl.col(col_name).cast(pl.Utf8))
                        .otherwise("")
                        .alias(col_name)
                    )

            buf = io.StringIO()
            df.write_csv(buf, separator=';')
            csv_bytes = buf.getvalue().encode('utf-8')
            with open(output_path, 'wb') as f:
                f.write(b'\xef\xbb\xbf')
                f.write(csv_bytes)
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            st.success(f"✅ Данные экспортированы: {output_path} ({size_mb:.2f} МБ)")
            return True
        except Exception as e:
            st.exception(e)
            return False

    def export_to_excel(self, output_path: Path, selected_columns: List[str] | None = None):
        total_records = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False, None

        num_files = (total_records + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
        progress = st.progress(0, text=f"Подготовка {num_files} файла(ов)...")
        exported_files = []

        base_query = self.build_export_query(selected_columns)

        for i in range(num_files):
            offset = i * EXCEL_ROW_LIMIT
            query = f"{base_query} LIMIT {EXCEL_ROW_LIMIT} OFFSET {offset}"
            df = self.conn.execute(query).pl()

            # Обработка цен
            if 'price' in df.columns:
                df = df.with_columns(
                    pl.col('price').apply(lambda p: self.apply_markup(p, brand_norm='')).alias('Цена с наценкой')
                )

            # Обработка числовых колонок
            for col_name in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if col_name in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col_name).is_not_null())
                        .then(pl.col(col_name).cast(pl.Utf8))
                        .otherwise("")
                        .alias(col_name)
                    )

            file_name = output_path.with_name(f"{output_path.stem}_part_{i + 1}.xlsx")
            df.write_excel(str(file_name))
            exported_files.append(file_name)
            progress.progress((i + 1) / num_files)

        # Архивация, если несколько
        if len(exported_files) > 1:
            zip_path = output_path.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in exported_files:
                    zf.write(f, arcname=f.name)
                    os.remove(f)
            final_path = zip_path
        else:
            final_path = exported_files[0]
            if final_path != output_path:
                os.rename(final_path, output_path)
                final_path = output_path
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        st.success(f"✅ Экспорт завершен: {final_path.name} ({size_mb:.2f} МБ)")
        return True, final_path

    def export_to_parquet(self, output_path: str, selected_columns: List[str] | None = None):
        total_records = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📤 Экспорт {total_records:,} в Parquet...")
        try:
            query = self.build_export_query(selected_columns)
            df = self.conn.execute(query).pl()
            if 'price' in df.columns:
                df = df.with_columns(
                    pl.col('price').apply(lambda p: self.apply_markup(p, brand_norm='')).alias('Цена с наценкой')
                )
            df.write_parquet(output_path)
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            st.success(f"✅ Экспорт завершен: {output_path} ({size_mb:.2f} МБ)")
            return True
        except Exception as e:
            st.exception(e)
            return False

    # --- интерфейс ---
    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total_records = self.conn.execute("SELECT count(DISTINCT (artikul_norm, brand_norm)) FROM parts_data").fetchone()[0]
        st.info(f"Всего записей для экспорта: {total_records:,}")
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return

        available_columns = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена с наценкой"
        ]
        selected_columns = st.multiselect("Выберите столбцы для экспорта", options=available_columns, default=available_columns)

        # Настройки фильтров исключений
        st.subheader("Фильтр по исключаемым наименованиям")
        exclude_names_input = st.text_input("Исключить позиции, разделённые | (частичный поиск)")
        exclude_names = [name.strip() for name in exclude_names_input.split('|')] if exclude_names_input else []

        # Выбор формата
        export_format = st.radio("Формат экспорта", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)

        # Общая наценка
        st.subheader("Настройка наценки")
        total_markup = st.slider("Общая наценка (%)", 0, 100, 0)
        brand_markups: Dict[str, float] = {}
        if st.checkbox("Настроить наценки по брендам"):
            brands = self.conn.execute("SELECT DISTINCT brand, brand_norm FROM parts_data WHERE brand IS NOT NULL").fetchall()
            for b, bn in brands:
                markup = st.slider(f"Наценка для бренда '{b}'", 0, 100, 0)
                brand_markups[bn] = markup
        if st.button("Сохранить настройки наценки"):
            self.set_markups(total_markup, brand_markups)

        # Обработка кнопки
        if st.button("🚀 Экспортировать", type="primary"):
            output_path = self.data_dir / "auto_parts_export"
            if export_format == "CSV":
                output_file = str(output_path.with_suffix('.csv'))
                success = self.export_to_csv_optimized(output_file, selected_columns)
                if success:
                    with open(output_file, "rb") as f:
                        st.download_button("📥 Скачать CSV", f, "auto_parts_report.csv", "text/csv")
            elif export_format == "Excel (.xlsx)":
                output_file = output_path.with_suffix('.xlsx')
                success, final_path = self.export_to_excel(output_file, selected_columns)
                if success and final_path and final_path.exists():
                    with open(final_path, "rb") as f:
                        st.download_button("📥 Скачать XLSX", f, final_path.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            elif export_format == "Parquet":
                output_file = str(output_path.with_suffix('.parquet'))
                success = self.export_to_parquet(output_file, selected_columns)
                if success:
                    with open(output_file, "rb") as f:
                        st.download_button("📥 Скачать Parquet", f, "auto_parts_report.parquet", "application/octet-stream")

    # --- управление операциями удаления ---
    def delete_by_brand(self, brand_norm: str) -> int:
        try:
            count = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE brand_norm = ?", [brand_norm]).fetchone()[0]
            if count:
                self.conn.execute("DELETE FROM parts_data WHERE brand_norm = ?", [brand_norm])
                self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT artikul_norm, brand_norm FROM parts_data)")
            return count
        except Exception as e:
            logger.exception(f"Ошибка удаления по бренду: {e}")
            return 0

    def delete_by_artikul(self, artikul_norm: str) -> int:
        try:
            count = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE artikul_norm = ?", [artikul_norm]).fetchone()[0]
            if count:
                self.conn.execute("DELETE FROM parts_data WHERE artikul_norm = ?", [artikul_norm])
                self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT artikul_norm, brand_norm FROM parts_data)")
            return count
        except Exception as e:
            logger.exception(f"Ошибка удаления по артикула: {e}")
            return 0

# --- Основной интерфейс ---
def main():
    st.title("🚗 AutoParts Catalog - Профессиональная система для 10+ млн записей")
    st.markdown("""
    ### 💪 Мощная платформа для управления большими объемами данных автозапчастей
    - **Инкрементальные обновления**: добавляйте новые файлы безопасно.
    - **Объединение данных**: 5 типов файлов в единую базу.
    - **Оптимизация хранения**: DuckDB.
    - **Умный экспорт**: CSV, Excel, Parquet.
    """)

    catalog = HighVolumeAutoPartsCatalog()

    # Навигация
    menu = st.sidebar.radio("Выберите раздел", ["Загрузка данных", "Экспорт", "Статистика", "Управление данными"])

    if menu == "Загрузка данных":
        # Загрузка файлов
        is_empty_db = catalog.get_total_records() == 0
        st.subheader("Загрузка и подготовка данных")
        col1, col2 = st.columns(2)
        with col1:
            file_oe = st.file_uploader("Основные данные (OE)", type=['xlsx', 'xls'])
            file_cross = st.file_uploader("Кроссы (OE -> Артикул)", type=['xlsx', 'xls'])
            file_barcode = st.file_uploader("Штрих-коды", type=['xlsx', 'xls'])
        with col2:
            file_dim = st.file_uploader("Весогабариты", type=['xlsx', 'xls'])
            file_img = st.file_uploader("Изображения", type=['xlsx', 'xls'])

        files_map = {
            'oe': file_oe,
            'cross': file_cross,
            'barcode': file_barcode,
            'dimensions': file_dim,
            'images': file_img
        }

        if st.button("🚀 Начать загрузку"):
            paths = {}
            for ftype, uploaded in files_map.items():
                if uploaded:
                    filename = f"{ftype}_{int(time.time())}_{uploaded.name}"
                    path = catalog.data_dir / filename
                    with open(path, "wb") as f:
                        f.write(uploaded.getvalue())
                    paths[ftype] = str(path)

            # Проверка для начальной
            if is_empty_db:
                missing = [f for f in ['oe', 'cross', 'barcode', 'dimensions', 'images'] if f not in paths]
                if missing:
                    st.error(f"Для начальной загрузки нужны все 5 файлов. Отсутствуют: {', '.join(missing)}")
                elif len(paths) == 5:
                    stats = catalog.merge_all_data_parallel(paths)
                    if stats:
                        st.subheader("📊 Статистика")
                        st.metric("Время", f"{stats.get('processing_time', 0):.2f} сек")
                        st.metric("Артикулов", f"{stats.get('total_records', 0):,}")
                else:
                    st.warning("Загружено не все обязательные файлы.")
            else:
                if len(paths) > 0:
                    stats = catalog.merge_all_data_parallel(paths)
                    if stats:
                        st.subheader("📊 Статистика")
                        st.metric("Время", f"{stats.get('processing_time', 0):.2f} сек")
                        st.metric("Артикулов", f"{stats.get('total_records', 0):,}")
                else:
                    st.info("Загрузите файлы для добавления.")

    elif menu == "Экспорт":
        catalog.show_export_interface()

    elif menu == "Статистика":
        st.header("📈 Статистика")
        with st.spinner("Сбор данных..."):
            stats = catalog.get_statistics()
        st.metric("Всего артикула", f"{stats.get('total_parts', 0):,}")
        st.metric("OE", f"{stats.get('total_oe', 0):,}")
        st.metric("Брендов", f"{stats.get('total_brands', 0):,}")
        st.subheader("Топ брендов")
        st.dataframe(stats.get('top_brands', pl.DataFrame()).to_pandas())
        st.subheader("Распределение по категориям")
        st.bar_chart(stats.get('categories', pl.DataFrame()).to_pandas().set_index('category'))

    elif menu == "Управление данными":
        st.header("🗑️ Управление")
        op = st.radio("Действие", ["Удалить по бренду", "Удалить по артикулу"])
        if op == "Удалить по бренду":
            brands = catalog.conn.execute("SELECT DISTINCT brand, brand_norm FROM parts_data WHERE brand IS NOT NULL").fetchall()
            if brands:
                brand_list = [b for b, bn in brands]
                selected_b = st.selectbox("Выберите бренд", brand_list)
                # Получение normalized
                bn_row = catalog.conn.execute("SELECT brand_norm FROM parts_data WHERE brand=?", [selected_b]).fetchone()
                bn = bn_row[0] if bn_row else ''
                count_del = catalog.delete_by_brand(bn)
                st.success(f"Удалено {count_del} записей для бренда {selected_b}")
            else:
                st.info("Нет брендов для удаления.")
        else:
            art_input = st.text_input("Артикул для удаления")
            if art_input:
                norm_series = catalog.normalize_key(pl.Series([art_input]))
                artikul_norm = norm_series[0] if len(norm_series) > 0 else ''
                count = catalog.delete_by_artikul(artikul_norm)
                st.success(f"Удалено {count} записей по артикулу {art_input}")

if __name__ == "__main__":
    main()
