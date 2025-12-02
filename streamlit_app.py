import polars as pl
import duckdb
import streamlit as st
import os
import time
import io
import zipfile
from pathlib import Path
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
EXCEL_ROW_LIMIT = 1_000_000

class HighVolumeAutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        self.setup_database()

        st.set_page_config(
            page_title="AutoParts Catalog 10M+", 
            layout="wide",
            page_icon="🚗"
        )

    def setup_database(self):
        # Создаем таблицы, если не существуют
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
            CREATE TABLE IF NOT EXISTS prices (
                artikul VARCHAR PRIMARY KEY,
                recommended_price DOUBLE,
                brand VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS supplier_prices (
                artikul VARCHAR,
                quantity INTEGER,
                brand VARCHAR,
                supplier_price DOUBLE,
                PRIMARY KEY (artikul, brand)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                key VARCHAR PRIMARY KEY,
                name VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS markup_settings (
                id INTEGER PRIMARY KEY,
                global_markup DOUBLE
            )
        """)
        # Инициализация глобальной скидки (если нужно)
        self.conn.execute("INSERT OR IGNORE INTO markup_settings (id, global_markup) VALUES (1, 0.0)")

    def create_indexes(self):
        # Создаем индексы для ускорения поиска
        st.info("Создаю индексы для базы данных...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_prices ON prices(artikul)"
        ]
        for sql in indexes:
            self.conn.execute(sql)
        st.success("Индексы созданы.")

    @staticmethod
    def normalize_key(key_series: pl.Series) -> pl.Series:
        # Нормализация ключей
        return (
            key_series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all("'", "")
            .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .str.to_lowercase()
        )

    @staticmethod
    def clean_values(value_series: pl.Series) -> pl.Series:
        # Очистка значений
        return (
            value_series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all("'", "")
            .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    @staticmethod
    def determine_category_vectorized(name_series: pl.Series) -> pl.Series:
        # Определение категории по названию
        categories_map = {
            'Фильтр': 'фильтр|filter', 'Тормоза': 'тормоз|brake|колодк|диск|суппорт',
            'Подвеска': 'амортизатор|стойк|spring|подвеск|рычаг', 'Двигатель': 'двигатель|engine|свеч|поршень|клапан',
            'Трансмиссия': 'трансмиссия|сцеплен|коробк|transmission', 'Электрика': 'аккумулятор|генератор|стартер|провод|ламп',
            'Рулевое': 'рулевой|тяга|наконечник|steering', 'Выпуск': 'глушитель|катализатор|выхлоп|exhaust',
            'Охлаждение': 'радиатор|вентилятор|термостат|cooling', 'Топливо': 'топливный|бензонасос|форсунк|fuel'
        }
        name_lower = name_series.str.to_lowercase()
        categorization_expr = pl.when(pl.lit(False)).then(pl.lit(None))
        for category, pattern in categories_map.items():
            categorization_expr = categorization_expr.when(name_lower.str.contains(pattern)).then(pl.lit(category))
        return categorization_expr.otherwise(pl.lit('Разное')).alias('category')

    def detect_columns(self, actual_columns: list, expected_columns: list) -> dict:
        # Детектирование колонок по вариациям
        mapping = {}
        col_variants = {
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
            variants = [v.lower() for v in col_variants.get(expected, [expected])]
            for variant in variants:
                for actual_l, actual_orig in actual_lower.items():
                    if variant in actual_l:
                        mapping[actual_orig] = expected
                        break
                if expected in mapping.values():
                    break
        return mapping

    def read_and_prepare_file(self, file_path: str, file_type: str) -> pl.DataFrame:
        logger.info(f"Обработка файла: {file_type} ({file_path})")
        try:
            df = pl.read_excel(file_path, engine='calamine')
            if df.is_empty():
                return pl.DataFrame()
        except Exception as e:
            logger.exception(f"Ошибка чтения файла {file_path}: {e}")
            return pl.DataFrame()

        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand']
        }
        expected_cols = schemas.get(file_type, [])
        col_map = self.detect_columns(df.columns, expected_cols)
        if not col_map:
            return pl.DataFrame()
        df = df.rename(col_map)

        # Очистка значений
        if 'artikul' in df.columns:
            df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand=self.clean_values(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))
        # Создаем нормализованные ключи
        if 'artikul' in df.columns:
            df = df.with_columns(artikul_norm=self.normalize_key(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand_norm=self.normalize_key(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number_norm=self.normalize_key(pl.col('oe_number')))
        # Удаление дубликатов по ключам
        key_cols = [col for col in ['oe_number', 'artikul', 'brand'] if col in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')
        return df

    def upsert_data(self, table_name: str, df: pl.DataFrame, pk: list):
        if df.is_empty():
            return
        df = df.unique(keep='first')
        cols = df.columns
        pk_str = ", ".join(f'"{c}"' for c in pk)
        temp_view_name = f"temp_{table_name}_{int(time.time())}"
        self.conn.register(temp_view_name, df.to_arrow())

        # Создание SQL-запроса для UPSERT
        update_cols = [col for col in cols if col not in pk]
        if update_cols:
            set_clause = ", ".join([f'"{col}"=excluded."{col}"' for col in update_cols])
            on_conflict = f"ON CONFLICT ({pk_str}) DO UPDATE SET {set_clause}"
        else:
            on_conflict = "ON CONFLICT DO NOTHING"

        sql = f"""
        INSERT INTO {table_name}
        SELECT * FROM {temp_view_name}
        {on_conflict};
        """
        try:
            self.conn.execute(sql)
        finally:
            self.conn.unregister(temp_view_name)

    def process_and_load_data(self, dataframes: dict):
        st.info("🔄 Начинаю обработку и обновление базы...")
        total_steps = 3
        progress_bar = st.progress(0, text="Обработка данных...")
        step = 0

        # Обработка OE
        if 'oe' in dataframes:
            step += 1
            progress_bar.progress(step / total_steps, text=f"({step}/{total_steps}) Обработка OE")
            df = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'])
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self.determine_category_vectorized(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])

            cross_df = df.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка кроссов
        if 'cross' in dataframes:
            step += 1
            progress_bar.progress(step / total_steps, f"({step}/{total_steps}) Обработка кроссов")
            df = dataframes['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            self.upsert_data('cross_references', df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка артикула и данных по ним
        step += 1
        progress_bar.progress(step / total_steps, f"({step}/{total_steps}) Обработка артикула")
        all_parts = None
        for key in ['oe', 'barcode', 'images', 'dimensions']:
            df = dataframes.get(key)
            if df is None or df.is_empty():
                continue
            if 'artikul_norm' in df.columns:
                temp = df.select(['artikul', 'artikul_norm', 'brand', 'brand_norm'])
                if all_parts is None:
                    all_parts = temp
                else:
                    all_parts = all_parts.join(temp, on=['artikul_norm', 'brand_norm'], how='left', coalesce=True)

        if all_parts is not None and not all_parts.is_empty():
            # Обработка размеров
            for col in ['length', 'width', 'height']:
                if col not in all_parts.columns:
                    all_parts = all_parts.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
            if 'dimensions_str' not in all_parts.columns:
                all_parts = all_parts.with_columns(dimensions_str=pl.lit(None).cast(pl.Utf8))
            # Создаём строку размеров
            all_parts = all_parts.with_columns([
                pl.col('length').cast(pl.Utf8).fill_null('0').alias('_length_str'),
                pl.col('width').cast(pl.Utf8).fill_null('0').alias('_width_str'),
                pl.col('height').cast(pl.Utf8).fill_null('0').alias('_height_str'),
            ])
            all_parts = all_parts.with_columns(
                pl.when(pl.col('dimensions_str').is_not_null() & (pl.col('dimensions_str') != ''))
                .then(pl.col('dimensions_str'))
                .otherwise(
                    pl.concat_str([pl.col('_length_str'), pl.lit('x'), pl.col('_width_str'), pl.lit('x'), pl.col('_height_str')], separator='')
                ).alias('dimensions_str')
            ).drop(['_length_str', '_width_str', '_height_str'])

            # Обработка description
            if 'artikul' not in all_parts.columns:
                all_parts = all_parts.with_columns(artikul=pl.lit(''))
            if 'brand' not in all_parts.columns:
                all_parts = all_parts.with_columns(brand=pl.lit(''))

            all_parts = all_parts.with_columns([
                pl.col('artikul').cast(pl.Utf8).fill_null(''),
                pl.col('brand').cast(pl.Utf8).fill_null(''),
                pl.col('multiplicity').cast(pl.Int32).fill_null(1),
                pl.col('length').cast(pl.Float64).fill_null(0),
                pl.col('width').cast(pl.Float64).fill_null(0),
                pl.col('height').cast(pl.Float64).fill_null(0),
                pl.col('weight').cast(pl.Float64).fill_null(0),
            ])

            # Формируем описание
            all_parts = all_parts.with_columns(
                description=pl.concat_str([
                    pl.lit('Артикул: '), pl.col('artikul'), pl.lit(', Бренд: '), pl.col('brand'),
                    pl.lit(', Кратность: '), pl.col('multiplicity').cast(pl.Utf8), pl.lit(' шт.')
                ], separator='')
            )

            # Финальные колонки
            final_cols = [
                'artikul_norm', 'brand_norm', 'artikul', 'brand', 'multiplicity', 'barcode',
                'length', 'width', 'height', 'weight', 'image_url', 'dimensions_str', 'description'
            ]
            for c in final_cols:
                if c not in all_parts.columns:
                    all_parts = all_parts.with_columns(pl.lit(None).alias(c))
            all_parts = all_parts.select([pl.col(c) if c in all_parts.columns else pl.lit(None).alias(c) for c in final_cols])

            self.upsert_data('parts_data', all_parts, ['artikul_norm', 'brand_norm'])

        progress_bar.progress(1.0)
        time.sleep(1)
        progress_bar.empty()
        st.success("✅ Обработка и загрузка завершены.")

    def merge_all_data_parallel(self, file_paths: dict):
        start_time = time.time()
        stats = {}
        st.info("🚀 Начинаю параллельную обработку файлов...")
        n_files = len(file_paths)
        progress_bar = st.progress(0)
        processed = 0

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.read_and_prepare_file, path, ftype): ftype
                for ftype, path in file_paths.items()
            }
            dataframes = {}
            for future in futures:
                ftype = futures[future]
                try:
                    df = future.result()
                    if not df.is_empty():
                        dataframes[ftype] = df
                        st.success(f"Файл {ftype} обработан: {len(df):,} строк")
                    else:
                        st.warning(f"Файл {ftype} пуст или не обработан.")
                except Exception as e:
                    st.error(f"Ошибка при обработке файла {ftype}: {e}")
                processed += 1
                progress_bar.progress(processed / n_files)
        progress_bar.empty()

        if not dataframes:
            st.warning("Нет данных для обработки.")
            return {}

        self.process_and_load_data(dataframes)

        stats['processing_time'] = time.time() - start_time
        stats['total_records'] = self.get_total_records()

        st.success(f"Обработка завершена за {stats['processing_time']:.2f} сек")
        st.success(f"Всего артикулов: {stats['total_records']:,}")
        self.create_indexes()
        return stats

    def get_total_records(self):
        try:
            res = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()
            return res[0] if res else 0
        except Exception:
            return 0

    def get_statistics(self):
        stats = {}
        try:
            stats['total_parts'] = self.get_total_records()
            if stats['total_parts'] == 0:
                return {
                    'total_parts': 0, 'total_oe': 0, 'total_brands': 0,
                    'top_brands': pl.DataFrame(), 'categories': pl.DataFrame()
                }
            total_oe_res = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()
            stats['total_oe'] = total_oe_res[0] if total_oe_res else 0

            total_brands_res = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data WHERE brand IS NOT NULL").fetchone()
            stats['total_brands'] = total_brands_res[0] if total_brands_res else 0

            # Топ брендов
            brands = self.conn.execute("SELECT brand, COUNT(*) FROM parts_data WHERE brand IS NOT NULL GROUP BY brand ORDER BY COUNT(*) DESC LIMIT 10").fetchall()
            if brands:
                stats['top_brands'] = pl.DataFrame(brands, schema=["brand", "count"])
            else:
                stats['top_brands'] = pl.DataFrame(schema=["brand", "count"])

            # Категории
            categories = self.conn.execute("SELECT category, COUNT(*) FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY COUNT(*) DESC").fetchall()
            if categories:
                stats['categories'] = pl.DataFrame(categories, schema=["category", "count"])
            else:
                stats['categories'] = pl.DataFrame(schema=["category", "count"])

        except Exception as e:
            logger.exception("Ошибка при сборе статистики")
            return {
                'total_parts': 0, 'total_oe': 0, 'total_brands': 0,
                'top_brands': pl.DataFrame(), 'categories': pl.DataFrame()
            }
        return stats

    # --- Методы экспорта ---
    def build_export_query(self, selected_columns: list = None):
        standard_description = """Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""
        # Отображение колонок
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

        select_exprs = ",\n            ".join(selected_exprs)

        query = ctes + f"""
        SELECT
            {select_exprs}
        FROM RankedData r
        CROSS JOIN DescriptionTemplate dt
        WHERE r.rn = 1
        ORDER BY r.brand, r.artikul
        """

        return query

    def export_to_csv(self, output_path: str, selected_columns: list = None):
        total = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        try:
            query = self.build_export_query(selected_columns)
            df = self.conn.execute(query).pl()
            # Обработка числовых колонок
            for col_name in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if col_name in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col_name).is_not_null())
                        .then(pl.col(col_name).cast(pl.Utf8))
                        .otherwise('')
                        .alias(col_name)
                    )
            buf = io.StringIO()
            df.write_csv(buf, separator=';')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\ufeff')  # BOM
                f.write(buf.getvalue())
            st.success(f"Экспорт завершен: {output_path}")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта в CSV")
            st.error(f"Ошибка при экспорте: {e}")
            return False

    def export_to_excel(self, output_path: Path, selected_columns: list = None):
        total = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False, None
        num_files = (total + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
        files = []
        for i in range(num_files):
            offset = i * EXCEL_ROW_LIMIT
            query = f"{self.build_export_query(selected_columns)} LIMIT {EXCEL_ROW_LIMIT} OFFSET {offset}"
            df = self.conn.execute(query).pl()
            # Обработка размеров
            for col in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if col in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col).is_not_null())
                        .then(pl.col(col).cast(pl.Utf8))
                        .otherwise('')
                        .alias(col)
                    )
            file_part = output_path.with_name(f"{output_path.stem}_part_{i+1}.xlsx")
            df.write_excel(str(file_part))
            files.append(file_part)
        if len(files) > 1:
            zip_path = output_path.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in files:
                    zf.write(str(f), f.name)
                    os.remove(f)
            return True, zip_path
        else:
            os.rename(str(files[0]), str(output_path))
            return True, output_path

    def export_to_parquet(self, output_path: str, selected_columns: list = None):
        total = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        try:
            query = self.build_export_query(selected_columns)
            df = self.conn.execute(query).pl()
            df.write_parquet(output_path)
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            st.success(f"Экспорт завершен: {output_path} ({size_mb:.2f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта в Parquet")
            st.error(f"Ошибка при экспорте: {e}")
            return False

    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total = self.conn.execute("SELECT count(DISTINCT (artikul_norm, brand_norm)) FROM parts_data").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта.")
            return
        options_cols = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение"
        ]
        selected_cols = st.multiselect("Выберите колонки для экспорта", options=options_cols, default=options_cols)
        format_choice = st.radio("Формат экспорта:", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)
        if format_choice == "CSV":
            if st.button("🚀 Экспортировать в CSV"):
                output_path = self.data_dir / "auto_parts_export.csv"
                success = self.export_to_csv(str(output_path), selected_cols)
                if success:
                    with open(output_path, "rb") as f:
                        st.download_button("📥 Скачать CSV", f, "auto_parts_export.csv", "text/csv")
        elif format_choice == "Excel (.xlsx)":
            if st.button("📊 Экспортировать в Excel"):
                output_path = self.data_dir / "auto_parts_export.xlsx"
                success, file_path = self.export_to_excel(output_path, selected_cols)
                if success and file_path:
                    with open(file_path, "rb") as f:
                        mime_type = "application/zip" if file_path.suffix == ".zip" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        st.download_button(f"📥 Скачать {file_path.name}", f, file_path.name, mime_type)
        elif format_choice == "Parquet":
            if st.button("⚡️ Экспортировать в Parquet"):
                output_path = self.data_dir / "auto_parts_export.parquet"
                success = self.export_to_parquet(str(output_path), selected_cols)
                if success:
                    with open(output_path, "rb") as f:
                        st.download_button("📥 Скачать Parquet", f, "auto_parts_export.parquet", "application/octet-stream")

    def delete_by_brand(self, brand_norm: str):
        # Удаление по бренду
        try:
            count_res = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE brand_norm = ?", [brand_norm]).fetchone()
            count_deleted = count_res[0] if count_res else 0
            if count_deleted == 0:
                st.info("Нет записей для удаления по этому бренду.")
                return 0
            self.conn.execute("DELETE FROM parts_data WHERE brand_norm = ?", [brand_norm])
            self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)")
            return count_deleted
        except Exception as e:
            logger.exception("Ошибка удаления по бренду")
            return 0

    def delete_by_artikul(self, artikul_norm: str):
        # Удаление по артикула
        try:
            res = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE artikul_norm = ?", [artikul_norm]).fetchone()
            count_deleted = res[0] if res else 0
            if count_deleted == 0:
                st.info("Нет записей для удаления по этому артикула.")
                return 0
            self.conn.execute("DELETE FROM parts_data WHERE artikul_norm = ?", [artikul_norm])
            self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)")
            return count_deleted
        except Exception as e:
            logger.exception("Ошибка удаления по артикула")
            return 0

# --- Основная функция интерфейса ---
def main():
    st.title("🚗 AutoParts Catalog - Профессиональная система для 10+ млн записей")
    st.markdown("""
    ### 💪 Мощная платформа для управления большими объемами данных автозапчастей
    - **Инкрементальные обновления**: Безопасно добавляйте новые файлы для дополнения и обновления каталога.
    - **Надежное объединение**: Данные из 5-ти типов файлов корректно сливаются в единую базу.
    - **Оптимизированное хранение**: Использование DuckDB для мгновенного доступа и анализа.
    - **Умный экспорт**: Быстрый и надежный экспорт в CSV, Excel или Parquet с гарантией отсутствия дубликатов.
    """)
    catalog = HighVolumeAutoPartsCatalog()

    # Навигация
    st.sidebar.title("🧭 Навигация")
    menu_option = st.sidebar.radio("Выберите действие:", ["Загрузка данных", "Экспорт", "Статистика", "Управление данными"])

    if menu_option == "Загрузка данных":
        # Загрузка и подготовка файлов
        st.header("📥 Загрузка и обработка данных")
        is_db_empty = catalog.get_total_records() == 0
        if is_db_empty:
            st.warning("⚠️ База пуста. Требуется начальная загрузка всех 5 файлов.")
            st.info("""
            **Типы файлов для начальной загрузки:**
            1. Основные данные (OE)
            2. Кроссы (OE -> Артикул)
            3. Штрих-коды
            4. Весогабариты
            5. Изображения
            """)
        else:
            st.success("✅ База содержит данные. Можно добавлять файлы по одному или нескольким одновременно.")
            st.info("💡 Для добавления новых данных загружайте файлы, они будут объединены и обновлены.")

        col1, col2 = st.columns(2)
        with col1:
            oe_file = st.file_uploader("1. Основные данные (OE)", type=['xlsx', 'xls'])
            cross_file = st.file_uploader("2. Кроссы (OE -> Артикул)", type=['xlsx', 'xls'])
            barcode_file = st.file_uploader("3. Штрих-коды", type=['xlsx', 'xls'])
        with col2:
            dimensions_file = st.file_uploader("4. Весогабариты", type=['xlsx', 'xls'])
            images_file = st.file_uploader("5. Изображения", type=['xlsx', 'xls'])

        file_map = {
            'oe': oe_file,
            'cross': cross_file,
            'barcode': barcode_file,
            'dimensions': dimensions_file,
            'images': images_file
        }

        if st.button("🚀 Начать обработку данных"):
            paths = {}
            count_files = 0
            for key, upl in file_map.items():
                if upl:
                    filename = f"{key}_{int(time.time())}_{upl.name}"
                    path = catalog.data_dir / filename
                    with open(path, "wb") as f:
                        f.write(upl.getvalue())
                    paths[key] = str(path)
                    count_files += 1

            if catalog.get_total_records() == 0:
                # Начальная загрузка
                required_files = ['oe', 'cross', 'barcode', 'dimensions', 'images']
                missing = [f for f in required_files if f not in paths]
                if missing:
                    st.error(f"❌ Для начальной загрузки необходимо загрузить все 5 файлов. Отсутствуют: {', '.join(missing)}")
                elif count_files == len(required_files):
                    stats = catalog.merge_all_data_parallel(paths)
                    if stats:
                        st.subheader("📊 Статистика обработки")
                        st.metric("Общее время", f"{stats.get('processing_time', 0):.2f} сек")
                        st.metric("Всего артикулов", f"{stats.get('total_records', 0):,}")
                        st.metric("Обработано файлов", f"{len(paths)}")
                else:
                    st.error("❌ Не все файлы загружены для начальной загрузки.")
            else:
                # Дозагрузка
                if count_files > 0:
                    stats = catalog.merge_all_data_parallel(paths)
                    if stats:
                        st.subheader("📊 Статистика обработки")
                        st.metric("Общее время", f"{stats.get('processing_time', 0):.2f} сек")
                        st.metric("Всего артикулов", f"{stats.get('total_records', 0):,}")
                        st.metric("Обработано файлов", f"{len(paths)}")
                else:
                    st.warning("⚠️ Загрузите хотя бы один файл.")

    elif menu_option == "Экспорт":
        catalog.show_export_interface()

    elif menu_option == "Статистика":
        st.header("📈 Статистика по каталогу")
        with st.spinner("Сбор статистики..."):
            stats = catalog.get_statistics()
        if stats.get('total_parts', 0) > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Уникальных артикулов", f"{stats.get('total_parts', 0):,}")
            col2.metric("Уникальных OE", f"{stats.get('total_oe', 0):,}")
            col3.metric("Уникальных брендов", f"{stats.get('total_brands', 0):,}")

            st.subheader("🏆 Топ-10 брендов по артикулов")
            if 'top_brands' in stats and not stats['top_brands'].is_empty():
                st.dataframe(stats['top_brands'].to_pandas(), width='stretch')
            else:
                st.write("Нет данных по брендам.")
            st.subheader("📊 Распределение по категориям")
            if 'categories' in stats and not stats['categories'].is_empty():
                st.bar_chart(stats['categories'].to_pandas().set_index('category'))
            else:
                st.write("Нет данных по категориям.")
        else:
            st.info("Данных нет. Загрузите файлы.")

    elif menu_option == "Управление данными":
        st.header("🗑️ Управление данными")
        st.warning("⚠️ Осторожно! Операции необратимы.")
        option = st.radio("Выберите операцию:", ["Удалить по бренду", "Удалить по артикулу"])

        if option == "Удалить по бренду":
            # Получение брендов
            try:
                brands_res = catalog.conn.execute("SELECT DISTINCT brand FROM parts_data WHERE brand IS NOT NULL ORDER BY brand").fetchall()
                brands = [row[0] for row in brands_res]
            except:
                brands = []

            if brands:
                selected_brand = st.selectbox("Выберите бренд для удаления", brands)
                # Получение normalized
                res_norm = catalog.conn.execute("SELECT brand_norm FROM parts_data WHERE brand = ? LIMIT 1", [selected_brand]).fetchone()
                brand_norm = res_norm[0] if res_norm else ''
                # Подсчет
                count_res = catalog.conn.execute("SELECT COUNT(*) FROM parts_data WHERE brand_norm = ?", [brand_norm]).fetchone()
                count_del = count_res[0] if count_res else 0
                st.info(f"Удалить {count_del} записей для бренда '{selected_brand}'")
                if st.checkbox("Я подтверждаю удаление", key=f"del_brand_{selected_brand}"):
                    if st.button("❌ Удалить"):
                        deleted = catalog.delete_by_brand(brand_norm)
                        st.success(f"Удалено {deleted} записей.")
                        st.rerun()
            else:
                st.info("Нет доступных брендов.")

        elif option == "Удалить по артикулу":
            arti_input = st.text_input("Введите артикул для удаления")
            if arti_input:
                # normalize
                ser = pl.Series([arti_input])
                norm_ser = catalog.normalize_key(ser)
                artikul_norm = norm_ser[0] if len(norm_ser) > 0 else ''
                res = catalog.conn.execute("SELECT COUNT(*) FROM parts_data WHERE artikul_norm = ?", [artikul_norm]).fetchone()
                count_del = res[0] if res else 0
                if count_del > 0:
                    st.info(f"Удалить {count_del} записей артикула '{arti_input}'")
                    if st.checkbox("Я подтверждаю удаление", key=f"del_arti_{arti_input}"):
                        if st.button("❌ Удалить"):
                            deleted = catalog.delete_by_artikul(artikul_norm)
                            st.success(f"Удалено {deleted} записей.")
                            st.rerun()
                else:
                    st.warning("Артикул не найден.")

# Запуск
if __name__ == "__main__":
    main()
