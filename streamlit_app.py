import polars as pl
import duckdb
import streamlit as st
import os
import time
import io
import zipfile
import json
from pathlib import Path

EXCEL_ROW_LIMIT = 1_000_000

class AutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        self._setup_database()
        self._create_indexes()
        st.set_page_config(
            page_title="AutoParts Catalog 10M+",
            layout="wide",
            page_icon="🚗"
        )

    def _setup_database(self):
        # Создание таблиц
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
        # Таблица для цен рекомендаций
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS recommended_prices (
                artikul_norm VARCHAR PRIMARY KEY,
                price DOUBLE
            )
        """)
        # Таблица для прайс-листа (артикул, бренд, кол-во, цена)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS price_list (
                artikul VARCHAR,
                brand VARCHAR,
                quantity INTEGER,
                price DOUBLE,
                PRIMARY KEY (artikul, brand)
            )
        """)
        # Таблица для настроек наценки
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS markup_settings (
                id INTEGER PRIMARY KEY,
                total_markup DOUBLE,
                brand_markup JSON
            )
        """)
        if not self.conn.execute("SELECT 1 FROM markup_settings").fetchone():
            self.conn.execute("INSERT INTO markup_settings (id, total_markup, brand_markup) VALUES (1, 0, '{}')")

    def _create_indexes(self):
        # Индексы для ускорения поиска
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_oe ON oe_data(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_parts ON parts_data(artikul_norm, brand_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross ON cross_references(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross_art ON cross_references(artikul_norm, brand_norm)")

    # --- Методы нормализации и очистки ---
    @staticmethod
    def normalize_key(series: pl.Series) -> pl.Series:
        return (
            series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all("'", "")
            .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .str.to_lowercase()
        )

    @staticmethod
    def clean_values(series: pl.Series) -> pl.Series:
        return (
            series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all("'", "")
            .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    def detect_columns(self, actual_cols, expected_cols):
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
        actual_lower = {c.lower(): c for c in actual_cols}
        for exp_col in expected_cols:
            variants = [v.lower() for v in col_variants.get(exp_col, [exp_col])]
            for var in variants:
                for act_lower, act_orig in actual_lower.items():
                    if var in act_lower:
                        mapping[act_orig] = exp_col
                        break
                if exp_col in mapping.values():
                    break
        return mapping

    # --- Обработка файлов ---
    def read_and_prepare_file(self, path, file_type):
        try:
            df = pl.read_excel(path, engine='calamine')
            if df.is_empty():
                return pl.DataFrame()
        except Exception as e:
            st.error(f"Ошибка чтения файла {path}: {e}")
            return pl.DataFrame()
        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand']
        }
        expected_cols = schemas.get(file_type, [])
        col_mapping = self.detect_columns(df.columns, expected_cols)
        if not col_mapping:
            return pl.DataFrame()
        df = df.rename(col_mapping)
        # Очистка данных
        if 'artikul' in df.columns:
            df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand=self.clean_values(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))
        # Уникальность по ключам
        key_cols = [c for c in ['oe_number', 'artikul', 'brand'] if c in df.columns]
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

    def upsert_data(self, table_name, df, pk):
        if df.is_empty():
            return
        df = df.unique(keep='first')
        self.conn.register(f"temp_{table_name}_{int(time.time())}", df.to_arrow())
        pk_str = ", ".join([f'"{col}"' for col in pk])
        update_cols = [col for col in df.columns if col not in pk]
        if update_cols:
            update_clause = ", ".join([f'"{col}"=excluded."{col}"' for col in update_cols])
            conflict_action = f"DO UPDATE SET {update_clause}"
        else:
            conflict_action = "DO NOTHING"
        sql = f"""
            INSERT INTO {table_name}
            SELECT * FROM "temp_{table_name}_{int(time.time())}"
            ON CONFLICT ({pk_str}) {conflict_action}
        """
        try:
            self.conn.execute(sql)
        except Exception as e:
            st.error(f"Ошибка при вставке/обновлении {table_name}: {e}")
        finally:
            self.conn.unregister(f"temp_{table_name}_{int(time.time())}")

    def process_and_load_data(self, dataframes):
        st.info("🔄 Обновление базы данных...")
        steps = ['oe', 'cross', 'parts']
        total_steps = len(steps)
        progress = st.progress(0)
        step_idx = 0

        # Обработка OE
        if 'oe' in dataframes:
            step_idx += 1
            progress.progress(step_idx / total_steps, text=f"Обработка OE ({step_idx}/{total_steps})")
            df_oe = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df_oe.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'])
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self.determine_category_vectorized(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])
            cross_df = df_oe.filter(pl.col('artikul_norm') != "").select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка cross
        if 'cross' in dataframes:
            step_idx += 1
            progress.progress(step_idx / total_steps, f"Обработка кроссов ({step_idx}/{total_steps})")
            df_cross = dataframes['cross'].filter(
                (pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != "")
            )
            self.upsert_data('cross_references', df_cross, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка parts
        step_idx += 1
        progress.progress(step_idx / total_steps, f"Обработка артикула ({step_idx}/{total_steps})")
        # Формируем объединённые данные артикула
        parts_df = None
        files_order = ['oe', 'barcode', 'images', 'dimensions']
        key_files = {f: dataframes[f] for f in files_order if f in dataframes}
        if key_files:
            combined = pl.concat([df.select(['artikul', 'artikul_norm', 'brand', 'brand_norm']) for df in key_files.values()]).filter(pl.col('artikul_norm') != "").unique(subset=['artikul', 'brand'])
            parts_df = combined

            # Объединение данных по файлам
            for ftype in files_order:
                df = key_files.get(ftype)
                if not df or df.is_empty():
                    continue
                if 'artikul_norm' not in df.columns:
                    continue
                # Объединение
                join_cols = [col for col in df.columns if col not in ['artikul', 'artikul_norm', 'brand', 'brand_norm']]
                existing_cols = set(parts_df.columns)
                join_cols = [col for col in join_cols if col not in existing_cols]
                if not join_cols:
                    continue
                df_select = df.select(['artikul_norm', 'brand_norm'] + join_cols).unique(subset=['artikul_norm', 'brand_norm'])
                parts_df = parts_df.join(df_select, on=['artikul_norm', 'brand_norm'], how='left', coalesce=True)

        # Обработка размеров, описание
        if parts_df and not parts_df.is_empty():
            # multiplicity
            if 'multiplicity' not in parts_df.columns:
                parts_df = parts_df.with_columns(pl.lit(1).cast(pl.Int32).alias('multiplicity'))
            else:
                parts_df = parts_df.with_columns(pl.col('multiplicity').fill_null(1).cast(pl.Int32))
            # размеры
            for c in ['length', 'width', 'height']:
                if c not in parts_df.columns:
                    parts_df = parts_df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
            if 'dimensions_str' not in parts_df.columns:
                parts_df = parts_df.with_columns(dimensions_str=pl.lit(None).cast(pl.Utf8))
            # создаем dimensions_str
            parts_df = parts_df.with_columns([
                pl.col('length').cast(pl.Utf8).fill_null('').alias('_length_str'),
                pl.col('width').cast(pl.Utf8).fill_null('').alias('_width_str'),
                pl.col('height').cast(pl.Utf8).fill_null('').alias('_height_str')
            ])
            parts_df = parts_df.with_columns(
                pl.when(
                    (pl.col('dimensions_str').is_not_null()) &
                    (pl.col('dimensions_str') != '') &
                    (pl.col('dimensions_str').cast(pl.Utf8).str.upper() != 'XX')
                )
                .then(pl.col('dimensions_str'))
                .otherwise(
                    pl.concat_str([pl.col('_length_str'), pl.lit('x'), pl.col('_width_str'), pl.lit('x'), pl.col('_height_str')], separator='')
                ).alias('dimensions_str')
            )
            parts_df = parts_df.drop(['_length_str', '_width_str', '_height_str'])

            # описание
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

            # В финальный набор колонок
            final_cols = [
                'artikul_norm', 'brand_norm', 'artikul', 'brand', 'multiplicity', 'barcode',
                'length', 'width', 'height', 'weight', 'image_url', 'dimensions_str', 'description'
            ]
            select_exprs = [pl.col(c) if c in parts_df.columns else pl.lit(None) for c in final_cols]
            parts_df = parts_df.select(select_exprs)

            # Вставка данных с ценами
            self.upsert_data('parts_data', parts_df, ['artikul_norm', 'brand_norm'])

        progress.progress(1.0)
        time.sleep(1)
        st.success("💾 Обновление базы завершено.")

    def merge_all_data_parallel(self, paths: Dict[str, str]):
        start_time = time.time()
        # Обработка файлов параллельно
        import concurrent.futures
        dataframes = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.read_and_prepare_file, p, t): t
                for t, p in paths.items()
            }
            for f in concurrent.futures.as_completed(futures):
                t = futures[f]
                df = f.result()
                if not df.is_empty():
                    dataframes[t] = df
        if not dataframes:
            st.warning("Нет данных для обработки.")
            return {}
        self.process_and_load_data(dataframes)
        return {
            'processing_time': time.time() - start_time,
            'total_records': self.get_total_records()
        }

    def get_total_records(self):
        try:
            return self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]
        except:
            return 0

    def get_statistics(self):
        stats = {}
        try:
            stats['total_parts'] = self.get_total_records()
            if stats['total_parts'] == 0:
                return {
                    'total_parts': 0, 'total_oe': 0, 'total_brands': 0,
                    'top_brands': None, 'categories': None
                }
            stats['total_oe'] = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()[0]
            stats['total_brands'] = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data WHERE brand IS NOT NULL").fetchone()[0]
            brs = self.conn.execute("SELECT brand, COUNT(*) FROM parts_data WHERE brand IS NOT NULL GROUP BY brand ORDER BY COUNT(*) DESC LIMIT 10").fetchall()
            stats['top_brands'] = pl.DataFrame(brs, schema=["brand", "count"])
            cats = self.conn.execute("SELECT category, COUNT(*) FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY COUNT(*) DESC").fetchall()
            stats['categories'] = pl.DataFrame(cats, schema=["category", "count"])
        except Exception as e:
            st.error(f"Ошибка статистики: {e}")
            stats = {}
        return stats

    # --- Методы загрузки цен и настроек ---
    def load_price_recommendation(self, file_bytes):
        df = pl.read_excel(io.BytesIO(file_bytes))
        if 'артикул' not in df.columns or 'цена' not in df.columns:
            st.error("Файл должен содержать 'артикул' и 'цена'")
            return
        df = df.select([
            pl.col('артикул').alias('artikul'),
            pl.col('цена').cast(pl.Float64)
        ])
        for row in df.iter_rows():
            artikul, price = row
            norm_series = self.normalize_key(pl.Series([artikul]))
            artikul_norm = norm_series[0] if len(norm_series) > 0 else ''
            self.conn.execute("""
                INSERT INTO recommended_prices (artikul_norm, price)
                VALUES (?, ?)
                ON CONFLICT (artikul_norm) DO UPDATE SET price=excluded.price
            """, [artikul_norm, price])
        st.success("Рекомендованные цены загружены.")

    def load_price_list(self, file_bytes):
        df = pl.read_excel(io.BytesIO(file_bytes))
        required_cols = ['артикул', 'бренд', 'кол-во', 'цена']
        if not all(c in df.columns for c in required_cols):
            st.error("Проверьте наличие колонок 'артикул', 'бренд', 'кол-во', 'цена'")
            return
        df = df.select([
            pl.col('артикул'),
            pl.col('бренд'),
            pl.col('кол-во').cast(pl.Int32),
            pl.col('цена').cast(pl.Float64)
        ])
        for row in df.iter_rows():
            artikul, brand, qty, price = row
            norm_artikul = self.normalize_key(pl.Series([artikul]))[0]
            norm_brand = self.normalize_key(pl.Series([brand]))[0]
            self.conn.execute("""
                INSERT INTO price_list (artikul, brand, quantity, price)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (artikul, brand) DO UPDATE SET quantity=excluded.quantity, price=excluded.price
            """, [artikul, brand, qty, price])
        st.success("Прайс-лист загружен.")

    def set_markups(self, total_markup, brand_markups):
        self.conn.execute("""
            UPDATE markup_settings SET total_markup=?, brand_markup=?
            WHERE id=1
        """, [total_markup, json.dumps(brand_markups)])
        st.success("Настройки наценки обновлены.")

    def get_markups(self):
        row = self.conn.execute("SELECT total_markup, brand_markup FROM markup_settings WHERE id=1").fetchone()
        if row:
            total_markup, brand_markup_json = row
            brand_markup = json.loads(brand_markup_json) if brand_markup_json else {}
            return total_markup, brand_markup
        return 0, {}

    def apply_markup(self, price, brand_norm=''):
        total_markup, brand_markup = self.get_markups()
        markup = total_markup
        if brand_norm and brand_norm in brand_markup:
            markup += brand_markup[brand_norm]
        return price * (1 + markup / 100)

    # --- Экспорт ---
    def build_export_query(self, selected_columns=None):
        desc_text = """Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""
        ctes = f"""
        WITH DescriptionTemplate AS (
            SELECT CHR(10) || CHR(10) || $${desc_text}$$ AS text
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
        select_parts = "*, ROW_NUMBER() OVER (PARTITION BY artikul_norm, brand_norm ORDER BY representative_name DESC, oe_list DESC) as rn"
        select_cols = ", ".join([expr for _, expr in [(k, v) for k, v in {
            "Артикул бренда": 'p.artikul AS "Артикул бренда"',
            "Бренд": 'p.brand AS "Бренд"',
            "Наименование": 'COALESCE(p.representative_name, p.analog_representative_name) AS "Наименование"',
            "Применимость": 'COALESCE(p.representative_applicability, p.analog_representative_applicability) AS "Применимость"',
            "Описание": 'CONCAT(COALESCE(p.description, \'\'), dt.text) AS "Описание"',
            "Категория товара": 'COALESCE(p.representative_category, p.analog_representative_category) AS "Категория товара"',
            "Кратность": 'p.multiplicity AS "Кратность"',
            "Длинна": 'COALESCE(p.length, p.analog_length) AS "Длинна"',
            "Ширина": 'COALESCE(p.width, p.analog_width) AS "Ширина"',
            "Высота": 'COALESCE(p.height, p.analog_height) AS "Высота"',
            "Вес": 'COALESCE(p.weight, p.analog_weight) AS "Вес"',
            "Длинна/Ширина/Высота": 'COALESCE(CASE WHEN p.dimensions_str IS NULL OR p.dimensions_str = \'\' OR UPPER(TRIM(p.dimensions_str)) = \'XX\' THEN NULL ELSE p.dimensions_str END, p.analog_dimensions_str) AS "Длинна/Ширина/Высота"',
            "OE номер": 'p.oe_list AS "OE номер"',
            "аналоги": 'p.analog_list AS "аналоги"',
            "Ссылка на изображение": 'p.image_url AS "Ссылка на изображение"',
            "Цена с наценкой": 'p.price_with_markup AS "Цена с наценкой"'
        }.items()]))
        query = f"{ctes} SELECT {select_cols} FROM RankedData p CROSS JOIN DescriptionTemplate dt WHERE p.rn=1 ORDER BY p.brand, p.artikul"

        return query

    def export_csv(self, output_path, selected_columns=None, exclude_names=None):
        total = self.conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)"
        ).fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        query = self.build_export_query(selected_columns)
        df = self.conn.execute(query).pl()

        # Фильтр исключений по названиям
        if exclude_names:
            pattern = '|'.join(exclude_names)
            df = df.filter(~pl.col("Наименование").str.contains(pattern, case=False))

        # Применение наценки к цене
        if 'Цена с наценкой' in df.columns:
            df = df.with_columns(
                pl.col('Цена с наценкой').apply(lambda p: self.apply_markup(p, brand_norm='')).alias('Цена с наценкой')
            )

        # Обработка числовых колонок
        for c in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
            if c in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(c).is_not_null())
                    .then(pl.col(c).cast(pl.Utf8))
                    .otherwise("")
                    .alias(c)
                )

        buf = io.StringIO()
        df.write_csv(buf, separator=';')
        with open(output_path, 'wb') as f:
            f.write(b'\xef\xbb\xbf')
            f.write(buf.getvalue().encode('utf-8'))
        size_mb = os.path.getsize(output_path) / (1024*1024)
        st.success(f"Экспорт завершен: {output_path} ({size_mb:.2f} МБ)")
        return True

    def export_excel(self, output_path, selected_columns=None, exclude_names=None):
        total = self.conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)"
        ).fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False, None
        num_files = (total + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
        progress = st.progress(0)
        files = []
        base_query = self.build_export_query(selected_columns)
        for i in range(num_files):
            offset = i * EXCEL_ROW_LIMIT
            query = f"{base_query} LIMIT {EXCEL_ROW_LIMIT} OFFSET {offset}"
            df = self.conn.execute(query).pl()

            # Фильтр исключений
            if exclude_names:
                pattern = '|'.join(exclude_names)
                df = df.filter(~pl.col("Наименование").str.contains(pattern, case=False))
            # Цены
            if 'Цена с наценкой' in df.columns:
                df = df.with_columns(
                    pl.col('Цена с наценкой').apply(lambda p: self.apply_markup(p, brand_norm='')).alias('Цена с наценкой')
                )

            # Обработка чисел
            for c in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if c in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(c).is_not_null())
                        .then(pl.col(c).cast(pl.Utf8))
                        .otherwise("")
                        .alias(c)
                    )

            fname = output_path.with_name(f"{output_path.stem}_part_{i+1}.xlsx")
            df.write_excel(str(fname))
            files.append(fname)
            progress.progress((i+1)/num_files)
        # Архивация
        if len(files) > 1:
            zip_path = output_path.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in files:
                    zf.write(f, arcname=f.name)
                    os.remove(f)
            final_path = zip_path
        else:
            final_path = files[0]
            if final_path != output_path:
                os.rename(final_path, output_path)
                final_path = output_path
        size_mb = os.path.getsize(final_path) / (1024*1024)
        st.success(f"Экспорт завершен: {final_path.name} ({size_mb:.2f} МБ)")
        return True, final_path

    def export_parquet(self, output_path, selected_columns=None, exclude_names=None):
        total = self.conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)"
        ).fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        query = self.build_export_query(selected_columns)
        df = self.conn.execute(query).pl()

        # Фильтр исключений
        if exclude_names:
            pattern = '|'.join(exclude_names)
            df = df.filter(~pl.col("Наименование").str.contains(pattern, case=False))
        # Цены
        if 'Цена с наценкой' in df.columns:
            df = df.with_columns(
                pl.col('Цена с наценкой').apply(lambda p: self.apply_markup(p, brand_norm='')).alias('Цена с наценкой')
            )

        df.write_parquet(output_path)
        size_mb = os.path.getsize(output_path) / (1024*1024)
        st.success(f"Экспорт завершен: {output_path} ({size_mb:.2f} МБ)")
        return True

    # --- Основной интерфейс ---
    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total = self.conn.execute(
            "SELECT COUNT(DISTINCT artikul_norm, brand_norm) FROM parts_data"
        ).fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта.")
            return
        # Выбор колонок
        options = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена с наценкой"
        ]
        selected_columns = st.multiselect("Выберите колонки для экспорта", options=options, default=options)
        # Фильтр исключений
        exclude_input = st.text_input("Исключить позиции по названию (через |)")
        exclude_names = [n.strip() for n in exclude_input.split('|')] if exclude_input else []

        # Формат
        format_opt = st.radio("Формат экспорта", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)

        # Настройка наценки
        st.subheader("Настройка наценки")
        total_markup = st.slider("Общая наценка (%)", 0, 100, 0)
        brand_markups = {}
        if st.checkbox("Настроить наценки по брендам"):
            brands = self.conn.execute("SELECT DISTINCT brand, brand_norm FROM parts_data WHERE brand IS NOT NULL").fetchall()
            for b, bn in brands:
                mark = st.slider(f"Наценка для бренда '{b}'", 0, 100, 0)
                brand_markups[bn] = mark
        if st.button("Сохранить настройки"):
            self.set_markups(total_markup, brand_markups)

        # Запуск экспорта
        if st.button("🚀 Экспортировать"):
            output_path = self.data_dir / "auto_parts_export"
            if format_opt == "CSV":
                out_file = str(output_path.with_suffix('.csv'))
                self.export_csv(out_file, selected_columns, exclude_names)
                with open(out_file, "rb") as f:
                    st.download_button("📥 Скачать CSV", f, "auto_parts_report.csv", "text/csv")
            elif format_opt == "Excel (.xlsx)":
                out_file = output_path.with_suffix('.xlsx')
                self.export_excel(out_file, selected_columns, exclude_names)
                with open(out_file, "rb") as f:
                    st.download_button("📥 Скачать XLSX", f, out_file.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            elif format_opt == "Parquet":
                out_file = str(output_path.with_suffix('.parquet'))
                self.export_parquet(out_file, selected_columns, exclude_names)
                with open(out_file, "rb") as f:
                    st.download_button("📥 Скачать Parquet", f, "auto_parts_report.parquet", "application/octet-stream")

    # --- Управление удалением ---
    def delete_by_brand(self, brand_norm):
        try:
            count = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE brand_norm=?", [brand_norm]).fetchone()[0]
            self.conn.execute("DELETE FROM parts_data WHERE brand_norm=?", [brand_norm])
            self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT artikul_norm, brand_norm FROM parts_data)")
            return count
        except Exception as e:
            st.error(f"Ошибка удаления по бренду: {e}")
            return 0

    def delete_by_artikul(self, artikul_norm):
        try:
            count = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE artikul_norm=?", [artikul_norm]).fetchone()[0]
            self.conn.execute("DELETE FROM parts_data WHERE artikul_norm=?", [artikul_norm])
            self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT artikul_norm, brand_norm FROM parts_data)")
            return count
        except Exception as e:
            st.error(f"Ошибка удаления по артикула: {e}")
            return 0

# --- Основной интерфейс ---
def main():
    st.title("🚗 AutoParts Catalog - 10+ млн записей")
    st.markdown("Мощная система для управления большими данными автозапчастей.")
    catalog = AutoPartsCatalog()

    menu = st.sidebar.radio("Меню", ["Загрузка данных", "Экспорт", "Статистика", "Управление"])

    if menu == "Загрузка данных":
        # Загрузка файлов
        is_empty = catalog.get_total_records() == 0
        st.subheader("Загрузка данных")
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
        if st.button("🚀 Начать обработку"):
            paths = {}
            for ftype, uploaded in files_map.items():
                if uploaded:
                    filename = f"{ftype}_{int(time.time())}_{uploaded.name}"
                    path = catalog.data_dir / filename
                    with open(path, "wb") as f:
                        f.write(uploaded.getvalue())
                    paths[ftype] = str(path)
            if is_empty:
                missing = [f for f in ['oe', 'cross', 'barcode', 'dimensions', 'images'] if f not in paths]
                if missing:
                    st.error(f"Для начальной загрузки нужны все файлы: {', '.join(missing)}")
                elif len(paths)==5:
                    catalog.merge_all_data_parallel(paths)
                else:
                    st.warning("Загружены не все обязательные файлы.")
            else:
                if paths:
                    catalog.merge_all_data_parallel(paths)
                else:
                    st.info("Загрузите хотя бы один файл.")

    elif menu == "Экспорт":
        catalog.show_export_interface()

    elif menu == "Статистика":
        st.header("📈 Статистика")
        stats = catalog.get_statistics()
        st.metric("Артикулов", stats.get('total_parts', 0))
        st.metric("OE", stats.get('total_oe', 0))
        st.metric("Брендов", stats.get('total_brands', 0))
        st.subheader("Топ брендов")
        st.dataframe(stats.get('top_brands', None))
        st.subheader("Распределение по категориям")
        st.bar_chart(stats.get('categories', None))

    elif menu == "Управление":
        st.header("🗑️ Управление")
        op = st.radio("Действие", ["Удалить по бренду", "Удалить по артикулу"])
        if op == "Удалить по бренду":
            brands = catalog.conn.execute("SELECT DISTINCT brand, brand_norm FROM parts_data WHERE brand IS NOT NULL").fetchall()
            if brands:
                b_list = [b for b, bn in brands]
                selected_b = st.selectbox("Выберите бренд", b_list)
                bn_row = catalog.conn.execute("SELECT brand_norm FROM parts_data WHERE brand=?", [selected_b]).fetchone()
                bn = bn_row[0] if bn_row else ''
                count_del = catalog.delete_by_brand(bn)
                st.success(f"Удалено {count_del} записей для бренда {selected_b}")
            else:
                st.info("Нет брендов для удаления.")
        else:
            arti = st.text_input("Артикул для удаления")
            if arti:
                norm_series = catalog.normalize_key(pl.Series([arti]))
                arti_norm = norm_series[0] if len(norm_series) > 0 else ''
                count_del = catalog.delete_by_artikul(arti_norm)
                st.success(f"Удалено {count_del} записей по артикулу {arti}")

if __name__ == "__main__":
    main()
