import streamlit as st
import duckdb
import polars as pl
import io
import os
import time
import zipfile
import json
from pathlib import Path

# Константы
DATA_DIR = Path("./auto_parts_data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "catalog.duckdb"
EXCEL_ROW_LIMIT = 1_000_000

class AutoPartsCatalog:
    def __init__(self):
        self.conn = duckdb.connect(str(DB_PATH))
        self._setup_database()
        self._create_indexes()
        self._init_settings()

    def _setup_database(self):
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
                price_with_markup DOUBLE,
                category VARCHAR,
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
        if not self.conn.execute("SELECT 1 FROM markup_settings").fetchone():
            self.conn.execute("INSERT INTO markup_settings (id, total_markup, brand_markup) VALUES (1, 0, '{}')")

    def _create_indexes(self):
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_oe ON oe_data(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_parts ON parts_data(artikul_norm, brand_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross ON cross_references(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross_art ON cross_references(artikul_norm, brand_norm)")

    def _init_settings(self):
        # Инициализация настроек наценки
        pass

    @staticmethod
    def normalize_key(series):
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
    def clean_values(series):
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
        if 'artikul' in df.columns:
            df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand=self.clean_values(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))
        key_cols = [c for c in ['oe_number', 'artikul', 'brand'] if c in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')
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
        df_unique = df.unique(keep='first')
        timestamp = int(time.time())
        self.conn.register(f"temp_{table_name}_{timestamp}", df_unique.to_arrow())
        pk_str = ", ".join([f'"{col}"' for col in pk])
        update_cols = [col for col in df_unique.columns if col not in pk]
        if update_cols:
            update_clause = ", ".join([f'"{col}"=excluded."{col}"' for col in update_cols])
        else:
            update_clause = "DO NOTHING"
        sql = f"""
            INSERT INTO {table_name}
            SELECT * FROM "temp_{table_name}_{timestamp}"
            ON CONFLICT ({pk_str}) {update_clause}
        """
        try:
            self.conn.execute(sql)
        except Exception as e:
            st.error(f"Ошибка при вставке/обновлении {table_name}: {e}")
        finally:
            self.conn.unregister(f"temp_{table_name}_{timestamp}")

    def process_and_load(self, dataframes):
        st.info("🔄 Обновление базы данных...")
        total_steps = 2
        progress = st.progress(0)
        step = 0

        # Обработка OE
        if 'oe' in dataframes:
            step +=1
            progress.progress(step/total_steps, text=f"Обработка OE ({step}/{total_steps})")
            df_oe = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df_oe.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'])
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self._category_by_name(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])
            cross_df = df_oe.filter(pl.col('artikul_norm') != "").select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка cross
        if 'cross' in dataframes:
            step +=1
            progress.progress(step/total_steps, text=f"Обработка кроссов ({step}/{total_steps})")
            df_cross = dataframes['cross'].filter(
                (pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != "")
            )
            self.upsert_data('cross_references', df_cross, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка parts
        step +=1
        progress.progress(step/total_steps, text=f"Обработка артикула ({step}/{total_steps})")
        # Можно расширить обработку
        progress.progress(1)
        time.sleep(0.5)
        st.success("🗃️ Обновление завершено!")

    def _category_by_name(self, name_col):
        categories_map = {
            'аккумулятор': 'Автоэлектрика',
            'фильтр': 'Фильтры',
            'масло': 'Масла',
            'тормоз': 'Тормозные системы',
            'свеча': 'Автоэлектрика'
        }
        def get_category(name):
            n = name.lower()
            for k, v in categories_map.items():
                if k in n:
                    return v
            return 'Разное'
        return name_col.apply(get_category)

    def merge_all_data(self, paths: dict):
        start_time = time.time()
        import concurrent.futures
        futures = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for key, path in paths.items():
                futures[executor.submit(self.read_and_prepare_file, path, key)] = key
            dataframes = {}
            for future in concurrent.futures.as_completed(futures):
                t = futures[future]
                df = future.result()
                if not df.is_empty():
                    dataframes[t] = df
        if dataframes:
            self.process_and_load(dataframes)
        st.success(f"Обработка завершена за {time.time() - start_time:.2f} сек.")

    def get_statistics(self):
        total_parts = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]
        total_oe = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()[0]
        total_brands = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data WHERE brand IS NOT NULL").fetchone()[0]
        top_brands = self.conn.execute("SELECT brand, COUNT(*) as cnt FROM parts_data WHERE brand IS NOT NULL GROUP BY brand ORDER BY cnt DESC LIMIT 10").fetchdf()
        categories = self.conn.execute("SELECT category, COUNT(*) as cnt FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY cnt DESC").fetchdf()
        return {
            'total_parts': total_parts,
            'total_oe': total_oe,
            'total_brands': total_brands,
            'top_brands': top_brands,
            'categories': categories
        }

    def load_recommended_prices(self, file_bytes):
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
            artikul_norm = norm_series[0]
            self.conn.execute("""
                INSERT INTO recommended_prices (artikul_norm, price)
                VALUES (?, ?)
                ON CONFLICT (artikul_norm) DO UPDATE SET price=excluded.price
            """, [artikul_norm, price])
        st.success("Рекомендованные цены обновлены!")

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
        st.success("Прайс-лист загружен!")

    def set_markups(self, total_markup, brand_markup):
        self.conn.execute("""
            UPDATE markup_settings SET total_markup=?, brand_markup=?
            WHERE id=1
        """, [total_markup, json.dumps(brand_markup)])
        st.success("Настройки наценки сохранены!")

    def get_markups(self):
        row = self.conn.execute("SELECT total_markup, brand_markup FROM markup_settings WHERE id=1").fetchone()
        if row:
            total_markup, brand_markup_json = row
            brand_markup = json.loads(brand_markup_json) if brand_markup_json else {}
            return total_markup, brand_markup
        return 0, {}

    def get_marked_brands(self):
        _, brand_markup = self.get_markups()
        return json.loads(brand_markup) if brand_markup else {}

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
        RankedData AS (
            -- Можно реализовать ранжирование по необходимости
        )
        """
        # Колонки по умолчанию
        if not selected_columns:
            select_exprs = [
                'p.artikul AS "Артикул бренда"',
                'p.brand AS "Бренд"',
                'COALESCE(p.representative_name, p.analog_representative_name) AS "Наименование"',
                'COALESCE(p.representative_applicability, p.analog_representative_applicability) AS "Применимость"',
                'CONCAT(COALESCE(p.description, \'\'), dt.text) AS "Описание"',
                'COALESCE(p.representative_category, p.analog_representative_category) AS "Категория товара"',
                'p.multiplicity AS "Кратность"',
                'COALESCE(p.length, p.analog_length) AS "Длинна"',
                'COALESCE(p.width, p.analog_width) AS "Ширина"',
                'COALESCE(p.height, p.analog_height) AS "Высота"',
                'COALESCE(p.weight, p.analog_weight) AS "Вес"',
                'COALESCE(CASE WHEN p.dimensions_str IS NULL OR p.dimensions_str = \'\' OR UPPER(TRIM(p.dimensions_str)) = \'XX\' THEN NULL ELSE p.dimensions_str END, p.analog_dimensions_str) AS "Длинна/Ширина/Высота"',
                'p.oe_list AS "OE номер"',
                'p.analog_list AS "аналоги"',
                'p.image_url AS "Ссылка на изображение"',
                'p.price_with_markup AS "Цена с наценкой"'
            ]
        else:
            select_exprs = []
            for col_name in selected_columns:
                if col_name == "Артикул бренда":
                    select_exprs.append('p.artikul AS "Артикул бренда"')
                elif col_name == "Бренд":
                    select_exprs.append('p.brand AS "Бренд"')
                elif col_name == "Наименование":
                    select_exprs.append('COALESCE(p.representative_name, p.analog_representative_name) AS "Наименование"')
                elif col_name == "Применимость":
                    select_exprs.append('COALESCE(p.representative_applicability, p.analog_representative_applicability) AS "Применимость"')
                elif col_name == "Описание":
                    select_exprs.append('CONCAT(COALESCE(p.description, \'\'), dt.text) AS "Описание"')
                elif col_name == "Категория товара":
                    select_exprs.append('COALESCE(p.representative_category, p.analog_representative_category) AS "Категория товара"')
                elif col_name == "Кратность":
                    select_exprs.append('p.multiplicity AS "Кратность"')
                elif col_name == "Длинна":
                    select_exprs.append('COALESCE(p.length, p.analog_length) AS "Длинна"')
                elif col_name == "Ширина":
                    select_exprs.append('COALESCE(p.width, p.analog_width) AS "Ширина"')
                elif col_name == "Высота":
                    select_exprs.append('COALESCE(p.height, p.analog_height) AS "Высота"')
                elif col_name == "Вес":
                    select_exprs.append('COALESCE(p.weight, p.analog_weight) AS "Вес"')
                elif col_name == "Длинна/Ширина/Высота":
                    select_exprs.append('COALESCE(CASE WHEN p.dimensions_str IS NULL OR p.dimensions_str = \'\' OR UPPER(TRIM(p.dimensions_str)) = \'XX\' THEN NULL ELSE p.dimensions_str END, p.analog_dimensions_str) AS "Длинна/Ширина/Высота"')
                elif col_name == "OE номер":
                    select_exprs.append('p.oe_list AS "OE номер"')
                elif col_name == "аналоги":
                    select_exprs.append('p.analog_list AS "аналоги"')
                elif col_name == "Ссылка на изображение":
                    select_exprs.append('p.image_url AS "Ссылка на изображение"')
                elif col_name == "Цена с наценкой":
                    select_exprs.append('p.price_with_markup AS "Цена с наценкой"')
        select_clause = ", ".join(select_exprs)
        query = f"""
        {ctes}
        SELECT {select_clause}
        FROM RankedData p
        CROSS JOIN DescriptionTemplate dt
        WHERE p.rn=1
        ORDER BY p.brand, p.artikul
        """
        return query

    def export_csv(self, output_path, selected_columns=None, exclude_names=None):
        total = self.conn.execute("""
            SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)
        """).fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        query = self.build_export_query(selected_columns)
        df = self.conn.execute(query).pl()

        if exclude_names:
            pattern = '|'.join(exclude_names)
            df = df.filter(~pl.col("Наименование").str.contains(pattern, case=False))
        if 'Цена с наценкой' in df.columns:
            df = df.with_columns(
                pl.col('Цена с наценкой').apply(lambda p: self.apply_markup(p)).alias('Цена с наценкой')
            )
        buf = io.StringIO()
        df.write_csv(buf, separator=';')
        with open(output_path, 'wb') as f:
            f.write(b'\xef\xbb\xbf')
            f.write(buf.getvalue().encode('utf-8'))
        st.success(f"Экспорт завершен: {output_path}")
        return True

    def export_excel(self, output_path, selected_columns=None, exclude_names=None):
        total = self.conn.execute("""
            SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)
        """).fetchone()[0]
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

            if exclude_names:
                pattern = '|'.join(exclude_names)
                df = df.filter(~pl.col("Наименование").str.contains(pattern, case=False))
            if 'Цена с наценкой' in df.columns:
                df = df.with_columns(
                    pl.col('Цена с наценкой').apply(lambda p: self.apply_markup(p)).alias('Цена с наценкой')
                )
            fname = output_path.with_name(f"{output_path.stem}_part_{i+1}.xlsx")
            df.write_excel(str(fname))
            files.append(fname)
            progress.progress((i+1)/num_files)
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
        st.success(f"Экспорт завершен: {final_path}")
        return True, final_path

    def export_parquet(self, output_path, selected_columns=None, exclude_names=None):
        total = self.conn.execute("""
            SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)
        """).fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        query = self.build_export_query(selected_columns)
        df = self.conn.execute(query).pl()

        if exclude_names:
            pattern = '|'.join(exclude_names)
            df = df.filter(~pl.col("Наименование").str.contains(pattern, case=False))
        if 'Цена с наценкой' in df.columns:
            df = df.with_columns(
                pl.col('Цена с наценкой').apply(lambda p: self.apply_markup(p)).alias('Цена с наценкой')
            )
        df.write_parquet(output_path)
        st.success(f"Экспорт завершен: {output_path}")
        return True

    def apply_markup(self, price):
        total, brand = self.get_markups()
        markup = total
        return price * (1 + markup / 100)

    def get_markups(self):
        row = self.conn.execute("SELECT total_markup, brand_markup FROM markup_settings WHERE id=1").fetchone()
        if row:
            total_markup, brand_markup_json = row
            brand_markup = json.loads(brand_markup_json) if brand_markup_json else {}
            return total_markup, brand_markup
        return 0, {}

    def show_export_interface(self):
        st.header("📤 Экспорт данных")
        total = self.conn.execute("""
            SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)
        """).fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return
        options = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена с наценкой"
        ]
        selected_columns = st.multiselect("Выберите колонки для экспорта и порядок", options=options, default=options)
        exclude_input = st.text_input("Исключить по названию через | (точное или частичное совпадение)")
        exclude_names = [n.strip() for n in exclude_input.split('|')] if exclude_input else []

        format_opt = st.radio("Формат экспорта", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)

        st.subheader("Настройки наценки")
        total_markup = st.slider("Общая наценка (%)", 0, 100, 0)
        brand_markups = {}
        if st.checkbox("Настроить наценки по брендам"):
            brands = self.conn.execute("SELECT DISTINCT brand, brand_norm FROM parts_data WHERE brand IS NOT NULL").fetchall()
            for b, bn in brands:
                mark = st.slider(f"Наценка для бренда '{b}'", 0, 100, 0)
                brand_markups[bn] = mark
        if st.button("Сохранить настройки"):
            self.set_markups(total_markup, brand_markups)

        if st.button("🚀 Начать экспорт"):
            output_path = self.data_dir / "auto_parts_export"
            if format_opt == "CSV":
                out_file = output_path.with_suffix('.csv')
                self.export_csv(str(out_file), selected_columns, exclude_names)
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

# Основной интерфейс
def main():
    st.set_page_config(page_title="AutoParts Catalog", layout="wide")
    st.title("🚗 AutoParts Catalog — Управление и экспорт")
    catalog = AutoPartsCatalog()

    menu = st.sidebar.radio("Меню", ["Загрузка данных", "Экспорт", "Статистика", "Рекомендации цен", "Прайс-лист"])

    if menu == "Загрузка данных":
        st.subheader("Загрузка файлов")
        col1, col2 = st.columns(2)
        with col1:
            file_oe = st.file_uploader("Основные данные (OE)", type=['xlsx', 'xls'])
            file_cross = st.file_uploader("Кроссы (OE → Артикул)", type=['xlsx', 'xls'])
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
        if st.button("🚀 Обработать файлы"):
            paths = {}
            for key, uploaded in files_map.items():
                if uploaded:
                    filename = f"{key}_{int(time.time())}_{uploaded.name}"
                    path = DATA_DIR / filename
                    with open(path, "wb") as f:
                        f.write(uploaded.read())
                    paths[key] = str(path)
            if paths:
                catalog.merge_all_data(paths)
            else:
                st.info("Загрузите файлы для обработки.")
    elif menu == "Экспорт":
        catalog.show_export_interface()
    elif menu == "Статистика":
        stats = catalog.get_statistics()
        st.metric("Артикулов", stats['total_parts'])
        st.metric("OE", stats['total_oe'])
        st.metric("Брендов", stats['total_brands'])
        st.subheader("Топ брендов")
        st.dataframe(stats['top_brands'])
        st.subheader("Распределение по категориям")
        st.dataframe(stats['categories'])
        st.bar_chart(stats['categories'].set_index('category')['cnt'])
    elif menu == "Рекомендации цен":
        st.subheader("Загрузка рекомендаций по ценам")
        uploaded = st.file_uploader("Загрузите файл с ценами", type=['xlsx', 'xls'])
        if uploaded:
            catalog.load_recommended_prices(uploaded.read())
    elif menu == "Прайс-лист":
        st.subheader("Загрузка прайс-листа")
        uploaded = st.file_uploader("Загрузите прайс-лист", type=['xlsx', 'xls'])
        if uploaded:
            catalog.load_price_list(uploaded.read())


if __name__ == "__main__":
    main()
