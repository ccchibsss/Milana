import polars as pl
import duckdb
import streamlit as st
import os
import time
import io
import zipfile
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

EXCEL_ROW_LIMIT = 1_000_000

class AutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        self.setup_database()
        self.load_recommended_prices()

        self.categories = {}  # категории по ключу
        self.brand_markups: dict = {}
        self.global_markup: float = 0.0

    def setup_database(self):
        # Создаем таблицы
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
                category VARCHAR,
                multiplicity INTEGER,
                barcode VARCHAR,
                image_url VARCHAR,
                dimensions_str VARCHAR,
                description VARCHAR,
                recommended_price DOUBLE,
                price_with_markup DOUBLE,
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
            CREATE TABLE IF NOT EXISTS supplier_prices (
                artikul VARCHAR,
                quantity INTEGER,
                brand VARCHAR,
                supplier_price DOUBLE,
                PRIMARY KEY (artikul, brand)
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
        # Инициализация
        self.conn.execute("INSERT OR IGNORE INTO markup_settings (id, global_markup) VALUES (1, 0.0)")

    def load_recommended_prices(self):
        filepath = self.data_dir / "recommended_prices.xlsx"
        if filepath.exists():
            df = pl.read_excel(str(filepath))
            for row in df.rows():
                artikul, price, brand = row
                self.conn.execute("""
                    INSERT OR REPLACE INTO prices (artikul, recommended_price, brand)
                    VALUES (?, ?, ?)
                """, [artikul, price, brand])
            st.info("Рекомендованные цены загружены.")
        else:
            pass

    def save_recommended_prices(self, df: pl.DataFrame):
        for row in df.rows():
            artikul, price, brand = row
            self.conn.execute("""
                INSERT OR REPLACE INTO prices (artikul, recommended_price, brand)
                VALUES (?, ?, ?)
            """, [artikul, price, brand])
        st.success("Цены обновлены.")

    def get_global_markup(self):
        res = self.conn.execute("SELECT global_markup FROM markup_settings WHERE id=1").fetchone()
        return res[0] if res else 0.0

    def set_global_markup(self, percent: float):
        self.conn.execute("UPDATE markup_settings SET global_markup=?", [percent])
        self.global_markup = percent

    def get_brand_markup(self, brand: str):
        return self.brand_markups.get(brand, 0.0)

    def set_brand_markup(self, brand: str, percent: float):
        self.brand_markups[brand] = percent

    def create_indexes(self):
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_oe ON oe_data(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_parts ON parts_data(artikul_norm, brand_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross ON cross_references(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_prices ON prices(artikul)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_categories ON categories(key)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_markup ON markup_settings(id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_supplier ON supplier_prices(artikul, brand)")

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
        actual_lower = {col.lower(): col for col in actual_cols}
        for key, variants in col_variants.items():
            for variant in variants:
                for actual_l, original_name in actual_lower.items():
                    if variant in actual_l:
                        mapping[original_name] = key
                        break
        return mapping

    def read_and_prepare_file(self, file_path: str, file_type: str):
        try:
            if not os.path.exists(file_path):
                return pl.DataFrame()
            df = pl.read_excel(file_path, engine='calamine')
            if df.is_empty():
                return pl.DataFrame()
            expected = {
                'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
                'barcode': ['barcode', 'artikul', 'brand', 'multiplicity'],
                'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
                'images': ['artikul', 'brand', 'image_url'],
                'cross': ['oe_number', 'artikul', 'brand']
            }
            expected_cols = expected.get(file_type, [])
            col_map = self.detect_columns(df.columns, expected_cols)
            if not col_map:
                return pl.DataFrame()
            df = df.rename(col_map)
            # Очистка
            if 'artikul' in df.columns:
                df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
            if 'brand' in df.columns:
                df = df.with_columns(brand=self.clean_values(pl.col('brand')))
            if 'oe_number' in df.columns:
                df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))
            # Удаление дубликатов
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
        except Exception as e:
            st.error(f"Ошибка при чтении файла {file_path}: {e}")
            return pl.DataFrame()

    def upsert(self, table: str, df: pl.DataFrame, pk: list):
        if df.is_empty():
            return
        df = df.unique(keep='first')
        cols = df.columns
        pk_str = ", ".join([f'"{c}"' for c in pk])
        temp_name = f"temp_{table}_{int(time.time())}"
        self.conn.register(temp_name, df.to_arrow())
        set_clause = ", ".join([f'"{col}"=excluded."{col}"' for col in cols if col not in pk])
        sql = f"""
        INSERT INTO {table} ({', '.join(['"'+c+'"' for c in cols])})
        SELECT * FROM {temp_name}
        ON CONFLICT ({pk_str}) DO UPDATE SET {set_clause}
        """
        self.conn.execute(sql)
        self.conn.unregister(temp_name)

    def process_and_load(self, dfs: dict):
        st.info("🔄 Обработка и загрузка данных...")
        # Обработка OE
        if 'oe' in dfs:
            df_oe = dfs['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df_oe.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'])
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self.determine_category_vectorized(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert('oe_data', oe_df, ['oe_number_norm'])
            cross_df = df_oe.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка cross
        if 'cross' in dfs:
            df_cross = dfs['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            self.upsert('cross_references', df_cross, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка parts
        all_parts = None
        for key in ['oe', 'barcode', 'images', 'dimensions']:
            df = dfs.get(key)
            if df is None or df.is_empty():
                continue
            if 'artikul_norm' in df.columns:
                temp = df.select(['artikul', 'artikul_norm', 'brand', 'brand_norm', 'category'])
                if all_parts is None:
                    all_parts = temp
                else:
                    all_parts = all_parts.join(temp, on=['artikul_norm', 'brand_norm'], how='left', coalesce=True)
        if all_parts is None:
            all_parts = pl.DataFrame()

        # Обработка цен
        prices_df = self.conn.execute("SELECT artikul, recommended_price, brand FROM prices").pl()
        if not all_parts.is_empty():
            all_parts = all_parts.join(prices_df, on='artikul', how='left')
            # Размеры
            for c in ['length', 'width', 'height']:
                if c not in all_parts.columns:
                    all_parts = all_parts.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
            if 'dimensions_str' not in all_parts.columns:
                all_parts = all_parts.with_columns(dimensions_str=pl.lit(''))
            # Описание
            if 'artikul' not in all_parts.columns:
                all_parts = all_parts.with_columns(artikul=pl.lit(''))
            if 'brand' not in all_parts.columns:
                all_parts = all_parts.with_columns(brand=pl.lit(''))
            # Расчет цены с наценкой
            def compute_price(row):
                base_price = row['recommended_price']
                if base_price is None or base_price == 0:
                    return None
                markup_percent = self.get_global_markup()
                brand_markup = self.get_brand_markup(row['brand']) if row['brand'] in self.brand_markups else 0.0
                total_markup = markup_percent + brand_markup
                return base_price * (1 + total_markup / 100)

            all_parts = all_parts.with_columns(
                pl.struct([pl.col('brand'), pl.col('recommended_price')])
                .apply(compute_price)
                .alias('price_with_markup')
            )

            # Финальные колонки
            all_parts = all_parts.select([
                'artikul_norm', 'brand_norm', 'artikul', 'brand', 'category', 'multiplicity', 'barcode',
                'length', 'width', 'height', 'weight', 'image_url', 'dimensions_str', 'description', 'price_with_markup'
            ])

            self.upsert('parts_data', all_parts, ['artikul_norm', 'brand_norm'])

        # Работа с прайсами поставщиков
        self.handle_supplier_prices()

        self.create_indexes()
        st.success("Обработка завершена.")

    def handle_supplier_prices(self):
        # Предположим, что у вас есть файл с прайсами поставщиков (артикул, кол-во, бренд, цена)
        # В реальности его нужно загрузить и обработать, например, из файла или базы.
        # Для примера создадим фиктивные данные или добавим сюда код загрузки файла.
        # Пока что заглушка:
        # Пример загрузки из файла:
        uploaded = st.file_uploader("Загрузить прайсы поставщиков (Excel)", type=['xlsx','xls'])
        if uploaded:
            df_sup = pl.read_excel(uploaded)
            # Предположим, структура: артикул, количество, бренд, цена
            for row in df_sup.rows():
                artikul, quantity, brand, price = row
                self.conn.execute("""
                    INSERT OR REPLACE INTO supplier_prices (artikul, quantity, brand, supplier_price)
                    VALUES (?, ?, ?, ?)
                """, [artikul, quantity, brand, price])
            st.success("Прайсы поставщиков загружены.")
        # Можно дополнительно связать с артикулами в части или таблице цен.

    def get_total_parts(self):
        return self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]

    def get_statistics(self):
        total_parts = self.get_total_parts()
        total_oe = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()[0]
        total_brands = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data").fetchone()[0]
        top_brands = self.conn.execute("SELECT brand, COUNT(*) as cnt FROM parts_data GROUP BY brand ORDER BY cnt DESC LIMIT 10").fetchdf()
        categories = self.conn.execute("SELECT category, COUNT(*) as cnt FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY cnt DESC").fetchdf()
        return {
            'total_parts': total_parts,
            'total_oe': total_oe,
            'total_brands': total_brands,
            'top_brands': top_brands,
            'categories': categories
        }

    def build_export_query(self, selected_cols=None, exclude_terms=None, filters=None):
        # Построение SQL для экспорта
        exclude_where = ""
        if exclude_terms:
            clauses = []
            for term in exclude_terms:
                clauses.append(f"r.\"Наименование\" NOT LIKE '%{term}%'")
            if clauses:
                exclude_where = " AND ".join(clauses)
            if exclude_where:
                exclude_where = "WHERE " + exclude_where

        filter_clauses = []
        if filters:
            if 'brand' in filters:
                brands = filters['brand']
                brand_list = ", ".join([f"'{b}'" for b in brands])
                filter_clauses.append(f"p.brand IN ({brand_list})")
            if 'category' in filters:
                cats = filters['category']
                cat_list = ", ".join([f"'{c}'" for c in cats])
                filter_clauses.append(f"p.category IN ({cat_list})")
            if 'artikul' in filters:
                arts = filters['artikul']
                arts_list = ", ".join([f"'{a}'" for a in arts])
                filter_clauses.append(f"p.artikul IN ({arts_list})")
            if 'oe' in filters:
                o_list = ", ".join([f"'{o}'" for o in filters['oe']])
                # Предположим, что связка с oe_data
                filter_clauses.append(f"pd.oe_list LIKE ANY (ARRAY[{o_list}])")
        filter_where = ""
        if filter_clauses:
            filter_where = " AND ".join(filter_clauses)
            filter_where = "WHERE " + filter_where

        columns_map = [
            ("Артикул бренда", 'p.artikul AS "Артикул бренда"'),
            ("Бренд", 'p.brand AS "Бренд"'),
            ("Категория", 'p.category AS "Категория"'),
            ("Наименование", 'COALESCE(p.description, "") AS "Наименование"'),
            ("Применимость", 'COALESCE(pd.representative_applicability, "") AS "Применимость"'),
            ("Описание", 'CONCAT(COALESCE(p.description, ""), dt.text) AS "Описание"'),
            ("Кратность", 'p.multiplicity AS "Кратность"'),
            ("Длинна", 'p.length AS "Длинна"'),
            ("Ширина", 'p.width AS "Ширина"'),
            ("Высота", 'p.height AS "Высота"'),
            ("Вес", 'p.weight AS "Вес"'),
            ("Длинна/Ширина/Высота", 'p.dimensions_str AS "Длинна/Ширина/Высота"'),
            ("OE номер", 'pd.oe_list AS "OE номер"'),
            ("аналоги", 'aa.analog_list AS "аналоги"'),
            ("Ссылка на изображение", 'p.image_url AS "Ссылка на изображение"'),
            ("Цена с наценкой", 'p.price_with_markup AS "Цена с наценкой"')
        ]

        if not selected_cols:
            selected_exprs = [expr for _, expr in columns_map]
        else:
            selected_exprs = []
            for name, expr in columns_map:
                if name in selected_cols:
                    selected_exprs.append(expr)
            if not selected_exprs:
                selected_exprs = [expr for _, expr in columns_map]

        query = f"""
        WITH DescriptionText AS (
            SELECT ' ' AS text
        ),
        pd AS (
            SELECT
                cr.artikul_norm,
                cr.brand_norm,
                STRING_AGG(DISTINCT regexp_replace(regexp_replace(o.oe_number, '''', ''), '[^0-9A-Za-zА-Яа-яЁё`\-\s]', '', 'g'), ', ') AS oe_list,
                ANY_VALUE(o.name) AS representative_name,
                ANY_VALUE(o.applicability) AS representative_applicability,
                ANY_VALUE(o.category) AS representative_category
            FROM cross_references cr
            JOIN oe_data o ON cr.oe_number_norm = o.oe_number_norm
            GROUP BY cr.artikul_norm, cr.brand_norm
        ),
        aa AS (
            SELECT
                cr1.artikul_norm,
                cr1.brand_norm,
                STRING_AGG(DISTINCT regexp_replace(regexp_replace(p2.artikul, '''', ''), '[^0-9A-Za-zА-Яа-яЁё`\-\s]', '', 'g'), ', ') as analog_list
            FROM cross_references cr1
            JOIN cross_references cr2 ON cr1.oe_number_norm = cr2.oe_number_norm
            JOIN parts_data p2 ON cr2.artikul_norm = p2.artikul_norm AND cr2.brand_norm = p2.brand_norm
            WHERE cr1.artikul_norm != p2.artikul_norm OR cr1.brand_norm != p2.brand_norm
            GROUP BY cr1.artikul_norm, cr1.brand_norm
        )
        SELECT
            {', '.join(selected_exprs)}
        FROM parts_data p
        LEFT JOIN pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
        LEFT JOIN aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
        LEFT JOIN DescriptionText dt ON 1=1
        {filter_where}
        {(' AND ' + exclude_where[6:]) if exclude_where else ''}
        ORDER BY p.brand, p.artikul
        """
        return query

    def export_csv(self, output_path, selected_cols=None, exclude_terms=None, filters=None):
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        query = self.build_export_query(selected_cols, exclude_terms, filters)
        df = self.conn.execute(query).pl()
        # Форматировать размеры и цену
        for c in ["Длинна", "Ширина", "Высота", "Цена с наценкой"]:
            if c in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(c).is_not_null())
                    .then(pl.col(c).cast(pl.Utf8))
                    .otherwise("")
                    .alias(c)
                )
        buf = io.StringIO()
        df.write_csv(buf, separator=';')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(buf.getvalue())
        st.success(f"Экспортировано {total:,} записей.")
        return True

    def export_excel(self, output_path: Path, selected_cols=None, exclude_terms=None, filters=None):
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False, None
        num_files = (total + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
        filenames = []
        for i in range(num_files):
            offset = i * EXCEL_ROW_LIMIT
            query = self.build_export_query(selected_cols, exclude_terms, filters) + f" LIMIT {EXCEL_ROW_LIMIT} OFFSET {offset}"
            df = self.conn.execute(query).pl()
            # Форматировать размеры и цену
            for c in ["Длинна", "Ширина", "Высота", "Цена с наценкой"]:
                if c in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(c).is_not_null())
                        .then(pl.col(c).cast(pl.Utf8))
                        .otherwise("")
                        .alias(c)
                    )
            filename = output_path.with_name(f"{output_path.stem}_part_{i+1}.xlsx")
            df.write_excel(str(filename))
            filenames.append(filename)
        if num_files > 1:
            zip_path = output_path.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for filename in filenames:
                    zipf.write(str(filename), filename.name)
                    os.remove(str(filename))
            return True, zip_path
        else:
            return True, filenames[0]

    def export_parquet(self, output_path, selected_cols=None, exclude_terms=None, filters=None):
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        query = self.build_export_query(selected_cols, exclude_terms, filters)
        df = self.conn.execute(query).pl()
        df.write_parquet(output_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        st.success(f"Экспортировано {total} записей. Размер файла: {size_mb:.2f} МБ")
        return True

    def show_export_ui(self):
        st.header("📤 Экспорт данных")
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта.")
            return

        # Фильтры
        st.subheader("Фильтры")
        brands = self.conn.execute("SELECT DISTINCT brand FROM parts_data").fetchdf()['brand'].dropna().tolist()
        selected_brands = st.multiselect("Бренды", options=brands)
        categories = self.conn.execute("SELECT DISTINCT category FROM oe_data").fetchdf()['category'].dropna().tolist()
        selected_categories = st.multiselect("Категории", options=categories)
        arts = self.conn.execute("SELECT DISTINCT artikul FROM parts_data").fetchdf()['artikul'].dropna().tolist()
        selected_arts = st.multiselect("Артикулы", options=arts)
        o_list = self.conn.execute("SELECT DISTINCT oe_number FROM oe_data").fetchdf()['oe_number'].dropna().tolist()
        selected_oes = st.multiselect("OE номера", options=o_list)

        filters = {}
        if selected_brands:
            filters['brand'] = selected_brands
        if selected_categories:
            filters['category'] = selected_categories
        if selected_arts:
            filters['artikul'] = selected_arts
        if selected_oes:
            filters['oe'] = selected_oes

        # Колонки
        default_cols = [
            "Артикул бренда", "Бренд", "Категория", "Наименование", "Применимость", "Описание",
            "Кратность", "Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота",
            "OE номер", "аналоги", "Ссылка на изображение", "Цена с наценкой"
        ]
        selected_cols = st.multiselect("Выберите колонки для экспорта", options=default_cols, default=default_cols)

        format_choice = st.radio("Формат экспорта", ["CSV", "Excel (.xlsx)", "Parquet"])
        if st.button("Экспортировать"):
            filename = self.data_dir / "auto_parts_export"
            if format_choice == "CSV":
                full_path = filename.with_suffix('.csv')
                self.export_csv(str(full_path), selected_cols, exclude_terms=None, filters=filters)
                with open(str(full_path), "rb") as f:
                    st.download_button("📥 Скачать CSV", f, "auto_parts_export.csv", "text/csv")
            elif format_choice == "Excel (.xlsx)":
                success, out_path = self.export_excel(filename, selected_cols, exclude_terms=None, filters=filters)
                if success and out_path:
                    with open(str(out_path), "rb") as f:
                        st.download_button("📥 Скачать Excel", f, out_path.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                out_path = filename.with_suffix('.parquet')
                self.export_parquet(str(out_path), selected_cols, exclude_terms=None, filters=filters)
                with open(str(out_path), "rb") as f:
                    st.download_button("📥 Скачать Parquet", f, "auto_parts_export.parquet", "application/octet-stream")

    def show_settings_ui(self):
        st.header("⚙️ Настройки")
        st.subheader("Наценки")
        new_markup = st.slider("Общая наценка (%)", 0.0, 100.0, value=self.get_global_markup(), step=0.5)
        if st.button("Применить общую наценку"):
            self.set_global_markup(new_markup)
            st.success(f"Общая наценка установлена {new_markup}%")
        # Для брендов
        st.subheader("Наценки по брендам")
        brands = self.conn.execute("SELECT DISTINCT brand FROM parts_data").fetchdf()['brand'].dropna().tolist()
        for b in brands:
            current_markup = self.get_brand_markup(b)
            new_b_markup = st.slider(f"Наценка для '{b}'", 0.0, 100.0, value=current_markup, step=0.5)
            if st.button(f"Применить для {b}"):
                self.set_brand_markup(b, new_b_markup)
                st.success(f"Наценка для {b} установлена {new_b_markup}%")
        # Загрузка цен
        st.subheader("Загрузка цен из файла")
        uploaded = st.file_uploader("Загрузить Excel файл с ценами", type=['xlsx','xls'])
        if uploaded:
            df_prices = pl.read_excel(uploaded)
            self.save_recommended_prices(df_prices)
        # Категории
        st.subheader("Категории")
        key_input = st.text_input("Ключ (например, 'engine')")
        name_input = st.text_input("Название категории")
        if st.button("Добавить/Обновить категорию"):
            if key_input and name_input:
                self.categories[key_input] = name_input
                st.success(f"Категория '{name_input}' добавлена или обновлена.")
        st.write("Текущие категории:")
        for k, v in self.categories.items():
            st.write(f"{k}: {v}")

    def delete_brand(self, brand_norm):
        count = self.conn.execute("DELETE FROM parts_data WHERE brand_norm=?", [brand_norm]).fetchone()[0]
        self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)")
        return count

    def delete_artikul(self, artikul_norm):
        count = self.conn.execute("DELETE FROM parts_data WHERE artikul_norm=?", [artikul_norm]).fetchone()[0]
        self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)")
        return count

# Основная функция
def main():
    st.title("🚗 Полный проект управления автозапчастями")
    st.markdown("""
    ### 🛠️ Расширенная платформа для работы с большими данными автозапчастей
    - Гибкое управление ценами, наценками и категориями
    - Продвинутый экспорт с фильтрацией
    - Удобное управление артикулами и брендами
    - Импорт данных и резервное копирование
    """)

    catalog = AutoPartsCatalog()

    menu = st.sidebar.radio("Навигация", ["Загрузка данных", "Настройки", "Экспорт", "Статистика", "Управление"])

    if menu == "Загрузка данных":
        # Загрузка файлов
        st.header("📥 Загрузка файлов")
        total_parts = catalog.get_total_parts()
        if total_parts == 0:
            st.warning("База пуста. Загрузите все файлы для начальной инициализации.")
        else:
            st.info("База содержит данные. Можно добавлять или обновлять файлы.")
        col1, col2 = st.columns(2)
        with col1:
            file_oe = st.file_uploader("Основные данные (OE)", type=['xlsx', 'xls'])
            file_cross = st.file_uploader("Кроссы", type=['xlsx', 'xls'])
            file_prices = st.file_uploader("Цены (рекомендованные)", type=['xlsx', 'xls'])
        with col2:
            file_dim = st.file_uploader("Весогабариты", type=['xlsx', 'xls'])
            file_img = st.file_uploader("Изображения", type=['xlsx', 'xls'])
        files_map = {
            'oe': file_oe,
            'cross': file_cross,
            'dimensions': file_dim,
            'images': file_img,
            'prices': file_prices
        }
        if st.button("Обработать файлы"):
            dfs = {}
            for key, file in files_map.items():
                if file:
                    path = catalog.data_dir / f"{key}_{int(time.time())}.xlsx"
                    with open(path, 'wb') as f:
                        f.write(file.read())
                    df = catalog.read_and_prepare_file(str(path), key)
                    dfs[key] = df
            catalog.process_and_load(dfs)

    elif menu == "Настройки":
        catalog.show_settings_ui()

    elif menu == "Экспорт":
        catalog.show_export_ui()

    elif menu == "Статистика":
        stats = catalog.get_statistics()
        st.subheader("📊 Статистика")
        st.metric("Артикулов", stats['total_parts'])
        st.metric("OE", stats['total_oe'])
        st.metric("Брендов", stats['total_brands'])
        st.subheader("Топ брендов")
        st.dataframe(stats['top_brands'])
        st.subheader("Категории")
        st.dataframe(stats['categories'])

    elif menu == "Управление":
        st.header("🗑️ Управление данными")
        action = st.radio("Действие", ["Удалить по бренду", "Удалить по артикулу", "Редактировать цены"])
        if action == "Удалить по бренду":
            brands = catalog.conn.execute("SELECT DISTINCT brand FROM parts_data").fetchdf()['brand'].dropna().tolist()
            if not brands:
                st.info("Нет брендов для удаления.")
            else:
                brand = st.selectbox("Выберите бренд для удаления", brands)
                norm_b = catalog.conn.execute("SELECT brand_norm FROM parts_data WHERE brand=?", [brand]).fetchone()
                if norm_b:
                    norm_b = norm_b[0]
                else:
                    norm_b = catalog.normalize_key(pl.Series([brand]))[0]
                count = catalog.delete_brand(norm_b)
                st.success(f"Удалено {count} записей по бренду {brand}")
        elif action == "Удалить по артикулу":
            artikul_input = st.text_input("Введите артикул для удаления")
            if artikul_input:
                norm_a = catalog.normalize_key(pl.Series([artikul_input]))[0]
                count = catalog.delete_artikul(norm_a)
                st.success(f"Удалено {count} записей по артикулу {artikul_input}")
        elif action == "Редактировать цены":
            st.subheader("Редактировать цену")
            artikul_edit = st.text_input("Артикул")
            brand_edit = st.text_input("Бренд")
            price_edit = st.number_input("Цена", min_value=0.0, step=0.01)
            if st.button("Обновить цену"):
                catalog.conn.execute(
                    "INSERT OR REPLACE INTO prices (artikul, recommended_price, brand) VALUES (?, ?, ?)",
                    [artikul_edit, price_edit, brand_edit]
                )
                st.success(f"Цена для {artikul_edit} ({brand_edit}) обновлена.")

if __name__ == "__main__":
    main()
