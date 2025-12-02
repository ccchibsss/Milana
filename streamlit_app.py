import polars as pl
import duckdb
import streamlit as st
import os
import time
import io
import zipfile
from pathlib import Path
from typing import Dict, List
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
        st.set_page_config(
            page_title="AutoParts Catalog 10M+",
            layout="wide",
            page_icon="🚗"
        )

        # Наценки
        self.global_markup_percent = 0.0
        self.brand_markups: Dict[str, float] = {}
        self.load_recommended_prices()

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
                length DOUBLE, 
                width DOUBLE,
                height DOUBLE, 
                weight DOUBLE,
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
            CREATE TABLE IF NOT EXISTS prices (
                artikul VARCHAR PRIMARY KEY,
                recommended_price DOUBLE,
                brand VARCHAR
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
        self.load_recommended_prices()

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
            st.success("Загружены рекомендованные цены из файла.")
        else:
            pass

    def save_recommended_prices(self, df: pl.DataFrame):
        for row in df.rows():
            artikul, price, brand = row
            self.conn.execute("""
                INSERT OR REPLACE INTO prices (artikul, recommended_price, brand)
                VALUES (?, ?, ?)
            """, [artikul, price, brand])
        st.success("Рекомендованные цены сохранены.")

    def set_global_markup(self, percent: float):
        self.global_markup_percent = percent
        self.conn.execute("UPDATE markup_settings SET global_markup = ?", [percent])

    def get_global_markup(self):
        res = self.conn.execute("SELECT global_markup FROM markup_settings WHERE id=1").fetchone()
        return res[0] if res else 0.0

    def set_brand_markup(self, brand: str, percent: float):
        self.brand_markups[brand] = percent

    def get_brand_markup(self, brand: str):
        return self.brand_markups.get(brand, 0.0)

    def create_indexes(self):
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_oe ON oe_data(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_parts ON parts_data(artikul_norm, brand_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross ON cross_references(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_prices ON prices(artikul)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_markup ON markup_settings(id)")

    @staticmethod
    def normalize_key(series: pl.Series) -> pl.Series:
        return (
            series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all("'", "")
            .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
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
            .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    def detect_columns(self, actual_cols: List[str], expected_cols: List[str]) -> Dict[str, str]:
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
        except:
            return pl.DataFrame()

    def upsert(self, table: str, df: pl.DataFrame, pk: List[str]):
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
        ON CONFLICT ({pk_str}) DO UPDATE SET {set_clause};
        """
        self.conn.execute(sql)
        self.conn.unregister(temp_name)

    def process_and_load(self, dfs: Dict[str, pl.DataFrame]):
        st.info("🔄 Обработка и загрузка данных...")
        # OE
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

        # cross
        if 'cross' in dfs:
            df_cross = dfs['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            self.upsert('cross_references', df_cross, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # parts
        all_parts = None
        for f in ['oe', 'barcode', 'images', 'dimensions']:
            df = dfs.get(f)
            if df is None or df.is_empty():
                continue
            if 'artikul_norm' in df.columns:
                temp = df.select(['artikul', 'artikul_norm', 'brand', 'brand_norm'])
                if all_parts is None:
                    all_parts = temp
                else:
                    all_parts = all_parts.join(temp, on=['artikul_norm', 'brand_norm'], how='left', coalesce=True)
        if all_parts is None:
            all_parts = pl.DataFrame()

        # загрузка цен
        prices_df = self.conn.execute("SELECT artikul, recommended_price, brand FROM prices").pl()
        if not all_parts.is_empty():
            all_parts = all_parts.join(prices_df, on='artikul', how='left')
            # размеры
            for c in ['length','width','height']:
                if c not in all_parts.columns:
                    all_parts = all_parts.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
            if 'dimensions_str' not in all_parts.columns:
                all_parts = all_parts.with_columns(dimensions_str=pl.lit(''))
            # строки размеров
            all_parts = all_parts.with_columns([
                pl.col('length').cast(pl.Utf8).fill_null('').alias('_length_str'),
                pl.col('width').cast(pl.Utf8).fill_null('').alias('_width_str'),
                pl.col('height').cast(pl.Utf8).fill_null('').alias('_height_str'),
            ])
            all_parts = all_parts.with_columns(
                pl.when(pl.col('dimensions_str') != '').then(pl.col('dimensions_str'))
                .otherwise(pl.concat_str([pl.col('_length_str'), pl.lit('x'), pl.col('_width_str'), pl.lit('x'), pl.col('_height_str')], separator=''))
                .alias('dimensions_str')
            ).drop(['_length_str', '_width_str', '_height_str'])
            # описание
            if 'artikul' not in all_parts.columns:
                all_parts = all_parts.with_columns(artikul=pl.lit(''))
            if 'brand' not in all_parts.columns:
                all_parts = all_parts.with_columns(brand=pl.lit(''))
            all_parts = all_parts.with_columns([
                pl.col('artikul').cast(pl.Utf8).fill_null('').alias('_artikul'),
                pl.col('brand').cast(pl.Utf8).fill_null('').alias('_brand'),
                pl.col('recommended_price').fill_null(0).cast(pl.Float64).alias('_rec_price'),
                pl.col('multiplicity').fill_null(1).cast(pl.Int32).alias('multiplicity'),
            ])
            all_parts = all_parts.with_columns(
                pl.concat_str([
                    'Артикул: ', pl.col('_artikul'),
                    ', Бренд: ', pl.col('_brand'),
                    ', Кратность: ', pl.col('multiplicity').cast(pl.Utf8),
                    ' шт.'
                ], separator='').alias('description')
            )
            # цена с наценкой
            def compute_price(row):
                base_price = row['recommended_price']
                if base_price is None or base_price==0:
                    return None
                markup_percent = self.get_global_markup()
                brand_markup = self.get_brand_markup(row['brand']) if row['brand'] in self.get_brand_markups() else 0.0
                total_markup = markup_percent + brand_markup
                return base_price * (1 + total_markup/100)

            all_parts = all_parts.with_columns(
                pl.struct([pl.col('brand'), pl.col('recommended_price')])
                .apply(compute_price)
                .alias('price_with_markup')
            )

            # финальные колонки
            all_parts = all_parts.select([
                'artikul_norm', 'brand_norm', 'artikul', 'brand', 'multiplicity', 'barcode',
                'length','width','height','weight','image_url','dimensions_str','description','price_with_markup'
            ])

            self.upsert('parts_data', all_parts, ['artikul_norm', 'brand_norm'])

        self.create_indexes()
        st.success("Обработка и загрузка завершена.")

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

    def build_export_query(self, selected_cols: List[str]=None, exclude_terms: List[str]=None):
        # Основной запрос
        exclude_where = ""
        if exclude_terms:
            clauses = []
            for term in exclude_terms:
                clauses.append(f"r.\"Наименование\" NOT LIKE '%{term}%'")
            if clauses:
                exclude_where = "WHERE " + " AND ".join(clauses)

        columns_map = [
            ("Артикул бренда", 'p.artikul AS "Артикул бренда"'),
            ("Бренд", 'p.brand AS "Бренд"'),
            ("Наименование", 'COALESCE(p.description, "") AS "Наименование"'),
            ("Применимость", 'COALESCE(pd.representative_applicability, "") AS "Применимость"'),
            ("Описание", 'CONCAT(COALESCE(p.description, ""), dt.text) AS "Описание"'),
            ("Категория товара", 'COALESCE(pd.representative_category, "") AS "Категория товара"'),
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

        # Общий текст
        query = f"""
        WITH DescriptionText AS (
            SELECT CHR(10) || CHR(10) || $${"""Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""}$$ AS text
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
        FROM ranked r
        LEFT JOIN DescriptionText dt ON 1=1
        WHERE r.rn=1
        {('AND ' + exclude_where) if exclude_where else ''}
        ORDER BY r.brand, r.artikul
        """
        return query

    def export_csv(self, output_path: str, selected_cols: List[str]=None, exclude_terms: List[str]=None):
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        query = self.build_export_query(selected_cols, exclude_terms)
        df = self.conn.execute(query).pl()
        # строки размеров
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
        with open(output_path, 'wb') as f:
            f.write(b'\xef\xbb\xbf')
            f.write(buf.getvalue().encode('utf-8'))
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        st.success(f"Экспортировано {total:,} записей. Размер файла: {size_mb:.2f} МБ")
        return True

    def export_excel(self, output_path: Path, selected_cols: List[str]=None, exclude_terms: List[str]=None):
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False, None
        num_files = (total + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
        for i in range(num_files):
            offset = i * EXCEL_ROW_LIMIT
            query = self.build_export_query(selected_cols, exclude_terms) + f" LIMIT {EXCEL_ROW_LIMIT} OFFSET {offset}"
            df = self.conn.execute(query).pl()
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
        if num_files > 1:
            zip_path = output_path.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for i in range(num_files):
                    filename = output_path.with_name(f"{output_path.stem}_part_{i+1}.xlsx")
                    zipf.write(str(filename), filename.name)
                    os.remove(str(filename))
            return True, zip_path
        else:
            return True, output_path

    def export_parquet(self, output_path: str, selected_cols: List[str]=None, exclude_terms: List[str]=None):
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        query = self.build_export_query(selected_cols, exclude_terms)
        df = self.conn.execute(query).pl()
        df.write_parquet(output_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        st.success(f"Экспортировано {total} записей. Размер файла: {size_mb:.2f} МБ")
        return True

    def show_export_ui(self):
        st.header("📤 Умный экспорт данных")
        total = self.conn.execute("SELECT COUNT(DISTINCT artikul_norm, brand_norm) FROM parts_data").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта.")
            return

        # Колонки с возможностью drag-and-drop
        default_cols = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена с наценкой"
        ]
        selected_cols = st.multiselect("Выберите колонки для экспорта (пусто — все по умолчанию):", options=default_cols, default=default_cols)

        # фильтр исключений
        exclude_input = st.text_input("Исключить позиции по названиям (через |):")
        exclude_terms = [t.strip() for t in exclude_input.split('|')] if exclude_input else []

        format_choice = st.radio("Экспортировать в:", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)
        if st.button("Экспортировать"):
            if format_choice == "CSV":
                path = self.data_dir / "auto_parts_export.csv"
                self.export_csv(str(path), selected_cols, exclude_terms)
                with open(str(path), "rb") as f:
                    st.download_button("📥 Скачать CSV", f, "auto_parts_export.csv", "text/csv")
            elif format_choice == "Excel (.xlsx)":
                path = self.data_dir / "auto_parts_export.xlsx"
                success, final_path = self.export_excel(path, selected_cols, exclude_terms)
                if success and final_path:
                    with open(str(final_path), "rb") as f:
                        mime = "application/zip" if final_path.suffix == ".zip" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        st.download_button("📥 Скачать файл", f, final_path.name, mime)
            else:
                path = self.data_dir / "auto_parts_export.parquet"
                self.export_parquet(str(path), selected_cols, exclude_terms)
                with open(str(path), "rb") as f:
                    st.download_button("📥 Скачать Parquet", f, "auto_parts_export.parquet", "application/octet-stream")

    def show_settings_ui(self):
        st.header("⚙️ Настройки")
        st.subheader("Наценки")
        # Глобальная
        new_markup = st.slider("Общая наценка %:", min_value=0.0, max_value=100.0, value=self.get_global_markup(), step=0.5)
        if st.button("Применить наценку"):
            self.set_global_markup(new_markup)
            st.success(f"Общая наценка установлена {new_markup}%")
        # По брендам
        st.subheader("Наценки по брендам")
        # Загрузка текущих
        brands = self.conn.execute("SELECT DISTINCT brand FROM parts_data").fetchdf()
        for b in brands['brand']:
            current = self.get_brand_markup(b)
            new_b_markup = st.slider(f"Наценка для бренда '{b}':", min_value=0.0, max_value=100.0, value=current, step=0.5)
            if st.button(f"Применить для {b}"):
                self.set_brand_markup(b, new_b_markup)
                st.success(f"Наценка для {b} установлена {new_b_markup}%")
        # Загрузка цен
        st.subheader("Загрузка цен из файла")
        uploaded_files = st.file_uploader("Выберите Excel файл с ценами", type=['xlsx','xls'])
        if uploaded_files:
            df_prices = pl.read_excel(uploaded_files)
            self.save_recommended_prices(df_prices)
            st.success("Цены обновлены.")

    def delete_brand(self, brand_norm: str):
        count = self.conn.execute("DELETE FROM parts_data WHERE brand_norm=?", [brand_norm]).fetchone()[0]
        self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)")
        return count

    def delete_artikul(self, artikul_norm: str):
        count = self.conn.execute("DELETE FROM parts_data WHERE artikul_norm=?", [artikul_norm]).fetchone()[0]
        self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)")
        return count

# Основная функция
def main():
    st.title("🚗 AutoParts Catalog - Управление большими данными")
    st.markdown("""
    ### 💪 Мощная платформа для работы с автозапчастями
    - Инкрементальные обновления
    - Быстрый поиск
    - Экспорт без дубликатов
    - Настройка цен и наценок
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
            file_oe = st.file_uploader("Основные данные (OE)", type=['xlsx','xls'])
            file_cross = st.file_uploader("Кроссы", type=['xlsx','xls'])
            file_prices = st.file_uploader("Цены (рекомендованные)", type=['xlsx','xls'])
        with col2:
            file_dim = st.file_uploader("Весогабариты", type=['xlsx','xls'])
            file_img = st.file_uploader("Изображения", type=['xlsx','xls'])
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
            # Обновление цен
            if files_map['prices']:
                df_prices = pl.read_excel(str(path))
                catalog.save_recommended_prices(df_prices)

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
        st.header("🗑️ Управление")
        action = st.radio("Действие", ["Удалить по бренду", "Удалить по артикулу"])
        if action == "Удалить по бренду":
            brands = catalog.conn.execute("SELECT DISTINCT brand FROM parts_data").fetchdf()
            if not brands.empty:
                brand = st.selectbox("Выберите бренд для удаления", brands['brand'].to_list())
                norm = catalog.conn.execute("SELECT brand_norm FROM parts_data WHERE brand=?", [brand]).fetchone()
                if norm:
                    norm = norm[0]
                else:
                    norm = catalog.normalize_key(pl.Series([brand]))[0]
                count = catalog.delete_brand(norm)
                st.success(f"Удалено {count} записей по бренду {brand}")
            else:
                st.info("Нет брендов для удаления.")
        else:
            artikul = st.text_input("Введите артикул для удаления")
            if artikul:
                norm = catalog.normalize_key(pl.Series([artikul]))[0]
                count = catalog.delete_artikul(norm)
                st.success(f"Удалено {count} записей по артикулу {artikul}")

if __name__ == "__main__":
    main()
