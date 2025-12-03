import polars as pl
import duckdb
import streamlit as st
import os
import time
import io
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="AutoParts Catalog 10M+", layout="wide", page_icon="🚗")


class AutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        self.setup_database()
        self.global_markup = 0.0  # Глобальная наценка
        self.brand_markup: dict = {}  # Наценка для брендов
        self.excluded_brands: list = []  # Исключенные бренды
        self.excluded_artikuls: list = []  # Исключенные артикула
        self.create_indexes()

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
            CREATE TABLE IF NOT EXISTS price_recommendations (
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                artikul VARCHAR,
                brand VARCHAR,
                quantity INTEGER,
                price DOUBLE,
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)
        self.create_indexes()

    def create_indexes(self):
        for sql in [
            "CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)"
        ]:
            self.conn.execute(sql)

    def normalize_key(self, series):
        return (series
                .fill_null("")
                .cast(pl.Utf8)
                .str.replace_all("'", "")
                .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .str.to_lowercase())

    def clean_values(self, series):
        return (series
                .fill_null("")
                .cast(pl.Utf8)
                .str.replace_all("'", "")
                .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars())

    def detect_columns(self, actual_cols, expected_cols):
        mapping = {}
        variants = {
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
        for key, keys in variants.items():
            for v in keys:
                for ac, ac_orig in actual_lower.items():
                    if v in ac:
                        mapping[ac_orig] = key
                        break
        return mapping

    def read_and_prepare_file(self, path, ftype):
        try:
            if not os.path.exists(path) or os.path.getsize(path)==0:
                return pl.DataFrame()
            df = pl.read_excel(str(path), engine='calamine')
            if df.is_empty():
                return pl.DataFrame()
        except:
            return pl.DataFrame()
        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand']
        }
        exp_cols = schemas.get(ftype, [])
        mapping = self.detect_columns(df.columns, exp_cols)
        if not mapping:
            return pl.DataFrame()
        df = df.rename(mapping)
        if 'artikul' in df.columns:
            df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand=self.clean_values(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))
        key_cols = [c for c in ['oe_number','artikul','brand'] if c in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')
        if 'artikul' in df.columns:
            df = df.with_columns(artikul_norm=self.normalize_key(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand_norm=self.normalize_key(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number_norm=self.normalize_key(pl.col('oe_number')))
        return df

    def upsert_data(self, tablename, df, pk):
        if df.is_empty():
            return
        df = df.unique(keep='first')
        cols = df.columns
        pk_str = ", ".join(f'"{c}"' for c in pk)
        tname = f"temp_{tablename}_{int(time.time())}"
        self.conn.register(tname, df.to_arrow())
        update_cols = [c for c in cols if c not in pk]
        if not update_cols:
            sql = f"INSERT INTO {tablename} SELECT * FROM {tname} ON CONFLICT ({pk_str}) DO NOTHING;"
        else:
            set_clause = ", ".join([f'"{c}"=excluded."{c}"' for c in update_cols])
            sql = f"INSERT INTO {tablename} SELECT * FROM {tname} ON CONFLICT ({pk_str}) DO UPDATE SET {set_clause};"
        try:
            self.conn.execute(sql)
        finally:
            self.conn.unregister(tname)

    def process_and_load_data(self, dfs):
        st.info("🔄 Обновление базы данных...")
        steps = ['oe', 'cross', 'parts']
        n_step = len(steps)
        pbar = st.progress(0)
        idx=0
        # Обработка OE
        if 'oe' in dfs:
            idx+=1
            pbar.progress(idx/n_step, "Обработка OE")
            df_oe = dfs['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df_oe.select(['oe_number_norm','oe_number','name','applicability']).unique(subset=['oe_number_norm'])
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self.determine_category_vectorized(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])
            cross_df = df_oe.filter(pl.col('artikul_norm') != "").select(['oe_number_norm','artikul_norm','brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm','artikul_norm','brand_norm'])
        # Обработка cross
        if 'cross' in dfs:
            idx+=1
            pbar.progress(idx/n_step, "Обработка кроссов")
            df_cross = dfs['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            self.upsert_data('cross_references', df_cross, ['oe_number_norm','artikul_norm','brand_norm'])
        # Обработка parts
        idx+=1
        pbar.progress(idx/n_step, "Обработка артикулов")
        parts_df = None
        p_files = {k:v for k,v in dfs.items() if k in ['oe','barcode','images','dimensions']}
        if p_files:
            all_parts = pl.concat([v.select(['artikul','artikul_norm','brand','brand_norm']) for v in p_files.values() if 'artikul_norm' in v.columns])
            all_parts = all_parts.filter(pl.col('artikul_norm') != "").unique(subset=['artikul_norm'])
            parts_df = all_parts
            for f in ['oe','barcode','images','dimensions']:
                if f not in p_files:
                    continue
                df = p_files[f]
                if df.is_empty() or 'artikul_norm' not in df.columns:
                    continue
                join_cols = [c for c in df.columns if c not in ['artikul','artikul_norm','brand','brand_norm']]
                if not join_cols:
                    continue
                existing_cols = set(parts_df.columns)
                join_cols = [c for c in join_cols if c not in existing_cols]
                if not join_cols:
                    continue
                df2 = df.select(['artikul_norm','brand_norm']+join_cols).unique(subset=['artikul_norm'])
                parts_df = parts_df.join(df2, on=['artikul_norm','brand_norm'], how='left', coalesce=True)
        if parts_df and not parts_df.is_empty():
            if 'multiplicity' not in parts_df.columns:
                parts_df = parts_df.with_columns(multiplicity=pl.lit(1).cast(pl.Int32))
            else:
                parts_df = parts_df.with_columns(pl.col('multiplicity').fill_null(1).cast(pl.Int32))
            for col in ['length','width','height']:
                if col not in parts_df.columns:
                    parts_df = parts_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
            if 'dimensions_str' not in parts_df.columns:
                parts_df = parts_df.with_columns(dimensions_str=pl.lit(None).cast(pl.Utf8))
            parts_df = parts_df.with_columns([
                pl.col('length').cast(pl.Utf8).fill_null('').alias('_length_str'),
                pl.col('width').cast(pl.Utf8).fill_null('').alias('_width_str'),
                pl.col('height').cast(pl.Utf8).fill_null('').alias('_height_str'),
            ])
            parts_df = parts_df.with_columns(
                dimensions_str=pl.when(
                    (pl.col('dimensions_str').is_not_null()) & (pl.col('dimensions_str') != '')
                ).then(
                    pl.col('dimensions_str')
                ).otherwise(
                    pl.concat_str([pl.col('_length_str'), 'x', pl.col('_width_str'), 'x', pl.col('_height_str')], separator='')
                )
            )
            parts_df = parts_df.drop(['_length_str','_width_str','_height_str'])
            # Создаем описание
            if 'artikul' not in parts_df.columns:
                parts_df = parts_df.with_columns(artikul=pl.lit(''))
            if 'brand' not in parts_df.columns:
                parts_df = parts_df.with_columns(brand=pl.lit(''))
            parts_df = parts_df.with_columns([
                pl.col('artikul').cast(pl.Utf8).fill_null('').alias('_artikul'),
                pl.col('brand').cast(pl.Utf8).fill_null('').alias('_brand'),
                pl.col('multiplicity').cast(pl.Utf8).alias('_multiplicity'),
            ])
            parts_df = parts_df.with_columns(
                description=pl.concat_str([
                    'Артикул: ', pl.col('_artikul'),
                    ', Бренд: ', pl.col('_brand'),
                    ', Кратность: ', pl.col('_multiplicity'), ' шт.'
                ], separator='')
            )
            parts_df = parts_df.drop(['_artikul','_brand','_multiplicity'])
            final_cols = ['artikul_norm','brand_norm','artikul','brand','multiplicity','barcode','length','width','height','weight','image_url','dimensions_str','description']
            parts_df = parts_df.select([pl.col(c) if c in parts_df.columns else pl.lit(None).alias(c) for c in final_cols])
            self.upsert_data('parts_data', parts_df, ['artikul_norm','brand_norm'])
        pbar.progress(1.0)
        time.sleep(1)
        pbar.empty()
        st.success("💾 Обновление завершено.")

    def merge_all_data_parallel(self, paths):
        start_time = time.time()
        dataframes = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.read_and_prepare_file, p, t): t for t, p in paths.items()}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    df = f.result()
                    if not df.is_empty():
                        dataframes[t] = df
                        st.success(f"✅ Загружен: {t} ({len(df):,} строк)")
                except Exception as e:
                    st.error(f"🚫 Ошибка {t}: {e}")
        if not dataframes:
            st.warning("Нет данных")
            return {}
        self.process_and_load_data(dataframes)
        stats = {
            'processing_time': time.time() - start_time,
            'total_records': self.get_total_records()
        }
        self.create_indexes()
        st.success(f"Обработка завершена за {stats['processing_time']:.2f} сек")
        return stats

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
                return {'total_parts': 0, 'total_oe': 0, 'total_brands': 0,
                        'top_brands': pl.DataFrame(), 'categories': pl.DataFrame()}
            res1 = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()
            stats['total_oe'] = res1[0] if res1 else 0
            res2 = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data WHERE brand IS NOT NULL").fetchone()
            stats['total_brands'] = res2[0] if res2 else 0
            brands = self.conn.execute("SELECT brand, COUNT(*) FROM parts_data WHERE brand IS NOT NULL GROUP BY brand ORDER BY 2 DESC LIMIT 10").fetchall()
            stats['top_brands'] = pl.DataFrame(brands, schema=["brand", "count"])
            cats = self.conn.execute("SELECT category, COUNT(*) FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY 2 DESC").fetchall()
            stats['categories'] = pl.DataFrame(cats, schema=["category", "count"])
        except:
            return {'total_parts':0,'total_oe':0,'total_brands':0,'top_brands':pl.DataFrame(),'categories':pl.DataFrame()}
        return stats

    def determine_category_vectorized(self, col):
        # Ваша логика определения категории
        return pl.lit('Категория')

    # --- разделы загрузки ---
    def start_initial_load(self, files_dict):
        required = ['oe','cross','barcode','dimensions','images']
        missing = [k for k in required if k not in files_dict]
        if missing:
            st.error(f"Нужно загрузить все 5 файлов для начальной загрузки: {', '.join(missing)}")
            return
        paths = {}
        for key, file in files_dict.items():
            filename = self.data_dir / f"{key}_{int(time.time())}_{file.name}"
            with open(filename, "wb") as f:
                f.write(file.getvalue())
            paths[key] = str(filename)
        self.merge_all_data_parallel(paths)

    def load_additional_files(self, files_dict):
        paths = {}
        for key, file in files_dict.items():
            filename = self.data_dir / f"{key}_{int(time.time())}_{file.name}"
            with open(filename, "wb") as f:
                f.write(file.getvalue())
            paths[key] = str(filename)
        self.merge_all_data_parallel(paths)

    # --- управление настройками ---
    def set_global_markup(self, value):
        self.global_markup = value

    def set_brand_markup(self, brand, value):
        self.brand_markup[brand] = value

    def set_excluded_brands(self, brands_list):
        self.excluded_brands = brands_list

    def set_excluded_artikuls(self, artikuls_list):
        self.excluded_artikuls = artikuls_list

    # --- экспорт ---
    def show_export_interface(self):
        st.header("📤 Экспорт данных")
        total = self.get_total_records()
        st.info(f"Всего записей: {total:,}")
        if total == 0:
            st.warning("Нет данных для экспорта")
            return
        columns = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена с наценкой"
        ]
        selected_cols = st.multiselect("Выберите колонки", columns, default=columns)
        format_option = st.radio("Формат экспорта", ["CSV", "Excel (.xlsx)", "Parquet"])

        if st.button("Экспортировать"):
            filename = None
            if format_option == "CSV":
                filename = self.export_to_csv_optimized("auto_parts_export.csv", selected_cols)
            elif format_option == "Excel (.xlsx)":
                filename = self.export_to_excel("auto_parts_export.xlsx", selected_cols)
            elif format_option == "Parquet":
                filename = self.export_to_parquet("auto_parts_export.parquet", selected_cols)
            if filename:
                with open(filename, "rb") as f:
                    st.download_button("Скачать файл", f, filename, mime="application/octet-stream")

    def build_export_query(self, selected_cols):
        # Полный SQL-запрос с CTE
        cols_map = {
            "артикул": "p.artikul",
            "бренд": "p.brand",
            "наименование": "p.name",
            "применимость": "p.applicability",
            "описание": "p.description",
            "категория": "o.category",
            "кратность": "p.multiplicity",
            "длина": "p.length",
            "ширина": "p.width",
            "высота": "p.height",
            "вес": "p.weight",
            "длина/ширина/высота": "p.dimensions_str",
            "oe номер": "o.oe_number",
            "аналоги": "array_agg(DISTINCT c.artikul_norm || '/' || c.brand_norm)",
            "ссылка на изображение": "p.image_url",
            "цена с наценкой": "pr.price"
        }
        select_list = []
        for col_name in selected_cols:
            sql_col = cols_map.get(col_name)
            if sql_col:
                select_list.append(sql_col + f" AS \"{col_name}\"")
            else:
                select_list.append("NULL AS \"" + col_name + "\"")
        select_clause = ", ".join(select_list)
        sql = f"""
WITH oe AS (
    SELECT oe_number_norm, oe_number, name, applicability, category
    FROM oe_data
),
cross AS (
    SELECT oe_number_norm, artikul_norm, brand_norm
    FROM cross_references
),
parts AS (
    SELECT
        artikul_norm,
        brand_norm,
        artikul,
        brand,
        multiplicity,
        barcode,
        length,
        width,
        height,
        weight,
        image_url,
        dimensions_str,
        description
    FROM parts_data
),
pr AS (
    SELECT artikul_norm, brand_norm, price
    FROM price_recommendations
)
SELECT
    {select_clause}
FROM parts p
LEFT JOIN oe o ON p.artikul_norm = o.oe_number_norm
LEFT JOIN cross c ON p.artikul_norm = c.artikul_norm AND p.brand_norm = c.brand_norm
LEFT JOIN pr ON p.artikul_norm = pr.artikul_norm AND p.brand_norm = pr.brand_norm
"""
        return sql

    def export_to_csv_optimized(self, filename, selected_cols):
        total = self.get_total_records()
        if total == 0:
            st.warning("Нет данных для экспорта")
            return
        query = self.build_export_query(selected_cols)
        df = self.conn.execute(query).pl()
        buf = io.StringIO()
        df.write_csv(buf, separator=';')
        csv_bytes = buf.getvalue().encode('utf-8')
        with open(filename, 'wb') as f:
            f.write(b'\xef\xbb\xbf')  # BOM
            f.write(csv_bytes)
        return filename

    def export_to_excel(self, filename, selected_cols):
        total = self.get_total_records()
        if total == 0:
            st.warning("Нет данных для экспорта")
            return
        # разбиваем по частям, создаем несколько файлов
        num_parts = (total + 999999) // 1000000
        files = []
        for i in range(num_parts):
            offset = i * 1000000
            query = self.build_export_query(selected_cols) + f" LIMIT 1000000 OFFSET {offset}"
            df = self.conn.execute(query).pl()
            part_filename = self.data_dir / f"{Path(filename).stem}_part{i+1}.xlsx"
            df.write_excel(str(part_filename))
            files.append(str(part_filename))
        if len(files) > 1:
            zip_path = self.data_dir / (Path(filename).stem + ".zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in files:
                    zf.write(f, arcname=os.path.basename(f))
                    os.remove(f)
            return str(zip_path)
        else:
            return str(files[0])

    def export_to_parquet(self, filename, selected_cols):
        query = self.build_export_query(selected_cols)
        df = self.conn.execute(query).pl()
        df.write_parquet(filename)
        return filename

    # --- установка настроек ---
    def set_global_markup(self, value):
        self.global_markup = value

    def set_brand_markup(self, brand, value):
        self.brand_markup[brand] = value

    def set_excluded_brands(self, brands_list):
        self.excluded_brands = brands_list

    def set_excluded_artikuls(self, artikuls_list):
        self.excluded_artikuls = artikuls_list

    # --- фильтры ---
    def apply_filters(self, df):
        # Исключение брендов и артикула
        if self.excluded_brands:
            df = df.filter(~pl.col('brand').is_in(self.excluded_brands))
        if self.excluded_artikuls:
            df = df.filter(~pl.col('artikul').is_in(self.excluded_artikuls))
        return df

    def get_filtered_parts(self):
        query = "SELECT * FROM parts_data"
        df = self.conn.execute(query).pl()
        return self.apply_filters(df)

    def determine_category_vectorized(self, col):
        return pl.lit('Категория')


def main():
    catalog = AutoPartsCatalog()

    st.title("🚗 AutoParts Catalog - 10+ млн записей")
    st.sidebar.title("🧭 Навигация")
    choice = st.sidebar.radio("Выберите действие", ["Загрузка данных", "Экспорт", "Статистика", "Управление"])

    if choice == "Загрузка данных":
        st.header("📥 Загрузка и обработка данных")
        is_empty = catalog.get_total_records() == 0
        if is_empty:
            st.warning("⚠️ База пуста. Необходимо выполнить полную начальную загрузку всех 5 файлов.")
        else:
            st.info("Можно добавлять файлы по одному или несколько.")

        files = {
            'oe': st.file_uploader("Основные данные (OE)", type=['xlsx']),
            'cross': st.file_uploader("Кроссы", type=['xlsx']),
            'barcode': st.file_uploader("Штрих-коды", type=['xlsx']),
            'dimensions': st.file_uploader("Весогабариты", type=['xlsx']),
            'images': st.file_uploader("Изображения", type=['xlsx'])
        }

        if st.button("🚀 Начать обработку данных"):
            files_paths = {}
            for key, file in files.items():
                if file:
                    filename = catalog.data_dir / f"{key}_{int(time.time())}_{file.name}"
                    with open(filename, "wb") as f:
                        f.write(file.getvalue())
                    files_paths[key] = str(filename)
            if catalog.get_total_records() == 0:
                missing = [k for k in ['oe','cross','barcode','dimensions','images'] if k not in files_paths]
                if missing:
                    st.error(f"Нужно загрузить все 5 файлов. Отсутствуют: {', '.join(missing)}")
                else:
                    catalog.start_initial_load(files_paths)
            else:
                if files_paths:
                    catalog.load_additional_files(files_paths)

    elif choice == "Экспорт":
        catalog.show_export_interface()

        # дополнительно: установка наценки и исключений
        st.sidebar.header("Настройки экспорта")
        with st.sidebar.expander("Настройки цен и исключений"):
            global_markup_input = st.number_input("Глобальная наценка (%)", min_value=0.0, max_value=100.0, value=catalog.global_markup)
            catalog.set_global_markup(global_markup_input)
            brand_markup_input = st.text_input("Наценки по брендам (через запятую, бренд=процент)", "")
            if brand_markup_input:
                for item in brand_markup_input.split(','):
                    if '=' in item:
                        brand, percent = item.split('=')
                        try:
                            catalog.set_brand_markup(brand.strip(), float(percent.strip()))
                        except:
                            continue
            exclude_brands_input = st.text_input("Исключить бренды (через запятую)", "")
            if exclude_brands_input:
                catalog.set_excluded_brands([b.strip() for b in exclude_brands_input.split(',')])
            exclude_artikul_input = st.text_input("Исключить артикулы (через запятую)", "")
            if exclude_artikul_input:
                catalog.set_excluded_artikuls([a.strip() for a in exclude_artikul_input.split(',')])

    elif choice == "Статистика":
        stats = catalog.get_statistics()
        st.subheader("Статистика")
        st.write(f"Общее количество артикулов: {stats.get('total_parts',0):,}")
        st.write(f"Общее количество OE: {stats.get('total_oe',0):,}")
        st.write(f"Общее количество брендов: {stats.get('total_brands',0):,}")
        st.subheader("Топ брендов")
        st.dataframe(stats.get('top_brands', pl.DataFrame()))
        st.subheader("Категории")
        st.dataframe(stats.get('categories', pl.DataFrame()))

    elif choice == "Управление":
        st.header("🗑️ Удаление данных")
        op = st.radio("Выберите операцию", ["Удалить по бренду", "Удалить по артикулу"])
        if op == "Удалить по бренду":
            brands_result = catalog.conn.execute("SELECT DISTINCT brand FROM parts_data WHERE brand IS NOT NULL ORDER BY 1").fetchall()
            brands = [b[0] for b in brands_result]
            selected_brand = st.selectbox("Выберите бренд", brands)
            res = catalog.conn.execute("SELECT brand_norm FROM parts_data WHERE brand = ? LIMIT 1", [selected_brand]).fetchone()
            brand_norm = res[0] if res else ""
            count = catalog.conn.execute("SELECT COUNT(*) FROM parts_data WHERE brand_norm = ?", [brand_norm]).fetchone()[0]
            st.write(f"Записей для удаления: {count}")
            if st.button("Удалить все для этого бренда"):
                catalog.conn.execute("DELETE FROM parts_data WHERE brand_norm = ?", [brand_norm])
                catalog.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT artikul_norm, brand_norm FROM parts_data)")
                st.success("Удалено.")
        else:
            artikul_input = st.text_input("Введите артикул для удаления")
            if artikul_input:
                normalized = catalog.normalize_key(pl.Series([artikul_input]))[0]
                count = catalog.conn.execute("SELECT COUNT(*) FROM parts_data WHERE artikul_norm = ?", [normalized]).fetchone()[0]
                st.write(f"Записей для удаления: {count}")
                if st.button("Удалить по артикулу"):
                    catalog.conn.execute("DELETE FROM parts_data WHERE artikul_norm = ?", [normalized])
                    catalog.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT artikul_norm, brand_norm FROM parts_data)")
                    st.success("Удалено.")


if __name__ == "__main__":
    main()
