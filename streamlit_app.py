import streamlit as st
import duckdb
import polars as pl
import io
import os
import time
import json
from pathlib import Path
from difflib import get_close_matches

# Константы
DATA_DIR = Path("./auto_parts_data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "catalog.duckdb"

class AutoPartsCatalog:
    def __init__(self):
        self.conn = duckdb.connect(str(DB_PATH))
        self._setup_database()
        self._create_indexes()

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
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                name VARCHAR PRIMARY KEY,
                description VARCHAR
            )
        """)

    def _create_indexes(self):
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_oe ON oe_data(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_parts ON parts_data(artikul_norm, brand_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross ON cross_references(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross_art ON cross_references(artikul_norm, brand_norm)")

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
            step += 1
            progress.progress(step / total_steps, text=f"Обработка OE ({step}/{total_steps})")
            df_oe = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df_oe.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'])
            # добавляем категорию (здесь убрали, так как просили)
            # if 'name' in oe_df.columns:
            #     oe_df = oe_df.with_columns(self._category_by_name(pl.col('name')).alias('category'))
            # else:
            #     oe_df = oe_df.with_columns(pl.lit('Разное').alias('category'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])
            cross_df = df_oe.filter(pl.col('artikul_norm') != "").select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка cross
        if 'cross' in dataframes:
            step += 1
            progress.progress(step / total_steps, text=f"Обработка кроссов ({step}/{total_steps})")
            df_cross = dataframes['cross'].filter(
                (pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != "")
            )
            self.upsert_data('cross_references', df_cross, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка parts
        step += 1
        progress.progress(step / total_steps, text=f"Обработка артикула ({step}/{total_steps})")
        # Можно расширить обработку
        progress.progress(1)
        time.sleep(0.5)
        st.success("🗃️ Обновление завершено!")

    def load_category_data(self, file_bytes):
        """Загрузка файла с наименованиями и категориями"""
        df = pl.read_excel(io.BytesIO(file_bytes))
        # Предполагается, что файл содержит наименование и категорию
        if 'наименование' not in df.columns or 'категория' not in df.columns:
            st.error("Файл должен содержать 'наименование' и 'категория'")
            return
        df = df.select([
            pl.col('наименование'),
            pl.col('категория')
        ])
        # Загрузка данных в память
        self._category_data = df

    def assign_categories(self, df_names):
        """Поиск приблизительных совпадений и присвоение категории"""
        if not hasattr(self, '_category_data'):
            return pl.Series([''] * len(df_names))
        category_series = []
        categories = self._category_data['наименование'].to_list()
        categories_lower = [cat.lower() for cat in categories]
        for name in df_names:
            name_lower = name.lower() if name else ''
            matches = get_close_matches(name_lower, categories_lower, n=1, cutoff=0.6)
            if matches:
                idx = categories_lower.index(matches[0])
                category_series.append(self._category_data['категория'][idx])
            else:
                category_series.append('')
        return pl.Series(category_series)

    def build_export_query(self, selected_columns=None, category_filter=None):
        desc_text = """Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""
        ctes = f"""
        WITH DescriptionTemplate AS (
            SELECT '{desc_text}' AS text
        ),
        PartDetails AS (
            SELECT
                cr.artikul_norm,
                cr.brand_norm,
                STRING_AGG(DISTINCT regexp_replace(regexp_replace(o.oe_number, '''', ''), '[^0-9A-Za-zА-Яа-яЁё`\\-\\s]', '', 'g'), ', ') AS oe_list,
                ANY_VALUE(o.name) AS representative_name,
                ANY_VALUE(o.applicability) AS representative_applicability,
                ANY_VALUE(o.category) AS representative_category,
                ANY_VALUE(p.description) AS description,
                ANY_VALUE(p.category) AS category,
                ANY_VALUE(p.length) AS length,
                ANY_VALUE(p.width) AS width,
                ANY_VALUE(p.height) AS height,
                ANY_VALUE(p.weight) AS weight,
                ANY_VALUE(p.dimensions_str) AS dimensions_str,
                ANY_VALUE(p.analog_list) AS analog_list,
                ANY_VALUE(p.image_url) AS image_url,
                ANY_VALUE(p.oe_list) AS oe_list,
                ANY_VALUE(p.price_with_markup) AS price_with_markup,
                ROW_NUMBER() OVER (PARTITION BY cr.artikul_norm, cr.brand_norm ORDER BY o.oe_number) AS rn
            FROM cross_references cr
            LEFT JOIN oe_data o ON cr.oe_number_norm = o.oe_number_norm
            LEFT JOIN parts_data p ON cr.artikul_norm = p.artikul_norm AND cr.brand_norm = p.brand_norm
        """
        # Фильтр по категориям
        if category_filter:
            categories_str = "', '".join(category_filter)
            ctes += f"\nWHERE p.category IN ('{categories_str}')\n"
        ctes += """
            GROUP BY cr.artikul_norm, cr.brand_norm
        )
        SELECT
        """
        if selected_columns:
            select_cols = ', '.join(selected_columns)
        else:
            select_cols = '*'
        ctes += f" {select_cols} FROM PartDetails p WHERE p.rn=1"
        return ctes

    def get_markups(self):
        row = self.conn.execute("SELECT total_markup, brand_markup FROM markup_settings WHERE id=1").fetchone()
        if row:
            total_markup, brand_markup_json = row
            brand_markup = json.loads(brand_markup_json) if brand_markup_json else {}
            return total_markup, brand_markup
        return 0, {}

    def set_markups(self, total_markup, brand_markup):
        self.conn.execute("""
            UPDATE markup_settings SET total_markup=?, brand_markup=?
            WHERE id=1
        """, [total_markup, json.dumps(brand_markup)])
        st.success("Настройки наценки сохранены!")

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

    def get_filtered_exclusions(self, exclude_terms):
        """Возвращает список строк, которые нужно исключить по названиям"""
        exclude_list = [term.strip() for term in exclude_terms.split('|') if term.strip()]
        return exclude_list

    def filter_exclusions(self, df, exclude_terms):
        """Фильтрует DataFrame, исключая строки по названиям"""
        exclude_list = self.get_filtered_exclusions(exclude_terms)
        if not exclude_list:
            return df
        mask = pl.Series([False] * len(df))
        for term in exclude_list:
            mask = mask | df['name'].str.contains(term, case=False)
        return df.filter(~mask)

    def export_data(self, columns=None, exclude_terms=None, category_filter=None):
        """
        Экспорт данных с возможностью исключений и выбора колонок.
        columns - список выбранных колонок в желаемом порядке.
        exclude_terms - строки для исключения (через |), могут быть точные или частичные совпадения.
        category_filter - список категорий для фильтрации.
        """
        query = self.build_export_query(selected_columns=columns, category_filter=category_filter)
        df = self._run_query(query)
        if df is None or df.is_empty():
            st.info("Нет данных для экспорта.")
            return None
        if exclude_terms:
            df = self.filter_exclusions(df, exclude_terms)
        return df

    def _run_query(self, query):
        try:
            return pl.read_sql(query, self.conn)
        except Exception as e:
            st.error(f"Ошибка выполнения запроса: {e}")
            return None

    def get_statistics(self):
        total_parts = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]
        total_oe = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()[0]
        total_brands = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data").fetchone()[0]
        top_brands = self.conn.execute("""
            SELECT brand, COUNT(*) as cnt FROM parts_data GROUP BY brand ORDER BY cnt DESC LIMIT 10
        """).fetchdf()
        categories = self.conn.execute("""
            SELECT category, COUNT(*) as cnt FROM parts_data GROUP BY category ORDER BY cnt DESC
        """).fetchdf()
        return {
            'total_parts': total_parts,
            'total_oe': total_oe,
            'total_brands': total_brands,
            'top_brands': top_brands,
            'categories': categories
        }

    def show_export_interface(self):
        st.subheader("Экспорт данных")
        all_cols = [
            'artikul', 'brand', 'category', 'length', 'width', 'height', 'weight',
            'image_url', 'description', 'oe_list', 'price_with_markup'
        ]
        selected_cols = st.multiselect("Выберите колонки для экспорта (в желаемом порядке)", all_cols, default=all_cols)
        selected_cols = list(selected_cols)

        exclude_input = st.text_input("Исключить позиции по названию (через |)", "")

        categories = self.conn.execute("SELECT DISTINCT category FROM parts_data").fetchdf()['category'].tolist()
        categories.insert(0, 'Все')
        category_filter = st.multiselect("Фильтр по категориям", categories, default=['Все'])
        if 'Все' in category_filter:
            category_filter = None
        elif not category_filter:
            category_filter = None
        else:
            category_filter = category_filter

        total_markup_value = st.number_input("Общая наценка (%)", value=0.0, step=0.1)
        brand_markup_df = self.conn.execute("SELECT brand, COUNT(*) as cnt FROM parts_data GROUP BY brand").fetchdf()
        brand_markup_dict = {}
        st.write("Настройки наценок по брендам:")
        for index, row in brand_markup_df.iterrows():
            brand = row['brand']
            default_markup = 0.0
            markup_value = st.number_input(f"{brand}", value=default_markup, step=0.1, key=f"markup_{brand}")
            brand_markup_dict[brand] = markup_value

        if st.button("Сохранить настройки наценки"):
            self.set_markups(total_markup_value, brand_markup_dict)

        if st.button("Экспортировать данные"):
            df = self.export_data(
                columns=selected_cols,
                exclude_terms=exclude_input,
                category_filter=None if category_filter is None or 'Все' in category_filter else category_filter
            )
            if df is not None:
                total_markup, brand_markup = self.get_markups()
                if 'price_with_markup' in df.columns:
                    df = df.with_columns(
                        pl.col('price_with_markup').apply(
                            lambda p: self.apply_markup(p, total_markup, brand_markup, None)
                        ).alias('price_with_markup')
                    )
                buffer = io.BytesIO()
                df.write_excel(buffer)
                buffer.seek(0)
                filename = f"export_{int(time.time())}.xlsx"
                st.download_button("Скачать файл", data=buffer, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    def apply_markup(self, price, total_markup, brand_markup_dict, brand):
        markup = total_markup
        if brand and brand in brand_markup_dict:
            markup += brand_markup_dict[brand]
        return price * (1 + markup / 100)

    def get_markups(self):
        row = self.conn.execute("SELECT total_markup, brand_markup FROM markup_settings WHERE id=1").fetchone()
        if row:
            total_markup, brand_markup_json = row
            brand_markup = json.loads(brand_markup_json) if brand_markup_json else {}
            return total_markup, brand_markup
        return 0, {}

    def set_markups(self, total_markup, brand_markup):
        self.conn.execute("""
            UPDATE markup_settings SET total_markup=?, brand_markup=?
            WHERE id=1
        """, [total_markup, json.dumps(brand_markup)])
        st.success("Настройки наценки сохранены!")

# Основной интерфейс
def main():
    st.set_page_config(page_title="AutoParts Catalog", layout="wide")
    st.title("🚗 AutoParts Catalog — Расширенное управление и экспорт")
    catalog = AutoPartsCatalog()

    menu = st.sidebar.radio("Меню", ["Загрузка данных", "Экспорт", "Статистика", "Рекомендации цен", "Прайс-лист"])

    if menu == "Загрузка данных":
        st.subheader("Загрузите файлы для обработки")
        col1, col2 = st.columns(2)
        with col1:
            file_oe = st.file_uploader("Основные данные (OE)", type=['xlsx', 'xls'])
            file_cross = st.file_uploader("Кроссы (OE → Артикул)", type=['xlsx', 'xls'])
            file_barcode = st.file_uploader("Штрих-коды", type=['xlsx', 'xls'])
        with col2:
            file_dim = st.file_uploader("Весогабариты", type=['xlsx', 'xls'])
            file_img = st.file_uploader("Изображения", type=['xlsx', 'xls'])
            file_category = st.file_uploader("Категории (наименование, категория)", type=['xlsx', 'xls'])

        files_map = {
            'oe': file_oe,
            'cross': file_cross,
            'barcode': file_barcode,
            'dimensions': file_dim,
            'images': file_img,
            'categories': file_category
        }

        if st.button("🚀 Обработать файлы"):
            dataframes = {}
            for key, uploaded in files_map.items():
                if uploaded:
                    filename = f"{key}_{int(time.time())}_{uploaded.name}"
                    path = DATA_DIR / filename
                    with open(path, "wb") as f:
                        f.write(uploaded.read())
                    df = catalog.read_and_prepare_file(str(path), key)
                    dataframes[key] = df
            # Загрузка файла с категориями
            if 'categories' in files_map and files_map['categories']:
                catalog.load_category_data(files_map['categories'].read())

            if dataframes:
                catalog.process_and_load(dataframes)
                # После загрузки, присвоение категорий по названию
                if hasattr(catalog, '_category_data'):
                    # Обновляем категории по совпадениям
                    # Для всех строк в parts_data
                    df_parts = catalog.conn.execute("SELECT artikul_norm, brand_norm, artikul, brand FROM parts_data").fetchdf()
                    if not df_parts.empty:
                        categories_assigned = catalog.assign_categories(pl.Series(df_parts['artikul']))
                        # Обновляем категории в parts_data
                        for idx, row in df_parts.iterrows():
                            category_name = categories_assigned[idx]
                            catalog.conn.execute("""
                                UPDATE parts_data SET category=? WHERE artikul_norm=? AND brand_norm=?
                            """, [category_name, row['artikul_norm'], row['brand_norm']])
                        st.success("Категории по наименованиям успешно присвоены!")
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
