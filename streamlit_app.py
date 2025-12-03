import streamlit as st
import duckdb
import polars as pl
import io
import time
import json
from pathlib import Path

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

    # --- Категории ---
    def get_categories(self):
        return self.conn.execute("SELECT name, description FROM categories").fetchdf()

    def add_category(self, name, description=''):
        try:
            self.conn.execute("INSERT INTO categories (name, description) VALUES (?, ?)", [name, description])
        except Exception:
            pass

    def delete_category(self, name):
        self.conn.execute("DELETE FROM categories WHERE name=?", [name])

    def update_category_name(self, old_name, new_name):
        self.conn.execute("UPDATE categories SET name=? WHERE name=?", [new_name, old_name])

    def load_categories(self, file_bytes):
        try:
            df = pl.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(f"Ошибка чтения файла категорий: {e}")
            return
        if 'наименование' not in df.columns or 'категория' not in df.columns:
            st.error("Файл должен содержать 'наименование' и 'категория'")
            return
        df = df.select([pl.col('наименование'), pl.col('категория')])
        self.conn.execute("DELETE FROM categories")
        for row in df.iter_rows():
            self.add_category(row[0], row[1])

    def show_categories(self):
        df = self.get_categories()
        st.subheader("Текущие категории")
        edited_df = st.experimental_data_editor(df, use_container_width=True)
        if st.button("Обновить категории"):
            # Обновление названий и удаление
            original_names = df['name'].tolist()
            updated_names = edited_df['name'].tolist()
            for old_name, new_name in zip(original_names, edited_df['name']):
                if old_name != new_name:
                    self.update_category_name(old_name, new_name)
            # Удаление отсутствующих
            for name in original_names:
                if name not in edited_df['name'].tolist():
                    self.delete_category(name)
            st.success("Категории обновлены!")

    # --- Загрузка и обработка файлов ---
    def read_and_prepare_file(self, filepath, key):
        try:
            df = pl.read_excel(filepath)
            # Предобработка зависит от типа файла
            if key == 'oe':
                df = df.rename({"№ OE": "oe_number", "Наименование": "name", "Применимость": "applicability", "Категория": "category"})
                df['oe_number_norm'] = df['oe_number'].str.strip().str.upper()
            elif key == 'cross':
                df = df.rename({"OE": "oe_number", "Артикул": "artikul", "Бренд": "brand"})
                df['oe_number_norm'] = df['oe_number'].str.strip().str.upper()
                df['artikul_norm'] = df['artikul'].str.strip().str.upper()
            elif key == 'barcode':
                df = df.rename({"Артикул": "artikul", "Бренд": "brand", "Штрихкод": "barcode"})
            elif key == 'dimensions':
                df = df.rename({"Артикул": "artikul", "Бренд": "brand", "Длина": "length", "Ширина": "width", "Высота": "height", "Вес": "weight"})
            elif key == 'images':
                df = df.rename({"Артикул": "artikul", "Бренд": "brand", "Изображение": "image_url"})
            elif key == 'categories':
                # уже обработано при загрузке
                pass
            return df
        except Exception as e:
            st.error(f"Ошибка чтения файла {filepath}: {e}")
            return None

    def process_and_load(self, dataframes):
        # Обработка и загрузка в базу
        if 'oe' in dataframes:
            df_oe = dataframes['oe']
            for row in df_oe.iter_rows():
                self.conn.execute("""
                    INSERT OR REPLACE INTO oe_data (oe_number_norm, oe_number, name, applicability, category)
                    VALUES (?, ?, ?, ?, ?)
                """, [row['oe_number_norm'], row['oe_number'], row['name'], row.get('applicability', ''), row.get('category', '')])
        if 'cross' in dataframes:
            df_cross = dataframes['cross']
            for row in df_cross.iter_rows():
                self.conn.execute("""
                    INSERT OR REPLACE INTO cross_references (oe_number_norm, artikul_norm, brand_norm)
                    VALUES (?, ?, ?)
                """, [row['oe_number'].strip().upper(), row['artikul'].strip().upper(), row['brand'].strip().upper()])
        if 'barcode' in dataframes:
            # обработка штрихкодов
            pass
        if 'dimensions' in dataframes:
            df_dim = dataframes['dimensions']
            for row in df_dim.iter_rows():
                self.conn.execute("""
                    UPDATE OR INSERT INTO parts_data (artikul_norm, brand_norm, length, width, height, weight)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [row['artikul'].strip().upper(), row['brand'].strip().upper(), row['length'], row['width'], row['height'], row['weight']])
        # добавьте остальные обработки по необходимости

    def load_recommended_prices(self, file_bytes):
        try:
            df = pl.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(f"Ошибка чтения файла рекомендаций: {e}")
            return
        if 'Артикул' not in df.columns or 'Цена' not in df.columns:
            st.error("Файл должен содержать 'Артикул' и 'Цена'")
            return
        for row in df.iter_rows():
            artikul = row['Артикул'].strip().upper()
            price = row['Цена']
            self.conn.execute("""
                INSERT OR REPLACE INTO recommended_prices (artikul_norm, price)
                VALUES (?, ?)
            """, [artikul, price])
        st.success("Рекомендации цен успешно загружены.")

    def load_price_list(self, file_bytes):
        try:
            df = pl.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(f"Ошибка чтения файла прайс-листа: {e}")
            return
        if not {'Артикул', 'Бренд', 'Количество', 'Цена'}.issubset(df.columns):
            st.error("Файл должен содержать 'Артикул', 'Бренд', 'Количество', 'Цена'")
            return
        for row in df.iter_rows():
            artikul = row['Артикул'].strip().upper()
            brand = row['Бренд'].strip()
            quantity = row['Количество']
            price = row['Цена']
            self.conn.execute("""
                INSERT OR REPLACE INTO price_list (artikul, brand, quantity, price)
                VALUES (?, ?, ?, ?)
            """, [artikul, brand, quantity, price])
        st.success("Прайс-лист успешно загружен.")

    # --- Настройки наценок ---
    def get_markups(self):
        row = self.conn.execute("SELECT total_markup, brand_markup FROM markup_settings WHERE id=1").fetchone()
        if row:
            total_markup, brand_markup_json = row
            try:
                brand_markup = json.loads(brand_markup_json) if brand_markup_json else {}
            except:
                brand_markup = {}
            return total_markup, brand_markup
        return 0, {}

    def set_markups(self, total_markup, brand_markup):
        self.conn.execute("""
            UPDATE markup_settings SET total_markup=?, brand_markup=?
            WHERE id=1
        """, [total_markup, json.dumps(brand_markup)])
        st.success("Настройки наценки сохранены!")

    def apply_markup(self, price, total_markup, brand_markup_dict, brand):
        markup = total_markup
        if brand and brand in brand_markup_dict:
            markup += brand_markup_dict[brand]
        return round(price * (1 + markup / 100), 2)

    # --- Экспорт ---
    def show_export_interface(self):
        st.subheader("Экспорт данных")
        all_cols = [
            'artikul', 'brand', 'category', 'length', 'width', 'height', 'weight',
            'image_url', 'description', 'oe_list', 'price_with_markup'
        ]
        selected_cols = st.multiselect("Выберите колонки для экспорта (в желаемом порядке)", all_cols, default=all_cols)
        exclude_input = st.text_input("Исключить позиции по названию (через |)", "")

        categories = self.get_categories()['name'].tolist()
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
                category_filter=None if 'Все' in category_filter else category_filter
            )
            if df is not None:
                total_markup, brand_markup = self.get_markups()
                if 'price_with_markup' in df.columns:
                    df = df.with_columns(
                        pl.col('price_with_markup').apply(
                            lambda p, br=pl.col('brand'): self.apply_markup(p, total_markup, brand_markup, br[0])
                        ).alias('price_with_markup')
                    )
                buffer = io.BytesIO()
                df.write_excel(buffer)
                buffer.seek(0)
                filename = f"export_{int(time.time())}.xlsx"
                st.download_button("Скачать файл", data=buffer, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Главная логика ---
def main():
    st.set_page_config(page_title="AutoParts Catalog", layout="wide")
    st.title("🚗 AutoParts Catalog — Управление и экспорт")
    catalog = AutoPartsCatalog()

    menu = st.sidebar.radio("Меню", ["Загрузка данных", "Категории", "Настройки", "Экспорт", "Статистика", "Рекомендации цен", "Прайс-лист"])

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
                catalog.load_categories(files_map['categories'].read())

            if dataframes:
                catalog.process_and_load(dataframes)
                st.success("Обработка завершена.")
            else:
                st.info("Загрузите файлы для обработки.")

    elif menu == "Категории":
        catalog.show_categories()

    elif menu == "Настройки":
        st.subheader("Настройки наценок")
        total_markup, brand_markup = catalog.get_markups()
        new_total_markup = st.number_input("Общая наценка (%)", value=total_markup, step=0.1)
        brand_df = catalog.conn.execute("SELECT brand FROM parts_data GROUP BY brand").fetchdf()
        brand_markup_dict = {}
        st.write("Наценки по брендам:")
        for index, row in brand_df.iterrows():
            brand = row['brand']
            default_markup = brand_markup.get(brand, 0)
            markup_value = st.number_input(f"{brand}", value=default_markup, step=0.1, key=f"markup_{brand}")
            brand_markup_dict[brand] = markup_value
        if st.button("Сохранить настройки наценки"):
            catalog.set_markups(new_total_markup, brand_markup_dict)

    elif menu == "Экспорт":
        catalog.show_export_interface()

    elif menu == "Статистика":
        stats = catalog.get_statistics()
        st.subheader("Статистика")
        st.write(f"Общее количество запчастей: {stats['total_parts']}")
        st.write(f"Общее количество категорий: {stats['total_categories']}")

    elif menu == "Рекомендации цен":
        uploaded = st.file_uploader("Загрузите файл с рекомендациями по ценам", type=['xlsx', 'xls'])
        if uploaded:
            catalog.load_recommended_prices(uploaded.read())

    elif menu == "Прайс-лист":
        uploaded = st.file_uploader("Загрузите прайс-лист", type=['xlsx', 'xls'])
        if uploaded:
            catalog.load_price_list(uploaded.read())

if __name__ == "__main__":
    main()
