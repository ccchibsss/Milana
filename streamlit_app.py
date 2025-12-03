import polars as pl
import duckdb
import streamlit as st
import os
import time
import io
import zipfile
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXCEL_ROW_LIMIT = 1_000_000

class AutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(database=str(self.db_path))
        self.setup_database()

        # Встроенные функции обработки файлов
        self._setup_file_processing_functions()

        # Настройки цен
        self.overall_markup = 1.2  # 20% наценка по умолчанию
        self.brand_markups = {}  # по брендам

        st.set_page_config(
            page_title="AutoParts Catalog 10M+", 
            layout="wide",
            page_icon="🚗"
        )

    def _setup_file_processing_functions(self):
        def read_and_prepare_file(file_path: str, file_type: str):
            logger.info(f"Начинаю обработку файла: {file_type} ({file_path})")
            try:
                if not os.path.exists(file_path):
                    logger.error(f"Файл не найден: {file_path}")
                    return pl.DataFrame()
                if os.path.getsize(file_path) == 0:
                    logger.warning(f"Файл пуст: {file_path}")
                    return pl.DataFrame()
                df = pl.read_excel(file_path, engine='calamine')
                if df.is_empty():
                    logger.warning(f"Файл прочитан, но не содержит данных: {file_path}")
                    return pl.DataFrame()
            except Exception as e:
                logger.exception(f"Не удалось прочитать файл {file_path}: {e}")
                return pl.DataFrame()

            schemas = {
                'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
                'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
                'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
                'images': ['artikul', 'brand', 'image_url'],
                'cross': ['oe_number', 'artikul', 'brand'],
                'price': ['artikul', 'brand', 'quantity', 'price']
            }
            expected_cols = schemas.get(file_type, [])
            column_mapping = self.detect_columns(df.columns, expected_cols)
            if not column_mapping:
                logger.warning(f"Не удалось определить колонки для файла {file_type}. Доступные колонки: {df.columns}")
                return pl.DataFrame()
            df = df.rename(column_mapping)

            # Очистка значений
            for col in ['artikul', 'brand', 'oe_number', 'name', 'applicability']:
                if col in df.columns:
                    df = df.with_columns(**{col: self.clean_values(pl.col(col))})

            # Для файла цен, приведем цену к float
            if 'price' in df.columns:
                df = df.with_columns(price=pl.col('price').cast(pl.Float64))
            if 'quantity' in df.columns:
                df = df.with_columns(quantity=pl.col('quantity').cast(pl.Int64))

            # Удаляем дубликаты по ключу
            key_cols = [col for col in ['artikul', 'brand', 'oe_number'] if col in df.columns]
            if key_cols:
                df = df.unique(subset=key_cols, keep='first')
            return df

        def detect_columns(self, actual_columns, expected_columns):
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
                'dimensions_str': ['весогабариты', 'размеры', 'dimensions', 'size'],
                'quantity': ['количество', 'quantity', 'qty'],
                'price': ['цена', 'price']
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

        def clean_values(self, value_series: pl.Series) -> pl.Series:
            return (
                value_series
                .fill_null("")
                .cast(pl.Utf8)
                .str.replace_all("'", "")
                .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
            )

        def normalize_key(self, key_series: pl.Series) -> pl.Series:
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

        self.read_and_prepare_file = read_and_prepare_file
        self.detect_columns = detect_columns
        self.clean_values = clean_values
        self.normalize_key = normalize_key

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
                price FLOAT,
                quantity INTEGER,
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

    def create_indexes(self):
        st.info("Создание индексов...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)"
        ]
        for sql in indexes:
            self.conn.execute(sql)
        st.success("Индексы созданы.")

    def process_and_load_data(self, dataframes: dict):
        # Основная логика - обновление базы данными из файлов
        # Обновление или вставка данных по артикулам
        if 'price' in dataframes:
            df_price = dataframes['price']
            # Обновляем цену и количество по артикулу
            for _, row in df_price.iterrows():
                artikul_norm = self.normalize_key(pl.Series([row['artikul']]))[0]
                brand_norm = self.normalize_key(pl.Series([row['brand']]))[0]
                quantity = row.get('quantity', 0)
                price = row.get('price', 0.0)
                self.conn.execute("""
                    UPDATE parts_data SET price = ?, quantity = ? WHERE artikul_norm = ? AND brand_norm = ?
                """, [price, quantity, artikul_norm, brand_norm])
                # Или вставляем, если не существует
                self.conn.execute("""
                    INSERT INTO parts_data (artikul_norm, brand_norm, artikul, brand, price, quantity)
                    SELECT ?, ?, ?, ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM parts_data WHERE artikul_norm=? AND brand_norm=?
                    )
                """, [artikul_norm, brand_norm, row['artikul'], row['brand'], price, quantity, artikul_norm, brand_norm])

        # Можно дополнительно обновлять остальные данные, если необходимо.

    def update_pricing(self, overall_markup=None, brand_markups=None):
        # Устанавливаем общую наценку и по брендам
        if overall_markup:
            self.overall_markup = overall_markup
        if brand_markups:
            self.brand_markups.update(brand_markups)

        # Обновляем цены в базе
        cursor = self.conn.execute("SELECT artikul_norm, brand_norm, price FROM parts_data WHERE price IS NOT NULL")
        for artikul_norm, brand_norm, base_price in cursor.fetchall():
            markup = self.brand_markups.get(brand_norm, self.overall_markup)
            new_price = base_price * markup
            self.conn.execute("""
                UPDATE parts_data SET price = ? WHERE artikul_norm = ? AND brand_norm = ?
            """, [new_price, artikul_norm, brand_norm])
        st.success("Цены обновлены с учетом наценок.")

    def load_price_file(self, file_path):
        df = self.read_and_prepare_file(file_path, 'price')
        if df and not df.is_empty():
            self.process_and_load_data({'price': df})
            st.success("Прайс успешно загружен и применен.")
        else:
            st.warning("Прайс файл пуст или не удалось его распарсить.")

    def merge_all_data_parallel(self, file_paths: dict):
        start_time = time.time()
        dataframes = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.read_and_prepare_file, path, ftype): ftype
                for ftype, path in file_paths.items()
            }
            for future in as_completed(futures):
                ftype = futures[future]
                try:
                    df = future.result()
                    if not df.is_empty():
                        dataframes[ftype] = df
                        st.success(f"Файл {ftype} загружен: {len(df)} строк")
                except Exception as e:
                    st.error(f"Ошибка при загрузке {ftype}: {e}")

        if not dataframes:
            st.warning("Нет данных для загрузки.")
            return {}
        self.process_and_load_data(dataframes)
        stats = {
            'processing_time': time.time() - start_time,
            'total_records': self.get_total_records()
        }
        self.create_indexes()
        return stats

    def get_total_records(self):
        try:
            res = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()
            return res[0] if res else 0
        except:
            return 0

    def get_statistics(self):
        stats = {}
        try:
            stats['total_parts'] = self.get_total_records()
            res_oe = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()
            stats['total_oe'] = res_oe[0] if res_oe else 0
            res_b = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data WHERE brand IS NOT NULL").fetchone()
            stats['total_brands'] = res_b[0] if res_b else 0
            # Топ брендов
            top_b = self.conn.execute("SELECT brand, COUNT(*) FROM parts_data WHERE brand IS NOT NULL GROUP BY brand ORDER BY 2 DESC LIMIT 10").fetchall()
            stats['top_brands'] = pl.DataFrame(top_b, schema=["brand", "count"])
            # Категории
            cats = self.conn.execute("SELECT category, COUNT(*) FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY 2 DESC").fetchall()
            stats['categories'] = pl.DataFrame(cats, schema=["category", "count"])
        except:
            pass
        return stats

    def build_export_query(self, selected_columns=None, exclude_positions=None):
        # Построение запроса с учетом исключений
        # exclude_positions - строка, где позиции через | для точного совпадения
        base_select = """
        SELECT
            a.artikul_norm, a.brand_norm,
            a.artikul, a.brand,
            a.price, a.quantity,
            p.length, p.width, p.height, p.weight, p.dimensions_str, p.description, p.image_url
        FROM parts_data a
        LEFT JOIN oe_data o ON a.oe_number_norm = o.oe_number_norm
        LEFT JOIN parts_data p ON a.artikul_norm = p.artikul_norm AND a.brand_norm = p.brand_norm
        """
        if selected_columns:
            # Можно расширить
            pass

        where_clauses = []
        if exclude_positions:
            # Разделяем по | и добавляем условие исключения
            positions = [pos.strip() for pos in exclude_positions.split('|')]
            for pos in positions:
                where_clauses.append(f"a.artikul NOT LIKE '%{pos}%'")
        if where_clauses:
            base_select += " WHERE " + " AND ".join(where_clauses)
        return base_select

    def export_to_csv(self, filename, exclude_positions=None):
        query = self.build_export_query(exclude_positions=exclude_positions)
        df = self.conn.execute(query).pl()
        # Обработка числовых колонок
        for col in ['length', 'width', 'height', 'weight', 'price']:
            if col in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(col).is_not_null())
                    .then(pl.col(col).cast(pl.Utf8))
                    .otherwise(pl.lit(""))
                    .alias(col)
                )

        buf = io.StringIO()
        df.write_csv(buf, separator=';')
        csv_text = buf.getvalue()
        with open(filename, 'wb') as f:
            f.write(b'\xef\xbb\xbf')  # BOM
            f.write(csv_text.encode('utf-8'))
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        st.success(f"Экспорт завершен: {filename} ({size_mb:.2f} МБ)")

    def export_to_excel(self, filename, exclude_positions=None):
        query = self.build_export_query(exclude_positions=exclude_positions)
        total_records = self.get_total_records()
        num_files = (total_records + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
        exported_files = []

        for i in range(num_files):
            offset = i * EXCEL_ROW_LIMIT
            q = query + f" LIMIT {EXCEL_ROW_LIMIT} OFFSET {offset}"
            df = self.conn.execute(q).pl()
            for col in ['length', 'width', 'height', 'weight', 'price']:
                if col in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col).is_not_null())
                        .then(pl.col(col).cast(pl.Utf8))
                        .otherwise(pl.lit(""))
                        .alias(col)
                    )
            part_path = Path(filename).with_name(f"{Path(filename).stem}_part_{i+1}.xlsx")
            df.write_excel(str(part_path))
            exported_files.append(part_path)

        # ZIP если более 1 файла
        if len(exported_files) > 1:
            zip_path = Path(filename).with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for f in exported_files:
                    zipf.write(f, f.name)
                    os.remove(f)
            final_path = zip_path
        else:
            final_path = exported_files[0]
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        st.success(f"Экспорт завершен: {final_path} ({size_mb:.2f} МБ)")

    def export_to_parquet(self, filename, exclude_positions=None):
        query = self.build_export_query(exclude_positions=exclude_positions)
        df = self.conn.execute(query).pl()
        for col in ['length', 'width', 'height', 'weight', 'price']:
            if col in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(col).is_not_null())
                    .then(pl.col(col).cast(pl.Utf8))
                    .otherwise(pl.lit(""))
                    .alias(col)
                )
        df.write_parquet(filename)
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        st.success(f"Экспорт в Parquet завершен: {filename} ({size_mb:.2f} МБ)")

    def add_price_data(self, filepath):
        # Загрузка прайса и обновление базы
        df = self.read_and_prepare_file(filepath, 'price')
        if df and not df.is_empty():
            for _, row in df.iterrows():
                artikul_norm = self.normalize_key(pl.Series([row['artikul']]))[0]
                brand_norm = self.normalize_key(pl.Series([row['brand']]))[0]
                quantity = row.get('quantity', 0)
                price = row.get('price', 0.0)
                # Обновление
                self.conn.execute("""
                    UPDATE parts_data SET price = ?, quantity = ? WHERE artikul_norm = ? AND brand_norm = ?
                """, [price, quantity, artikul_norm, brand_norm])
                # Или вставка
                self.conn.execute("""
                    INSERT INTO parts_data (artikul_norm, brand_norm, artikul, brand, price, quantity)
                    SELECT ?, ?, ?, ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM parts_data WHERE artikul_norm=? AND brand_norm=?
                    )
                """, [artikul_norm, brand_norm, row['artikul'], row['brand'], price, quantity, artikul_norm, brand_norm])
            st.success("Прайс успешно добавлен и обновлен.")
        else:
            st.warning("Прайс файл пуст или не удалось его распарсить.")

    def set_markup(self, overall=None, brand_dict=None):
        if overall:
            self.overall_markup = overall
        if brand_dict:
            self.brand_markups.update(brand_dict)
        self.update_prices()

    def update_prices(self):
        # Обновляем цены по текущим наценкам
        cursor = self.conn.execute("SELECT artikul_norm, brand_norm, price FROM parts_data WHERE price IS NOT NULL")
        for artikul_norm, brand_norm, base_price in cursor.fetchall():
            markup = self.brand_markups.get(brand_norm, self.overall_markup)
            new_price = base_price * markup
            self.conn.execute("""
                UPDATE parts_data SET price = ? WHERE artikul_norm = ? AND brand_norm = ?
            """, [new_price, artikul_norm, brand_norm])
        st.success("Цены обновлены с учетом наценок.")

    def partial_search(self, search_text):
        # Поиск по артикулам, брендам или названиям
        query = """
        SELECT a.artikul, a.brand, a.description, a.price, a.quantity
        FROM parts_data a
        LEFT JOIN oe_data o ON a.oe_number_norm = o.oe_number_norm
        WHERE a.artikul LIKE ? OR a.brand LIKE ? OR o.name LIKE ?
        """
        pattern = f"%{search_text}%"
        df = self.conn.execute(query, [pattern, pattern, pattern]).pl()
        if df.shape[0] == 0:
            st.info("Ничего не найдено.")
        else:
            st.dataframe(df.to_pandas())

# ================== Основное приложение ==================

def main():
    st.title("🚗 AutoParts Catalog - Расширенная система")
    catalog = AutoPartsCatalog()

    st.sidebar.title("🧭 Меню")
    option = st.sidebar.radio("Действия", ["Загрузка данных", "Обновление прайса", "Настройка цен", "Экспорт", "Статистика", "Поиск"])

    if option == "Загрузка данных":
        st.header("📥 Загрузка исходных файлов")
        cols = st.columns(2)
        with cols[0]:
            oe_file = st.file_uploader("OE (Базовые данные)", type=['xlsx'])
            cross_file = st.file_uploader("Кроссы", type=['xlsx'])
            barcode_file = st.file_uploader("Штрихкоды", type=['xlsx'])
        with cols[1]:
            dimensions_file = st.file_uploader("Весогабариты", type=['xlsx'])
            images_file = st.file_uploader("Изображения", type=['xlsx'])
            price_file = st.file_uploader("Прайс (новый)", type=['xlsx'])

        if st.button("Обработать и загрузить"):
            file_paths = {}
            for name, file in [('oe', oe_file), ('cross', cross_file), ('barcode', barcode_file),
                               ('dimensions', dimensions_file), ('images', images_file), ('price', price_file)]:
                if file:
                    filename = f"{name}_{int(time.time())}_{file.name}"
                    path = catalog.data_dir / filename
                    with open(path, 'wb') as f:
                        f.write(file.getvalue())
                    file_paths[name] = str(path)

            # Загрузка прайса отдельно
            if 'price' in file_paths:
                catalog.add_price_data(file_paths['price'])

            # Объединение всех файлов
            if file_paths:
                catalog.merge_all_data_parallel(file_paths)

    elif option == "Обновление прайса":
        st.header("🔧 Обновить цены")
        overall_markup = st.number_input("Общая наценка (%)", min_value=0.0, max_value=100.0, value=20.0)
        brand_markups_input = st.text_input("Наценки по брендам (через запятую: бренд=коэффициент)", "")
        brand_dict = {}
        if brand_markups_input:
            pairs = [p.strip() for p in brand_markups_input.split(',')]
            for pair in pairs:
                if '=' in pair:
                    brand, coeff = pair.split('=')
                    try:
                        brand_dict[brand.strip()] = float(coeff.strip()) / 100.0
                    except:
                        pass
        catalog.set_markup(overall=1 + overall_markup/100.0, brand_dict=brand_dict)

    elif option == "Настройка цен":
        st.header("⚙️ Настройка цен")
        overall_markup = st.number_input("Общая наценка (%)", min_value=0.0, max_value=100.0, value=20.0)
        brand_markups_input = st.text_input("Наценки по брендам (через запятую: бренд=коэффициент)", "")
        brand_dict = {}
        if brand_markups_input:
            pairs = [p.strip() for p in brand_markups_input.split(',')]
            for pair in pairs:
                if '=' in pair:
                    brand, coeff = pair.split('=')
                    try:
                        brand_dict[brand.strip()] = float(coeff.strip()) / 100.0
                    except:
                        pass
        catalog.set_markup(overall=1 + overall_markup/100.0, brand_dict=brand_dict)

    elif option == "Экспорт":
        catalog.show_export_interface()

    elif option == "Статистика":
        with st.spinner("Сбор статистики..."):
            stats = catalog.get_statistics()
        st.write(f"Всего артикулов: {stats.get('total_parts', 0):,}")
        st.write(f"Всего OE: {stats.get('total_oe', 0):,}")
        st.write(f"Брендов: {stats.get('total_brands', 0):,}")
        if not stats['top_brands'].is_empty():
            st.subheader("Топ брендов")
            st.dataframe(stats['top_brands'].to_pandas())
        if not stats['categories'].is_empty():
            st.subheader("Распределение по категориям")
            st.bar_chart(stats['categories'].to_pandas().set_index('category'))

    elif option == "Поиск":
        search_text = st.text_input("Введите название, артикул или бренд для поиска")
        if st.button("Искать"):
            catalog.partial_search(search_text)

if __name__ == "__main__":
    main()
