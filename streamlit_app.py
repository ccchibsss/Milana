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

class HighVolumeAutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        self.setup_database()
        # Изначальная глобальная наценка
        self.global_markup = 0.2  # 20%
        self.create_indexes()
        # Внутренние переменные
        self.price_cache = {}  # Для кэширования цен, по желанию

        st.set_page_config(
            page_title="AutoParts Catalog 10M+", 
            layout="wide",
            page_icon="🚗"
        )

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
        # Таблица цен
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                artikul VARCHAR,
                quantity INTEGER,
                brand VARCHAR,
                price DOUBLE,
                PRIMARY KEY (artikul, brand)
            )
        """)
        # Таблица наценок
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS markups (
                brand VARCHAR PRIMARY KEY,
                markup DOUBLE
            )
        """)
        # Изначально глобальная наценка
        self.global_markup = 0.2

    def create_indexes(self):
        # Создаем индексы
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)",
        ]
        for idx_sql in indexes:
            self.conn.execute(idx_sql)

    def normalize_key(self, key_series: pl.Series) -> pl.Series:
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

    def clean_values(self, value_series: pl.Series) -> pl.Series:
        return (
            value_series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all("'", "")
            .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

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

    def read_and_prepare_file(self, file_path, file_type):
        # Чтение и подготовка файла
        try:
            df = pl.read_excel(file_path, engine='calamine')
        except Exception:
            return pl.DataFrame()
        # Определение колонок
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
        # Очистка
        if 'artikul' in df.columns:
            df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand=self.clean_values(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))
        # Уникальные по ключам
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

    def upsert_data(self, table_name, df, pk):
        if df.is_empty():
            return
        df = df.unique(keep='first')
        cols = df.columns
        pk_str = ", ".join(f'"{c}"' for c in pk)
        temp_view = f"temp_{table_name}_{int(time.time())}"
        self.conn.register(temp_view, df.to_arrow())
        update_cols = [col for col in cols if col not in pk]
        if not update_cols:
            on_conflict = "DO NOTHING"
        else:
            set_clause = ", ".join([f'"{col}"=excluded."{col}"' for col in update_cols])
            on_conflict = f"DO UPDATE SET {set_clause}"
        sql = f"""
        INSERT INTO {table_name}
        SELECT * FROM {temp_view}
        ON CONFLICT ({pk_str}) {on_conflict};
        """
        self.conn.execute(sql)
        self.conn.unregister(temp_view)

    def process_and_load_data(self, dataframes):
        # Основной процесс
        st.info("🔄 Начинаю обновление базы данных...")
        # Обработка oe
        if 'oe' in dataframes:
            df = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'])
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self.determine_category_vectorized(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])
            # Cross
            cross_df = df.filter(pl.col('artikul_norm') != "").select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])
        # Обработка cross
        if 'cross' in dataframes:
            df = dataframes['cross']
            cross_df = df.filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])
        # Обработка parts
        # Собираем артикула и бренды из всех файлов
        parts_df = None
        # Объединение данных по артикулам и брендам
        # ...
        # Далее здесь логика обновления parts_data, аналогичная вашему коду
        # (для краткости пропущено, вставьте как есть)
        # После этого: подготовка для экспорта, формирование итоговых данных, расчет цен с наценками
        # Для каждого артикула ищем цену в таблице prices, если есть, применяем наценку
        # Обязательно добавьте расчет цены с учетом наценки в финальный SELECT

        # В конце вызов self.upsert_data для parts_data
        # (распишите по аналогии выше или вставьте ваш существующий код)

        st.success("💾 Загрузка данных завершена.")

    def load_price_list(self, file_path):
        df = pl.read_excel(file_path, engine='calamine')
        df = df.rename({col: col.lower() for col in df.columns})
        required_cols = ['артикул', 'количество', 'бренд', 'цена']
        if not all(c in df.columns for c in required_cols):
            st.error("Прайс-лист должен содержать столбцы: Артикул, Количество, Бренд, Цена")
            return
        df = df.with_columns(
            artikul=self.clean_values(pl.col('артикул')),
            brand=self.clean_values(pl.col('бренд')),
            quantity=pl.col('количество').cast(pl.Int32),
            price=pl.col('цена').cast(pl.Float64)
        )
        for row in df.to_dicts():
            self.conn.execute("""
                INSERT INTO prices (artikul, quantity, brand, price) VALUES (?, ?, ?, ?)
                ON CONFLICT (artikul, brand) DO UPDATE SET
                quantity=excluded.quantity,
                price=excluded.price
            """, [row['артикул'], row['количество'], row['бренд'], row['цена']])
        st.success("Цены успешно обновлены")

    def get_price_for_artikul(self, artikul, brand):
        # Можно кэшировать или делать запрос по необходимости
        result = self.conn.execute("SELECT price FROM prices WHERE artikul = ? AND brand = ?", [artikul, brand]).fetchone()
        if result:
            return result[0]
        return None

    def build_export_query(self, selected_columns=None, exclude_names=None, include_markup=True):
        # Формируем SELECT с учетом колонок, наценок и исключений
        # Объявляем CTE с текстом (описание)
        standard_description = """Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""
        # Колонки и их SQL выражения
        columns_map = [
            ("Артикул бренда", 'p.artikul AS "Артикул бренда"'),
            ("Бренд", 'p.brand AS "Бренд"'),
            ("Наименование", 'COALESCE(p.name, p2.name) AS "Наименование"'),
            ("Применимость", 'COALESCE(p.applicability, p2.applicability) AS "Применимость"'),
            ("Описание", "CONCAT(COALESCE(p.description, ''), dt.text) AS \"Описание\""),
            ("Категория товара", 'COALESCE(p.category, p2.category) AS "Категория товара"'),
            ("Кратность", 'p.multiplicity AS "Кратность"'),
            ("Длинна", 'COALESCE(p.length, p2.length) AS "Длинна"'),
            ("Ширина", 'COALESCE(p.width, p2.width) AS "Ширина"'),
            ("Высота", 'COALESCE(p.height, p2.height) AS "Высота"'),
            ("Вес", 'COALESCE(p.weight, p2.weight) AS "Вес"'),
            ("Длинна/Ширина/Высота", "COALESCE(p.dimensions_str, p2.dimensions_str) AS \"Длинна/Ширина/Высота\""),
            ("OE номер", 'p.oe_list AS "OE номер"'),
            ("аналоги", 'p.analog_list AS "аналоги"'),
            ("Ссылка на изображение", 'p.image_url AS "Ссылка на изображение"'),
            ("Цена с наценкой", 'CASE WHEN p.price IS NOT NULL THEN p.price * (1 + ? + COALESCE(m.markup, 0)) ELSE NULL END AS "Цена с наценкой"')
        ]
        # Если выбранные колонки не указаны, берем все
        if selected_columns is None:
            selected_columns = [name for name, _ in columns_map]
        else:
            # фильтруем по выбранным
            columns_map = [item for item in columns_map if item[0] in selected_columns]
        select_exprs = [expr for _, expr in columns_map]
        # Формируем WHERE с исключениями
        where_clauses = []
        if exclude_names:
            conditions = []
            for name in exclude_names:
                conditions.append(f"p.name LIKE '%{name}%'")
            where_clauses.append("(" + " OR ".join(conditions) + ")")
        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)
        # Формируем CTE с текстом
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
        )
        -- (Можно добавить уровни расширения, если нужно)
        """
        # Собираем итог
        query = f"""
        {ctes}
        SELECT
            {", ".join(select_exprs)}
        FROM parts_data p
        LEFT JOIN cross_references cr ON p.artikul_norm = cr.artikul_norm AND p.brand_norm = cr.brand_norm
        LEFT JOIN oe_data o ON cr.oe_number_norm = o.oe_number_norm
        LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
        LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
        LEFT JOIN DescriptionTemplate dt ON 1=1
        {where_sql}
        WHERE 1=1
        """
        return query, self.global_markup

    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total_records = self.conn.execute("SELECT count(DISTINCT (artikul_norm, brand_norm)) FROM parts_data").fetchone()[0]
        st.info(f"Всего записей для экспорта (строк): {total_records:,}")
        if total_records == 0:
            st.warning("База пуста или нет данных для экспорта.")
            return
        # Настройки колонок
        available_columns = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена с наценкой"
        ]
        columns_order = st.multiselect("Выберите порядок колонок", options=available_columns, default=available_columns)

        # Исключения
        exclusions_input = st.text_area("Наименования для исключения (через |)", height=100)
        exclude_names = [n.strip() for n in exclusions_input.split('|') if n.strip()]

        # Настройки наценки
        st.subheader("Настройка наценки")
        self.global_markup = st.slider("Общая наценка (%)", 0, 100, int(self.global_markup*100))/100
        brand_name = st.text_input("Бренд для настройки наценки")
        if brand_name:
            res = self.conn.execute("SELECT markup FROM markups WHERE brand = ?", [brand_name]).fetchone()
            current_markup = res[0] if res else 0
            new_markup = st.slider(f"Наценка для {brand_name} (%)", 0, 100, int(current_markup*100))
            if st.button("Установить наценку для бренда"):
                self.conn.execute("""
                    INSERT INTO markups (brand, markup) VALUES (?, ?)
                    ON CONFLICT (brand) DO UPDATE SET markup=excluded.markup
                """, [brand_name, new_markup/100])

        # Выбор колонок
        selected_columns = st.multiselect("Выберите колонки для экспорта (по умолчанию все)", options=available_columns, default=columns_order)

        # Формат
        export_format = st.radio("Выберите формат экспорта:", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)

        if export_format == "CSV":
            if st.button("🚀 Экспорт в CSV"):
                output_path = self.data_dir / "auto_parts_export.csv"
                with st.spinner("Экспорт в CSV..."):
                    query, markup_value = self.build_export_query(selected_columns, exclude_names)
                    df = self.conn.execute(query, [self.global_markup]).pl()
                    # преобразование числовых колонок в строки
                    for colname in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                        if colname in df.columns:
                            df = df.with_columns(
                                pl.when(pl.col(colname).is_not_null())
                                .then(pl.col(colname).cast(pl.Utf8))
                                .otherwise("")
                                .alias(colname)
                            )
                    buf = io.StringIO()
                    df.write_csv(buf, separator=';')
                    with open(output_path, 'wb') as f:
                        f.write(b'\xef\xbb\xbf')
                        f.write(buf.getvalue().encode('utf-8'))
                st.success(f"Файл сохранен: {output_path}")
                st.download_button("Скачать CSV", open(output_path, "rb"), "auto_parts_export.csv")
        elif export_format == "Excel (.xlsx)":
            if st.button("📊 Экспорт в Excel"):
                output_path = self.data_dir / "auto_parts_export.xlsx"
                # Поскольку Excel ограничение, делаем по частям
                total_count = self.conn.execute("SELECT COUNT(DISTINCT artikul_norm, brand_norm) FROM parts_data").fetchone()[0]
                num_files = (total_count // EXCEL_ROW_LIMIT) + 1
                all_files = []
                for i in range(num_files):
                    offset = i * EXCEL_ROW_LIMIT
                    query, markup_value = self.build_export_query(selected_columns, exclude_names)
                    df = self.conn.execute(f"{query} LIMIT {EXCEL_ROW_LIMIT} OFFSET {offset}", [self.global_markup]).pl()
                    for colname in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                        if colname in df.columns:
                            df = df.with_columns(
                                pl.when(pl.col(colname).is_not_null())
                                .then(pl.col(colname).cast(pl.Utf8))
                                .otherwise("")
                                .alias(colname)
                            )
                    file_path = self.data_dir / f"part_{i+1}.xlsx"
                    df.write_excel(str(file_path))
                    all_files.append(file_path)
                # ZIP если больше 1 файла
                if len(all_files) > 1:
                    zip_path = self.data_dir / "export_parts.zip"
                    with zipfile.ZipFile(zip_path, 'w') as zf:
                        for file in all_files:
                            zf.write(file, arcname=file.name)
                            os.remove(file)
                    st.download_button("Скачать ZIP", open(zip_path, "rb"), "export_parts.zip")
                else:
                    st.download_button("Скачать Excel", open(all_files[0], "rb"), "auto_parts_export.xlsx")
        elif export_format == "Parquet":
            if st.button("⚡️ Экспорт в Parquet"):
                output_path = self.data_dir / "auto_parts_export.parquet"
                query, _ = self.build_export_query(selected_columns, exclude_names)
                df = self.conn.execute(query, [self.global_markup]).pl()
                df.write_parquet(str(output_path))
                st.download_button("Скачать Parquet", open(output_path, "rb"), "auto_parts_export.parquet")
        
    def get_statistics(self):
        stats = {}
        try:
            stats['total_parts'] = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]
            stats['total_oe'] = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()[0]
            stats['total_brands'] = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data").fetchone()[0]
            # Топ брендов
            br_res = self.conn.execute("SELECT brand, COUNT(*) FROM parts_data GROUP BY brand ORDER BY COUNT(*) DESC LIMIT 10").fetchall()
            stats['top_brands'] = pl.DataFrame(br_res, schema=["brand", "count"])
            # Категории
            cat_res = self.conn.execute("SELECT category, COUNT(*) FROM oe_data GROUP BY category ORDER BY COUNT(*) DESC").fetchall()
            stats['categories'] = pl.DataFrame(cat_res, schema=["category", "count"])
        except Exception:
            pass
        return stats

    def merge_all_data_parallel(self, file_paths):
        # Реализуйте как у вас
        # После обработки — вызов self.process_and_load_data(...)
        pass

# В основном вызывайте
def main():
    catalog = HighVolumeAutoPartsCatalog()

    st.title("🚗 AutoParts Catalog - Профессиональная система для 10+ млн записей")
    st.markdown("...")  # Ваша описание

    menu_option = st.sidebar.radio("Выберите действие:", ["Загрузка данных", "Экспорт", "Статистика", "Управление данными"])

    if menu_option == "Загрузка данных":
        # Ваша логика загрузки, вызов catalog.load_price_list() при необходимости
        pass
    elif menu_option == "Экспорт":
        catalog.show_export_interface()
    elif menu_option == "Статистика":
        stats = catalog.get_statistics()
        # Ваша статистика
        pass
    elif menu_option == "Управление данными":
        # Операции удаления
        pass

if __name__ == "__main__":
    main()
