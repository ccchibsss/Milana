import polars as pl
import duckdb
import streamlit as st
import os
import time
import io
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import warnings
warnings.filterwarnings('ignore')

# Константы
EXCEL_ROW_LIMIT = 1_000_000

class AutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        self.setup_database()

        # Таблицы для цен рекомендаций и цен по артикулам
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS price_recommendations (
                artikul_norm VARCHAR PRIMARY KEY,
                recommended_price DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS part_prices (
                artikul_norm VARCHAR PRIMARY KEY,
                brand_norm VARCHAR,
                price DOUBLE
            )
        """)

        # Настройки
        self.global_markup = 1.2
        self.brand_markups: Dict[str, float] = {}
        self.exclusions: List[str] = []
        self.exclusions_partial: List[str] = []

        # Для экспорта
        self.export_columns: List[str] = []

        # Визуальные настройки
        st.set_page_config(page_title="AutoParts Catalog 10M+", layout="wide", page_icon="🚗")

    def setup_database(self):
        # Создаём таблицы, если отсутствуют
        # Основные таблицы
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

    # ===================== Загрузка цен рекомендаций =====================
    def load_price_recommendations(self, file_path):
        df = pl.read_excel(file_path)
        df = df.select([
            pl.col("артикул").alias("artikul"),
            pl.col("цена").alias("recommended_price")
        ]).drop_nulls()
        df = df.with_columns(
            pl.col("artikul").str.replace_all("'", "").str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "").str.strip_chars().str.to_lowercase()
        )
        for row in df.iter_rows():
            self.conn.execute("""
                INSERT INTO price_recommendations (artikul_norm, recommended_price)
                VALUES (?, ?)
                ON CONFLICT (artikul_norm) DO UPDATE SET recommended_price=excluded.recommended_price
            """, [row[0], row[1]])

    def get_price_for_artikul(self, artikul_norm):
        res = self.conn.execute("SELECT recommended_price FROM price_recommendations WHERE artikul_norm = ?", [artikul_norm]).fetchone()
        return res[0] if res else None

    # ===================== Загрузка прайса с артикулами =====================
    def load_price_list(self, file_path):
        df = pl.read_excel(file_path)
        df = df.select([
            pl.col("артикул").alias("artikul"),
            pl.col("количество").alias("quantity"),
            pl.col("бренд").alias("brand"),
            pl.col("цена").alias("price")
        ]).drop_nulls()

        # Очистка
        df = df.with_columns(
            pl.col("artikul").str.replace_all("'", "").str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "").str.strip_chars().str.to_lowercase(),
            pl.col("brand").str.replace_all("'", "").str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "").str.strip_chars().str.to_lowercase()
        )

        # Обновление таблицы цен
        for row in df.iter_rows():
            artikul_norm = row[0]
            brand_norm = row[2]
            price = row[3]
            self.conn.execute("""
                INSERT INTO part_prices (artikul_norm, brand_norm, price)
                VALUES (?, ?, ?)
                ON CONFLICT (artikul_norm) DO UPDATE SET price=excluded.price
            """, [artikul_norm, brand_norm, price])

    # ===================== Настройка наценки =====================
    def set_global_markup(self, markup):
        self.global_markup = markup

    def set_brand_markup(self, brand, markup):
        self.brand_markups[brand.lower()] = markup

    def get_markup_for_brand(self, brand):
        return self.brand_markups.get(brand.lower(), self.global_markup)

    # ===================== Настройки исключений =====================
    def load_exclusions(self, exact_list, partial_list):
        self.exclusions = [s.lower() for s in exact_list]
        self.exclusions_partial = [s.lower() for s in partial_list]

    def check_exclusions(self, name):
        name_lower = name.lower()
        for excl in self.exclusions:
            if excl == name_lower:
                return True
        for excl in self.exclusions_partial:
            if excl in name_lower:
                return True
        return False

    # ===================== Получение финальной цены =====================
    def get_final_price(self, artikul_norm, brand, base_price):
        if self.check_exclusions(brand):
            return None
        brand_markup = self.get_markup_for_brand(brand)
        total_markup = self.global_markup * brand_markup
        final_price = base_price * total_markup
        return final_price

    # ===================== Основной метод загрузки и обработки =====================
    def merge_all_data_parallel(self, file_paths: Dict[str, str]) -> Dict:
        start_time = time.time()
        stats = {}
        dataframes = {}

        # Чтение файлов параллельно
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.read_and_prepare_file, path, ftype) for ftype, path in file_paths.items()]
            for future in as_completed(futures):
                # Получение типа файла по порядку
                index = list(file_paths.keys()).index(list(future.result().values())[0])
                ftype = list(file_paths.keys())[index]
                df = future.result()
                if not df.is_empty():
                    dataframes[ftype] = df

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
            return self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]
        except:
            return 0

    def create_indexes(self):
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_oe ON oe_data(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_parts ON parts_data(artikul_norm, brand_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross ON cross_references(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross_art ON cross_references(artikul_norm, brand_norm)")

    def read_and_prepare_file(self, file_path, file_type):
        df = pl.read_excel(file_path)
        # Можно тут добавить обработку по типу файла
        return df

    def process_and_load_data(self, dataframes):
        # Обработка и вставка данных
        # Для примера вставляю только базовые операции
        # Здесь можно расширять обработку
        pass

    # ===================== Построение SQL-запроса для экспорта =====================
    def build_export_query(self, selected_columns=None):
        # Встроенный текст описания
        description_text = """
        Состояние товара: новый (в упаковке).
        Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
        Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

        В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

        Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

        Выбирайте только лучшее — надежность и качество от ведущих производителей."""
        # Оформляем CTE с текстом
        query = f"""
        WITH DescriptionText AS (
            SELECT chr(10) || chr(10) || $${description_text}$$ AS text
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
            -- Аналоги логика
            -- Можно расширить
        )
        SELECT
        """

        columns_map = [
            ("Артикул бренда", 'p.artikul AS "Артикул бренда"'),
            ("Бренд", 'p.brand AS "Бренд"'),
            ("Наименование", 'COALESCE(pd.representative_name, p.description) AS "Наименование"'),
            ("Применимость", 'COALESCE(pd.representative_applicability, "") AS "Применимость"'),
            ("Описание", 'CONCAT(COALESCE(p.description, ""), dt.text) AS "Описание"'),
            ("Категория товара", 'COALESCE(pd.representative_category, "") AS "Категория товара"'),
            ("Кратность", 'p.multiplicity AS "Кратность"'),
            ("Длинна", 'COALESCE(p.length, 0) AS "Длинна"'),
            ("Ширина", 'COALESCE(p.width, 0) AS "Ширина"'),
            ("Высота", 'COALESCE(p.height, 0) AS "Высота"'),
            ("Вес", 'COALESCE(p.weight, 0) AS "Вес"'),
            ("Длинна/Ширина/Высота", 'COALESCE(p.dimensions_str, "") AS "Длинна/Ширина/Высота"'),
            ("OE номер", 'pd.oe_list AS "OE номер"'),
            ("аналоги", 'p.analog_list AS "аналоги"'),
            ("Ссылка на изображение", 'p.image_url AS "Ссылка на изображение"')
        ]

        select_exprs = []
        if selected_columns:
            for col in selected_columns:
                for name, expr in columns_map:
                    if col == name:
                        select_exprs.append(expr)
                        break
        else:
            select_exprs = [expr for _, expr in columns_map]

        query += "\n".join(select_exprs) + "\nFROM parts_data p\n"
        query += "LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm\n"
        query += "LEFT JOIN DescriptionText dt ON 1=1\n"
        query += "WHERE 1=1\n"
        query += "ORDER BY p.brand, p.artikul\n"
        return query

    def export_to_csv(self, output_path, selected_columns=None):
        try:
            query = self.build_export_query(selected_columns)
            df = self.conn.execute(query).pl()

            # Обработка числовых колонок для CSV
            for col in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if col in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col).is_not_null()).then(pl.col(col).cast(pl.Utf8)).otherwise("").alias(col)
                    )

            buf = io.StringIO()
            df.write_csv(buf, separator=';')
            csv_bytes = buf.getvalue().encode('utf-8-sig')

            with open(output_path, 'wb') as f:
                f.write(csv_bytes)
            st.success(f"Экспорт завершен: {output_path}")
            return True
        except Exception as e:
            st.error(f"Ошибка экспорта: {e}")
            return False

    # Аналогичные функции для excel, parquet, можно расширить
    def export_to_excel(self, output_path, selected_columns=None):
        # Реализуем с разделением по лимиту
        pass

    def export_to_parquet(self, output_path, selected_columns=None):
        # Реализуем
        pass

    # ===================== Визуальный интерфейс =====================
    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total_records = self.conn.execute("SELECT COUNT(DISTINCT (artikul_norm, brand_norm)) FROM parts_data").fetchone()[0]
        st.info(f"Всего записей для экспорта: {total_records:,}")
        if total_records == 0:
            st.warning("Нет данных для экспорта.")
            return

        # Выбор колонок и порядка
        available_columns = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение"
        ]
        selected_columns = st.multiselect(
            "Выберите столбцы для экспорта (порядок важен)", options=available_columns, default=available_columns
        )
        self.export_columns = selected_columns

        export_format = st.radio("Выберите формат экспорта:", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)

        if export_format == "CSV":
            if st.button("🚀 Экспорт в CSV"):
                output_path = self.data_dir / "auto_parts_report.csv"
                self.export_to_csv(str(output_path), self.export_columns)
                with open(output_path, "rb") as f:
                    st.download_button("📥 Скачать CSV", f, "auto_parts_report.csv", "text/csv")
        elif export_format == "Excel (.xlsx)":
            if st.button("📊 Экспорт в Excel"):
                output_path = self.data_dir / "auto_parts_report.xlsx"
                # Реализуйте экспорт
                pass
        elif export_format == "Parquet":
            if st.button("⚡️ Экспорт в Parquet"):
                output_path = self.data_dir / "auto_parts_report.parquet"
                # Реализуйте экспорт
                pass

# ===================== Основной запуск =====================
def main():
    catalog = AutoPartsCatalog()
    st.title("🚗 AutoParts Catalog - Профессиональная система для 10+ млн записей")
    st.markdown("---")
    menu = st.sidebar.radio("Навигация", ["Загрузка данных", "Экспорт", "Статистика", "Управление"])

    if menu == "Загрузка данных":
        st.header("📥 Загрузка и обработка данных")
        files_ui = {}
        files_ui['oe'] = st.file_uploader("Основные данные (OE)", type=['xlsx', 'xls'])
        files_ui['cross'] = st.file_uploader("Кроссы (OE -> Артикул)", type=['xlsx', 'xls'])
        files_ui['barcode'] = st.file_uploader("Штрихкоды и кратность", type=['xlsx', 'xls'])
        files_ui['dimensions'] = st.file_uploader("Весогабаритные данные", type=['xlsx', 'xls'])
        files_ui['images'] = st.file_uploader("Изображения", type=['xlsx', 'xls'])

        if st.button("🚀 Начать обработку"):
            file_paths = {}
            for key, uploaded in files_ui.items():
                if uploaded:
                    path = catalog.data_dir / f"{key}_{int(time.time())}_{uploaded.name}"
                    with open(path, "wb") as f:
                        f.write(uploaded.getvalue())
                    file_paths[key] = str(path)
            if file_paths:
                catalog.merge_all_data_parallel(file_paths)
                st.success("Обработка завершена.")
            else:
                st.warning("Загрузите хотя бы один файл.")

    elif menu == "Экспорт":
        catalog.show_export_interface()

    elif menu == "Статистика":
        st.header("📈 Статистика")
        # Реализуйте отображение статистики
        pass

    elif menu == "Управление":
        st.header("🗑️ Управление данными")
        # Реализуйте удаление по бренду или артикулу
        pass

if __name__ == "__main__":
    main()
