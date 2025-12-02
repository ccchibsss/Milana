import platform
import sys
import polars as pl
import duckdb
import streamlit as st
import os
import time
import io
import zipfile
from pathlib import Path
from typing import Dict, List

# Проверка архитектуры Python (требуется 64-bit)
if platform.architecture()[0] != '64bit':
    error_msg = """
    ❌ ОШИБКА: Обнаружена 32-bit версия Python!
    Это приложение требует 64-bit версию Python, так как библиотеки 
    pyarrow, polars и duckdb не поддерживают 32-bit архитектуру на Windows.
    Решение:
    1. Скачайте и установите 64-bit Python с https://www.python.org/downloads/
    2. Переустановите зависимости: pip install -r requirements.txt
    3. Запустите приложение снова
    Текущая архитектура: {}
    """.format(platform.architecture()[0])
    print(error_msg)
    sys.exit(1)

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

        # Настройки наценки
        self.overall_markup = 0.0  # в процентах
        self.brand_markups: Dict[str, float] = {}  # по брендам

        st.set_page_config(
            page_title="AutoParts Catalog 10M+", 
            layout="wide",
            page_icon="🚗"
        )

    def setup_database(self):
        # Таблица с артикулом и прочими данными
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS oe_data (
                oe_number_norm VARCHAR PRIMARY KEY,
                oe_number VARCHAR,
                name VARCHAR,
                applicability VARCHAR,
                category VARCHAR
            )
        """)
        # Таблица с артикулами
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
        # Таблица с кроссами
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_references (
                oe_number_norm VARCHAR,
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                PRIMARY KEY (oe_number_norm, artikul_norm, brand_norm)
            )
        """)
        # Таблица с рекомендованными ценами
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS recommended_prices (
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                recommended_price DOUBLE,
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)

    def create_indexes(self):
        # Создание индексов для ускорения поиска
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_recommended_prices ON recommended_prices(artikul_norm, brand_norm)")

    @staticmethod
    def normalize_key(key_series: pl.Series) -> pl.Series:
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

    def upsert_recommended_prices(self, df: pl.DataFrame):
        # Обновление или вставка цен
        df = df.with_columns(
            artikul_norm=self.normalize_key(pl.col('artikul')),
            brand_norm=self.normalize_key(pl.col('brand')),
            recommended_price=pl.col('recommended_price').cast(pl.Float64)
        )
        # UPSERT
        for row in df.iter_rows():
            self.conn.execute("""
                INSERT INTO recommended_prices (artikul_norm, brand_norm, recommended_price)
                VALUES (?, ?, ?)
                ON CONFLICT (artikul_norm, brand_norm) DO UPDATE SET recommended_price=excluded.recommended_price
            """, [row[1], row[2], row[3]])

    def load_recommended_prices(self, file_path: str):
        df = pl.read_excel(file_path, engine='calamine')
        # Предполагаемый формат: артикул, бренд, цена
        # Проверка и переименование колонок
        # Можно добавить проверки, чтобы убедиться в наличии колонок
        df = df.rename({df.columns[0]: 'artikul', df.columns[1]: 'brand', df.columns[2]: 'recommended_price'})
        self.upsert_recommended_prices(df)

    def set_markup(self, overall: float, brand_markups: Dict[str, float]):
        self.overall_markup = overall
        self.brand_markups = brand_markups

    def get_brand_markup(self, brand_norm: str) -> float:
        return self.brand_markups.get(brand_norm, self.overall_markup)

    def build_export_query(self, selected_columns: List[str] | None, exclude_terms: str = "") -> str:
        # Вводим параметры
        # Создаем CTE с текстом описания
        standard_description = """Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""
        # Формируем условия исключений
        exclude_sql = ""
        params = []

        if exclude_terms:
            terms = [term.strip() for term in exclude_terms.split('|') if term.strip()]
            # точное совпадение
            for term in terms:
                exclude_sql += " AND r.\"Наименование\" NOT IN ({})".format(', '.join(['?']*len(terms)))
                params.extend(terms)
            # частичное совпадение
            for term in terms:
                exclude_sql += " AND r.\"Наименование\" NOT LIKE ?"
                params.append(f"%{term}%")

        # Формируем SELECT выражения в зависимости от выбранных колонок
        columns_map = [
            ("Артикул бренда", 'r.artikul AS "Артикул бренда"'),
            ("Бренд", 'r.brand AS "Бренд"'),
            ("Наименование", 'COALESCE(r.representative_name, r.analog_representative_name) AS "Наименование"'),
            ("Применимость", 'COALESCE(r.representative_applicability, r.analog_representative_applicability) AS "Применимость"'),
            ("Описание", "CONCAT(COALESCE(r.description, ''), dt.text) AS \"Описание\""),
            ("Категория товара", 'COALESCE(r.representative_category, r.analog_representative_category) AS "Категория товара"'),
            ("Кратность", 'r.multiplicity AS "Кратность"'),
            ("Длинна", 'COALESCE(r.length, r.analog_length) AS "Длинна"'),
            ("Ширина", 'COALESCE(r.width, r.analog_width) AS "Ширина"'),
            ("Высота", 'COALESCE(r.height, r.analog_height) AS "Высота"'),
            ("Вес", 'COALESCE(r.weight, r.analog_weight) AS "Вес"'),
            ("Длинна/Ширина/Высота", "COALESCE(CASE WHEN r.dimensions_str IS NULL OR r.dimensions_str = '' OR UPPER(TRIM(r.dimensions_str)) = 'XX' THEN NULL ELSE r.dimensions_str END, r.analog_dimensions_str) AS \"Длинна/Ширина/Высота\""),
            ("OE номер", 'r.oe_list AS "OE номер"'),
            ("аналоги", 'r.analog_list AS "аналоги"'),
            ("Ссылка на изображение", 'r.image_url AS "Ссылка на изображение"'),
            ("Финальная цена", 'r.final_price AS "Цена с учетом наценки"')
        ]

        if not selected_columns:
            selected_exprs = [expr for _, expr in columns_map]
        else:
            selected_exprs = [expr for name, expr in columns_map if name in selected_columns]
            if not selected_exprs:
                selected_exprs = [expr for _, expr in columns_map]

        # CTE с текстом
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
                ROW_NUMBER() OVER(PARTITION BY p.artikul_norm, p.brand_norm ORDER BY pd.representative_name DESC NULLS LAST, pd.oe_list DESC NULLS LAST) as rn,
                -- добавляем расчет финальной цены
                -- присоединяем таблицу цен
                -- В основном запросе сделаем JOIN с таблицей цен и расчет
                -- В этом месте оставим placeholder
                0 AS final_price
            FROM parts_data p
            LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
            LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
            LEFT JOIN AggregatedAnalogData p_analog ON p.artikul_norm = p_analog.artikul_norm AND p.brand_norm = p_analog.brand_norm
        )
        """

        # В основном запросе сделаем JOIN с ценами
        # В конце добавим расчет финальной цены
        query = f"""
        {ctes}
        SELECT
            {', '.join(selected_exprs)}
        FROM RankedData r
        LEFT JOIN recommended_prices rp ON r.artikul_norm = rp.artikul_norm AND r.brand_norm = rp.brand_norm
        LEFT JOIN parts_data p ON r.artikul_norm = p.artikul_norm AND r.brand_norm = p.brand_norm
        LEFT JOIN (
            SELECT *,
            -- расчет финальной цены с учетом наценки
            COALESCE(rp.recommended_price, 0) * (1 + {self.overall_markup / 100}) * (1 + self.get_brand_markup('r.brand')/100) AS final_price
            FROM parts_data p2
            LEFT JOIN recommended_prices rp ON p2.artikul_norm = rp.artikul_norm AND p2.brand_norm = rp.brand_norm
        ) p2 ON r.artikul_norm = p2.artikul_norm AND r.brand_norm = p2.brand_norm
        WHERE r.rn = 1
        {exclude_sql}
        ORDER BY r.brand, r.artikul
        """
        return query, params

    def export_to_csv_optimized(self, output_path: str, selected_columns: List[str] | None = None, exclude_terms: str = "") -> bool:
        total_records = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📤 Экспорт {total_records:,} записей в CSV...")
        try:
            query, params = self.build_export_query(selected_columns, exclude_terms)
            df = self.conn.execute(query, params).pl()

            # Преобразование числовых колонок
            for col_name in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if col_name in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col_name).is_not_null())
                        .then(pl.col(col_name).cast(pl.Utf8))
                        .otherwise(pl.lit(""))
                        .alias(col_name)
                    )

            buf = io.StringIO()
            df.write_csv(buf, separator=';')
            csv_text = buf.getvalue()

            with open(output_path, 'wb') as f:
                f.write(b'\xef\xbb\xbf')
                f.write(csv_text.encode('utf-8'))

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            st.success(f"✅ Данные экспортированы в CSV: {output_path} ({file_size:.1f} МБ)")
            return True
        except Exception as e:
            print(e)
            st.error(f"❌ Ошибка экспорта в CSV: {e}")
            return False

    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total_records = self.conn.execute("SELECT count(DISTINCT (artikul_norm, brand_norm)) FROM parts_data").fetchone()[0]
        st.info(f"Всего записей для экспорта: {total_records:,}")
        if total_records == 0:
            st.warning("База пуста или нет данных для экспорта")
            return

        # Ввод исключений
        exclude_terms = st.text_input("Исключить позиции (через |):", value="")
        selected_columns = st.multiselect("Выберите столбцы для экспорта", options=[
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена с учетом наценки"
        ], default=[
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена с учетом наценки"
        ])

        export_format = st.radio("Формат экспорта:", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)

        if st.button("🚀 Экспортировать", type="primary"):
            output_path = self.data_dir / "auto_parts_export"
            if export_format == "CSV":
                out_file = str(output_path.with_suffix('.csv'))
                with st.spinner("Экспорт в CSV..."):
                    self.export_to_csv_optimized(out_file, selected_columns, exclude_terms)
                with open(out_file, "rb") as f:
                    st.download_button("📥 Скачать CSV", f, "auto_parts_export.csv", "text/csv")
            elif export_format == "Excel (.xlsx)":
                out_file = output_path.with_suffix('.xlsx')
                with st.spinner("Экспорт в Excel..."):
                    self.export_to_excel(out_file, selected_columns, exclude_terms)
                with open(out_file, "rb") as f:
                    st.download_button("📥 Скачать Excel", f, "auto_parts_export.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            elif export_format == "Parquet":
                out_file = str(output_path.with_suffix('.parquet'))
                with st.spinner("Экспорт в Parquet..."):
                    self.export_to_parquet(out_file, selected_columns, exclude_terms)
                with open(out_file, "rb") as f:
                    st.download_button("📥 Скачать Parquet", f, "auto_parts_export.parquet", "application/octet-stream")
 
    def export_to_excel(self, output_path: Path, selected_columns: List[str], exclude_terms: str):
        total_records = self.conn.execute("SELECT COUNT(DISTINCT (artikul_norm, brand_norm)) FROM parts_data").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        num_files = (total_records + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
        for i in range(num_files):
            query, params = self.build_export_query(selected_columns, exclude_terms)
            df = self.conn.execute(f"{query} LIMIT {EXCEL_ROW_LIMIT} OFFSET {i*EXCEL_ROW_LIMIT}", params).pl()
            # преобразуем числа
            for col_name in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if col_name in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col_name).is_not_null())
                        .then(pl.col(col_name).cast(pl.Utf8))
                        .otherwise(pl.lit(""))
                        .alias(col_name)
                    )
            file_part_path = output_path.with_name(f"{output_path.stem}_part_{i+1}.xlsx")
            df.write_excel(str(file_part_path))
        # ZIP если больше 1 файла
        # (можно оставить как есть)
        return True

    def export_to_parquet(self, output_path: str, selected_columns: List[str], exclude_terms: str):
        query, params = self.build_export_query(selected_columns, exclude_terms)
        df = self.conn.execute(query, params).pl()
        df.write_parquet(output_path)
        return True

    def build_export_query(self, selected_columns: List[str], exclude_terms: str):
        # возвращает строку SQL и параметры
        query, params = self._build_full_query(selected_columns, exclude_terms)
        return query, params

    def _build_full_query(self, selected_columns: List[str], exclude_terms: str):
        # Тут собирается полный SQL с учетом исключений
        # В этом месте встроим все вышеописанные конструкции
        # для краткости, пример с минимальной логикой:
        columns_map = [
            ("Артикул бренда", 'p.artikul AS "Артикул бренда"'),
            ("Бренд", 'p.brand AS "Бренд"'),
            ("Наименование", 'COALESCE(p.description, "") AS "Наименование"'),
            ("Применимость", '"" AS "Применимость"'),
            ("Описание", 'CONCAT(COALESCE(p.description, ""), dt.text) AS "Описание"'),
            ("Категория товара", '"" AS "Категория товара"'),
            ("Кратность", 'p.multiplicity AS "Кратность"'),
            ("Длинна", 'p.length AS "Длинна"'),
            ("Ширина", 'p.width AS "Ширина"'),
            ("Высота", 'p.height AS "Высота"'),
            ("Вес", 'p.weight AS "Вес"'),
            ("Длинна/Ширина/Высота", 'p.dimensions_str AS "Длинна/Ширина/Высота"'),
            ("OE номер", 'p.oe_list AS "OE номер"'),
            ("аналоги", 'p.analog_list AS "аналоги"'),
            ("Ссылка на изображение", 'p.image_url AS "Ссылка на изображение"'),
            ("Цена с учетом наценки", 'r.final_price AS "Цена с учетом наценки"')
        ]

        if not selected_columns:
            select_exprs = [expr for _, expr in columns_map]
        else:
            select_exprs = [expr for name, expr in columns_map if name in selected_columns]
            if not select_exprs:
                select_exprs = [expr for _, expr in columns_map]

        # Полный запрос
        sql = f"""
        WITH DescriptionTemplate AS (
            SELECT CHR(10) || CHR(10) || $${"""Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""}$$ AS text
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
            -- Аналоги
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
                ROW_NUMBER() OVER(PARTITION BY p.artikul_norm, p.brand_norm ORDER BY pd.representative_name DESC NULLS LAST, pd.oe_list DESC NULLS LAST) as rn,
                -- placeholder для final_price
                0 AS final_price
            FROM parts_data p
            LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
            LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
            LEFT JOIN AggregatedAnalogData p_analog ON p.artikul_norm = p_analog.artikul_norm AND p.brand_norm = p_analog.brand_norm
        )
        """

        # финальный запрос с расчетом цены
        query = f"""
        {ctes}
        SELECT
            {', '.join(selected_exprs)}
        FROM RankedData r
        LEFT JOIN recommended_prices rp ON r.artikul_norm = rp.artikul_norm AND r.brand_norm = rp.brand_norm
        LEFT JOIN parts_data p ON r.artikul_norm = p.artikul_norm AND r.brand_norm = p.brand_norm
        LEFT JOIN (
            SELECT
                p2.artikul_norm,
                p2.brand_norm,
                -- расчет финальной цены с учетом наценки
                COALESCE(rp.recommended_price, 0) * (1 + {self.overall_markup / 100}) * (1 + self.get_brand_markup('p2.brand')/100) AS final_price
            FROM parts_data p2
            LEFT JOIN recommended_prices rp ON p2.artikul_norm = rp.artikul_norm AND p2.brand_norm = rp.brand_norm
        ) p2 ON r.artikul_norm = p2.artikul_norm AND r.brand_norm = p2.brand_norm
        WHERE r.rn = 1
        {exclude_sql}
        ORDER BY r.brand, r.artikul
        """
        return query, params

    def export_to_csv_optimized(self, output_path: str, selected_columns: List[str] | None = None, exclude_terms: str = "") -> bool:
        total_records = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📤 Экспорт {total_records:,} записей в CSV...")
        try:
            query, params = self._build_full_query(selected_columns, exclude_terms)
            df = self.conn.execute(query, params).pl()

            # преобразуем числа
            for col_name in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if col_name in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col_name).is_not_null())
                        .then(pl.col(col_name).cast(pl.Utf8))
                        .otherwise(pl.lit(""))
                        .alias(col_name)
                    )

            buf = io.StringIO()
            df.write_csv(buf, separator=';')
            csv_text = buf.getvalue()

            with open(output_path, 'wb') as f:
                f.write(b'\xef\xbb\xbf')
                f.write(csv_text.encode('utf-8'))

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            st.success(f"✅ Данные экспортированы в CSV: {output_path} ({file_size:.1f} МБ)")
            return True
        except Exception as e:
            print(e)
            st.error(f"❌ Ошибка экспорта в CSV: {e}")
            return False

    def export_to_excel(self, output_path: Path, selected_columns: List[str], exclude_terms: str):
        total_records = self.conn.execute("SELECT COUNT(DISTINCT (artikul_norm, brand_norm)) FROM parts_data").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        num_files = (total_records + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
        for i in range(num_files):
            query, params = self._build_full_query(selected_columns, exclude_terms)
            df = self.conn.execute(f"{query} LIMIT {EXCEL_ROW_LIMIT} OFFSET {i*EXCEL_ROW_LIMIT}", params).pl()
            # преобразуем числа
            for col_name in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if col_name in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col_name).is_not_null())
                        .then(pl.col(col_name).cast(pl.Utf8))
                        .otherwise(pl.lit(""))
                        .alias(col_name)
                    )
            file_part_path = output_path.with_name(f"{output_path.stem}_part_{i+1}.xlsx")
            df.write_excel(str(file_part_path))
        return True

    def export_to_parquet(self, output_path: str, selected_columns: List[str], exclude_terms: str):
        query, params = self._build_full_query(selected_columns, exclude_terms)
        df = self.conn.execute(query, params).pl()
        df.write_parquet(output_path)
        return True

    def _build_full_query(self, selected_columns: List[str], exclude_terms: str):
        # Внутренний вызов для подготовки SQL
        query, params = self.build_export_query(selected_columns, exclude_terms)
        return query, params

    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total_records = self.conn.execute("SELECT count(DISTINCT (artikul_norm, brand_norm)) FROM parts_data").fetchone()[0]
        st.info(f"Всего записей для экспорта: {total_records:,}")
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return

        exclude_terms = st.text_input("Исключить позиции (через |):", value="")
        selected_columns = st.multiselect("Выберите столбцы", options=[
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена с учетом наценки"
        ], default=[
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена с учетом наценки"
        ])

        export_format = st.radio("Формат экспорта:", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)

        if st.button("🚀 Экспортировать", type="primary"):
            output_path = self.data_dir / "auto_parts_export"
            if export_format == "CSV":
                out_file = str(output_path.with_suffix('.csv'))
                with st.spinner("Экспорт в CSV..."):
                    self.export_to_csv_optimized(out_file, selected_columns, exclude_terms)
                with open(out_file, "rb") as f:
                    st.download_button("📥 Скачать CSV", f, "auto_parts_export.csv", "text/csv")
            elif export_format == "Excel (.xlsx)":
                out_file = output_path.with_suffix('.xlsx')
                with st.spinner("Экспорт в Excel..."):
                    self.export_to_excel(out_file, selected_columns, exclude_terms)
                with open(out_file, "rb") as f:
                    st.download_button("📥 Скачать Excel", f, "auto_parts_export.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            elif export_format == "Parquet":
                out_file = str(output_path.with_suffix('.parquet'))
                with st.spinner("Экспорт в Parquet..."):
                    self.export_to_parquet(out_file, selected_columns, exclude_terms)
                with open(out_file, "rb") as f:
                    st.download_button("📥 Скачать Parquet", f, "auto_parts_export.parquet", "application/octet-stream")

    # Методы загрузки файлов
    def load_files(self, files_dict: Dict[str, str]):
        for ftype, path in files_dict.items():
            if ftype == 'recommended_prices':
                self.load_recommended_prices(path)
            # Можно добавить другие загрузки по необходимости

    # Метод для установки наценки
    def configure_markups(self):
        self.overall_markup = st.number_input("Общая наценка (%)", value=0.0, step=1.0)
        brand_markups_input = st.text_area("Наценки по брендам (формат: бренд:процент, через запятую)", value="")
        self.brand_markups = {}
        if brand_markups_input:
            pairs = [pair.strip() for pair in brand_markups_input.split(',') if pair.strip()]
            for pair in pairs:
                if ':' in pair:
                    brand, percent = pair.split(':', 1)
                    self.brand_markups[brand.strip().lower()] = float(percent.strip())

    def get_brand_markup(self, brand_norm: str) -> float:
        return self.brand_markups.get(brand_norm.lower(), self.overall_markup)

    # Инициализация и запуск
    def run(self):
        # В UI добавьте загрузку файла цен
        st.sidebar.header("Загрузка дополнительных данных")
        prices_file = st.sidebar.file_uploader("Загрузить файл с рекомендованными ценами", type=['xlsx', 'xls'])
        if prices_file:
            save_path = self.data_dir / f"recommended_prices_{int(time.time())}_{prices_file.name}"
            with open(save_path, "wb") as f:
                f.write(prices_file.getvalue())
            self.load_recommended_prices(str(save_path))
            st.sidebar.success("Цены успешно загружены.")

        # Настройка наценок
        self.configure_markups()

        # Основной интерфейс
        self.show_export_interface()


# В основном запуске
def main():
    catalog = HighVolumeAutoPartsCatalog()
    catalog.run()

if __name__ == "__main__":
    main()
