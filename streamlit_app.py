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
import logging
import difflib

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

EXCEL_ROW_LIMIT = 1_000_000

class HighVolumeAutoPartsCatalog:
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
        # Для хранения цен и настроек
        self.prices_df = pl.DataFrame()
        self.price_markup = 1.0  # Общая наценка
        self.brand_markup = {}   # Наценка по брендам

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
                price DOUBLE,
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
        # Таблица цен (если потребуется)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                price DOUBLE,
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)

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

    def detect_category(self, name_series: pl.Series) -> pl.Series:
        categories_map = {
            'Фильтр': 'фильтр|filter',
            'Тормозная система': 'тормоз|brake|колодк|диск|суппорт',
            'Подвеска': 'амортизатор|стойк|spring|подвеск|Рычаг|Рычаги|Шаровая опора|Опора шаровая|Сайлентблок|Ступиц|подшипник ступицы|подшипники ступицы',
            'Двигатель': 'двигатель|engine|свеч|поршень|клапан',
            'Трансмиссия': 'трансмиссия|сцеплен|коробк|transmission',
            'Электрика': 'аккумулятор|генератор|стартер|провод|ламп',
            'Рулевое': 'рулевой|тяга|наконечник|steering',
            'Выхлопная система': 'глушитель|глушител|катализатор|выхлоп|exhaust',
            'Охлаждение': 'радиатор|вентилятор|термостат|cooling',
            'Топливо': 'топливный|бензонасос|форсунк|fuel',
        }
        name_lower = name_series.str.to_lowercase()
        categorization_expr = pl.when(pl.lit(False)).then(pl.lit(None))
        for cat, pattern in categories_map.items():
            categorization_expr = categorization_expr.when(name_lower.str.contains(pattern)).then(pl.lit(cat))
        return categorization_expr.otherwise(pl.lit('Разное')).alias('category')

    def read_and_prepare_file(self, file_path: str, file_type: str) -> pl.DataFrame:
        try:
            df = pl.read_excel(file_path, engine='calamine')
        except Exception as e:
            st.error(f"Не удалось прочитать файл {file_path}: {e}")
            return pl.DataFrame()

        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand']
        }
        expected_cols = schemas.get(file_type, [])
        column_mapping = self.detect_columns(df.columns, expected_cols)
        df = df.rename(column_mapping)

        # Очистка значений
        if 'artikul' in df.columns:
            df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand=self.clean_values(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))

        key_cols = [col for col in ['oe_number', 'artikul', 'brand'] if col in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')

        # Нормализация ключей
        if 'artikul' in df.columns:
            df = df.with_columns(artikul_norm=self.normalize_key(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand_norm=self.normalize_key(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number_norm= self.normalize_key(pl.col('oe_number')))
        return df

    def upsert_data(self, table_name: str, df: pl.DataFrame, pk: List[str]):
        if df.is_empty():
            return
        df = df.unique(keep='first')
        cols = df.columns
        pk_str = ", ".join(f'"{c}"' for c in pk)
        temp_view_name = f"temp_{table_name}_{int(time.time())}"
        self.conn.register(temp_view_name, df.to_arrow())

        update_cols = [col for col in cols if col not in pk]
        if not update_cols:
            on_conflict_action = "DO NOTHING"
        else:
            update_clause = ", ".join([f'"{col}" = excluded."{col}"' for col in update_cols])
            on_conflict_action = f"DO UPDATE SET {update_clause}"

        sql = f"""
        INSERT INTO {table_name}
        SELECT * FROM {temp_view_name}
        ON CONFLICT ({pk_str}) {on_conflict_action};
        """
        try:
            self.conn.execute(sql)
        finally:
            self.conn.unregister(temp_view_name)

    def process_and_load_data(self, dataframes: Dict[str, pl.DataFrame]):
        if 'oe' in dataframes:
            df = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'], keep='first')
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self.detect_category(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])
            cross_df = df.filter(pl.col('artikul_norm') != "").select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        if 'cross' in dataframes:
            df = dataframes['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            cross_df = df.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка артикула и прочих данных
        file_priority = ['oe', 'barcode', 'images', 'dimensions']
        key_files = {ftype: df for ftype, df in dataframes.items() if ftype in file_priority}
        if key_files:
            all_parts = pl.concat([
                df.select(['artikul', 'artikul_norm', 'brand', 'brand_norm']) 
                for ftype, df in key_files.items()
                if 'artikul_norm' in df.columns and 'brand_norm' in df.columns
            ]).filter(pl.col('artikul_norm') != "").unique(subset=['artikul_norm', 'brand_norm'])
            parts_df = all_parts

            for ftype in file_priority:
                if ftype not in key_files:
                    continue
                df = key_files[ftype]
                if df.is_empty() or 'artikul_norm' not in df.columns:
                    continue
                join_cols = [col for col in df.columns if col not in ['artikul', 'artikul_norm', 'brand', 'brand_norm']]
                existing_cols = set(parts_df.columns)
                join_cols = [col for col in join_cols if col not in existing_cols]
                if not join_cols:
                    continue
                df_subset = df.select(['artikul_norm', 'brand_norm'] + join_cols).unique(subset=['artikul_norm', 'brand_norm'])
                parts_df = parts_df.join(df_subset, on=['artikul_norm', 'brand_norm'], how='left', coalesce=True)

            # Обработка размеров, описание, цены
            if 'multiplicity' not in parts_df.columns:
                parts_df = parts_df.with_columns(multiplicity=pl.lit(1).cast(pl.Int32))
            else:
                parts_df = parts_df.with_columns(pl.col('multiplicity').fill_null(1).cast(pl.Int32))
            for col in ['length', 'width', 'height']:
                if col not in parts_df.columns:
                    parts_df = parts_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
            if 'dimensions_str' not in parts_df.columns:
                parts_df = parts_df.with_columns(dimensions_str=pl.lit(None).cast(pl.Utf8))
            # Создаем dimensions_str
            parts_df = parts_df.with_columns([
                pl.col('length').cast(pl.Utf8).fill_null('').alias('_length_str'),
                pl.col('width').cast(pl.Utf8).fill_null('').alias('_width_str'),
                pl.col('height').cast(pl.Utf8).fill_null('').alias('_height_str'),
            ])
            parts_df = parts_df.with_columns(
                dimensions_str=pl.when(
                    (pl.col('dimensions_str').is_not_null()) & (pl.col('dimensions_str') != '')
                ).then(pl.col('dimensions_str')).otherwise(
                    pl.concat_str([pl.col('_length_str'), pl.lit('x'), pl.col('_width_str'), pl.lit('x'), pl.col('_height_str')], separator='')
                )
            ).drop(['_length_str', '_width_str', '_height_str'])

            # Описание
            if 'artikul' not in parts_df.columns:
                parts_df = parts_df.with_columns(artikul=pl.lit(''))
            if 'brand' not in parts_df.columns:
                parts_df = parts_df.with_columns(brand=pl.lit(''))
            parts_df = parts_df.with_columns([
                pl.col('artikul').cast(pl.Utf8).fill_null('').alias('_artikul_str'),
                pl.col('brand').cast(pl.Utf8).fill_null('').alias('_brand_str'),
                pl.col('multiplicity').cast(pl.Utf8).alias('_multiplicity_str'),
            ])
            parts_df = parts_df.with_columns(
                description=pl.concat_str([
                    'Артикул: ', pl.col('_artikul_str'),
                    ', Бренд: ', pl.col('_brand_str'),
                    ', Кратность: ', pl.col('_multiplicity_str'), ' шт.'
                ], separator='')
            ).drop(['_artikul_str', '_brand_str', '_multiplicity_str'])

            # Обработка цен
            if 'price' not in parts_df.columns:
                parts_df = parts_df.with_columns(pl.lit(None).cast(pl.Float64).alias('price'))

            self.upsert_data('parts_data', parts_df, ['artikul_norm', 'brand_norm'])

        st.success("Данные успешно обновлены в базе.")

    def merge_all_data_parallel(self, file_paths: Dict[str, str]) -> dict:
        start_time = time.time()
        # Загрузка цен из прайса
        self.load_price_file(file_paths.get('price'))

        # Обработка всех файлов
        dataframes = {}
        for key, path in file_paths.items():
            if key != 'price':  # цена обрабатывается отдельно
                df = self.read_and_prepare_file(path, key)
                if not df.is_empty():
                    dataframes[key] = df
        self.process_and_load_data(dataframes)

        # Применение наценки
        self.apply_markups()

        total_time = time.time() - start_time
        total_records = self.get_total_records()
        return {
            'processing_time': total_time,
            'total_records': total_records
        }

    def load_price_file(self, price_file_path: str):
        if not price_file_path:
            return
        try:
            df_price = pl.read_excel(io.BytesIO(open(price_file_path, 'rb').read()), engine='calamine')
        except Exception as e:
            st.error(f"Ошибка при чтении прайса: {e}")
            return
        if all(c in df_price.columns for c in ['артикул', 'бренд', 'цена']):
            df_price = df_price.rename({'артикул': 'artikul', 'бренд': 'brand', 'цена': 'price'})
            df_price = df_price.with_columns([
                self.normalize_key(pl.col('artikul')).alias('artikul_norm'),
                self.normalize_key(pl.col('brand')).alias('brand_norm')
            ])
            self.prices_df = self.prices_df.vstack(df_price.select(['artikul_norm', 'brand_norm', 'price']))
            st.success("Прайс обновлен.")
        else:
            st.warning("Прайс должен содержать колонки: 'артикул', 'бренд', 'цена'.")

    def apply_price_markup(self):
        # Обновление цен по текущей наценке и бренду
        if self.prices_df.is_empty():
            return
        df_prices = self.prices_df
        for row in df_prices.iter_rows():
            artikul_norm = row[0]
            brand_norm = row[1]
            price = row[2]
            # Применяем наценки
            markup = self.brand_markup.get(brand_norm, self.price_markup)
            final_price = price * markup
            self.conn.execute("""
                UPDATE parts_data SET price = ? WHERE artikul_norm = ? AND brand_norm = ?
            """, [final_price, artikul_norm, brand_norm])

    def set_brand_markup(self, brand: str, percent: float):
        normalized_brand = self.normalize_key(pl.Series([brand]))[0]
        self.brand_markup[normalized_brand] = 1 + percent / 100
        self.apply_price_markup()
        st.info(f"Наценка для бренда '{brand}': {percent}% установлена.")

    def set_global_markup(self, percent: float):
        self.price_markup = 1 + percent / 100
        self.apply_price_markup()
        st.info(f"Общая наценка: {percent}% установлена.")

    def add_price_from_uploaded_file(self, file_bytes):
        try:
            df_price = pl.read_excel(io.BytesIO(file_bytes), engine='calamine')
        except Exception as e:
            st.error(f"Ошибка чтения прайса: {e}")
            return
        if all(c in df_price.columns for c in ['артикул', 'бренд', 'цена']):
            df_price = df_price.rename({'артикул': 'artikul', 'бренд': 'brand', 'цена': 'price'})
            df_price = df_price.with_columns([
                self.normalize_key(pl.col('artikul')).alias('artikul_norm'),
                self.normalize_key(pl.col('brand')).alias('brand_norm')
            ])
            self.prices_df = self.prices_df.vstack(df_price.select(['artikul_norm', 'brand_norm', 'price']))
            st.success("Цены успешно добавлены.")
        else:
            st.warning("Прайс-файл должен содержать колонки: 'артикул', 'бренд', 'цена'.")

    def set_brand_markup(self, brand: str, percent: float):
        normalized_brand = self.normalize_key(pl.Series([brand]))[0]
        self.brand_markup[normalized_brand] = 1 + percent / 100
        self.apply_price_markup()

    def set_global_markup(self, percent: float):
        self.price_markup = 1 + percent / 100
        self.apply_price_markup()

    def build_export_query(self, selected_columns: List[str]):
        standard_description = """Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""
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
            ("Ссылка на изображение", 'r.image_url AS "Ссылка на изображение"')
        ]
        if not selected_columns:
            selected_exprs = [expr for _, expr in columns_map]
        else:
            selected_exprs = [expr for name, expr in columns_map if name in selected_columns]
            if not selected_exprs:
                selected_exprs = [expr for _, expr in columns_map]

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
                ROW_NUMBER() OVER(PARTITION BY p.artikul_norm, p.brand_norm ORDER BY pd.representative_name DESC NULLS LAST, pd.oe_list DESC NULLS LAST) as rn
            FROM parts_data p
            LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
            LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
            LEFT JOIN (
                SELECT artikul_norm, brand_norm, length, width, height, weight, dimensions_str, representative_name, representative_applicability, representative_category
                FROM parts_data
            ) p_analog ON p.artikul_norm = p_analog.artikul_norm AND p.brand_norm = p_analog.brand_norm
        )
        """

        select_clause = ",\n            ".join(selected_exprs)
        query = ctes + f"""
        SELECT
            {select_clause}
        FROM RankedData r
        CROSS JOIN DescriptionTemplate dt
        WHERE r.rn = 1
        ORDER BY r.brand, r.artikul
        """
        return query

    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total_records = self.get_total_records()
        st.info(f"Всего записей для экспорта: {total_records:,}")
        if total_records == 0:
            st.warning("Нет данных для экспорта.")
            return
        options = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение"
        ]
        selected_columns = st.multiselect("Выберите столбцы для экспорта", options=options, default=options)

        format_type = st.radio("Формат экспорта:", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)
        if st.button("🚀 Начать экспорт"):
            output_path = Path("./auto_parts_export")
            output_path.mkdir(exist_ok=True)
            filename = f"auto_parts_{int(time.time())}"
            if format_type == "CSV":
                file_path = output_path / f"{filename}.csv"
                self.export_to_csv(selected_columns, str(file_path))
                with open(file_path, 'rb') as f:
                    st.download_button("📥 Скачать CSV", f, file_path.name, "text/csv")
            elif format_type == "Excel (.xlsx)":
                file_path = output_path / f"{filename}.xlsx"
                self.export_to_excel(selected_columns, file_path)
                with open(file_path, 'rb') as f:
                    st.download_button("📥 Скачать Excel", f, file_path.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                file_path = output_path / f"{filename}.parquet"
                self.export_to_parquet(selected_columns, str(file_path))
                with open(file_path, 'rb') as f:
                    st.download_button("📥 Скачать Parquet", f, file_path.name, "application/octet-stream")

    def delete_by_brand(self, brand_norm: str) -> int:
        try:
            res = self.conn.execute("DELETE FROM parts_data WHERE brand_norm = ?", [brand_norm])
            return res.rowcount
        except Exception as e:
            logging.exception(e)
            return 0

    def delete_by_artikul(self, artikul_norm: str) -> int:
        try:
            res = self.conn.execute("DELETE FROM parts_data WHERE artikul_norm = ?", [artikul_norm])
            return res.rowcount
        except Exception as e:
            logging.exception(e)
            return 0

    def get_statistics(self):
        total_parts = self.get_total_records()
        total_oe = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()[0]
        total_brands = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data").fetchone()[0]
        top_brands = self.conn.execute("SELECT brand, COUNT(*) as cnt FROM parts_data GROUP BY brand ORDER BY cnt DESC LIMIT 10").pl()
        categories = self.conn.execute("SELECT category, COUNT(*) as cnt FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY cnt DESC").pl()
        return {
            'total_parts': total_parts,
            'total_oe': total_oe,
            'total_brands': total_brands,
            'top_brands': top_brands,
            'categories': categories
        }

    def get_total_records(self):
        res = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()
        return res[0] if res else 0

    # Загрузка прайса
    def load_price_file(self, file_bytes):
        try:
            df_price = pl.read_excel(io.BytesIO(file_bytes), engine='calamine')
        except Exception as e:
            st.error(f"Ошибка чтения прайса: {e}")
            return
        if all(c in df_price.columns for c in ['артикул', 'бренд', 'цена']):
            df_price = df_price.rename({'артикул': 'artikul', 'бренд': 'brand', 'цена': 'price'})
            df_price = df_price.with_columns([
                self.normalize_key(pl.col('artikul')).alias('artikul_norm'),
                self.normalize_key(pl.col('brand')).alias('brand_norm')
            ])
            self.prices_df = self.prices_df.vstack(df_price.select(['artikul_norm', 'brand_norm', 'price']))
            st.success("Прайс обновлен.")
        else:
            st.warning("Прайс должен содержать колонки: 'артикул', 'бренд', 'цена'.")

    def apply_price_markup(self):
        # Обновление цен по текущей наценке и бренду
        if self.prices_df.is_empty():
            return
        df_prices = self.prices_df
        for row in df_prices.iter_rows():
            artikul_norm = row[0]
            brand_norm = row[1]
            price = row[2]
            markup = self.brand_markup.get(brand_norm, self.price_markup)
            final_price = price * markup
            self.conn.execute("""
                UPDATE parts_data SET price = ? WHERE artikul_norm = ? AND brand_norm = ?
            """, [final_price, artikul_norm, brand_norm])

    def set_brand_markup(self, brand: str, percent: float):
        normalized_brand = self.normalize_key(pl.Series([brand]))[0]
        self.brand_markup[normalized_brand] = 1 + percent / 100
        self.apply_price_markup()

    def set_global_markup(self, percent: float):
        self.price_markup = 1 + percent / 100
        self.apply_price_markup()

    def add_price_from_uploaded_file(self, file_bytes):
        self.load_price_file(file_bytes)
        self.apply_price_markup()

    def build_export_query(self, selected_columns: List[str]):
        standard_description = """Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""
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
            ("Ссылка на изображение", 'r.image_url AS "Ссылка на изображение"')
        ]
        if not selected_columns:
            selected_exprs = [expr for _, expr in columns_map]
        else:
            selected_exprs = [expr for name, expr in columns_map if name in selected_columns]
            if not selected_exprs:
                selected_exprs = [expr for _, expr in columns_map]

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
                ROW_NUMBER() OVER(PARTITION BY p.artikul_norm, p.brand_norm ORDER BY pd.representative_name DESC NULLS LAST, pd.oe_list DESC NULLS LAST) as rn
            FROM parts_data p
            LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
            LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
            LEFT JOIN (
                SELECT artikul_norm, brand_norm, length, width, height, weight, dimensions_str, representative_name, representative_applicability, representative_category
                FROM parts_data
            ) p_analog ON p.artikul_norm = p_analog.artikul_norm AND p.brand_norm = p_analog.brand_norm
        )
        """

        select_clause = ",\n            ".join(selected_exprs)
        query = ctes + f"""
        SELECT
            {select_clause}
        FROM RankedData r
        CROSS JOIN DescriptionTemplate dt
        WHERE r.rn = 1
        ORDER BY r.brand, r.artikul
        """
        return query

    def assign_category_by_name(self, search_name: str, category_name: str, similarity_threshold: float = 0.5):
        """
        Ищет в базе товары с похожими названиями и присваивает им указанную категорию.
        """
        # Получаем все уникальные названия
        res = self.conn.execute("SELECT DISTINCT name FROM oe_data WHERE name IS NOT NULL").fetchall()
        names = [row[0] for row in res]
        matched_names = []
        for name in names:
            ratio = difflib.SequenceMatcher(None, name.lower(), search_name.lower()).ratio()
            if ratio >= similarity_threshold:
                matched_names.append(name)
        if not matched_names:
            st.info("Похожих товаров не найдено.")
            return
        # Обновляем категорию для найденных названий
        for name in matched_names:
            self.conn.execute("""
                UPDATE oe_data SET category = ? WHERE name = ?
            """, [category_name, name])
        st.success(f"Названия, похожие на '{search_name}', обновлены на категорию '{category_name}'. Обновлено {len(matched_names)} записей.")

    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total_records = self.get_total_records()
        st.info(f"Всего записей для экспорта: {total_records:,}")
        if total_records == 0:
            st.warning("Нет данных для экспорта.")
            return
        options = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение"
        ]
        selected_columns = st.multiselect("Выберите столбцы для экспорта", options=options, default=options)

        format_type = st.radio("Формат экспорта:", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)
        if st.button("🚀 Начать экспорт"):
            output_path = Path("./auto_parts_export")
            output_path.mkdir(exist_ok=True)
            filename = f"auto_parts_{int(time.time())}"
            if format_type == "CSV":
                file_path = output_path / f"{filename}.csv"
                self.export_to_csv(selected_columns, str(file_path))
                with open(file_path, 'rb') as f:
                    st.download_button("📥 Скачать CSV", f, file_path.name, "text/csv")
            elif format_type == "Excel (.xlsx)":
                file_path = output_path / f"{filename}.xlsx"
                self.export_to_excel(selected_columns, file_path)
                with open(file_path, 'rb') as f:
                    st.download_button("📥 Скачать Excel", f, file_path.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                file_path = output_path / f"{filename}.parquet"
                self.export_to_parquet(selected_columns, str(file_path))
                with open(file_path, 'rb') as f:
                    st.download_button("📥 Скачать Parquet", f, file_path.name, "application/octet-stream")

    def delete_by_brand(self, brand_norm: str) -> int:
        try:
            res = self.conn.execute("DELETE FROM parts_data WHERE brand_norm = ?", [brand_norm])
            return res.rowcount
        except Exception as e:
            logging.exception(e)
            return 0

    def delete_by_artikul(self, artikul_norm: str) -> int:
        try:
            res = self.conn.execute("DELETE FROM parts_data WHERE artikul_norm = ?", [artikul_norm])
            return res.rowcount
        except Exception as e:
            logging.exception(e)
            return 0

    def get_statistics(self):
        total_parts = self.get_total_records()
        total_oe = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()[0]
        total_brands = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data").fetchone()[0]
        top_brands = self.conn.execute("SELECT brand, COUNT(*) as cnt FROM parts_data GROUP BY brand ORDER BY cnt DESC LIMIT 10").pl()
        categories = self.conn.execute("SELECT category, COUNT(*) as cnt FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY cnt DESC").pl()
        return {
            'total_parts': total_parts,
            'total_oe': total_oe,
            'total_brands': total_brands,
            'top_brands': top_brands,
            'categories': categories
        }

# Основная логика интерфейса
def main():
    st.title("🚗 AutoParts Catalog - Управление 10+ млн записей")
    catalog = HighVolumeAutoPartsCatalog()

    st.sidebar.title("🧭 Меню")
    menu = st.sidebar.radio("Выберите действие:", ["Загрузка данных", "Экспорт", "Статистика", "Управление ценами", "Управление данными"])

    if menu == "Загрузка данных":
        st.header("📥 Загрузка и обработка данных")
        st.info("Загрузите файлы Excel для обновления каталога.\n\n"
                "Можно загружать несколько файлов — они будут объединены.\n\n"
                "Типы файлов:\n"
                "- Основные данные (OE, артикул, бренд, наименование)\n"
                "- Кроссы (OE -> артикул)\n"
                "- Штрих-коды и кратность\n"
                "- Весогабариты\n"
                "- Изображения")
        col1, col2 = st.columns(2)
        with col1:
            oe_file = st.file_uploader("Основные данные (OE)", type=['xlsx', 'xls'])
            cross_file = st.file_uploader("Кроссы", type=['xlsx', 'xls'])
            barcode_file = st.file_uploader("Штрих-коды", type=['xlsx', 'xls'])
        with col2:
            dimensions_file = st.file_uploader("Весогабариты", type=['xlsx', 'xls'])
            images_file = st.file_uploader("Изображения", type=['xlsx', 'xls'])

        uploaded_files = {
            'oe': oe_file,
            'cross': cross_file,
            'barcode': barcode_file,
            'dimensions': dimensions_file,
            'images': images_file
        }

        if st.button("🚀 Начать обработку"):
            file_paths = {}
            for key, uploaded in uploaded_files.items():
                if uploaded:
                    path = catalog.data_dir / f"{key}_{int(time.time())}_{uploaded.name}"
                    with open(path, 'wb') as f:
                        f.write(uploaded.getvalue())
                    file_paths[key] = str(path)
            if file_paths:
                stats = catalog.merge_all_data_parallel(file_paths)
                st.success(f"Обработка завершена за {stats['processing_time']:.2f} сек. В базе {stats['total_records']:,} записей.")
            else:
                st.warning("Загрузите хотя бы один файл для обработки.")

    elif menu == "Экспорт":
        catalog.show_export_interface()

    elif menu == "Статистика":
        st.header("📈 Статистика по каталогу")
        stats = catalog.get_statistics()
        st.metric("Всего артикулов", stats['total_parts'])
        st.metric("OE номера", stats['total_oe'])
        st.metric("Бренды", stats['total_brands'])
        st.subheader("Топ брендов")
        if not stats['top_brands'].is_empty():
            st.dataframe(stats['top_brands'].to_pandas())
        st.subheader("Категории")
        if not stats['categories'].is_empty():
            st.bar_chart(stats['categories'].to_pandas().set_index('category'))

    elif menu == "Управление ценами":
        st.header("🛠️ Управление ценами")
        # Загрузка цен
        uploaded_price_file = st.file_uploader("Загрузить прайс (артикул, бренд, цена)", type=['xlsx', 'xls'])
        if uploaded_price_file:
            catalog.add_price_from_uploaded_file(uploaded_price_file.read())

        # Установка общего процента
        markup = st.number_input("Общая наценка (%)", min_value=0.0, max_value=100.0, value=0.0)
        if st.button("Применить общую наценку"):
            catalog.set_global_markup(markup)

        # Установка наценки по брендам
        st.subheader("Настройка наценки по брендам")
        brand_name = st.text_input("Бренд")
        brand_markup_percent = st.number_input("Наценка (%)", min_value=0.0, max_value=100.0, value=0.0)
        if st.button("Установить для бренда"):
            if brand_name:
                catalog.set_brand_markup(brand_name, brand_markup_percent)

    elif menu == "Управление данными":
        st.header("🗑️ Управление данными")
        option = st.radio("Действия", ["Удалить по бренду", "Удалить по артикулу", "Добавить категорию по названию"])
        if option == "Удалить по бренду":
            brands = []
            res = catalog.conn.execute("SELECT DISTINCT brand FROM parts_data WHERE brand IS NOT NULL").fetchall()
            for row in res:
                brands.append(row[0])
            selected_brand = st.selectbox("Выберите бренд для удаления", brands)
            if selected_brand:
                norm_brand = catalog.normalize_key(pl.Series([selected_brand]))[0]
                count = catalog.delete_by_brand(norm_brand)
                st.success(f"Удалено {count} записей по бренду '{selected_brand}'")
        elif option == "Удалить по артикулу":
            artikul_input = st.text_input("Артикул")
            if artikul_input:
                norm_artikul = catalog.normalize_key(pl.Series([artikul_input]))[0]
                count = catalog.delete_by_artikul(norm_artikul)
                st.success(f"Удалено {count} записей по артикулу '{artikul_input}'")
        elif option == "Добавить категорию по названию":
            search_name_input = st.text_input("Название товара для поиска")
            category_name_input = st.text_input("Категория товара")
            similarity_threshold = st.slider("Порог похожести", 0.0, 1.0, 0.5, step=0.05)
            if st.button("Присвоить категорию"):
                if search_name_input and category_name_input:
                    catalog.assign_category_by_name(search_name_input, category_name_input, similarity_threshold)
                else:
                    st.warning("Заполните оба поля: название и категория.")

if __name__ == "__main__":
    main()
