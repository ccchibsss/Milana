import os
import time
import io
import zipfile
import warnings
import logging
from pathlib import Path
from typing import Dict, List

import polars as pl
import duckdb
import streamlit as st

from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXCEL_ROW_LIMIT = 1_000_000

class HighVolumeAutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(database=str(self.db_path))
        self.setup_database()

        st.set_page_config(
            page_title="AutoParts Catalog 10M+", 
            layout="wide",
            page_icon="🚗"
        )

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
                цена DOUBLE DEFAULT NULL,
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
            CREATE TABLE IF NOT EXISTS settings (
                key VARCHAR PRIMARY KEY,
                value FLOAT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS brand_markups (
                brand_norm VARCHAR PRIMARY KEY,
                markup FLOAT
            )
        """)

    def create_indexes(self):
        st.info("Создание индексов для ускорения поиска...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)",
        ]
        for index_sql in indexes:
            self.conn.execute(index_sql)
        st.success("Индексы созданы.")

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

    @staticmethod
    def clean_values(value_series: pl.Series) -> pl.Series:
        return (
            value_series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all("'", "")
            .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    @staticmethod
    def determine_category_vectorized(name_series: pl.Series) -> pl.Series:
        categories_map = {
            'Фильтр': 'фильтр|filter', 'Тормоза': 'тормоз|brake|колодк|диск|суппорт',
            'Подвеска': 'амортизатор|стойк|spring|подвеск|рычаг', 'Двигатель': 'двигатель|engine|свеч|поршень|клапан',
            'Трансмиссия': 'трансмиссия|сцеплен|коробк|transmission', 'Электрика': 'аккумулятор|генератор|стартер|провод|ламп',
            'Рулевое': 'рулевой|тяга|наконечник|steering', 'Выпуск': 'глушитель|катализатор|выхлоп|exhaust',
            'Охлаждение': 'радиатор|вентилятор|термостат|cooling', 'Топливо': 'топливный|бензонасос|форсунк|fuel'
        }
        name_lower = name_series.str.to_lowercase()
        categorization_expr = pl.when(pl.lit(False)).then(pl.lit(None))
        for category, pattern in categories_map.items():
            categorization_expr = categorization_expr.when(name_lower.str.contains(pattern)).then(pl.lit(category))
        return categorization_expr.otherwise(pl.lit('Разное')).alias('category')

    def detect_columns(self, actual_columns: List[str], expected_columns: List[str]) -> Dict[str, str]:
        mapping = {}
        column_variants = {
            'oe_number': ['oe номер', 'oe', 'оe', 'номер', 'code', 'OE'], 'artikul': ['артикул', 'article', 'sku'],
            'brand': ['бренд', 'brand', 'производитель', 'manufacturer'], 'name': ['наименование', 'название', 'name', 'описание', 'description'],
            'applicability': ['применимость', 'автомобиль', 'vehicle', 'applicability'], 'barcode': ['штрих-код', 'barcode', 'штрихкод', 'ean', 'eac13'],
            'multiplicity': ['кратность шт', 'кратность', 'multiplicity'], 'length': ['длина (см)', 'длина', 'length', 'длинна'],
            'width': ['ширина (см)', 'ширина', 'width'], 'height': ['высота (см)', 'высота', 'height'],
            'weight': ['вес (кг)', 'вес, кг', 'вес', 'weight'], 'image_url': ['ссылка', 'url', 'изображение', 'image', 'картинка'],
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

    def read_and_prepare_file(self, file_path: str, file_type: str) -> pl.DataFrame:
        logger.info(f"Начинаю обработку файла: {file_type} ({file_path})")
        try:
            if not os.path.exists(file_path):
                logger.error(f"Файл не найден: {file_path}")
                return pl.DataFrame()
            file_size = os.path.getsize(file_path)
            if file_size == 0:
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
            'cross': ['oe_number', 'artikul', 'brand']
        }
        expected_cols = schemas.get(file_type, [])
        column_mapping = self.detect_columns(df.columns, expected_cols)
        if not column_mapping:
            logger.warning(f"Не удалось определить колонки для файла {file_type}. Доступные колонки: {df.columns}")
            return pl.DataFrame()
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
        # Нормализация
        if 'artikul' in df.columns:
            df = df.with_columns(artikul_norm=self.normalize_key(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand_norm=self.normalize_key(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number_norm=self.normalize_key(pl.col('oe_number')))
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
            logger.info(f"Успешно обновлено/вставлено {len(df)} записей в таблицу {table_name}.")
        except Exception as e:
            logger.error(f"Ошибка при UPSERT в {table_name}: {e}")
            st.error(f"Ошибка при записи в таблицу {table_name}. Детали в логе.")
        finally:
            self.conn.unregister(temp_view_name)

    def process_and_load(self, dataframes: Dict[str, pl.DataFrame]):
        st.info("🔄 Начинаю загрузку и обновление данных в базе...")
        steps = [s for s in ['oe', 'cross', 'parts'] if s in dataframes or s == 'parts']
        num_steps = len(steps)
        progress_bar = st.progress(0, text="Подготовка к обновлению базы данных...")
        step_counter = 0
        # Обработка oe
        if 'oe' in dataframes:
            step_counter += 1
            progress_bar.progress(step_counter / (num_steps + 1), text=f"({step_counter}/{num_steps}) Обработка OE данных...")
            df = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'], keep='first')
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self.determine_category_vectorized(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])
            cross_df_from_oe = df.filter(pl.col('artikul_norm') != "").select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df_from_oe, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка cross
        if 'cross' in dataframes:
            step_counter += 1
            progress_bar.progress(step_counter / (num_steps + 1), text=f"({step_counter}/{num_steps}) Обработка кроссов...")
            df = dataframes['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            cross_df_from_cross = df.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df_from_cross, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка parts
        step_counter += 1
        progress_bar.progress(step_counter / (num_steps + 1), text=f"({step_counter}/{num_steps}) Обработка деталей...")

        # Объединяем артикула и бренды из всех файлов типа
        file_priority = ['oe', 'barcode', 'images', 'dimensions']
        key_files = {ftype: df for ftype, df in dataframes.items() if ftype in file_priority}
        if key_files:
            all_parts = pl.concat([
                df.select(['artikul', 'artikul_norm', 'brand', 'brand_norm']) 
                for ftype, df in key_files.items() if 'artikul_norm' in df.columns and 'brand_norm' in df.columns
            ]).filter(pl.col('artikul_norm') != "").unique(subset=['artikul_norm', 'brand_norm'], keep='first')
            parts_df = all_parts
            for ftype in file_priority:
                if ftype not in key_files: continue
                df = key_files[ftype]
                if df.is_empty() or 'artikul_norm' not in df.columns:
                    continue
                join_cols = [col for col in df.columns if col not in ['artikul', 'artikul_norm', 'brand', 'brand_norm']]
                if not join_cols:
                    continue
                existing_cols = set(parts_df.columns)
                join_cols = [col for col in join_cols if col not in existing_cols]
                if not join_cols:
                    continue
                df_subset = df.select(['artikul_norm', 'brand_norm'] + join_cols).unique(subset=['artikul_norm', 'brand_norm'], keep='first')
                parts_df = parts_df.join(df_subset, on=['artikul_norm', 'brand_norm'], how='left', coalesce=True)

        if 'parts_df' in locals() and not parts_df.is_empty():
            # Обработка dimensions и description
            if 'multiplicity' not in parts_df.columns:
                parts_df = parts_df.with_columns(multiplicity=pl.lit(1).cast(pl.Int32))
            else:
                parts_df = parts_df.with_columns(pl.col('multiplicity').fill_null(1).cast(pl.Int32))
            for col in ['length', 'width', 'height']:
                if col not in parts_df.columns:
                    parts_df = parts_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
            if 'dimensions_str' not in parts_df.columns:
                parts_df = parts_df.with_columns(dimensions_str=pl.lit(None).cast(pl.Utf8))
            # Создаем строки размеров
            parts_df = parts_df.with_columns([
                pl.col('length').cast(pl.Utf8).fill_null('').alias('_length_str'),
                pl.col('width').cast(pl.Utf8).fill_null('').alias('_width_str'),
                pl.col('height').cast(pl.Utf8).fill_null('').alias('_height_str'),
            ])
            parts_df = parts_df.with_columns(
                pl.when(pl.col('dimensions_str').is_not_null() & (pl.col('dimensions_str') != '')).then(pl.col('dimensions_str'))
                .otherwise(
                    pl.concat_str([pl.col('_length_str'), pl.lit('x'), pl.col('_width_str'), pl.lit('x'), pl.col('_height_str')], separator='')
                ).alias('dimensions_str')
            )
            parts_df = parts_df.drop(['_length_str', '_width_str', '_height_str'])
            # Обработка description
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
                pl.concat_str([
                    'Артикул: ', pl.col('_artikul_str'),
                    ', Бренд: ', pl.col('_brand_str'),
                    ', Кратность: ', pl.col('_multiplicity_str'), ' шт.'
                ], separator='').alias('description')
            )
            parts_df = parts_df.drop(['_artikul_str', '_brand_str', '_multiplicity_str'])

            final_columns = [
                'artikul_norm', 'brand_norm', 'artikul', 'brand', 'multiplicity', 'barcode', 
                'length', 'width', 'height', 'weight', 'image_url', 'dimensions_str', 'description'
            ]
            select_exprs = [pl.col(c) if c in parts_df.columns else pl.lit(None).alias(c) for c in final_columns]
            parts_df = parts_df.select(select_exprs)
            self.upsert_data('parts_data', parts_df, ['artikul_norm', 'brand_norm'])
        progress_bar.progress(1.0, text="Обновление базы данных завершено!")
        time.sleep(1)
        progress_bar.empty()
        st.success("💾 Загрузка данных в базу завершена.")

    def merge_all_data_parallel(self, file_paths: Dict[str, str]) -> Dict[str, any]:
        start_time = time.time()
        stats = {}
        st.info("🚀 Начало параллельного чтения и подготовки файлов...")
        n_files = len(file_paths)
        file_progress_bar = st.progress(0, text=f"Обработка файлов...")
        dataframes = {}
        processed_files = 0
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(self.read_and_prepare_file, path, ftype): ftype for ftype, path in file_paths.items()}
            for future in as_completed(future_to_file):
                ftype = future_to_file[future]
                try:
                    df = future.result()
                    if not df.is_empty():
                        dataframes[ftype] = df
                        st.success(f"✅ Файл '{ftype}' прочитан: {len(df):,} строк.")
                        logger.info(f"Файл '{ftype}' успешно обработан: {len(df):,} строк, колонки: {df.columns}")
                    else:
                        logger.warning(f"Файл '{ftype}' вернул пустой DataFrame после обработки")
                        st.warning(f"⚠️ Файл '{ftype}' пуст или не удалось обработать.")
                except Exception as e:
                    logger.exception(f"Ошибка обработки файла {ftype}")
                    st.error(f"❌ Ошибка в {ftype}: {e}")
                finally:
                    processed_files += 1
                    file_progress_bar.progress(processed_files / n_files, text=f"Обработка файла: {ftype} ({processed_files}/{n_files})")
        file_progress_bar.empty()
        if not dataframes:
            st.error("❌ Ни один файл не был загружен. Обработка остановлена.")
            return {}
        self.process_and_load(dataframes)
        processing_time = time.time() - start_time
        total_records = self.get_total_records()
        stats['processing_time'] = processing_time
        stats['total_records'] = total_records
        st.success(f"🎉 Обработка завершена за {processing_time:.2f} секунд")
        st.success(f"📊 Всего уникальных артикулов в базе: {total_records:,}")
        self.create_indexes()
        return stats

    def get_total_records(self) -> int:
        try:
            result = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()
            return result[0] if result else 0
        except (duckdb.Error, TypeError):
            return 0

    def get_export_query(self) -> str:
        # Ваша логика формирования запроса
        return "SELECT * FROM parts_data"

    def generate_exclude_filter(self, exclude_input: str):
        """Создает SQL условие для исключения позиций по названиям."""
        terms = [term.strip() for term in exclude_input.split('|') if term.strip()]
        if not terms:
            return ""
        conditions = []
        for term in terms:
            escaped_term = term.replace("'", "''")
            conditions.append(f"(name LIKE '%{escaped_term}%' OR name = '{escaped_term}')")
        return " OR ".join(conditions)

    def export_to_csv_optimized(self, output_path: str, selected_columns: List[str] | None = None, exclude_names: str = "") -> bool:
        total_records = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📤 Экспорт {total_records:,} записей в CSV...")
        try:
            query = self.get_export_query()

            # Добавляем фильтр исключений по наименованиям
            exclude_filter = self.generate_exclude_filter(exclude_names)
            if exclude_filter:
                query += f" WHERE NOT ({exclude_filter})"

            df = self.conn.execute(query).pl()
            # преобразование чисел
            dimension_cols = ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]
            for col_name in dimension_cols:
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
            st.success(f"✅ Данные экспортированы: {output_path} ({file_size:.1f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта в CSV")
            st.error(f"❌ Ошибка экспорта в CSV: {e}")
            return False

    def export_to_excel(self, output_path: Path, selected_columns: List[str] | None = None, exclude_names: str = "") -> tuple[bool, Path | None]:
        total_records = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False, None
        st.info(f"📤 Экспорт {total_records:,} записей в Excel...")
        try:
            num_files = (total_records + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
            exported_files = []
            progress_bar = st.progress(0, text=f"Подготовка к экспорту {num_files} файла(ов)...")
            for i in range(num_files):
                progress_bar.progress((i + 1) / num_files, text=f"Экспорт части {i+1} из {num_files}...")
                offset = i * EXCEL_ROW_LIMIT
                query = f"{self.get_export_query()} LIMIT {EXCEL_ROW_LIMIT} OFFSET {offset}"

                # добавляем фильтр исключений
                exclude_filter = self.generate_exclude_filter(exclude_names)
                if exclude_filter:
                    query += f" WHERE NOT ({exclude_filter})"

                df = self.conn.execute(query).pl()
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
                exported_files.append(file_part_path)
            progress_bar.empty()
            if len(exported_files) > 1:
                st.info("Архивация файлов в ZIP...")
                zip_path = output_path.with_suffix('.zip')
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in exported_files:
                        zipf.write(file, file.name)
                        os.remove(file)
                final_path = zip_path
            else:
                final_path = exported_files[0]
                if final_path.name != output_path.name:
                    os.rename(final_path, output_path)
                    final_path = output_path
            file_size = os.path.getsize(final_path) / (1024 * 1024)
            st.success(f"✅ Данные экспортированы: {final_path.name} ({file_size:.1f} МБ)")
            return True, final_path
        except Exception as e:
            logger.exception("Ошибка экспорта в Excel")
            st.error(f"❌ Ошибка экспорта в Excel: {e}")
            return False, None

    def export_to_parquet(self, output_path: str, selected_columns: List[str] | None = None, exclude_names: str = "") -> bool:
        total_records = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📤 Экспорт {total_records:,} записей в Parquet...")
        try:
            query = self.get_export_query()
            # добавляем исключение по имени
            exclude_filter = self.generate_exclude_filter(exclude_names)
            if exclude_filter:
                query += f" WHERE NOT ({exclude_filter})"

            df = self.conn.execute(query).pl()
            df.write_parquet(output_path)
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            st.success(f"✅ Данные экспортированы в Parquet: {output_path} ({file_size:.1f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта в Parquet")
            st.error(f"❌ Ошибка экспорта в Parquet: {e}")
            return False

    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total_records = self.conn.execute("SELECT COUNT(DISTINCT (artikul_norm, brand_norm)) FROM parts_data").fetchone()[0]
        st.info(f"Всего записей для экспорта (строк): {total_records:,}")
        if total_records == 0:
            st.warning("База данных пуста или нет связей для экспорта. Сначала загрузите данные.")
            return
        # Выбор колонок
        available_columns = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение"
        ]
        selected_columns = st.multiselect("Выберите столбцы для экспорта (можно менять порядок):", options=available_columns, default=available_columns, key='columns_select')
        # сохранение порядка
        # В будущем можно сохранять выбранный порядок в сессии или файле

        # Ввод исключений по наименованию
        exclude_names = st.text_input("Исключить по названию (через |):", key='exclude_names')

        export_format = st.radio("Выберите формат экспорта:", ["CSV", "Excel (.xlsx)", "Parquet (для разработчиков)"], index=0)

        if export_format == "CSV":
            if st.button("🚀 Экспорт в CSV", key='export_csv'):
                output_path = self.data_dir / "auto_parts_report.csv"
                with st.spinner("Идет экспорт в CSV..."):
                    success = self.export_to_csv_optimized(str(output_path), selected_columns, exclude_names)
                if success:
                    with open(output_path, "rb") as f:
                        st.download_button("📥 Скачать CSV файл", f, "auto_parts_report.csv", "text/csv")
        elif export_format == "Excel (.xlsx)":
            st.info("ℹ️ Если записей больше 1 млн, результат будет разделен на несколько файлов и упакован в ZIP-архив.")
            if st.button("📊 Экспорт в Excel", key='export_excel'):
                output_path = self.data_dir / "auto_parts_report.xlsx"
                with st.spinner("Идет экспорт в Excel..."):
                    success, final_path = self.export_to_excel(output_path, selected_columns, exclude_names)
                if success and final_path and final_path.exists():
                    with open(final_path, "rb") as f:
                        mime = "application/zip" if final_path.suffix == ".zip" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        st.download_button(f"📥 Скачать {final_path.name}", f, final_path.name, mime)
        elif export_format == "Parquet (для разработчиков)":
            if st.button("⚡️ Экспорт в Parquet", key='export_parquet'):
                output_path = self.data_dir / "auto_parts_report.parquet"
                with st.spinner("Идет экспорт в Parquet..."):
                    success = self.export_to_parquet(str(output_path), selected_columns, exclude_names)
                if success:
                    with open(output_path, "rb") as f:
                        st.download_button("📥 Скачать Parquet файл", f, "auto_parts_report.parquet", "application/octet-stream")

    def delete_by_brand(self, brand_norm: str) -> int:
        try:
            count_result = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE brand_norm = ?", [brand_norm]).fetchone()
            deleted_count = count_result[0] if count_result else 0
            if deleted_count == 0:
                logger.info(f"No records for brand {brand_norm}")
                return 0
            self.conn.execute("DELETE FROM parts_data WHERE brand_norm = ?", [brand_norm])
            self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)")
            logger.info(f"Deleted {deleted_count} records for brand: {brand_norm}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting by brand {brand_norm}: {e}")
            raise

    def delete_by_artikul(self, artikul_norm: str) -> int:
        try:
            count_result = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE artikul_norm = ?", [artikul_norm]).fetchone()
            deleted_count = count_result[0] if count_result else 0
            if deleted_count == 0:
                logger.info(f"No records for artikul {artikul_norm}")
                return 0
            self.conn.execute("DELETE FROM parts_data WHERE artikul_norm = ?", [artikul_norm])
            self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)")
            logger.info(f"Deleted {deleted_count} records for artikul: {artikul_norm}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting by artikul {artikul_norm}: {e}")
            raise

    def get_statistics(self) -> Dict:
        stats = {}
        try:
            stats['total_parts'] = self.get_total_records()
            if stats['total_parts'] == 0:
                return {
                    'total_parts': 0, 'total_oe': 0, 'total_brands': 0,
                    'top_brands': pl.DataFrame(), 'categories': pl.DataFrame()
                }
            total_oe_res = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()
            stats['total_oe'] = total_oe_res[0] if total_oe_res else 0
            total_brands_res = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data WHERE brand IS NOT NULL").fetchone()
            stats['total_brands'] = total_brands_res[0] if total_brands_res else 0
            try:
                result = self.conn.execute("SELECT brand, COUNT(*) as count FROM parts_data WHERE brand IS NOT NULL GROUP BY brand ORDER BY count DESC LIMIT 10")
                rows = result.fetchall()
                if rows:
                    stats['top_brands'] = pl.DataFrame(rows, schema=["brand", "count"])
                else:
                    stats['top_brands'] = pl.DataFrame(schema=["brand", "count"])
            except Exception as e:
                logger.error(f"Ошибка при получении статистики брендов: {e}")
                stats['top_brands'] = pl.DataFrame(schema=["brand", "count"])
            try:
                result = self.conn.execute("SELECT category, COUNT(*) as count FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY count DESC")
                rows = result.fetchall()
                if rows:
                    stats['categories'] = pl.DataFrame(rows, schema=["category", "count"])
                else:
                    stats['categories'] = pl.DataFrame(schema=["category", "count"])
            except Exception as e:
                logger.error(f"Ошибка при получении статистики категорий: {e}")
                stats['categories'] = pl.DataFrame(schema=["category", "count"])
        except Exception as e:
            logger.error(f"Ошибка при сборе статистики: {e}")
            return {
                'total_parts': 0, 'total_oe': 0, 'total_brands': 0,
                'top_brands': pl.DataFrame(), 'categories': pl.DataFrame()
            }
        return stats

    def load_recommended_prices(self, file_path: str):
        """Загрузка рекомендаций цен из файла"""
        try:
            df = pl.read_excel(file_path, engine='calamine')
            if not all(c in df.columns for c in ['артикул', 'количество', 'бренд', 'цена']):
                st.error("Некорректный формат файла цен. Требуются столбцы: артикул, количество, бренд, цена.")
                return
            for _, row in df.iterrows():
                artikul_norm = self.normalize_key(pl.Series([row['артикул']]))[0]
                brand_norm = self.normalize_key(pl.Series([row['бренд']]))[0]
                цена = row['цена']
                self.conn.execute("""
                    INSERT INTO parts_data (artikul_norm, brand_norm, цена)
                    VALUES (?, ?, ?)
                    ON CONFLICT (artikul_norm, brand_norm) DO UPDATE SET цена=excluded.цена
                """, [artikul_norm, brand_norm, цена])
            st.success("Цены успешно загружены и обновлены.")
        except Exception as e:
            st.error(f"Ошибка загрузки цен: {e}")

    def set_global_markup(self, percent: float):
        """Устанавливает общую наценку"""
        self.conn.execute("INSERT INTO settings (key, value) VALUES ('global_markup', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [percent])
        st.success(f"Общая наценка установлена на {percent}%.")

    def set_brand_markup(self, brand: str, percent: float):
        """Устанавливает наценку по бренду"""
        brand_norm = self.normalize_key(pl.Series([brand]))[0]
        self.conn.execute("INSERT INTO brand_markups (brand_norm, markup) VALUES (?, ?) ON CONFLICT(brand_norm) DO UPDATE SET markup=excluded.markup", [brand_norm, percent])
        st.success(f"Наценка для бренда '{brand}' установлена на {percent}%.")

    def generate_exclude_filter(self, exclude_input: str):
        """Создает SQL условие для исключения позиций по названиям."""
        terms = [term.strip() for term in exclude_input.split('|') if term.strip()]
        if not terms:
            return ""
        conditions = []
        for term in terms:
            escaped_term = term.replace("'", "''")
            conditions.append(f"(name LIKE '%{escaped_term}%' OR name = '{escaped_term}')")
        return " OR ".join(conditions)

# Основная функция
def main():
    st.title("🚗 AutoParts Catalog - Профессиональная система для 10+ млн записей")
    st.markdown("""
    ### 💪 Мощная платформа для управления большими объемами данных автозапчастей
    - **Инкрементальные обновления**: Безопасно добавляйте новые файлы для дополнения и обновления каталога.
    - **Надежное объединение**: Данные из 5-ти типов файлов корректно сливаются в единую базу.
    - **Оптимизированное хранение**: Использование DuckDB для мгновенного доступа и анализа.
    - **Умный экспорт**: Быстрый и надежный экспорт в CSV, Excel или Parquet с гарантией отсутствия дубликатов.
    """)
    
    catalog = HighVolumeAutoPartsCatalog()
    
    st.sidebar.title("🧭 Навигация")
    menu_option = st.sidebar.radio("Выберите действие:", ["Загрузка данных", "Экспорт", "Статистика", "Управление данными"])
    
    if menu_option == "Загрузка данных":
        st.header("📥 Загрузка и обработка данных")
        is_database_empty = catalog.get_total_records() == 0
        if is_database_empty:
            st.warning("⚠️ **База данных пуста. Требуется начальная загрузка всех файлов.**")
            st.info("**Типы Файлов (все обязательны для начальной загрузки):**\n"
                    "1. Основные данные (OE): OE номера, артикулы, бренд, наименование.\n"
                    "2. Кроссы (OE -> Артикул): Связь OE номеров с артикулами и брендами.\n"
                    "3. Штрих-коды: Связь артикулов со штрихкодами и кратностью.\n"
                    "4. Весогабариты: Размеры и вес товаров.\n"
                    "5. Изображения: Ссылки на изображения.")
        else:
            st.success("✅ База содержит данные. Можно добавлять файлы.")
        col1, col2 = st.columns(2)
        with col1:
            oe_file = st.file_uploader("1. Основные данные (OE)", type=['xlsx', 'xls'], key='oe_upload')
            cross_file = st.file_uploader("2. Кроссы (OE -> Артикул)", type=['xlsx', 'xls'], key='cross_upload')
            barcode_file = st.file_uploader("3. Штрих-коды и кратность", type=['xlsx', 'xls'], key='barcode_upload')
        with col2:
            dimensions_file = st.file_uploader("4. Весогабаритные", type=['xlsx', 'xls'], key='dimensions_upload')
            images_file = st.file_uploader("5. Изображения", type=['xlsx', 'xls'], key='images_upload')
        file_map = {
            'oe': oe_file, 'cross': cross_file, 'barcode': barcode_file,
            'dimensions': dimensions_file, 'images': images_file
        }
        if st.button("🚀 Начать обработку данных", key='start_processing'):
            paths_to_process = {}
            uploaded_files_count = 0
            for ftype, uploaded_file in file_map.items():
                if uploaded_file:
                    uploaded_files_count += 1
                    path = catalog.data_dir / f"{ftype}_data_{int(time.time())}_{uploaded_file.name}"
                    with open(path, "wb") as f: f.write(uploaded_file.getvalue())
                    paths_to_process[ftype] = str(path)
            if catalog.get_total_records() == 0:
                # Начальная загрузка
                required_files = ['oe', 'cross', 'barcode', 'dimensions', 'images']
                missing = [f for f in required_files if f not in paths_to_process]
                if missing:
                    st.error(f"❌ Для начальной загрузки нужно загрузить все 5 файлов. Отсутствуют: {', '.join(missing)}")
                elif len(paths_to_process) == 5:
                    stats = catalog.merge_all_data_parallel(paths_to_process)
                    if stats:
                        st.subheader("📊 Статистика обработки")
                        st.metric("Общее время", f"{stats.get('processing_time', 0):.2f} сек")
                        st.metric("Всего артикулов", f"{stats.get('total_records', 0):,}")
                        st.metric("Обработано файлов", len(paths_to_process))
                else:
                    st.error("Загрузите все 5 файлов для начальной загрузки.")
            else:
                if uploaded_files_count > 0:
                    stats = catalog.merge_all_data_parallel(paths_to_process)
                    if stats:
                        st.subheader("📊 Статистика обработки")
                        st.metric("Общее время", f"{stats.get('processing_time', 0):.2f} сек")
                        st.metric("Всего артикулов", f"{stats.get('total_records', 0):,}")
                        st.metric("Обработано файлов", len(paths_to_process))
                else:
                    st.warning("⚠️ Загрузите хотя бы один файл для обработки.")
    elif menu_option == "Экспорт":
        catalog.show_export_interface()
    elif menu_option == "Статистика":
        st.header("📈 Статистика по каталогу")
        with st.spinner("Сбор статистики..."):
            stats = catalog.get_statistics()
        if stats.get('total_parts', 0) > 0:
            st.metric("Уникальных артикулов", f"{stats.get('total_parts', 0):,}")
            st.metric("Уникальных OE", f"{stats.get('total_oe', 0):,}")
            st.metric("Уникальных брендов", f"{stats.get('total_brands', 0):,}")
            st.subheader("🏆 Топ-10 брендов по количеству артикулов")
            if 'top_brands' in stats and not stats['top_brands'].is_empty():
                st.dataframe(stats['top_brands'].to_pandas())
            else:
                st.write("Нет данных по брендам.")
            st.subheader("📊 Распределение по категориям")
            if 'categories' in stats and not stats['categories'].is_empty():
                st.bar_chart(stats['categories'].to_pandas().set_index('category'))
            else:
                st.write("Нет данных по категориям.")
        else:
            st.info("Нет данных.")
    elif menu_option == "Управление данными":
        st.header("🗑️ Управление данными")
        st.warning("⚠️ Будьте осторожны! Операции необратимы.")
        management_option = st.radio("Операция:", ["Удалить по бренду", "Удалить по артикулу"])
        if management_option == "Удалить по бренду":
            try:
                brands_result = catalog.conn.execute("SELECT DISTINCT brand FROM parts_data WHERE brand IS NOT NULL ORDER BY brand").fetchall()
                available_brands = [row[0] for row in brands_result] if brands_result else []
            except Exception:
                available_brands = []
            if available_brands:
                selected_brand = st.selectbox("Бренд для удаления", available_brands)
                brand_norm_res = catalog.conn.execute("SELECT brand_norm FROM parts_data WHERE brand = ? LIMIT 1", [selected_brand]).fetchone()
                brand_norm = brand_norm_res[0] if brand_norm_res else catalog.normalize_key(pl.Series([selected_brand]))[0]
                count_res = catalog.conn.execute("SELECT COUNT(*) FROM parts_data WHERE brand_norm = ?", [brand_norm]).fetchone()
                count_del = count_res[0] if count_res else 0
                st.info(f"Удаляется {count_del} записей для бренда '{selected_brand}'")
                if st.button("❌ Удалить все для бренда", key='delete_brand'):
                    deleted = catalog.delete_by_brand(brand_norm)
                    st.success(f"Удалено {deleted} записей")
            else:
                st.warning("Нет доступных брендов.")
        else:
            # Удаление по артикулу
            artikul_input = st.text_input("Артикул для удаления", key='artikul_delete')
            if artikul_input:
                artikul_norm = catalog.normalize_key(pl.Series([artikul_input]))[0]
                count_res = catalog.conn.execute("SELECT COUNT(*) FROM parts_data WHERE artikul_norm = ?", [artikul_norm]).fetchone()
                deleted_count = count_res[0] if count_res else 0
                st.info(f"Удаляется {deleted_count} записей для артикула '{artikul_input}'")
                if st.button("❌ Удалить по артикулу", key='delete_artikul'):
                    deleted = catalog.delete_by_artikul(artikul_norm)
                    st.success(f"Удалено {deleted} записей")
    # Настройки цен и наценок в разделе управления
    st.markdown("## 🏷️ Настройки цен и наценок")
    with st.expander("Загрузка рекомендованных цен"):
        price_file = st.file_uploader("Загрузить файл цен", type=['xlsx'])
        if price_file:
            temp_path = catalog.data_dir / f"recommended_prices_{int(time.time())}.xlsx"
            with open(temp_path, "wb") as f:
                f.write(price_file.getvalue())
            if st.button("Загрузить цены из файла", key='load_prices'):
                catalog.load_recommended_prices(str(temp_path))
    # Наценки только здесь
    with st.expander("Общая наценка"):
        markup_percent = st.number_input(
            "Процент наценки", min_value=0.0, max_value=100.0, step=0.1, key='global_markup_percent'
        )
        if st.button("Установить общую наценку", key='set_global_markup'):
            catalog.set_global_markup(markup_percent)
    with st.expander("Наценка по бренду"):
        brand_name = st.text_input("Название бренда", key='brand_name_markup')
        brand_markup_percent = st.number_input(
            "Процент наценки", min_value=0.0, max_value=100.0, step=0.1, key='brand_markup_percent'
        )
        if st.button("Установить для бренда", key='set_brand_markup'):
            if brand_name:
                catalog.set_brand_markup(brand_name, brand_markup_percent)
            else:
                st.warning("Введите название бренда.")

if __name__ == "__main__":
    main()
