import polars as pl
import duckdb
import streamlit as st
import os
import time
import io
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXCEL_ROW_LIMIT = 1_000_000

class HighVolumeAutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        self.setup_database()

        # Визуальные настройки
        st.set_page_config(
            page_title="AutoParts Catalog 10M+", 
            layout="wide",
            page_icon="🚗"
        )

    def setup_database(self):
        # Создаем таблицы, если их нет
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
    def normalize_key(series: pl.Series) -> pl.Series:
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
    def clean_values(series: pl.Series) -> pl.Series:
        return (
            series
            .fill_null("")
            .cast(pl.Utf8)
            .str.replace_all("'", "")
            .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    def detect_columns(self, actual_cols: List[str], expected_cols: List[str]) -> Dict[str, str]:
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
        actual_lower = {col.lower(): col for col in actual_cols}
        for expected in expected_cols:
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
        try:
            df = pl.read_excel(file_path, engine='calamine')
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
            return pl.DataFrame()

        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand']
        }
        expected_cols = schemas.get(file_type, [])
        col_mapping = self.detect_columns(df.columns, expected_cols)
        df = df.rename(col_mapping)

        # Очистка оригинальных значений
        if 'artikul' in df.columns:
            df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand=self.clean_values(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))

        # Убираем дубли
        key_cols = [col for col in ['oe_number', 'artikul', 'brand'] if col in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')

        # Создаем нормализованные ключи
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
            on_conflict = "DO NOTHING"
        else:
            set_clause = ", ".join([f'"{col}" = excluded."{col}"' for col in update_cols])
            on_conflict = f"DO UPDATE SET {set_clause}"

        sql = f"""
        INSERT INTO {table_name}
        SELECT * FROM {temp_view_name}
        ON CONFLICT ({pk_str}) {on_conflict};
        """
        try:
            self.conn.execute(sql)
            logger.info(f"Upsert {len(df)} records into {table_name}")
        except Exception as e:
            logger.exception(f"Error upserting into {table_name}: {e}")
            st.error(f"Ошибка при записи в таблицу {table_name}")
        finally:
            self.conn.unregister(temp_view_name)

    def process_and_load_data(self, dataframes: Dict[str, pl.DataFrame]):
        st.info("🔄 Начинаю загрузку и обновление базы данных...")
        progress = st.progress(0.0)
        total_steps = 3
        step = 0

        # Обработка OE данных
        if 'oe' in dataframes:
            step += 1
            progress.progress(step / total_steps, text=f"Обработка OE данных ({step}/{total_steps})")
            df = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(subset=['oe_number_norm'])
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self.determine_category_vectorized(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])

            # Cross ссылки
            cross_df = df.filter(pl.col('artikul_norm') != "").select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка кроссов
        if 'cross' in dataframes:
            step += 1
            progress.progress(step / total_steps, text=f"Обработка кроссов ({step}/{total_steps})")
            df = dataframes['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            self.upsert_data('cross_references', df, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка артикула и связанных данных
        step += 1
        progress.progress(step / total_steps, text=f"Обработка артикула ({step}/{total_steps})")
        # Объединяем все файлы по артикулам
        relevant_files = ['oe', 'barcode', 'images', 'dimensions']
        parts_df = None
        combined_artikuls = []

        for ftype in relevant_files:
            if ftype in dataframes:
                df = dataframes[ftype]
                if 'artikul_norm' in df.columns:
                    combined_artikuls.append(df.select(['artikul_norm', 'brand_norm']))
        if combined_artikuls:
            all_parts = pl.concat(combined_artikuls).unique(subset=['artikul_norm', 'brand_norm'])
            parts_df = all_parts

            # Обработка размеров и описания
            if parts_df is not None and not parts_df.is_empty():
                # Обработка размеров
                for col in ['length', 'width', 'height']:
                    if col not in parts_df.columns:
                        parts_df = parts_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
                # Создание dimensions_str
                parts_df = parts_df.with_columns([
                    pl.col('length').cast(pl.Utf8).fill_null(''),
                    pl.col('width').cast(pl.Utf8).fill_null(''),
                    pl.col('height').cast(pl.Utf8).fill_null(''),
                ]).with_columns(
                    dimensions_str=pl.when(
                        (pl.col('dimensions_str').is_not_null()) & (pl.col('dimensions_str') != '') & (pl.col('dimensions_str') != 'XX')
                    ).then(pl.col('dimensions_str')).otherwise(
                        pl.concat_str([pl.col('length'), pl.lit('x'), pl.col('width'), pl.lit('x'), pl.col('height')], separator='')
                    )
                )

                # Обработка description
                if 'artikul' not in parts_df.columns:
                    parts_df = parts_df.with_columns(artikul=pl.lit(''))
                if 'brand' not in parts_df.columns:
                    parts_df = parts_df.with_columns(brand=pl.lit(''))

                parts_df = parts_df.with_columns([
                    pl.col('artikul').cast(pl.Utf8).fill_null(''),
                    pl.col('brand').cast(pl.Utf8).fill_null(''),
                    pl.col('multiplicity').fill_null(1).cast(pl.Int32),
                ])

                parts_df = parts_df.with_columns([
                    pl.concat_str(['Артикул: ', pl.col('artikul'), ', Бренд: ', pl.col('brand'), ', Кратность: ', pl.col('multiplicity').cast(pl.Utf8), ' шт.'], separator='').alias('description')
                ])

                # Стандартизация
                final_cols = ['artikul_norm', 'brand_norm', 'artikul', 'brand', 'multiplicity', 'barcode', 
                              'length', 'width', 'height', 'weight', 'image_url', 'dimensions_str', 'description']
                # Для отсутствующих колонок создаем пустые
                for c in final_cols:
                    if c not in parts_df.columns:
                        parts_df = parts_df.with_columns(pl.lit(None).cast(pl.Utf8).alias(c))
                parts_df = parts_df.select([pl.col(c) if c in parts_df.columns else pl.lit(None).alias(c) for c in final_cols])

                self.upsert_data('parts_data', parts_df, ['artikul_norm', 'brand_norm'])

        progress.progress(1.0)
        time.sleep(0.5)
        st.success("✅ Обработка данных завершена!")

    def merge_all_data_parallel(self, file_paths: Dict[str, str]) -> Dict:
        start_time = time.time()
        stats = {}

        # Чтение файлов параллельно
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.read_and_prepare_file, path, ftype): ftype
                for ftype, path in file_paths.items()
            }
            dataframes = {}
            for future in as_completed(futures):
                ftype = futures[future]
                try:
                    df = future.result()
                    if not df.is_empty():
                        dataframes[ftype] = df
                        st.success(f"Файл '{ftype}' прочитан: {len(df):,} строк")
                except Exception as e:
                    logger.exception(f"Ошибка при чтении файла {ftype}")
                    st.error(f"Ошибка при чтении файла {ftype}")
        if not dataframes:
            st.warning("Нет данных для обработки.")
            return {}

        self.process_and_load_data(dataframes)
        stats['processing_time'] = time.time() - start_time
        stats['total_records'] = self.get_total_records()
        st.success(f"Обработка завершена за {stats['processing_time']:.2f} с.")
        st.success(f"Всего артикулов: {stats['total_records']:,}")
        self.create_indexes()
        return stats

    def get_total_records(self) -> int:
        try:
            res = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()
            return res[0] if res else 0
        except:
            return 0

    def get_statistics(self):
        stats = {}
        try:
            stats['total_parts'] = self.get_total_records()
            if stats['total_parts'] == 0:
                return stats
            total_oe = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()
            total_b = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data WHERE brand IS NOT NULL").fetchone()
            top_brands = self.conn.execute("SELECT brand, COUNT(*) as cnt FROM parts_data WHERE brand IS NOT NULL GROUP BY brand ORDER BY cnt DESC LIMIT 10").pl()
            categories = self.conn.execute("SELECT category, COUNT(*) as cnt FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY cnt DESC").pl()
            stats['total_oe'] = total_oe[0]
            stats['total_brands'] = total_b[0]
            stats['top_brands'] = top_brands
            stats['categories'] = categories
        except:
            pass
        return stats

    def build_export_query(self, selected_columns: List[str] = None) -> str:
        # Внутренний текст
        description_text = """Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. 

Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. 

Выбирайте только лучшее — надежность и качество от ведущих производителей."""
        # Отображение
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
            select_exprs = [expr for _, expr in columns_map]
        else:
            select_exprs = [expr for name, expr in columns_map if name in selected_columns]
            if not select_exprs:
                select_exprs = [expr for _, expr in columns_map]

        ctes = f"""
        WITH DescriptionTemplate AS (
            SELECT CHR(10) || CHR(10) || $${description_text}$$ AS text
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
        """

        select_clause = ",\n            ".join(select_exprs)

        query = ctes + f"""
        SELECT
        {select_clause}
        FROM RankedData r
        CROSS JOIN DescriptionTemplate dt
        WHERE r.rn = 1
        ORDER BY r.brand, r.artikul
        """

        return query

    def export_to_csv(self, output_path: str, selected_columns: List[str] = None) -> bool:
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()
        total_records = total[0] if total else 0
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        try:
            query = self.build_export_query(selected_columns)
            df = self.conn.execute(query).pl()

            # Преобразуем числа в строки
            for col in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                if col in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col).is_not_null())
                        .then(pl.col(col).cast(pl.Utf8))
                        .otherwise("")
                        .alias(col)
                    )
            buf = io.StringIO()
            df.write_csv(buf, separator=';')
            csv_bytes = buf.getvalue().encode('utf-8-sig')
            with open(output_path, 'wb') as f:
                f.write(csv_bytes)
            st.success(f"Экспорт в CSV завершен: {output_path}")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта CSV")
            st.error(f"Ошибка экспорта CSV: {e}")
            return False

    def export_to_excel(self, output_path: Path, selected_columns: List[str] = None) -> Tuple[bool, Path]:
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()
        total_records = total[0] if total else 0
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False, None
        try:
            num_files = (total_records + EXCEL_ROW_LIMIT - 1) // EXCEL_ROW_LIMIT
            files_created = []
            for i in range(num_files):
                offset = i * EXCEL_ROW_LIMIT
                query = f"{self.build_export_query(selected_columns)} LIMIT {EXCEL_ROW_LIMIT} OFFSET {offset}"
                df = self.conn.execute(query).pl()
                # Чтоб не интерпретировали как даты
                for col in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота", "Кратность"]:
                    if col in df.columns:
                        df = df.with_columns(
                            pl.when(pl.col(col).is_not_null())
                            .then(pl.col(col).cast(pl.Utf8))
                            .otherwise("")
                            .alias(col)
                        )
                file_part = output_path.with_name(f"{output_path.stem}_part_{i+1}.xlsx")
                df.write_excel(str(file_part))
                files_created.append(file_part)
            if num_files > 1:
                # Архивация
                zip_path = output_path.with_suffix('.zip')
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in files_created:
                        zipf.write(file, file.name)
                        os.remove(file)
                return True, zip_path
            else:
                # Только один файл
                os.rename(files_created[0], output_path)
                return True, output_path
        except Exception as e:
            logger.exception("Ошибка экспорта Excel")
            st.error(f"Ошибка экспорта Excel: {e}")
            return False, None

    def export_to_parquet(self, output_path: str, selected_columns: List[str] = None) -> bool:
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data) AS t").fetchone()
        total_records = total[0] if total else 0
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        try:
            query = self.build_export_query(selected_columns)
            df = self.conn.execute(query).pl()
            df.write_parquet(output_path)
            st.success(f"Экспорт в Parquet завершен: {output_path}")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта Parquet")
            st.error(f"Ошибка экспорта Parquet: {e}")
            return False

    def show_export_interface(self):
        st.header("📤 Умный экспорт данных")
        total = self.conn.execute("SELECT COUNT(DISTINCT artikul_norm, brand_norm) FROM parts_data").fetchone()
        total_records = total[0] if total else 0
        if total_records == 0:
            st.warning("Нет данных для экспорта. Загрузите файлы.")
            return
        options = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота",
            "Вес", "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение"
        ]
        selected_cols = st.multiselect("Выберите колонки для экспорта (пусто = все)", options=options, default=options)
        format_choice = st.radio("Формат экспорта:", ["CSV", "Excel (.xlsx)", "Parquet"], index=0)

        if format_choice == "CSV":
            if st.button("🚀 Экспорт в CSV"):
                output_file = self.data_dir / "auto_parts_report.csv"
                success = self.export_to_csv(str(output_file), selected_cols if selected_cols else None)
                if success:
                    with open(output_file, "rb") as f:
                        st.download_button("📥 Скачать CSV", f, "auto_parts_report.csv", "text/csv")
        elif format_choice == "Excel (.xlsx)":
            if st.button("📊 Экспорт в Excel"):
                output_file = self.data_dir / "auto_parts_report.xlsx"
                success, path = self.export_to_excel(output_file, selected_cols if selected_cols else None)
                if success and path:
                    with open(path, "rb") as f:
                        st.download_button("📥 Скачать Excel", f, path.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            if st.button("⚡️ Экспорт в Parquet"):
                output_file = self.data_dir / "auto_parts_report.parquet"
                success = self.export_to_parquet(str(output_file), selected_cols if selected_cols else None)
                if success:
                    with open(output_file, "rb") as f:
                        st.download_button("📥 Скачать Parquet", f, "auto_parts_report.parquet", "application/octet-stream")

    def delete_by_brand(self, brand_norm: str) -> int:
        try:
            count_res = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE brand_norm = ?", [brand_norm]).fetchone()
            count = count_res[0] if count_res else 0
            if count == 0:
                return 0
            self.conn.execute("DELETE FROM parts_data WHERE brand_norm = ?", [brand_norm])
            # Удаляем старые cross, которые уже не нужны
            self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT artikul_norm, brand_norm FROM parts_data)")
            return count
        except:
            return 0

    def delete_by_artikul(self, artikul_norm: str) -> int:
        try:
            count_res = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE artikul_norm = ?", [artikul_norm]).fetchone()
            count = count_res[0] if count_res else 0
            if count == 0:
                return 0
            self.conn.execute("DELETE FROM parts_data WHERE artikul_norm = ?", [artikul_norm])
            self.conn.execute("DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT artikul_norm, brand_norm FROM parts_data)")
            return count
        except:
            return 0

# ===================== Основная функция =====================
def main():
    st.title("🚗 AutoParts Catalog - Профессиональная система для 10+ млн записей")
    st.markdown("""
    ### 💪 Мощная платформа для управления большими объемами данных автозапчастей
    - **Инкрементальные обновления**: добавляйте новые файлы без потери данных.
    - **Объединение**: корректно сливает данные из многофайлов.
    - **Оптимизация**: использует DuckDB для быстрого доступа.
    - **Умный экспорт**: быстрый и надежный вывод в CSV, Excel, Parquet.
    """)
    catalog = HighVolumeAutoPartsCatalog()

    menu = st.sidebar.radio("Навигация", ["Загрузка данных", "Экспорт", "Статистика", "Управление"])

    if menu == "Загрузка данных":
        st.header("📥 Загрузка файлов")
        col1, col2 = st.columns(2)
        with col1:
            oe_file = st.file_uploader("Основные данные (OE)", type=['xlsx', 'xls'])
            cross_file = st.file_uploader("Кроссы (OE -> Артикул)", type=['xlsx', 'xls'])
            barcode_file = st.file_uploader("Штрих-коды", type=['xlsx', 'xls'])
        with col2:
            dimensions_file = st.file_uploader("Весогабариты", type=['xlsx', 'xls'])
            images_file = st.file_uploader("Изображения", type=['xlsx', 'xls'])

        if st.button("🚀 Начать обработку"):
            files_to_process = {}
            for name, uploaded in [('oe', oe_file), ('cross', cross_file),
                                   ('barcode', barcode_file), ('dimensions', dimensions_file),
                                   ('images', images_file)]:
                if uploaded:
                    path = Path("./auto_parts_data") / f"{name}_{int(time.time())}_{uploaded.name}"
                    with open(path, 'wb') as f:
                        f.write(uploaded.read())
                    files_to_process[name] = str(path)
            if files_to_process:
                catalog.merge_all_data_parallel(files_to_process)
            else:
                st.warning("Загрузите хотя бы один файл.")

    elif menu == "Экспорт":
        catalog.show_export_interface()

    elif menu == "Статистика":
        st.header("📊 Статистика")
        stats = catalog.get_statistics()
        st.metric("Всего артикулов", stats.get('total_parts', 0))
        st.metric("OE номеров", stats.get('total_oe', 0))
        st.metric("Брендов", stats.get('total_brands', 0))
        if 'top_brands' in stats and not stats['top_brands'].is_empty():
            st.subheader("Топ брендов")
            st.dataframe(stats['top_brands'].to_pandas())
        if 'categories' in stats and not stats['categories'].is_empty():
            st.subheader("Распределение по категориям")
            st.bar_chart(stats['categories'].to_pandas().set_index('category'))

    elif menu == "Управление":
        st.header("🗑️ Удаление данных")
        option = st.radio("Удалить по:", ["Бренду", "Артикулу"])
        if option == "Бренду":
            brands = catalog.conn.execute("SELECT DISTINCT brand FROM parts_data WHERE brand IS NOT NULL").fetchall()
            brand_list = [b[0] for b in brands]
            selected = st.selectbox("Выберите бренд", brand_list)
            norm = catalog.conn.execute("SELECT brand_norm FROM parts_data WHERE brand = ? LIMIT 1", [selected]).fetchone()
            brand_norm = norm[0] if norm else ''
            count = catalog.delete_by_brand(brand_norm)
            st.info(f"Удалено {count} записей для бренда {selected}")
        else:
            artikul_input = st.text_input("Введите артикул для удаления")
            if artikul_input:
                # нормализуем
                norm_series = catalog.normalize_key(pl.Series([artikul_input]))
                artikul_norm = norm_series[0]
                count = catalog.delete_by_artikul(artikul_norm)
                st.info(f"Удалено {count} записей для артикула {artikul_input}")

if __name__ == "__main__":
    main()
