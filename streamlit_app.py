import polars as pl
import duckdb
import streamlit as st
import os
import time
import logging
from pathlib import Path
from typing import Dict, List
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
            CREATE TABLE IF NOT EXISTS prices (
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                price DOUBLE,
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)
        self.create_indexes()

    def create_indexes(self):
        st.info("Создание индексов для ускорения поиска...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_prices ON prices(artikul_norm, brand_norm)"
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
            .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
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
            .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    @staticmethod
    def determine_category_vectorized(name_series: pl.Series) -> pl.Series:
        categories_map = {
            'Фильтр': 'фильтр|filter', 
            'Тормозная система': 'тормоз|brake|колодк|диск|суппорт',
            'Подвеска': 'амортизатор|стойк|spring|подвеск|Рычаг|Рычаги|Шаровая опора|Опора шаровая|Сайлентблок|Ступиц|подшипник ступицы|подшипники ступицы', 
            'Двигатель': 'двигатель|engine|свеч|поршень|клапан',
            'Трансмиссия': 'трансмиссия|сцеплен|коробк|transmission', 
            'Электрика': 'аккумулятор|генератор|стартер|провод|ламп',
            'Рулевое': 'рулевой|тяга|наконечник|steering', 
            'Выхлопная система': 'глушитель|глушител|катализатор|выхлоп|exhaust|',
            'Охлаждение': 'радиатор|вентилятор|термостат|cooling', 
            'Топливо': 'топливный|бензонасос|форсунк|fuel',
        }
        name_lower = name_series.str.to_lowercase()
        categorization_expr = pl.when(pl.lit(False)).then(pl.lit(None))
        for category, pattern in categories_map.items():
            categorization_expr = categorization_expr.when(name_lower.str.contains(pattern)).then(pl.lit(category))
        return categorization_expr.otherwise(pl.lit('Разное')).alias('category')

    def detect_columns(self, actual_columns: List[str], expected_columns: List[str]) -> Dict[str, str]:
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
            'price': ['рекомендованная цена', 'рекомендуемая цена', 'price']
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
            df = pl.read_excel(file_path, engine='calamine')
        except Exception as e:
            logger.error(f"Не удалось прочитать файл {file_path}: {e}")
            return pl.DataFrame()

        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand'],
            'prices': ['artikul', 'brand', 'price']
        }
        expected_cols = schemas.get(file_type, [])
        column_mapping = self.detect_columns(df.columns, expected_cols)
        df = df.rename(column_mapping)
        
        # Очистить оригинальные значения от апострофов и мусора на входе
        if 'artikul' in df.columns:
            df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand=self.clean_values(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))
        
        key_cols = [col for col in ['oe_number', 'artikul', 'brand'] if col in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')

        # Создать нормализованные версии для ключей (нижний регистр)
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

    def load_prices(self, file_path: str) -> pl.DataFrame:
        logger.info(f"Загрузка цен из файла: {file_path}")
        df = pl.read_excel(file_path, engine='calamine')
        df = df.rename({"artikul": "artikul", "brand": "brand", "price": "price"})  # Убедитесь, что имена соответствуют
        df = df.with_columns(
            artikul_norm=self.normalize_key(pl.col('artikul')),
            brand_norm=self.normalize_key(pl.col('brand'))
        )
        return df

    def upsert_prices(self, df: pl.DataFrame, general_markup: float, brand_markup: Dict[str, float]):
        df = df.with_columns([
            (pl.col('price') * (1 + general_markup)).alias('final_price'),
        ])
        for brand, markup in brand_markup.items():
            df = df.with_columns(
                pl.when(pl.col('brand_norm') == brand)
                .then(pl.col('price') * (1 + markup))
                .otherwise(pl.col('final_price'))
                .alias('final_price')
            )

        self.upsert_data('prices', df.select(['artikul_norm', 'brand_norm', 'final_price']), ['artikul_norm', 'brand_norm'])

    def process_and_load_data(self, dataframes: Dict[str, pl.DataFrame]):
        st.info("🔄 Начало загрузки и обновления данных в базе...")

        steps = [s for s in ['oe', 'cross', 'parts', 'prices'] if s in dataframes or s == 'parts']
        num_steps = len(steps)
        progress_bar = st.progress(0, text="Подготовка к обновлению базы данных...")
        step_counter = 0

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

        if 'cross' in dataframes:
            step_counter += 1
            progress_bar.progress(step_counter / (num_steps + 1), text=f"({step_counter}/{num_steps}) Обработка кроссов...")
            df = dataframes['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            cross_df_from_cross = df.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_df_from_cross, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        step_counter += 1
        progress_bar.progress(step_counter / (num_steps + 1), text=f"({step_counter}/{num_steps}) Сборка и обновление данных по артикулам...")
        parts_df = None

        file_priority = ['oe', 'barcode', 'images', 'dimensions']
        key_files = {ftype: df for ftype, df in dataframes.items() if ftype in file_priority}
        
        if key_files:
            all_parts = pl.concat([ 
                df.select(['artikul', 'artikul_norm', 'brand', 'brand_norm']) 
                for df in key_files.values() if 'artikul_norm' in df.columns and 'brand_norm' in df.columns
            ]).filter(pl.col('artikul_norm') != "").unique(subset=['artikul_norm', 'brand_norm'], keep='first')

            parts_df = all_parts

            for ftype in file_priority:
                if ftype not in key_files: continue
                df = key_files[ftype]
                if df.is_empty() or 'artikul_norm' not in df.columns: continue
                
                join_cols = [col for col in df.columns if col not in ['artikul', 'artikul_norm', 'brand', 'brand_norm']]
                if not join_cols: continue
                
                existing_cols = set(parts_df.columns)
                join_cols = [col for col in join_cols if col not in existing_cols]
                if not join_cols: continue
                
                df_subset = df.select(['artikul_norm', 'brand_norm'] + join_cols).unique(subset=['artikul_norm', 'brand_norm'], keep='first')
                parts_df = parts_df.join(df_subset, on=['artikul_norm', 'brand_norm'], how='left', coalesce=True)

        if parts_df is not None and not parts_df.is_empty():
            if 'multiplicity' not in parts_df.columns:
                parts_df = parts_df.with_columns(multiplicity=pl.lit(1).cast(pl.Int32))
            else:
                parts_df = parts_df.with_columns(
                    pl.col('multiplicity').fill_null(1).cast(pl.Int32)
                )
            
            for col in ['length', 'width', 'height']:
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
                    (pl.col('dimensions_str').is_not_null()) & 
                    (pl.col('dimensions_str').cast(pl.Utf8) != '')
                ).then(
                    pl.col('dimensions_str').cast(pl.Utf8)
                ).otherwise(
                    pl.concat_str([
                        pl.col('_length_str'), pl.lit('x'), 
                        pl.col('_width_str'), pl.lit('x'), 
                        pl.col('_height_str')
                    ], separator='')
                )
            )
            
            parts_df = parts_df.drop(['_length_str', '_width_str', '_height_str'])
            
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
                    pl.lit('Артикул: '), pl.col('_artikul_str'),
                    pl.lit(', Бренд: '), pl.col('_brand_str'),
                    pl.lit(', Кратность: '), pl.col('_multiplicity_str'), pl.lit(' шт.')
                ], separator='')
            )
            
            parts_df = parts_df.drop(['_artikul_str', '_brand_str', '_multiplicity_str'])
            final_columns = [
                'artikul_norm', 'brand_norm', 'artikul', 'brand', 'multiplicity', 'barcode', 
                'length', 'width', 'height', 'weight', 'image_url', 'dimensions_str', 'description'
            ]
            select_exprs = [pl.col(c) if c in parts_df.columns else pl.lit(None).alias(c) for c in final_columns]
            parts_df = parts_df.select(select_exprs)
            
            self.upsert_data('parts_data', parts_df, ['artikul_norm', 'brand_norm'])

        if 'prices' in dataframes:
            price_df = dataframes['prices']
            general_markup = st.number_input("Общая наценка (%)", min_value=0.0, step=0.01) / 100.0
            brand_markup = {}
            brand_list = price_df['brand_norm'].unique().to_list()
            for brand in brand_list:
                markup = st.number_input(f"Наценка для {brand} (%)", min_value=0.0, step=0.01) / 100.0
                brand_markup[brand] = markup
            self.upsert_prices(price_df, general_markup, brand_markup)

        progress_bar.progress(1.0, text="Обновление базы данных завершено!")
        time.sleep(1)
        progress_bar.empty()
        st.success("💾 Загрузка данных в базу завершена.")

    def merge_all_data_parallel(self, file_paths: Dict[str, str]) -> Dict[str, any]:
        start_time = time.time()
        stats = {}
        
        st.info("🚀 Начало параллельного чтения и подготовки файлов...")
        n_files = len(file_paths)
        file_progress_bar = st.progress(0, text="Ожидание...")
        
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
                    else:
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

        self.process_and_load_data(dataframes)
        
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

    def get_export_query(self, selected_columns: List[str], exclusions: str) -> str:
        exclusions_condition = " AND ".join([
            f'NOT (brand_norm LIKE "%{exclusion}%" OR artikul_norm LIKE "%{exclusion}%")'
            for exclusion in exclusions.split('|') if exclusion
        ])
        
        query = f"""
        SELECT {', '.join(selected_columns)}
        FROM parts_data
        WHERE {exclusions_condition}
        """
        return query

    def show_pricing_interface(self):
        st.header("💸 Установка наценок")
        general_markup = st.number_input("Общая наценка (%)", min_value=0.0, step=0.01) / 100.0
        brand_markup = {}
        brand_list = self.conn.execute("SELECT DISTINCT brand_norm FROM prices").fetchall()
        for brand in brand_list:
            markup = st.number_input(f"Наценка для {brand[0]} (%)", min_value=0.0, step=0.01) / 100.0
            brand_markup[brand[0]] = markup

        if st.button("Загрузить цены"):
            price_file = st.file_uploader("Выберите файл с ценами", type=['xlsx', 'xls'])
            if price_file:
                df_prices = self.load_prices(price_file)
                self.upsert_prices(df_prices, general_markup, brand_markup)

        exclusions = st.text_input("Исключения (разделяйте '|' для нескольких наименований)")
        selected_columns = st.multiselect("Выберите столбцы для экспорта", options=["artikul", "brand", "price", "description"], default=["artikul", "brand", "price"])

        if st.button("Экспортировать"):
            query = self.get_export_query(selected_columns, exclusions)
            df_export = self.conn.execute(query).pl()
            output_path = self.data_dir / "exported_data.csv"
            df_export.write_csv(str(output_path))
            st.success(f"Данные успешно экспортированы: {output_path.name}")

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
        st.info("""
        **Порядок работы:**
        1. Загрузите один или несколько файлов Excel (`.xlsx`, `.xls`). Не обязательно загружать все сразу.
        2. Нажмите кнопку "Начать обработку".
        3. Система автоматически прочитает, объединит данные и обновит/дополнит существующую базу.
        
        **💡 Дополнение данных:**
        - Вы можете загружать файлы **по одному** или **пачками** (несколько файлов одновременно).
        - Система использует механизм UPSERT: новые записи добавляются, существующие обновляются.
        - При повторной загрузке файла с теми же артикулами/брендами данные будут обновлены, а не продублированы.
        - Можно загружать только те типы файлов, которые у вас есть - остальные просто пропускаются.
        
        **Типы Файлов:**
        - **Основные данные**: OE номера, артикулы, бренд, наименование.
        - **Кроссы (OE -> Артикул)**: Связь OE номеров с артикулами и брендами.
        - **Штрих-коды**: Связь артикулов со штрих-кодами и кратностью.
        - **Весогабариты**: Размеры и вес товаров.
        - **Изображения**: Ссылки на изображения.
        - **Рекомендованные цены**: Цены на артикулы.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            oe_file = st.file_uploader("1. Основные данные (OE)", type=['xlsx', 'xls'])
            cross_file = st.file_uploader("2. Кроссы (OE -> Артикул)", type=['xlsx', 'xls'])
            barcode_file = st.file_uploader("3. Штрих-коды и кратность", type=['xlsx', 'xls'])
        with col2:
            dimensions_file = st.file_uploader("4. Весогабаритные данные", type=['xlsx', 'xls'])
            images_file = st.file_uploader("5. Ссылки на изображения", type=['xlsx', 'xls'])
            prices_file = st.file_uploader("6. Рекомендованные цены", type=['xlsx', 'xls'])

        file_map = {
            'oe': oe_file, 
            'cross': cross_file, 
            'barcode': barcode_file,
            'dimensions': dimensions_file,
            'images': images_file,
            'prices': prices_file
        }
        
        if st.button("🚀 Начать обработку данных", type="primary"):
            paths_to_process = {}
            any_file_uploaded = False
            for ftype, uploaded_file in file_map.items():
                if uploaded_file:
                    any_file_uploaded = True
                    path = catalog.data_dir / f"{ftype}_data_{int(time.time())}_{uploaded_file.name}"
                    with open(path, "wb") as f: f.write(uploaded_file.getvalue())
                    paths_to_process[ftype] = str(path)
            
            if any_file_uploaded:
                stats = catalog.merge_all_data_parallel(paths_to_process)
                if stats:
                    st.subheader("📊 Статистика обработки")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Общее время", f"{stats.get('processing_time', 0):.2f} сек")
                    col2.metric("Всего артикулов в базе", f"{stats.get('total_records', 0):,}")
                    col3.metric("Обработано файлов", f"{len(paths_to_process)}")
            else:
                st.warning("⚠️ Пожалуйста, загрузите хотя бы один файл для начала обработки.")

    elif menu_option == "Экспорт":
        catalog.show_pricing_interface()
    
    elif menu_option == "Статистика":
        st.header("📈 Статистика по каталогу")
        with st.spinner("Сбор статистики..."):
            stats = catalog.get_statistics()
        
        if stats.get('total_parts', 0) > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Уникальных артикулов", f"{stats.get('total_parts', 0):,}")
            col2.metric("Уникальных OE", f"{stats.get('total_oe', 0):,}")
            col3.metric("Уникальных брендов", f"{stats.get('total_brands', 0):,}")
            
            st.subheader("🏆 Топ-10 брендов по количеству артикулов")
            if 'top_brands' in stats and not stats['top_brands'].is_empty():
                st.dataframe(stats['top_brands'].to_pandas(), width='stretch')
            else:
                st.write("Нет данных по брендам.")

            st.subheader("📊 Распределение по категориям")
            if 'categories' in stats and not stats['categories'].is_empty():
                st.bar_chart(stats['categories'].to_pandas().set_index('category'))
            else:
                st.write("Нет данных по категориям.")
        else:
            st.info("Данные отсутствуют. Перейдите в раздел 'Загрузка данных', чтобы начать.")
    
    elif menu_option == "Управление данными":
        st.header("🗑️ Управление данными в базе")
        st.warning("⚠️ Будьте осторожны! Операции удаления необратимы.")
        
        management_option = st.radio("Выберите операцию:", ["Удалить по бренду", "Удалить по артикулу"])
        
        if management_option == "Удалить по бренду":
            st.subheader("🏭 Удалить все артикулы определенного бренда")
            
            # Get list of available brands
            brands_result = catalog.conn.execute("SELECT DISTINCT brand FROM parts_data WHERE brand IS NOT NULL ORDER BY brand").pl()
            available_brands = brands_result['brand'].to_list() if not brands_result.is_empty() else []
            
            if available_brands:
                selected_brand = st.selectbox("Выберите бренд для удаления:", available_brands)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    brand_norm_result = catalog.conn.execute("SELECT brand_norm FROM parts_data WHERE brand = ? LIMIT 1", [selected_brand]).fetchone()
                    if brand_norm_result:
                        brand_norm = brand_norm_result[0]
                    else:
                        brand_series = pl.Series([selected_brand])
                        normalized_series = catalog.normalize_key(brand_series)
                        brand_norm = normalized_series[0] if len(normalized_series) > 0 else ""
                    
                    count_result = catalog.conn.execute("SELECT COUNT(*) FROM parts_data WHERE brand_norm = ?", [brand_norm]).fetchone()
                    count_to_delete = count_result[0] if count_result else 0
                    
                    st.info(f"К удалению: **{count_to_delete}** записей из бренда '{selected_brand}'")
                
                with col2:
                    confirm_delete_brand = st.checkbox("Я подтверждаю удаление всех записей этого бренда", key=f"confirm_brand_{selected_brand}")
                    if st.button("❌ Удалить все записи бренда", type="secondary", disabled=not confirm_delete_brand):
                        try:
                            deleted = catalog.delete_by_brand(brand_norm)
                            st.success(f"✅ Успешно удалено {deleted} записей для бренда '{selected_brand}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Ошибка при удалении: {e}")
                    if not confirm_delete_brand:
                        st.caption("⚠️ Отметьте чекбокс для активации кнопки удаления")
            else:
                st.warning("Нет доступных брендов для удаления.")
        
        elif management_option == "Удалить по артикулу":
            st.subheader("📦 Удалить все записи определенного артикула")
            st.info("💡 Введите артикул (поиск без учета регистра и спецсимволов)")
            
            input_artikul = st.text_input("Введите артикул для удаления:")
            
            if input_artikul:
                col1, col2 = st.columns(2)
                
                with col1:
                    if input_artikul:
                        input_series = pl.Series([input_artikul])
                        normalized_series = catalog.normalize_key(input_series)
                        artikul_norm = normalized_series[0] if len(normalized_series) > 0 else ""
                    
                    count_result = catalog.conn.execute("SELECT COUNT(*) FROM parts_data WHERE artikul_norm = ?", [artikul_norm]).fetchone()
                    count_to_delete = count_result[0] if count_result else 0
                    
                    if count_to_delete > 0:
                        st.info(f"К удалению: **{count_to_delete}** записей артикула '{input_artikul}'")
                    else:
                        st.warning(f"Артикул '{input_artikul}' не найден в базе")
                
                with col2:
                    if count_to_delete > 0:
                        confirm_delete_artikul = st.checkbox("Я подтверждаю удаление всех записей этого артикула", key=f"confirm_artikul_{artikul_norm}")
                        if st.button("❌ Удалить все записи артикула", type="secondary", disabled=not confirm_delete_artikul):
                            try:
                                deleted = catalog.delete_by_artikul(artikul_norm)
                                st.success(f"✅ Успешно удалено {deleted} записей для артикула '{input_artikul}'")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Ошибка при удалении: {e}")
                        if not confirm_delete_artikul:
                            st.caption("⚠️ Отметьте чекбокс для активации кнопки удаления")

if __name__ == "__main__":
    main()
