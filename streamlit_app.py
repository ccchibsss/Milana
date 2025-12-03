import polars as pl
import duckdb
import streamlit as st
import os
import time
import logging
import io
import zipfile
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
        self.conn = duckdb.connect(str(self.db_path))
        self.setup_database()
        self.global_markup = 0.0
        self.brand_markup: Dict[str, float] = {}
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
            CREATE TABLE IF NOT EXISTS price_recommendations (
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                artikul VARCHAR,
                brand VARCHAR,
                quantity INTEGER,
                price DOUBLE,
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)
    def create_indexes(self):
        st.info("Создание индексов...")
        for sql in [
            "CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)"
        ]:
            self.conn.execute(sql)
        st.success("Индексы созданы.")
    @staticmethod
    def normalize_key(series: pl.Series) -> pl.Series:
        return (series
                .fill_null("")
                .cast(pl.Utf8)
                .str.replace_all("'", "")
                .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .str.to_lowercase())
    @staticmethod
    def clean_values(series: pl.Series) -> pl.Series:
        return (series
                .fill_null("")
                .cast(pl.Utf8)
                .str.replace_all("'", "")
                .str.replace_all(r"[^0-9A-Za-zA-za-яЁё`\-\s]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars())
    @staticmethod
    def determine_category_vectorized(name_series: pl.Series) -> pl.Series:
        cats = {
            'Фильтр': 'фильтр|filter',
            'Тормоза': 'тормоз|brake|колодк|диск|суппорт',
            'Подвеска': 'амортизатор|стойк|spring|подвеск|рычаг',
            'Двигатель': 'двигатель|engine|свеч|поршень|клапан',
            'Трансмиссия': 'трансмиссия|сцеплен|коробк|transmission',
            'Электрика': 'аккумулятор|генератор|стартер|провод|ламп',
            'Рулевое': 'рулевой|тяга|наконечник|steering',
            'Выпуск': 'глушитель|катализатор|выхлоп|exhaust',
            'Охлаждение': 'радиатор|вентилятор|термостат|cooling',
            'Топливо': 'топливный|бензонасос|форсунк|fuel'
        }
        lower = name_series.str.to_lowercase()
        expr = pl.when(pl.lit(False)).then(pl.lit(None))
        for cat, pattern in cats.items():
            expr = expr.when(lower.str.contains(pattern)).then(pl.lit(cat))
        return expr.otherwise(pl.lit('Разное')).alias('category')
    def detect_columns(self, actual_cols, expected_cols):
        mapping = {}
        variants = {
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
        for key, keys in variants.items():
            for v in keys:
                for ac, ac_orig in actual_lower.items():
                    if v in ac:
                        mapping[ac_orig] = key
                        break
        return mapping
    def read_and_prepare_file(self, path, ftype):
        try:
            if not os.path.exists(path) or os.path.getsize(path)==0:
                return pl.DataFrame()
            df = pl.read_excel(path, engine='calamine')
            if df.is_empty():
                return pl.DataFrame()
        except:
            return pl.DataFrame()
        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand']
        }
        exp_cols = schemas.get(ftype, [])
        mapping = self.detect_columns(df.columns, exp_cols)
        if not mapping:
            return pl.DataFrame()
        df = df.rename(mapping)
        if 'artikul' in df.columns:
            df = df.with_columns(artikul=self.clean_values(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand=self.clean_values(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number=self.clean_values(pl.col('oe_number')))
        key_cols = [c for c in ['oe_number','artikul','brand'] if c in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')
        if 'artikul' in df.columns:
            df = df.with_columns(artikul_norm=self.normalize_key(pl.col('artikul')))
        if 'brand' in df.columns:
            df = df.with_columns(brand_norm=self.normalize_key(pl.col('brand')))
        if 'oe_number' in df.columns:
            df = df.with_columns(oe_number_norm=self.normalize_key(pl.col('oe_number')))
        return df
    def upsert_data(self, tablename, df, pk):
        if df.is_empty():
            return
        df = df.unique(keep='first')
        cols = df.columns
        pk_str = ", ".join(f'"{c}"' for c in pk)
        tname = f"temp_{tablename}_{int(time.time())}"
        self.conn.register(tname, df.to_arrow())
        update_cols = [c for c in cols if c not in pk]
        if not update_cols:
            conflict_sql = f"INSERT INTO {tablename} SELECT * FROM {tname} ON CONFLICT ({pk_str}) DO NOTHING;"
        else:
            set_clause = ", ".join([f'"{c}"=excluded."{c}"' for c in update_cols])
            conflict_sql = f"INSERT INTO {tablename} SELECT * FROM {tname} ON CONFLICT ({pk_str}) DO UPDATE SET {set_clause};"
        try:
            self.conn.execute(conflict_sql)
        finally:
            self.conn.unregister(tname)
    def process_and_load_data(self, dfs):
        st.info("🔄 Обновление базы данных...")
        steps = ['oe','cross','parts']
        n_step = len(steps)
        pbar = st.progress(0)
        idx=0
        # Обработка OE
        if 'oe' in dfs:
            idx+=1
            pbar.progress(idx/n_step, "Обработка OE")
            df_oe = dfs['oe'].filter(pl.col('oe_number_norm') != "")
            oe_df = df_oe.select(['oe_number_norm','oe_number','name','applicability']).unique(subset=['oe_number_norm'])
            if 'name' in oe_df.columns:
                oe_df = oe_df.with_columns(self.determine_category_vectorized(pl.col('name')))
            else:
                oe_df = oe_df.with_columns(category=pl.lit('Разное'))
            self.upsert_data('oe_data', oe_df, ['oe_number_norm'])
            cross_df = df_oe.filter(pl.col('artikul_norm') != "").select(['oe_number_norm','artikul_norm','brand_norm']).unique()
            self.upsert_data('cross_references', cross_df, ['oe_number_norm','artikul_norm','brand_norm'])
        # Обработка cross
        if 'cross' in dfs:
            idx+=1
            pbar.progress(idx/n_step, "Обработка кроссов")
            df_cross = dfs['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            self.upsert_data('cross_references', df_cross, ['oe_number_norm','artikul_norm','brand_norm'])
        # Обработка parts
        idx+=1
        pbar.progress(idx/n_step, "Обработка артикулов")
        parts_df = None
        p_files = {k:v for k,v in dfs.items() if k in ['oe','barcode','images','dimensions']}
        if p_files:
            all_parts = pl.concat([v.select(['artikul','artikul_norm','brand','brand_norm']) for v in p_files.values() if 'artikul_norm' in v.columns])
            all_parts = all_parts.filter(pl.col('artikul_norm') != "").unique(subset=['artikul_norm'])
            parts_df = all_parts
            for f in ['oe','barcode','images','dimensions']:
                if f not in p_files:
                    continue
                df = p_files[f]
                if df.is_empty() or 'artikul_norm' not in df.columns:
                    continue
                join_cols = [c for c in df.columns if c not in ['artikul','artikul_norm','brand','brand_norm']]
                if not join_cols:
                    continue
                existing_cols = set(parts_df.columns)
                join_cols = [c for c in join_cols if c not in existing_cols]
                if not join_cols:
                    continue
                df2 = df.select(['artikul_norm','brand_norm']+join_cols).unique(subset=['artikul_norm'])
                parts_df = parts_df.join(df2, on=['artikul_norm','brand_norm'], how='left', coalesce=True)
        if parts_df and not parts_df.is_empty():
            if 'multiplicity' not in parts_df.columns:
                parts_df = parts_df.with_columns(multiplicity=pl.lit(1).cast(pl.Int32))
            else:
                parts_df = parts_df.with_columns(pl.col('multiplicity').fill_null(1).cast(pl.Int32))
            for col in ['length','width','height']:
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
                    (pl.col('dimensions_str').is_not_null()) & (pl.col('dimensions_str') != '')
                ).then(
                    pl.col('dimensions_str')
                ).otherwise(
                    pl.concat_str([pl.col('_length_str'), 'x', pl.col('_width_str'), 'x', pl.col('_height_str')], separator='')
                )
            )
            parts_df = parts_df.drop(['_length_str','_width_str','_height_str'])
            # Создаем описание
            if 'artikul' not in parts_df.columns:
                parts_df = parts_df.with_columns(artikul=pl.lit(''))
            if 'brand' not in parts_df.columns:
                parts_df = parts_df.with_columns(brand=pl.lit(''))
            parts_df = parts_df.with_columns([
                pl.col('artikul').cast(pl.Utf8).fill_null('').alias('_artikul'),
                pl.col('brand').cast(pl.Utf8).fill_null('').alias('_brand'),
                pl.col('multiplicity').cast(pl.Utf8).alias('_multiplicity'),
            ])
            parts_df = parts_df.with_columns(
                description=pl.concat_str([
                    'Артикул: ', pl.col('_artikul'),
                    ', Бренд: ', pl.col('_brand'),
                    ', Кратность: ', pl.col('_multiplicity'), ' шт.'
                ], separator='')
            )
            parts_df = parts_df.drop(['_artikul','_brand','_multiplicity'])
            final_cols = ['artikul_norm','brand_norm','artikul','brand','multiplicity','barcode','length','width','height','weight','image_url','dimensions_str','description']
            parts_df = parts_df.select([pl.col(c) if c in parts_df.columns else pl.lit(None).alias(c) for c in final_cols])
            self.upsert_data('parts_data', parts_df, ['artikul_norm','brand_norm'])
        pbar.progress(1.0)
        time.sleep(1)
        pbar.empty()
        st.success("💾 Обновление завершено.")
    def merge_all_data_parallel(self, paths: Dict[str,str]):
        start = time.time()
        dataframes = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.read_and_prepare_file, p, t): t for t,p in paths.items()}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    df = f.result()
                    if not df.is_empty():
                        dataframes[t] = df
                        st.success(f"✅ {t} загружен: {len(df):,} строк")
                except Exception as e:
                    st.error(f"🚫 Ошибка: {t} {e}")
        if not dataframes:
            st.error("Нет данных для обработки")
            return {}
        self.process_and_load_data(dataframes)
        stats = {
            'processing_time': time.time()-start,
            'total_records': self.get_total_records()
        }
        self.create_indexes()
        st.success(f"Обработка завершена за {stats['processing_time']:.2f} сек")
        return stats
    def get_total_records(self):
        try:
            return self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]
        except:
            return 0
    
    def get_statistics(self):
        stats = {}
        try:
            stats['total_parts'] = self.get_total_records()
            if stats['total_parts']==0:
                return {'total_parts':0,'total_oe':0,'total_brands':0,'top_brands':pl.DataFrame(),'categories':pl.DataFrame()}
            res1 = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()
            stats['total_oe'] = res1[0] if res1 else 0
            res2 = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data WHERE brand IS NOT NULL").fetchone()
            stats['total_brands'] = res2[0] if res2 else 0
            result = self.conn.execute("SELECT brand, COUNT(*) as c FROM parts_data WHERE brand IS NOT NULL GROUP BY brand ORDER BY c DESC LIMIT 10").fetchall()
            if result:
                stats['top_brands'] = pl.DataFrame(result, schema=["brand","c"])
            else:
                stats['top_brands'] = pl.DataFrame(schema=["brand","c"])
            result2 = self.conn.execute("SELECT category, COUNT(*) as c FROM oe_data WHERE category IS NOT NULL GROUP BY category ORDER BY c DESC").fetchall()
            if result2:
                stats['categories'] = pl.DataFrame(result2, schema=["category","c"])
            else:
                stats['categories'] = pl.DataFrame(schema=["category","c"])
        except:
            return {'total_parts':0,'total_oe':0,'total_brands':0,'top_brands':pl.DataFrame(),'categories':pl.DataFrame()}
        return stats
    def load_price_list(self, path):
        df = self.read_and_prepare_file(path,'price')
        if df.is_empty():
            st.warning("Пустой прайс-лист")
            return
        if not all(c in df.columns for c in ['artikul','brand','quantity','price']):
            st.error("Нет обязательных колонок")
            return
        df = df.select(['artikul','brand','quantity','price'])
        df = df.with_columns(
            artikul_norm=self.normalize_key(pl.col('artikul')),
            brand_norm=self.normalize_key(pl.col('brand'))
        )
        self.upsert_data('price_recommendations',df,['artikul_norm','brand_norm'])
        st.success("Цены загружены")
    def set_global_markup(self, percent):
        self.global_markup = percent/100
    def set_brand_markup(self, brand_norm, percent):
        self.brand_markup[brand_norm] = percent/100
    def get_price_with_markup(self, artikul_norm, brand_norm, base_price):
        markup = self.global_markup
        markup += self.brand_markup.get(brand_norm,0)
        return round(base_price*(1+markup),2)
    def build_export_query(self, selected_columns=None):
        # Собственно, тут полный SQL с исключениями и колонками
        standard_description = """Состояние товара: новый (в упаковке).
Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. 
Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей.

В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электрику, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности."""
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
            ("Длинна/Ширина/Высота", 'COALESCE(r.dimensions_str, r.analog_dimensions_str) AS "Длинна/Ширина/Высота"'),
            ("OE номер", 'r.oe_list AS "OE номер"'),
            ("аналоги", 'r.analog_list AS "аналоги"'),
            ("Ссылка на изображение", 'r.image_url AS "Ссылка на изображение"'),
            ("Цена с наценкой", "ROUND(COALESCE(pr.price, 0) * (1 + {self.global_markup} + COALESCE(self.brand_markup.get(r.brand, 0), 0)), 2) AS \"Цена с наценкой\"")
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
                ROW_NUMBER() OVER(PARTITION BY p.artikul_norm, p.brand_norm ORDER BY pd.representative_name DESC NULLS LAST, pd.oe_list DESC NULLS LAST) as rn
            FROM parts_data p
            LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
            LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
            LEFT JOIN AggregatedAnalogData p_analog ON p.artikul_norm = p_analog.artikul_norm AND p.brand_norm = p_analog.brand_norm
        )
        """
        select_exprs = ",\n            ".join(selected_exprs)
        query = ctes + "\nSELECT " + select_exprs + "\nFROM RankedData r\nCROSS JOIN DescriptionTemplate dt\nWHERE r.rn=1\nORDER BY r.\"Бренд\", r.\"Артикул\""
        return query
    
# В основном интерфейс
def main():
    st.title("🚗 AutoParts Catalog - 10+ млн записей")
    catalog = HighVolumeAutoPartsCatalog()
    # Тут далее весь интерфейс, как описано в предыдущем ответе.
    # Вызовы методов для загрузки, экспорта, статистики — через интерфейс.
    
    # Пример: загрузка файла
    if st.sidebar.button("Загрузить данные"):
        # сюда вставьте логику для загрузки файлов, вызова merge_all_data_parallel
        pass
    
    # Экспорт
    if st.sidebar.button("Экспорт данных"):
        catalog.show_export_interface()
    
    # Статистика
    if st.sidebar.button("Посмотреть статистику"):
        stats = catalog.get_statistics()
        # вывод статистики
    
    # Управление
    if st.sidebar.button("Управление данными"):
        # операции удаления
        pass

if __name__ == "__main__":
    main()
