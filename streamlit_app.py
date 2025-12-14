#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — улучшенная версия
- Структурированное решение
- CLI + Streamlit интерфейсы
- Многопроцессная обработка
- Конфигурация через dataclass
- Надежное логирование
"""

import argparse
import json
import logging
import os
import sys
import io
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Импорт опциональных модулей
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# --- Конфигурация ---
@dataclass
class Config:
    remove_bg: bool = True
    remove_wm: bool = False
    wm_threshold: int = 220
    wm_radius: int = 5
    fmt: str = "PNG"
    jpeg_q: int = 95
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    input_dir: Path = Path("./input")
    output_dir: Path = Path("./output")

    def validate(self):
        if not self.input_dir.exists():
            raise ValueError(f"Папка входных изображений не существует: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise ValueError(f"Путь входных изображений не папка: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

# --- Логирование ---
logger = logging.getLogger(__name__)
def setup_logger():
    log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

setup_logger()

# --- Обработка изображений ---
def rembg_background(pil_img: Image.Image) -> Image.Image:
    if not HAS_REMBG:
        return pil_img
    try:
        out_bytes = rembg_remove(pil_img)
        if isinstance(out_bytes, (bytes, bytearray)):
            return Image.open(io.BytesIO(out_bytes))
        if isinstance(out_bytes, Image.Image):
            return out_bytes
    except Exception:
        logger.exception("Ошибка rembg")
    return pil_img

def grabcut_background(pil_img: Image.Image) -> Image.Image:
    try:
        img = np.array(pil_img.convert("RGB"))
        h, w = img.shape[:2]
        scale = 512 / max(h, w) if max(h, w) > 512 else 1.0
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros(small.shape[:2], np.uint8)
        rect = (5, 5, small.shape[1] - 10, small.shape[0] - 10)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        small_rgba = cv2.cvtColor(small, cv2.COLOR_RGB2RGBA)
        small_rgba[..., 3] = mask2 * 255
        alpha = cv2.resize(small_rgba[..., 3], (w, h), interpolation=cv2.INTER_LINEAR)
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img_rgba[..., 3] = alpha
        return Image.fromarray(img_rgba)
    except Exception:
        logger.exception("Ошибка grabcut")
        return pil_img

def remove_background(pil_img: Image.Image, config: Config) -> Image.Image:
    if config.remove_bg:
        if HAS_REMBG:
            return rembg_background(pil_img)
        else:
            logger.warning("rembg не установлен, используется grabcut")
    return grabcut_background(pil_img)

def remove_watermark(img_cv: np.ndarray, config: Config) -> np.ndarray:
    if not config.remove_wm:
        return img_cv
    try:
        bgr = img_cv[..., :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, config.wm_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in contours:
            if cv2.contourArea(c) > 50:
                cv2.drawContours(mask, [c], -1, 255, -1)
        if np.any(mask):
            inpainted = cv2.inpaint(bgr, mask, config.wm_radius, cv2.INPAINT_TELEA)
            if img_cv.ndim == 3 and img_cv.shape[2] == 4:
                out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
                out[..., 3] = img_cv[..., 3]
                return out
            return inpainted
        return img_cv
    except Exception:
        logger.exception("Ошибка удаления водяных знаков")
        return img_cv

def resize_image(img_cv: np.ndarray, target_w: Optional[int], target_h: Optional[int]) -> np.ndarray:
    h, w = img_cv.shape[:2]
    if not target_w and not target_h:
        return img_cv
    if target_w and target_h:
        return cv2.resize(img_cv, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if target_w:
        scale = target_w / w
        return cv2.resize(img_cv, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    # target_h
    scale = target_h / h
    return cv2.resize(img_cv, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)

def save_image(img_cv: np.ndarray, out_path: Path, config: Config) -> bool:
    try:
        # Проверка размеров
        if config.target_width and (config.target_width <= 0 or config.target_width > 10000):
            logger.error(f"Недопустимая ширина: {config.target_width}")
            return False
        if config.target_height and (config.target_height <= 0 or config.target_height > 10000):
            logger.error(f"Недопустимая высота: {config.target_height}")
            return False

        # Создать папку
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Изменение размера
        img_cv = resize_image(img_cv, config.target_width, config.target_height)

        # Сохранение
        if config.fmt.upper() == "PNG":
            cv2.imwrite(str(out_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            return True
        else:
            # JPEG
            bgr = img_cv
            if img_cv.ndim == 3 and img_cv.shape[2] == 4:
                bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
            success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_q])
            if success:
                out_path.write_bytes(buf.tobytes())
                return True
        return False
    except Exception:
        logger.exception("Ошибка сохранения")
        return False

# --- Обработка задач ---
def process_single_task(task: Tuple[str, str, Any], config: Config) -> str:
    src_type, name, payload = task
    try:
        if src_type == "disk":
            src_path = config.input_dir / name
            pil_img = Image.open(src_path).convert("RGBA")
        else:
            data = payload
            if hasattr(data, "read"):
                buf = data.read()
            else:
                buf = data
            pil_img = Image.open(io.BytesIO(buf)).convert("RGBA")

        # Обработка фона
        processed_pil = remove_background(pil_img, config)

        # Конвертация в OpenCV
        img_cv = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGBA2BGRA)

        # Удаление водяных знаков
        img_cv = remove_watermark(img_cv, config)

        ext = ".png" if config.fmt.upper() == "PNG" else ".jpg"
        out_name = Path(name).stem + ext
        out_path = config.output_dir / out_name

        if save_image(img_cv, out_path, config):
            return f"✅ {name} -> {out_name}"
        else:
            return f"❌ Ошибка сохранения {name}"
    except UnidentifiedImageError:
        return f"❌ Не удалось открыть {name} (не изображение/повреждён)"
    except Exception:
        logger.exception(f"Ошибка обработки {name}")
        return f"❌ Ошибка обработки {name}"

# --- Обработка батчей с многопроцессорностью ---
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_batch(
    config: Config,
    filenames: Optional[List[str]] = None,
    uploaded_files: Optional[List[Any]] = None
) -> List[str]:

    # Валидация путей
    config.validate()

    tasks: List[Tuple[str, str, Any]] = []

    # Загруженные файлы
    if uploaded_files:
        for f in uploaded_files:
            name = getattr(f, "name", None)
            if name and validate_file_extension(Path(name)):
                tasks.append(("uploaded", name, f))
    else:
        # Файлы на диске
        for p in config.input_dir.iterdir():
            if p.is_file() and validate_file_extension(p):
                if not filenames or p.name in filenames:
                    tasks.append(("disk", p.name, None))

    if not tasks:
        return ["[ПРЕДУПРЕЖДЕНИЕ] Нет изображений для обработки"]

    logs: List[str] = []

    max_workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_task, task, config) for task in tasks]
        for future in as_completed(futures):
            try:
                res = future.result()
                logs.append(res)
                logger.info(res)
            except Exception:
                logger.exception("Ошибка в воркере")
                logs.append("❌ В процессе возникла ошибка")
    return logs

# --- Проверка расширения файла ---
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def validate_file_extension(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS

# --- CLI ---
def run_cli():
    parser = argparse.ArgumentParser(description="Photo Processor Pro CLI")
    parser.add_argument("--input", "-i", default="./input", help="Папка с изображениями")
    parser.add_argument("--output", "-o", default=None, help="Куда сохранять")
    parser.add_argument("--no-bg", dest="remove_bg", action="store_false", help="Отключить удаление фона")
    parser.add_argument("--wm", dest="remove_wm", action="store_true", help="Удалять водяные знаки")
    parser.add_argument("--wm-threshold", type=int, default=220, help="Порог для удаления водяных знаков")
    parser.add_argument("--wm-radius", type=int, default=5, help="Радиус inpaint")
    parser.add_argument("--fmt", choices=["PNG", "JPEG"], default="PNG", help="Формат сохранения")
    parser.add_argument("--jpeg-q", type=int, default=95, help="Качество JPEG")
    parser.add_argument("--width", type=int, default=None, help="Ширина")
    parser.add_argument("--height", type=int, default=None, help="Высота")
    parser.add_argument("--config", default="config.json", help="Путь к файлу конфигурации")

    args = parser.parse_args()

    # Загрузка конфигурации
    cfg_data = {}
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
    except Exception:
        logger.warning("Нет файла конфигурации или ошибка чтения, используют параметры по умолчанию.")

    config = Config(
        input_dir=Path(args.input).expanduser().resolve(),
        output_dir=Path(args.output).expanduser().resolve() if args.output else Path("./output").resolve(),
        remove_bg=args.remove_bg,
        remove_wm=args.remove_wm,
        wm_threshold=args.wm_threshold,
        wm_radius=args.wm_radius,
        fmt=args.fmt,
        jpeg_q=args.jpeg_q,
        target_width=args.width,
        target_height=args.height
    )

    try:
        config.validate()
    except Exception as e:
        logger.error(f"Ошибка конфигурации: {e}")
        sys.exit(1)

    logs = process_batch(config)
    for log in logs:
        print(log)

# --- Streamlit UI ---
def run_streamlit():
    if not HAS_STREAMLIT:
        print("Streamlit не установлен. Установите: pip install streamlit")
        return
    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("🖼️ Photo Processor Pro")

    # Настройки
    with st.sidebar:
        st.header("Настройки обработки")
        remove_bg = st.checkbox("Удалить фон", value=True)
        remove_wm = st.checkbox("Удалить водяные знаки", value=False)
        wm_threshold = st.slider("Порог для водяных знаков", 0, 255, 220)
        wm_radius = st.slider("Радиус inpaint", 1, 20, 5)
        fmt = st.selectbox("Формат вывода", ["PNG", "JPEG"])
        jpeg_q = st.slider("Качество JPEG", 1, 100, 95) if fmt == "JPEG" else 95
        width = st.number_input("Ширина (px)", min_value=1, max_value=10000, value=None, step=1)
        height = st.number_input("Высота (px)", min_value=1, max_value=10000, value=None, step=1)

    # Загрузка файлов
    uploaded_files = st.file_uploader(
        "Выберите изображения",
        accept_multiple_files=True,
        type=list(SUPPORTED_EXTENSIONS)
    )

    if uploaded_files:
        st.subheader("Предварительный просмотр")
        cols = st.columns(5)
        for idx, file in enumerate(uploaded_files[:10]):
            with cols[idx % 5]:
                try:
                    img = Image.open(file).convert("RGBA")
                    st.image(img, caption=file.name, use_column_width=True)
                except:
                    st.write("❌ Не удалось отобразить изображение.")

        if st.button("Начать обработку"):
            with st.spinner("Обработка..."):
                temp_dir = Path("./temp_uploaded")
                temp_dir.mkdir(exist_ok=True)
                for file in uploaded_files:
                    with open(temp_dir / file.name, "wb") as f:
                        f.write(file.read())

                config = Config(
                    remove_bg=remove_bg,
                    remove_wm=remove_wm,
                    wm_threshold=wm_threshold,
                    wm_radius=wm_radius,
                    fmt=fmt,
                    jpeg_q=jpeg_q,
                    target_width=width,
                    target_height=height,
                    input_dir=temp_dir,
                    output_dir=Path("./streamlit_output")
                )

                # Обработка
                logs = process_batch(config)
                for log in logs:
                    if "✅" in log:
                        st.success(log)
                    elif "❌" in log:
                        st.error(log)
                    else:
                        st.info(log)

                # Создать ZIP архива
                try:
                    zip_path = create_zip_of_output(str(config.output_dir))
                    with open(zip_path, "rb") as f:
                        st.download_button("Скачать ZIP-архив", data=f, file_name=zip_path.name, mime="application/zip")
                except Exception as e:
                    st.error(f"Ошибка создания архива: {e}")

# --- Утилита для архива ---
def create_zip_of_output(output_dir: str, zip_name: Optional[str] = None) -> Path:
    outp = Path(output_dir).expanduser().resolve()
    if not outp.exists() or not outp.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {outp}")
    base_name = zip_name or f"{outp.name}_results"
    tmp_dir = Path(tempfile.gettempdir())
    zip_base = tmp_dir / base_name
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(outp))
    return Path(zip_path)

# --- Основной вход ---
def main():
    parser = argparse.ArgumentParser(description="Photo Processor Pro")
    parser.add_argument("--mode", choices=["cli", "streamlit"], default="cli")
    args = parser.parse_args()

    if args.mode == "streamlit":
        run_streamlit()
    else:
        run_cli()

if __name__ == "__main__":
    main()
