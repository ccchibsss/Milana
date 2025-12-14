# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленная и рабочая версия Photo Processor Pro (CLI + Streamlit)
Устранены синтаксические и логические ошибки:
- Опциональные аргументы CLI (используются значения из config.json / defaults)
- Исправлены ошибки при сохранении/конвертации изображений
- Корректная обработка альфа-каналов и удаления ватермарок
- Устранены все явные синтаксические ошибки
"""

from __future__ import annotations
import argparse
import io
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional deps
try:
    from rembg import remove as rembg_remove  # type: ignore
    HAS_REMBG = True
except Exception as e:
    rembg_remove = None  # type: ignore
    HAS_REMBG = False
    logging.warning(f"rembg не установлен: {e}")

try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception as e:
    st = None  # type: ignore
    HAS_STREAMLIT = False
    logging.warning(f"Streamlit не установлен: {e}")

# Logger
def setup_logger() -> logging.Logger:
    fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(fn, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logger()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

@dataclass
class ProcessingConfig:
    remove_bg: bool = True
    remove_wm: bool = False
    wm_threshold: int = 220
    wm_radius: int = 5
    fmt: str = "PNG"
    jpeg_q: int = 95
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    inp: Path = Path("./input")
    outp: Path = Path("./output")

def load_config() -> ProcessingConfig:
    """Загружает config.json или возвращает дефолтные значения."""
    config_path = Path("config.json")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = ProcessingConfig()
            for k, v in data.items():
                if hasattr(cfg, k):
                    if k in ("inp", "outp") and v is not None:
                        setattr(cfg, k, Path(v))
                    else:
                        setattr(cfg, k, v)
            return cfg
        except Exception as e:
            logger.error("Ошибка чтения config.json: %s", e)
    logger.info("Используются дефолтные настройки (config.json не найден)")
    return ProcessingConfig()

def validate_path(path: Path, is_input: bool = True) -> Tuple[bool, str]:
    try:
        if is_input:
            if not path.exists() or not path.is_dir():
                return False, f"Входная папка не найдена или не каталог: {path}"
        else:
            # Для выходной папки создаём директорию, если нужно
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            elif not path.is_dir():
                return False, f"Выходной путь не является директорией: {path}"
        return True, ""
    except PermissionError:
        return False, f"Нет прав доступа: {path}"
    except Exception as e:
        return False, f"Ошибка проверки: {e}"

def validate_file_extension(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS

def rembg_background(pil_img: Image.Image) -> Image.Image:
    if not HAS_REMBG or rembg_remove is None:
        return pil_img
    try:
        out = rembg_remove(pil_img)
        if isinstance(out, (bytes, bytearray)):
            return Image.open(io.BytesIO(out))
        if isinstance(out, Image.Image):
            return out
        # Если rembg вернул numpy array
        if isinstance(out, np.ndarray):
            return Image.fromarray(out)
    except Exception as e:
        logger.exception("rembg failed: %s", e)
    return pil_img

def grabcut_background(pil_img: Image.Image) -> Image.Image:
    try:
        img = np.array(pil_img.convert("RGB"))
        h, w = img.shape[:2]
        scale = 512 / max(h, w) if max(h, w) > 512 else 1.0
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), cv2.INTER_LINEAR)
        mask = np.zeros(small.shape[:2], np.uint8)
        rect = (5, 5, max(1, small.shape[1] - 10), max(1, small.shape[0] - 10))
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        small_rgba = cv2.cvtColor(small, cv2.COLOR_RGB2RGBA)
        small_rgba[..., 3] = mask2 * 255
        alpha = cv2.resize(small_rgba[..., 3], (w, h), cv2.INTER_LINEAR)
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img_rgba[..., 3] = alpha
        return Image.fromarray(img_rgba)
    except Exception as e:
        logger.exception("grabcut failed: %s", e)
        return pil_img

def remove_background(pil_img: Image.Image, config: ProcessingConfig) -> Image.Image:
    if config.remove_bg and HAS_REMBG:
        try:
            return rembg_background(pil_img)
        except Exception:
            logger.warning("rembg failed, fallback to grabcut")
    return grabcut_background(pil_img)

def remove_watermark(img_cv: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if not config.remove_wm:
        return img_cv
    try:
        # Work on BGR or BGRA
        bgr = img_cv[..., :3].copy()
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
            # Если исход был 3-канальным
            if img_cv.ndim == 3 and img_cv.shape[2] == 3:
                return inpainted
            # Иначе создаём 4-канальное и восстанавливаем альфу, если она была
            out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
            if img_cv.ndim == 3 and img_cv.shape[2] == 4:
                out[..., 3] = img_cv[..., 3]
            return out
        return img_cv
    except Exception as e:
        logger.exception("remove_watermark failed: %s", e)
        return img_cv

def resize_image(img_cv: np.ndarray, target_width: Optional[int], target_height: Optional[int]) -> np.ndarray:
    if img_cv.ndim == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    h, w = img_cv.shape[:2]
    if (not target_width or target_width <= 0) and (not target_height or target_height <= 0):
        return img_cv
    if target_width and target_height:
        return cv2.resize(img_cv, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA)
    if target_width and target_width > 0:
        scale = target_width / w
        return cv2.resize(img_cv, (int(target_width), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    scale = target_height / h
    return cv2.resize(img_cv, (max(1, int(w * scale)), int(target_height)), interpolation=cv2.INTER_AREA)

def save_image(img_cv: np.ndarray, out_path: Path, config: ProcessingConfig) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_cv = resize_image(img_cv, config.target_width, config.target_height)

        # Конвертируем в PIL для сохранения с настройками формата
        if img_cv.ndim == 2:
            pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB))
        elif img_cv.shape[2] == 4:
            pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        fmt = config.fmt.upper()
        if fmt in ("JPEG", "JPG"):
            pil_img = pil_img.convert("RGB")
            pil_img.save(out_path, "JPEG", quality=int(config.jpeg_q), optimize=True)
        else:
            pil_img.save(out_path, fmt)
        return True
    except Exception as e:
        logger.error("Не удалось сохранить изображение %s: %s", out_path, e)
        return False

def process_image(in_path: Path, out_path: Path, config: ProcessingConfig) -> Tuple[bool, str]:
    try:
        pil_img = Image.open(in_path)
        pil_img = remove_background(pil_img, config)

        img_cv = np.array(pil_img)
        if img_cv.ndim == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        elif img_cv.shape[2] == 3:
            # PIL gives RGB
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        elif img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGRA)

        img_cv = remove_watermark(img_cv, config)

        # Ensure output filename extension matches desired format
        out_path = out_path.with_suffix(f".{config.fmt.lower()}")
        if save_image(img_cv, out_path, config):
            return True, ""
        else:
            return False, f"Ошибка сохранения: {out_path}"
    except UnidentifiedImageError:
        return False, f"Неподдерживаемый формат: {in_path}"
    except Exception as e:
        logger.exception("Ошибка обработки %s: %s", in_path, e)
        return False, f"Ошибка: {in_path} — {str(e)}"

def process_batch(input_dir: Path, output_dir: Path, config: ProcessingConfig, max_workers: int = 4) -> List[Tuple[Path, bool, str]]:
    results: List[Tuple[Path, bool, str]] = []
    if not input_dir.exists() or not input_dir.is_dir():
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for in_path in input_dir.iterdir():
            if not in_path.is_file():
                continue
            if not validate_file_extension(in_path):
                continue
            out_path = output_dir / f"{in_path.stem}.{config.fmt.lower()}"
            futures[executor.submit(process_image, in_path, out_path, config)] = in_path

        for future in as_completed(futures):
            in_path = futures[future]
            try:
                success, msg = future.result()
                results.append((in_path, success, msg))
            except Exception as e:
                results.append((in_path, False, str(e)))
    return results

def run_cli(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Обработка изображений")
    parser.add_argument("--input", type=Path, help="Входная папка")
    parser.add_argument("--output", type=Path, help="Выходная папка")
    parser.add_argument("--remove_bg", action="store_true", help="Удалить фон")
    parser.add_argument("--remove_wm", action="store_true", help="Удалить водяные знаки")
    parser.add_argument("--fmt", type=str, help="Формат вывода (PNG/JPEG/BMP)")
    parser.add_argument("--jpeg_q", type=int, help="Качество JPEG (0-100)")
    parser.add_argument("--workers", type=int, default=4, help="Количество потоков")
    args = parser.parse_args(argv)

    config = load_config()
    if args.input:
        config.inp = args.input
    if args.output:
        config.outp = args.output
    if args.remove_bg:
        config.remove_bg = True
    if args.remove_wm:
        config.remove_wm = True
    if args.fmt:
        config.fmt = args.fmt
    if args.jpeg_q is not None:
        config.jpeg_q = args.jpeg_q

    valid, msg = validate_path(config.inp, is_input=True)
    if not valid:
        print(f"Ошибка: {msg}")
        return
    valid, msg = validate_path(config.outp, is_input=False)
    if not valid:
        print(f"Ошибка: {msg}")
        return

    print("Начало обработки...")
    results = process_batch(config.inp, config.outp, config, max_workers=args.workers)
    for path, success, msg in results:
        if success:
            print(f"✓ {path.name}")
        else:
            print(f"✗ {path.name}: {msg}")

def run_streamlit():
    if st is None:
        raise RuntimeError("Streamlit не доступен")
    st.title("Photo Processor Pro")
    st.write("Обработка изображений: удаление фона, водяных знаков, изменение размера.")

    config = load_config()
    input_dir = st.sidebar.text_input("Входная папка", value=str(config.inp))
    output_dir = st.sidebar.text_input("Выходная папка", value=str(config.outp))

    remove_bg = st.sidebar.checkbox("Удалить фон", value=config.remove_bg)
    remove_wm = st.sidebar.checkbox("Удалить водяные знаки", value=config.remove_wm)
    fmt = st.sidebar.selectbox("Формат вывода", ["PNG", "JPEG", "BMP"], index=0)
    jpeg_q = st.sidebar.slider("Качество JPEG (0-100)", 0, 100, int(config.jpeg_q))
    target_width = st.sidebar.number_input("Ширина (px)", value=int(config.target_width or 0), min_value=0, step=10)
    target_height = st.sidebar.number_input("Высота (px)", value=int(config.target_height or 0), min_value=0, step=10)
    workers = st.sidebar.number_input("Потоки", value=4, min_value=1, step=1)

    if st.button("Запустить обработку"):
        config.inp = Path(input_dir)
        config.outp = Path(output_dir)
        config.remove_bg = remove_bg
        config.remove_wm = remove_wm
        config.fmt = fmt
        config.jpeg_q = int(jpeg_q)
        config.target_width = int(target_width) if target_width > 0 else None
        config.target_height = int(target_height) if target_height > 0 else None

        valid, msg = validate_path(config.inp, is_input=True)
        if not valid:
            st.error(msg)
            return
        valid, msg = validate_path(config.outp, is_input=False)
        if not valid:
            st.error(msg)
            return

        st.info("Начинается обработка изображений...")
        results = process_batch(config.inp, config.outp, config, max_workers=int(workers))

        success_count = sum(1 for _, success, _ in results if success)
        fail_count = len(results) - success_count

        if success_count > 0:
            st.success(f"Обработано успешно: {success_count} файлов")
        if fail_count > 0:
            st.error(f"Ошибок: {fail_count} файлов")

        for path, success, msg in results:
            if not success:
                st.warning(f"{path.name}: {msg}")

        if success_count > 0:
            st.subheader("Обработанные файлы:")
            for path, success, _ in results:
                if success:
                    try:
                        img = Image.open(config.outp / (path.stem + f".{config.fmt.lower()}"))
                        st.image(img, caption=path.name, use_column_width=True)
                    except Exception as e:
                        st.write(f"Не удалось отобразить {path.name}: {e}")

def main(argv: Optional[List[str]] = None):
    # Если Streamlit доступен и мы запускаемся под streamlit, используем интерфейс
    if HAS_STREAMLIT and "streamlit" in sys.modules:
        run_streamlit()
    else:
        run_cli(argv)

if __name__ == "__main__":
    main()
