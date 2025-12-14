#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — профессиональная обработка изображений (CLI + Streamlit)
"""

from __future__ import annotations
import argparse
import io
import json
import sys
import zipfile
import tempfile
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import concurrent.futures

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Опциональные зависимости
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except Exception:
    rembg_remove = None
    HAS_REMBG = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    st = None
    HAS_STREAMLIT = False

import logging

# Настройка логгера
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
    return logging.getLogger("photo_processor")

logger = setup_logger()

# Конфигурации
@dataclass
class WatermarkParams:
    threshold: int = 220
    adaptive: bool = True
    block_size: int = 31
    c: int = 10
    min_area: int = 50
    max_area: int = 5000
    radius: int = 5
    use_ns: bool = True

@dataclass
class ProcessingConfig:
    remove_bg: bool = True
    remove_wm: bool = True
    wm_params: WatermarkParams = field(default_factory=WatermarkParams)
    fmt: str = "PNG"
    jpeg_q: int = 95
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    inp: Path = Path("./input")
    outp: Path = Path("./output")


# --- Функции загрузки и сохранения конфигурации ---
def load_config(path: Path = Path("config.json")) -> ProcessingConfig:
    cfg = ProcessingConfig()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(cfg, k):
                    if k in ("inp", "outp") and v is not None:
                        setattr(cfg, k, Path(v))
                    else:
                        setattr(cfg, k, v)
        except Exception as e:
            logger.warning("Не удалось прочесть %s: %s", path, e)
    return cfg

def save_params(params: WatermarkParams, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(params.__dict__, f)

def analyze_image_for_params(pil_img: Image.Image) -> WatermarkParams:
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_color = np.mean(rgb, axis=(0,1))
    contrast_std = np.std(gray)
    threshold = 200 if mean_color[0] > 100 else 220
    adaptive = contrast_std > 15
    return WatermarkParams(
        threshold=threshold,
        adaptive=adaptive,
        block_size=31,
        c=10,
        min_area=50,
        max_area=5000,
        radius=5,
        use_ns=True
    )

# --- Детекция водяных знаков ---
def detect_watermark_auto(pil_img: Image.Image, params: WatermarkParams) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if params.adaptive:
        thr = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            params.block_size,
            params.c
        )
    else:
        _, thr = cv2.threshold(gray, int(params.threshold), 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)

    for c in contours:
        area = cv2.contourArea(c)
        if area < params.min_area or area > params.max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        roi_gray = gray[y:y+h, x:x+w]
        roi_mean = np.mean(roi_gray)
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        bg_roi = gray[y1:y2, x1:x2]
        bg_mean = np.mean(bg_roi)
        contrast = abs(roi_mean - bg_mean)
        if contrast < 15:
            continue
        cv2.drawContours(mask, [c], -1, 255, -1)

    return mask

# --- Основная функция автоматической детекции ---
def detect_watermark(pil_img: Image.Image, auto_mode=True, user_params: Optional[WatermarkParams]=None) -> np.ndarray:
    if auto_mode:
        params = analyze_image_for_params(pil_img)
    else:
        params = user_params or WatermarkParams()
    return detect_watermark_auto(pil_img, params)

# --- Основная обработка водяных знаков ---
def remove_watermark(img_cv: np.ndarray, cfg: ProcessingConfig) -> np.ndarray:
    if not cfg.remove_wm:
        return img_cv
    try:
        bgr = img_cv[..., :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if cfg.wm_params.adaptive:
            thr = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                cfg.wm_params.block_size,
                cfg.wm_params.c
            )
        else:
            _, thr = cv2.threshold(gray, int(cfg.wm_params.threshold), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg.wm_params.min_area or area > cfg.wm_params.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            roi_gray = gray[y:y+h, x:x+w]
            roi_mean = np.mean(roi_gray)
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            bg_roi = gray[y1:y2, x1:x2]
            bg_mean = np.mean(bg_roi)
            contrast = abs(roi_mean - bg_mean)
            if contrast < 15:
                continue
            cv2.drawContours(mask, [c], -1, 255, -1)
        if np.sum(mask) == 0:
            return img_cv
        # Двойной inpaint
        inpainted_telea = cv2.inpaint(bgr, mask, int(cfg.wm_params.radius), cv2.INPAINT_TELEA)
        if cfg.wm_params.use_ns:
            inpainted_ns = cv2.inpaint(bgr, mask, int(cfg.wm_params.radius), cv2.INPAINT_NS)
            # Можно сравнить или выбрать лучший
            # Для простоты оставим Telea
            inpainted = inpainted_telea
        else:
            inpainted = inpainted_telea
        # Восстановление альфа-канала
        if img_cv.shape[2] == 4:
            out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
            out[..., 3] = img_cv[..., 3]
        else:
            out = inpainted
        return out
    except Exception:
        logger.exception("remove_watermark error")
        return img_cv

# --- Вспомогательные функции ---
def resize_cv(img_cv: np.ndarray, w_target: Optional[int], h_target: Optional[int]) -> np.ndarray:
    h, w = img_cv.shape[:2]
    if not w_target and not h_target:
        return img_cv
    if w_target and h_target:
        return cv2.resize(img_cv, (w_target, h_target), interpolation=cv2.INTER_AREA)
    elif w_target:
        scale = w_target / w
        return cv2.resize(img_cv, (w_target, int(h * scale)), interpolation=cv2.INTER_AREA)
    elif h_target:
        scale = h_target / h
        return cv2.resize(img_cv, (int(w * scale), h_target), interpolation=cv2.INTER_AREA)
    return img_cv

def save_cv_image(img_cv: np.ndarray, out_path: Path, cfg: ProcessingConfig) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_cv = resize_cv(img_cv, cfg.target_width, cfg.target_height)
        if img_cv.ndim == 2:
            pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB))
        elif img_cv.shape[2] == 4:
            pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))
        else:
            pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        fmt = cfg.fmt.upper()
        if fmt in ("JPEG", "JPG"):
            pil = pil.convert("RGB")
            pil.save(out_path, "JPEG", quality=int(cfg.jpeg_q))
        else:
            pil.save(out_path, fmt)
        return True
    except Exception:
        logger.exception("Failed to save %s", out_path)
        return False

def process_image(in_path: Path, out_path: Path, cfg: ProcessingConfig) -> Tuple[bool, str]:
    try:
        pil = Image.open(in_path)
        pil = remove_background(pil, cfg)
        img_cv = np.array(pil)
        if img_cv.ndim == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        elif img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        elif img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGRA)
        # Обработка водяных знаков
        img_cv = remove_watermark(img_cv, cfg)
        # Название файла
        out_final = out_path.with_suffix("." + cfg.fmt.lower())
        if save_cv_image(img_cv, out_final, cfg):
            return True, ""
        return False, f"Error saving {out_final}"
    except UnidentifiedImageError:
        return False, f"Unidentified image: {in_path.name}"
    except Exception:
        logger.exception("Error processing %s", in_path)
        return False, str(e)

# --- Вспомогательные функции ---
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def validate_ext(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# --- Batch processing ---
def process_batch(input_dir: Path, output_dir: Path, cfg: ProcessingConfig, max_workers: int=4):
    ensure_dir(input_dir)
    ensure_dir(output_dir)
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and validate_ext(p)]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, p, output_dir / p.stem, cfg): p for p in files}
        for f in concurrent.futures.as_completed(futures):
            p = futures[f]
            try:
                ok, msg = f.result()
                results.append((p, ok, msg))
            except Exception:
                results.append((p, False, "Ошибка при обработке"))
    return results

def zip_results(out_dir: Path, results: List[Tuple[Path, bool, str]], format_ext: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zipf:
        for p, ok, _ in results:
            if not ok:
                continue
            filename = f"{p.stem}.{format_ext}"
            filepath = out_dir / filename
            if filepath.exists():
                zipf.write(filepath, arcname=filename)
    buf.seek(0)
    return buf.read()

# --- Основной CLI ---
def run_cli(argv=None):
    parser = argparse.ArgumentParser(description="Photo Processor Pro CLI")
    parser.add_argument("--input", type=Path, required=True, help="Входная папка")
    parser.add_argument("--output", type=Path, required=True, help="Выходная папка")
    parser.add_argument("--calibrate", action="store_true", help="Калибровка (анализ и сохранение параметров)")
    parser.add_argument("--params_file", type=str, default="detected_params.json", help="Файл с параметрами")
    parser.add_argument("--remove_bg", action="store_true", help="Удалять фон")
    parser.add_argument("--remove_wm", action="store_true", help="Удалять водяные знаки")
    parser.add_argument("--workers", type=int, default=4, help="Потоки")
    args = parser.parse_args(argv)

    cfg = load_config()
    cfg.inp = args.input
    cfg.outp = args.output
    if args.remove_bg:
        cfg.remove_bg = True
    if args.remove_wm:
        cfg.remove_wm = True

    # Анализ или обработка
    if args.calibrate:
        sample_file = next(cfg.inp.glob("*.*"), None)
        if sample_file:
            pil_sample = Image.open(sample_file)
            params = analyze_image_for_params(pil_sample)
            save_params(params, args.params_file)
            print(f"Параметры сохранены в {args.params_file}")
        else:
            print("Нет файлов для анализа")
        return

    # Используем сохранённые параметры
    wm_params = load_params(Path(args.params_file))
    cfg.wm_params = wm_params

    # Обработка
    results = process_batch(cfg.inp, cfg.outp, cfg, max_workers=args.workers)
    for p, ok, msg in results:
        print(f"{'✓' if ok else '✗'} {p.name}: {msg}")

# --- Streamlit UI ---
def run_streamlit():
    if st is None:
        raise RuntimeError("Streamlit не установлен")
    cfg = load_config()

    st.title("Photo Processor Pro — обработка изображений")
    st.sidebar.header("Настройки")
    inp_dir_str = st.sidebar.text_input("Входная папка", str(cfg.inp))
    out_dir_str = st.sidebar.text_input("Выходная папка", str(cfg.outp))

    remove_bg = st.sidebar.checkbox("Удалить фон", value=cfg.remove_bg)
    remove_wm = st.sidebar.checkbox("Удалить водяные знаки", value=cfg.remove_wm)
    wm_adaptive = st.sidebar.checkbox("Адаптивный порог", value=cfg.wm_params.adaptive)
    wm_block_size = st.sidebar.number_input("Размер блока adaptiveThreshold", value=cfg.wm_params.block_size, min_value=3, step=2)
    wm_c = st.sidebar.number_input("Коррекция adaptiveThreshold", value=cfg.wm_params.c, min_value=-100, max_value=100)
    wm_min_area = st.sidebar.number_input("Мин. площадь водяного знака", value=cfg.wm_params.min_area, min_value=1)
    wm_max_area = st.sidebar.number_input("Макс. площадь водяного знака", value=cfg.wm_params.max_area, min_value=1)
    wm_radius = st.sidebar.number_input("Радиус inpaint", value=cfg.wm_params.radius, min_value=1)
    wm_use_ns = st.sidebar.checkbox("Использовать inpaint NS", value=cfg.wm_params.use_ns)
    fmt = st.sidebar.selectbox("Формат", ["PNG", "JPEG", "BMP"], index=["PNG", "JPEG", "BMP"].index(cfg.fmt))
    jpeg_q = st.sidebar.slider("Качество JPEG", 0, 100, cfg.jpeg_q)
    tw = st.sidebar.number_input("Ширина", value=int(cfg.target_width or 0))
    th = st.sidebar.number_input("Высота", value=int(cfg.target_height or 0))
    workers = st.sidebar.number_input("Потоки", value=4, min_value=1)

    uploaded_files = st.sidebar.file_uploader("Загрузить файлы", type=["jpg","jpeg","png","bmp","tiff","webp"], accept_multiple_files=True)
    temp_dir = None
    if uploaded_files:
        temp_dir = Path(tempfile.mkdtemp())
        for f in uploaded_files:
            (temp_dir / f.name).write_bytes(f.read())
        st.sidebar.success(f"Загружено {len(uploaded_files)} файлов в {temp_dir}")

    use_uploaded = st.sidebar.checkbox("Использовать загруженные файлы", value=bool(uploaded_files))
    input_dir = Path(inp_dir_str) if not use_uploaded else (temp_dir or Path("."))
    output_dir = Path(out_dir_str)

    if st.button("Начать обработку"):
        # Создаем локальную копию cfg с новыми параметрами
        cfg_local = ProcessingConfig(
            remove_bg=remove_bg,
            remove_wm=remove_wm,
            wm_params=WatermarkParams(
                threshold=cfg.wm_params.threshold,
                adaptive=wm_adaptive,
                block_size=int(wm_block_size),
                c=int(wm_c),
                min_area=int(wm_min_area),
                max_area=int(wm_max_area),
                radius=int(wm_radius),
                use_ns=wm_use_ns
            ),
            fmt=fmt,
            jpeg_q=jpeg_q,
            target_width=int(tw) if tw > 0 else None,
            target_height=int(th) if th > 0 else None,
            inp=input_dir,
            outp=output_dir
        )

        with st.spinner("Обработка..."):
            results = process_batch(input_dir, output_dir, cfg_local, max_workers=int(workers))
        success_count = sum(1 for _, ok, _ in results if ok)
        fail_count = len(results) - success_count
        st.success(f"Обработка завершена: {success_count} успешно, {fail_count} ошибок")
        zip_data = zip_results(output_dir, results, cfg_local.fmt.lower())
        st.download_button("Скачать все результаты ZIP", zip_data, "results.zip", mime="application/zip")
        st.subheader("Превью результатов")
        cols = st.columns(3)
        for i, (p, ok, _) in enumerate(results):
            if ok:
                img_path = output_dir / f"{p.stem}.{cfg_local.fmt.lower()}"
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        cols[i % 3].image(img, caption=p.name, use_column_width=True)
                    except:
                        pass

    # Очистка временных файлов
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)

# --- Основной запуск ---
def main():
    if HAS_STREAMLIT and len(sys.argv) <= 1:
        run_streamlit()
    else:
        run_cli()

if __name__ == "__main__":
    main()
