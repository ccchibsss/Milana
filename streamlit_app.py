# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — профессиональная обработка изображений (CLI + Streamlit)
Функции:
- Удаление фона (rembg при наличии, иначе GrabCut — фон прозрачный, сохраняется передний план)
- Простое удаление водяных знаков (inpaint)
- Массовая и одиночная обработка
- Загрузка файлов через Streamlit, скачивание результата и ZIP
- Логирование и автоматическое создание папок
Все сообщения и комментарии — на русском.
"""

from __future__ import annotations
import argparse
import io
import json
import logging
import sys
import zipfile
import tempfile
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Опциональные зависимости
try:
    from rembg import remove as rembg_remove  # type: ignore
    HAS_REMBG = True
except Exception:
    rembg_remove = None  # type: ignore
    HAS_REMBG = False

try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception:
    st = None  # type: ignore
    HAS_STREAMLIT = False

# Логгер
def setup_logger() -> logging.Logger:
    fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(fn, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("photo_processor")

logger = setup_logger()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_TYPES = [e.lstrip(".") for e in SUPPORTED_EXTENSIONS]

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
            logger.warning("Не удалось прочесть config.json: %s", e)
    return cfg

def ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error("Не удалось создать папку %s: %s", path, e)
        raise

def validate_ext(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXTENSIONS

# GrabCut: сохраняем передний план, фон делаем прозрачным
def grabcut_foreground_rgba(pil_img: Image.Image, downscale_max: int = 512) -> Image.Image:
    try:
        rgb = np.array(pil_img.convert("RGB"))
        h, w = rgb.shape[:2]
        scale = downscale_max / max(h, w) if max(h, w) > downscale_max else 1.0
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)

        # Инициализация маски: вероятный фон
        mask = np.full(small.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
        # Прямоугольник немного внутри границ
        rx, ry = max(1, sw // 50), max(1, sh // 50)
        rw, rh = max(2, sw - 2 * rx), max(2, sh - 2 * ry)
        rect = (rx, ry, rw, rh)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        fg_small = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        small_rgba = cv2.cvtColor(small, cv2.COLOR_RGB2RGBA)
        small_rgba[..., 3] = fg_small

        alpha_full = cv2.resize(small_rgba[..., 3], (w, h), interpolation=cv2.INTER_LINEAR)
        img_rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        img_rgba[..., 3] = alpha_full
        return Image.fromarray(img_rgba)
    except Exception as e:
        logger.exception("grabcut failed: %s", e)
        return pil_img.convert("RGBA")

def rembg_background(pil_img: Image.Image) -> Image.Image:
    if not HAS_REMBG or rembg_remove is None:
        return pil_img
    try:
        out = rembg_remove(pil_img)
        if isinstance(out, Image.Image):
            return out
        if isinstance(out, (bytes, bytearray)):
            return Image.open(io.BytesIO(out))
        if isinstance(out, np.ndarray):
            return Image.fromarray(out)
    except Exception as e:
        logger.warning("rembg failed: %s", e)
    return pil_img

def remove_background(pil_img: Image.Image, cfg: ProcessingConfig) -> Image.Image:
    if cfg.remove_bg and HAS_REMBG:
        try:
            return rembg_background(pil_img)
        except Exception:
            logger.info("Fallback to GrabCut")
    return grabcut_foreground_rgba(pil_img)

def remove_watermark(img_cv: np.ndarray, cfg: ProcessingConfig) -> np.ndarray:
    if not cfg.remove_wm:
        return img_cv
    try:
        bgr = img_cv[..., :3].copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, int(cfg.wm_threshold), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in contours:
            if cv2.contourArea(c) > 50:
                cv2.drawContours(mask, [c], -1, 255, -1)
        if np.any(mask):
            inpainted = cv2.inpaint(bgr, mask, int(cfg.wm_radius), cv2.INPAINT_TELEA)
            if img_cv.ndim == 3 and img_cv.shape[2] == 3:
                return inpainted
            out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
            if img_cv.ndim == 3 and img_cv.shape[2] == 4:
                out[..., 3] = img_cv[..., 3]
            return out
        return img_cv
    except Exception as e:
        logger.exception("remove_watermark failed: %s", e)
        return img_cv

def resize_cv(img_cv: np.ndarray, w_target: Optional[int], h_target: Optional[int]) -> np.ndarray:
    if img_cv.ndim == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    h, w = img_cv.shape[:2]
    if (not w_target or w_target <= 0) and (not h_target or h_target <= 0):
        return img_cv
    if w_target and h_target:
        return cv2.resize(img_cv, (int(w_target), int(h_target)), interpolation=cv2.INTER_AREA)
    if w_target:
        scale = float(w_target) / w
        return cv2.resize(img_cv, (int(w_target), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    scale = float(h_target) / h
    return cv2.resize(img_cv, (max(1, int(w * scale)), int(h_target)), interpolation=cv2.INTER_AREA)

def save_cv_image(img_cv: np.ndarray, out_path: Path, cfg: ProcessingConfig) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_cv = resize_cv(img_cv, cfg.target_width, cfg.target_height)
        # Конвертация в PIL для корректного сохранения форматов и альфа
        if img_cv.ndim == 2:
            pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB))
        elif img_cv.shape[2] == 4:
            pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))
        else:
            pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        fmt = cfg.fmt.upper()
        if fmt in ("JPEG", "JPG"):
            pil = pil.convert("RGB")
            pil.save(out_path, "JPEG", quality=int(cfg.jpeg_q), optimize=True)
        else:
            pil.save(out_path, fmt)
        return True
    except Exception as e:
        logger.error("Не удалось сохранить %s: %s", out_path, e)
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

        img_cv = remove_watermark(img_cv, cfg)
        out_final = out_path.with_suffix(f".{cfg.fmt.lower()}")
        if save_cv_image(img_cv, out_final, cfg):
            return True, ""
        return False, f"Ошибка сохранения: {out_final}"
    except UnidentifiedImageError:
        return False, f"Неподдерживаемый файл: {in_path.name}"
    except Exception as e:
        logger.exception("Ошибка обработки %s: %s", in_path, e)
        return False, str(e)

def process_batch(input_dir: Path, output_dir: Path, cfg: ProcessingConfig, max_workers: int = 4) -> List[Tuple[Path, bool, str]]:
    results: List[Tuple[Path, bool, str]] = []
    ensure_dir(input_dir)
    ensure_dir(output_dir)
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and validate_ext(p)]
    if not files:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_image, p, output_dir / p.stem, cfg): p for p in files}
        for f in as_completed(futs):
            p = futs[f]
            try:
                ok, msg = f.result()
                results.append((p, ok, msg))
            except Exception as e:
                results.append((p, False, str(e)))
    return results

def zip_results_bytes(output_dir: Path, results: List[Tuple[Path, bool, str]], fmt: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p, ok, _ in results:
            if not ok:
                continue
            fn = f"{p.stem}.{fmt.lower()}"
            fp = output_dir / fn
            if fp.exists():
                z.write(fp, arcname=fn)
    buf.seek(0)
    return buf.read()

# CLI
def run_cli(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Photo Processor Pro — обработка изображений")
    parser.add_argument("--input", type=Path, help="Входная папка")
    parser.add_argument("--output", type=Path, help="Выходная папка")
    parser.add_argument("--remove_bg", action="store_true", help="Удалять фон")
    parser.add_argument("--remove_wm", action="store_true", help="Удалять водяные знаки")
    parser.add_argument("--fmt", type=str, choices=["PNG", "JPEG", "BMP"], help="Формат вывода")
    parser.add_argument("--jpeg_q", type=int, help="Качество JPEG 0-100")
    parser.add_argument("--workers", type=int, default=4, help="Потоки")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.input:
        cfg.inp = args.input
    if args.output:
        cfg.outp = args.output
    if args.remove_bg:
        cfg.remove_bg = True
    if args.remove_wm:
        cfg.remove_wm = True
    if args.fmt:
        cfg.fmt = args.fmt
    if args.jpeg_q is not None:
        cfg.jpeg_q = args.jpeg_q

    ensure_dir(cfg.inp)
    ensure_dir(cfg.outp)

    logger.info("Начало пакетной обработки: %s -> %s", cfg.inp, cfg.outp)
    results = process_batch(cfg.inp, cfg.outp, cfg, max_workers=args.workers)
    for p, ok, msg in results:
        if ok:
            print(f"✓ {p.name}")
        else:
            print(f"✗ {p.name}: {msg}")

# Streamlit UI (если доступен)
def run_streamlit():
    if st is None:
        raise RuntimeError("Streamlit не установлен")
    cfg0 = load_config()
    st.title("Photo Processor Pro — профессиональная обработка изображений")
    st.sidebar.header("Настройки")
    inp_text = st.sidebar.text_input("Входная папка (сервер)", value=str(cfg0.inp))
    out_text = st.sidebar.text_input("Выходная папка (сервер)", value=str(cfg0.outp))
    remove_bg = st.sidebar.checkbox("Удалить фон", value=cfg0.remove_bg)
    remove_wm = st.sidebar.checkbox("Удалить водяные знаки", value=cfg0.remove_wm)
    fmt = st.sidebar.selectbox("Формат", ["PNG", "JPEG", "BMP"], index=["PNG","JPEG","BMP"].index(cfg0.fmt if cfg0.fmt in ("PNG","JPEG","BMP") else "PNG"))
    jpeg_q = st.sidebar.slider("Качество JPEG", 0, 100, int(cfg0.jpeg_q))
    tw = st.sidebar.number_input("Ширина (px)", value=int(cfg0.target_width or 0), min_value=0, step=10)
    th = st.sidebar.number_input("Высота (px)", value=int(cfg0.target_height or 0), min_value=0, step=10)
    workers = st.sidebar.number_input("Потоки", value=4, min_value=1, step=1)

    uploaded = st.sidebar.file_uploader("Загрузить файлы (опционально)", type=SUPPORTED_TYPES, accept_multiple_files=True)
    temp_dir = None
    if uploaded:
        temp_dir = Path(tempfile.mkdtemp(prefix="pp_upload_"))
        for f in uploaded:
            (temp_dir / f.name).write_bytes(f.getbuffer())
        st.sidebar.success(f"Загружено {len(uploaded)} файлов в {temp_dir}")

    use_uploaded = st.sidebar.checkbox("Использовать загруженные файлы", value=bool(uploaded))
    inp_dir = Path(inp_text) if not use_uploaded else (temp_dir or Path("."))
    out_dir = Path(out_text)

    ensure_dir(inp_dir)
    ensure_dir(out_dir)

    st.write(f"Входная папка: {inp_dir}")
    files = sorted([p for p in inp_dir.iterdir() if p.is_file() and validate_ext(p)], key=lambda x: x.name)
    if not files:
        st.info("Нет изображений в входной папке")
    file_names = [p.name for p in files]
    sel = st.selectbox("Выбрать файл", options=["(ничего)"] + file_names)

    cfg = ProcessingConfig(remove_bg=remove_bg, remove_wm=remove_wm, fmt=fmt, jpeg_q=int(jpeg_q),
                           target_width=int(tw) if tw > 0 else None, target_height=int(th) if th > 0 else None,
                           inp=inp_dir, outp=out_dir)

    if st.button("Обработать выбранный файл") and sel and sel != "(ничего)":
        in_path = inp_dir / sel
        out_path = out_dir / in_path.stem
        ok, msg = process_image(in_path, out_path, cfg)
        if ok:
            out_file = out_path.with_suffix(f".{cfg.fmt.lower()}")
            st.success("Готово")
            st.image(out_file, caption=out_file.name, use_column_width=True)
            st.download_button("Скачать файл", out_file.read_bytes(), file_name=out_file.name, mime="application/octet-stream")
        else:
            st.error(msg)

    if st.button("Обработать все и скачать ZIP"):
        results = process_batch(cfg.inp, cfg.outp, cfg, max_workers=int(workers))
        ok_count = sum(1 for _, ok, _ in results if ok)
        st.write(f"Готово: {ok_count} файлов")
        if ok_count > 0:
            zipb = zip_results_bytes(cfg.outp, results, cfg.fmt)
            st.download_button("Скачать ZIP", zipb, file_name=f"results_{cfg.fmt.lower()}.zip", mime="application/zip")
        for p, ok, m in results:
            if not ok:
                st.warning(f"{p.name}: {m}")

    if uploaded and st.button("Обработать загруженные и скачать ZIP"):
        if not temp_dir:
            st.error("Нет загруженных файлов")
        else:
            results = process_batch(temp_dir, out_dir, cfg, max_workers=int(workers))
            ok_count = sum(1 for _, ok, _ in results if ok)
            st.write(f"Готово: {ok_count} файлов")
            if ok_count > 0:
                zipb = zip_results_bytes(out_dir, results, cfg.fmt)
                st.download_button("Скачать ZIP", zipb, file_name=f"uploaded_results_{cfg.fmt.lower()}.zip", mime="application/zip")
            for p, ok, m in results:
                if not ok:
                    st.warning(f"{p.name}: {m}")
            try:
                if temp_dir and temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception:
                pass

def main(argv: Optional[List[str]] = None):
    # Если Streamlit импортирован — используем UI, иначе CLI
    if HAS_STREAMLIT and "streamlit" in sys.modules:
        run_streamlit()
    else:
        run_cli(argv)

if __name__ == "__main__":
    main()
