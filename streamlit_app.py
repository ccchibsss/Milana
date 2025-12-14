# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленная и упрощённая версия Photo Processor Pro (CLI + Streamlit)
- Убраны синтаксические ошибки
- Стабилизирован CLI / Streamlit режим
- Добавлен скачиваемый ZIP в Streamlit
"""

from __future__ import annotations
import argparse
import io
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Optional deps
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

# logger
def setup_logger() -> logging.Logger:
    fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(fn, encoding="utf-8"), logging.StreamHandler()],
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

def validate_path(path: Path, is_input: bool = True) -> Tuple[bool, str]:
    try:
        if is_input:
            if not path.exists() or not path.is_dir():
                return False, f"Входная папка не найдена или не каталог: {path}"
        else:
            # output: allow not exists (we will create), but check parent permission
            parent = path if path.exists() else path.parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
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
    except Exception:
        logger.exception("rembg failed")
    return pil_img

def grabcut_background(pil_img: Image.Image) -> Image.Image:
    try:
        img = np.array(pil_img.convert("RGB"))
        h, w = img.shape[:2]
        scale = 512 / max(h, w) if max(h, w) > 512 else 1.0
        small = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_LINEAR)
        mask = np.zeros(small.shape[:2], np.uint8)
        rect = (5, 5, small.shape[1] - 10, small.shape[0] - 10)
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
    except Exception:
        logger.exception("grabcut failed")
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
            if img_cv.ndim == 3:
                return inpainted
            # preserve alpha
            out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
            out[..., 3] = img_cv[..., 3]
            return out
        return img_cv
    except Exception:
        logger.exception("remove_watermark failed")
        return img_cv

def resize_image(img_cv: np.ndarray, target_width: Optional[int], target_height: Optional[int]) -> np.ndarray:
    h, w = img_cv.shape[:2]
    if (not target_width or target_width <= 0) and (not target_height or target_height <= 0):
        return img_cv
    if target_width and target_height:
        return cv2.resize(img_cv, (target_width, target_height), interpolation=cv2.INTER_AREA)
    if target_width and target_width > 0:
        scale = target_width / w
        return cv2.resize(img_cv, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    scale = target_height / h
    return cv2.resize(img_cv, (int(w * scale), target_height), interpolation=cv2.INTER_AREA)

def save_image(img_cv: np.ndarray, out_path: Path, config: ProcessingConfig) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_cv = resize_image(img_cv, config.target_width, config.target_height)
        if config.fmt.upper() == "PNG":
            return bool(cv2.imwrite(str(out_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3]))
        # JPEG: drop alpha
        bgr = img_cv
        if img_cv.ndim == 3 and img_cv.shape[2] == 4:
            bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        success, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_q)])
        if success:
            out_path.write_bytes(buf.tobytes())
            return True
        return False
    except Exception:
        logger.exception("save_image failed")
        return False

def process_single_task(src_path: Path, out_dir: Path, config: ProcessingConfig) -> str:
    name = src_path.name
    try:
        pil = Image.open(src_path).convert("RGBA")
        processed = remove_background(pil, config)
        img_cv = cv2.cvtColor(np.array(processed), cv2.COLOR_RGBA2BGRA)
        img_cv = remove_watermark(img_cv, config)
        ext = ".png" if config.fmt.upper() == "PNG" else ".jpg"
        out_name = src_path.stem + ext
        out_path = out_dir / out_name
        if save_image(img_cv, out_path, config):
            return f"✅ {name} -> {out_name}"
        return f"❌ Ошибка сохранения {name}"
    except UnidentifiedImageError:
        return f"❌ Невозможно открыть {name} (не изображение/повреждён)"
    except Exception as e:
        logger.exception("Ошибка обработки %s: %s", name, e)
        return f"❌ Ошибка обработки {name}: {e}"

def process_batch(input_dir: str, output_dir: str, config: ProcessingConfig,
                  selected_filenames: Optional[List[str]] = None) -> List[str]:
    inp = Path(input_dir).expanduser().resolve()
    outp = Path(output_dir).expanduser().resolve()
    config.inp = inp
    config.outp = outp

    ok, msg = validate_path(inp, is_input=True)
    if not ok:
        return [f"[ОШИБКА] {msg}"]
    ok, msg = validate_path(outp, is_input=False)
    if not ok:
        return [f"[ОШИБКА] {msg}"]
    outp.mkdir(parents=True, exist_ok=True)

    files = [p for p in inp.iterdir() if p.is_file() and validate_file_extension(p)]
    if selected_filenames:
        files = [p for p in files if p.name in selected_filenames]
    if not files:
        return ["[ПРЕДУПРЕЖДЕНИЕ] Не найдено изображений для обработки."]

    logs: List[str] = []
    max_workers = min(4, max(1, mp.cpu_count()))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_single_task, p, outp, config): p for p in files}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                logs.append(res)
                logger.info(res)
            except Exception as e:
                logger.exception("Worker error")
                logs.append(f"❌ Неожиданная ошибка: {e}")
    return logs

def create_zip_of_output(output_dir: str, zip_name: Optional[str] = None) -> Path:
    outp = Path(output_dir).expanduser().resolve()
    if not outp.exists() or not outp.is_dir():
        raise FileNotFoundError(f"Выходная папка не найдена: {outp}")
    base_name = zip_name or f"{outp.name}_results"
    tmp_dir = Path(tempfile.gettempdir())
    zip_base = tmp_dir / base_name
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(outp))
    return Path(zip_path)

def choose_output_folder_cli(base: str = ".") -> Path:
    base_p = Path(base).expanduser().resolve()
    base_p.mkdir(parents=True, exist_ok=True)
    dirs = [d for d in [base_p] + sorted([p for p in base_p.iterdir() if p.is_dir()])][:10]
    print(f"База для выбора: {base_p}")
    for i, d in enumerate(dirs, start=1):
        print(f"{i}. {d}")
    print("0 - использовать базовую папку; или введите путь")
    choice = input("Выбор: ").strip()
    if choice == "0" or choice == "":
        return base_p
    try:
        idx = int(choice)
        if 1 <= idx <= len(dirs):
            return dirs[idx - 1]
    except Exception:
        pass
    p = Path(choice).expanduser().resolve()
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p

def run_cli(argv=None):
    parser = argparse.ArgumentParser(description="Photo Processor Pro (CLI)")
    parser.add_argument("--input", "-i", default="./input", help="Папка с изображениями")
    parser.add_argument("--output", "-o", default=None, help="Куда сохранять (если не указано — интерактивно)")
    parser.add_argument("--no-bg", dest="remove_bg", action="store_false", help="Отключить удаление фона")
    parser.add_argument("--wm", dest="remove_wm", action="store_true", help="Включить удаление водяных знаков")
    parser.add_argument("--wm-threshold", type=int, default=220)
    parser.add_argument("--wm-radius", type=int, default=5)
    parser.add_argument("--fmt", choices=["PNG", "JPEG"], default="PNG")
    parser.add_argument("--jpeg-q", type=int, default=95)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = ProcessingConfig(
        remove_bg=args.remove_bg,
        remove_wm=args.remove_wm,
        wm_threshold=args.wm_threshold,
        wm_radius=args.wm_radius,
        fmt=args.fmt,
        jpeg_q=args.jpeg_q,
        target_width=args.width,
        target_height=args.height,
        inp=Path(args.input).expanduser().resolve()
    )
    if args.output:
        cfg.outp = Path(args.output).expanduser().resolve()
    else:
        cfg.outp = choose_output_folder_cli(str(cfg.inp))

    logger.info("Начало обработки: %s -> %s", cfg.inp, cfg.outp)
    logs = process_batch(str(cfg.inp), str(cfg.outp), cfg)
    for L in logs:
        print(L)
    logger.info("Готово. Результаты в: %s", cfg.outp)

def run_streamlit():
    if not HAS_STREAMLIT:
        print("Streamlit не установлен.")
        return
    st.set_page_config(title="Photo Processor Pro", layout="wide")
    st.title("🖼️ Photo Processor Pro")

    with st.sidebar:
        st.header("Настройки")
        remove_bg = st.checkbox("Удалить фон", value=True)
        remove_wm = st.checkbox("Удалить водяные знаки", value=False)
        wm_threshold = st.slider("Порог WM", 0, 255, 220)
        wm_radius = st.slider("Радиус inpaint", 1, 20, 5)
        fmt = st.selectbox("Формат", ["PNG", "JPEG"])
        jpeg_q = st.slider("Качество JPEG", 1, 100, 95) if fmt == "JPEG" else 95
        target_width = st.number_input("Ширина (0 = авто)", min_value=0, max_value=10000, value=0)
        target_height = st.number_input("Высота (0 = авто)", min_value=0, max_value=10000, value=0)

    uploaded = st.file_uploader("Выберите изображения", accept_multiple_files=True,
                                type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS])
    if uploaded:
        cols = st.columns(5)
        for i, f in enumerate(uploaded[:10]):
            with cols[i % 5]:
                try:
                    st.image(Image.open(f), caption=f.name, use_column_width=True)
                except Exception:
                    st.write("Не удалось отобразить")

        if st.button("Начать обработку"):
            tw = target_width or None
            th = target_height or None
            temp_dir = Path(tempfile.mkdtemp(prefix="ppp_"))
            for f in uploaded:
                (temp_dir / f.name).write_bytes(f.read())
            out_dir = Path(tempfile.mkdtemp(prefix="ppp_out_"))
            cfg = ProcessingConfig(remove_bg=remove_bg, remove_wm=remove_wm,
                                   wm_threshold=wm_threshold, wm_radius=wm_radius,
                                   fmt=fmt, jpeg_q=jpeg_q, target_width=tw, target_height=th,
                                   inp=temp_dir, outp=out_dir)
            with st.spinner("Обработка..."):
                logs = process_batch(str(temp_dir), str(out_dir), cfg)
            st.subheader("Результаты")
            for L in logs:
                if "✅" in L:
                    st.success(L)
                elif "❌" in L:
                    st.error(L)
                else:
                    st.info(L)
            try:
                zip_path = create_zip_of_output(str(out_dir))
                with open(zip_path, "rb") as fh:
                    st.download_button("Скачать ZIP-архив", data=fh, file_name=zip_path.name, mime="application/zip")
            except Exception as e:
                st.error(f"Не удалось создать архив: {e}")

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["cli", "streamlit"], default="cli")
    ns, _ = parser.parse_known_args()
    if ns.mode == "streamlit":
        run_streamlit()
    else:
        run_cli()

if __name__ == "__main__":
    main()
