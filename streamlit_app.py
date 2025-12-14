# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — исправленный полный скрипт (CLI + Streamlit)
Включает все недостающие функции и защиты от ошибок.
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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import concurrent.futures

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Optional deps
try:
    from rembg import remove as rembg_remove  # type: ignore
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

# Logger
def setup_logger() -> logging.Logger:
    fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(fn, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("photo_processor")

logger = setup_logger()

# Config dataclasses
@dataclass
class WatermarkParams:
    threshold: int = 220
    adaptive: bool = True
    block_size: int = 31  # must be odd >=3
    c: int = 10
    min_area: int = 50
    max_area: int = 5000
    radius: int = 5
    use_ns: bool = True

    def normalized(self) -> "WatermarkParams":
        bs = max(3, int(self.block_size))
        if bs % 2 == 0:
            bs += 1
        return WatermarkParams(
            threshold=int(self.threshold),
            adaptive=bool(self.adaptive),
            block_size=bs,
            c=int(self.c),
            min_area=max(1, int(self.min_area)),
            max_area=max(1, int(self.max_area)),
            radius=max(1, int(self.radius)),
            use_ns=bool(self.use_ns),
        )

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

# Helpers
def ensure_dir(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("ensure_dir failed for %s", p)

def save_params(params: WatermarkParams, filename: str):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(asdict(params), f, ensure_ascii=False, indent=2)
        logger.info("Saved params to %s", filename)
    except Exception:
        logger.exception("save_params failed")

def load_params(filename: str) -> WatermarkParams:
    p = Path(filename)
    if not p.exists():
        return WatermarkParams()
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        merged = {**WatermarkParams().__dict__, **(data or {})}
        return WatermarkParams(**merged).normalized()
    except Exception:
        logger.exception("load_params failed, using defaults")
        return WatermarkParams()

def load_config(filename: str = "ppp_config.json") -> ProcessingConfig:
    p = Path(filename)
    if not p.exists():
        return ProcessingConfig()
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        wm = data.get("wm_params", {})
        wm_params = WatermarkParams(**{**WatermarkParams().__dict__, **(wm or {})}).normalized()
        cfg = ProcessingConfig(
            remove_bg=bool(data.get("remove_bg", True)),
            remove_wm=bool(data.get("remove_wm", True)),
            wm_params=wm_params,
            fmt=str(data.get("fmt", "PNG")),
            jpeg_q=int(data.get("jpeg_q", 95)),
            target_width=(int(data["target_width"]) if data.get("target_width") is not None else None),
            target_height=(int(data["target_height"]) if data.get("target_height") is not None else None),
            inp=Path(str(data.get("inp", "./input"))),
            outp=Path(str(data.get("outp", "./output"))),
        )
        return cfg
    except Exception:
        logger.exception("load_config failed, returning defaults")
        return ProcessingConfig()

def remove_background(pil_img: Image.Image, cfg: ProcessingConfig) -> Image.Image:
    if not cfg.remove_bg or not HAS_REMBG or rembg_remove is None:
        return pil_img
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        out = rembg_remove(buf.read())
        return Image.open(io.BytesIO(out)).convert("RGBA")
    except Exception:
        logger.exception("remove_background failed")
        return pil_img

# Analysis and detection
def analyze_image_for_params(pil_img: Image.Image) -> WatermarkParams:
    try:
        rgb = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mean_color = np.mean(rgb, axis=(0,1))
        contrast_std = np.std(gray)
        threshold = 200 if mean_color[0] > 100 else 220
        adaptive = contrast_std > 15
        return WatermarkParams(threshold=int(threshold), adaptive=bool(adaptive), block_size=31, c=10).normalized()
    except Exception:
        logger.exception("analyze_image_for_params failed")
        return WatermarkParams()

def detect_watermark_auto(pil_img: Image.Image, params: WatermarkParams) -> np.ndarray:
    params = params.normalized()
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if params.adaptive:
        try:
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, params.block_size, params.c)
        except Exception:
            _, thr = cv2.threshold(gray, int(params.threshold), 255, cv2.THRESH_BINARY)
    else:
        _, thr = cv2.threshold(gray, int(params.threshold), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    for c in contours:
        area = cv2.contourArea(c)
        if area < params.min_area or area > params.max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        roi_gray = gray[y:y+h, x:x+w]
        roi_mean = float(np.mean(roi_gray)) if roi_gray.size else 0.0
        pad = 5
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(gray.shape[1], x+w+pad), min(gray.shape[0], y+h+pad)
        bg_roi = gray[y1:y2, x1:x2]
        bg_mean = float(np.mean(bg_roi)) if bg_roi.size else 0.0
        if abs(roi_mean - bg_mean) < 15:
            continue
        cv2.drawContours(mask, [c], -1, 255, -1)
    return mask

def detect_watermark(pil_img: Image.Image, auto_mode=True, user_params: Optional[WatermarkParams]=None) -> np.ndarray:
    params = analyze_image_for_params(pil_img) if auto_mode else (user_params or WatermarkParams()).normalized()
    return detect_watermark_auto(pil_img, params)

# Inpainting & watermark removal
def remove_watermark(img_cv: np.ndarray, cfg: ProcessingConfig) -> np.ndarray:
    if not cfg.remove_wm:
        return img_cv
    try:
        cfg.wm_params = cfg.wm_params.normalized()
        has_alpha = img_cv.ndim == 3 and img_cv.shape[2] == 4
        bgr = img_cv[..., :3].copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if cfg.wm_params.adaptive:
            try:
                thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, cfg.wm_params.block_size, cfg.wm_params.c)
            except Exception:
                _, thr = cv2.threshold(gray, int(cfg.wm_params.threshold), 255, cv2.THRESH_BINARY)
        else:
            _, thr = cv2.threshold(gray, int(cfg.wm_params.threshold), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray, dtype=np.uint8)
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg.wm_params.min_area or area > cfg.wm_params.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            roi_gray = gray[y:y+h, x:x+w]
            roi_mean = float(np.mean(roi_gray)) if roi_gray.size else 0.0
            pad = 5
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(gray.shape[1], x+w+pad), min(gray.shape[0], y+h+pad)
            bg_roi = gray[y1:y2, x1:x2]
            bg_mean = float(np.mean(bg_roi)) if bg_roi.size else 0.0
            if abs(roi_mean - bg_mean) < 15:
                continue
            cv2.drawContours(mask, [c], -1, 255, -1)
        if mask.sum() == 0:
            logger.debug("No watermark mask detected")
            return img_cv
        inpaint_telea = cv2.inpaint(bgr, mask, int(cfg.wm_params.radius), cv2.INPAINT_TELEA)
        chosen = inpaint_telea
        if cfg.wm_params.use_ns:
            try:
                inpaint_ns = cv2.inpaint(bgr, mask, int(cfg.wm_params.radius), cv2.INPAINT_NS)
                m = mask.astype(bool)
                if m.any():
                    telea_err = np.mean(np.abs(inpaint_telea[m] - bgr[m]))
                    ns_err = np.mean(np.abs(inpaint_ns[m] - bgr[m]))
                    chosen = inpaint_ns if ns_err <= telea_err else inpaint_telea
            except Exception:
                logger.exception("INPAINT_NS failed, using TELEA")
                chosen = inpaint_telea
        if has_alpha:
            out = cv2.cvtColor(chosen, cv2.COLOR_BGR2BGRA)
            out[..., 3] = img_cv[..., 3]
        else:
            out = chosen
        return out
    except Exception:
        logger.exception("remove_watermark error")
        return img_cv

# Resize / save
def resize_cv(img_cv: np.ndarray, w_target: Optional[int], h_target: Optional[int]) -> np.ndarray:
    h, w = img_cv.shape[:2]
    if not w_target and not h_target:
        return img_cv
    if w_target and h_target:
        return cv2.resize(img_cv, (w_target, h_target), interpolation=cv2.INTER_AREA)
    if w_target:
        scale = w_target / w
        return cv2.resize(img_cv, (w_target, max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
    if h_target:
        scale = h_target / h
        return cv2.resize(img_cv, (max(1, int(w*scale)), h_target), interpolation=cv2.INTER_AREA)
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
        logger.exception("save_cv_image failed for %s", out_path)
        return False

# Processing single image
def process_image(in_path: Path, out_path: Path, cfg: ProcessingConfig) -> Tuple[bool, str]:
    try:
        pil = Image.open(in_path)
        pil = pil.convert("RGBA") if pil.mode in ("RGBA","LA") else pil.convert("RGB")
        pil = remove_background(pil, cfg)
        img_cv = np.array(pil)
        if img_cv.ndim == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        elif img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        elif img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGRA)
        img_cv = remove_watermark(img_cv, cfg)
        out_final = out_path.with_suffix("." + cfg.fmt.lower())
        if save_cv_image(img_cv, out_final, cfg):
            return True, ""
        return False, f"Error saving {out_final}"
    except UnidentifiedImageError:
        return False, f"Unidentified image: {in_path.name}"
    except Exception:
        logger.exception("process_image failed for %s", in_path)
        return False, "processing error"

# Batch
def validate_ext(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def process_batch(input_dir: Path, output_dir: Path, cfg: ProcessingConfig, max_workers: int = 4):
    ensure_dir(input_dir)
    ensure_dir(output_dir)
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and validate_ext(p)]
    results: List[Tuple[Path, bool, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_image, p, output_dir / p.stem, cfg): p for p in files}
        for f in concurrent.futures.as_completed(futures):
            p = futures[f]
            try:
                ok, msg = f.result()
                results.append((p, ok, msg))
            except Exception as e:
                results.append((p, False, str(e)))
    return results

def zip_results(out_dir: Path, results: List[Tuple[Path, bool, str]], format_ext: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p, ok, _ in results:
            if not ok:
                continue
            fname = f"{p.stem}.{format_ext}"
            fp = out_dir / fname
            if fp.exists():
                zf.write(fp, arcname=fname)
    buf.seek(0)
    return buf.read()

# CLI
def run_cli(argv=None):
    parser = argparse.ArgumentParser(description="Photo Processor Pro CLI")
    parser.add_argument("--input", type=Path, default=Path("./input"), help="Input folder (default ./input)")
    parser.add_argument("--output", type=Path, default=Path("./output"), help="Output folder (default ./output)")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate (analyze + save params)")
    parser.add_argument("--params_file", type=str, default="detected_params.json", help="Params file")
    parser.add_argument("--remove_bg", action="store_true", help="Remove background")
    parser.add_argument("--remove_wm", action="store_true", help="Remove watermark")
    parser.add_argument("--workers", type=int, default=4, help="Threads")
    args = parser.parse_args(argv)

    cfg = ProcessingConfig(inp=args.input, outp=args.output)
    if args.remove_bg:
        cfg.remove_bg = True
    if args.remove_wm:
        cfg.remove_wm = True

    if args.calibrate:
        ensure_dir(cfg.inp)
        sample = next(cfg.inp.glob("*.*"), None)
        if sample:
            try:
                pil = Image.open(sample)
                params = analyze_image_for_params(pil)
                save_params(params, args.params_file)
                print(f"Saved params to {args.params_file}")
            except Exception:
                logger.exception("Calibration failed")
                print("Calibration failed")
        else:
            print("No files for calibration")
        return

    cfg.wm_params = load_params(args.params_file)
    results = process_batch(cfg.inp, cfg.outp, cfg, max_workers=args.workers)
    for p, ok, msg in results:
        print(f"{'✓' if ok else '✗'} {p.name}: {msg}")

# Streamlit UI
def run_streamlit():
    if st is None:
        raise RuntimeError("Streamlit not installed")
    cfg = load_config()
    st.title("Photo Processor Pro — обработка изображений")
    st.sidebar.header("Настройки")
    inp_dir = Path(st.sidebar.text_input("Входная папка", str(cfg.inp)))
    out_dir = Path(st.sidebar.text_input("Выходная папка", str(cfg.outp)))
    remove_bg = st.sidebar.checkbox("Удалить фон", value=cfg.remove_bg)
    remove_wm = st.sidebar.checkbox("Удалить водяные знаки", value=cfg.remove_wm)
    wm_adaptive = st.sidebar.checkbox("Адаптивный порог", value=cfg.wm_params.adaptive)
    wm_block_size = st.sidebar.number_input("Размер блока adaptiveThreshold", value=cfg.wm_params.block_size, min_value=3, step=2)
    wm_c = st.sidebar.number_input("Коррекция adaptiveThreshold", value=cfg.wm_params.c, min_value=-100, max_value=100)
    wm_min_area = st.sidebar.number_input("Мин. площадь водяного знака", value=cfg.wm_params.min_area, min_value=1)
    wm_max_area = st.sidebar.number_input("Макс. площадь водяного знака", value=cfg.wm_params.max_area, min_value=1)
    wm_radius = st.sidebar.number_input("Радиус inpaint", value=cfg.wm_params.radius, min_value=1)
    wm_use_ns = st.sidebar.checkbox("Использовать inpaint NS", value=cfg.wm_params.use_ns)
    fmt_options = ["PNG", "JPEG", "BMP"]
    fmt = st.sidebar.selectbox("Формат", fmt_options, index=fmt_options.index(cfg.fmt if cfg.fmt in fmt_options else "PNG"))
    jpeg_q = st.sidebar.slider("Качество JPEG", 0, 100, cfg.jpeg_q)
    tw = st.sidebar.number_input("Ширина (px)", value=int(cfg.target_width or 0), min_value=0)
    th = st.sidebar.number_input("Высота (px)", value=int(cfg.target_height or 0), min_value=0)
    workers = st.sidebar.number_input("Потоки", value=4, min_value=1)

    uploaded_files = st.sidebar.file_uploader("Загрузить файлы", type=["jpg","jpeg","png","bmp","tiff","webp"], accept_multiple_files=True)
    temp_dir = None
    if uploaded_files:
        temp_dir = Path(tempfile.mkdtemp())
        for f in uploaded_files:
            (temp_dir / f.name).write_bytes(f.read())
        st.sidebar.success(f"Загружено {len(uploaded_files)} файлов в {temp_dir}")

    use_uploaded = st.sidebar.checkbox("Использовать загруженные файлы", value=bool(uploaded_files))
    input_dir = Path(inp_dir) if not use_uploaded else (temp_dir or Path("."))
    output_dir = Path(out_dir)

    if st.button("Начать обработку"):
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
            ).normalized(),
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
                    except Exception:
                        logger.exception("Failed to display %s", img_path)

    if temp_dir and temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            logger.exception("Failed to cleanup temp_dir")

# Entrypoint
def main():
    # If streamlit present and script launched without CLI args, run UI
    if HAS_STREAMLIT and len(sys.argv) <= 1:
        run_streamlit()
    else:
        run_cli()

if __name__ == "__main__":
    main()
