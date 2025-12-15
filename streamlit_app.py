# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — улучшенная и надёжная версия (CLI + Streamlit).
Исправления и улучшения:
- безопасный импорт onnxruntime и rembg;
- аккуратная обработка ошибок (без bare except);
- небольшие улучшения детекции и работы с PIL/NumPy;
- явное управление временными директориями в Streamlit;
- понятное логирование.
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

# optional deps
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

try:
    from rembg import remove as rembg_remove  # type: ignore
    HAS_REMBG = True
except Exception:
    rembg_remove = None
    HAS_REMBG = False

try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception:
    st = None
    HAS_STREAMLIT = False

# model session (optional)
MODEL_PATH = Path("watermark_segmentation.onnx")
model_session = None
if ort is not None and MODEL_PATH.exists():
    try:
        model_session = ort.InferenceSession(str(MODEL_PATH))
        logging.getLogger().info("ONNX model loaded: %s", MODEL_PATH)
    except Exception:
        model_session = None
        logging.getLogger().exception("Failed to load ONNX model; continuing without it")

# logger
def setup_logger() -> logging.Logger:
    log = logging.getLogger("photo_processor")
    if not log.handlers:
        fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handler = logging.FileHandler(fn, encoding="utf-8")
        sh = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        sh.setFormatter(fmt)
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        log.addHandler(sh)
    return log

logger = setup_logger()

# dataclasses
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

# helpers
def ensure_dir(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("ensure_dir failed for %s", p)

def save_json(obj, filename: str):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("save_json failed for %s", filename)

def load_json(filename: str) -> Optional[dict]:
    p = Path(filename)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("load_json failed for %s", filename)
        return None

def save_params(params: WatermarkParams, filename: str):
    save_json(asdict(params), filename)

def load_params(filename: str) -> WatermarkParams:
    data = load_json(filename)
    if not data:
        return WatermarkParams()
    merged = {**WatermarkParams().__dict__, **data}
    try:
        return WatermarkParams(**merged).normalized()
    except Exception:
        logger.exception("Invalid params file, using defaults")
        return WatermarkParams()

def load_config(filename: str = "ppp_config.json") -> ProcessingConfig:
    data = load_json(filename)
    if not data:
        return ProcessingConfig()
    try:
        wm = data.get("wm_params", {}) or {}
        wm_params = WatermarkParams(**{**WatermarkParams().__dict__, **wm}).normalized()
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

# background removal
def remove_background(pil_img: Image.Image, cfg: ProcessingConfig) -> Image.Image:
    if not cfg.remove_bg or not HAS_REMBG or rembg_remove is None:
        return pil_img
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        out = rembg_remove(buf.getvalue())
        return Image.open(io.BytesIO(out)).convert("RGBA")
    except Exception:
        logger.exception("remove_background failed")
        return pil_img

# model segmentation (optional)
def segment_watermark_with_model(pil_img: Image.Image) -> np.ndarray:
    if model_session is None:
        return np.zeros((pil_img.height, pil_img.width), dtype=np.uint8)
    try:
        inp_meta = model_session.get_inputs()[0].shape
        # assume (1,C,H,W) or similar
        _, c, h, w = inp_meta if len(inp_meta) == 4 else (1, 3, 256, 256)
        resized = pil_img.resize((w, h))
        arr = np.asarray(resized).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: tensor})
        pred = outputs[0]
        pred_map = pred[0, 0] if pred.ndim == 4 else pred[0]
        mask = cv2.resize(pred_map.astype(np.float32), (pil_img.width, pil_img.height))
        return (mask > 0.5).astype(np.uint8) * 255
    except Exception:
        logger.exception("segment_watermark_with_model failed")
        return np.zeros((pil_img.height, pil_img.width), dtype=np.uint8)

# adaptive auto-threshold using histogram percentile
def auto_threshold(gray: np.ndarray, percentile: float = 0.80) -> int:
    if gray.size == 0:
        return 220
    hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    cumsum = np.cumsum(hist)
    total = gray.size
    idx = np.searchsorted(cumsum, total * percentile)
    return int(idx if 0 <= idx < 256 else 220)

# traditional detection
def detect_watermark_auto(pil_img: Image.Image, params: WatermarkParams) -> np.ndarray:
    params = params.normalized()
    rgb = np.asarray(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    auto_t = auto_threshold(gray)
    threshold_value = max(int(params.threshold), auto_t)
    if params.adaptive:
        try:
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, params.block_size, params.c)
        except Exception:
            _, thr = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        _, thr = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    for c in contours:
        area = cv2.contourArea(c)
        if area < params.min_area or area > params.max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        roi_mean = float(np.mean(roi))
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        bg = gray[y1:y2, x1:x2]
        if bg.size == 0:
            continue
        bg_mean = float(np.mean(bg))
        if abs(roi_mean - bg_mean) < 15:
            continue
        cv2.drawContours(mask, [c], -1, 255, -1)
    return mask

# refine mask
def refine_mask(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    try:
        nb, out, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        refined = np.zeros_like(mask)
        for i in range(1, nb):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                refined[out == i] = 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k, iterations=2)
        return refined
    except Exception:
        logger.exception("refine_mask failed")
        return mask

# remove watermark (combine model + traditional)
def remove_watermark(img_cv: np.ndarray, cfg: ProcessingConfig, use_model: bool = True) -> np.ndarray:
    if not cfg.remove_wm:
        return img_cv
    try:
        cfg.wm_params = cfg.wm_params.normalized()
        h, w = img_cv.shape[:2]
        rgb_pil = Image.fromarray(cv2.cvtColor(img_cv[..., :3], cv2.COLOR_BGR2RGB))
        masks: List[np.ndarray] = []
        if use_model and model_session is not None:
            masks.append(segment_watermark_with_model(rgb_pil))
        masks.append(detect_watermark_auto(rgb_pil, cfg.wm_params))
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            if m.shape == combined.shape:
                combined = cv2.bitwise_or(combined, m)
        combined = refine_mask(combined, min_size=max(50, cfg.wm_params.min_area))
        if np.count_nonzero(combined) == 0:
            logger.debug("No watermark mask found")
            return img_cv
        has_alpha = img_cv.ndim == 3 and img_cv.shape[2] == 4
        bgr = img_cv[..., :3].copy()
        telea = cv2.inpaint(bgr, combined, max(1, int(cfg.wm_params.radius)), cv2.INPAINT_TELEA)
        chosen = telea
        if cfg.wm_params.use_ns:
            try:
                ns = cv2.inpaint(bgr, combined, max(1, int(cfg.wm_params.radius)), cv2.INPAINT_NS)
                m_bool = combined.astype(bool)
                if m_bool.any():
                    telea_err = float(np.mean(np.abs(telea[m_bool] - bgr[m_bool])))
                    ns_err = float(np.mean(np.abs(ns[m_bool] - bgr[m_bool])))
                    chosen = ns if ns_err <= telea_err else telea
            except Exception:
                logger.exception("INPAINT_NS failed, using TELEA")
        if has_alpha:
            out = cv2.cvtColor(chosen, cv2.COLOR_BGR2BGRA)
            out[..., 3] = img_cv[..., 3]
        else:
            out = chosen
        return out
    except Exception:
        logger.exception("remove_watermark error")
        return img_cv

# resize & save
def resize_cv(img_cv: np.ndarray, w_t: Optional[int], h_t: Optional[int]) -> np.ndarray:
    h, w = img_cv.shape[:2]
    if not w_t and not h_t:
        return img_cv
    if w_t and h_t:
        return cv2.resize(img_cv, (w_t, h_t), interpolation=cv2.INTER_AREA)
    if w_t:
        scale = w_t / w
        return cv2.resize(img_cv, (w_t, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    if h_t:
        scale = h_t / h
        return cv2.resize(img_cv, (max(1, int(w * scale)), h_t), interpolation=cv2.INTER_AREA)
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

# processing a single image
def process_image(in_path: Path, out_path: Path, cfg: ProcessingConfig, use_model_segmentation: bool = True) -> Tuple[bool, str]:
    try:
        with Image.open(in_path) as pil:
            pil = pil.convert("RGBA") if pil.mode in ("RGBA", "LA") else pil.convert("RGB")
            pil = remove_background(pil, cfg)
            img_cv = np.asarray(pil)
        if img_cv.ndim == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        elif img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        elif img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGRA)
        img_cv = remove_watermark(img_cv, cfg, use_model=use_model_segmentation)
        out_final = out_path.with_suffix("." + cfg.fmt.lower())
        if save_cv_image(img_cv, out_final, cfg):
            return True, ""
        return False, f"Failed to save {out_final}"
    except UnidentifiedImageError:
        return False, f"Unidentified image: {in_path.name}"
    except Exception:
        logger.exception("process_image failed for %s", in_path)
        return False, "processing error"

# batch
def validate_ext(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def process_batch(input_dir: Path, output_dir: Path, cfg: ProcessingConfig, max_workers: int = 4, use_model_segmentation: Optional[bool] = None):
    ensure_dir(input_dir)
    ensure_dir(output_dir)
    if use_model_segmentation is None:
        use_model_segmentation = model_session is not None
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and validate_ext(p)]
    results: List[Tuple[Path, bool, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_image, p, output_dir / p.stem, cfg, use_model_segmentation): p for p in files}
        for fut in concurrent.futures.as_completed(futures):
            p = futures[fut]
            try:
                ok, msg = fut.result()
                results.append((p, ok, msg))
            except Exception as e:
                logger.exception("Worker raised")
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
                try:
                    zf.write(fp, arcname=fname)
                except Exception:
                    logger.exception("Failed to add to ZIP: %s", fp)
    buf.seek(0)
    return buf.read()

# CLI
def run_cli(argv=None):
    parser = argparse.ArgumentParser(description="Photo Processor Pro CLI")
    parser.add_argument("--input", type=Path, default=Path("./input"))
    parser.add_argument("--output", type=Path, default=Path("./output"))
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--params_file", type=str, default="detected_params.json")
    parser.add_argument("--remove_bg", action="store_true")
    parser.add_argument("--remove_wm", action="store_true")
    parser.add_argument("--use_model", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
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
                with Image.open(sample) as pil:
                    params = detect_watermark_auto(pil, WatermarkParams())
                    save_params(params.__dict__, args.params_file)
                    print(f"Saved params to {args.params_file}")
            except Exception:
                logger.exception("Calibration failed")
                print("Calibration failed")
        else:
            print("No files for calibration")
        return

    cfg.wm_params = load_params(args.params_file)
    results = process_batch(cfg.inp, cfg.outp, cfg, max_workers=args.workers, use_model_segmentation=args.use_model)
    for p, ok, msg in results:
        print(f"{'✓' if ok else '✗'} {p.name}: {msg}")

# Streamlit
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
    use_model = st.sidebar.checkbox("Использовать ONNX модель (если есть)", value=(model_session is not None))
    fmt_options = ["PNG", "JPEG", "BMP"]
    fmt = st.sidebar.selectbox("Формат", fmt_options, index=fmt_options.index(cfg.fmt if cfg.fmt in fmt_options else "PNG"))
    jpeg_q = st.sidebar.slider("Качество JPEG", 0, 100, cfg.jpeg_q)
    tw = st.sidebar.number_input("Ширина (px)", value=int(cfg.target_width or 0), min_value=0)
    th = st.sidebar.number_input("Высота (px)", value=int(cfg.target_height or 0), min_value=0)
    workers = st.sidebar.number_input("Потоки", value=4, min_value=1)

    uploaded_files = st.sidebar.file_uploader("Загрузить файлы", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"], accept_multiple_files=True)
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
            results = process_batch(input_dir, output_dir, cfg_local, max_workers=int(workers), use_model_segmentation=use_model)
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
                        cols[i % 3].image(img, caption=p.name, width=300)
                    except Exception:
                        logger.exception("Failed to display %s", img_path)

    if temp_dir and temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            logger.exception("Failed to cleanup temp_dir")

# entrypoint
def main():
    if HAS_STREAMLIT and len(sys.argv) <= 1:
        run_streamlit()
    else:
        run_cli()

if __name__ == "__main__":
    main()
