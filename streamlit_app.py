#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — полный скрипт (CLI + Streamlit) с расширенной автоматической обработкой водяных знаков
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

# Импорт onnxruntime
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

MODEL_PATH = Path("watermark_segmentation.onnx")
model_session = None
if ort is not None and MODEL_PATH.exists():
    try:
        model_session = ort.InferenceSession(str(MODEL_PATH))
        print("ONNX модель загружена:", MODEL_PATH)
    except Exception:
        model_session = None
        print("Не удалось загрузить ONNX модель, продолжаем без нее")
else:
    model_session = None

# rembg
try:
    from rembg import remove as rembg_remove  # type: ignore
    HAS_REMBG = True
except Exception:
    rembg_remove = None
    HAS_REMBG = False

# streamlit
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception:
    st = None
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


# Вспомогательные функции
def ensure_dir(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except:
        logger.exception("ensure_dir failed for %s", p)

def save_params(params: WatermarkParams, filename: str):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(asdict(params), f, ensure_ascii=False, indent=2)
    except:
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
    except:
        logger.exception("load_params failed")
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
    except:
        logger.exception("load_config failed")
        return ProcessingConfig()

# Удаление фона
def remove_background(pil_img: Image.Image, cfg: ProcessingConfig) -> Image.Image:
    if not cfg.remove_bg or not HAS_REMBG or rembg_remove is None:
        return pil_img
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        out = rembg_remove(buf.read())
        return Image.open(io.BytesIO(out)).convert("RGBA")
    except:
        logger.exception("remove_background failed")
        return pil_img

# Модель сегментации
def segment_watermark_with_model(pil_img: Image.Image) -> np.ndarray:
    if model_session is None:
        return np.zeros((pil_img.height, pil_img.width), dtype=np.uint8)
    try:
        inp_shape = model_session.get_inputs()[0].shape
        _, c, h, w = inp_shape if len(inp_shape) == 4 else (1, 3, 256, 256)
        resized = pil_img.resize((w, h))
        arr = np.array(resized).astype(np.float32) / 255
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: tensor})
        pred = outputs[0]
        pred_map = pred[0, 0] if pred.ndim == 4 else pred[0]
        mask_resized = cv2.resize(pred_map.astype(np.float32), (pil_img.width, pil_img.height))
        mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
        return mask_bin
    except:
        logger.exception("segment_watermark_with_model failed")
        return np.zeros((pil_img.height, pil_img.width), dtype=np.uint8)

# Автоматическое определение порога
def auto_threshold(gray_img: np.ndarray) -> int:
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    total = gray_img.size
    cumsum = 0
    threshold = 0
    for i in range(256):
        cumsum += hist[i][0]
        if cumsum > total * 0.8:
            threshold = i
            break
    return threshold

# Детекция водяных знаков по thresholding
def detect_watermark_auto(pil_img: Image.Image, params: WatermarkParams) -> np.ndarray:
    params = params.normalized()
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Автоматический подбор порога
    auto_thresh = auto_threshold(gray)
    threshold_value = max(int(params.threshold), auto_thresh)

    if params.adaptive:
        try:
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, params.block_size, params.c)
        except:
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
        roi_gray = gray[y:y+h, x:x+w]
        if roi_gray.size == 0:
            continue
        roi_mean = float(np.mean(roi_gray))
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        bg_roi = gray[y1:y2, x1:x2]
        if bg_roi.size == 0:
            continue
        bg_mean = float(np.mean(bg_roi))
        if abs(roi_mean - bg_mean) < 15:
            continue
        cv2.drawContours(mask, [c], -1, 255, -1)

    return mask

# Рефайн маски: удаление мелких объектов и заполнение дыр
def refine_mask(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            output[output == i] = 0
    refined = np.zeros_like(mask)
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            refined[output == i] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return refined

# Основная функция удаления водяных знаков
def remove_watermark(
    img_cv: np.ndarray, 
    cfg: ProcessingConfig, 
    use_model: bool = True
) -> np.ndarray:
    if not cfg.remove_wm:
        return img_cv
    try:
        cfg.wm_params = cfg.wm_params.normalized()
        h_img, w_img = img_cv.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(img_cv[..., :3], cv2.COLOR_BGR2RGB))
        # Объединение методов
        masks = []

        # Модель
        if use_model and model_session is not None:
            masks.append(segment_watermark_with_model(pil_img))
        # Thresholding
        masks.append(detect_watermark_auto(pil_img, cfg.wm_params))
        # Объединяем маски
        combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for m in masks:
            combined_mask = cv2.bitwise_or(combined_mask, m)
        # Рефайн маски
        combined_mask = refine_mask(combined_mask, min_size=100)

        if np.count_nonzero(combined_mask) == 0:
            logger.debug("Маска водяного знака не обнаружена")
            return img_cv

        has_alpha = img_cv.ndim == 3 and img_cv.shape[2] == 4
        bgr = img_cv[..., :3].copy()

        # Inpaint
        inpaint_telea = cv2.inpaint(bgr, combined_mask, max(1, int(cfg.wm_params.radius)), cv2.INPAINT_TELEA)
        inpaint_ns = None
        if cfg.wm_params.use_ns:
            try:
                inpaint_ns = cv2.inpaint(bgr, combined_mask, max(1, int(cfg.wm_params.radius)), cv2.INPAINT_NS)
            except:
                logger.exception("INPAINT_NS не удалось, используем TELEA")
        # Выбор лучшего результата
        if inpaint_ns is not None:
            telea_err = np.mean(np.abs(inpaint_telea - bgr))
            ns_err = np.mean(np.abs(inpaint_ns - bgr))
            chosen = inpaint_ns if ns_err <= telea_err else inpaint_telea
        else:
            chosen = inpaint_telea

        if has_alpha:
            out = cv2.cvtColor(chosen, cv2.COLOR_BGR2BGRA)
            out[..., 3] = img_cv[..., 3]
        else:
            out = chosen
        return out

    except:
        logger.exception("Ошибка при удалении водяных знаков")
        return img_cv

# Resize и сохранение
def resize_cv(img_cv: np.ndarray, w_target: Optional[int], h_target: Optional[int]) -> np.ndarray:
    h, w = img_cv.shape[:2]
    if not w_target and not h_target:
        return img_cv
    if w_target and h_target:
        return cv2.resize(img_cv, (w_target, h_target), interpolation=cv2.INTER_AREA)
    if w_target:
        scale = w_target / w
        return cv2.resize(img_cv, (w_target, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    if h_target:
        scale = h_target / h
        return cv2.resize(img_cv, (max(1, int(w * scale)), h_target), interpolation=cv2.INTER_AREA)
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
    except:
        logger.exception("save_cv_image failed for %s", out_path)
        return False

# Обработка одного изображения
def process_image(in_path: Path, out_path: Path, cfg: ProcessingConfig, use_model_segmentation: bool = True) -> Tuple[bool, str]:
    try:
        pil = Image.open(in_path)
        pil = pil.convert("RGBA") if pil.mode in ("RGBA", "LA") else pil.convert("RGB")
        pil = remove_background(pil, cfg)
        img_cv = np.array(pil)
        if img_cv.ndim == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        elif img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        elif img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGRA)
        # Удаление водяных знаков
        img_cv = remove_watermark(img_cv, cfg, use_model=use_model_segmentation)
        out_final = out_path.with_suffix("." + cfg.fmt.lower())
        if save_cv_image(img_cv, out_final, cfg):
            return True, ""
        return False, f"Ошибка при сохранении {out_final}"
    except UnidentifiedImageError:
        return False, f"Некорректное изображение: {in_path.name}"
    except:
        logger.exception("Ошибка обработки %s", in_path)
        return False, "Ошибка обработки"

# Базовая валидация расширений
def validate_ext(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Обработка каталога
def process_batch(input_dir: Path, output_dir: Path, cfg: ProcessingConfig, max_workers: int = 4, use_model_segmentation: bool = True):
    ensure_dir(input_dir)
    ensure_dir(output_dir)
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and validate_ext(p)]
    results: List[Tuple[Path, bool, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_image, p, output_dir / p.stem, cfg, use_model_segmentation): p for p in files}
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
                try:
                    zf.write(fp, arcname=fname)
                except:
                    logger.exception("Ошибка при добавлении файла в ZIP: %s", fp)
    buf.seek(0)
    return buf.read()

# CLI
def run_cli(argv=None):
    parser = argparse.ArgumentParser(description="Photo Processor Pro CLI")
    parser.add_argument("--input", type=Path, default=Path("./input"), help="Входная папка (по умолчанию ./input)")
    parser.add_argument("--output", type=Path, default=Path("./output"), help="Выходная папка (по умолчанию ./output)")
    parser.add_argument("--calibrate", action="store_true", help="Калибровка (анализ и сохранение параметров)")
    parser.add_argument("--params_file", type=str, default="detected_params.json", help="Файл параметров")
    parser.add_argument("--remove_bg", action="store_true", help="Удалить фон")
    parser.add_argument("--remove_wm", action="store_true", help="Удалить водяные знаки")
    parser.add_argument("--use_model", action="store_true", help="Использовать ONNX модель для сегментации")
    parser.add_argument("--workers", type=int, default=4, help="Потоки")
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
                params = detect_watermark_auto(pil, WatermarkParams())
                save_params(params, args.params_file)
                print(f"Параметры сохранены в {args.params_file}")
            except:
                logger.exception("Калибровка не удалась")
                print("Калибровка не удалась")
        else:
            print("Нет файлов для калибровки")
        return

    cfg.wm_params = load_params(args.params_file)
    results = process_batch(cfg.inp, cfg.outp, cfg, max_workers=args.workers, use_model_segmentation=args.use_model)
    for p, ok, msg in results:
        print(f"{'✓' if ok else '✗'} {p.name}: {msg}")

# Streamlit UI
def run_streamlit():
    if st is None:
        raise RuntimeError("Streamlit не установлен")
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
                    except:
                        logger.exception("Не удалось отобразить %s", img_path)

    if temp_dir and temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except:
            logger.exception("Ошибка при удалении временной папки")

# Главная точка входа
def main():
    if HAS_STREAMLIT and len(sys.argv) <= 1:
        run_streamlit()
    else:
        run_cli()

if __name__ == "__main__":
    main()
