# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit app: Photo Processor Pro — robust fixed version.
- No matplotlib dependency (uses OpenCV/PIL for histogram visualization).
- Uses rembg if available; falls back to GrabCut if not.
- Handles missing optional packages gracefully and fixes prior bugs.
"""
from pathlib import Path
from datetime import datetime
import logging
import traceback
import io
import os

import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import streamlit as st

# Try optional rembg
try:
    from rembg import remove as rembg_remove  # type: ignore
    HAS_REMBG = True
except Exception:
    rembg_remove = None  # type: ignore
    HAS_REMBG = False

# Logger
def setup_logger():
    fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(fn, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# Helpers
def validate_paths(inp: Path, out: Path):
    if not inp.exists() or not inp.is_dir():
        return False, f"Входная папка '{inp}' не найдена или не директория."
    if not os.access(str(inp), os.R_OK):
        return False, f"Нет доступа для чтения: '{inp}'."
    try:
        out.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Не удалось создать папку вывода '{out}': {e}"
    if not os.access(str(out), os.W_OK):
        return False, f"Нет доступа для записи: '{out}'."
    return True, "OK"

def get_image_files(inp: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return [p for p in sorted(inp.iterdir()) if p.is_file() and p.suffix.lower() in exts]

def rembg_background(pil_img: Image.Image):
    """Use rembg if available. rembg may return PIL.Image or bytes."""
    if not HAS_REMBG or rembg_remove is None:
        return pil_img
    try:
        out = rembg_remove(pil_img)
        if isinstance(out, (bytes, bytearray)):
            return Image.open(io.BytesIO(out))
        if isinstance(out, Image.Image):
            return out
    except Exception as e:
        logger.warning(f"rembg failed: {e}")
    return pil_img

def grabcut_background(pil_img: Image.Image):
    """Fallback background removal with GrabCut (fast resized pass)."""
    try:
        img = np.array(pil_img.convert("RGB"))
        h, w = img.shape[:2]
        scale = 512 / max(h, w) if max(h, w) > 512 else 1.0
        small = cv2.resize(img, (max(1,int(w*scale)), max(1,int(h*scale))), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros(small.shape[:2], np.uint8)
        rect = (5, 5, small.shape[1]-10, small.shape[0]-10)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv2.grabCut(small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        small_rgba = cv2.cvtColor(small, cv2.COLOR_RGB2RGBA)
        small_rgba[...,3] = mask2*255
        alpha = cv2.resize(small_rgba[...,3], (w, h), interpolation=cv2.INTER_LINEAR)
        result = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        result[...,3] = alpha
        return Image.fromarray(result)
    except Exception as e:
        logger.warning(f"grabcut failed: {e}")
        return pil_img

def remove_background_pil(pil_img: Image.Image):
    """Prefer rembg, fallback to GrabCut."""
    if HAS_REMBG:
        try:
            out = rembg_background(pil_img)
            if isinstance(out, Image.Image):
                return out
        except Exception:
            logger.exception("rembg crashed, falling back to grabcut")
    return grabcut_background(pil_img)

def remove_watermark_cv(img_cv: np.ndarray, threshold: int = 220, radius: int = 5):
    """Simple inpaint by bright-thresholding."""
    try:
        bgr = img_cv[..., :3].copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in contours:
            if cv2.contourArea(c) > 50:
                cv2.drawContours(mask, [c], -1, 255, -1)
        if np.any(mask):
            inpainted = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
            if img_cv.shape[2] == 4:
                out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
                out[...,3] = img_cv[...,3]
                return out
            return inpainted
        return img_cv
    except Exception:
        logger.exception("remove_watermark_cv failed")
        return img_cv

def save_image(img_cv: np.ndarray, out_path: Path, fmt: str, jpeg_quality: int = 95):
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt.startswith("PNG") and img_cv.shape[2] == 4:
            cv2.imwrite(str(out_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            return True
        bgr = img_cv
        if img_cv.shape[2] == 4:
            bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        success, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if success:
            out_path.write_bytes(buf.tobytes())
            return True
    except Exception:
        logger.exception("save_image failed")
    return False

def bgr_to_display(img_cv: np.ndarray):
    """Return image suitable for streamlit.image (RGB or RGBA numpy array)."""
    if img_cv is None:
        return None
    if img_cv.ndim == 2:
        return img_cv
    if img_cv.shape[2] == 3:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    if img_cv.shape[2] == 4:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
    return img_cv

def histogram_image_rgb(img_rgb: np.ndarray, size=(256,120)):
    """Create a small RGB image visualizing R/G/B histograms using OpenCV drawing."""
    h, w = size[1], size[0]
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)
    if img_rgb is None:
        return canvas
    if img_rgb.ndim == 2:
        hist = cv2.calcHist([img_rgb], [0], None, [256], [0,256])
        cv2.normalize(hist, hist, 0, h-10, cv2.NORM_MINMAX)
        prev = None
        for x in range(256):
            y = h - int(hist[x])
            if prev is not None:
                cv2.line(canvas, (x-1, prev), (x, y), (200,200,200), 1)
            prev = y
    else:
        colors = [(255,0,0),(0,255,0),(0,0,255)]
        for ch in range(3):
            hist = cv2.calcHist([img_rgb], [ch], None, [256], [0,256])
            cv2.normalize(hist, hist, 0, h-10, cv2.NORM_MINMAX)
            prev = None
            for x in range(256):
                y = h - int(hist[x])
                if prev is not None:
                    cv2.line(canvas, (x-1, prev), (x, y), colors[ch], 1)
                prev = y
    return canvas

# Streamlit app
def main():
    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("🖼️ Photo Processor Pro")
    st.write("Пакетная обработка: удаление фона + инпейтинг водяных знаков.")

    if "logs" not in st.session_state:
        st.session_state.logs = []

    with st.sidebar:
        st.header("Настройки")
        input_dir = st.text_input("Входная папка", value="./input")
        output_dir = st.text_input("Выходная папка", value="./output")
        st.markdown("---")
        remove_bg = st.checkbox("Удалить фон (rembg если доступен)", value=True)
        if remove_bg and not HAS_REMBG:
            st.caption("rembg не установлен — используется fallback (GrabCut).")
        remove_wm = st.checkbox("Удалить водяные знаки (inpaint)", value=False)
        if remove_wm:
            wm_radius = st.slider("Радиус inpaint", 1, 25, 5)
            wm_threshold = st.slider("Порог яркости для маски", 120, 255, 220)
        st.markdown("---")
        fmt = st.radio("Формат вывода", ("PNG (с альфа)", "JPEG (без альфа)"))
        jpeg_q = st.slider("Качество JPEG (%)", 50, 100, 95) if fmt.startswith("JPEG") else 95
        st.markdown("---")
        run = st.button("🚀 Запустить обработку")

    progress_placeholder = st.empty()
    status = st.empty()
    preview = st.container()

    if run:
        inp = Path(input_dir)
        outp = Path(output_dir)
        ok, msg = validate_paths(inp, outp)
        if not ok:
            st.error(msg)
            return

        imgs = get_image_files(inp)
        if not imgs:
            st.warning("Входная папка не содержит поддерживаемых изображений.")
            return

        total = len(imgs)
        progress = progress_placeholder.progress(0)
        st.session_state.logs = []

        for i, p in enumerate(imgs):
            try:
                with Image.open(p) as pil:
                    pil_orig = pil.convert("RGBA")
                processed_pil = pil_orig
                mask_preview = None

                if remove_bg:
                    processed_pil = remove_background_pil(pil_orig)
                    if processed_pil.mode != "RGBA":
                        processed_pil = processed_pil.convert("RGBA")
                    alpha = np.array(processed_pil.split()[-1])
                    mask_preview = (alpha == 0).astype("uint8") * 255

                img_cv = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGBA2BGRA)

                if remove_wm:
                    img_cv = remove_watermark_cv(img_cv, threshold=wm_threshold, radius=wm_radius)

                out_name = p.stem + (".png" if fmt.startswith("PNG") else ".jpg")
                out_path = outp / out_name
                saved = save_image(img_cv, out_path, fmt, jpeg_q)

                if saved:
                    log = f"✅ {i+1}/{total}: {p.name} → {out_name}"
                else:
                    log = f"❌ {i+1}/{total}: Ошибка сохранения {out_name}"
                st.session_state.logs.append(log)
                logger.info(log)
                status.info(log)

                # Preview
                with preview:
                    st.markdown(f"### {i+1}. {p.name}")
                    c1, c2, c3 = st.columns(3)
                    orig_disp = bgr_to_display(cv2.cvtColor(np.array(pil_orig), cv2.COLOR_RGBA2BGRA))
                    c1.image(orig_disp, caption="Оригинал", use_column_width=True)
                    res_disp = bgr_to_display(img_cv)
                    c2.image(res_disp, caption="Результат", use_column_width=True)
                    if mask_preview is not None:
                        c3.image(mask_preview, caption="Маска прозрачности", use_column_width=True)
                    elif remove_wm:
                        gray = cv2.cvtColor(img_cv[..., :3], cv2.COLOR_BGR2GRAY)
                        _, m = cv2.threshold(gray, wm_threshold, 255, cv2.THRESH_BINARY)
                        c3.image(m, caption="WM маска (порог)", use_column_width=True)
                    else:
                        c3.write("—")
                    # histogram as image
                    hist_img = histogram_image_rgb(res_disp[..., :3] if res_disp is not None and res_disp.ndim==3 else None)
                    st.image(hist_img, caption="Гистограмма (R/G/B)", use_column_width=False)

            except UnidentifiedImageError:
                err = f"❌ {i+1}/{total}: Невозможно открыть {p.name}"
                st.session_state.logs.append(err)
                logger.warning(err)
            except Exception as e:
                err = f"❌ {i+1}/{total}: Ошибка {p.name}: {e}"
                st.session_state.logs.append(err)
                logger.error(f"{err}\n{traceback.format_exc()}")

            progress.progress((i + 1) / total)

        status.success("Обработка завершена.")
        st.balloons()

    st.markdown("---")
    st.subheader("Журнал")
    if st.session_state.logs:
        with st.expander("Показать лог", expanded=False):
            st.code("\n".join(st.session_state.logs))
    else:
        st.info("Лог пуст. Запустите обработку.")

if __name__ == "__main__":
    main()
