# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — объединённая версия:
- Работает в Streamlit (если установлен) или в CLI режиме.
- Поддерживает выбор файлов для обработки и выбор папки сохранения.
- Использует rembg (если доступен) или GrabCut (фоллбек).
- Удаление водяных знаков через простую inpaint-эвристику.
- Превью: гистограмма + маска, логирование.
Комментарии и сообщения на русском.
"""
from pathlib import Path
from datetime import datetime
import logging
import traceback
import io
import os
import sys
import argparse
from typing import List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

# Опционально: rembg
try:
    from rembg import remove as rembg_remove  # type: ignore
    HAS_REMBG = True
except Exception:
    rembg_remove = None
    HAS_REMBG = False

# Опционально: streamlit
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception:
    st = None  # type: ignore
    HAS_STREAMLIT = False

# Логгер
def setup_logger():
    fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(fn, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# --- Утилиты ---
def get_image_files(inp: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if not inp.exists() or not inp.is_dir():
        return []
    return [p for p in sorted(inp.iterdir()) if p.is_file() and p.suffix.lower() in exts]

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
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_LINEAR)
        mask = np.zeros(small.shape[:2], np.uint8)
        rect = (5, 5, max(1, small.shape[1] - 10), max(1, small.shape[0] - 10))
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        small_rgba = cv2.cvtColor(small, cv2.COLOR_RGB2RGBA)
        small_rgba[..., 3] = mask2 * 255
        alpha = cv2.resize(small_rgba[..., 3], (w, h), interpolation=cv2.INTER_LINEAR)
        result = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        result[..., 3] = alpha
        return Image.fromarray(result)
    except Exception:
        logger.exception("grabcut fallback failed")
        return pil_img

def remove_background_pil(pil_img: Image.Image) -> Image.Image:
    if HAS_REMBG:
        try:
            out = rembg_background(pil_img)
            if isinstance(out, Image.Image):
                return out
        except Exception:
            logger.exception("rembg crashed; falling back to grabcut")
    return grabcut_background(pil_img)

def remove_watermark_cv(img_cv: np.ndarray, threshold: int = 220, radius: int = 5) -> np.ndarray:
    try:
        bgr = img_cv[..., :3].copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
                out[..., 3] = img_cv[..., 3]
                return out
            return inpainted
        return img_cv
    except Exception:
        logger.exception("remove_watermark_cv failed")
        return img_cv

def resize_image(img_cv: np.ndarray, target_width: int = None, target_height: int = None) -> np.ndarray:
    h, w = img_cv.shape[:2]
    if target_width and target_height:
        return cv2.resize(img_cv, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA)
    if target_width and not target_height:
        scale = target_width / w
        return cv2.resize(img_cv, (int(target_width), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    if target_height and not target_width:
        scale = target_height / h
        return cv2.resize(img_cv, (max(1, int(w * scale)), int(target_height)), interpolation=cv2.INTER_AREA)
    return img_cv

def save_image(img_cv: np.ndarray, out_path: Path, fmt: str, jpeg_quality: int = 95,
               target_width: int = None, target_height: int = None) -> bool:
    try:
        if target_width or target_height:
            img_cv = resize_image(img_cv, target_width, target_height)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt.upper().startswith("PNG"):
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
    if img_cv is None:
        return None
    if img_cv.ndim == 2:
        return img_cv
    if img_cv.shape[2] == 3:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    if img_cv.shape[2] == 4:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
    return img_cv

def histogram_image_rgb(img_rgb: np.ndarray, size=(256, 120)):
    w, h = size
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)
    if img_rgb is None:
        return canvas
    if img_rgb.ndim == 2:
        hist = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, h - 10, cv2.NORM_MINMAX)
        prev = None
        for x in range(256):
            y = h - int(hist[x])
            if prev is not None:
                cv2.line(canvas, (x - 1, prev), (x, y), (200, 200, 200), 1)
            prev = y
    else:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for ch in range(3):
            hist = cv2.calcHist([img_rgb], [ch], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, h - 10, cv2.NORM_MINMAX)
            prev = None
            for x in range(256):
                y = h - int(hist[x])
                if prev is not None:
                    cv2.line(canvas, (x - 1, prev), (x, y), colors[ch], 1)
                prev = y
    return canvas

# --- Основная функция обработки: поддерживает выбор файлов и загруженные файлы
# ---
def process_batch(input_dir: str,
                  output_dir: str,
                  remove_bg: bool = True,
                  remove_wm: bool = False,
                  wm_threshold: int = 220,
                  wm_radius: int = 5,
                  fmt: str = "PNG",
                  jpeg_q: int = 95,
                  target_width: int = None,
                  target_height: int = None,
                  selected_filenames: Optional[List[str]] = None,
                  uploaded_files: Optional[List[Tuple[str, bytes]]] = None,
                  show_preview: bool = False) -> List[str]:
    """
    selected_filenames: список имён файлов в input_dir (если нужно обработать подмножество).
    uploaded_files: список (имя, bytes) — если пользователь загрузил файлы через UI.
    """
    inp = Path(input_dir)
    outp = Path(output_dir)
    logs: List[str] = []

    # Если входной папки нет — создаём и просим пользователя положить туда файлы.
    if not inp.exists():
        try:
            inp.mkdir(parents=True, exist_ok=True)
            msg = f"Входная папка '{inp}' не найдена — создана пустая папка. Поместите изображения и запустите снова."
            logger.error(msg); logs.append(msg); return logs
        except Exception:
            msg = f"Не удалось создать входную папку '{inp}'. Проверьте права."
            logger.error(msg); logs.append(msg); return logs

    outp.mkdir(parents=True, exist_ok=True)

    # Формируем задания: приоритет — uploaded_files, иначе файлы с диска (возможно фильтрация)
    tasks = []
    if uploaded_files:
        for name, data in uploaded_files:
            tasks.append(("uploaded", name, data))
    else:
        imgs = get_image_files(inp)
        if not imgs:
            msg = f"Входная папка '{inp}' не содержит поддерживаемых изображений."
            logger.warning(msg); logs.append(msg); return logs
        if selected_filenames:
            name_set = set(selected_filenames)
            imgs = [p for p in imgs if p.name in name_set]
            if not imgs:
                msg = "Нет совпадающих выбранных файлов в указанной папке."
                logger.warning(msg); logs.append(msg); return logs
        for p in imgs:
            tasks.append(("disk", p.name, p))

    total = len(tasks)
    for i, task in enumerate(tasks, start=1):
        src_type, name, payload = task
        try:
            if src_type == "uploaded":
                pil = Image.open(io.BytesIO(payload))
                pil_orig = pil.convert("RGBA")
            else:
                with Image.open(payload) as pil:
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

            out_name = Path(name).stem + (".png" if fmt.upper().startswith("PNG") else ".jpg")
            out_path = outp / out_name
            saved = save_image(img_cv, out_path, fmt, jpeg_q, target_width, target_height)

            msg = f"{'✅' if saved else '❌'} {i}/{total}: {name} -> {out_name}"
            logger.info(msg); logs.append(msg)

            # Сохраняем превьюы (гистограмма и маска) для CLI/проверки
            try:
                disp = bgr_to_display(img_cv)
                hist_img = histogram_image_rgb(disp[..., :3] if disp is not None and disp.ndim == 3 else None)
                hist_path = outp / f"{Path(name).stem}_hist.png"
                cv2.imwrite(str(hist_path), cv2.cvtColor(hist_img, cv2.COLOR_RGB2BGR))
                if mask_preview is not None:
                    mask_path = outp / f"{Path(name).stem}_mask.png"
                    cv2.imwrite(str(mask_path), mask_preview)
            except Exception:
                logger.debug("Не удалось сохранить превью (гистограмма/маска).", exc_info=True)

        except UnidentifiedImageError:
            msg = f"❌ {i}/{total}: Невозможно открыть {name} (не изображение/повреждён)"
            logger.warning(msg); logs.append(msg)
        except Exception:
            msg = f"❌ {i}/{total}: Ошибка при обработке {name}:\n{traceback.format_exc()}"
            logger.error(msg); logs.append(msg)

    return logs

# --- CLI: добавлена опция --files для подмножества ---
def run_cli(argv=None):
    parser = argparse.ArgumentParser(description="Photo Processor Pro (CLI)")
    parser.add_argument("--input", "-i", default="./input", help="Папка с изображениями")
    parser.add_argument("--output", "-o", default="./output", help="Куда сохранять")
    parser.add_argument("--no-bg", dest="remove_bg", action="store_false", help="Отключить удаление фона")
    parser.add_argument("--wm", dest="remove_wm", action="store_true", help="Включить удаление водяных знаков")
    parser.add_argument("--wm-threshold", type=int, default=220)
    parser.add_argument("--wm-radius", type=int, default=5)
    parser.add_argument("--fmt", choices=["PNG", "JPEG"], default="PNG")
    parser.add_argument("--jpeg-q", type=int, default=95)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--files", type=str, default=None,
                        help="Список имён файлов через запятую для обработки (в папке input)")
    args = parser.parse_args(argv)

    inp = Path(args.input).expanduser().resolve()
    outp = Path(args.output).expanduser().resolve()
    # Создадим input если нет (поведение как раньше)
    if not inp.exists():
        try:
            inp.mkdir(parents=True, exist_ok=True)
            msg = f"Входная папка '{inp}' не найдена — создана пустая папка. Поместите изображения и запустите снова."
            print(msg); logger.error(msg); return
        except Exception:
            msg = f"Не удалось создать входную папку '{inp}'. Проверьте права."
            print(msg); logger.error(msg); return
    outp.mkdir(parents=True, exist_ok=True)

    selected = [s.strip() for s in args.files.split(",")] if args.files else None

    logs = process_batch(
        input_dir=str(inp),
        output_dir=str(outp),
        remove_bg=args.remove_bg,
        remove_wm=args.remove_wm,
        wm_threshold=args.wm_threshold,
        wm_radius=args.wm_radius,
        fmt=args.fmt,
        jpeg_q=args.jpeg_q,
        target_width=args.width,
        target_height=args.height,
        selected_filenames=selected,
        uploaded_files=None,
        show_preview=False,
    )
    print("\n".join(logs))

# --- Streamlit UI: выбор файлов из папки или загрузка + выбор куда сохранять
# ---
def run_streamlit():
    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("🖼️ Photo Processor Pro — выбор файлов и папки сохранения")
    if "logs" not in st.session_state:
        st.session_state.logs = []

    with st.sidebar:
        st.header("Источник / Сохранение")
        input_dir = st.text_input("Входная папка (путь)", value="./input")
        output_dir = st.text_input("Выходная папка (куда сохранять)", value="./output")
        st.markdown("---")
        input_mode = st.radio("Источник файлов", ("Из папки", "Загрузить файлы"))
        selected_files = None
        uploaded = None
        if input_mode == "Из папки":
            p = Path(input_dir)
            if not p.exists():
                try:
                    p.mkdir(parents=True, exist_ok=True)
                    st.warning(f"Папка '{p}' не найдена — создана пустая. Поместите файлы туда.")
                except Exception:
                    st.error(f"Не удалось создать папку '{p}'. Проверьте права.")
            files = get_image_files(Path(input_dir)) if Path(input_dir).exists() else []
            names = [f.name for f in files]
            selected_files = st.multiselect("Выберите файлы для обработки (оставьте пустым = все)", options=names)
        else:
            uploaded = st.file_uploader("Загрузите файлы (множественная загрузка)", accept_multiple_files=True)

        st.markdown("---")
        st.header("Обработка")
        remove_bg = st.checkbox("Удалить фон (rembg если доступен)", value=True)
        if remove_bg and not HAS_REMBG:
            st.caption("rembg не установлен — используется GrabCut (фоллбек).")
        remove_wm = st.checkbox("Удалить водяные знаки (inpaint)", value=False)
        wm_radius = st.slider("Радиус inpaint", 1, 25, 5) if remove_wm else 5
        wm_threshold = st.slider("Порог яркости для маски", 120, 255, 220) if remove_wm else 220

        st.markdown("---")
        fmt = st.radio("Формат вывода", ("PNG", "JPEG"))
        jpeg_q = st.slider("Качество JPEG", 50, 100, 95) if fmt == "JPEG" else 95

        st.markdown("---")
        resize_option = st.selectbox("Изменение размера (опционально)", ("Оригинал", "Ширина", "Высота", "Оба параметра"))
        target_width = None; target_height = None
        if resize_option == "Ширина":
            target_width = st.number_input("Ширина (px)", min_value=1, value=1920)
        elif resize_option == "Высота":
            target_height = st.number_input("Высота (px)", min_value=1, value=1080)
        elif resize_option == "Оба параметра":
            target_width = st.number_input("Ширина (px)", min_value=1, value=1920)
            target_height = st.number_input("Высота (px)", min_value=1, value=1080)

        st.markdown("---")
        run = st.button("🚀 Запустить обработку")

    if run:
        uploaded_files = None
        if input_mode == "Загрузить файлы" and uploaded:
            uploaded_files = []
            for uf in uploaded:
                try:
                    data = uf.read()
                    uploaded_files.append((uf.name, data))
                except Exception:
                    st.warning(f"Не удалось прочитать загруженный файл {uf.name}")
        selected = selected_files if selected_files else None

        st.session_state.logs = []
        with st.spinner("Обработка..."):
            logs = process_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                remove_bg=remove_bg,
                remove_wm=remove_wm,
                wm_threshold=wm_threshold,
                wm_radius=wm_radius,
                fmt=fmt,
                jpeg_q=jpeg_q,
                target_width=target_width,
                target_height=target_height,
                selected_filenames=selected,
                uploaded_files=uploaded_files,
                show_preview=True,
            )
        for l in logs:
            st.session_state.logs.append(l)
        st.success("Готово. Проверьте папку сохранения.")
        # не принудительно перезапускаем, просто показываем результаты

    st.markdown("---")
    st.subheader("Журнал")
    if st.session_state.logs:
        with st.expander("Показать лог", expanded=False):
            st.code("\n".join(st.session_state.logs))
    else:
        st.info("Лог пуст. Запустите обработку.")

# Точка входа: Streamlit если доступен, иначе CLI
def main():
    if HAS_STREAMLIT:
        run_streamlit()
    else:
        run_cli(sys.argv[1:])

if __name__ == "__main__":
    main()
