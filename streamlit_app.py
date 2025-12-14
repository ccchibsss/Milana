# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — исправленная объединённая версия:
- CLI или Streamlit UI.
- Выбор входной/выходной папки, удаление фона (rembg/GrabCut), простое удаление водяных знаков (inpaint).
- Параллельная обработка (ThreadPool), валидация путей и форматов, сохранение конфигурации.
"""
from pathlib import Path
from datetime import datetime
import logging
import traceback
import io
import os
import sys
import argparse
import json
from typing import List, Optional, Tuple, Dict, Any
import concurrent.futures as cf
import multiprocessing as mp

import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

# optional rembg
try:
    from rembg import remove as rembg_remove  # type: ignore
    HAS_REMBG = True
except Exception:
    rembg_remove = None
    HAS_REMBG = False

# optional streamlit
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception:
    st = None  # type: ignore
    HAS_STREAMLIT = False

# --- Logger ---
def setup_logger():
    fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(fn, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# --- Validation / config helpers ---
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def validate_path(path: Path, is_input: bool = True) -> Tuple[bool, str]:
    try:
        if not path.exists():
            if is_input:
                return False, f"Путь не существует: {path}"
            return True, ""  # output can be created later
        if is_input and not path.is_dir():
            return False, f"Не каталог: {path}"
        # permission check
        if is_input:
            _ = next(path.iterdir(), None)
        else:
            tmp = path / ".tmp_permission_check"
            tmp.write_text("x")
            tmp.unlink()
        return True, ""
    except PermissionError:
        return False, f"Нет прав доступа: {path}"
    except Exception as e:
        return False, f"Ошибка проверки: {e}"

def validate_file_extension(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS

def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"Ошибка чтения config: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str):
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Не удалось сохранить конфигурацию")

# --- Interactive output folder chooser (CLI) ---
def choose_output_folder(base: str = ".") -> Path:
    base_p = Path(base).expanduser().resolve()
    if not base_p.exists():
        base_p.mkdir(parents=True, exist_ok=True)
    dirs = [base_p] + sorted([p for p in base_p.iterdir() if p.is_dir() and p != base_p])
    while True:
        print(f"\nБаза для выбора: {base_p}")
        print("Доступные каталоги:")
        for i, d in enumerate(dirs, start=1):
            print(f"  {i:2d}. {d}")
        print("  0. Ввести путь вручную")
        print("  c. Создать новую папку внутри базы")
        choice = input("Выберите номер, 0, c или Q для выхода: ").strip().lower()
        if choice == "q":
            raise SystemExit("Выход по запросу пользователя.")
        if choice == "0":
            p = Path(input("Введите путь (абсолютный или относительный): ").strip()).expanduser().resolve()
            if p.exists() and p.is_dir():
                print(f"Выбрана папка: {p}")
                return p
            create = input(f"Папка '{p}' не существует. Создать её? (y/N): ").strip().lower()
            if create == "y":
                p.mkdir(parents=True, exist_ok=True)
                print(f"Папка создана и выбрана: {p}")
                return p
            continue
        if choice == "c":
            name = input("Имя новой папки внутри базы: ").strip()
            if not name:
                print("Имя не указано.")
                continue
            p = base_p / name
            p.mkdir(parents=True, exist_ok=True)
            print(f"Папка создана и выбрана: {p}")
            return p
        try:
            idx = int(choice)
            if 1 <= idx <= len(dirs):
                selected = dirs[idx - 1]
                print(f"Выбрана папка: {selected}")
                return selected
            else:
                print("Неверный номер. Попробуйте ещё.")
        except ValueError:
            print("Неверный ввод. Введите номер, 0, c или Q.")

# --- Image utilities ---
def get_image_files(inp: Path) -> List[Path]:
    if not inp.exists() or not inp.is_dir():
        return []
    return [p for p in sorted(inp.iterdir()) if p.is_file() and validate_file_extension(p)]

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
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img_rgba[..., 3] = alpha
        return Image.fromarray(img_rgba)
    except Exception:
        logger.exception("grabcut failed")
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

def resize_image(img_cv: np.ndarray, target_width: Optional[int], target_height: Optional[int]) -> np.ndarray:
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

def validate_output_params(outp: Path, fmt: str, target_width: Optional[int], target_height: Optional[int]) -> Tuple[bool, str]:
    try:
        outp.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Не удалось создать родительскую директорию: {e}"
    valid_formats = {"PNG", "JPEG"}
    if fmt not in valid_formats:
        return False, f"Неподдерживаемый формат: {fmt}. Допустимые: {valid_formats}"
    if target_width is not None and (target_width <= 0 or target_width > 10000):
        return False, f"Недопустимая ширина: {target_width}. Диапазон: 1–10000"
    if target_height is not None and (target_height <= 0 or target_height > 10000):
        return False, f"Недопустимая высота: {target_height}. Диапазон: 1–10000"
    return True, ""

def save_image(img_cv: np.ndarray, out_path: Path, fmt: str, jpeg_quality: int = 95,
               target_width: Optional[int] = None, target_height: Optional[int] = None) -> bool:
    try:
        valid, msg = validate_output_params(out_path, fmt, target_width, target_height)
        if not valid:
            logger.error(msg)
            return False
        if target_width or target_height:
            img_cv = resize_image(img_cv, target_width, target_height)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt.upper().startswith("PNG"):
            cv2.imwrite(str(out_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            return True
        bgr = img_cv
        if img_cv.ndim == 3 and img_cv.shape[2] == 4:
            bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        success, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if success:
            out_path.write_bytes(buf.tobytes())
            return True
    except Exception:
        logger.exception("save_image failed")
    return False

# --- Processing tasks ---
def process_single_task(task: Tuple[str, str, Any], kwargs: Dict[str, Any]) -> str:
    src_type, name, payload = task
    try:
        inp: Path = kwargs["inp"]
        outp: Path = kwargs["outp"]
        remove_bg: bool = kwargs.get("remove_bg", True)
        remove_wm: bool = kwargs.get("remove_wm", False)
        wm_threshold: int = kwargs.get("wm_threshold", 220)
        wm_radius: int = kwargs.get("wm_radius", 5)
        fmt: str = kwargs.get("fmt", "PNG")
        jpeg_q: int = kwargs.get("jpeg_q", 95)
        tw: Optional[int] = kwargs.get("target_width", None)
        th: Optional[int] = kwargs.get("target_height", None)

        # Read image
        if src_type == "disk":
            src_path: Path = inp / name
            pil = Image.open(src_path).convert("RGBA")
        else:  # uploaded
            data = payload  # bytes or file-like
            if hasattr(data, "read"):
                buf = data.read()
            else:
                buf = data
            pil = Image.open(io.BytesIO(buf)).convert("RGBA")

        # Background removal
        processed_pil = pil
        if remove_bg:
            processed_pil = remove_background_pil(pil)
            if processed_pil.mode != "RGBA":
                processed_pil = processed_pil.convert("RGBA")

        img_cv = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGBA2BGRA)

        # Watermark removal (optional heuristic)
        if remove_wm:
            img_cv = remove_watermark_cv(img_cv, threshold=wm_threshold, radius=wm_radius)

        # Output path
        ext = ".png" if fmt.upper().startswith("PNG") else ".jpg"
        out_name = Path(name).stem + ext
        out_path = outp / out_name

        if save_image(img_cv, out_path, fmt, jpeg_q, tw, th):
            return f"✅ {name} -> {out_name}"
        else:
            return f"❌ Ошибка сохранения {name}"
    except UnidentifiedImageError:
        return f"❌ Невозможно открыть {name} (не изображение/повреждён)"
    except Exception as e:
        logger.exception(f"Ошибка в обработке {name}")
        return f"❌ Ошибка обработки {name}: {str(e)}"

def process_batch(
    input_dir: str,
    output_dir: str,
    remove_bg: bool = True,
    remove_wm: bool = False,
    wm_threshold: int = 220,
    wm_radius: int = 5,
    fmt: str = "PNG",
    jpeg_q: int = 95,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    selected_filenames: Optional[List[str]] = None,
    uploaded_files: Optional[List[Any]] = None,
    show_preview: bool = False
) -> List[str]:
    inp = Path(input_dir).expanduser().resolve()
    outp = Path(output_dir).expanduser().resolve()
    valid, msg = validate_path(inp, is_input=True)
    if not valid:
        return [f"[ОШИБКА] {msg}"]
    valid, msg = validate_path(outp, is_input=False)
    if not valid:
        return [f"[ОШИБКА] {msg}"]
    outp.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, str, Any]] = []
    if uploaded_files:
        # uploaded_files: list of (name, bytes) or streamlit UploadedFile
        for f in uploaded_files:
            if hasattr(f, "name") and hasattr(f, "read"):
                name = f.name
                data = f
            elif isinstance(f, (tuple, list)) and len(f) == 2:
                name, data = f
            else:
                continue
            if validate_file_extension(Path(name)):
                tasks.append(("uploaded", name, data))
            else:
                logger.warning(f"Пропущен файл (неподдерживаемый формат): {name}")
    else:
        files = get_image_files(inp)
        for p in files:
            name = p.name
            if selected_filenames and name not in selected_filenames:
                continue
            tasks.append(("disk", name, None))

    if not tasks:
        return ["[ПРЕДУПРЕЖДЕНИЕ] Не найдено изображений для обработки."]

    logs: List[str] = []
    max_workers = min(4, max(1, mp.cpu_count()))
    kwargs = dict(
        inp=inp,
        outp=outp,
        remove_bg=remove_bg,
        remove_wm=remove_wm,
        wm_threshold=wm_threshold,
        wm_radius=wm_radius,
        fmt=fmt,
        jpeg_q=jpeg_q,
        target_width=target_width,
        target_height=target_height,
    )
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_task, task, kwargs) for task in tasks]
        for future in cf.as_completed(futures):
            try:
                res = future.result()
                logs.append(res)
                logger.info(res)
            except Exception as e:
                logger.exception("Worker error")
                logs.append(f"❌ Неожиданная ошибка: {e}")
    return logs

# --- CLI ---
def run_cli(argv=None):
    parser = argparse.ArgumentParser(description="Photo Processor Pro (CLI)")
    parser.add_argument("--input", "-i", default="./input", help="Папка с изображениями")
    parser.add_argument("--output", "-o", default=None, help="Куда сохранять (если не указано — интерактивно)")
    parser.add_argument("--no-bg", dest="no_bg", action="store_true", help="Отключить удаление фона")
    parser.add_argument("--wm", dest="wm", action="store_true", help="Включить удаление водяных знаков")
    parser.add_argument("--wm-threshold", type=int, default=220, help="Порог для удаления водяных знаков (0–255)")
    parser.add_argument("--wm-radius", type=int, default=5, help="Радиус inpaint")
    parser.add_argument("--fmt", choices=["PNG", "JPEG"], default="PNG", help="Формат выходного изображения")
    parser.add_argument("--jpeg-q", type=int, default=95, help="Качество JPEG (1–100)")
    parser.add_argument("--width", type=int, default=None, help="Ширина выходного изображения")
    parser.add_argument("--height", type=int, default=None, help="Высота выходного изображения")
    parser.add_argument("--files", type=str, default=None, help="Список файлов через запятую (в input)")
    parser.add_argument("--config", default="config.json", help="Путь к конфигурационному файлу")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    # override args with config if arg is default/None
    for k, v in cfg.items():
        if hasattr(args, k) and getattr(args, k) in (None, "./input", "./output"):
            setattr(args, k, v)

    inp = Path(args.input).expanduser().resolve()
    valid, msg = validate_path(inp, is_input=True)
    if not valid:
        print(f"[ОШИБКА] {msg}")
        return

    if args.output:
        outp = Path(args.output).expanduser().resolve()
        valid, msg = validate_path(outp, is_input=False)
        if not valid:
            print(f"[ОШИБКА] {msg}")
            return
        outp.mkdir(parents=True, exist_ok=True)
    else:
        try:
            outp = choose_output_folder(base=".")
        except SystemExit:
            return

    selected = [s.strip() for s in args.files.split(",")] if args.files else None

    logs = process_batch(
        input_dir=str(inp),
        output_dir=str(outp),
        remove_bg=not args.no_bg,
        remove_wm=args.wm,
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

# --- Streamlit UI ---
def run_streamlit():
    if not HAS_STREAMLIT:
        print("Streamlit не установлен. Установите через `pip install streamlit`.")
        return

    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("🖼️ Photo Processor Pro")

    with st.sidebar:
        st.header("Настройки")
        config_file = st.file_uploader("Загрузить config.json", type=["json"])
        if config_file:
            try:
                cfg = json.load(config_file)
            except Exception:
                cfg = {}
        else:
            cfg = {}
        input_dir = st.text_input("Входная папка", value=cfg.get("input_dir", "./input"))
        output_dir = st.text_input("Выходная папка", value=cfg.get("output_dir", "./output"))
        remove_bg = st.checkbox("Удалить фон", value=cfg.get("remove_bg", True))
        remove_wm = st.checkbox("Удалить водяные знаки", value=cfg.get("remove_wm", False))
        wm_threshold = st.slider("Порог водяных знаков", 0, 255, cfg.get("wm_threshold", 220))
        wm_radius = st.slider("Радиус inpaint", 1, 20, cfg.get("wm_radius", 5))
        fmt = st.selectbox("Формат вывода", ["PNG", "JPEG"], index=0 if cfg.get("fmt", "PNG") == "PNG" else 1)
        jpeg_q = st.slider("Качество JPEG", 1, 100, cfg.get("jpeg_q", 95)) if fmt == "JPEG" else 95
        st.markdown("---")
        st.subheader("Изменение размера (необязательно)")
        resize_mode = st.selectbox("Режим", ("Оригинал", "Ширина", "Высота", "Оба"))
        target_width = None; target_height = None
        if resize_mode == "Ширина":
            target_width = st.number_input("Ширина (px)", min_value=1, value=cfg.get("target_width", 1920))
        elif resize_mode == "Высота":
            target_height = st.number_input("Высота (px)", min_value=1, value=cfg.get("target_height", 1080))
        elif resize_mode == "Оба":
            target_width = st.number_input("Ширина (px)", min_value=1, value=cfg.get("target_width", 1920))
            target_height = st.number_input("Высота (px)", min_value=1, value=cfg.get("target_height", 1080))
        st.markdown("---")
        st.subheader("Файлы")
        input_mode = st.radio("Источник файлов", ("Из папки", "Загрузить файлы"))
        selected_files = None
        uploaded = None
        if input_mode == "Из папки":
            p = Path(input_dir)
            if not p.exists():
                try:
                    p.mkdir(parents=True, exist_ok=True)
                    st.warning(f"Папка '{p}' создана.")
                except Exception:
                    st.error(f"Не удалось создать папку '{p}'.")
            files = get_image_files(Path(input_dir)) if Path(input_dir).exists() else []
            names = [f.name for f in files]
            selected_files = st.multiselect("Выберите файлы (если пусто — все)", options=names)
        else:
            uploaded = st.file_uploader("Загрузите файлы", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS], accept_multiple_files=True)

        run = st.button("🚀 Запустить обработку")

    if run:
        uploaded_files = None
        if input_mode != "Из папки" and uploaded:
            uploaded_files = uploaded
        selected = selected_files if selected_files else None

        valid, msg = validate_path(Path(input_dir), is_input=True)
        if not valid:
            st.error(msg); return
        valid, msg = validate_path(Path(output_dir), is_input=False)
        if not valid:
            st.error(msg); return
        Path(output_dir).mkdir(parents=True, exist_ok=True)

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
        st.success("Готово")
        for l in logs:
            if l.startswith("✅"):
                st.markdown(f"<span style='color:green'>{l}</span>", unsafe_allow_html=True)
            elif l.startswith("❌"):
                st.markdown(f"<span style='color:red'>{l}</span>", unsafe_allow_html=True)
            else:
                st.text(l)

        # preview first few processed images
        processed = [f for f in Path(output_dir).iterdir() if f.is_file() and validate_file_extension(f)]
        if processed:
            st.subheader("Превью")
            cols = st.columns(2)
            for i, f in enumerate(sorted(processed)[:8]):
                with cols[i % 2]:
                    st.image(str(f), caption=f.name, use_column_width=True)

    # Save config
    if st.sidebar.button("Сохранить конфигурацию"):
        cfg = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "remove_bg": remove_bg,
            "remove_wm": remove_wm,
            "wm_threshold": wm_threshold,
            "wm_radius": wm_radius,
            "fmt": fmt,
            "jpeg_q": jpeg_q,
            "target_width": target_width,
            "target_height": target_height,
        }
        try:
            save_config(cfg, "config.json")
            st.sidebar.success("config.json сохранён")
        except Exception as e:
            st.sidebar.error(f"Ошибка: {e}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] != "streamlit":
        run_cli(sys.argv[1:])
    else:
        run_streamlit()

if __name__ == "__main__":
    main()
