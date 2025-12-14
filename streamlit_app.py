# !/usr/bin/env python3
"""
Photo Processor Pro — полный скрипт со «меню» и боковой панелью (Streamlit).
Добавлена гибкая настройка куда сохранять обработанные файлы:
- В отдельную выходную папку (по умолчанию)
- Рядом с оригиналом (с суффиксом)
- В выходную папку с зеркальной структурой входных папок
Поддерживает CLI-режим, если streamlit не установлен.
"""

from pathlib import Path
from datetime import datetime
import logging
import os
import traceback
import argparse
from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# optional imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False

try:
    from rembg import remove as rembg_remove  # type: ignore
    REMBG_AVAILABLE = True
except Exception:
    rembg_remove = None
    REMBG_AVAILABLE = False

# logger
def setup_logger():
    fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(fn, encoding="utf-8"), logging.StreamHandler()]
    )
    return logging.getLogger("photo_processor_pro")

logger = setup_logger()

# helpers
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def validate_paths(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    if not input_path.exists():
        return False, f"Папка {input_path} не существует"
    if not os.access(input_path, os.R_OK):
        return False, f"Нет доступа на чтение: {input_path}"
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Не удалось создать выходную папку {output_path}: {e}"
    if not os.access(output_path, os.W_OK):
        return False, f"Нет доступа на запись: {output_path}"
    return True, "OK"

def list_subfolders(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    subs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    return [root] + subs

def get_image_files_from_dirs(dirs: List[Path], recursive: bool=False) -> List[Path]:
    found = []
    for d in dirs:
        if not d.exists() or not d.is_dir():
            continue
        if recursive:
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    found.append(f)
        else:
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    found.append(f)
    return sorted(set(found), key=lambda p: p.as_posix())

def find_input_root_for_path(p: Path, input_dirs: List[Path]) -> Optional[Path]:
    """Вернуть тот входной корень, которому принадлежит p (или ближайший ancestor)."""
    p_resolved = p.resolve()
    for root in sorted(input_dirs, key=lambda r: len(str(r)), reverse=True):
        try:
            if p_resolved.is_relative_to(root.resolve()):  # Python 3.9+
                return root
        except Exception:
            try:
                p_resolved.relative_to(root.resolve())
                return root
            except Exception:
                continue
    return None

def compute_output_path(original: Path, out_root: Path, save_mode: str,
                        input_roots: List[Path], suffix: str = "_proc") -> Path:
    """
    save_mode:
      - "out" — все в out_root
      - "inplace" — рядом с оригиналом, имя + suffix
      - "mirror" — в out_root с зеркальной структурой относительно одного из input_roots
    Возвращает Path без расширения (функция save_image добавит расширение).
    """
    if save_mode == "inplace":
        return original.parent / f"{original.stem}{suffix}"
    if save_mode == "mirror":
        root = find_input_root_for_path(original, input_roots)
        if root:
            try:
                rel = original.relative_to(root)
                target_dir = out_root / rel.parent
                return target_dir / original.stem
            except Exception:
                pass
        # fallback to flat out_root
        return out_root / original.stem
    # default "out"
    return out_root / original.stem

def remove_background_pil(img_pil: Image.Image) -> Image.Image:
    if REMBG_AVAILABLE and rembg_remove is not None:
        try:
            out = rembg_remove(img_pil)
            if isinstance(out, Image.Image):
                return out.convert("RGBA")
            try:
                return Image.open(BytesIO(out)).convert("RGBA")
            except Exception:
                return img_pil.convert("RGBA")
        except Exception as e:
            logger.warning(f"rembg failed, using fallback: {e}")
    rgb = img_pil.convert("RGB")
    arr = np.array(rgb)
    thr = 240
    bg_mask = np.all(arr > thr, axis=2)
    alpha = (~bg_mask).astype(np.uint8) * 255
    rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba, "RGBA")

def remove_watermark_cv(img_cv: np.ndarray, threshold: int, radius: int) -> np.ndarray:
    if img_cv is None:
        return img_cv
    if img_cv.ndim == 2:
        gray = img_cv
    else:
        if img_cv.shape[2] == 4:
            bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        else:
            bgr = img_cv
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    if np.any(mask):
        to_inpaint = img_cv
        converted = False
        if img_cv.ndim == 3 and img_cv.shape[2] == 4:
            to_inpaint = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
            converted = True
        inpainted = cv2.inpaint(to_inpaint, mask, radius=radius, flags=cv2.INPAINT_TELEA)
        if converted:
            alpha = img_cv[:, :, 3]
            inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
            inpainted[:, :, 3] = alpha
        return inpainted
    return img_cv

def save_image(img_cv: np.ndarray, out_path_base: Path, fmt: str, jpeg_q: int=95) -> Path:
    """Сохраняет изображение, возвращает фактический путь файла."""
    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "PNG (с альфа)" and img_cv.ndim == 3 and img_cv.shape[2] == 4:
        out_path = out_path_base.with_suffix(".png")
        cv2.imwrite(str(out_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return out_path
    if img_cv.ndim == 3 and img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
    out_path = out_path_base.with_suffix(".jpg")
    ok, buf = cv2.imencode(".jpg", img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
    if not ok:
        raise IOError("cv2.imencode failed")
    out_path.write_bytes(buf.tobytes())
    return out_path

def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    if img_pil.mode == "RGBA":
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
    else:
        rgb = img_pil.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

# Streamlit UI
def main_streamlit():
    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("🖼️ Photo Processor Pro")
    st.sidebar.title("Меню")
    page = st.sidebar.radio("Раздел", ["Обзор", "Настройки", "Обработка", "О программе"])

    if page == "Обзор":
        st.header("Обзор")
        st.write("Выберите раздел 'Настройки' в боковой панели, чтобы указать папки/файлы.")
        st.info(f"rembg available: {REMBG_AVAILABLE}")
    elif page == "Настройки":
        st.header("Настройки и выбор файлов")
        root = st.text_input("Корневая папка (локальная)", value=str(Path.cwd()))
        root_p = Path(root)
        folder_options = [str(p) for p in list_subfolders(root_p)]
        selected = st.multiselect("Выберите папки для поиска изображений", options=folder_options, default=[str(root_p)] if str(root_p) in folder_options else [])
        recursive = st.checkbox("Рекурсивно искать в подпапках", value=False)
        uploaded = st.file_uploader("Загрузите файлы (опционально)", accept_multiple_files=True, type=[e.strip(".") for e in IMG_EXTS])
        st.session_state["selected_folders"] = selected
        st.session_state["recursive"] = recursive
        st.session_state["uploaded"] = uploaded
        st.success("Настройки сохранены в сессии. Перейдите в 'Обработка' для запуска.")
    elif page == "Обработка":
        st.header("Запуск обработки")
        selected = st.session_state.get("selected_folders", [str(Path.cwd())])
        recursive = st.session_state.get("recursive", False)
        uploaded = st.session_state.get("uploaded", [])

        st.subheader("Выбор папок / файлов")
        root = st.text_input("Корневая папка (локальная)", value=str(Path.cwd()))
        root_p = Path(root)
        folder_options = [str(p) for p in list_subfolders(root_p)]
        selected = st.multiselect("Папки для обработки", options=folder_options, default=selected or [str(root_p)])
        recursive = st.checkbox("Рекурсивно", value=recursive)

        st.subheader("Куда сохранять обработанные файлы?")
        save_mode = st.selectbox("Режим сохранения", [
            ("out", "В отдельную выходную папку (по умолчанию)"),
            ("inplace", "Рядом с оригиналом (добавить суффикс)"),
            ("mirror", "В выходную папку с зеркальной структурой")
        ], format_func=lambda x: x[1])[0]  # store keys
        output_dir = st.text_input("Выходная папка (используется для режимов out/mirror)", value="./output")
        fname_suffix = st.text_input("Суффикс для inplace (например _proc)", value="_proc")

        st.subheader("Функции обработки")
        remove_bg = st.checkbox("Удалить фон (rembg или фолбэк)", value=True)
        remove_wm = st.checkbox("Убрать водяные знаки (OpenCV)", value=False)
        if remove_wm:
            wm_radius = st.slider("Радиус инпейнта", 1, 15, 5)
            wm_threshold = st.slider("Порог для маски (0-255)", 1, 255, 220)
        else:
            wm_radius, wm_threshold = 5, 220

        fmt = st.radio("Формат вывода", ("PNG (с альфа)", "JPEG (без альфа)"))
        jpeg_q = st.slider("Качество JPEG (%)", 70, 100, 95) if fmt == "JPEG (без альфа)" else 95

        uploaded_local = st.file_uploader("Или загрузите отдельные файлы (опционально)", accept_multiple_files=True, type=[e.strip(".") for e in IMG_EXTS])

        if st.button("🚀 Запустить"):
            dirs = [Path(p) for p in selected]
            images = get_image_files_from_dirs(dirs, recursive=recursive)
            mem = []
            if uploaded_local:
                for uf in uploaded_local:
                    try:
                        b = uf.read()
                        mem.append({"name": uf.name, "bytes": b})
                    except Exception as e:
                        logger.error(f"Ошибка чтения загруженного файла {uf.name}: {e}")
            if not images and not mem:
                st.warning("Нет изображений для обработки.")
                return

            out_path = Path(output_dir)
            if save_mode in ("out", "mirror"):
                ok, msg = validate_paths(dirs[0] if dirs else Path.cwd(), out_path)
                if not ok:
                    st.error(msg)
                    return
            st.info(f"Найдено {len(images)} файлов на диске и {len(mem)} загруженных файлов.")
            progress = st.progress(0.0)
            log_box = st.empty()
            logs: List[str] = []
            total = len(images) + len(mem)
            idx = 0

            for p in images:
                try:
                    with Image.open(p) as pil:
                        if remove_bg:
                            pil = remove_background_pil(pil)
                        img_cv = pil_to_cv(pil)
                        if remove_wm:
                            img_cv = remove_watermark_cv(img_cv, wm_threshold, wm_radius)
                        out_base = compute_output_path(p, out_path, save_mode, dirs, suffix=fname_suffix)
                        out_file = save_image(img_cv, out_base, fmt, jpeg_q)
                        msg = f"✅ {idx+1}/{total}: {p.name} → {out_file}"
                        logs.append(msg)
                        log_box.code("\n".join(logs[-10:]))
                        idx += 1
                except UnidentifiedImageError:
                    err = f"❌ {idx+1}/{total}: Не удалось открыть {p.name}"
                    logs.append(err); log_box.code("\n".join(logs[-10:])); logger.error(err); idx += 1
                except Exception as e:
                    err = f"❌ {idx+1}/{total}: Ошибка {p.name} — {e}"
                    logs.append(err); log_box.code("\n".join(logs[-10:])); logger.error(traceback.format_exc()); idx += 1
                progress.progress(idx / total)

            for mf in mem:
                try:
                    pil = Image.open(BytesIO(mf["bytes"]))
                    if remove_bg:
                        pil = remove_background_pil(pil)
                    img_cv = pil_to_cv(pil)
                    if remove_wm:
                        img_cv = remove_watermark_cv(img_cv, wm_threshold, wm_radius)
                    # For uploaded files we can't mirror original structure; use out_path or inplace isn't applicable
                    if save_mode == "inplace":
                        # save next to current working directory
                        out_base = Path.cwd() / f"{Path(mf['name']).stem}{fname_suffix}"
                    else:
                        out_base = compute_output_path(Path(mf["name"]), out_path, save_mode, dirs, suffix=fname_suffix)
                    out_file = save_image(img_cv, out_base, fmt, jpeg_q)
                    msg = f"✅ {idx+1}/{total}: {mf['name']} → {out_file}"
                    logs.append(msg); log_box.code("\n".join(logs[-10:])); idx += 1
                except Exception as e:
                    err = f"❌ {idx+1}/{total}: Ошибка {mf['name']} — {e}"
                    logs.append(err); log_box.code("\n".join(logs[-10:])); logger.error(traceback.format_exc()); idx += 1
                progress.progress(idx / total)

            st.success("Обработка завершена")
            st.code("\n".join(logs))
            st.write("Выходная папка:")
            try:
                for f in sorted(Path(output_dir).rglob("*")):
                    if f.is_file():
                        st.write(f.relative_to(Path(output_dir)))
            except Exception:
                pass
    else:
        st.header("О программе")
        st.markdown(
            """
            Photo Processor Pro — приложение для массовой обработки изображений.
            Поддерживает удаление фона (rembg если доступен) и простое удаление
            водяных знаков через OpenCV. Работает в Streamlit и в CLI.
            """
        )

# CLI fallback
def process_cli(input_dirs: List[str], output_dir: str, recursive: bool,
                remove_bg: bool, remove_wm: bool, wm_threshold: int, wm_radius: int,
                fmt: str, jpeg_q: int, save_mode: str, suffix: str):
    dirs = [Path(d) for d in input_dirs]
    images = get_image_files_from_dirs(dirs, recursive=recursive)
    if not images:
        print("Нет изображений для обработки.")
        return
    out_path = Path(output_dir)
    if save_mode in ("out", "mirror"):
        ok, msg = validate_paths(dirs[0], out_path)
        if not ok:
            print("Ошибка путей:", msg); return
    print(f"REMBG_AVAILABLE={REMBG_AVAILABLE}")
    logs = []
    for idx, p in enumerate(images):
        try:
            with Image.open(p) as pil:
                if remove_bg:
                    pil = remove_background_pil(pil)
                img_cv = pil_to_cv(pil)
                if remove_wm:
                    img_cv = remove_watermark_cv(img_cv, wm_threshold, wm_radius)
                out_base = compute_output_path(p, out_path, save_mode, dirs, suffix=suffix)
                out_file = save_image(img_cv, out_base, fmt, jpeg_q)
                msg = f"✅ {idx+1}/{len(images)}: {p.name} → {out_file}"
                logs.append(msg); print(msg)
        except Exception as e:
            err = f"❌ {idx+1}/{len(images)}: {p.name} — {e}"
            logs.append(err); print(err); logger.error(traceback.format_exc())
    print("Готово. Лог:")
    print("\n".join(logs))

# entrypoint
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        main_streamlit()
    else:
        parser = argparse.ArgumentParser(description="Photo Processor Pro — CLI")
        parser.add_argument("-i", "--input", nargs="+", default=[str(Path.cwd())], help="папки для обработки")
        parser.add_argument("-o", "--output", default="./output", help="выходная папка")
        parser.add_argument("-r", "--recursive", action="store_true", help="рекурсивно")
        parser.add_argument("--no-bg", dest="remove_bg", action="store_false", help="не удалять фон")
        parser.add_argument("--wm", dest="remove_wm", action="store_true", help="удалять водяные знаки")
        parser.add_argument("--wm-th", type=int, default=220, help="порог для водяных знаков")
        parser.add_argument("--wm-r", type=int, default=5, help="радиус инпейнта")
        parser.add_argument("--fmt", choices=["PNG", "JPEG"], default="PNG", help="формат вывода")
        parser.add_argument("--q", type=int, default=95, help="качество JPEG")
        parser.add_argument("--save-mode", choices=["out", "inplace", "mirror"], default="out",
                            help="куда сохранять: out (в выходную папку), inplace (рядом, с суффиксом), mirror (зеркальная структура)")
        parser.add_argument("--suffix", default="_proc", help="суффикс для inplace")
        args = parser.parse_args()
        fmt = "PNG (с альфа)" if args.fmt == "PNG" else "JPEG (без альфа)"
        process_cli(args.input, args.output, args.recursive, args.remove_bg, args.remove_wm,
                    args.wm_th, args.wm_r, fmt, args.q, args.save_mode, args.suffix)
