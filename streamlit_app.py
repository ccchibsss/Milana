# !/usr/bin/env python3
"""
Photo Processor Pro — массовое удаление фона и водяных знаков

Исправлена обработка отсутствующих опциональных библиотек (streamlit, rembg).
Если streamlit отсутствует — доступен простой CLI.
Если rembg отсутствует — используется простой фолбэк для удаления фона (на основе порога на белый фон).
Поддерживается выбор нескольких локальных папок и рекурсивный поиск.
"""

from pathlib import Path
from datetime import datetime
import logging
import os
import traceback
import argparse

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Попытка импортировать rembg (опционально)
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    rembg_remove = None
    REMBG_AVAILABLE = False

# Попытка импортировать streamlit (опционально)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False

# --- Логирование ---
def setup_logger():
    log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("photo_processor_pro")

logger = setup_logger()

# --- Вспомогательные функции ---
def validate_paths(input_path: Path, output_path: Path) -> tuple[bool, str]:
    """Проверяет доступность путей."""
    if not input_path.exists():
        return False, f"Папка {input_path} не существует!"
    if not os.access(input_path, os.R_OK):
        return False, f"Нет доступа для чтения: {input_path}"
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Не удалось создать/доступна ли запись в: {output_path} — {e}"
    if not os.access(output_path, os.W_OK):
        return False, f"Нет доступа для записи: {output_path}"
    return True, "OK"

def get_image_files_from_dirs(dirs: list[Path], recursive: bool = False) -> list[Path]:
    """Собирает список изображений из списка директорий."""
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    found = []
    for d in dirs:
        if not d.exists() or not d.is_dir():
            continue
        if recursive:
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in img_extensions:
                    found.append(f)
        else:
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in img_extensions:
                    found.append(f)
    # убрать дубликаты и отсортировать
    return sorted(set(found), key=lambda p: p.as_posix())

def remove_background_pil(img_pil: Image.Image) -> Image.Image:
    """
    Удаляет фон. Если rembg доступен — использует его.
    Иначе — простой фолбэк: делает пиксели близкие к белому прозрачными.
    Возвращает PIL.Image (обычно RGBA).
    """
    if REMBG_AVAILABLE and rembg_remove is not None:
        try:
            out = rembg_remove(img_pil)
            if isinstance(out, Image.Image):
                return out
            # rembg может вернуть bytes; попытаемся прочитать
            try:
                from io import BytesIO
                return Image.open(BytesIO(out)).convert("RGBA")
            except Exception:
                return img_pil.convert("RGBA")
        except Exception as e:
            logger.warning(f"rembg failed, fallback will be used: {e}")
    # Фолбэк: прозрачный белый фон
    img_rgb = img_pil.convert("RGB")
    arr = np.array(img_rgb)  # H x W x 3
    # Порог: считать фон белым, если все каналы > threshold
    threshold = 240
    bg_mask = np.all(arr > threshold, axis=2)  # True for background
    alpha = (~bg_mask).astype(np.uint8) * 255
    rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba, mode="RGBA")

def remove_watermark_cv(img_cv: np.ndarray, threshold: int, radius: int) -> np.ndarray:
    """Инпейнтинг водяных знаков через OpenCV."""
    if img_cv is None:
        return img_cv
    # Обработка для цветных и одно-канальных изображений
    if img_cv.ndim == 2:
        gray = img_cv
    else:
        # Если есть альфа в 4 каналах, игнорируем альфу при поиске маски
        if img_cv.ndim == 3 and img_cv.shape[2] == 4:
            bgr_for_mask = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        else:
            bgr_for_mask = img_cv
        gray = cv2.cvtColor(bgr_for_mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    if np.any(mask):
        # Inpaint expects 1- or 3-channel BGR image; handle BGRA -> BGR
        to_inpaint = img_cv
        converted_back = False
        if img_cv.ndim == 3 and img_cv.shape[2] == 4:
            to_inpaint = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
            converted_back = True
        inpainted = cv2.inpaint(to_inpaint, mask, radius=radius, flags=cv2.INPAINT_TELEA)
        if converted_back:
            # добавить альфа обратно (если была)
            alpha = img_cv[:, :, 3]
            inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
            inpainted[:, :, 3] = alpha
        return inpainted
    return img_cv

def save_image(img_cv: np.ndarray, output_path: Path, format: str, jpeg_quality: int = 95):
    """Сохраняет изображение с учётом формата."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # PNG with alpha
        if format == "PNG (с альфа)" and img_cv.ndim == 3 and img_cv.shape[2] == 4:
            output_path = output_path.with_suffix(".png")
            cv2.imwrite(str(output_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            return

        # Если есть альфа, убрать его для JPEG
        if img_cv.ndim == 3 and img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)

        # Save JPEG
        output_path = output_path.with_suffix(".jpg")
        success, buffer = cv2.imencode(".jpg", img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if success:
            with open(output_path, "wb") as f:
                f.write(buffer.tobytes())
        else:
            raise IOError("cv2.imencode failed")
    except Exception as e:
        logger.error(f"Ошибка сохранения {output_path}: {e}")
        raise

# --- Streamlit UI ---
def main_streamlit():
    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("🖼️ Photo Processor Pro — выбор локальных папок")
    st.caption("Выберите одну или несколько локальных папок для массовой обработки изображений")

    input_dir = st.text_input("Корневая папка (локальная)", value=str(Path.cwd()))
    output_dir = st.text_input("Выходная папка", value="./output")

    input_path = Path(input_dir)
    folder_options = []
    if input_path.exists() and input_path.is_dir():
        folder_options = [str(input_path)] + [str(p) for p in sorted(input_path.iterdir()) if p.is_dir()]
    else:
        st.warning("Указанная корневая папка не найдена.")

    selected_folders = st.multiselect("Выберите папки для обработки (можно несколько)", options=folder_options,
                                      default=[str(input_path)] if str(input_path) in folder_options else [])
    recursive = st.checkbox("Рекурсивно: включать вложенные подпапки", value=False)

    st.subheader("Функции")
    remove_bg = st.checkbox("Удалить фон (rembg или фолбэк)", value=True)
    remove_wm = st.checkbox("Убрать водяные знаки (OpenCV)", value=False)
    if remove_wm:
        wm_radius = st.slider("Радиус инпейнта", 1, 15, 5)
        wm_threshold = st.slider("Порог яркости", 1, 255, 220)
    else:
        wm_radius, wm_threshold = 5, 220

    st.subheader("Вывод")
    fmt = st.radio("Формат", ("PNG (с альфа)", "JPEG (без альфа)"))
    jpeg_q = st.slider("Качество JPEG (%)", 70, 100, 95) if fmt == "JPEG (без альфа)" else 95

    st.info(f"rembg available: {REMBG_AVAILABLE}")

    if st.button("🚀 Запустить обработку"):
        if not selected_folders:
            st.error("Выберите хотя бы одну папку для обработки.")
            return

        out_path = Path(output_dir)
        ok, msg = validate_paths(Path(selected_folders[0]), out_path)
        if not ok:
            st.error(msg)
            return

        dirs = [Path(p) for p in selected_folders]
        images = get_image_files_from_dirs(dirs, recursive=recursive)
        if not images:
            st.warning("Не найдено изображений в выбранных папках.")
            return

        st.info(f"Найдено {len(images)} изображений. Обработка началась.")
        progress = st.progress(0.0)
        log_area = st.empty()
        logs: list[str] = []

        for idx, img_path in enumerate(images):
            try:
                with Image.open(img_path) as img_pil:
                    if remove_bg:
                        img_pil = remove_background_pil(img_pil)
                    # Конвертация PIL->OpenCV
                    if img_pil.mode == "RGBA":
                        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
                    else:
                        rgb = img_pil.convert("RGB")
                        img_cv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

                if remove_wm:
                    img_cv = remove_watermark_cv(img_cv, wm_threshold, wm_radius)

                out_name = img_path.stem
                out_file = out_path / out_name
                save_image(img_cv, out_file, fmt, jpeg_q)

                msg = f"✅ {idx+1}/{len(images)}: {img_path.name} → {out_file.with_suffix('.png' if fmt.startswith('PNG') else '.jpg').name}"
                logs.append(msg)
                log_area.code("\n".join(logs[-6:]))
            except UnidentifiedImageError:
                err = f"❌ {idx+1}/{len(images)}: Не удалось открыть {img_path.name}"
                logs.append(err)
                log_area.code("\n".join(logs[-6:]))
                logger.error(err)
            except Exception as e:
                err = f"❌ {idx+1}/{len(images)}: Ошибка {img_path.name} — {e}"
                logs.append(err)
                log_area.code("\n".join(logs[-6:]))
                logger.error(f"{err}\n{traceback.format_exc()}")

            progress.progress((idx + 1) / len(images))

        st.success("Обработка завершена")
        st.write("Полный лог:")
        st.code("\n".join(logs))

# --- CLI-альтернатива ---
def process_cli(input_dirs: list[str], output_dir: str,
                recursive: bool, remove_bg: bool, remove_wm: bool,
                wm_threshold: int, wm_radius: int,
                fmt: str, jpeg_q: int):
    dirs = [Path(d) for d in input_dirs]
    images = get_image_files_from_dirs(dirs, recursive=recursive)
    if not images:
        print("Нет изображений для обработки.")
        return
    out_path = Path(output_dir)
    ok, msg = validate_paths(dirs[0], out_path)
    if not ok:
        print("Ошибка путей:", msg)
        return
    logs = []
    print(f"REMBG_AVAILABLE={REMBG_AVAILABLE}")
    for idx, img_path in enumerate(images):
        try:
            with Image.open(img_path) as img_pil:
                if remove_bg:
                    img_pil = remove_background_pil(img_pil)
                if img_pil.mode == "RGBA":
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
                else:
                    rgb = img_pil.convert("RGB")
                    img_cv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

            if remove_wm:
                img_cv = remove_watermark_cv(img_cv, wm_threshold, wm_radius)

            out_name = img_path.stem
            out_file = out_path / out_name
            save_image(img_cv, out_file, fmt, jpeg_q)
            msg = f"✅ {idx+1}/{len(images)}: {img_path.name} → {out_file.with_suffix('.png' if fmt.startswith('PNG') else '.jpg').name}"
            logs.append(msg)
            print(msg)
        except UnidentifiedImageError:
            err = f"❌ {idx+1}/{len(images)}: Не удалось открыть {img_path.name}"
            logs.append(err)
            print(err)
            logger.error(err)
        except Exception as e:
            err = f"❌ {idx+1}/{len(images)}: {img_path.name} — {e}"
            logs.append(err)
            print(err)
            logger.error(f"{err}\n{traceback.format_exc()}")
    print("Готово. Лог:")
    print("\n".join(logs))

# --- Точка входа ---
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        main_streamlit()
    else:
        p = argparse.ArgumentParser(description="Photo Processor Pro — CLI режим (если нет streamlit)")
        p.add_argument("--input", "-i", nargs="+", required=False, default=[str(Path.cwd())],
                       help="Одна или несколько локальных папок для обработки")
        p.add_argument("--output", "-o", required=False, default="./output")
        p.add_argument("--recursive", action="store_true", help="Рекурсивно искать во вложенных папках")
        p.add_argument("--no-bg", dest="remove_bg", action="store_false", help="Не удалять фон")
        p.add_argument("--wm", dest="remove_wm", action="store_true", help="Удалять водяные знаки")
        p.add_argument("--wm-th", type=int, default=220, help="Порог для поиска водяных знаков")
        p.add_argument("--wm-r", type=int, default=5, help="Радиус инпейнта")
        p.add_argument("--fmt", choices=["PNG", "JPEG"], default="PNG", help="Формат вывода")
        p.add_argument("--q", type=int, default=95, help="Качество JPEG")
        args = p.parse_args()

        fmt = "PNG (с альфа)" if args.fmt == "PNG" else "JPEG (без альфа)"
        process_cli(args.input, args.output, args.recursive, args.remove_bg, args.remove_wm,
                    args.wm_th, args.wm_r, fmt, args.q)
