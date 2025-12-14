#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — оптимизированная версия (CLI + Streamlit)
- Повышена производительность за счёт ProcessPoolExecutor
- Улучшена структура кода (классы, типизация)
- Добавлена валидация конфигурации
- Оптимизировано логирование
- Упрощён UI (Streamlit)
"""


import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ProcessPoolExecutor as PPE, as_completed
import multiprocessing as mp

# Optional dependencies
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    rembg_remove = None
    HAS_REMBG = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False



# --- Конфигурация и логирование ---
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


def setup_logger() -> logging.Logger:
    fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(fn, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logger()


# --- Валидация путей и файлов ---
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def validate_path(path: Path, is_input: bool = True) -> Tuple[bool, str]:
    try:
        if not path.exists():
            if is_input:
                return False, f"Путь не существует: {path}"
            return True, ""
        if is_input and not path.is_dir():
            return False, f"Не каталог: {path}"
        # Проверка прав
        if is_input:
            next(path.iterdir(), None)
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


# --- Загрузка/сохранение конфигурации ---
def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            # Валидация схемы
            required = ["input_dir", "output_dir"]
            for key in required:
                if key not in cfg:
                    raise ValueError(f"Отсутствует ключ {key} в config")
            return cfg
    except FileNotFoundError:
        logger.warning("config.json не найден, используются значения по умолчанию")
        return {}
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Ошибка конфигурации: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str):
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.exception(f"Не удалось сохранить конфигурацию: {e}")


# --- Обработка изображений ---
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
            logger.warning("rembg не удалось, используем grabcut")
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
            inpainted = cv2.inpaint(bgr, mask, config.wmradius, cv2.INPAINT_TELEA)
                    if img_cv.ndim == 3 and img_cv.shape[2] == 4:
            out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
            out[..., 3] = img_cv[..., 3]
            return out
        return inpainted
    except Exception:
        logger.exception("remove_watermark failed")
        return img_cv

def resize_image(img_cv: np.ndarray, target_width: Optional[int], target_height: Optional[int]) -> np.ndarray:
    h, w = img_cv.shape[:2]
    if not target_width and not target_height:
        return img_cv
    
    if target_width and target_height:
        return cv2.resize(img_cv, (target_width, target_height), interpolation=cv2.INTER_AREA)
    if target_width:
        scale = target_width / w
        return cv2.resize(img_cv, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    scale = target_height / h
    return cv2.resize(img_cv, (int(w * scale), target_height), interpolation=cv2.INTER_AREA)


def save_image(
    img_cv: np.ndarray,
    out_path: Path,
    config: ProcessingConfig
) -> bool:
    try:
        # Валидация параметров
        if config.target_width and (config.target_width <= 0 or config.target_width > 10000):
            logger.error(f"Недопустимая ширина: {config.target_width}")
            return False
        if config.target_height and (config.target_height <= 0 or config.target_height > 10000):
            logger.error(f"Недопустимая высота: {config.target_height}")
            return False

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Изменение размера
        img_cv = resize_image(img_cv, config.target_width, config.target_height)

        # Сохранение
        if config.fmt.upper() == "PNG":
            cv2.imwrite(str(out_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            return True

        bgr = img_cv
        if img_cv.ndim == 3 and img_cv.shape[2] == 4:
            bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)

        success, buf = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_q)]
        )
        if success:
            out_path.write_bytes(buf.tobytes())
            return True
        return False
    except Exception as e:
        logger.exception(f"save_image failed: {e}")
        return False


# --- Обработка задач ---
def process_single_task(task: Tuple[str, str, Any], config: ProcessingConfig) -> str:
    src_type, name, payload = task
    try:
        # Чтение изображения
        if src_type == "disk":
            src_path = config.inp / name
            pil = Image.open(src_path).convert("RGBA")
        else:  # uploaded
            data = payload if hasattr(payload, "read") else payload
            if hasattr(data, "read"):
                buf = data.read()
            else:
                buf = data
            pil = Image.open(io.BytesIO(buf)).convert("RGBA")

        # Удаление фона
        processed_pil = remove_background(pil, config)


        # Преобразование в OpenCV
        img_cv = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGBA2BGRA)


        # Удаление водяных знаков
        img_cv = remove_watermark(img_cv, config)


        # Формирование пути сохранения
        ext = ".png" if config.fmt.upper() == "PNG" else ".jpg"
        out_name = Path(name).stem + ext
        out_path = config.outp / out_name


        if save_image(img_cv, out_path, config):
            return f!✅ {name} -> {out_name}"
        else:
            return f!❌ Ошибка сохранения {name}"


    except UnidentifiedImageError:
        return f!❌ Невозможно открыть {name} (не изображение/повреждён)"
    except Exception as e:
        logger.exception(f!Ошибка обработки {name}")
        return f!❌ Ошибка обработки {name}: {str(e)}"


def process_batch(
    input_dir: str,
    output_dir: str,
    config: ProcessingConfig,
    selected_filenames: Optional[List[str]] = None,
    uploaded_files: Optional[List[Any]] = None
) -> List[str]:
    inp = Path(input_dir).expanduser().resolve()
    outp = Path(output_dir).expanduser().resolve()


    # Валидация путей
    valid, msg = validate_path(inp, is_input=True)
    if not valid:
        return [f"[ОШИБКА] {msg}"]
    valid, msg = validate_path(outp, is_input=False)
    if not valid:
        return [f"[ОШИБКА] {msg}"]
    outp.mkdir(parents=True, exist_ok=True)

    # Формирование списка задач
    tasks: List[Tuple[str, str, Any]] = []
    if uploaded_files:
        for f in uploaded_files:
            if hasattr(f, "name") and hasattr(f, "read"):
                name, data = f.name, f
            elif isinstance(f, (tuple, list)) and len(f) == 2:
                name, data = f
            else:
                continue
            if validate_file_extension(Path(name)):
                tasks.append(("uploaded", name, data))
            else:
                logger.warning(f!Пропущен файл (неподдерживаемый формат): {name}")
    else:
        files = [p for p in inp.iterdir() if p.is_file() and validate_file_extension(p)]
        for p in files:
            name = p.name
            if selected_filenames and name not in selected_filenames:
                continue
            tasks.append(("disk", name, None))

    if not tasks:
        return ["[ПРЕДУПРЕЖДЕНИЕ] Не найдено изображений для обработки."]

    logs: List[str] = []
    max_workers = min(4, mp.cpu_count())

    with PPE(max_workers=max_workers) as executor:
        futures = [executor.submit(processsingle_task, task, config) for task in tasks]
        for future in as_completed(futures):
            try:
                res = future.result()
                logs.append(res)
                logger.info(res)
            except Exception as e:
                logger.exception("Worker error")
                logs.append(f!❌ Неожиданная ошибка: {e}")

    return logs

# --- CLI ---
def run_cli(argv=None):
    parser = argparse.ArgumentParser(description="Photo Processor Pro (CLI)")
    parser.add_argument("--input", "-i", default="./input", help="Папка с изображениями")
    parser.add_argument("--output", "-o", default=None, help="Куда сохранять (если не указано — интерактивно)")
    parser.add_argument("--no-bg", dest="remove_bg", action="store_false", help="Отключить удаление фона")
    parser.add_argument("--wm", dest="remove_wm", action="store_true", help="Включить удаление водяных знаков")
    parser.add_argument("--wm-threshold", type=int, default=220, help="Порог для удаления водяных знаков (0–255)")
    parser.add_argument("--wm-radius", type=int, default=5, help="Радиус inpaint")
    parser.add_argument("--fmt", choices=["PNG", "JPEG"], default="PNG", help="Формат выходного изображения")
    parser.add_argument("--jpeg-q", type=int, default=95, help="Качество JPEG (1–100)")
    parser.add_argument("--width", type=int, default
            parser.add_argument("--width", type=int, default=None, help="Ширина выходного изображения")
        parser.add_argument("--height", type=int, default=None, help="Высота выходного изображения")
        parser.add_argument("--config", default="config.json", help="Путь к конфигурационному файлу")


        args = parser.parse_args(argv)

        # Загрузка конфигурации из файла (если есть)
        cfg_data = load_config(args.config)
        config = ProcessingConfig()

        # Переопределение параметров из CLI
        config.inp = Path(args.input).expanduser().resolve()
        if args.output:
            config.outp = Path(args.output).expanduser().resolve()
        else:
            # Интерактивный выбор выходной папки
            config.outp = choose_output_folder(str(config.inp))

        config.remove_bg = args.remove_bg
        config.remove_wm = args.wm
        config.wm_threshold = args.wm_threshold
        config.wm_radius = args.wm_radius
        config.fmt = args.fmt
        config.jpeg_q = args.jpeg_q
        config.target_width = args.width
        config.target_height = args.height

        # Сохранение актуальной конфигурации
        save_config({
            "input_dir": str(config.inp),
            "output_dir": str(config.outp),
            "remove_bg": config.remove_bg,
            "remove_wm": config.remove_wm,
            "wm_threshold": config.wm_threshold,
            "wm_radius": config.wm_radius,
            "fmt": config.fmt,
            "jpeg_q": config.jpeg_q,
            "target_width": config.target_width,
            "target_height": config.target_height
        }, args.config)

        logger.info(f!Начало обработки: {config.inp} → {config.outp}")
        logs = process_batch(
            str(config.inp), str(config.outp), config, selected_filenames=None, uploaded_files=None
        )
        for log in logs:
            print(log)
        logger.info("Обработка завершена.")


    except KeyboardInterrupt:
        logger.warning("Прервано пользователем.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Неожиданная ошибка в CLI")
        sys.exit(1)



# --- Streamlit UI ---
def run_streamlit():
    if not HAS_STREAMLIT:
        st.error("Streamlit не установлен. Установите: pip install streamlit")
        return

    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("🖼️ Photo Processor Pro")


    # Боковая панель настроек
    with st.sidebar:
        st.header("Настройки обработки")
        remove_bg = st.checkbox("Удалить фон", value=True)
        remove_wm = st.checkbox("Удалить водяные знаки", value=False)
        wm_threshold = st.slider("Порог для водяных знаков", 0, 255, 220)
        wm_radius = st.slider("Радиус inpaint", 1, 20, 5)
        fmt = st.selectbox("Формат вывода", ["PNG", "JPEG"])
        jpeg_q = st.slider("Качество JPEG", 1, 100, 95) if fmt == "JPEG" else 95
        target_width = st.number_input("Ширина (px)", min_value=1, max_value=10000, value=None, step=1)
        target_height = st.number_input("Высота (px)", min_value=1, max_value=10000, value=None, step=1)


    # Основной интерфейс
    st.subheader("1. Загрузка изображений")
    uploaded_files = st.file_uploader(
        "Выберите изображения",
        accept_multiple_files=True,
        type=list(SUPPORTED_EXTENSIONS)
    )

    if uploaded_files:
        st.subheader("2. Предварительный просмотр")
        cols = st.columns(5)
        for idx, file in enumerate(uploaded_files[:10]):  # Ограничение для превью
            with cols[idx % 5]:
                try:
                    img = Image.open(file).convert("RGBA")
                    st.image(img, caption=file.name, use_column_width=True)
                except Exception:
                    st.write("❌ Не удалось отобразить")


        if st.button("Начать обработку"):
            with st.spinner("Обработка..."):
                config = ProcessingConfig(
                    remove_bg=remove_bg,
                    remove_wm=remove_wm,
                    wm_threshold=wm_threshold,
                    wm_radius=wm_radius,
                    fmt=fmt,
                    jpeg_q=jpeg_q,
                    target_width=target_width,
                    target_height=target_height,
                    inp=Path("./temp_uploaded"),
                    outp=Path("./streamlit_output")
                )

                # Временное сохранение загруженных файлов
                temp_dir = Path("./temp_uploaded")
                temp_dir.mkdir(exist_ok=True)
                for file in uploaded_files:
                    with open(temp_dir / file.name, "wb") as f:
                        f.write(file.read())

                logs = process_batch(
                    str(temp_dir), str(config.outp), config, uploaded_files=uploaded_files
                )

                # Отображение логов
                st.subheader("Результаты")
                for log in logs:
                    if "✅" in log:
                        st.success(log)
                    elif "❌" in log:
                        st.error(log)
                    else:
                        st.info(log)

                # Ссылка для скачивания ZIP
                try:
                    zip_path = create_zip_of_output(str(config.outp))
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="Скачать ZIP-архив",
                            data=f,
                            file_name=zip_path.name,
                            mime="application/zip"
                        )
                except Exception as e:
                    st.error(f"Не удалось создать архив: {e}")



# --- Утилиты ---
def choose_output_folder(base: str = ".") -> Path:
    base_p = Path(base).expanduser().resolve()
    if not base_p.exists():
        base_p.mkdir(parents=True, exist_ok=True)
    dirs = [base_p] + sorted([p for p in base_p.iterdir() if p.is_dir() and p != base_p])

    st.info(f"База для выбора: {base_p}")
    for i, d in enumerate(dirs, start=1):
        st.write(f"{i:2d}. {d}")
    st.write("0. Ввести путь вручную")
    st.write("c. Создать новую папку внутри базы")

    choice = st.text_input("Выберите номер, 0, c или Q для выхода").strip().lower()
    if choice == "q":
        st.stop()
    if choice == "0":
        p = Path(st.text_input("Введите путь:").strip()).expanduser().resolve()
        if p.exists() and p.is_dir():
            st.success(f"Выбрана папка: {p}")
            return p
        create = st.text_input(f"Папка '{p}' не существует. Создать? (y/N)").strip().lower()
        if create == "y":
            p.mkdir(parents=True, exist_ok=True)
            st.success(f"Папка создана и выбрана: {p}")
            return p
    elif choice == "c":
        name = st.text_input("Имя новой папки:").strip()
        if name:
            p = base_p / name
            p.mkdir(parents=True, exist_ok=True)
            st.success(f"Папка создана и выбрана: {p}")
            return p
        else:
            st.warning("Имя не указано.")
    else:
        try:
            idx = int(choice)
            if 1 <= idx <= len(dirs):
                selected = dirs[idx - 1
                                return selected
        except (ValueError, IndexError):
            st.warning("Неверный выбор. Попробуйте ещё раз.")


    st.error("Не удалось выбрать папку. Обработка прервана.")
    st.stop()

def create_zip_of_output(output_dir: str, zip_name: Optional[str] = None) -> Path:
    """
    Создать ZIP-архив выходной папки во временной директории системы.
    Возвращает путь к созданному архиву.
    """
    outp = Path(output_dir).expanduser().resolve()
    if not outp.exists() or not outp.is_dir():
        raise FileNotFoundError(f"Выходная папка не найдена: {outp}")

    base_name = zip_name or f"{outp.name}_results"
    tmp_dir = Path(tempfile.gettempdir())
    zip_base = tmp_dir / base_name
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(outp))
    return Path(zip_path)


# --- Точка входа ---
def main():
    parser = argparse.ArgumentParser(description="Photo Processor Pro")
    parser.add_argument("--mode", choices=["cli", "streamlit"], default="cli",
                        help="Режим работы (cli или streamlit)")
    args = parser.parse_args()


    if args.mode == "streamlit":
        if not HAS_STREAMLIT:
            print("Ошибка: Streamlit не установлен. Установите через `pip install streamlit`")
            sys.exit(1)
        run_streamlit()
    else:
        run_cli()

if __name__ == "__main__":
    main()
