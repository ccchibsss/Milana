# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — full improved script (CLI + Streamlit)
- Creates missing input/output dirs automatically (avoids "input not found")
- Uses rembg when available, otherwise GrabCut (foreground kept, background -> transparent)
- Watermark removal (simple inpaint)
- Streamlit: upload files, pick from folder, process single/all, download single or ZIP results
- CLI: optional args, uses config.json/defaults, creates dirs if missing
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional deps
try:
    from rembg import remove as rembg_remove  # type: ignore
    HAS_REMBG = True
except Exception as e:
    rembg_remove = None  # type: ignore
    HAS_REMBG = False
    logging.warning("rembg не установлен: %s", e)

try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception as e:
    st = None  # type: ignore
    HAS_STREAMLIT = False
    logging.warning("Streamlit не установлен: %s", e)

# Logger
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
SUPPORTED_TYPES = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]


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


def load_config() -> ProcessingConfig:
    cfg = ProcessingConfig()
    try:
        p = Path("config.json")
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    if k in ("inp", "outp") and v is not None:
                        setattr(cfg, k, Path(v))
                    else:
                        setattr(cfg, k, v)
    except Exception as e:
        logger.error("Ошибка чтения config.json: %s", e)
    return cfg


def ensure_dir(path: Path) -> Tuple[bool, str]:
    """Create directory if missing. Returns (True, '') on success or (False, msg)."""
    try:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            msg = f"Создана директория: {path}"
            logger.info(msg)
            return True, msg
        if not path.is_dir():
            return False, f"Путь существует и не является директорией: {path}"
        return True, ""
    except PermissionError:
        return False, f"Нет прав доступа: {path}"
    except Exception as e:
        return False, f"Ошибка создания директории {path}: {e}"


def validate_file_extension(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


# GrabCut helper that keeps foreground (subject) and makes background
# transparent
def grabcut_make_foreground_transparent(pil_img: Image.Image, downscale_max: int = 512) -> Image.Image:
    try:
        rgb = np.array(pil_img.convert("RGB"))
        h, w = rgb.shape[:2]
        scale = downscale_max / max(h, w) if max(h, w) > downscale_max else 1.0
        small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # init mask with probable background
        mask = np.full(small.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
        # rect slightly inset
        rx = max(1, int(0.02 * small_w)); ry = max(1, int(0.02 * small_h))
        rw = max(2, small_w - max(2, int(0.04 * small_w))); rh = max(2, small_h - max(2, int(0.04 * small_h)))
        rect = (rx, ry, rw, rh)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        fg_mask_small = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        small_rgba = cv2.cvtColor(small, cv2.COLOR_RGB2RGBA)
        small_rgba[..., 3] = fg_mask_small

        alpha_full = cv2.resize(small_rgba[..., 3], (w, h), interpolation=cv2.INTER_LINEAR)
        img_rgba_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        img_rgba_full[..., 3] = alpha_full
        return Image.fromarray(img_rgba_full)
    except Exception as e:
        logger.exception("grabcut_make_foreground_transparent failed: %s", e)
        try:
            return pil_img.convert("RGBA")
        except Exception:
            return pil_img


def rembg_background(pil_img: Image.Image) -> Image.Image:
    if not HAS_REMBG or rembg_remove is None:
        return pil_img
    try:
        out = rembg_remove(pil_img)
        if isinstance(out, Image.Image):
            return out
        if isinstance(out, (bytes, bytearray)):
            return Image.open(io.BytesIO(out))
        if isinstance(out, np.ndarray):
            return Image.fromarray(out)
    except Exception as e:
        logger.exception("rembg failed: %s", e)
    return pil_img


def remove_background(pil_img: Image.Image, config: ProcessingConfig) -> Image.Image:
    if config.remove_bg and HAS_REMBG:
        try:
            return rembg_background(pil_img)
        except Exception:
            logger.warning("rembg failed, fallback to grabcut")
    # ensure foreground kept, background transparent
    return grabcut_make_foreground_transparent(pil_img)


def remove_watermark(img_cv: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if not config.remove_wm:
        return img_cv
    try:
        bgr = img_cv[..., :3].copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, int(config.wm_threshold), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in contours:
            if cv2.contourArea(c) > 50:
                cv2.drawContours(mask, [c], -1, 255, -1)
        if np.any(mask):
            inpainted = cv2.inpaint(bgr, mask, int(config.wm_radius), cv2.INPAINT_TELEA)
            if img_cv.ndim == 3 and img_cv.shape[2] == 3:
                return inpainted
            out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
            if img_cv.ndim == 3 and img_cv.shape[2] == 4:
                out[..., 3] = img_cv[..., 3]
            return out
        return img_cv
    except Exception as e:
        logger.exception("remove_watermark failed: %s", e)
        return img_cv


def resize_image(img_cv: np.ndarray, target_width: Optional[int], target_height: Optional[int]) -> np.ndarray:
    if img_cv.ndim == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    h, w = img_cv.shape[:2]
    if (not target_width or target_width <= 0) and (not target_height or target_height <= 0):
        return img_cv
    if target_width and target_height:
        return cv2.resize(img_cv, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA)
    if target_width and target_width > 0:
        scale = float(target_width) / w
        return cv2.resize(img_cv, (int(target_width), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    scale = float(target_height) / h
    return cv2.resize(img_cv, (max(1, int(w * scale)), int(target_height)), interpolation=cv2.INTER_AREA)


def save_image(img_cv: np.ndarray, out_path: Path, config: ProcessingConfig) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_cv = resize_image(img_cv, config.target_width, config.target_height)

        if img_cv.ndim == 2:
            pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB))
        elif img_cv.shape[2] == 4:
            pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        fmt = config.fmt.upper()
        if fmt in ("JPEG", "JPG"):
            pil_img = pil_img.convert("RGB")
            pil_img.save(out_path, "JPEG", quality=int(config.jpeg_q), optimize=True)
        else:
            pil_img.save(out_path, fmt)
        return True
    except Exception as e:
        logger.error("Не удалось сохранить изображение %s: %s", out_path, e)
        return False


def process_image(in_path: Path, out_path: Path, config: ProcessingConfig) -> Tuple[bool, str]:
    try:
        pil_img = Image.open(in_path)
        pil_img = remove_background(pil_img, config)
        img_cv = np.array(pil_img)
        if img_cv.ndim == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        elif img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        elif img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGRA)

        img_cv = remove_watermark(img_cv, config)
        out_path = out_path.with_suffix(f".{config.fmt.lower()}")
        if save_image(img_cv, out_path, config):
            return True, ""
        return False, f"Ошибка сохранения: {out_path}"
    except UnidentifiedImageError:
        return False, f"Неподдерживаемый формат: {in_path}"
    except Exception as e:
        logger.exception("Ошибка обработки %s: %s", in_path, e)
        return False, f"Ошибка: {in_path} — {e}"


def process_batch(input_dir: Path, output_dir: Path, config: ProcessingConfig, max_workers: int = 4) -> List[Tuple[Path, bool, str]]:
    results: List[Tuple[Path, bool, str]] = []
    ok, msg = ensure_dir(input_dir)
    if not ok:
        return results
    ok, msg = ensure_dir(output_dir)
    if not ok:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for in_path in input_dir.iterdir():
            if not in_path.is_file():
                continue
            if not validate_file_extension(in_path):
                continue
            out_path = output_dir / f"{in_path.stem}.{config.fmt.lower()}"
            futures[executor.submit(process_image, in_path, out_path, config)] = in_path
        for future in as_completed(futures):
            in_path = futures[future]
            try:
                success, msg = future.result()
                results.append((in_path, success, msg))
            except Exception as e:
                results.append((in_path, False, str(e)))
    return results


def _zip_results_bytes(output_dir: Path, results: List[Tuple[Path, bool, str]], fmt: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for in_path, success, _ in results:
            if not success:
                continue
            out_name = f"{in_path.stem}.{fmt.lower()}"
            out_path = output_dir / out_name
            if out_path.exists() and out_path.is_file():
                z.write(out_path, arcname=out_name)
    buf.seek(0)
    return buf.read()


def run_cli(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Обработка изображений")
    parser.add_argument("--input", type=Path, help="Входная папка (по умолчанию из config или ./input)")
    parser.add_argument("--output", type=Path, help="Выходная папка (по умолчанию из config или ./output)")
    parser.add_argument("--remove_bg", action="store_true", help="Удалить фон")
    parser.add_argument("--remove_wm", action="store_true", help="Удалить водяные знаки")
    parser.add_argument("--fmt", type=str, help="Формат вывода (PNG/JPEG/BMP)")
    parser.add_argument("--jpeg_q", type=int, help="Качество JPEG (0-100)")
    parser.add_argument("--workers", type=int, default=4, help="Количество потоков")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.input:
        cfg.inp = args.input
    if args.output:
        cfg.outp = args.output
    if args.remove_bg:
        cfg.remove_bg = True
    if args.remove_wm:
        cfg.remove_wm = True
    if args.fmt:
        cfg.fmt = args.fmt
    if args.jpeg_q is not None:
        cfg.jpeg_q = args.jpeg_q

    # Ensure dirs exist (creates if missing)
    ok_in, msg_in = ensure_dir(cfg.inp)
    if not ok_in:
        print(f"Ошибка входной папки: {msg_in}")
        return
    ok_out, msg_out = ensure_dir(cfg.outp)
    if not ok_out:
        print(f"Ошибка выходной папки: {msg_out}")
        return

    print("Начало обработки...")
    results = process_batch(cfg.inp, cfg.outp, cfg, max_workers=args.workers)
    for path, success, msg in results:
        if success:
            print(f"✓ {path.name}")
        else:
            print(f"✗ {path.name}: {msg}")


def run_streamlit():
    if st is None:
        raise RuntimeError("Streamlit не доступен")
    st.title("Photo Processor Pro")
    st.write("Удаление фона (foreground kept), удаление водяных знаков, изменение размера. "
             "Можно выбрать серверную папку или загрузить файлы.")

    cfg0 = load_config()
    st.sidebar.header("Настройки")
    input_text = st.sidebar.text_input("Входная папка (сервер)", value=str(cfg0.inp))
    output_text = st.sidebar.text_input("Выходная папка (сервер)", value=str(cfg0.outp))
    remove_bg = st.sidebar.checkbox("Удалить фон", value=cfg0.remove_bg)
    remove_wm = st.sidebar.checkbox("Удалить водяные знаки", value=cfg0.remove_wm)
    fmt = st.sidebar.selectbox("Формат вывода", ["PNG", "JPEG", "BMP"], index=["PNG", "JPEG", "BMP"].index(cfg0.fmt if cfg0.fmt in ("PNG", "JPEG", "BMP") else "PNG"))
    jpeg_q = st.sidebar.slider("Качество JPEG (0-100)", 0, 100, int(cfg0.jpeg_q))
    target_width = st.sidebar.number_input("Ширина (px)", value=int(cfg0.target_width or 0), min_value=0, step=10)
    target_height = st.sidebar.number_input("Высота (px)", value=int(cfg0.target_height or 0), min_value=0, step=10)
    workers = st.sidebar.number_input("Потоки", value=4, min_value=1, step=1)

    st.sidebar.markdown("---")
    st.sidebar.write("Или загрузите файлы:")
    uploaded = st.sidebar.file_uploader("Загрузить изображения", type=SUPPORTED_TYPES, accept_multiple_files=True)

    temp_input_dir: Optional[Path] = None
    if uploaded:
        temp_input_dir = Path(tempfile.mkdtemp(prefix="pp_upload_"))
        for up in uploaded:
            target = temp_input_dir / up.name
            with open(target, "wb") as f:
                f.write(up.getbuffer())
        st.sidebar.success(f"Загружено {len(uploaded)} файлов (временная папка создана)")

    use_uploaded = st.sidebar.checkbox("Использовать загруженные файлы (вместо серверной папки)", value=bool(uploaded))
    input_dir = Path(input_text) if not use_uploaded else (temp_input_dir or Path("."))
    output_dir = Path(output_text)

    ok_in, msg_in = ensure_dir(input_dir)
    if msg_in:
        st.sidebar.info(msg_in)
    ok_out, msg_out = ensure_dir(output_dir)
    if msg_out:
        st.sidebar.info(msg_out)

    st.write("Файлы в текущей входной папке:")
    if not input_dir.exists() or not input_dir.is_dir():
        st.error(f"Входная папка не найдена или не каталог: {input_dir}")
        file_list = []
    else:
        file_list = sorted([p for p in input_dir.iterdir() if p.is_file() and validate_file_extension(p)], key=lambda p: p.name)
        if not file_list:
            st.info("Входная папка пуста или нет поддерживаемых изображений.")

    file_names = [p.name for p in file_list]
    selected_file = st.selectbox("Выбрать файл для одиночной обработки", options=["(ничего)"] + file_names)

    run_cfg = ProcessingConfig(
        remove_bg=remove_bg,
        remove_wm=remove_wm,
        wm_threshold=cfg0.wm_threshold,
        wm_radius=cfg0.wm_radius,
        fmt=fmt,
        jpeg_q=int(jpeg_q),
        target_width=int(target_width) if target_width > 0 else None,
        target_height=int(target_height) if target_height > 0 else None,
        inp=input_dir,
        outp=output_dir,
    )

    if st.button("Обработать выбранный файл") and selected_file and selected_file != "(ничего)":
        in_path = run_cfg.inp / selected_file
        out_path = run_cfg.outp / f"{in_path.stem}.{run_cfg.fmt.lower()}"
        st.info(f"Обработка {selected_file} ...")
        ok, msg = process_image(in_path, out_path, run_cfg)
        if ok:
            st.success("Успешно")
            try:
                with open(out_path, "rb") as f:
                    data = f.read()
                st.image(out_path, caption=out_path.name, use_column_width=True)
                st.download_button("Скачать результат", data, file_name=out_path.name, mime="application/octet-stream")
            except Exception as e:
                st.error(f"Не удалось отобразить/скачать результат: {e}")
        else:
            st.error(f"Ошибка: {msg}")

    if st.button("Обработать все (папка)"):
        st.info("Запущена пакетная обработка...")
        results = process_batch(run_cfg.inp, run_cfg.outp, run_cfg, max_workers=int(workers))
        success_count = sum(1 for _, ok, _ in results if ok)
        fail_count = len(results) - success_count
        st.write(f"Успешно: {success_count}, Ошибок: {fail_count}")
        for p, ok, m in results:
            if not ok:
                st.warning(f"{p.name}: {m}")
        if success_count > 0:
            zipdata = _zip_results_bytes(run_cfg.outp, results, run_cfg.fmt)
            st.download_button("Скачать все результаты (ZIP)", zipdata, file_name=f"results_{run_cfg.fmt.lower()}.zip", mime="application/zip")

    if uploaded and st.button("Обработать загруженные файлы и скачать ZIP"):
        if not temp_input_dir:
            st.error("Нет загруженных файлов")
        else:
            results = process_batch(temp_input_dir, run_cfg.outp, run_cfg, max_workers=int(workers))
            success_count = sum(1 for _, ok, _ in results if ok)
            fail_count = len(results) - success_count
            st.write(f"Успешно: {success_count}, Ошибок: {fail_count}")
            if success_count > 0:
                zipdata = _zip_results_bytes(run_cfg.outp, results, run_cfg.fmt)
                st.download_button("Скачать ZIP обработанных загруженных файлов", zipdata, file_name=f"uploaded_results_{run_cfg.fmt.lower()}.zip", mime="application/zip")
            for p, ok, m in results:
                if not ok:
                    st.warning(f"{p.name}: {m}")
            try:
                if temp_input_dir and temp_input_dir.exists():
                    shutil.rmtree(temp_input_dir)
            except Exception:
                pass

    st.markdown("---")
    st.write("Предпросмотр (без обработки)")
    if selected_file and selected_file != "(ничего)":
        try:
            img = Image.open(input_dir / selected_file)
            st.image(img, caption=f"Исходник: {selected_file}", use_column_width=True)
        except Exception as e:
            st.write(f"Не удалось открыть {selected_file}: {e}")


def main(argv: Optional[List[str]] = None):
    if HAS_STREAMLIT and "streamlit" in sys.modules:
        run_streamlit()
    else:
        run_cli(argv)


if __name__ == "__main__":
    main()
