# -*- coding: utf-8 -*-
"""
watermark_tool.py

Исправленная и улучшенная версия:
- устранены синтаксические ошибки;
- добавлена загрузка/сохранение метаданных (EXIF, ICC) при наличии PIL;
- при возможности inpaint выполняется в Lab-пространстве (лучше сохраняет светопередачу/тон),
  а при сохранении JPEG используются высокие настройки качества и отключение сабсемплинга.
"""

from __future__ import annotations
import sys
import os
import io
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# опциональные зависимости
try:
    import numpy as np
except Exception:
    print("Требуется numpy. Установите пакет: pip install numpy")
    raise

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    Image = None
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

# -------------------- Утилиты обработки --------------------

def make_sample_image() -> np.ndarray:
    """Создать простое тестовое изображение (BGR numpy array)."""
    h, w = 400, 700
    img = np.full((h, w, 3), 230, dtype=np.uint8)
    if HAS_CV2:
        cv2.putText(img, "SAMPLE IMAGE", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 80, 200), 4, cv2.LINE_AA)
        cv2.putText(img, "WATERMARK", (300, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    else:
        img[300:340, 260:560] = (200, 200, 200)
    return img

def load_image_with_meta(path: str) -> Optional[Dict[str, Any]]:
    """
    Загрузить изображение и вернуть словарь с данными:
      {
        'img_bgr': numpy.ndarray (H,W,3) или (H,W,4) в BGR(A) порядке,
        'format': PIL format string or None,
        'mode': original PIL mode or None,
        'exif': raw exif bytes or None,
        'info': pil.info dict (may contain quality, dpi, icc_profile, etc.)
      }
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"Файл не найден: {path}")
        return None

    if HAS_PIL:
        pil = Image.open(str(p))
        pil_format = pil.format
        pil_mode = pil.mode
        info = dict(getattr(pil, "info", {}) or {})
        exif = info.get("exif", None)
        icc = info.get("icc_profile", None)

        # Конвертируем в RGB(A) чтобы не потерять альфу
        if pil_mode == "P":
            pil_conv = pil.convert("RGBA") if "A" in pil.getbands() else pil.convert("RGB")
        elif pil_mode not in ("RGB", "RGBA", "L"):
            try:
                pil_conv = pil.convert("RGBA")
            except Exception:
                pil_conv = pil.convert("RGB")
        else:
            pil_conv = pil

        arr = np.array(pil_conv)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        # RGB(A) -> BGR(A)
        img_bgr = arr[:, :, ::-1].copy()
        meta = {"img_bgr": img_bgr, "format": pil_format, "mode": pil_mode, "exif": exif, "info": info}
        if icc:
            meta["icc_profile"] = icc
        return meta
    elif HAS_CV2:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return {"img_bgr": img, "format": None, "mode": None, "exif": None, "info": {}}
    else:
        print("PIL и OpenCV отсутствуют; чтение поддерживается не полностью.")
        return None

def save_image_with_meta(img_bgr: np.ndarray, out_path: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Сохранить изображение, пытаясь сохранить формат и метаданные исходного файла.
    - img_bgr: BGR or BGRA uint8
    - meta: результат load_image_with_meta
    """
    p = Path(out_path)
    fmt = None
    exif = None
    info = {}
    icc_profile = None
    if meta:
        fmt = meta.get("format")
        exif = meta.get("exif")
        info = meta.get("info", {}) or {}
        icc_profile = meta.get("icc_profile")

    # Определим формат по расширению, если не задан
    if not fmt:
        suf = p.suffix.lower().lstrip(".")
        fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP", "tiff": "TIFF"}.get(suf, None)

    if HAS_PIL:
        arr = img_bgr
        # Перевод из BGR(A) в RGB(A)
        if arr.ndim == 3 and arr.shape[2] == 4:
            pil_img = Image.fromarray(arr[:, :, ::-1], mode="RGBA")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pil_img = Image.fromarray(arr[:, :, ::-1], mode="RGB")
        else:
            # fallback grayscale
            if arr.ndim == 3:
                pil_img = Image.fromarray(arr[:, :, 0], mode="L")
            else:
                pil_img = Image.fromarray(arr, mode="L")

        save_kwargs = {}
        if exif is not None:
            save_kwargs["exif"] = exif
        if icc_profile is not None:
            save_kwargs["icc_profile"] = icc_profile

        # Настройки по формату для минимизации потерь
        fmt_upper = (fmt or "").upper()
        if fmt_upper in ("JPEG", "JPG"):
            # Попробуем взять качество из info, иначе высокий quality
            q = info.get("quality")
            save_kwargs["quality"] = int(q) if isinstance(q, int) else 95
            # отключить сжатие хромаканалов (если поддерживается)
            try:
                save_kwargs.setdefault("subsampling", 0)
            except Exception:
                pass
            dpi = info.get("dpi")
            if dpi:
                save_kwargs.setdefault("dpi", dpi)
            save_kwargs.setdefault("optimize", True)
            # progressive can be optional
            save_kwargs.setdefault("progressive", False)
        elif fmt_upper == "PNG":
            save_kwargs.setdefault("optimize", True)

        try:
            if fmt:
                pil_img.save(str(p), format=fmt, **save_kwargs)
            else:
                pil_img.save(str(p), **save_kwargs)
        except TypeError:
            # Некоторый kwargs могут быть неподдерживаемыми в старых версиях PIL
            try:
                if fmt:
                    pil_img.save(str(p), format=fmt)
                else:
                    pil_img.save(str(p))
            except Exception as e:
                print("Ошибка при сохранении через PIL:", e)
                raise
    elif HAS_CV2:
        img_to_write = img_bgr.astype(np.uint8) if img_bgr.dtype != np.uint8 else img_bgr
        cv2.imwrite(str(p), img_to_write)
    else:
        # PPM fallback (теряем метаданные и альфу)
        h, w = img_bgr.shape[:2]
        with open(p, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            rgb = img_bgr[:, :, ::-1].astype("uint8")
            f.write(rgb.tobytes())

def make_mask_from_gray(gray: np.ndarray, thresh: int = 150, invert: bool = False, k: int = 5) -> np.ndarray:
    """Создать бинарную маску из серого изображения (uint8 0/255)."""
    if HAS_CV2:
        _, m = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
        if invert:
            m = cv2.bitwise_not(m)
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        return m.astype(np.uint8)
    else:
        m = (gray > thresh).astype(np.uint8) * 255
        if invert:
            m = 255 - m
        return m.astype(np.uint8)

def inpaint_bgr(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Выполнить inpaint, пытаясь сохранить светопередачу:
    - при наличии OpenCV: переводим в Lab, инпейнтим в Lab и возвращаем в BGR.
    - иначе возвращаем копию.
    """
    if not HAS_CV2:
        print("OpenCV (cv2) не установлен: inpaint недоступен, возвращаю исходное изображение.")
        return img_bgr.copy()

    # cv2.inpaint ожидает 8-bit 1- or 3-channel image; mask 0/255 uint8
    m = mask.astype(np.uint8)
    # Если изображение 4-канальное (BGRA) — инпейнтим только RGB
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        bgr = img_bgr[:, :, :3]
        alpha = img_bgr[:, :, 3]
    else:
        bgr = img_bgr
        alpha = None

    try:
        # Переводим в Lab для лучшей работы по светопередаче
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_inpaint = cv2.inpaint(lab, m, 3, cv2.INPAINT_TELEA)
        result_bgr = cv2.cvtColor(lab_inpaint, cv2.COLOR_LAB2BGR)
    except Exception:
        # В крайнем случае инпейнтим в исходном пространстве по каналам
        try:
            result_bgr = cv2.inpaint(bgr, m, 3, cv2.INPAINT_TELEA)
        except Exception:
            result_bgr = bgr.copy()

    if alpha is not None:
        # Восстанавливаем альфу без изменений
        result = np.dstack([result_bgr, alpha])
        return result
    return result_bgr

def overlay_mask_on_bgr(img_bgr: np.ndarray, mask: np.ndarray, color: tuple = (0, 0, 255), alpha: float = 0.3) -> np.ndarray:
    """
    Наложить цветную полупрозрачную маску на BGR-изображение.
    color - (B,G,R) значение 0-255 для подсветки.
    mask - uint8 0/255, single channel.
    alpha - непрозрачность маски (0..1).
    """
    img = img_bgr.copy().astype(np.float32)
    # Если есть альфа-канал — не трогаем его
    if img.ndim == 3 and img.shape[2] == 4:
        base = img[:, :, :3]
        alpha_channel = img[:, :, 3].copy()
    else:
        base = img
        alpha_channel = None

    overlay = np.zeros_like(base, dtype=np.float32)
    if mask.ndim == 2:
        m3 = np.stack([mask] * 3, axis=-1) / 255.0
    else:
        m3 = (mask.astype(np.uint8) != 0).astype(np.float32)
    overlay[:, :, 0] = float(color[0])
    overlay[:, :, 1] = float(color[1])
    overlay[:, :, 2] = float(color[2])
    alpha_mask = (m3[..., 0] > 0).astype(np.float32) * float(alpha)
    alpha_mask = np.expand_dims(alpha_mask, axis=-1)
    out_rgb = base * (1.0 - alpha_mask) + overlay * alpha_mask
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
    if alpha_channel is not None:
        out = np.dstack([out_rgb, alpha_channel])
    else:
        out = out_rgb
    return out

# -------------------- Streamlit-приложение (опционально) --------------------

def streamlit_app_entry():
    """Запуск UI внутри процесса Streamlit."""
    import streamlit as st
    import io as _io
    import numpy as _np
    from PIL import Image as _Image
    try:
        import cv2 as st_cv2
        st_has_cv2 = True
    except Exception:
        st_cv2 = None
        st_has_cv2 = False

    st.set_page_config(page_title="Image Watermark Remover", layout="wide")
    st.title("Image Watermark Remover (streamlit)")

    def to_pil(bgr: np.ndarray) -> _Image.Image:
        # Если BGRA — вернём RGBA; если BGR — RGB
        if bgr.ndim == 3 and bgr.shape[2] == 4:
            return _Image.fromarray(bgr[:, :, ::-1], mode="RGBA")
        return _Image.fromarray(bgr[:, :, ::-1])

    uploaded = st.file_uploader("Upload image (PNG/JPG). If none uploaded, a sample will be used.",
                               type=["png", "jpg", "jpeg"])
    if uploaded:
        data = uploaded.read()
        try:
            pil = _Image.open(_io.BytesIO(data)).convert("RGB")
            img_bgr = np.array(pil)[:, :, ::-1]
        except Exception:
            st.error("Не удалось прочитать загруженный файл.")
            st.stop()
    else:
        img_bgr = make_sample_image()

    st.sidebar.header("Settings")
    method = st.sidebar.selectbox("Method", ("Inpaint (auto mask)", "Binary Threshold (preview only)"))
    thresh = st.sidebar.slider("Threshold for mask", 0, 255, 150)
    kernel_size = st.sidebar.slider("Morph kernel", 1, 25, 5)
    invert_mask = st.sidebar.checkbox("Invert mask", False)
    apply = st.sidebar.button("Apply removal")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(to_pil(img_bgr), use_column_width=True)

    gray = (st_cv2.cvtColor(img_bgr, st_cv2.COLOR_BGR2GRAY) if st_has_cv2
            else (np.dot(img_bgr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)))
    mask_preview = make_mask_from_gray(gray, thresh, invert_mask, kernel_size)

    with col2:
        st.subheader("Mask preview")
        st.image(_Image.fromarray(mask_preview), use_column_width=True, caption="White = detected area")

    if apply:
        if method.startswith("Binary"):
            result = overlay_mask_on_bgr(img_bgr, mask_preview, color=(0, 0, 255), alpha=0.35)
        else:
            if st_has_cv2:
                result = inpaint_bgr(img_bgr, mask_preview)
            else:
                st.warning("OpenCV не доступен: покажем маску вместо inpaint.")
                result = overlay_mask_on_bgr(img_bgr, mask_preview, color=(0, 0, 255), alpha=0.35)

        st.subheader("Result")
        st.image(to_pil(result), use_column_width=True)

        buf = _io.BytesIO()
        # сохраняем результат как PNG в буфер
        pil_res = to_pil(result)
        pil_res.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        st.download_button("Download PNG", data=buf, file_name="result.png", mime="image/png")
    else:
        st.info("Adjust settings in the sidebar and click 'Apply removal' to process the image.")

# -------------------- CLI и запуск --------------------

def run_cli_improved(args) -> int:
    """CLI-режим с сохранением качества/метаданных."""
    if args.input is None:
        print("Входной файл не указан: создаю тестовое изображение.")
        meta = {"img_bgr": make_sample_image(), "format": None, "mode": None, "exif": None, "info": {}}
        img = meta["img_bgr"]
    else:
        meta = load_image_with_meta(args.input)
        if meta is None:
            print("Не удалось загрузить изображение. Выход.")
            return 2
        img = meta["img_bgr"]

    # Проверяем наличие альфа-канала
    has_alpha = (img.ndim == 3 and img.shape[2] == 4)
    if has_alpha:
        bgr = img[:, :, :3].copy()  # Отделяем BGR
        alpha = img[:, :, 3].copy()  # Сохраняем альфа
    else:
        bgr = img

    # Преобразуем в grayscale для создания маски
    if HAS_CV2:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = (np.dot(bgr[..., :3], [0.2989, 0.5870, 0.1140])).astype(np.uint8)

    # Создаём маску водяного знака
    mask = make_mask_from_gray(gray, thresh=args.thresh, invert=args.invert, k=args.kernel)

    # Применяем выбранный метод обработки
    if args.method == "threshold":
        result_bgr = overlay_mask_on_bgr(bgr, mask, color=(0, 0, 255), alpha=0.35)
    else:
        result_bgr = inpaint_bgr(bgr, mask)

    # Если был альфа-канал — восстанавливаем его в результате
    if has_alpha:
        if result_bgr.dtype != np.uint8:
            result_bgr = result_bgr.astype(np.uint8)
        result = np.dstack([result_bgr, alpha])
    else:
        result = result_bgr

    out = args.output or "result.png"
    save_image_with_meta(result, out, meta if args.input else None)
    print(f"Готово: {out}")
    return 0

def try_launch_streamlit_here():
    """Попытаться запустить streamlit как отдельный процесс."""
    try:
        filepath = os.path.abspath(__file__)
    except NameError:
        filepath = os.path.abspath(sys.argv[0])
    cmd = [sys.executable, "-m", "streamlit", "run", filepath, "--", "--streamlit-app"]
    print("Запуск Streamlit командой:")
    print(" ".join(cmd))
    try:
        subprocess.Popen(cmd)
    except FileNotFoundError:
        print("streamlit не найден в окружении. Установите пакет: pip install streamlit")
    except Exception as e:
        print("Ошибка при запуске streamlit:", e)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Watermark remover (CLI or streamlit).")
    parser.add_argument("--serve", action="store_true", help="Попытаться запустить Streamlit UI (если доступен).")
    parser.add_argument("--streamlit-app", action="store_true", help="Внутренний маркер: запустить streamlit-приложение.")
    parser.add_argument("--input", "-i", help="Входной файл (PNG/JPG). Если не указан, используется пример.")
    parser.add_argument("--output", "-o", help="Выходной файл (PNG/JPG). По умолчанию result.png")
    parser.add_argument("--method", "-m", choices=("inpaint", "threshold"), default="inpaint", help="Метод: inpaint (с cv2) или threshold.")
    parser.add_argument("--thresh", type=int, default=150, help="Порог для маски (0-255).")
    parser.add_argument("--kernel", type=int, default=5, help="Размер ядра морфологии для очистки маски.")
    parser.add_argument("--invert", action="store_true", help="Инвертировать маску.")
    args = parser.parse_args(argv)

    # Если скрипт запущен внутри Streamlit (модуль streamlit уже импортирован) или вызван напрямую
    if "streamlit" in sys.modules or args.streamlit_app:
        try:
            streamlit_app_entry()
        except Exception as e:
            print("Ошибка в streamlit-приложении:", e)
        return 0

    if args.serve:
        try:
            import streamlit  # проверка доступности
            try_launch_streamlit_here()
        except Exception:
            print("Streamlit не установлен или недоступен в текущем окружении.")
            print("Установите streamlit: pip install streamlit")
        return 0

    # CLI
    return run_cli_improved(args)

if __name__ == "__main__":
    sys.exit(main())
