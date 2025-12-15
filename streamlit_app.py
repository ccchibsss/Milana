# -*- coding: utf-8 -*-
"""
watermark_tool.py

Универсальный скрипт для удаления простых водяных знаков.
Работает как CLI; при наличии streamlit можно запустить UI: python watermark_tool.py --serve
Если streamlit запускает файл (streamlit run ...), внутри будет вызван streamlit_app_entry().

Изменение: в режиме "Binary Threshold (preview only)" теперь результат
сохраняет оригинальные цвета и добавляет полупрозрачную цветовую подсветку
обнаруженной маски, вместо вывода чисто серой маски.
"""

from __future__ import annotations
import sys
import os
import io
import argparse
import subprocess
from pathlib import Path

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

def load_image(path: str) -> np.ndarray | None:
    """Загрузить изображение в BGR numpy array. Возвращает None при ошибке."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"Файл не найден: {path}")
        return None
    if HAS_PIL:
        pil = Image.open(str(p)).convert("RGB")
        arr = np.array(pil)[:, :, ::-1]  # RGB -> BGR
        return arr
    elif HAS_CV2:
        img = cv2.imread(str(p))
        return img
    else:
        print("PIL и OpenCV отсутствуют; чтение поддерживается не полностью.")
        return None

def save_image_bgr(img_bgr: np.ndarray, out_path: str) -> None:
    """Сохранить BGR numpy в файл (PIL/ OpenCV / PPM fallback)."""
    p = Path(out_path)
    if HAS_PIL:
        pil = Image.fromarray(img_bgr[:, :, ::-1])  # BGR -> RGB
        pil.save(str(p))
        return
    elif HAS_CV2:
        cv2.imwrite(str(p), img_bgr)
        return
    else:
        h, w = img_bgr.shape[:2]
        with open(p, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            rgb = img_bgr[:, :, ::-1].astype("uint8")
            f.write(rgb.tobytes())
        return

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
    """Если доступен cv2, выполнить inpaint, иначе вернуть копию."""
    if HAS_CV2:
        # cv2.inpaint ожидает uint8 mask (0 или 255)
        m = mask.astype(np.uint8)
        return cv2.inpaint(img_bgr, m, 3, cv2.INPAINT_TELEA)
    else:
        print("OpenCV (cv2) не установлен: inpaint недоступен, возвращаю исходное изображение.")
        return img_bgr.copy()

def overlay_mask_on_bgr(img_bgr: np.ndarray, mask: np.ndarray, color: tuple = (0, 0, 255), alpha: float = 0.3) -> np.ndarray:
    """
    Наложить цветную полупрозрачную маску на BGR-изображение.
    color - (B,G,R) значение 0-255 для подсветки.
    mask - uint8 0/255, single channel.
    alpha - непрозрачность маски (0..1).
    Возвращает BGR uint8.
    """
    img = img_bgr.copy().astype(np.float32)
    overlay = np.zeros_like(img, dtype=np.float32)
    # Broadcast mask to 3 channels and set color
    if mask.ndim == 2:
        m3 = np.stack([mask]*3, axis=-1) / 255.0  # 0..1
    else:
        m3 = (mask.astype(np.uint8) != 0).astype(np.float32)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    # Blend only where mask is present
    alpha_mask = (m3[..., 0] > 0).astype(np.float32) * alpha
    alpha_mask = np.expand_dims(alpha_mask, axis=-1)
    out = img * (1.0 - alpha_mask) + overlay * alpha_mask
    out = np.clip(out, 0, 255).astype(np.uint8)
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
            # keep original colors, add semi-transparent red overlay on mask
            result = overlay_mask_on_bgr(img_bgr, mask_preview, color=(0, 0, 255), alpha=0.35)
        else:
            if st_has_cv2:
                result = st_cv2.inpaint(img_bgr, mask_preview, 3, st_cv2.INPAINT_TELEA)
            else:
                st.warning("OpenCV не доступен: покажем маску вместо inpaint.")
                result = overlay_mask_on_bgr(img_bgr, mask_preview, color=(0, 0, 255), alpha=0.35)

        st.subheader("Result")
        st.image(to_pil(result), use_column_width=True)

        # Скачивание
        buf = _io.BytesIO()
        _Image.fromarray(result[:, :, ::-1]).save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download PNG", data=buf, file_name="result.png", mime="image/png")
    else:
        st.info("Adjust settings in the sidebar and click 'Apply removal' to process the image.")

# -------------------- CLI и запуск --------------------

def run_cli(args):
    """Выполнить обработку через CLI."""
    if args.input is None:
        print("Входной файл не указан: создаю тестовое изображение.")
        img = make_sample_image()
    else:
        img = load_image(args.input)
        if img is None:
            print("Не удалось загрузить изображение. Выход.")
            return 2

    gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if HAS_CV2
            else (np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)))
    mask = make_mask_from_gray(gray, thresh=args.thresh, invert=args.invert, k=args.kernel)

    if args.method == "threshold":
        # сохранить оригинальные цвета и подсветить найденную маску (полупрозрачная красная подсветка)
        result = overlay_mask_on_bgr(img, mask, color=(0, 0, 255), alpha=0.35)
    else:
        result = inpaint_bgr(img, mask)

    out = args.output or "result.png"
    save_image_bgr(result, out)
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

def main(argv=None):
    parser = argparse.ArgumentParser(description="Watermark remover (CLI or streamlit).")
    parser.add_argument("--serve", action="store_true", help="Попытаться запустить Streamlit UI (если доступен).")
    parser.add_argument("--streamlit-app", action="store_true", help="Внутренний маркер: запустить streamlit-приложение.")
    parser.add_argument("--input", "-i", help="Входной файл (PNG/JPG). Если не указан, используется пример.")
    parser.add_argument("--output", "-o", help="Выходной файл (PNG/PPM). По умолчанию result.png")
    parser.add_argument("--method", "-m", choices=("inpaint", "threshold"), default="inpaint", help="Метод: inpaint (с cv2) или threshold.")
    parser.add_argument("--thresh", type=int, default=150, help="Порог для маски (0-255).")
    parser.add_argument("--kernel", type=int, default=5, help="Размер ядра морфологии для очистки маски.")
    parser.add_argument("--invert", action="store_true", help="Инвертировать маску.")
    args = parser.parse_args(argv)

    # Если скрипт запущен внутри Streamlit (модуль streamlit уже импортирован)
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

    # Иначе режим CLI
    return run_cli(args)

if __name__ == "__main__":
    sys.exit(main())
