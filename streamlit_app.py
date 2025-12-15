# -*- coding: utf-8 -*-
"""
watermark_tool.py

Универсальный скрипт для удаления простых водяных знаков.
- Если установлен streamlit, можно запустить веб-интерфейс:
    python watermark_tool.py --serve
  Скрипт сам попытается запустить `python -m streamlit run ... -- --streamlit-app`.
- В противном случае работает как CLI:
    python watermark_tool.py --input in.jpg --output out.png --method inpaint --thresh 150
Примечания:
- OpenCV (cv2) желателен для inpaint и морфологии маски, но не обязателен.
- Pillow (PIL) желателен для чтения/записи; если его нет, сохраняется PPM.
"""

from __future__ import annotations
import sys
import os
import argparse
import subprocess
from pathlib import Path

# Опциональные зависимости
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
        # простая имитация текста: цветной прямоугольник как "водяной знак"
        cv2_like = img
        cv2_like[300:340, 260:560] = (200, 200, 200)
        img = cv2_like
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
        # Попытка простого чтения через numpy (только для PPM/PGM)
        with open(p, "rb") as f:
            header = f.read(2)
            if header != b"P6":
                print("Без PIL и без OpenCV, поддерживаются только PPM (P6).")
                return None
        print("PIL и OpenCV не установлены; чтение ограничено PPM.")
        return None

def save_image_bgr(img_bgr: np.ndarray, out_path: str) -> None:
    """Сохранить BGR numpy в файл PNG (или PPM как запасной вариант)."""
    p = Path(out_path)
    if HAS_PIL:
        pil = Image.fromarray(img_bgr[:, :, ::-1])  # BGR -> RGB
        pil.save(str(p))
        return
    elif HAS_CV2:
        cv2.imwrite(str(p), img_bgr)
        return
    else:
        # Сохранение в PPM (P6)
        h, w = img_bgr.shape[:2]
        with open(p, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            rgb = img_bgr[:, :, ::-1].astype("uint8")
            f.write(rgb.tobytes())
        return

def make_mask_from_gray(gray: np.ndarray, thresh: int = 150, invert: bool = False, k: int = 5) -> np.ndarray:
    """Создать бинарную маску из серого изображения."""
    if HAS_CV2:
        _, m = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
        if invert:
            m = cv2.bitwise_not(m)
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        return m
    else:
        m = (gray > thresh).astype(np.uint8) * 255
        if invert:
            m = 255 - m
        return m

def inpaint_bgr(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Если доступен cv2, выполнить inpaint, иначе вернуть копию."""
    if HAS_CV2:
        return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    else:
        print("OpenCV (cv2) не установлен: inpaint недоступен, возвращаю исходное изображение.")
        return img_bgr.copy()

# -------------------- Streamlit-приложение (опционально) --------------------

def streamlit_app_entry():
    """Функция, выполняемая внутри Streamlit процесса. Импорт streamlit здесь локально."""
    import streamlit as st
    st.set_page_config(page_title="Image Watermark Remover", layout="wide")
    st.title("Image Watermark Remover (streamlit)")

    import numpy as np
    from PIL import Image
    try:
        import cv2 as st_cv2
        st_has_cv2 = True
    except Exception:
        st_cv2 = None
        st_has_cv2 = False

    def to_pil(bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(bgr[:, :, ::-1])

    uploaded = st.file_uploader("Upload image (PNG/JPG). If none uploaded, a sample will be used.", type=["png", "jpg", "jpeg"])
    if uploaded:
        data = uploaded.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        img_bgr = np.array(pil)[:, :, ::-1]
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
        st.image(Image.fromarray(mask_preview), use_column_width=True, caption="White = detected area")

    if apply:
        if method.startswith("Binary"):
            result = (cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR) if st_has_cv2 else np.stack([mask_preview]*3, axis=-1))
        else:
            if st_has_cv2:
                result = st_cv2.inpaint(img_bgr, mask_preview, 3, st_cv2.INPAINT_TELEA)
            else:
                st.warning("OpenCV не доступен: покажем маску вместо inpaint.")
                result = np.stack([mask_preview]*3, axis=-1)
        st.subheader("Result")
        st.image(to_pil(result), use_column_width=True)
        # Скачивание
        import io as _io
        buf = _io.BytesIO()
        Image.fromarray(result[:, :, ::-1]).save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download PNG", data=buf, file_name="result.png", mime="image/png")

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if HAS_CV2 else (np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8))
    mask = make_mask_from_gray(gray, thresh=args.thresh, invert=args.invert, k=args.kernel)

    if args.method == "threshold":
        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if HAS_CV2 else np.stack([mask]*3, axis=-1)
    else:
        result = inpaint_bgr(img, mask)

    out = args.output or "result.png"
    save_image_bgr(result, out)
    print(f"Готово: {out}")
    return 0

def try_launch_streamlit_here():
    """Попытаться запустить streamlit как отдельный процесс, если модуль доступен."""
    # определяем путь к этому файлу
    try:
        filepath = os.path.abspath(__file__)
    except NameError:
        filepath = os.path.abspath(sys.argv[0])
    cmd = [sys.executable, "-m", "streamlit", "run", filepath, "--", "--streamlit-app"]
    print("Запуск Streamlit командой:")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        print("streamlit не найден в окружении. Установите пакет: pip install streamlit")
    except Exception as e:
        print("Ошибка при запуске streamlit:", e)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Watermark remover (CLI or streamlit).")
    parser.add_argument("--serve", action="store_true", help="Попытаться запустить Streamlit UI (если доступен).")
    parser.add_argument("--streamlit-app", action="store_true", help="Внутренний маркер: запустить streamlit-приложение (не указывать вручную).")
    parser.add_argument("--input", "-i", help="Входной файл (PNG/JPG). Если не указан, используется пример.")
    parser.add_argument("--output", "-o", help="Выходной файл (PNG/PPM). По умолчанию result.png")
    parser.add_argument("--method", "-m", choices=("inpaint", "threshold"), default="inpaint", help="Метод: inpaint (с cv2) или threshold.")
    parser.add_argument("--thresh", type=int, default=150, help="Порог для маски (0-255).")
    parser.add_argument("--kernel", type=int, default=5, help="Размер ядра морфологии для очистки маски.")
    parser.add_argument("--invert", action="store_true", help="Инвертировать маску.")
    args = parser.parse_args(argv)

    # Если запускают внутренний streamlit маркер, импортируем и запускаем приложение здесь.
    if args.streamlit_app:
        # Это выполняется внутри процесса streamlit; streamlit уже доступен.
        try:
            streamlit_app_entry()
        except Exception as e:
            print("Ошибка в streamlit-приложении:", e)
        return

    if args.serve:
        # Попытаться найти streamlit и запустить его как отдельный процесс.
        try:
            import streamlit  # availability check
            # Запускаем как внешнюю команду, чтобы streamlit корректно поднял сервер
            try_launch_streamlit_here()
        except Exception:
            print("Streamlit не установлен или недоступен в текущем окружении.")
            print("Установите streamlit: pip install streamlit")
        return

    # Иначе режим CLI
    return_code = run_cli(args)
    return return_code

if __name__ == "__main__":
    sys.exit(main())
