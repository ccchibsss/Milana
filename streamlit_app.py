# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная версия Photo Processor Pro с исправлениями и расширенной визуализацией.
- Исправлены синтаксические ошибки и обработка исключений.
- Добавлены предпросмотры (оригинал / результат / маска), гистограмма цветов.
- Более робастная работа с форматами и сохранением.
- Логирование и прогресс корректно обновляются.
"""
import os
from pathlib import Path
from datetime import datetime
import logging
import traceback

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from rembg import remove  # требует установленный пакет rembg
import streamlit as st
import matplotlib.pyplot as plt

# --- Логгер ---
def setup_logger():
    log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# --- Вспомогательные функции ---
def validate_paths(input_path: Path, output_path: Path) -> tuple[bool, str]:
    if not input_path.exists() or not input_path.is_dir():
        return False, f"Папка '{input_path}' не существует или не является директорией."
    if not os.access(input_path, os.R_OK):
        return False, f"Нет доступа для чтения: '{input_path}'."
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Не удалось создать/доступ к папке вывода '{output_path}': {e}"
    if not os.access(output_path, os.W_OK):
        return False, f"Нет доступа для записи: '{output_path}'."
    return True, "OK"

def get_image_files(input_path: Path) -> list[Path]:
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = [p for p in sorted(input_path.iterdir()) if p.is_file() and p.suffix.lower() in img_extensions]
    return files

def remove_background_pil(img_pil: Image.Image) -> Image.Image:
    """
    Удаляет фон с помощью rembg; возвращает PIL.Image (обычно RGBA).
    Если что-то пошло не так, возвращает исходное изображение.
    """
    try:
        # rembg поддерживает PIL.Image
        out = remove(img_pil)
        if isinstance(out, Image.Image):
            return out
        # Если вернулся bytes/bytearray, преобразуем
        if isinstance(out, (bytes, bytearray)):
            return Image.open(io.BytesIO(out))
    except Exception as e:
        logger.warning(f"remove_background_pil: ошибка rembg — {e}")
    return img_pil

def remove_watermark_cv(img_cv: np.ndarray, threshold: int, radius: int) -> np.ndarray:
    """
    Простая эвристика: порог яркости -> контуры -> inpaint.
    Работает на BGR или BGRA.
    """
    try:
        bgr = img_cv[..., :3].copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        # Удалим мелкий шум
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 30:  # порог минимальной площади
                cv2.drawContours(mask, [cnt], -1, 255, -1)

        if np.any(mask):
            inpainted = cv2.inpaint(bgr, mask, radius=radius, flags=cv2.INPAINT_TELEA)
            # Если был альфа, восстановим канал альфа
            if img_cv.shape[2] == 4:
                alpha = img_cv[..., 3]
                result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
                result[..., 3] = alpha
                return result
            return inpainted
        return img_cv
    except Exception as e:
        logger.error(f"remove_watermark_cv error: {e}\n{traceback.format_exc()}")
        return img_cv

def save_image(img_cv: np.ndarray, output_path: Path, fmt: str, jpeg_quality: int = 95):
    """
    Сохраняет изображение. fmt: "PNG (с альфа)" или "JPEG (без альфа)".
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Если PNG и есть альфа канал, сохраним PNG
        if fmt == "PNG (с альфа)" and img_cv.shape[2] == 4:
            # cv2.imwrite поддерживает BGRA -> PNG
            cv2.imwrite(str(output_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            return True
        # Для JPEG, убедимся что нет альфа
        bgr = img_cv
        if img_cv.shape[2] == 4:
            bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        # Запишем JPEG с заданным качеством
        success, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if success:
            with open(output_path, "wb") as f:
                f.write(buffer)
            return True
        return False
    except Exception as e:
        logger.error(f"save_image error: {e}\n{traceback.format_exc()}")
        return False

def bgr_to_rgb_for_display(img_cv: np.ndarray):
    """Преобразует BGR/BGRA -> RGB/RGBA для streamlit."""
    if img_cv is None:
        return None
    if img_cv.ndim == 2:
        return img_cv
    if img_cv.shape[2] == 3:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    if img_cv.shape[2] == 4:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
    return img_cv

def plot_color_histogram(img_rgb):
    """Возвращает matplotlib фигуру с гистограммой цветов для RGB изображения."""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    if img_rgb is None:
        return fig
    if img_rgb.ndim == 2:
        ax.hist(img_rgb.ravel(), bins=256, color='k', alpha=0.7)
        ax.set_title("Гистограмма (градации серого)")
    else:
        colors = ("r", "g", "b")
        for i, col in enumerate(colors):
            hist, bins = np.histogram(img_rgb[..., i].ravel(), bins=256, range=(0, 256))
            ax.plot(bins[:-1], hist, color=col, linewidth=1)
        ax.set_title("Гистограмма цветов (R,G,B)")
    ax.set_xlim([0, 255])
    plt.tight_layout()
    return fig

# --- Streamlit интерфейс ---
def main():
    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("🖼️ Photo Processor Pro")
    st.markdown("Массовая обработка: удаление фона (rembg) + инпейтинг водяных знаков (OpenCV).")
    # Инициализация логов в сессии
    if "logs" not in st.session_state:
        st.session_state.logs = []

    with st.sidebar:
        st.header("⚙️ Настройки")
        input_dir = st.text_input("Входная папка", value="./input")
        output_dir = st.text_input("Выходная папка", value="./output")

        st.subheader("Функции")
        remove_bg = st.checkbox("Удалить фон (rembg)", value=True)
        remove_wm = st.checkbox("Удалить водяные знаки (inpaint)", value=False)
        if remove_wm:
            wm_radius = st.slider("Радиус инпейта", 1, 25, 5)
            wm_threshold = st.slider("Порог яркости для маски", 120, 255, 220)

        st.subheader("Формат вывода")
        fmt = st.radio("Формат", ("PNG (с альфа)", "JPEG (без альфа)"))
        jpeg_q = st.slider("Качество JPEG (%)", 50, 100, 95) if fmt == "JPEG (без альфа)" else 95

        st.markdown("---")
        run = st.button("🚀 Запустить обработку")

    # UI область для прогресса и предпросмотра
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    preview_container = st.container()

    if run:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        ok, msg = validate_paths(input_path, output_path)
        if not ok:
            st.error(msg)
            return

        images = get_image_files(input_path)
        if not images:
            st.warning("Входная папка не содержит поддерживаемых изображений.")
            return

        total = len(images)
        progress_bar = progress_placeholder.progress(0)
        status_placeholder.info(f"Начинаем обработку {total} файлов...")

        st.session_state.logs.clear()

        for idx, img_path in enumerate(images):
            try:
                # Открываем через PIL (чтобы rembg корректно работал)
                with Image.open(img_path) as pil_img:
                    orig_mode = pil_img.mode
                    pil_img = pil_img.convert("RGBA")  # работаем в RGBA для единобразия
                    orig_for_display = pil_img.copy()
                # Преобразуем в cv2 формат (BGRA)
                img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)

                mask_preview = None
                # Удаляем фон
                if remove_bg:
                    processed_pil = remove_background_pil(orig_for_display)
                    # Гарантируем, что получим RGBA
                    if processed_pil.mode != "RGBA":
                        processed_pil = processed_pil.convert("RGBA")
                    img_cv = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGBA2BGRA)
                    # Маска прозрачности для визуализации
                    alpha = np.array(processed_pil.split()[-1])
                    mask_preview = (alpha == 0).astype(np.uint8) * 255
                # Удаление водяных знаков (простая эвристика)
                if remove_wm:
                    img_cv = remove_watermark_cv(img_cv, wm_threshold, wm_radius)

                # Сохранение
                out_name = img_path.stem + (".png" if fmt == "PNG (с альфа)" else ".jpg")
                out_path = output_path / out_name
                saved = save_image(img_cv, out_path, fmt, jpeg_q)

                # Логирование
                if saved:
                    msg = f"✅ {idx + 1}/{total}: {img_path.name} → {out_name}"
                else:
                    msg = f"❌ {idx + 1}/{total}: Ошибка при сохранении {out_name}"
                st.session_state.logs.append(msg)
                logger.info(msg)

                # Обновление прогресса
                progress_bar.progress(int((idx + 1) / total * 100) / 100.0)
                status_placeholder.info(msg)

                # Визуализация: оригинал / результат / маска / гистограмма
                with preview_container.container():
                    st.markdown(f"### {idx + 1}. {img_path.name}")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    # Оригинал
                    orig_disp = bgr_to_rgb_for_display(cv2.cvtColor(np.array(orig_for_display), cv2.COLOR_RGBA2BGRA))
                    col1.image(orig_disp, caption="Оригинал", use_column_width=True)
                    # Результат
                    res_disp = bgr_to_rgb_for_display(img_cv)
                    col2.image(res_disp, caption="Результат", use_column_width=True)
                    # Маска / альфа
                    if mask_preview is not None:
                        col3.image(mask_preview, caption="Маска (прозрачность)", use_column_width=True)
                    else:
                        # Попробуем показать выделенную маску от порога (если remove_wm)
                        if remove_wm:
                            # создаём для отображения маску методом remove_watermark_cv (reuse)
                            gray = cv2.cvtColor(img_cv[..., :3], cv2.COLOR_BGR2GRAY)
                            _, m = cv2.threshold(gray, wm_threshold, 255, cv2.THRESH_BINARY)
                            col3.image(m, caption="WM маска (порог)", use_column_width=True)
                        else:
                            col3.write("—")
                    # Гистограмма (маленький график)
                    fig = plot_color_histogram(bgr_to_rgb_for_display(img_cv)[..., :3] if res_disp is not None else None)
                    st.pyplot(fig)
            except UnidentifiedImageError:
                err = f"❌ {idx + 1}/{total}: Файл {img_path.name} не является изображением или повреждён."
                st.session_state.logs.append(err)
                logger.warning(err)
            except Exception as e:
                err = f"❌ {idx + 1}/{total}: Ошибка при обработке {img_path.name}: {e}"
                st.session_state.logs.append(err)
                logger.error(f"{err}\n{traceback.format_exc()}")

        # Финал
        status_placeholder.success("Обработка завершена.")
        st.balloons()
        progress_bar.progress(1.0)

    # Показ логов (всегда доступно)
    st.markdown("---")
    st.subheader("Журнал обработки")
    if st.session_state.logs:
        with st.expander("Показать весь лог", expanded=False):
            st.code("\n".join(st.session_state.logs))
    else:
        st.info("Пока нет записей в логе. Запустите обработку.")

    st.markdown("---")
    st.markdown("""
    **Как пользоваться**
    1. Поместите изображения в папку `./input` или укажите другую.
    2. Настройте опции в боковой панели.
    3. Нажмите «🚀 Запустить обработку».
    4. Результаты появятся в указанной выходной папке.
    """)

if __name__ == "__main__":
    main()
