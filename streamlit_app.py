# !/usr/bin/env python3
"""
Photo Processor Pro — массовое удаление фона и водяных знаков
"""

import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from rembg import remove
from pathlib import Path
import logging
from datetime import datetime
import traceback

# --- Настройка логгера ---
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
    return logging.getLogger(__name__)

logger = setup_logger()

# --- Вспомогательные функции ---
def validate_paths(input_path: Path, output_path: Path) -> tuple[bool, str]:
    """Проверяет доступность путей."""
    if not input_path.exists():
        return False, f"Папка {input_path} не существует!"
    if not os.access(input_path, os.R_OK):
        return False, f"Нет доступа для чтения: {input_path}"
    output_path.mkdir(parents=True, exist_ok=True)
    if not os.access(output_path, os.W_OK):
        return False, f"Нет доступа для записи: {output_path}"
    return True, "OK"

def get_image_files(input_path: Path) -> list[Path]:
    """Возвращает список поддерживаемых изображений."""
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in img_extensions
    ]

def remove_background_pil(img_pil: Image.Image) -> Image.Image:
    """Удаляет фон через rembg."""
    return remove(img_pil)

def remove_watermark_cv(img_cv: np.ndarray, threshold: int, radius: int) -> np.ndarray:
    """Инпейнтинг водяных знаков через OpenCV."""
    if img_cv.ndim == 2:
        gray = img_cv
    else:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    if np.any(mask):
        img_cv = cv2.inpaint(img_cv, mask, radius=radius, flags=cv2.INPAINT_TELEA)
    return img_cv

def save_image(img_cv: np.ndarray, output_path: Path, format: str, jpeg_quality: int = 95):
    """Сохраняет изображение с учётом формата."""
    try:
        # Ensure parent exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # PNG with alpha
        if format == "PNG (с альфа)" and img_cv.ndim == 3 and img_cv.shape[2] == 4:
            output_path = output_path.with_suffix(".png")
            cv2.imwrite(str(output_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            return

        # Convert BGRA->BGR if needed
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

# --- Основной интерфейс Streamlit ---
def main():
    st.set_page_config(page_title="Photo Processor Pro", layout="wide")

    st.title("🖼️ Photo Processor Pro")
    st.caption("Массовая обработка изображений: удаление фона и водяных знаков")

    logs: list[str] = []

    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки обработки")

        input_dir = st.text_input("Входная папка", value="./input")
        output_dir = st.text_input("Выходная папка", value="./output")

        st.subheader("Функции")
        remove_bg = st.checkbox("Удалить фон", value=True)
        remove_wm = st.checkbox("Убрать водяные знаки", value=False)

        if remove_wm:
            wm_radius = st.slider("Радиус инпейнта", 1, 15, 5)
            wm_threshold = st.slider("Порог яркости", 180, 255, 220)
        else:
            wm_radius = 5
            wm_threshold = 220

        st.subheader("Вывод")
        fmt = st.radio("Формат", ("PNG (с альфа)", "JPEG (без альфа)"))
        jpeg_q = st.slider("Качество JPEG (%)", 70, 100, 95) if fmt == "JPEG (без альфа)" else 95

        st.divider()
        if st.button("🚀 Запустить обработку"):
            # Валидация
            input_path = Path(input_dir)
            output_path = Path(output_dir)

            is_valid, msg = validate_paths(input_path, output_path)
            if not is_valid:
                st.error(f"❌ {msg}")
                return

            # Поиск файлов
            images = get_image_files(input_path)
            if not images:
                st.warning("⚠️ Нет изображений для обработки.")
                return

            st.info(f"📂 Найдено: {len(images)} файлов")

            # Прогресс и лог
            progress_bar = st.progress(0.0)
            status_box = st.empty()
            log_area = st.empty()

            # Обработка
            for idx, img_path in enumerate(images):
                try:
                    # Чтение
                    with Image.open(img_path) as img_pil:
                        # Если нужно удалить фон, передаём оригинал в rembg
                        if remove_bg:
                            img_pil = remove_background_pil(img_pil)
                        # Конвертируем PIL->OpenCV (BGR or BGRA)
                        if img_pil.mode == "RGBA":
                            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
                        else:
                            rgb = img_pil.convert("RGB")
                            img_cv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

                    # Водяные знаки
                    if remove_wm:
                        img_cv = remove_watermark_cv(img_cv, wm_threshold, wm_radius)

                    # Сохранение
                    out_name = img_path.stem  # имя без расширения
                    out_path = output_path / out_name
                    save_image(img_cv, out_path, fmt, jpeg_q)

                    # Лог успешного завершения
                    log_msg = f"✅ {idx+1}/{len(images)}: {img_path.name} → {out_path.with_suffix('.png' if fmt.startswith('PNG') else '.jpg').name}"
                    logs.append(log_msg)
                    log_area.code("\n".join(logs[-5:]))  # показываем последние 5 строк
                    status_box.info(log_msg)

                except UnidentifiedImageError:
                    err_msg = f"❌ {idx+1}/{len(images)}: Не удалось открыть {img_path.name}"
                    logs.append(err_msg)
                    log_area.code("\n".join(logs[-5:]))
                    logger.error(err_msg)

                except Exception as e:
                    err_msg = f"❌ {idx+1}/{len(images)}: Ошибка при обработке {img_path.name} — {str(e)}"
                    logs.append(err_msg)
                    log_area.code("\n".join(logs[-5:]))
                    logger.error(f"{err_msg}\n{traceback.format_exc()}")

                # Обновление прогресс‑бара внутри цикла
                progress_bar.progress((idx + 1) / len(images))

    # Финальное сообщение
    if logs:
        st.success("✅ Обработка завершена!")
        try:
            st.balloons()
        except Exception:
            pass
    else:
        st.warning("⚠️ Ничего не обработано.")

    # Показ полного лога
    with st.expander("Полный лог обработки"):
        st.code("\n".join(logs))

    st.markdown("---")
    st.info(
        """
**Как использовать:**
1. Поместите изображения в папку `./input` или укажите другую в поле «Входная папка».
2. Настройте параметры обработки в боковой панели.
3. Нажмите «🚀 Запустить обработку».
4. Результаты будут в указанной выходной папке.

**Примечания:**
- **Удаление фона** использует модель `u2net` (библиотека `rembg`). Лучше всего работает на контрастном фоне.
- **Удаление водяных знаков** — экспериментальная функция. Эффективно для:
  - ярких/белых надписей;
  - простых геометрических элементов;
  - однородного фона.
- Для сложных водяных знаков требуется ручная доработка или специализированные нейросети.
- Логи сохраняются в файл `log_*.log` в текущей директории.
"""
    )

# --- Запуск приложения ---
if __name__ == "__main__":
    main()
