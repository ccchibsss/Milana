import streamlit as st
import os
from PIL import Image
from io import BytesIO
from rembg import remove
import cv2
import numpy as np
import pyttsx3
from typing import Optional, Union

def main():
    # Настройка страницы
    st.set_page_config(
        page_title="Удаление фона и водяных знаков",
        page_icon="🖼️",
        layout="wide"
    )

    # Стили для внешнего вида
    st.markdown("""
    <style>
        .stProgress > div > div > div > div {
            background-color: #4CAF50 !important;
        }
        .success-box {
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .error-box {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🖼️ Массовое удаление фона и водяных знаков из изображений")
    st.markdown("Загрузите одно или несколько изображений для автоматического удаления фона и водяных знаков.")

    # Интерфейс: два столбца
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Загрузите изображения (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Выберите одно или несколько изображений.",
            key="file_uploader"
        )

    with col2:
        st.subheader("Настройки обработки:")
        save_folder = st.text_input("Папка для сохранения:", value="processed_images", key="save_folder")
        remove_bg = st.checkbox("Удалять фон", value=True, key="remove_bg")
        remove_watermark = st.checkbox("Удалять водяные знаки", value=False, key="remove_watermark")
        quality = st.slider("Качество выходного файла (%):", min_value=50, max_value=100, value=95, key="quality_slider")

    # Создаем папку для сохранения
    os.makedirs(save_folder, exist_ok=True)

    # Инициализация говорителя
    engine = pyttsx3.init()

    def speak(text):
        engine.say(text)
        engine.runAndWait()

    # Класс для обработки изображений
    class BackgroundAndWatermarkRemover:
        def __init__(self, save_folder: str = "processed_images"):
            self.save_folder = save_folder

        def create_thumbnail(self, image: Image.Image, size=(200, 200)) -> Image.Image:
            img = image.copy()
            img.thumbnail(size)
            return img

        def remove_watermark(self, image: Image.Image) -> Image.Image:
            open_cv_image = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            result = cv2.inpaint(open_cv_image, clean_mask, 3, cv2.INPAINT_TELEA)
            return Image.fromarray(result)

        def process_image(
            self,
            image_data: Union[BytesIO, bytes],
            remove_bg: bool = True,
            remove_watermark: bool = False
        ) -> Optional[Image.Image]:
            try:
                if hasattr(image_data, "read"):
                    image_bytes = image_data.read()
                else:
                    image_bytes = image_data
                image = Image.open(BytesIO(image_bytes)).convert("RGBA")
                # Удаление фона
                if remove_bg:
                    try:
                        output = remove(image_bytes)
                        bg_image = Image.open(BytesIO(output)).convert("RGBA")
                    except Exception as e:
                        st.error(f"Ошибка при удалении фона: {e}")
                        bg_image = image
                else:
                    bg_image = image
                # Удаление водяных знаков
                if remove_watermark:
                    try:
                        final_image = self.remove_watermark(bg_image)
                    except Exception as e:
                        st.error(f"Ошибка при удалении водяных знаков: {e}")
                        final_image = bg_image
                else:
                    final_image = bg_image
                return final_image
            except Exception as e:
                st.error(f"Ошибка при обработке изображения: {e}")
                return None

        def save_image(self, filename: str, image: Image.Image, quality: int = 95) -> str:
            save_path = os.path.join(self.save_folder, filename)
            ext = os.path.splitext(filename)[1].lower()
            try:
                if ext in ['.jpg', '.jpeg']:
                    image = image.convert("RGB")
                    image.save(save_path, format='JPEG', quality=quality, optimize=True)
                else:
                    image.save(save_path, format='PNG', optimize=True)
            except Exception as e:
                st.error(f"Ошибка при сохранении файла {filename}: {e}")
                return ""
            return save_path

    # Основной блок обработки
    if uploaded_files and len(uploaded_files) > 0:
        speak("Обработка изображений началась.")
        remover = BackgroundAndWatermarkRemover(save_folder=save_folder)
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        processed_files = []  # список для хранения байтовых данных файлов для скачивания

        for idx, uploaded_file in enumerate(uploaded_files):
            status_message = f"Обрабатываю файл {idx+1}/{total_files}: {uploaded_file.name}"
            with st.spinner(status_message):
                processed_img = remover.process_image(
                    uploaded_file,
                    remove_bg=remove_bg,
                    remove_watermark=remove_watermark
                )
                if processed_img is None:
                    st.error(f"Не удалось обработать {uploaded_file.name}. Пропускаю.")
                    continue
                thumbnail = remover.create_thumbnail(processed_img)
                st.image(thumbnail, caption=f"Предпросмотр {uploaded_file.name}", use_column_width=True)
                save_path = remover.save_image(uploaded_file.name, processed_img, quality=quality)
                if save_path:
                    st.success(f"Файл сохранён: `{save_path}`")
                    # читаем байты файла для скачивания
                    with open(save_path, "rb") as f:
                        file_bytes = f.read()
                    processed_files.append((uploaded_file.name, file_bytes))
                else:
                    st.error(f"Ошибка при сохранении {uploaded_file.name}")
            progress_bar.progress((idx + 1) / total_files)

        st.balloons()
        st.write(f"Все файлы успешно сохранены в папку: `{save_folder}`.")

        # отображаем кнопки скачивания
        st.subheader("Скачать обработанные файлы:")
        for filename, file_bytes in processed_files:
            st.download_button(
                label=f"Скачать {filename}",
                data=file_bytes,
                file_name=filename
            )

    elif uploaded_files is None or len(uploaded_files) == 0:
        st.info("Пожалуйста, загрузите хотя бы одно изображение для начала обработки.")

if __name__ == "__main__":
    main()
