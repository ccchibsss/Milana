import streamlit as st
import os
from PIL import Image
from io import BytesIO
from rembg import remove
import cv2
import numpy as np

# Настройка страницы
st.set_page_config(
    page_title="Удаление фона и водяных знаков",
    page_icon="🖼️",
    layout="wide"
)

# Стили для улучшения внешнего вида
st.markdown("""
<style>
    .stProgress .st-bo {
        background-color: #4CAF50;
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

# Разделяем интерфейс на два столбца
col1, col2 = st.columns([1, 1])

with col1:
    # Загрузка изображений
    uploaded_files = st.file_uploader(
        "Загрузите изображения (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Выберите одно или несколько изображений."
    )

with col2:
    # Настройки обработки
    st.subheader("Настройки обработки:")
    save_folder = st.text_input("Папка для сохранения:", value="processed_images")
    remove_bg = st.checkbox("Удалять фон", value=True)
    remove_watermark = st.checkbox("Удалять водяные знаки", value=False)
    quality = st.slider("Качество выходного файла (%):", min_value=50, max_value=100, value=95)

# Функциональность класса для обработки изображений
class BackgroundAndWatermarkRemover:
    def __init__(self, save_folder="processed_images"):
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def create_thumbnail(self, image, size=(200, 200)):
        """Создание миниатюры изображения."""
        img = image.copy()
        img.thumbnail(size)
        return img

    def remove_watermark(self, image):
        """
        Эффективно удаляет водяные знаки с изображения путём анализа и маскировки областей с низкой непрозрачностью.
        """
        open_cv_image = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        result = cv2.inpaint(open_cv_image, clean_mask, 3, cv2.INPAINT_TELEA)
        return Image.fromarray(result)

    def process_image(self, image_data, remove_bg=True, remove_watermark=False):
        """
        Метод для полной обработки изображения: удаление фона и водяных знаков.
        """
        image = Image.open(image_data).convert("RGBA")
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGRA)

        # Удаление фона
        if remove_bg:
            try:
                output = remove(image_data.read())
                bg_image = Image.open(BytesIO(output))
            except Exception as e:
                st.error(f"Ошибка при удалении фона: {e}")
                bg_image = image
        else:
            bg_image = image

        # Удаление водяных знаков
        if remove_watermark and isinstance(bg_image, Image.Image):
            try:
                watermark_removed = self.remove_watermark(bg_image)
                final_image = watermark_removed
            except Exception as e:
                st.error(f"Ошибка при удалении водяного знака: {e}")
                final_image = bg_image
        else:
            final_image = bg_image

        return final_image

    def save_image(self, filename, image, quality=95):
        """Метод для сохранения обработанных изображений в заданную папку."""
        save_path = os.path.join(self.save_folder, filename)
        image.save(save_path, format='PNG', quality=quality)
        return save_path

# Начнём обрабатывать изображения, если они были загружены
if uploaded_files:
    remover = BackgroundAndWatermarkRemover(save_folder=save_folder)
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f"Обрабатываю файл {uploaded_file.name}..."):
            # Основной цикл обработки изображений
            processed_img = remover.process_image(uploaded_file, remove_bg=remove_bg, remove_watermark=remove_watermark)
            
            # Показываем превью результата
            thumbnail = remover.create_thumbnail(processed_img)
            st.image(thumbnail, caption=f"Предпросмотр {uploaded_file.name}", use_column_width=True)
        
            # Сохраняем обработанный файл
            save_filename = remover.save_image(uploaded_file.name, processed_img, quality=quality)
            st.success(f"Файл успешно сохранён: {save_filename}.")
        
        # Продвигаем прогресс-бар
        progress_bar.progress((idx+1)/total_files)

    st.balloons()  # Анимационная иконка после завершения обработки
    st.write(f"Все файлы успешно сохранены в папку: `{save_folder}`.")

else:
    st.info("Пожалуйста, загрузите хотя бы одно изображение для начала обработки.")
