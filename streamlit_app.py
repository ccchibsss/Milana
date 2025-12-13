import streamlit as st
import os
from PIL import Image
from io import BytesIO
from rembg import remove
import cv2
import numpy as np

st.title("Массовое удаление фона и водяных знаков из изображений")

# Загрузка изображений
uploaded_files = st.file_uploader("Загрузите изображения (можно несколько)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Выбор папки для сохранения
save_folder = st.text_input("Введите путь для сохранения обработанных изображений", value="processed_images")

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Функция для удаления водяных знаков (пример простого удаления)
def remove_watermark(image):
    # Пример: преобразование изображения в градации серого и пороговая обработка
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Маска для водяных знаков
    mask = cv2.bitwise_and(image, image, mask=thresh)
    # Заменяем области водяных знаков на прозрачные или белые
    result = cv2.inpaint(image, thresh, 3, cv2.INPAINT_TELEA)
    return result

if uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        bytes_data = uploaded_file.read()

        # Открываем изображение
        input_image = Image.open(BytesIO(bytes_data)).convert("RGBA")
        # Удаляем фон
        image_no_bg = remove(input_image)
        # Конвертируем в формат OpenCV для удаления водяных знаков
        image_cv = cv2.cvtColor(np.array(image_no_bg), cv2.COLOR_RGBA2BGR)
        # Удаляем водяные знаки
        image_clean = remove_watermark(image_cv)
        # Конвертируем обратно в PIL
        final_image = Image.fromarray(cv2.cvtColor(image_clean, cv2.COLOR_BGR2RGB))
        
        # Сохраняем изображение
        save_path = os.path.join(save_folder, filename)
        final_image.save(save_path)
        st.success(f"Обработано и сохранено: {save_path}")

st.write("Обработка завершена.")
