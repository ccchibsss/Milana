import streamlit as st
import os
from PIL import Image
from io import BytesIO
from rembg import remove
import cv2
import numpy as np

st.title("Массовое удаление фона и водяных знаков из изображений")

# Загрузка изображений
uploaded_files = st.file_uploader(
    "Загрузите изображения (можно несколько)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

# Ввод папки для сохранения
save_folder = st.text_input("Введите путь для сохранения обработанных изображений", value="processed_images")

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Функция для удаления водяных знаков (пример)
def remove_watermark(image):
    # Если изображение прозрачное, используем альфа-канал
    if image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        # Создаем маску на основе альфа-канала
        mask = cv2.threshold(alpha_channel, 250, 255, cv2.THRESH_BINARY_INV)[1]
        # Восстановление области водяных знаков
        inpainted = cv2.inpaint(image[:, :, :3], mask, 3, cv2.INPAINT_TELEA)
        # Добавляем обратно альфа-канал
        inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGBA)
        inpainted[:, :, 3] = alpha_channel
        return inpainted
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        mask = thresh
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return result

if uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        bytes_data = uploaded_file.read()
        try:
            # Открываем изображение
            input_image = Image.open(BytesIO(bytes_data)).convert("RGBA")
            # Удаляем фон
            image_no_bg = remove(input_image)
            # Конвертируем в формат OpenCV
            image_cv = cv2.cvtColor(np.array(image_no_bg), cv2.COLOR_RGBA2BGRA)
            # Удаляем водяные знаки
            image_clean = remove_watermark(image_cv)
            # Конвертируем обратно в PIL Image
            final_image = Image.fromarray(cv2.cvtColor(image_clean, cv2.COLOR_BGRA2RGBA))
            # Сохраняем
            save_path = os.path.join(save_folder, filename)
            final_image.save(save_path)
            st.success(f"Обработано и сохранено: {save_path}")
        except Exception as e:
            st.error(f"Ошибка при обработке {filename}: {e}")

st.write("Обработка завершена.")
