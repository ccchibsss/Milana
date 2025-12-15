import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import sys
import platform
import subprocess
from pathlib import Path

# Функция для безопасной установки пакетов
def safe_install_package(package_name):
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--user", package_name
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

# Проверяем наличие необходимых пакетов
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV не установлен. Пытаемся установить...")
    
    success, message = safe_install_package("opencv-python-headless")
    if success:
        try:
            import cv2
            OPENCV_AVAILABLE = True
            st.success("✅ OpenCV успешно установлен!")
            st.rerun()
        except ImportError:
            OPENCV_AVAILABLE = False
            st.error("OpenCV установлен, но импорт не работает")

try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("SciPy не установлен. Пытаемся установить...")
    
    success, message = safe_install_package("scipy")
    if success:
        try:
            import scipy
            SCIPY_AVAILABLE = True
            st.success("✅ SciPy успешно установлен!")
            st.rerun()
        except ImportError:
            SCIPY_AVAILABLE = False
            st.error("SciPy установлен, но импорт не работает")

# Утилиты обработки изображений
class ImageProcessor:
    @staticmethod
    def make_sample_image() -> np.ndarray:
        """Создать тестовое изображение"""
        h, w = 400, 700
        img = np.full((h, w, 3), 230, dtype=np.uint8)
        
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font_large = ImageFont.truetype("arial.ttf", 40)
            font_small = ImageFont.truetype("arial.ttf", 24)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        draw.text((40, 140), "SAMPLE IMAGE", fill=(80, 80, 200), font=font_large)
        draw.text((300, 300), "WATERMARK", fill=(200, 200, 200), font=font_small)
        
        return np.array(pil_img)

    @staticmethod
    def load_image(uploaded_file) -> np.ndarray:
        """Загрузить изображение"""
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)

    @staticmethod
    def save_image(image_array: np.ndarray) -> bytes:
        """Сохранить изображение в bytes"""
        img_pil = Image.fromarray(image_array)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
        """Конвертация RGB в grayscale"""
        return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    @staticmethod
    def create_mask(gray_image: np.ndarray, threshold: int = 150, invert: bool = False) -> np.ndarray:
        """Создать бинарную маску"""
        if OPENCV_AVAILABLE:
            _, mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
            if invert:
                mask = cv2.bitwise_not(mask)
            return mask
        
        mask = np.where(gray_image > threshold, 255, 0).astype(np.uint8)
        if invert:
            mask = 255 - mask
        return mask

    @staticmethod
    def apply_morphology(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Применить морфологические операции"""
        if not OPENCV_AVAILABLE or kernel_size <= 1:
            return mask
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def remove_watermark(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Удалить водяной знак"""
        result = image.copy()
        
        if OPENCV_AVAILABLE:
            # Используем OpenCV inpaint
            return cv2.inpaint(image, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        elif SCIPY_AVAILABLE:
            # Используем scipy для размытия
            mask_bool = mask > 0
            
            for channel in range(3):
                channel_data = result[:, :, channel].copy()
                blurred = scipy.ndimage.gaussian_filter(channel_data, sigma=3)
                channel_data[mask_bool] = blurred[mask_bool]
                result[:, :, channel] = channel_data
            
            return result
        else:
            # Простая замена на среднее значение
            mask_bool = mask > 0
            for channel in range(3):
                channel_data = result[:, :, channel].copy()
                avg_value = np.mean(channel_data[~mask_bool])
                channel_data[mask_bool] = avg_value
                result[:, :, channel] = channel_data
            
            return result

    @staticmethod
    def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.3) -> np.ndarray:
        """Наложить маску на изображение"""
        overlay = image.copy()
        mask_indices = mask > 0
        
        for channel in range(3):
            overlay[mask_indices, channel] = (
                overlay[mask_indices, channel] * (1 - alpha) + 
                color[channel] * alpha
            )
        
        return overlay.astype(np.uint8)

# Основное приложение
def main():
    st.title("🖼️ Watermark Removal Tool")
    st.write("Удаление водяных знаков с изображений")
    
    # Информация о системе
    st.sidebar.header("Информация о системе")
    st.sidebar.write(f"Python: {sys.version.split()[0]}")
    st.sidebar.write(f"Платформа: {platform.system()} {platform.release()}")
    st.sidebar.write(f"OpenCV доступен: {'✅' if OPENCV_AVAILABLE else '❌'}")
    st.sidebar.write(f"SciPy доступен: {'✅' if SCIPY_AVAILABLE else '❌'}")
    
    if not OPENCV_AVAILABLE or not SCIPY_AVAILABLE:
        st.info("""
        ℹ️ **Для лучшей работы рекомендуется установить все зависимости:**
        
        **В терминале:**
        ```bash
        pip install --user opencv-python-headless scipy numpy pillow
        ```
        
        **Или перезапустите приложение для автоматической установки**
        """)
    
    processor = ImageProcessor()
    
    # Создание примера изображения
    if st.button("Создать пример изображения"):
        sample_img = processor.make_sample_image()
        st.image(sample_img, caption="Пример изображения с водяным знаком", width=None)
        
        img_bytes = processor.save_image(sample_img)
        st.download_button(
            label="Скачать пример",
            data=img_bytes,
            file_name="sample_image.png",
            mime="image/png"
        )
    
    # Загрузка изображения
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Отображение оригинального изображения
        original_image = processor.load_image(uploaded_file)
        st.image(original_image, caption="Оригинальное изображение", width=None)
        
        # Параметры обработки
        st.sidebar.header("Настройки обработки")
        threshold = st.sidebar.slider("Порог бинаризации", 0, 255, 150)
        invert_mask = st.sidebar.checkbox("Инвертировать маску", False)
        kernel_size = st.sidebar.slider("Размер ядра морфологии", 1, 15, 5)
        
        # Конвертация в grayscale
        gray_image = processor.rgb_to_grayscale(original_image)
        
        # Создание маски
        mask = processor.create_mask(gray_image, threshold, invert_mask)
        
        # Применение морфологических операций
        if OPENCV_AVAILABLE and kernel_size > 1:
            mask = processor.apply_morphology(mask, kernel_size)
        
        # Показ маски
        st.image(mask, caption="Обнаруженная маска", width=None, clamp=True)
        
        # Показ маски наложенной на изображение
        masked_image = processor.overlay_mask(original_image, mask)
        st.image(masked_image, caption="Маска на изображении", width=None)
        
        # Удаление водяного знака
        if st.button("Удалить водяной знак"):
            with st.spinner("Обработка..."):
                result = processor.remove_watermark(original_image, mask)
            
            st.image(result, caption="Результат удаления", width=None)
            
            # Кнопка скачивания
            result_bytes = processor.save_image(result)
            st.download_button(
                label="Скачать результат",
                data=result_bytes,
                file_name="watermark_removed.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
