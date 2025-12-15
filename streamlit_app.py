import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import sys
from pathlib import Path
import subprocess
import platform

# Функция для установки пакетов
def install_package(package_name):
    try:
        # Проверяем, какой пакетный менеджер доступен
        if hasattr(sys, 'real_prefix') or hasattr(sys, 'base_prefix'):
            # Мы в виртуальном окружении
            pip_cmd = [sys.executable, "-m", "pip", "install", package_name]
        else:
            pip_cmd = ["pip", "install", package_name]
            
        result = subprocess.run(
            pip_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Ошибка установки {package_name}: {e.stderr}"

# Проверяем наличие OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV не установлен. Пытаемся установить...")
    
    # Пробуем установить с разными вариантами
    success, message = install_package("opencv-python-headless")
    if success:
        try:
            import cv2
            OPENCV_AVAILABLE = True
            st.success("OpenCV успешно установлен!")
        except ImportError:
            OPENCV_AVAILABLE = False
            st.error("OpenCV установлен, но импорт не работает")
    else:
        st.error(f"Не удалось установить OpenCV: {message}")
        st.info("Попробуем альтернативный вариант...")
        
        # Пробуем другой пакет
        success2, message2 = install_package("opencv-python")
        if success2:
            try:
                import cv2
                OPENCV_AVAILABLE = True
                st.success("OpenCV успешно установлен!")
            except ImportError:
                OPENCV_AVAILABLE = False
                st.error("OpenCV установлен, но импорт не работает")
        else:
            st.error(f"Не удалось установить OpenCV: {message2}")

# Проверяем наличие torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("PyTorch не установлен. Нейросетевые функции будут недоступны.")

# -------------------- Утилиты обработки --------------------
class WatermarkUtils:
    @staticmethod
    def make_sample_image() -> np.ndarray:
        """Создать простое тестовое изображение."""
        h, w = 400, 700
        img = np.full((h, w, 3), 230, dtype=np.uint8)
        
        if OPENCV_AVAILABLE:
            cv2.putText(img, "SAMPLE IMAGE", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 80, 200), 4, cv2.LINE_AA)
            cv2.putText(img, "WATERMARK", (300, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        else:
            # Альтернатива без OpenCV
            from PIL import ImageDraw, ImageFont
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("arial.ttf", 40)
                draw.text((40, 140), "SAMPLE IMAGE", fill=(80, 80, 200), font=font)
                draw.text((300, 300), "WATERMARK", fill=(200, 200, 200), font=font)
            except:
                draw.text((40, 140), "SAMPLE IMAGE", fill=(80, 80, 200))
                draw.text((300, 300), "WATERMARK", fill=(200, 200, 200))
            img = np.array(pil_img)
        
        return img

    @staticmethod
    def load_image(uploaded_file) -> np.ndarray:
        """Загрузить изображение из Streamlit uploaded file."""
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)

    @staticmethod
    def save_image(image_array: np.ndarray) -> bytes:
        """Сохранить изображение в bytes."""
        img_pil = Image.fromarray(image_array)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def make_mask_from_gray(gray: np.ndarray, thresh: int = 150, invert: bool = False, k: int = 5) -> np.ndarray:
        """Создать бинарную маску из серого изображения."""
        if not OPENCV_AVAILABLE:
            # Простая альтернатива без OpenCV
            mask = np.where(gray > thresh, 255, 0).astype(np.uint8)
            if invert:
                mask = 255 - mask
            return mask
        
        _, m = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
        if invert:
            m = cv2.bitwise_not(m)
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        return m.astype(np.uint8)

    @staticmethod
    def inpaint_bgr(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Выполнить inpaint."""
        if not OPENCV_AVAILABLE:
            # Простая альтернатива без OpenCV - размытие области
            result = img_bgr.copy()
            mask_bool = mask > 0
            
            # Применяем размытие только к области маски
            from scipy import ndimage
            for c in range(3):
                channel = result[:, :, c]
                # Заменяем пиксели под маской на среднее значение окружающих пикселей
                blurred = ndimage.gaussian_filter(channel, sigma=3)
                channel[mask_bool] = blurred[mask_bool]
                result[:, :, c] = channel
            
            return result
        
        m = mask.astype(np.uint8)
        return cv2.inpaint(img_bgr, m, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def overlay_mask_on_image(img: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.3) -> np.ndarray:
        """Наложить маску на изображение."""
        overlay = img.copy()
        mask_indices = mask > 0
        
        for c in range(3):
            overlay[mask_indices, c] = (
                overlay[mask_indices, c] * (1 - alpha) + 
                color[c] * alpha
            )
        
        return overlay.astype(np.uint8)

# -------------------- Streamlit App --------------------

def main():
    st.title("🖼️ Watermark Removal Tool")
    st.write("Удаление водяных знаков с изображений")
    
    # Информация о системе
    st.sidebar.header("Информация о системе")
    st.sidebar.write(f"Python: {sys.version}")
    st.sidebar.write(f"Платформа: {platform.system()} {platform.release()}")
    st.sidebar.write(f"OpenCV доступен: {'✅' if OPENCV_AVAILABLE else '❌'}")
    st.sidebar.write(f"PyTorch доступен: {'✅' if TORCH_AVAILABLE else '❌'}")
    
    if not OPENCV_AVAILABLE:
        st.warning("""
        ⚠️ OpenCV не установлен. Некоторые функции будут ограничены.
        
        Попробуйте установить вручную:
        ```
        pip install opencv-python-headless
        ```
        
        Или перезапустите приложение - оно попытается установить автоматически.
        """)
    
    utils = WatermarkUtils()
    
    # Создание примера изображения
    if st.button("Создать пример изображения"):
        sample_img = utils.make_sample_image()
        st.image(sample_img, caption="Пример изображения с водяным знаком", use_column_width=True)
        
        # Сохранение для скачивания
        img_bytes = utils.save_image(sample_img)
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
        original_image = utils.load_image(uploaded_file)
        st.image(original_image, caption="Оригинальное изображение", use_column_width=True)
        
        # Параметры обработки
        st.sidebar.header("Настройки обработки")
        thresh = st.sidebar.slider("Порог бинаризации", 0, 255, 150)
        invert = st.sidebar.checkbox("Инвертировать маску", False)
        kernel_size = st.sidebar.slider("Размер ядра", 1, 15, 5)
        
        # Конвертация в grayscale для создания маски
        if OPENCV_AVAILABLE:
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            # Альтернатива без OpenCV
            gray = np.mean(original_image, axis=2).astype(np.uint8)
        
        # Создание маски
        mask = utils.make_mask_from_gray(gray, thresh=thresh, invert=invert, k=kernel_size)
        
        # Показ маски
        st.image(mask, caption="Обнаруженная маска", use_column_width=True, clamp=True)
        
        # Показ маски наложенной на изображение
        masked_image = utils.overlay_mask_on_image(original_image, mask)
        st.image(masked_image, caption="Маска на изображении", use_column_width=True)
        
        # Удаление водяного знака
        if st.button("Удалить водяной знак"):
            with st.spinner("Обработка..."):
                result = utils.inpaint_bgr(original_image, mask)
            
            st.image(result, caption="Результат удаления", use_column_width=True)
            
            # Кнопка скачивания
            result_bytes = utils.save_image(result)
            st.download_button(
                label="Скачать результат",
                data=result_bytes,
                file_name="watermark_removed.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
