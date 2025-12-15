# ai_watermark_removal.py
# Streamlit app that removes watermarks and optionally uses a simple "AI" (MLP)
# model trained on the image.
# Saves missing dependencies automatically (opencv-python-headless, scipy,
# scikit-learn) if needed.

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import sys
import platform
import subprocess
from pathlib import Path
import time

# safe installer
def safe_install_package(package_name):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", package_name],
            capture_output=True, text=True, timeout=240
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

# optional libs
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

try:
    import scipy
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from sklearn.neural_network import MLPRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Attempt installs if missing (non-blocking; not recommended for heavy envs)
if not OPENCV_AVAILABLE:
    st = st if 'st' in globals() else None
    success, msg = safe_install_package("opencv-python-headless")
    if success:
        try:
            import cv2
            OPENCV_AVAILABLE = True
        except Exception:
            OPENCV_AVAILABLE = False

if not SCIPY_AVAILABLE:
    success, msg = safe_install_package("scipy")
    if success:
        try:
            import scipy
            SCIPY_AVAILABLE = True
        except Exception:
            SCIPY_AVAILABLE = False

if not SKLEARN_AVAILABLE:
    success, msg = safe_install_package("scikit-learn")
    if success:
        try:
            from sklearn.neural_network import MLPRegressor
            SKLEARN_AVAILABLE = True
        except Exception:
            SKLEARN_AVAILABLE = False

# Image utilities
class ImageProcessor:
    @staticmethod
    def make_sample_image() -> np.ndarray:
        h, w = 400, 700
        img = np.full((h, w, 3), 230, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        try:
            font_large = ImageFont.truetype("arial.ttf", 40)
            font_small = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        draw.text((40, 140), "SAMPLE IMAGE", fill=(80, 80, 200), font=font_large)
        draw.text((300, 300), "WATERMARK", fill=(200, 200, 200), font=font_small)
        return np.array(pil_img)

    @staticmethod
    def load_image(uploaded_file) -> np.ndarray:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)

    @staticmethod
    def save_image(image_array: np.ndarray) -> bytes:
        img_pil = Image.fromarray(image_array)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
        return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    @staticmethod
    def create_mask(gray_image: np.ndarray, threshold: int = 150, invert: bool = False) -> np.ndarray:
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
        if not OPENCV_AVAILABLE or kernel_size <= 1:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def remove_watermark(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = image.copy()
        if OPENCV_AVAILABLE:
            return cv2.inpaint(image, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        elif SCIPY_AVAILABLE:
            mask_bool = mask > 0
            for channel in range(3):
                channel_data = result[:, :, channel].copy()
                blurred = scipy.ndimage.gaussian_filter(channel_data, sigma=3)
                channel_data[mask_bool] = blurred[mask_bool]
                result[:, :, channel] = channel_data
            return result
        else:
            mask_bool = mask > 0
            for channel in range(3):
                channel_data = result[:, :, channel].copy()
                valid = ~mask_bool
                if np.any(valid):
                    avg_value = np.mean(channel_data[valid])
                else:
                    avg_value = 128
                channel_data[mask_bool] = avg_value
                result[:, :, channel] = channel_data
            return result

    @staticmethod
    def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.3) -> np.ndarray:
        overlay = image.copy().astype(np.float32)
        mask_indices = mask > 0
        for channel in range(3):
            overlay[..., channel][mask_indices] = (
                overlay[..., channel][mask_indices] * (1 - alpha) + color[channel] * alpha
            )
        return np.clip(overlay, 0, 255).astype(np.uint8)

    @staticmethod
    def ai_inpaint_mlp(image: np.ndarray, mask: np.ndarray, subsample: int = 5000, hidden_size: int = 64, max_iter: int = 200, random_state: int = 42) -> np.ndarray:
        """
        Train a small MLP on unmasked pixels to predict RGB from (x_norm, y_norm, r,g,b, neigh_mean_r,g,b).
        This is a lightweight "AI" demo suitable for small images and for educational use.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not available")

        h, w = mask.shape
        mask_bool = mask > 0

        # prepare padded image for 3x3 neighborhood mean
        pad = np.pad(image, ((1,1),(1,1),(0,0)), mode='reflect')
        neigh_mean = np.zeros_like(image, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                block = pad[i:i+3, j:j+3, :]
                neigh_mean[i, j, :] = block.mean(axis=(0,1))

        # sample unmasked pixels for training
        ys, xs = np.where(~mask_bool)
        if len(xs) == 0:
            return image.copy()
        idxs = np.arange(len(xs))
        if len(idxs) > subsample:
            rng = np.random.RandomState(random_state)
            idxs = rng.choice(idxs, subsample, replace=False)
        xs_s = xs[idxs]
        ys_s = ys[idxs]

        # build features and targets
        X = []
        y = []
        for xi, yi in zip(xs_s, ys_s):
            r,g,b = image[xi, yi].astype(np.float32) / 255.0
            nm_r, nm_g, nm_b = neigh_mean[xi, yi] / 255.0
            X.append([yi / w, xi / h, r, g, b, nm_r, nm_g, nm_b])
            y.append([r, g, b])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # train MLP
        mlp = MLPRegressor(hidden_layer_sizes=(hidden_size,), activation='relu', solver='adam',
                            learning_rate_init=0.001, max_iter=max_iter, random_state=random_state, verbose=False)
        mlp.fit(X, y)

        # predict for masked pixels
        out = image.copy().astype(np.float32) / 255.0
        ys_m, xs_m = np.where(mask_bool)
        if len(xs_m) == 0:
            return image.copy()
        Xm = []
        for xi, yi in zip(xs_m, ys_m):
            r,g,b = image[xi, yi].astype(np.float32) / 255.0
            nm_r, nm_g, nm_b = neigh_mean[xi, yi] / 255.0
            Xm.append([yi / w, xi / h, r, g, b, nm_r, nm_g, nm_b])
        Xm = np.array(Xm, dtype=np.float32)
        preds = mlp.predict(Xm)  # normalized values
        preds = np.clip(preds, 0.0, 1.0)

        for k, (xi, yi) in enumerate(zip(xs_m, ys_m)):
            out[xi, yi, :] = preds[k, :]  # replace masked pixel with prediction

        return (np.clip(out * 255.0, 0, 255)).astype(np.uint8)


# Streamlit app
def main():
    st.set_page_config(page_title="AI Watermark Removal", layout="wide")
    st.title("🖼️ Watermark Removal Tool with AI Demo")
    st.write("Удаление водяных знаков — обычные методы + демонстрация простого обучения (MLP).")

    st.sidebar.header("Система")
    st.sidebar.write(f"Python: {sys.version.split()[0]}")
    st.sidebar.write(f"Platform: {platform.system()} {platform.release()}")
    st.sidebar.write(f"OpenCV: {'✅' if OPENCV_AVAILABLE else '❌'}")
    st.sidebar.write(f"SciPy: {'✅' if SCIPY_AVAILABLE else '❌'}")
    st.sidebar.write(f"scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")

    processor = ImageProcessor()

    if st.button("Создать пример изображения"):
        sample_img = processor.make_sample_image()
        st.image(sample_img, caption="Пример изображения", use_column_width=True)
        st.download_button("Скачать пример", data=processor.save_image(sample_img), file_name="sample_image.png", mime="image/png")

    uploaded_file = st.file_uploader("Загрузите изображение (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        original = processor.load_image(uploaded_file)
        st.image(original, caption="Оригинал", use_column_width=True)

        st.sidebar.header("Параметры маски")
        threshold = st.sidebar.slider("Порог бинаризации", 0, 255, 150)
        invert_mask = st.sidebar.checkbox("Инвертировать маску", False)
        kernel_size = st.sidebar.slider("Размер ядра морфологии", 1, 15, 5)

        gray = processor.rgb_to_grayscale(original)
        mask = processor.create_mask(gray, threshold=threshold, invert=invert_mask)
        if OPENCV_AVAILABLE and kernel_size > 1:
            mask = processor.apply_morphology(mask, kernel_size)

        st.image(mask, caption="Маска", clamp=True, use_column_width=True)
        st.image(processor.overlay_mask(original, mask), caption="Маска на изображении", use_column_width=True)

        # standard removal
        if st.button("Удалить водяной знак (обычный метод)"):
            with st.spinner("Обработка..."):
                res = processor.remove_watermark(original, mask)
            st.image(res, caption="Результат (обычный метод)", use_column_width=True)
            st.download_button("Скачать результат (обычный)", data=processor.save_image(res), file_name="result_standard.png", mime="image/png")

        # AI inpainting options
        st.sidebar.header("AI (MLP) параметры")
        use_ai = st.sidebar.checkbox("Использовать AI (тренировать MLP на изображении)", value=False)
        subsample = st.sidebar.slider("Макс обучающих точек", 1000, 20000, 5000, step=500)
        hidden_size = st.sidebar.slider("Размер скрытого слоя", 8, 256, 64, step=8)
        max_iter = st.sidebar.slider("Эпохи (итерации) MLP", 50, 500, 200, step=10)

        if use_ai:
            if not SKLEARN_AVAILABLE:
                st.error("scikit-learn не установлен — установка не всегда возможна в этом окружении.")
            else:
                if st.button("Удалить водяной знак (AI)"):
                    with st.spinner("Тренируем MLP и применяем..."):
                        t0 = time.time()
                        try:
                            res_ai = processor.ai_inpaint_mlp(original, mask, subsample=subsample, hidden_size=hidden_size, max_iter=max_iter)
                            dt = time.time() - t0
                            st.success(f"Готово за {dt:.1f}s")
                            st.image(res_ai, caption="Результат (AI MLP)", use_column_width=True)
                            st.download_button("Скачать результат (AI)", data=processor.save_image(res_ai), file_name="result_ai.png", mime="image/png")
                        except Exception as e:
                            st.error(f"Ошибка AI процесса: {e}")

if __name__ == "__main__":
    main()
