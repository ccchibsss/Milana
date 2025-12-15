import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from torchvision.models import detection
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config
import io
import tempfile
import os
from typing import List, Tuple
import time

# Настройки страницы
st.set_page_config(
    page_title="AI Watermark Remover Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния сессии
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "lama"
if 'mask_points' not in st.session_state:
    st.session_state.mask_points = []
if 'drawing_mode' not in st.session_state:
    st.session_state.drawing_mode = False

class WatermarkRemover:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.current_model = None
        
    def load_model(self, model_name: str):
        """Загрузка выбранной модели"""
        try:
            if model_name == "lama" and "lama" not in self.models:
                st.info("🔄 Загрузка LaMa модели...")
                self.models["lama"] = ModelManager(
                    name="lama",
                    device=self.device
                )
                
            elif model_name == "gfpgan" and "gfpgan" not in self.models:
                st.info("🔄 Загрузка GFPGAN модели...")
                # Здесь будет код для GFPGAN
                pass
                
            elif model_name == "detection" and "detection" not in self.models:
                st.info("🔄 Загрузка модели детекции...")
                self.models["detection"] = detection.maskrcnn_resnet50_fpn(
                    pretrained=True
                ).to(self.device).eval()
                
            self.current_model = model_name
            return True
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            return False

    def auto_detect_watermark(self, image: np.ndarray) -> np.ndarray:
        """Автоматическое обнаружение водяных знаков"""
        if "detection" not in self.models:
            self.load_model("detection")
        
        transform = T.Compose([T.ToTensor()])
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.models["detection"](input_tensor)
        
        # Создание маски на основе предсказаний
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for score, label, box, mask_pred in zip(
            predictions[0]['scores'], predictions[0]['labels'],
            predictions[0]['boxes'], predictions[0]['masks']
        ):
            if score > 0.7:  # Порог уверенности
                mask_pred = mask_pred[0].cpu().numpy() > 0.5
                mask[mask_pred] = 255
        
        return mask

    def remove_watermark(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        config: Config
    ) -> np.ndarray:
        """Удаление водяного знака с помощью выбранной модели"""
        if self.current_model == "lama":
            return self._remove_with_lama(image, mask, config)
        elif self.current_model == "gfpgan":
            return self._remove_with_gfpgan(image, mask)
        else:
            return image

    def _remove_with_lama(self, image: np.ndarray, mask: np.ndarray, config: Config) -> np.ndarray:
        """Удаление с помощью LaMa"""
        try:
            result = self.models["lama"](image, mask, config)
            return result
        except Exception as e:
            st.error(f"Ошибка обработки LaMa: {e}")
            return image

    def _remove_with_gfpgan(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Удаление с помощью GFPGAN"""
        # Заглушка для GFPGAN реализации
        return image

def create_mask_from_points(image_size: Tuple[int, int], points: List[Tuple[int, int]]) -> np.ndarray:
    """Создание маски из точек"""
    mask = Image.new('L', image_size, 0)
    if points:
        draw = ImageDraw.Draw(mask)
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=255, width=20)
        draw.line([points[-1], points[0]], fill=255, width=20)
    return np.array(mask)

def main():
    st.title("🎨 AI Watermark Remover Pro")
    st.markdown("Мощный инструмент для удаления водяных знаков с использованием AI")
    
    # Инициализация обработчика
    remover = WatermarkRemover()
    
    # Сайдбар с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Выбор модели
        model_choice = st.selectbox(
            "Выберите модель",
            ["lama", "gfpgan"],
            index=0,
            help="LaMa - для общего удаления, GFPGAN - для лиц"
        )
        
        # Настройки обработки
        st.subheader("Параметры обработки")
        hd_option = st.checkbox("HD режим", False)
        quality = st.slider("Качество обработки", 1, 10, 7)
        
        # Пакетная обработка
        st.subheader("Пакетная обработка")
        batch_files = st.file_uploader(
            "Выберите несколько изображений",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if st.button("🚀 Обработать все", type="primary") and batch_files:
            process_batch(remover, batch_files, model_choice, hd_option, quality)

    # Основная область
    tab1, tab2, tab3 = st.tabs(["📤 Загрузка", "🎯 Выбор области", "⚡ Обработка"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Загрузите изображение с водяным знаком",
            type=['png', 'jpg', 'jpeg'],
            key="main_uploader"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.original_image = np.array(image)
            st.image(image, caption="Исходное изображение", use_column_width=True)
            
            # Автодетекция
            if st.button("🔍 Автоматическое обнаружение водяных знаков"):
                with st.spinner("Ищем водяные знаки..."):
                    mask = remover.auto_detect_watermark(st.session_state.original_image)
                    if mask.any():
                        st.session_state.auto_mask = mask
                        st.success("Найдены потенциальные водяные знаки!")
                        st.image(mask, caption="Обнаруженная область", use_column_width=True)
                    else:
                        st.warning("Водяные знаки не обнаружены автоматически")

    with tab2:
        if 'original_image' in st.session_state:
            st.subheader("Выделите область водяного знака")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Интерактивное выделение
                if st.button("✏️ Режим рисования"):
                    st.session_state.drawing_mode = not st.session_state.drawing_mode
                
                if st.session_state.drawing_mode:
                    st.info("Кликните на изображении чтобы добавить точки")
                    
                    # Отображение изображения для кликов
                    fig = st.empty()
                    fig.image(st.session_state.original_image, use_column_width=True)
                    
                    # Обработка кликов
                    points = st.session_state.get('mask_points', [])
                    if fig.clickable:
                        click_data = fig.get_click_data()
                        if click_data:
                            x, y = click_data['x'], click_data['y']
                            points.append((x, y))
                            st.session_state.mask_points = points
                
                if st.button("🧹 Очистить выделение"):
                    st.session_state.mask_points = []
                
                if st.button("✅ Применить выделение"):
                    if st.session_state.mask_points:
                        mask = create_mask_from_points(
                            st.session_state.original_image.shape[:2][::-1],
                            st.session_state.mask_points
                        )
                        st.session_state.custom_mask = mask
                        st.success("Маска создана!")
            
            with col2:
                if 'custom_mask' in st.session_state:
                    st.image(st.session_state.custom_mask, caption="Ваша маска", use_column_width=True)

    with tab3:
        if 'original_image' in st.session_state:
            st.subheader("Обработка изображения")
            
            if st.button("✨ Запустить обработку", type="primary"):
                # Загрузка модели
                if remover.load_model(model_choice):
                    with st.spinner("Обрабатываем изображение..."):
                        # Конфигурация обработки
                        config = Config(
                            ldm_steps=20,
                            hd_strategy='Crop' if hd_option else 'Original',
                            quality=quality
                        )
                        
                        # Выбор маски
                        if 'custom_mask' in st.session_state:
                            mask = st.session_state.custom_mask
                        elif 'auto_mask' in st.session_state:
                            mask = st.session_state.auto_mask
                        else:
                            st.error("Сначала создайте маску!")
                            return
                        
                        # Обработка
                        result = remover.remove_watermark(
                            st.session_state.original_image,
                            mask,
                            config
                        )
                        
                        # Сохранение результата
                        st.session_state.processed_images.append(result)
                        
                        # Отображение
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(st.session_state.original_image, 
                                   caption="До", use_column_width=True)
                        with col2:
                            st.image(result, caption="После", use_column_width=True)
                        
                        # Кнопка скачивания
                        result_pil = Image.fromarray(result)
                        buf = io.BytesIO()
                        result_pil.save(buf, format="PNG", quality=95)
                        
                        st.download_button(
                            "📥 Скачать результат",
                            buf.getvalue(),
                            "watermark_removed.png",
                            "image/png",
                            use_container_width=True
                        )

def process_batch(remover, files, model_choice, hd_option, quality):
    """Обработка нескольких изображений"""
    progress_bar = st.progress(0)
    results = []
    
    for i, file in enumerate(files):
        try:
            image = Image.open(file).convert('RGB')
            img_array = np.array(image)
            
            # Автоматическое создание маски
            mask = remover.auto_detect_watermark(img_array)
            
            # Обработка
            config = Config(
                ldm_steps=20,
                hd_strategy='Crop' if hd_option else 'Original',
                quality=quality
            )
            
            result = remover.remove_watermark(img_array, mask, config)
            results.append((file.name, result))
            
        except Exception as e:
            st.error(f"Ошибка обработки {file.name}: {e}")
        
        progress_bar.progress((i + 1) / len(files))
    
    # Предоставление результатов для скачивания
    for filename, result in results:
        result_pil = Image.fromarray(result)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG", quality=95)
        
        st.download_button(
            f"📥 Скачать {filename}",
            buf.getvalue(),
            f"processed_{filename}",
            "image/png"
        )

# Информация о системе
with st.sidebar:
    st.markdown("---")
    st.subheader("Системная информация")
    st.write(f"Устройство: {remover.device}")
    st.write(f"CUDA доступно: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.write(f"GPU: {torch.cuda.get_device_name(0)}")
        st.write(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

if __name__ == "__main__":
    main()
