import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
import argparse
import subprocess
from pathlib import Path

# -------------------- Утилиты обработки из watermark_tool.py --------------------

class WatermarkUtils:
    @staticmethod
    def make_sample_image() -> np.ndarray:
        """Создать простое тестовое изображение (BGR numpy array)."""
        h, w = 400, 700
        img = np.full((h, w, 3), 230, dtype=np.uint8)
        cv2.putText(img, "SAMPLE IMAGE", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 80, 200), 4, cv2.LINE_AA)
        cv2.putText(img, "WATERMARK", (300, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        return img

    @staticmethod
    def load_image(path: str) -> np.ndarray | None:
        """Загрузить изображение в BGR numpy array. Возвращает None при ошибке."""
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            print(f"Файл не найден: {path}")
            return None
        
        pil = Image.open(str(p)).convert("RGB")
        arr = np.array(pil)[:, :, ::-1]  # RGB -> BGR
        return arr

    @staticmethod
    def save_image_bgr(img_bgr: np.ndarray, out_path: str) -> None:
        """Сохранить BGR numpy в файл."""
        p = Path(out_path)
        pil = Image.fromarray(img_bgr[:, :, ::-1])  # BGR -> RGB
        pil.save(str(p))

    @staticmethod
    def detect_watermark_areas(image_path, threshold=200):
        """
        Автоматическое обнаружение областей с водяными знаками
        """
        img = WatermarkUtils.load_image(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Бинаризация для выделения светлых областей
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        watermark_areas = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Минимальная площадь
                x, y, w, h = cv2.boundingRect(contour)
                watermark_areas.append((x, y, w, h))
        
        return watermark_areas

    @staticmethod
    def make_mask_from_gray(gray: np.ndarray, thresh: int = 150, invert: bool = False, k: int = 5) -> np.ndarray:
        """Создать бинарную маску из серого изображения (uint8 0/255)."""
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
        """Выполнить inpaint с помощью cv2."""
        m = mask.astype(np.uint8)
        return cv2.inpaint(img_bgr, m, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def overlay_mask_on_bgr(img_bgr: np.ndarray, mask: np.ndarray, color: tuple = (0, 0, 255), alpha: float = 0.3) -> np.ndarray:
        """
        Наложить цветную полупрозрачную маску на BGR-изображение.
        color - (B,G,R) значение 0-255 для подсветки.
        mask - uint8 0/255, single channel.
        alpha - непрозрачность маски (0..1).
        Возвращает BGR uint8.
        """
        img = img_bgr.copy().astype(np.float32)
        overlay = np.zeros_like(img, dtype=np.float32)
        # Broadcast mask to 3 channels and set color
        if mask.ndim == 2:
            m3 = np.stack([mask]*3, axis=-1) / 255.0  # 0..1
        else:
            m3 = (mask.astype(np.uint8) != 0).astype(np.float32)
        overlay[:, :, 0] = color[0]
        overlay[:, :, 1] = color[1]
        overlay[:, :, 2] = color[2]
        # Blend only where mask is present
        alpha_mask = (m3[..., 0] > 0).astype(np.float32) * alpha
        alpha_mask = np.expand_dims(alpha_mask, axis=-1)
        out = img * (1.0 - alpha_mask) + overlay * alpha_mask
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def blend_images(original, processed, alpha=0.7):
        """
        Смешивание оригинального и обработанного изображения
        """
        return cv2.addWeighted(original, alpha, processed, 1 - alpha, 0)

# -------------------- Нейросетевая часть --------------------

class WatermarkDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

class WatermarkRemoverCNN(nn.Module):
    def __init__(self):
        super(WatermarkRemoverCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class WatermarkRemover:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = WatermarkRemoverCNN().to(self.device)
        self.utils = WatermarkUtils()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def remove_watermark_simple(self, image_path, watermark_coords=None, thresh=150, invert=False, kernel=5):
        """
        Удаление водяного знака методом клонирования (OpenCV)
        """
        img = self.utils.load_image(image_path)
        
        if watermark_coords is None:
            # Автоматическое обнаружение водяных знаков
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = self.utils.make_mask_from_gray(gray, thresh=thresh, invert=invert, k=kernel)
        else:
            # Используем предоставленные координаты
            mask = np.zeros(img.shape[:2], np.uint8)
            for coord in watermark_coords:
                x, y, w, h = coord
                mask[y:y+h, x:x+w] = 255
        
        # Применяем inpainting
        result = self.utils.inpaint_bgr(img, mask)
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), mask
    
    def train(self, train_loader, epochs=10, save_path='watermark_remover_model.pth'):
        """
        Обучение нейросетевой модели
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
        
        # Сохранение модели после обучения
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')
    
    def remove_watermark_advanced(self, image_path):
        """
        Удаление водяного знака с помощью нейросети
        """
        self.model.eval()
        with torch.no_grad():
            # Загрузка и преобразование изображения
            image = Image.open(image_path).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            
            # Конвертируем обратно в изображение
            output_image = output.squeeze(0).cpu().numpy()
            output_image = np.transpose(output_image, (1, 2, 0))
            output_image = (output_image * 255).astype(np.uint8)
            
            return output_image
    
    def compare_results(self, image_path, watermark_coords=None, thresh=150, invert=False, kernel=5):
        """
        Сравнение результатов разных методов
        """
        original = self.utils.load_image(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Простое удаление
        simple_result, mask = self.remove_watermark_simple(image_path, watermark_coords, thresh, invert, kernel)
        
        # Визуализация маски с наложением
        mask_preview = self.utils.overlay_mask_on_bgr(original, mask, color=(0, 0, 255), alpha=0.35)
        mask_preview_rgb = cv2.cvtColor(mask_preview, cv2.COLOR_BGR2RGB)
        
        # Нейросетевое удаление (если модель обучена)
        try:
            advanced_result = self.remove_watermark_advanced(image_path)
            has_advanced = True
        except:
            advanced_result = original_rgb
            has_advanced = False
        
        # Визуализация результатов
        plt.figure(figsize=(15, 10 if has_advanced else 15))
        
        plt.subplot(2, 2, 1)
        plt.imshow(original_rgb)
        plt.title('Оригинал')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(mask_preview_rgb)
        plt.title('Обнаруженная маска')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(simple_result)
        plt.title('Простое удаление (inpaint)')
        plt.axis('off')
        
        if has_advanced:
            plt.subplot(2, 2, 4)
            plt.imshow(advanced_result)
            plt.title('Нейросетевое удаление')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return simple_result, advanced_result if has_advanced else simple_result

# -------------------- CLI интерфейс --------------------

def run_cli(args):
    """Выполнить обработку через CLI."""
    remover = WatermarkRemover(args.model)
    
    if args.input is None:
        print("Входной файл не указан: создаю тестовое изображение.")
        img = WatermarkUtils.make_sample_image()
        WatermarkUtils.save_image_bgr(img, "sample_input.png")
        image_path = "sample_input.png"
    else:
        image_path = args.input
    
    if args.method == "simple":
        result, _ = remover.remove_watermark_simple(
            image_path, 
            watermark_coords=None if args.auto_detect else [(args.x, args.y, args.w, args.h)],
            thresh=args.thresh,
            invert=args.invert,
            kernel=args.kernel
        )
    elif args.method == "advanced":
        result = remover.remove_watermark_advanced(image_path)
    else:  # compare
        result_simple, result_advanced = remover.compare_results(
            image_path,
            watermark_coords=None if args.auto_detect else [(args.x, args.y, args.w, args.h)],
            thresh=args.thresh,
            invert=args.invert,
            kernel=args.kernel
        )
        result = result_advanced
    
    out = args.output or "result.png"
    # Convert RGB to BGR for saving
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    WatermarkUtils.save_image_bgr(result_bgr, out)
    print(f"Готово: {out}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Watermark Removal Tool")
    parser.add_argument("--input", "-i", help="Входной файл изображения")
    parser.add_argument("--output", "-o", help="Выходной файл")
    parser.add_argument("--method", "-m", choices=["simple", "advanced", "compare"], 
                       default="compare", help="Метод удаления водяного знака")
    parser.add_argument("--model", help="Путь к предобученной модели")
    parser.add_argument("--thresh", type=int, default=150, help="Порог для бинаризации")
    parser.add_argument("--invert", action="store_true", help="Инвертировать маску")
    parser.add_argument("--kernel", type=int, default=5, help="Размер ядра морфологии")
    parser.add_argument("--auto-detect", action="store_true", help="Автоматическое обнаружение водяного знака")
    parser.add_argument("--x", type=int, default=0, help="X координата водяного знака")
    parser.add_argument("--y", type=int, default=0, help="Y координата водяного знака")
    parser.add_argument("--w", type=int, default=100, help="Ширина водяного знака")
    parser.add_argument("--h", type=int, default=50, help="Высота водяного знака")
    
    args = parser.parse_args()
    
    return run_cli(args)

if __name__ == "__main__":
    # Если запущено как скрипт, использовать CLI
    if len(sys.argv) > 1:
        sys.exit(main())
    
    # Иначе показать пример использования
    print("Watermark Removal Tool")
    print("Использование: python script.py --input image.jpg --output result.png --method compare")
    
    # Создать пример и показать результат
    remover = WatermarkRemover()
    sample_img = WatermarkUtils.make_sample_image()
    WatermarkUtils.save_image_bgr(sample_img, "sample_image.jpg")
    
    print("\nСоздан пример изображения: sample_image.jpg")
    print("Запуск сравнения методов...")
    
    remover.compare_results("sample_image.jpg")
