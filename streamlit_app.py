#!/usr/bin/env python3
"""
Инструмент для удаления водяных знаков с изображений с помощью нейросети LaMa.

Usage:
    python watermark_remover.py input.jpg output.jpg 100 50 300 150
"""

import argparse
import cv2
import numpy as np
import torch
import sys
from omegaconf import OmegaConf
from lama.model import build_model
from lama.utils import load_checkpoint, move_to_device, minmax_scale



def remove_watermark_with_lama(
    input_path: str,
    output_path: str,
    x1: int, y1: int, x2: int, y2: int,
    device: str = 'cpu'
):
    """
    Удаление водяного знака с помощью нейросети LaMa.
    """
    try:
        # Загрузка изображения
        image = cv2.imread(input_path)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {input_path}")

        h, w = image.shape[:2]

        # Проверка корректности координат
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("Некорректные координаты области для удаления")

        # Создание маски
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Преобразование в формат для модели
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        # Конфигурация модели
        config = OmegaConf.create({
            'model': {
                'backbone': 'swin_unet',
                'width': 128,
                'num_stages': 4,
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
            }
        })

        # Инициализация и загрузка модели
        model = build_model(config)
        model.eval()

        checkpoint = load_checkpoint('big-lama', strict=False)
        model.load_state_dict(checkpoint['model_state'])

        model = move_to_device(model, device)
        image_tensor = move_to_device(image_tensor, device)
        mask_tensor = move_to_device(mask_tensor, device)

        # Инпейнтинг
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))
            result = output[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()

        # Конвертация результата
        result_rgb = (result * 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        # Сохранение
        cv2.imwrite(output_path, result_bgr)
        print(f"[SUCCESS] Результат сохранён: {output_path}")

    except Exception as e:
        print(f"[ERROR] Ошибка обработки: {str(e)}")
        sys.exit(1)



def main():
    parser = argparse.ArgumentParser(
        description="Удаление водяных знаков с помощью нейросети LaMa"
    )
    parser.add_argument("input", help="Путь к исходному изображению")
    parser.add_argument("output", help="Путь для сохранения результата")
    parser.add_argument("x1", type=int, help="X координата левого верхнего угла")
    parser.add_argument("y1", type=int, help="Y координата левого верхнего угла")
    parser.add_argument("x2", type=int, help="X координата правого нижнего угла")
    parser.add_argument("y2", type=int, help="Y координата правого нижнего угла")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Устройство для вычислений (cpu или cuda)"
    )

    args = parser.parse_args()

    remove_watermark_with_lama(
        input_path=args.input,
        output_path=args.output,
        x1=args.x1,
        y1=args.y1,
        x2=args.x2,
        y2=args.y2,
        device=args.device
    )



if __name__ == "__main__":
    main()
