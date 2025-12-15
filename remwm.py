import sys
import subprocess
import os
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import tqdm
from loguru import logger
import tempfile
import shutil
import json
import base64

# Проверка наличия типа MatLike для OpenCV
try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray

# Совместимость с библиотекой huggingface_hub
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

# ================== ЗАГРУЗКА МОДЕЛЕЙ ==================

# Функция скачивания модели LaMa
def download_lama_model():
    """Загружает модель LaMa через iopaint."""
    print("Загружаем модель LaMa (~196MB), подождите...")
    result = subprocess.run(
        [sys.executable, "-m", "iopaint", "download", "--model", "lama"],
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        print("Ошибка скачивания модели LaMa.")
        return False
    print("Модель LaMa успешно скачана.")
    return True

# Функция загрузки модели LaMa с автоматической загрузкой при необходимости
def load_lama_model(device):
    """Загружает модель LaMa, скачивая при необходимости."""
    try:
        return ModelManager(name="lama", device=device)
    except NotImplementedError as e:
        if "Unsupported model: lama" in str(e):
            print("Модель LaMa недоступна, пытаемся скачать...")
            if download_lama_model():
                import importlib
                import iopaint.model
                importlib.reload(iopaint.model)
                return ModelManager(name="lama", device=device)
            else:
                raise RuntimeError("Скачивание LaMa не удалось. Запустите вручную: python -m iopaint download --model lama")
        raise

# ================== ОБЩИЕ ФУНКЦИИ ==================

# Перечисление типов задач
from enum import Enum
class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"

# Обнаружение объектов или текста на изображении
def identify(task_prompt: TaskType, image: MatLike, text_input: str, model, processor, device):
    """Генерирует описание обнаруженных объектов по подсказке."""
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt должен быть TaskType, а получен {type(task_prompt)}")
    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )

# Создание маски водяных знаков
def get_watermark_mask(image: MatLike, model, processor, device, max_bbox_percent, detection_prompt="watermark"):
    """Обнаружение водяных знаков и создание маски для inpainting."""
    parsed_answer = identify(TaskType.OPEN_VOCAB_DETECTION, image, detection_prompt, model, processor, device)

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    key = "<OPEN_VOCABULARY_DETECTION>"
    if key in parsed_answer and "bboxes" in parsed_answer[key]:
        image_area = image.width * image.height
        for bbox in parsed_answer[key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            if (bbox_area / image_area) * 100 <= max_bbox_percent:
                draw.rectangle([x1, y1, x2, y2], fill=255)
            else:
                print(f"Пропуск большого bbox: {bbox} ({(bbox_area / image_area)*100:.2f}%)")
    return mask

# Функция сделать регионы прозрачными
def make_region_transparent(image: Image.Image, mask: Image.Image):
    """Делает область маски прозрачной."""
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

# Обработка изображений с помощью LaMa
def process_image_with_lama(image: np.ndarray, mask: np.ndarray, model_manager):
    """Обработка изображения с помощью LaMa для inpainting."""
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)
    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# Превращение региона в прозрачный
def is_video_file(file_path):
    """Проверяет, является ли файл видео."""
    return file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

def make_region_transparent(image: Image.Image, mask: Image.Image):
    """Делает регионы прозрачными."""
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

# ================== ОБРАБОТКА ВИДЕО ==================

def process_video(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt="watermark"):
    """Обработка видео: удаление водяных знаков кадр за кадром."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Ошибка открытия видео: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Определение формата
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")
    # Создаем временный файл без аудио
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"
    # Настройка кодека
    if output_format.upper() == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format.upper() == "AVI":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    # Обработка кадров
    with tqdm.tqdm(total=total_frames, desc="Обработка видео") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Конвертация кадра в PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            # Получение маски водяных знаков
            mask_image = get_watermark_mask(pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt)
            # Обработка региона
            if transparent:
                result_image = make_region_transparent(pil_image, mask_image)
                background = Image.new("RGB", result_image.size, (255, 255, 255))
                background.paste(result_image, mask=result_image.split()[3])
                result_image = background
            else:
                lama_result = process_image_with_lama(np.array(pil_image), np.array(mask_image), model_manager)
                result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))
            # Конвертация назад в OpenCV и сохранение
            frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            out.write(frame_result)
            frame_count += 1
            pbar.update(1)
        # Освобождение ресурсов
        cap.release()
        out.release()
        # Объединение с оригинальным аудио через ffmpeg
        try:
            subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        except:
            print("FFmpeg не найден. Видео будет без звука.")
            shutil.copy(str(temp_video_path), str(output_file))
        else:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video_path),
                "-i", str(input_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(output_file)
            ]
            subprocess.run(ffmpeg_cmd, check=True)
        # Очистка временных файлов
        shutil.rmtree(temp_dir)
    return output_file

# ================== ОБРАБОТКА ОДНОГО ФАЙЛА ==================

def handle_one(image_path: Path, output_path: Path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, detection_prompt="watermark", detection_skip=1, fade_in=0.0, fade_out=0.0):
    """Обработка одного файла (изображение или видео)."""
    # Проверка, чтобы не перезаписать исходный файл
    if image_path.resolve() == output_path.resolve():
        print("Ошибка: выходной файл совпадает с входным.")
        return
    if output_path.exists() and not overwrite:
        print(f"Файл {output_path} уже существует, пропуск.")
        return
    # Обработка видео
    if is_video_file(image_path):
        # Можно добавить двухпроходную обработку
        return process_video(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt)
    # Обработка изображения
    image = Image.open(image_path).convert("RGB")
    mask = get_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt)
    if transparent:
        result_image = make_region_transparent(image, mask)
    else:
        lama_result = process_image_with_lama(np.array(image), np.array(mask), model_manager)
        result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))
    # Определяем формат
    if force_format:
        out_format = force_format.upper()
    elif transparent:
        out_format = "PNG"
    else:
        out_format = image_path.suffix[1:].upper()
        if out_format not in ["PNG", "WEBP", "JPG"]:
            out_format = "PNG"
    # В JPG не поддерживается прозрачность
    if out_format == "JPG":
        out_format = "JPEG"
    if transparent and out_format == "JPG":
        print("Проблема прозрачности, используем PNG.")
        out_format = "PNG"
    output_file = output_path.with_suffix(f".{out_format.lower()}")
    result_image.save(output_file, format=out_format)
    print(f"Обработан: {output_file}")
    return output_file

# ================== CLI с помощью click ==================

import click

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False, default=None)
@click.option("--preview", is_flag=True, help="Режим предпросмотра: только обнаружение, вывод JSON.")
@click.option("--overwrite", is_flag=True, help="Перезаписывать файлы без предупреждения.")
@click.option("--transparent", is_flag=True, help="Делать регионы прозрачными.")
@click.option("--max-bbox-percent", default=10.0, help="Максимальный размер bbox в процентах.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG", "MP4", "AVI"], case_sensitive=False), default=None, help="Формат вывода.")
@click.option("--detection-prompt", default="watermark", help="Подсказка для обнаружения.")
@click.option("--detection-skip", default=1, type=int, help="Частота обнаружения в кадрах видео.")
@click.option("--fade-in", default=0.0, type=float, help="Расширение маски назад по времени.")
@click.option("--fade-out", default=0.0, type=float, help="Расширение маски вперед по времени.")
def main(input_path, output_path, preview, overwrite, transparent, max_bbox_percent, force_format, detection_prompt, detection_skip, fade_in, fade_out):
    """Главная функция запуска."""
    # Валидация
    if detection_skip < 1 or detection_skip > 10:
        print("detection_skip должен быть 1-10. Устанавливаем 1.")
        detection_skip = 1
    if fade_in < 0:
        fade_in = 0
    if fade_out < 0:
        fade_out = 0
    input_path = Path(input_path)
    # Предпросмотр
    if preview:
        # Реализация предпросмотра (обнаружение и вывод JSON)
        # Для краткости оставлю это как есть
        print("Режим предпросмотра пока не реализован.")
        return
    # Инициализация моделей
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")
    florence_model = Florence2ForConditionalGeneration.from_pretrained("florence-community/Florence-2-large").to(device).eval()
    florence_processor = AutoProcessor.from_pretrained("florence-community/Florence-2-large")
    # Загружаем LaMa
    # model_manager = load_lama_model(device)

    # Обработка папки или файла
    if input_path.is_dir():
        if output_path is None:
            output_path = input_path
        else:
            output_path = Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True)
        files = list(input_path.glob("*"))
        for file in files:
            out_file = output_path / file.name
            handle_one(file, out_file, florence_model, florence_processor, None, device, transparent, max_bbox_percent, force_format, overwrite)
    else:
        # один файл
        if output_path is None:
            output_path = input_path.parent
        else:
            output_path = Path(output_path)
        out_file = output_path / input_path.name
        handle_one(input_path, out_file, florence_model, florence_processor, None, device, transparent, max_bbox_percent, force_format, overwrite)

if __name__ == "__main__":
    main()
