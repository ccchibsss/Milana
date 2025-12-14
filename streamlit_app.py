"""
prepare_photo_processor_files.py

Writes two files to the current directory:
- photo_processor_pro.py
- requirements.txt

Run: python prepare_photo_processor_files.py
"""
from pathlib import Path

APP_PATH = Path("photo_processor_pro.py")
REQS_PATH = Path("requirements.txt")

APP_CODE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photo_processor_pro.py

Robust image batch processor:
- optional Streamlit UI (if streamlit is installed)
- optional rembg background removal (if rembg is installed)
- watermark removal via OpenCV inpainting
- safe PIL <-> OpenCV conversions
- clear logging and CLI
"""
from pathlib import Path
from datetime import datetime
import logging
import traceback
import io
import sys
import argparse
import os

# Optional imports
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    rembg_remove = None
    REMBG_AVAILABLE = False

try:
    import cv2
except Exception:
    raise SystemExit("OpenCV (cv2) is required. Install with: pip install opencv-python-headless")

try:
    from PIL import Image, UnidentifiedImageError
except Exception:
    raise SystemExit("Pillow is required. Install with: pip install pillow")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# Logging
def setup_logger(log_dir: Path = Path(".")) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("photo_processor_pro")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_filename, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

logger = setup_logger(Path("."))

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def validate_paths(input_path: Path, output_path: Path) -> tuple[bool, str]:
    if not input_path.exists():
        try:
            input_path.mkdir(parents=True, exist_ok=True)
            return False, f"Input folder '{input_path}' was created. Add images and re-run."
        except Exception as e:
            return False, f"Cannot create input folder '{input_path}': {e}"
    if not input_path.is_dir():
        return False, f"Input path '{input_path}' is not a directory."
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create or use output folder '{output_path}': {e}"
    return True, "OK"

def get_image_files(input_path: Path) -> list[Path]:
    return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])

def read_image_pil(path: Path) -> Image.Image:
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            return im.convert("RGBA")
        return im.convert("RGB")

def pil_to_cv2(img: Image.Image):
    import numpy as np
    arr = np.array(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    raise ValueError("Unsupported image shape")

def cv2_to_pil(img_cv):
    import numpy as np
    arr = img_cv
    if arr.ndim == 2:
        return Image.fromarray(arr)
    if arr.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    if arr.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA))
    raise ValueError("Unsupported image shape")

def remove_background_safe(pil_img: Image.Image) -> Image.Image:
    if not REMBG_AVAILABLE:
        logger.warning("rembg is not available; skipping background removal.")
        return pil_img
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    try:
        out_bytes = rembg_remove(buf.read())
    except Exception as e:
        logger.error("rembg error: %s", e)
        raise
    out_buf = io.BytesIO(out_bytes)
    out_img = Image.open(out_buf).convert("RGBA")
    return out_img

def remove_watermark_cv(img_cv, threshold: int = 220, radius: int = 5):
    import numpy as np
    # Convert 4-channel to 3-channel for processing
    if img_cv.shape[2] == 4:
        bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img_cv.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) > 30:
            cv2.drawContours(final_mask, [c], -1, 255, -1)
    if final_mask.sum() > 0:
        inpainted = cv2.inpaint(bgr, final_mask, radius, cv2.INPAINT_TELEA)
        if img_cv.shape[2] == 4:
            alpha = img_cv[:, :, 3]
            inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
            inpainted[:, :, 3] = alpha
        return inpainted
    return img_cv

def save_image(img_cv, out_path: Path, fmt: str, jpeg_quality: int = 95) -> Path:
    pil = cv2_to_pil(img_cv)
    if fmt.lower().startswith("png"):
        out = out_path.with_suffix(".png")
        pil.save(out, format="PNG", optimize=True)
    else:
        out = out_path.with_suffix(".jpg")
        pil.convert("RGB").save(out, format="JPEG", quality=jpeg_quality, optimize=True)
    return out

def process_images(input_dir: Path, output_dir: Path,
                   remove_bg: bool = True, remove_wm: bool = False,
                   wm_threshold: int = 220, wm_radius: int = 5,
                   fmt: str = "PNG", jpeg_q: int = 95):
    ok, msg = validate_paths(input_dir, output_dir)
    if not ok:
        logger.info(msg)
        return [msg]
    images = get_image_files(input_dir)
    if not images:
        msg = f"No supported images found in '{input_dir}'. Add files and re-run."
        logger.info(msg)
        return [msg]

    logs = []
    iterator = (tqdm(images, desc="Processing") if TQDM_AVAILABLE else images)
    total = len(images)
    for idx, img_path in enumerate(iterator, 1):
        try:
            pil_img = read_image_pil(img_path)
            if remove_bg:
                pil_img = remove_background_safe(pil_img)
            img_cv = pil_to_cv2(pil_img)
            if remove_wm:
                img_cv = remove_watermark_cv(img_cv, wm_threshold, wm_radius)
            out_name = img_path.stem
            out_path = output_dir / out_name
            saved = save_image(img_cv, out_path, "PNG" if fmt.lower().startswith("png") else "JPEG", jpeg_q)
            msg = f"[OK] {idx}/{total}: {img_path.name} -> {saved.name}"
            logger.info(msg)
            logs.append(msg)
        except UnidentifiedImageError:
            msg = f"[ERR] {idx}/{total}: cannot open {img_path.name}"
            logger.error(msg)
            logs.append(msg)
        except Exception as e:
            msg = f"[ERR] {idx}/{total}: error {img_path.name}: {e}"
            logger.error(msg + "\\n" + traceback.format_exc())
            logs.append(msg)
    return logs

def run_cli():
    parser = argparse.ArgumentParser(description="Photo Processor Pro (CLI)")
    parser.add_argument("--input", "-i", type=str, default="./input")
    parser.add_argument("--output", "-o", type=str, default="./output")
    parser.add_argument("--no-bg", dest="remove_bg", action="store_false", help="Disable background removal")
    parser.add_argument("--remove-wm", dest="remove_wm", action="store_true", help="Enable watermark removal (inpaint)")
    parser.add_argument("--wm-threshold", type=int, default=220)
    parser.add_argument("--wm-radius", type=int, default=5)
    parser.add_argument("--format", choices=["PNG", "JPEG"], default="PNG")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        try:
            input_dir.mkdir(parents=True, exist_ok=True)
            msg = f"Input folder '{input_dir}' created. Please add images and re-run."
            logger.info(msg)
            print(msg)
            return
        except Exception as e:
            logger.error("Cannot create input folder '%s': %s", input_dir, e)
            sys.exit(1)

    logs = process_images(input_dir, output_dir,
                          remove_bg=args.remove_bg,
                          remove_wm=args.remove_wm,
                          wm_threshold=args.wm_threshold,
                          wm_radius=args.wm_radius,
                          fmt=args.format,
                          jpeg_q=args.jpeg_quality)
    for line in logs:
        print(line)
    print("Done. Logs written to current directory.")

def run_streamlit_app():
    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("Photo Processor Pro")
    st.caption("Batch image processing - background removal and watermark inpaint")

    with st.sidebar:
        input_dir = st.text_input("Input folder", "./input")
        output_dir = st.text_input("Output folder", "./output")
        remove_bg = st.checkbox("Remove background (rembg)", value=True)
        remove_wm = st.checkbox("Remove watermark (inpaint)", value=False)
        wm_threshold = st.slider("Watermark detection threshold", 100, 255, 220) if remove_wm else 220
        wm_radius = st.slider("Inpaint radius", 1, 30, 5) if remove_wm else 5
        fmt = st.radio("Format", ("PNG", "JPEG"))
        jpeg_q = st.slider("JPEG quality", 70, 100, 95) if fmt == "JPEG" else 95
        run = st.button("Start")

    logs = []
    if run:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        if not input_path.exists():
            try:
                input_path.mkdir(parents=True, exist_ok=True)
                st.info(f"Input folder '{input_path}' created. Please add images and re-run.")
                return
            except Exception as e:
                st.error(f"Cannot create input folder '{input_path}': {e}")
                return

        ok, msg = validate_paths(input_path, output_path)
        if not ok:
            st.error(msg)
            return

        images = get_image_files(input_path)
        if not images:
            st.warning("No supported images found in the input folder.")
            return

        progress = st.progress(0)
        log_box = st.empty()
        total = len(images)
        for idx, img_path in enumerate(images, 1):
            try:
                pil_img = read_image_pil(img_path)
                if remove_bg:
                    pil_img = remove_background_safe(pil_img)
                img_cv = pil_to_cv2(pil_img)
                if remove_wm:
                    img_cv = remove_watermark_cv(img_cv, wm_threshold, wm_radius)
                out_name = img_path.stem
                out_path = output_path / out_name
                saved = save_image(img_cv, out_path, fmt, jpeg_q)
                msg = f"[OK] {idx}/{total}: {img_path.name} -> {saved.name}"
                logs.append(msg)
                logger.info(msg)
            except Exception as e:
                err = f"[ERR] {idx}/{total}: {img_path.name} - {e}"
                logs.append(err)
                logger.error(err + "\\n" + traceback.format_exc())
            progress.progress(idx / total)
            log_box.code("\\n".join(logs[-20:]))
        st.success("Processing finished.")

    if logs:
        with st.expander("Full log"):
            st.code("\\n".join(logs))
    else:
        st.info("Logs will appear here after processing.")

if __name__ == "__main__":
    # Prefer Streamlit if available and likely invoked by streamlit
    cli_call = " ".join(os.sys.argv).lower()
    if STREAMLIT_AVAILABLE and ("streamlit" in cli_call or "streamlit" in os.path.basename(sys.argv[0]).lower()):
        run_streamlit_app()
    else:
        run_cli()
'''

REQS_CONTENT = '''# pinned minimal versions
pillow==9.5.0
numpy==1.26.0
opencv-python-headless==4.8.1.78
tqdm==4.65.0

# optional for improved background removal
rembg==0.3.6
onnxruntime==1.15.1

# optional UI
streamlit==1.26.0
'''

def write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")
    print(f"Wrote: {path.resolve()}")

def main():
    write_file(APP_PATH, APP_CODE)
    write_file(REQS_PATH, REQS_CONTENT)
    print("Files created. Install dependencies with:")
    print("  python -m pip install -r requirements.txt")

if __name__ == "__main__":
    main()
