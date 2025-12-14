# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_photo_processor_files.py

Creates a corrected, runnable 'photo_processor_pro.py' and a pinned 'requirements.txt'.
Run:
    python create_photo_processor_files.py
"""
from pathlib import Path

APP_PATH = Path("photo_processor_pro.py")
REQS_PATH = Path("requirements.txt")

APP_CODE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photo_processor_pro.py

Robust, professional image batch processor:
- optional Streamlit UI (if streamlit is installed)
- optional rembg background removal (if rembg is installed)
- watermark removal via OpenCV inpainting
- safe PIL <-> OpenCV conversions
- scaling, config file support, multithreading
- clear logging and CLI
"""
from pathlib import Path
from datetime import datetime
import json
import io
import sys
import argparse
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

# Third-party imports with graceful degradation
try:
    import cv2
except Exception:
    raise SystemExit("OpenCV (cv2) is required. Install with: pip install opencv-python-headless")

try:
    import numpy as np
except Exception:
    raise SystemExit("numpy is required. Install with: pip install numpy")

try:
    from PIL import Image, UnidentifiedImageError
except Exception:
    raise SystemExit("Pillow is required. Install with: pip install pillow")

try:
    from rembg import remove as rembg_remove  # optional
    REMBG_AVAILABLE = True
except Exception:
    rembg_remove = None
    REMBG_AVAILABLE = False

try:
    from tqdm import tqdm  # optional progress for CLI
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

try:
    import streamlit as st  # optional UI
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# --- Config ---
CONFIG_FILE = Path("config.json")
DEFAULT_CONFIG: Dict[str, Any] = {
    "input_dir": "./input",
    "output_dir": "./output",
    "remove_bg": True,
    "remove_wm": False,
    "wm_threshold": 220,
    "wm_radius": 5,
    "format": "PNG",          # "PNG" or "JPEG"
    "jpeg_quality": 95,
    "max_width": 1920,
    "max_height": 1080,
    "num_threads": 4
}

# --- Logging ---
def setup_logger(log_dir: Path = Path(".")) -> "logging.Logger":
    import logging
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

# --- Config IO ---
def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                return {**DEFAULT_CONFIG, **cfg}
        except Exception as e:
            logger.warning("Cannot load config: %s. Using defaults.", e)
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save config: %s", e)

# --- Helpers ---
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def validate_paths(input_path: Path, output_path: Path) -> (bool, str):
    if not input_path.exists():
        try:
            input_path.mkdir(parents=True, exist_ok=True)
            return False, f"Input folder '{input_path}' created. Add images and re-run."
        except Exception as e:
            return False, f"Cannot create input folder '{input_path}': {e}"
    if not input_path.is_dir():
        return False, f"Input path '{input_path}' is not a directory."
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create/use output folder '{output_path}': {e}"
    return True, "OK"

def get_image_files(input_path: Path) -> List[Path]:
    return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])

def read_image_pil(path: Path) -> Image.Image:
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            return im.convert("RGBA")
        return im.convert("RGB")

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    raise ValueError("Unsupported image shape")

def cv2_to_pil(img_cv: np.ndarray) -> Image.Image:
    if img_cv.ndim == 2:
        return Image.fromarray(img_cv)
    if img_cv.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    if img_cv.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))
    raise ValueError("Unsupported image shape")

def scale_image(pil_img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    w, h = pil_img.size
    if w <= max_width and h <= max_height:
        return pil_img
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return pil_img.resize((new_w, new_h), Image.LANCZOS)

def remove_background_safe(pil_img: Image.Image) -> Image.Image:
    if not REMBG_AVAILABLE:
        logger.debug("rembg not available; skipping background removal.")
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

def remove_watermark_cv(img_cv: np.ndarray, threshold: int = 220, radius: int = 5) -> np.ndarray:
    # Accept BGR or BGRA
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

def save_image(img_cv: np.ndarray, out_path: Path, fmt: str, jpeg_quality: int = 95) -> Path:
    pil = cv2_to_pil(img_cv)
    if fmt.upper().startswith("PNG"):
        out = out_path.with_suffix(".png")
        pil.save(out, format="PNG", optimize=True)
    else:
        out = out_path.with_suffix(".jpg")
        pil.convert("RGB").save(out, format="JPEG", quality=int(jpeg_quality), optimize=True)
    return out

# --- Single image processing ---
def process_single_image(img_path: Path, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Optional[str]]:
    try:
        pil_img = read_image_pil(img_path)
        pil_img = scale_image(pil_img, int(config.get("max_width", 1920)), int(config.get("max_height", 1080)))

        if config.get("remove_bg", True):
            pil_img = remove_background_safe(pil_img)

        img_cv = pil_to_cv2(pil_img)

        if config.get("remove_wm", False):
            img_cv = remove_watermark_cv(img_cv, int(config.get("wm_threshold", 220)), int(config.get("wm_radius", 5)))

        out_name = img_path.stem
        out_path = output_dir / out_name
        saved = save_image(img_cv, out_path, config.get("format", "PNG"), int(config.get("jpeg_quality", 95)))

        return {"status": "OK", "message": f"[OK] {img_path.name} -> {saved.name}", "input": str(img_path), "output": str(saved)}
    except UnidentifiedImageError:
        return {"status": "ERROR", "message": f"[ERR] Cannot open {img_path.name}", "input": str(img_path), "output": None}
    except Exception as e:
        logger.exception("Error processing %s", img_path)
        return {"status": "ERROR", "message": f"[ERR] {img_path.name}: {e}", "input": str(img_path), "output": None}

# --- Batch processing with threads ---
def process_images(input_dir: Path, output_dir: Path, config: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    ok, msg = validate_paths(input_dir, output_dir)
    if not ok:
        logger.info(msg)
        return [{"status": "ERROR", "message": msg}]
    images = get_image_files(input_dir)
    if not images:
        msg = f"No supported images found in '{input_dir}'. Add files and re-run."
        logger.info(msg)
        return [{"status": "ERROR", "message": msg}]

    logs: List[Dict[str, Optional[str]]] = []
    num_threads = max(1, int(config.get("num_threads", 4)))
    use_tqdm = TQDM_AVAILABLE and not STREAMLIT_AVAILABLE
    if use_tqdm:
        iterator = tqdm(images, desc="Processing", unit="img")
    else:
        iterator = images

    # Use ThreadPoolExecutor; submit tasks and collect results to preserve order
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_single_image, img, output_dir, config): img for img in images}
        if use_tqdm:
            completed = 0
            for fut in as_completed(futures):
                res = fut.result()
                logs.append(res)
                completed += 1
                iterator.update(1)
                tqdm.write(res["message"])
            iterator.close()
        else:
            # preserve original ordering by image list
            for img in images:
                fut = next(f for f, p in futures.items() if p == img)
                res = fut.result()
                logs.append(res)
                logger.info(res["message"])

    return logs

# --- CLI ---
def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Photo Processor Pro (CLI)")
    parser.add_argument("--input", "-i", type=str, default="./input")
    parser.add_argument("--output", "-o", type=str, default="./output")
    parser.add_argument("--no-bg", dest="remove_bg", action="store_false", help="Disable background removal")
    parser.add_argument("--remove-wm", dest="remove_wm", action="store_true", help="Enable watermark removal")
    parser.add_argument("--wm-threshold", type=int, default=220)
    parser.add_argument("--wm-radius", type=int, default=5)
    parser.add_argument("--format", choices=["PNG", "JPEG"], default="PNG")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--max-height", type=int, default=1080)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    config = load_config()
    config.update({
        "input_dir": args.input,
        "output_dir": args.output,
        "remove_bg": args.remove_bg,
        "remove_wm": args.remove_wm,
        "wm_threshold": args.wm_threshold,
        "wm_radius": args.wm_radius,
        "format": args.format,
        "jpeg_quality": args.jpeg_quality,
        "max_width": args.max_width,
        "max_height": args.max_height,
        "num_threads": args.num_threads
    })
    save_config(config)

    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])

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

    logs = process_images(input_dir, output_dir, config)
    for entry in logs:
        print(entry["message"])
    print("Done. Logs written to current directory.")

# --- Streamlit UI ---
def run_streamlit_app() -> None:
    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("Photo Processor Pro")
    st.caption("Batch image processing — background removal, watermark inpaint, scaling")

    config = load_config()

    with st.sidebar:
        st.header("Settings")
        input_dir = st.text_input("Input folder", config["input_dir"])
        output_dir = st.text_input("Output folder", config["output_dir"])
        remove_bg = st.checkbox("Remove background (rembg)", value=config["remove_bg"])
        remove_wm = st.checkbox("Remove watermark (inpaint)", value=config["remove_wm"])
        wm_threshold = st.slider("Watermark detection threshold", 100, 255, int(config["wm_threshold"])) if remove_wm else int(config["wm_threshold"])
        wm_radius = st.slider("Inpaint radius", 1, 30, int(config["wm_radius"])) if remove_wm else int(config["wm_radius"])
        fmt = st.radio("Format", ("PNG", "JPEG"), index=0 if config["format"] == "PNG" else 1)
        jpeg_q = st.slider("JPEG quality", 70, 100, int(config["jpeg_quality"])) if fmt == "JPEG" else int(config["jpeg_quality"])
        max_width = st.number_input("Max width (px)", min_value=100, value=int(config["max_width"]), step=100)
        max_height = st.number_input("Max height (px)", min_value=100, value=int(config["max_height"]), step=100)
        num_threads = st.number_input("Threads count", min_value=1, max_value=32, value=int(config["num_threads"]), step=1)

        if st.button("Save config"):
            cfg = dict(input_dir=input_dir, output_dir=output_dir, remove_bg=remove_bg, remove_wm=remove_wm,
                       wm_threshold=wm_threshold, wm_radius=wm_radius, format=fmt, jpeg_quality=jpeg_q,
                       max_width=int(max_width), max_height=int(max_height), num_threads=int(num_threads))
            save_config(cfg)
            st.success("Configuration saved!")

        run = st.button("Start processing")

    logs: List[str] = []
    preview_col1, preview_col2 = st.columns(2)

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

        cfg = dict(input_dir=str(input_path), output_dir=str(output_path), remove_bg=remove_bg, remove_wm=remove_wm,
                   wm_threshold=int(wm_threshold), wm_radius=int(wm_radius), format=fmt, jpeg_quality=int(jpeg_q),
                   max_width=int(max_width), max_height=int(max_height), num_threads=int(num_threads))
        save_config(cfg)

        progress = st.progress(0)
        log_box = st.empty()
        total = len(images)

        for idx, img_path in enumerate(images, 1):
            result = process_single_image(img_path, output_path, cfg)
            logs.append(result["message"])
            log_box.code("\\n".join(logs[-20:]))

            if idx == 1 and result.get("status") == "OK" and result.get("output"):
                try:
                    preview_col1.image(str(result["input"]), caption="Input", use_column_width=True)
                    preview_col2.image(str(result["output"]), caption="Output", use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not display preview: {e}")

            progress.progress(idx / total)

        st.success("Processing finished!")

    if logs:
        with st.expander("Full log"):
            st.code("\\n".join(logs))
    else:
        st.info("Logs will appear here after processing.")

    with st.expander("Current configuration"):
        st.json(load_config())

# --- Entrypoint ---
if __name__ == "__main__":
    cli_call = " ".join(os.sys.argv).lower()
    if STREAMLIT_AVAILABLE and ("streamlit" in cli_call or "streamlit" in os.path.basename(sys.argv[0]).lower()):
        run_streamlit_app()
    else:
        run_cli()
'''

REQS_CONTENT = '''# Minimal, pinned recommended versions
pillow==9.5.0
numpy==1.26.0
opencv-python-headless==4.8.1.78
tqdm==4.65.0

# Optional for background removal (install if you need rembg)
rembg==0.3.6
onnxruntime==1.15.1

# Optional UI
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
