# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Processor Pro — исправленная версия с поддержкой нескольких наборов параметров.
- CLI: можно передать один набор параметров через флаги или несколько через JSON-файл (--presets).
- Streamlit: поддержка тех же настроек (если установлен).
- Результаты каждого набора параметров сохраняются в отдельной подпапке внутри выходной папки.
- ZIP архива создаётся вне выходной папки (в temp) для скачивания.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

# Optional deps
try:
    import cv2
except Exception as e:
    raise SystemExit("OpenCV (cv2) is required. Install via `pip install opencv-python`") from e

try:
    from rembg import remove as rembg_remove  # type: ignore
    HAS_REMBG = True
except Exception:
    rembg_remove = None
    HAS_REMBG = False

try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception:
    st = None  # type: ignore
    HAS_STREAMLIT = False

# --- Logging ---
def setup_logger() -> logging.Logger:
    fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(fn, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("photo_processor_pro")

logger = setup_logger()

# --- Types / Config ---
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

@dataclass
class Preset:
    label: str = "default"
    remove_bg: bool = True
    remove_wm: bool = False
    wm_threshold: int = 220
    wm_radius: int = 5
    fmt: str = "PNG"  # "PNG" or "JPEG"
    jpeg_q: int = 95
    target_width: Optional[int] = None
    target_height: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any], label: Optional[str] = None) -> "Preset":
        return cls(
            label=(label or d.get("label") or "preset"),
            remove_bg=bool(d.get("remove_bg", True)),
            remove_wm=bool(d.get("remove_wm", False)),
            wm_threshold=int(d.get("wm_threshold", 220)),
            wm_radius=int(d.get("wm_radius", 5)),
            fmt=str(d.get("fmt", "PNG")).upper(),
            jpeg_q=int(d.get("jpeg_q", 95)),
            target_width=(None if d.get("target_width") is None else int(d.get("target_width"))),
            target_height=(None if d.get("target_height") is None else int(d.get("target_height"))),
        )

# --- Utilities ---
def validate_file_extension(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS

def get_image_files(inp: Path) -> List[Path]:
    if not inp.exists() or not inp.is_dir():
        return []
    return [p for p in sorted(inp.iterdir()) if p.is_file() and validate_file_extension(p)]

def create_zip_of_output(output_dir: str, zip_name: Optional[str] = None) -> Path:
    outp = Path(output_dir).expanduser().resolve()
    if not outp.exists() or not outp.is_dir():
        raise FileNotFoundError(f"Output folder not found: {outp}")
    base_name = zip_name or f"{outp.name}_results"
    tmp_dir = Path(tempfile.gettempdir())
    zip_base = tmp_dir / base_name
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(outp))
    return Path(zip_path)

# --- Background removal ---
def rembg_background(pil_img: Image.Image) -> Image.Image:
    if not HAS_REMBG or rembg_remove is None:
        return pil_img
    try:
        out = rembg_remove(pil_img)
        if isinstance(out, (bytes, bytearray)):
            return Image.open(io.BytesIO(out))
        if isinstance(out, Image.Image):
            return out
    except Exception:
        logger.exception("rembg failed")
    return pil_img

def grabcut_background(pil_img: Image.Image) -> Image.Image:
    try:
        img = np.array(pil_img.convert("RGB"))
        h, w = img.shape[:2]
        scale = 512 / max(h, w) if max(h, w) > 512 else 1.0
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros(small.shape[:2], np.uint8)
        rect = (5, 5, max(1, small.shape[1] - 10), max(1, small.shape[0] - 10))
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        small_rgba = cv2.cvtColor(small, cv2.COLOR_RGB2RGBA)
        small_rgba[..., 3] = mask2 * 255
        alpha = cv2.resize(small_rgba[..., 3], (w, h), interpolation=cv2.INTER_LINEAR)
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img_rgba[..., 3] = alpha
        return Image.fromarray(img_rgba)
    except Exception:
        logger.exception("grabcut failed")
        return pil_img

def remove_background_pil(pil_img: Image.Image, use_rembg: bool) -> Image.Image:
    if use_rembg and HAS_REMBG:
        try:
            return rembg_background(pil_img)
        except Exception:
            logger.exception("rembg error; falling back to grabcut")
    return grabcut_background(pil_img)

# --- Watermark removal ---
def remove_watermark_cv(img_cv: np.ndarray, threshold: int = 220, radius: int = 5) -> np.ndarray:
    try:
        bgr = img_cv[..., :3].copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in contours:
            if cv2.contourArea(c) > 50:
                cv2.drawContours(mask, [c], -1, 255, -1)
        if np.any(mask):
            inpainted = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
            if img_cv.ndim == 3 and img_cv.shape[2] == 4:
                out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
                out[..., 3] = img_cv[..., 3]
                return out
            return inpainted
        return img_cv
    except Exception:
        logger.exception("remove_watermark_cv failed")
        return img_cv

# --- Resize / Save ---
def resize_image(img_cv: np.ndarray, target_width: Optional[int], target_height: Optional[int]) -> np.ndarray:
    h, w = img_cv.shape[:2]
    if target_width and target_height:
        return cv2.resize(img_cv, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA)
    if target_width and not target_height:
        scale = target_width / w
        return cv2.resize(img_cv, (int(target_width), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    if target_height and not target_width:
        scale = target_height / h
        return cv2.resize(img_cv, (max(1, int(w * scale)), int(target_height)), interpolation=cv2.INTER_AREA)
    return img_cv

def save_image(img_cv: np.ndarray, out_path: Path, preset: Preset) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_cv = resize_image(img_cv, preset.target_width, preset.target_height)
        if preset.fmt.upper().startswith("PNG"):
            # keep alpha channel if present
            cv2.imwrite(str(out_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            return True
        bgr = img_cv
        if img_cv.ndim == 3 and img_cv.shape[2] == 4:
            bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        success, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(preset.jpeg_q)])
        if success:
            out_path.write_bytes(buf.tobytes())
            return True
        return False
    except Exception:
        logger.exception("save_image failed")
        return False

# --- Processing logic ---
def process_one_image_bytes(name: str, data: bytes, preset: Preset, out_base: Path) -> str:
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGBA")
    except UnidentifiedImageError:
        return f"❌ {name}: not an image or corrupted"
    except Exception as e:
        logger.exception("open uploaded failed")
        return f"❌ {name}: open failed: {e}"

    processed_pil = remove_background_pil(pil, preset.remove_bg)
    if processed_pil.mode != "RGBA":
        processed_pil = processed_pil.convert("RGBA")
    img_cv = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGBA2BGRA)

    if preset.remove_wm:
        img_cv = remove_watermark_cv(img_cv, threshold=preset.wm_threshold, radius=preset.wm_radius)

    ext = ".png" if preset.fmt.upper().startswith("PNG") else ".jpg"
    out_folder = out_base / preset.label
    out_folder.mkdir(parents=True, exist_ok=True)
    out_name = Path(name).stem + ext
    out_path = out_folder / out_name

    ok = save_image(img_cv, out_path, preset)
    return (f"✅ {name} -> {preset.label}/{out_name}") if ok else (f"❌ {name} -> {preset.label}/{out_name} (save failed)")

def process_one_image_disk(path: Path, preset: Preset, out_base: Path) -> str:
    try:
        pil = Image.open(path).convert("RGBA")
    except UnidentifiedImageError:
        return f"❌ {path.name}: not an image or corrupted"
    except Exception as e:
        logger.exception("open disk failed")
        return f"❌ {path.name}: open failed: {e}"

    processed_pil = remove_background_pil(pil, preset.remove_bg)
    if processed_pil.mode != "RGBA":
        processed_pil = processed_pil.convert("RGBA")
    img_cv = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGBA2BGRA)

    if preset.remove_wm:
        img_cv = remove_watermark_cv(img_cv, threshold=preset.wm_threshold, radius=preset.wm_radius)

    ext = ".png" if preset.fmt.upper().startswith("PNG") else ".jpg"
    out_folder = out_base / preset.label
    out_folder.mkdir(parents=True, exist_ok=True)
    out_name = path.stem + ext
    out_path = out_folder / out_name

    ok = save_image(img_cv, out_path, preset)
    return (f"✅ {path.name} -> {preset.label}/{out_name}") if ok else (f"❌ {path.name} -> {preset.label}/{out_name} (save failed)")

def process_batch(
    input_dir: Optional[str],
    output_dir: str,
    presets: List[Preset],
    selected: Optional[List[str]] = None,
    uploaded_files: Optional[List[Tuple[str, bytes]]] = None,
    max_workers: int = 4
) -> List[str]:
    out_base = Path(output_dir).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    logs: List[str] = []

    tasks: List[Tuple[str, Any, Preset]] = []
    if uploaded_files:
        for name, data in uploaded_files:
            for preset in presets:
                tasks.append(("uploaded", (name, data), preset))
    else:
        if not input_dir:
            return ["[ERROR] input_dir required when uploaded_files is not provided"]
        inp = Path(input_dir).expanduser().resolve()
        if not inp.exists() or not inp.is_dir():
            return [f"[ERROR] input folder not found: {inp}"]
        files = get_image_files(inp)
        if selected:
            files = [p for p in files if p.name in set(selected)]
        for p in files:
            for preset in presets:
                tasks.append(("disk", p, preset))

    if not tasks:
        return ["[WARN] no tasks to process"]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = max(1, min(max_workers, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for kind, payload, preset in tasks:
            if kind == "uploaded":
                name, data = payload
                futures.append(ex.submit(process_one_image_bytes, name, data, preset, out_base))
            else:
                path: Path = payload
                futures.append(ex.submit(process_one_image_disk, path, preset, out_base))
        for fut in as_completed(futures):
            try:
                res = fut.result()
                logs.append(res)
                logger.info(res)
            except Exception as e:
                logger.exception("worker failed")
                logs.append(f"❌ worker exception: {e}")
    return logs

# --- CLI / Helpers ---
def load_presets_from_json(path: Path) -> List[Preset]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # single preset object or a dict of presets
        # allow {"presets":[...]} or single
        if "presets" in data and isinstance(data["presets"], list):
            arr = data["presets"]
        else:
            arr = [data]
    elif isinstance(data, list):
        arr = data
    else:
        raise ValueError("Invalid presets JSON")
    presets = []
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        label = item.get("label") or f"preset_{i+1}"
        presets.append(Preset.from_dict(item, label=label))
    return presets

def cli_main(argv=None):
    parser = argparse.ArgumentParser(description="Photo Processor Pro")
    parser.add_argument("--input", "-i", default="./input", help="Input folder with images")
    parser.add_argument("--output", "-o", default="./output", help="Output folder")
    parser.add_argument("--presets", help="JSON file with list of presets (overrides CLI flags)")
    # single preset flags (used if --presets not provided)
    parser.add_argument("--label", default="default", help="Label for this preset")
    parser.add_argument("--no-bg", dest="remove_bg", action="store_false", help="Disable background removal")
    parser.add_argument("--remove-wm", dest="remove_wm", action="store_true", help="Enable watermark removal")
    parser.add_argument("--wm-threshold", type=int, default=220)
    parser.add_argument("--wm-radius", type=int, default=5)
    parser.add_argument("--fmt", choices=["PNG", "JPEG"], default="PNG")
    parser.add_argument("--jpeg-q", type=int, default=95)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--files", default=None, help="Comma-separated list of filenames in input to process (optional)")
    args = parser.parse_args(argv)

    if args.presets:
        presets = load_presets_from_json(Path(args.presets))
    else:
        presets = [Preset(
            label=args.label,
            remove_bg=args.remove_bg,
            remove_wm=args.remove_wm,
            wm_threshold=args.wm_threshold,
            wm_radius=args.wm_radius,
            fmt=args.fmt,
            jpeg_q=args.jpeg_q,
            target_width=args.width,
            target_height=args.height,
        )]

    selected = [s.strip() for s in args.files.split(",")] if args.files else None
    logs = process_batch(args.input, args.output, presets, selected=selected, uploaded_files=None)
    for l in logs:
        print(l)
    # create zip
    try:
        zip_path = create_zip_of_output(args.output)
        print(f"ZIP created: {zip_path}")
    except Exception as e:
        print(f"Could not create ZIP: {e}", file=sys.stderr)

# --- Streamlit UI (simple) ---
def run_streamlit_app():
    if not HAS_STREAMLIT:
        raise SystemExit("Streamlit not installed. `pip install streamlit` to use UI.")

    st.set_page_config(page_title="Photo Processor Pro", layout="wide")
    st.title("Photo Processor Pro")

    st.sidebar.header("Global")
    output_dir = st.sidebar.text_input("Output folder", value="./streamlit_output")
    use_uploaded = st.sidebar.checkbox("Upload files (instead of folder)", value=True)

    st.sidebar.markdown("Presets (you can add multiple)")
    presets_data = st.sidebar.text_area("Presets JSON (list of preset objects) or leave blank for single preset",
                                       value=json.dumps([{
                                           "label": "preview",
                                           "remove_bg": True,
                                           "remove_wm": False,
                                           "wm_threshold": 220,
                                           "wm_radius": 5,
                                           "fmt": "PNG",
                                           "jpeg_q": 95,
                                           "target_width": None,
                                           "target_height": None
                                       }], ensure_ascii=False, indent=2))

    try:
        presets = load_presets_from_json(Path(tempfile.gettempdir()) / "tmp_presets.json") if False else []
        # parse text area
        parsed = json.loads(presets_data)
        if isinstance(parsed, dict) and "presets" in parsed:
            arr = parsed["presets"]
        elif isinstance(parsed, list):
            arr = parsed
        else:
            arr = [parsed]
        presets = [Preset.from_dict(item, label=(item.get("label") if isinstance(item, dict) else None)) for item in arr]
    except Exception:
        st.sidebar.error("Invalid presets JSON. Using default single preset.")
        presets = [Preset()]

    st.sidebar.markdown("Preview / Run")
    if use_uploaded:
        uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS])
    else:
        input_dir = st.sidebar.text_input("Input folder", value="./input")
        uploaded_files = None

    if st.button("Run processing"):
        st.info("Starting processing...")
        if use_uploaded and uploaded_files:
            up_list = []
            for uf in uploaded_files:
                try:
                    up_list.append((uf.name, uf.read()))
                except Exception:
                    st.warning(f"Could not read {uf.name}")
            logs = process_batch(None, output_dir, presets, selected=None, uploaded_files=up_list)
        else:
            logs = process_batch(input_dir, output_dir, presets, selected=None, uploaded_files=None)
        for l in logs:
            if l.startswith("✅"):
                st.success(l)
            elif l.startswith("❌"):
                st.error(l)
            else:
                st.info(l)
        try:
            zip_path = create_zip_of_output(output_dir)
            with open(zip_path, "rb") as f:
                st.download_button("Download ZIP", data=f, file_name=zip_path.name, mime="application/zip")
            st.info(f"ZIP created: {zip_path}")
        except Exception as e:
            st.error(f"Could not create ZIP: {e}")

# --- Entrypoint ---
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit_app()
    else:
        cli_main(sys.argv[1:])

if __name__ == "__main__":
    main()
