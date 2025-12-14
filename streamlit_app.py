"""
Photo Processor Pro — русская версия с поддержкой готового фона и надписей.
Добавлены CLI-опции:
  --bg-image PATH       : путь к изображению фонa (cover)
  --bg-color COLOR      : цвет фона, если нет bg-image (напр. "#ffffff" или "white")
  --caption TEXT        : текст для добавления
  --caption-font PATH   : путь к .ttf шрифту (если не указан, используется встроенный)
  --caption-size N      : размер шрифта (px)
  --caption-color COL   : цвет текста
  --caption-pos POS     : позиция текста: top/center/bottom or "x,y" в пикселях
  --caption-outline W   : ширина обводки текста (0 = нет)
Все сообщения на русском. Скрипт автоматически создаёт папки и README.txt.
Требования: Pillow, numpy, opencv-python. tqdm опционально.
"""

from pathlib import Path
from datetime import datetime
import logging, os, argparse, traceback, sys, csv, html, time
from io import BytesIO
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2, numpy as np
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont, ImageColor

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from rembg import remove as rembg_remove  # type: ignore
    REMBG_AVAILABLE = True
except Exception:
    rembg_remove = None
    REMBG_AVAILABLE = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------- utils -----------------------------------------------------
def _write_marker_file(folder: Path, text: Optional[str] = None) -> None:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        t = text or f"Photo Processor Pro\nПапка создана: {datetime.now().isoformat(sep=' ', timespec='seconds')}\n"
        (folder / "README.txt").write_text(t, encoding="utf-8")
    except Exception:
        pass


def setup_logger(out_dir: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("photo_processor_pro")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if out_dir:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            logfile = out_dir / f"ppp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            fh = logging.FileHandler(logfile, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
        except Exception as e:
            logger.warning("Не удалось создать файл лога в %s: %s. Используется консоль.", out_dir, e)
    return logger


def list_images_in_dirs(dirs: List[Path], recursive: bool = False) -> List[Path]:
    out = []
    for d in dirs:
        if not d.exists() or not d.is_dir():
            continue
        it = d.rglob("*") if recursive else d.iterdir()
        for p in it:
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                out.append(p)
    return sorted(out, key=lambda p: p.as_posix())


def compute_output_base(original: Path, out_root: Path, mode: str, input_roots: List[Path], suffix: str = "_proc") -> Path:
    if mode == "inplace":
        return original.parent / f"{original.stem}{suffix}"
    if mode == "mirror":
        try:
            p_res = original.resolve()
            roots = sorted((r.resolve() for r in input_roots), key=lambda r: len(str(r)), reverse=True)
            for r in roots:
                try:
                    if hasattr(p_res, "is_relative_to") and p_res.is_relative_to(r):
                        rel = original.relative_to(r)
                        return out_root / rel.parent / original.stem
                    else:
                        original.relative_to(r)
                        rel = original.relative_to(r)
                        return out_root / rel.parent / original.stem
                except Exception:
                    continue
        except Exception:
            pass
        return out_root / original.stem
    return out_root / original.stem


# ---------------- image helpers ---------------------------------------------
def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil)
    if img_pil.mode == "RGBA":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    else:
        rgb = img_pil.convert("RGB"); return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    if img_cv.ndim == 2:
        return Image.fromarray(img_cv)
    if img_cv.shape[2] == 4:
        arr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(arr)
    arr = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB); return Image.fromarray(arr)


def remove_background_pil(img_pil: Image.Image) -> Image.Image:
    if REMBG_AVAILABLE and rembg_remove is not None:
        try:
            out = rembg_remove(img_pil)
            if isinstance(out, Image.Image):
                return out.convert("RGBA")
            try:
                return Image.open(BytesIO(out)).convert("RGBA")
            except Exception:
                return img_pil.convert("RGBA")
        except Exception:
            pass
    rgb = img_pil.convert("RGB"); arr = np.array(rgb)
    thr = 240; bg_mask = np.all(arr > thr, axis=2)
    alpha = (~bg_mask).astype(np.uint8) * 255
    rgba = np.dstack([arr, alpha]); return Image.fromarray(rgba, "RGBA")


def remove_watermark_cv(img_cv: np.ndarray, threshold: int, radius: int) -> np.ndarray:
    if img_cv is None: return img_cv
    if img_cv.ndim == 2:
        gray = img_cv
    else:
        bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR) if img_cv.shape[2] == 4 else img_cv
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    if np.any(mask):
        to_inpaint = img_cv
        converted = False
        if img_cv.ndim == 3 and img_cv.shape[2] == 4:
            to_inpaint = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR); converted = True
        inpainted = cv2.inpaint(to_inpaint, mask, radius=radius, flags=cv2.INPAINT_TELEA)
        if converted:
            alpha = img_cv[:, :, 3]; inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA); inpainted[:, :, 3] = alpha
        return inpainted
    return img_cv


def save_cv_image(img_cv: np.ndarray, out_base: Path, fmt: str, jpeg_q: int = 95) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    if fmt.startswith("PNG") and img_cv.ndim == 3 and img_cv.shape[2] == 4:
        out_path = out_base.with_suffix(".png"); cv2.imwrite(str(out_path), img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 9]); return out_path
    if img_cv.ndim == 3 and img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
    out_path = out_base.with_suffix(".jpg")
    ok, buf = cv2.imencode(".jpg", img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
    if not ok:
        raise IOError("cv2.imencode не удался")
    out_path.write_bytes(buf.tobytes()); return out_path


# ---------------- background & caption -------------------------------------
def _parse_color(col: str):
    try:
        return ImageColor.getrgb(col)
    except Exception:
        # fallback: try hex cleanup
        col = col.strip()
        if not col:
            return (255,255,255)
        if col.startswith("#") and len(col) in (4,7):
            return ImageColor.getrgb(col)
        return (255,255,255)


def _load_font(path: Optional[str], size: int):
    try:
        if path and Path(path).exists():
            return ImageFont.truetype(str(path), size=size)
    except Exception:
        pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _cover_resize(img: Image.Image, target_size: Tuple[int,int]) -> Image.Image:
    # scale image to cover target size (center-crop style)
    w, h = img.size; tw, th = target_size
    if w == 0 or h == 0:
        return Image.new("RGBA", target_size, (255,255,255,255))
    scale = max(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img2 = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2; top = (nh - th) // 2
    return img2.crop((left, top, left+tw, top+th))


def apply_background_and_caption(img_pil: Image.Image,
                                 bg_image_path: Optional[Path],
                                 bg_color: Optional[str],
                                 caption_text: Optional[str],
                                 caption_font: Optional[str],
                                 caption_size: int,
                                 caption_color: str,
                                 caption_pos: str,
                                 caption_outline: int,
                                 logger: logging.Logger) -> Image.Image:
    """
    img_pil: processed image (PIL), can be RGBA (with transparency) or RGB.
    Returns PIL image composited on background and with caption.
    """
    try:
        # ensure RGBA for compositing
        src = img_pil.convert("RGBA")
        tw, th = src.size
        # create background
        if bg_image_path and Path(bg_image_path).exists():
            try:
                bg = Image.open(bg_image_path).convert("RGBA")
                bg = _cover_resize(bg, (tw, th))
            except Exception as e:
                logger.warning("Не удалось открыть фон %s: %s. Использую цветной фон.", bg_image_path, e)
                bg = Image.new("RGBA", (tw, th), _parse_color(bg_color or "#ffffff") + (255,))
        else:
            bg = Image.new("RGBA", (tw, th), _parse_color(bg_color or "#ffffff") + (255,))
        # composite: paste src centered on bg if src smaller, but usually same size
        bg.paste(src, ( (tw - src.width)//2, (th - src.height)//2 ), mask=src)
        # add caption if requested
        if caption_text:
            draw = ImageDraw.Draw(bg)
            font = _load_font(caption_font, caption_size or 36) or ImageFont.load_default()
            text = str(caption_text)
            # outline
            fill = _parse_color(caption_color or "#000000")
            outline_w = int(max(0, caption_outline or 0))
            # calculate position
            w, h = draw.textsize(text, font=font)
            pos = (0,0)
            pos_arg = caption_pos or "bottom"
            if "," in pos_arg:
                try:
                    x_str, y_str = pos_arg.split(",",1)
                    pos = (int(x_str.strip()), int(y_str.strip()))
                except Exception:
                    pos = ((tw - w)//2, th - h - 10)
            else:
                p = pos_arg.lower()
                if p in ("top","t"):
                    pos = ((tw - w)//2, 10)
                elif p in ("center","c","middle","m"):
                    pos = ((tw - w)//2, (th - h)//2)
                else:  # bottom
                    pos = ((tw - w)//2, th - h - 10)
            # draw outline by drawing text multiple times
            if outline_w > 0:
                outline_color = (0,0,0)
                for dx in range(-outline_w, outline_w+1):
                    for dy in range(-outline_w, outline_w+1):
                        if dx == 0 and dy == 0:
                            continue
                        draw.text((pos[0]+dx, pos[1]+dy), text, font=font, fill=outline_color)
            draw.text(pos, text, font=font, fill=fill)
        return bg.convert("RGBA")
    except Exception as e:
        logger.error("Ошибка при добавлении фона/надписи: %s", e)
        return img_pil.convert("RGBA")


def create_side_by_side_thumbnail(original_p: Path, processed_cv: Optional[np.ndarray], thumb_path: Path, size: Tuple[int,int]=(320,240)) -> Path:
    try:
        with Image.open(original_p) as im_orig:
            orig_thumb = im_orig.copy(); orig_thumb.thumbnail(size)
            if processed_cv is not None:
                if processed_cv.ndim == 2:
                    pil_proc = Image.fromarray(processed_cv)
                else:
                    if processed_cv.shape[2] == 4:
                        arr = cv2.cvtColor(processed_cv, cv2.COLOR_BGRA2RGBA)
                    else:
                        arr = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)
                    pil_proc = Image.fromarray(arr)
                pil_proc.thumbnail(size)
                w = orig_thumb.width + pil_proc.width; h = max(orig_thumb.height, pil_proc.height)
                canvas = Image.new("RGBA", (w, h), (255,255,255,255))
                canvas.paste(orig_thumb.convert("RGBA"), (0,0)); canvas.paste(pil_proc.convert("RGBA"), (orig_thumb.width,0))
            else:
                canvas = Image.new("RGBA", (orig_thumb.width*2, orig_thumb.height), (255,255,255,255)); canvas.paste(orig_thumb.convert("RGBA"), (0,0))
            thumb_path.parent.mkdir(parents=True, exist_ok=True); canvas.save(thumb_path, format="PNG"); return thumb_path
    except Exception:
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", size, (240,240,240,255)).save(thumb_path, format="PNG"); return thumb_path


# ---------------- processing task ------------------------------------------
def process_single(p: Path, out_root: Path, mode: str, input_roots: List[Path], fmt: str, jpeg_q: int,
                   remove_bg: bool, remove_wm: bool, wm_threshold: int, wm_radius: int,
                   suffix: str, dry_run: bool, retries: int, preview_dir: Path,
                   bg_image: Optional[str], bg_color: Optional[str],
                   caption_text: Optional[str], caption_font: Optional[str], caption_size: int,
                   caption_color: str, caption_pos: str, caption_outline: int,
                   logger: logging.Logger) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"src": str(p), "name": p.name, "status": "skipped", "out": None, "thumb": None, "error": None, "time": 0.0, "attempts": 0}
    base_out = compute_output_base(p, out_root, mode, input_roots, suffix); rec["out_base"] = str(base_out)
    if dry_run:
        rec["status"] = "dry-run"; return rec
    t0 = time.time(); attempt = 0
    while attempt <= retries:
        attempt += 1; rec["attempts"] = attempt
        try:
            with Image.open(p) as pil:
                # remove background (PIL)
                if remove_bg:
                    pil = remove_background_pil(pil)  # RGBA
                # convert to cv for watermark removal
                img_cv = pil_to_cv(pil)
                if remove_wm:
                    img_cv = remove_watermark_cv(img_cv, wm_threshold, wm_radius)
                # convert back to PIL for background/caption compositing
                pil2 = cv_to_pil(img_cv).convert("RGBA")
                # apply background & caption (if any)
                if bg_image or bg_color or caption_text:
                    pil2 = apply_background_and_caption(pil2, Path(bg_image) if bg_image else None, bg_color, caption_text, caption_font, caption_size, caption_color, caption_pos, caption_outline, logger)
                # convert back to cv for saving
                final_cv = pil_to_cv(pil2)
                out_file = save_cv_image(final_cv, base_out, fmt, jpeg_q)
                # thumbnail
                try:
                    rel = base_out.relative_to(out_root); rel_thumb = Path("_previews") / rel.with_suffix(".png")
                except Exception:
                    rel_thumb = Path("_previews") / (p.parent.name + "_" + p.stem + ".png")
                thumb_path = preview_dir / rel_thumb
                create_side_by_side_thumbnail(p, final_cv, thumb_path)
                rec.update({"status": "ok", "out": str(out_file), "thumb": str(thumb_path)})
                break
        except Exception as e:
            logger.debug("Ошибка обработки %s: попытка %d -> %s", p, attempt, e)
            rec["error"] = f"{type(e).__name__}: {e}"
            if attempt > retries:
                rec["status"] = "error"; break
            time.sleep(0.5 * attempt)
    rec["time"] = round(time.time() - t0, 3); return rec


# ---------------- reporting -------------------------------------------------
def save_csv_report(out_root: Path, records: List[Dict[str,Any]]) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "report.csv"
    fields = ["src","name","status","out","out_base","thumb","error","time","attempts"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records: w.writerow({k: r.get(k,"") for k in fields})
    return csv_path


def save_html_report(out_root: Path, records: List[Dict[str,Any]], title: str="Отчёт Photo Processor Pro") -> Path:
    out_root.mkdir(parents=True, exist_ok=True); html_path = out_root / "report.html"; now = datetime.now().isoformat(sep=" ", timespec="seconds")
    rows = []
    for r in records:
        thumb = r.get("thumb"); thumb_rel = os.path.relpath(thumb, out_root) if thumb else ""
        outf = r.get("out") or ""; outf_rel = os.path.relpath(outf, out_root) if outf else ""
        err = html.escape(str(r.get("error") or ""))
        rows.append(f"<tr><td>{html.escape(r.get('name',''))}</td><td>{r.get('status')}</td><td>{r.get('attempts')}</td><td>{r.get('time')}</td><td>{err}</td><td>{f'<a href=\"{outf_rel}\">файл</a>' if outf_rel else ''}</td><td>{f'<a href=\"{thumb_rel}\"><img src=\"{thumb_rel}\" style=\"max-width:240px\"></a>' if thumb_rel else ''}</td></tr>")
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>body{{font-family:Arial}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px}}</style></head><body><h1>{html.escape(title)}</h1><p>Сгенерировано: {now}</p><table><tr><th>имя</th><th>статус</th><th>попытки</th><th>время(с)</th><th>ошибка</th><th>файл</th><th>превью</th></tr>{''.join(rows)}</table></body></html>"
    html_path.write_text(html_doc, encoding="utf-8"); return html_path


# ---------------- main CLI --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Photo Processor Pro — фон и надписи")
    parser.add_argument("-i","--input", nargs="+", default=[str(Path.cwd())], help="входные папки")
    parser.add_argument("-o","--output", default="./output", help="выходная папка")
    parser.add_argument("-r","--recursive", action="store_true", help="искать рекурсивно")
    parser.add_argument("--no-bg", dest="remove_bg", action="store_false", help="не удалять фон")
    parser.add_argument("--wm", dest="remove_wm", action="store_true", help="удалять водяные знаки")
    parser.add_argument("--wm-th", type=int, default=220, help="порог для водяных знаков")
    parser.add_argument("--wm-r", type=int, default=5, help="радиус инпейнта")
    parser.add_argument("--fmt", choices=["PNG","JPEG"], default="PNG", help="формат вывода")
    parser.add_argument("--q", type=int, default=95, help="качество JPEG")
    parser.add_argument("--save-mode", choices=["out","inplace","mirror"], default="out", help="куда сохранять")
    parser.add_argument("--suffix", default="_proc", help="суффикс для inplace")
    parser.add_argument("--workers", type=int, default=4, help="рабочие (0/1 = последовательная)")
    parser.add_argument("--dry-run", action="store_true", help="показать куда сохранится")
    parser.add_argument("--retries", type=int, default=1, help="повторы при ошибке")
    parser.add_argument("--report", action="store_true", help="создать отчёт CSV+HTML")
    # new options
    parser.add_argument("--bg-image", default=None, help="путь к картинке фона (cover)")
    parser.add_argument("--bg-color", default="#ffffff", help="цвет фона, если нет bg-image")
    parser.add_argument("--caption", default=None, help="текст надписи (если нужен)")
    parser.add_argument("--caption-font", default=None, help="путь к .ttf шрифту")
    parser.add_argument("--caption-size", type=int, default=36, help="размер шрифта (px)")
    parser.add_argument("--caption-color", default="#000000", help="цвет текста")
    parser.add_argument("--caption-pos", default="bottom", help="позиция: top/center/bottom или 'x,y'")
    parser.add_argument("--caption-outline", type=int, default=1, help="обводка текста (px), 0 = нет")
    args = parser.parse_args()

    input_roots = [Path(p) for p in args.input]; images = list_images_in_dirs(input_roots, recursive=args.recursive)
    out_root = Path(args.output)
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        _write_marker_file(out_root, text=f"Photo Processor Pro\nПапка вывода: {out_root}\nСоздана: {datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
    except Exception:
        pass
    logger = setup_logger(out_root if not args.dry_run else None)

    if not images:
        logger.info("Изображений не найдено в папках: %s", ", ".join(str(p) for p in input_roots))
        try:
            (out_root / "_previews").mkdir(parents=True, exist_ok=True); _write_marker_file(out_root / "_previews", text="Превью обработанных файлов.")
        except Exception:
            pass
        if args.report:
            csv_p = out_root / "report.csv"
            if not csv_p.exists(): csv_p.write_text("src,name,status,out,out_base,thumb,error,time,attempts\n", encoding="utf-8")
            html_p = out_root / "report.html"
            if not html_p.exists(): html_p.write_text("<html><body><p>Нет обработанных файлов.</p></body></html>", encoding="utf-8")
        return

    logger.info("Найдено %d файлов. REMBG доступен: %s", len(images), REMBG_AVAILABLE)
    fmt_str = "PNG (с альфа)" if args.fmt == "PNG" else "JPEG (без альфа)"
    if args.dry_run:
        logger.info("DRY RUN: первые 20 соответствий")
        for p in images[:20]:
            base = compute_output_base(p, out_root, args.save_mode, input_roots, args.suffix)
            logger.info("%s -> base=%s (.png/.jpg)", p, base)
        return

    preview_dir = out_root; records = []; total = len(images); use_tqdm = tqdm is not None
    workers = args.workers if args.workers is not None else 0
    if workers <= 1:
        iterable = images
        if use_tqdm: iterable = tqdm(images, desc="Обработка", unit="файл")
        for p in iterable:
            rec = process_single(p, out_root, args.save_mode, input_roots, fmt_str, args.q,
                                 args.remove_bg, args.remove_wm, args.wm_th, args.wm_r,
                                 args.suffix, False, args.retries, preview_dir,
                                 args.bg_image, args.bg_color, args.caption, args.caption_font, args.caption_size,
                                 args.caption_color, args.caption_pos, args.caption_outline, logger)
            records.append(rec)
    else:
        logger.info("Запуск с %d рабочими потоками...", workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_single, p, out_root, args.save_mode, input_roots, fmt_str, args.q,
                                 args.remove_bg, args.remove_wm, args.wm_th, args.wm_r,
                                 args.suffix, False, args.retries, preview_dir,
                                 args.bg_image, args.bg_color, args.caption, args.caption_font, args.caption_size,
                                 args.caption_color, args.caption_pos, args.caption_outline, logger): p for p in images}
            if use_tqdm:
                with tqdm(total=total, desc="Обработка", unit="файл") as pbar:
                    for fut in as_completed(futures):
                        try: rec = fut.result()
                        except Exception:
                            rec = {"src": str(futures.get(fut,"unknown")), "name": "unknown", "status":"error", "error": traceback.format_exc(), "time":0, "attempts":0}
                            logger.error("Ошибка в воркере: %s", rec["error"])
                        records.append(rec); pbar.update(1)
            else:
                for fut in as_completed(futures):
                    try: rec = fut.result()
                    except Exception:
                        rec = {"src": str(futures.get(fut,"unknown")), "name": "unknown", "status":"error", "error": traceback.format_exc(), "time":0, "attempts":0}
                        logger.error("Ошибка в воркере: %s", rec["error"])
                    records.append(rec)

    try: (out_root / "_previews").mkdir(parents=True, exist_ok=True); _write_marker_file(out_root / "_previews", text="Превью обработанных файлов.")
    except Exception: pass
    csv_p = save_csv_report(out_root, records); logger.info("CSV отчёт: %s", csv_p)
    if args.report:
        html_p = save_html_report(out_root, records, title="Отчёт Photo Processor Pro"); logger.info("HTML отчёт: %s", html_p)
    ok = sum(1 for r in records if r.get("status")=="ok"); err = sum(1 for r in records if r.get("status")=="error")
    logger.info("Готово: всего=%d успешно=%d ошибки=%d. Выход: %s", len(records), ok, err, out_root)


if __name__ == "__main__":
    main()
