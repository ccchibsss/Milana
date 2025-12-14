CLI + Streamlit example. Adds Streamlit UI block (кнопка "Начать обработку") that
creates ProcessingConfig, runs process_batch, shows results, provides ZIP download
and previews. CLI mode remains as in the original example.
"""
import sys
import shutil
import tempfile
import logging
from dataclasses import dataclass
from pathlib import Path
import io
import zipfile
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import Streamlit and PIL.Image; if unavailable, fall back to CLI mode.
try:
    import streamlit as st
    from PIL import Image
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False
    Image = None  # type: ignore

@dataclass
class ProcessingConfig:
    remove_bg: bool
    remove_wm: bool
    wm_adaptive: bool
    wm_block_size: int
    wm_c: int
    wm_min_area: int
    wm_max_area: int
    wm_radius: int
    wm_use_ns: bool
    fmt: str
    jpeg_q: int
    target_width: Optional[int]
    target_height: Optional[int]
    inp: Path
    outp: Path

def process_batch(inp_dir: Path, out_dir: Path, cfg: ProcessingConfig, max_workers: int = 1) -> List[Tuple[Path, bool, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Tuple[Path, bool, str]] = []
    for p in sorted(inp_dir.glob("*")):
        if p.is_dir():
            continue
        try:
            out_path = out_dir / f"{p.stem}.{cfg.fmt.lower()}"
            # simple imitation of processing: copy and append a marker
            with p.open("rb") as fin, out_path.open("wb") as fout:
                data = fin.read()
                fout.write(data)
                fout.write(b"\n")
                fout.write(f"Processed as {cfg.fmt}".encode("utf-8"))
            results.append((p, True, ""))
        except Exception as e:
            results.append((p, False, str(e)))
    return results

def zip_results_bytes(out_dir: Path, results: List[Tuple[Path, bool, str]], fmt: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p, ok, _ in results:
            if ok:
                candidate = out_dir / f"{p.stem}.{fmt.lower()}"
                if candidate.exists():
                    zf.write(candidate, arcname=candidate.name)
    buf.seek(0)
    return buf.read()

# CLI flow (unchanged behavior)
def run_cli(argv):
    # default parameters
    remove_bg = False
    remove_wm = False
    wm_adaptive = False
    wm_block_size = 3
    wm_c = 5
    wm_min_area = 10
    wm_max_area = 1000
    wm_radius = 2
    wm_use_ns = False
    fmt = "PNG"
    jpeg_q = 90
    tw = 0
    th = 0
    workers = 2

    temp_dir = Path(tempfile.mkdtemp(prefix="proc_demo_"))
    inp_dir = temp_dir / "in"
    out_dir = temp_dir / "out"
    inp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 6):
        (inp_dir / f"image_{i}.txt").write_text(f"dummy content {i}")

    cfg = ProcessingConfig(
        remove_bg=remove_bg,
        remove_wm=remove_wm,
        wm_adaptive=wm_adaptive,
        wm_block_size=wm_block_size,
        wm_c=wm_c,
        wm_min_area=wm_min_area,
        wm_max_area=wm_max_area,
        wm_radius=wm_radius,
        wm_use_ns=wm_use_ns,
        fmt=fmt,
        jpeg_q=jpeg_q,
        target_width=tw if tw > 0 else None,
        target_height=th if th > 0 else None,
        inp=inp_dir,
        outp=out_dir,
    )

    print("Начинаем обработку...")
    results = process_batch(inp_dir, out_dir, cfg, max_workers=workers)

    success_count = sum(1 for _, ok, _ in results if ok)
    fail_count = len(results) - success_count
    print(f"Готово: {success_count} успешно, {fail_count} ошибок")

    if success_count > 0:
        zip_data = zip_results_bytes(out_dir, results, cfg.fmt.lower())
        zip_path = temp_dir / "processed_images.zip"
        zip_path.write_bytes(zip_data)
        print(f"ZIP результатов сохранён: {zip_path}")
        print("\nПревью результатов (список файлов):")
        for i, (p, ok, _) in enumerate(results):
            if ok:
                img_path = out_dir / f"{p.stem}.{cfg.fmt.lower()}"
                print(f" - {img_path.name} (size {img_path.stat().st_size} bytes)")

    if fail_count > 0:
        print("\nОшибки обработки:")
        for p, ok, msg in results:
            if not ok:
                print(f" - {p.name}: {msg}")

    try:
        shutil.rmtree(temp_dir)
        print(f"\nВременная папка {temp_dir} удалена.")
    except Exception as e:
        logger.warning("Не удалось удалить временную папку: %s", e)

# Streamlit UI: includes the requested block with st.button("Начать обработку")
def run_streamlit():
    st.title("Пример обработки файлов")
    uploaded = st.file_uploader("Загрузите файлы для обработки", accept_multiple_files=True)
    fmt = st.selectbox("Формат вывода", ["PNG", "JPEG"], index=0)
    jpeg_q = st.number_input("JPEG качество", min_value=10, max_value=100, value=90)
    workers = st.number_input("Число воркеров", min_value=1, max_value=16, value=2)
    # watermark/bg options (example)
    remove_bg = st.checkbox("Удалять фон", value=False)
    remove_wm = st.checkbox("Удалять водяные знаки", value=False)
    wm_adaptive = st.checkbox("WM adaptive", value=False)
    wm_block_size = st.number_input("WM block size", min_value=1, value=3)
    wm_c = st.number_input("WM C", min_value=0, value=5)
    wm_min_area = st.number_input("WM min area", min_value=0, value=10)
    wm_max_area = st.number_input("WM max area", min_value=0, value=1000)
    wm_radius = st.number_input("WM radius", min_value=0, value=2)
    wm_use_ns = st.checkbox("WM use non-specified", value=False)
    tw = st.number_input("Target width (0 = keep)", min_value=0, value=0)
    th = st.number_input("Target height (0 = keep)", min_value=0, value=0)

    # prepare temporary folders in session_state so user can clean up later
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None

    if st.button("Начать обработку"):
        if not uploaded:
            st.warning("Загрузите хотя бы один файл.")
        else:
            # create temp dirs
            temp_dir = Path(tempfile.mkdtemp(prefix="st_proc_"))
            inp_dir = temp_dir / "in"
            out_dir = temp_dir / "out"
            inp_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            # save uploaded files
            for f in uploaded:
                target = inp_dir / f.name
                # f is a UploadedFile-like object: .getbuffer() or .read()
                target.write_bytes(f.read())
            # store temp_dir to session for later cleanup
            st.session_state.temp_dir = str(temp_dir)

            cfg = ProcessingConfig(
                remove_bg=remove_bg,
                remove_wm=remove_wm,
                wm_adaptive=wm_adaptive,
                wm_block_size=int(wm_block_size),
                wm_c=int(wm_c),
                wm_min_area=int(wm_min_area),
                wm_max_area=int(wm_max_area),
                wm_radius=int(wm_radius),
                wm_use_ns=wm_use_ns,
                fmt=fmt,
                jpeg_q=int(jpeg_q),
                target_width=tw if tw > 0 else None,
                target_height=th if th > 0 else None,
                inp=inp_dir,
                outp=out_dir,
            )

            with st.spinner("Обработка..."):
                results = process_batch(inp_dir, out_dir, cfg, max_workers=int(workers))

            success_count = sum(1 for _, ok, _ in results if ok)
            fail_count = len(results) - success_count
            st.success(f"Готово: {success_count} успешно, {fail_count} ошибок")

            if success_count > 0:
                zip_data = zip_results_bytes(out_dir, results, cfg.fmt.lower())
                st.download_button(
                    label="Скачать все результаты (ZIP)",
                    data=zip_data,
                    file_name="processed_images.zip",
                    mime="application/zip",
                )

                st.subheader("Превью результатов")
                cols = st.columns(3)
                for i, (p, ok, _) in enumerate(results):
                    if ok:
                        img_path = out_dir / f"{p.stem}.{cfg.fmt.lower()}"
                        if img_path.exists():
                            try:
                                if Image is None:
                                    st.write(f"{img_path.name} (PIL не установлен, превью недоступно)")
                                else:
                                    img = Image.open(img_path)
                                    with cols[i % 3]:
                                        st.image(img, caption=p.name, use_column_width=True)
                            except Exception as e:
                                st.warning(f"Не удалось показать {p.name}: {e}")

            if fail_count > 0:
                st.subheader("Ошибки обработки")
                for p, ok, msg in results:
                    if not ok:
                        st.error(f"{p.name}: {msg}")

    # Option to clean up temporary files created in this session
    if st.session_state.get("temp_dir"):
        if st.button("Удалить временные файлы"):
            td = Path(st.session_state.temp_dir)
            try:
                if td.exists():
                    shutil.rmtree(td)
                st.success(f"Удалена временная папка {td}")
            except Exception as e:
                st.warning(f"Не удалось удалить временную папку: {e}")
            st.session_state.temp_dir = None

def main():
    if HAS_STREAMLIT and len(sys.argv) <= 1:
        run_streamlit()
    else:
        run_cli(sys.argv[1:])

if __name__ == "__main__":
    main()
