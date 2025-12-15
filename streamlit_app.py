wrapper_app.py

"Обернуть" Streamlit-приложение: если streamlit установлен — автоматически запустить
его через `streamlit run ... -- --as-streamlit`; если нет — запустить fallback на Tkinter.

Запуск:
- С Python: python wrapper_app.py
  (если streamlit установлен, откроется Streamlit сервер)
- Или явно: streamlit run wrapper_app.py -- --as-streamlit

Требования (fallback): Pillow; для лучших результатов — OpenCV (cv2).
"""

import sys
import os
import subprocess

# --- Streamlit app definition (keeps UI code here) ---
def streamlit_app():
    import streamlit as st
    import numpy as np
    import cv2
    from PIL import Image

    st.set_page_config(page_title="Image Watermark Remover", layout="wide")
    st.title("Image Watermark Remover")

    def make_sample():
        h, w = 400, 700
        img = np.full((h, w, 3), 230, dtype=np.uint8)
        cv2.putText(img, "SAMPLE IMAGE", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (80, 80, 200), 4, cv2.LINE_AA)
        cv2.putText(img, "WATERMARK", (290, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
        return img

    uploaded = st.file_uploader("Upload an image (PNG/JPG). If none uploaded, a sample will be used.", type=["png", "jpg", "jpeg"])
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Couldn't read the uploaded image. Please try another file.")
            st.stop()
    else:
        img = make_sample()

    st.sidebar.header("Removal settings")
    method = st.sidebar.selectbox("Method", ("Inpaint (auto mask)", "Binary Threshold (preview only)"))
    thresh = st.sidebar.slider("Threshold for mask", 0, 255, 150)
    kernel_size = st.sidebar.slider("Morph kernel (to clean mask)", 1, 25, 5)
    invert_mask = st.sidebar.checkbox("Invert mask (useful for dark watermarks)", False)
    apply_btn = st.sidebar.button("Apply removal")

    def to_pil(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def make_mask(gray, thresh_val, invert=False, k=5):
        _, m = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        if invert:
            m = cv2.bitwise_not(m)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        return m

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(to_pil(img), use_column_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_preview = make_mask(gray, thresh, invert=invert_mask, k=kernel_size)
    with col2:
        st.subheader("Mask Preview")
        st.image(Image.fromarray(mask_preview), caption="White = detected watermark area", use_column_width=True)

    result = None
    if apply_btn:
        if method == "Binary Threshold (preview only)":
            _, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            result = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
        else:
            mask = make_mask(gray, thresh, invert=invert_mask, k=kernel_size)
            inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            result = inpainted

        st.subheader("Result")
        st.image(to_pil(result), use_column_width=True)

        buf = cv2.imencode(".png", result)[1].tobytes()
        st.download_button("Download result (PNG)", data=buf, file_name="result.png", mime="image/png")
    else:
        st.info("Adjust settings in the sidebar and click 'Apply removal' to process the image.")


# --- Tkinter fallback app definition ---
def tkinter_fallback():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        from PIL import Image, ImageTk, ImageOps
        import cv2
        import numpy as np
    except Exception as e:
        print("Fallback GUI requires tkinter, Pillow and (recommended) OpenCV (cv2).")
        print("Missing or failed import:", e)
        return

    root = tk.Tk()
    root.title("Image Watermark Remover (Fallback)")
    root.geometry("1000x700")

    lbl = tk.Label(root, text="Image Watermark Remover", font=("Arial", 30), fg="magenta")
    lbl.pack(pady=10)

    canvas = tk.Label(root)
    canvas.pack()

    info = tk.Label(root, text="Select image to remove watermark (simple threshold/inpaint).", font=("Arial", 12))
    info.pack(pady=5)

    state = {"img_cv": None, "mask": None, "result": None}

    def show_pil(pil_img):
        pil_img = ImageOps.contain(pil_img, (800, 450))
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.configure(image=tk_img)
        canvas.image = tk_img

    def open_image():
        path = filedialog.askopenfilename(title="Select file", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Couldn't read image.")
            return
        state["img_cv"] = img
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        show_pil(pil)

    def preview_mask():
        if state["img_cv"] is None:
            messagebox.showinfo("Info", "Open an image first.")
            return
        gray = cv2.cvtColor(state["img_cv"], cv2.COLOR_BGR2GRAY)
        t = int(slider_thresh.get())
        _, m = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        k = int(slider_kernel.get())
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        state["mask"] = m
        pil = Image.fromarray(m)
        show_pil(pil.convert("RGB"))

    def apply_remove():
        if state["img_cv"] is None:
            messagebox.showinfo("Info", "Open an image first.")
            return
        if state["mask"] is None:
            preview_mask()
        img = state["img_cv"]
        m = state["mask"]
        res = cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)
        state["result"] = res
        pil = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        show_pil(pil)

    btn_open = tk.Button(root, text="Open Image", command=open_image, bg="lightgreen")
    btn_open.pack(side="left", padx=10, pady=10)

    slider_thresh = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Threshold", length=300)
    slider_thresh.set(150)
    slider_thresh.pack(side="left", padx=10)

    slider_kernel = tk.Scale(root, from_=1, to=25, orient="horizontal", label="Kernel", length=300)
    slider_kernel.set(5)
    slider_kernel.pack(side="left", padx=10)

    btn_preview = tk.Button(root, text="Preview Mask", command=preview_mask, bg="lightblue")
    btn_preview.pack(side="left", padx=10)

    btn_apply = tk.Button(root, text="Apply Remove", command=apply_remove, bg="orange")
    btn_apply.pack(side="left", padx=10)

    root.mainloop()


# --- Main launcher: decide which UI to run ---
def main():
    try:
        import streamlit  # check availability only
        streamlit_available = True
    except Exception:
        streamlit_available = False

    if streamlit_available:
        # If launched with the marker arg, run the streamlit app inline (this happens when streamlit runs the file)
        if "--as-streamlit" in sys.argv:
            streamlit_app()
        else:
            # spawn `streamlit run thisfile -- --as-streamlit` to avoid recursive spawning
            cmd = ["streamlit", "run", os.path.abspath(__file__), "--", "--as-streamlit"]
            print("Launching Streamlit app (if streamlit is installed):")
            print(" ".join(cmd))
            try:
                subprocess.run(cmd)
            except FileNotFoundError:
                print("Error: 'streamlit' executable not found even though streamlit package imported.")
    else:
        print("Streamlit not found. Launching Tkinter fallback GUI.")
        tkinter_fallback()


if __name__ == "__main__":
    main()
