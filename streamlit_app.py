# !/usr/bin/env python3
"""
Self-contained version of your WatermarkRemover GUI logic with:
- Streamlit UI used only if streamlit is installed
- Otherwise a headless test that:
    * creates a dummy remwm.py (simulates processing + prints progress)
    * creates a sample input file in uploads/
    * runs Api.start_processing to verify processing flow
    * serves the output directory over HTTP so you can download results
This adds "download after processing" behavior (via Streamlit when available,
or via simple HTTP server when not).
"""
import logging
import threading
import subprocess
import sys
import os
import json
import yaml
import time
import shutil
from pathlib import Path
import http.server
import socketserver

# Try to import streamlit lazily
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# Suppress noisy pywebview warnings if present
class PyWebviewFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if 'Error while processing window.native' in msg:
            return False
        if 'CoreWebView2 members can only be accessed' in msg:
            return False
        return True
logging.getLogger('pywebview').addFilter(PyWebviewFilter())

# Minimal dependencies done
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.yml")

class Api:
    """Lightweight Api similar to your original one."""
    def __init__(self):
        self.window = None
        self.process = None
        self.is_running = False
        self.config = self._load_config()

    def _load_config(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception:
            pass
        return {}

    def _save_config(self, config):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            print("Failed to save config:", e)

    def save_config(self, config):
        self.config = config
        self._save_config(config)

    # Basic conflict checks so code is self-contained
    def _would_overwrite_input(self, input_path, output_path):
        try:
            inp = os.path.abspath(input_path)
            out = os.path.abspath(output_path)
            if inp == out:
                return True
            if os.path.isfile(inp) and os.path.isdir(out):
                if os.path.abspath(os.path.join(out, os.path.basename(inp))) == inp:
                    return True
            if os.path.isfile(out) and inp == out:
                return True
        except Exception:
            return False
        return False

    def _check_file_conflicts(self, input_path, output_path):
        conflicts = []
        try:
            if os.path.isfile(input_path):
                out_dir = output_path if os.path.isdir(output_path) else os.path.dirname(output_path)
                target = os.path.join(out_dir, os.path.basename(input_path))
                if os.path.exists(target) and os.path.abspath(target) != os.path.abspath(input_path):
                    conflicts.append(os.path.basename(target))
            elif os.path.isdir(input_path):
                out_dir = output_path if os.path.isdir(output_path) else output_path
                if os.path.isdir(out_dir):
                    for root, _, files in os.walk(input_path):
                        for fn in files:
                            target = os.path.join(out_dir, fn)
                            if os.path.exists(target):
                                conflicts.append(fn)
                        break
        except Exception:
            pass
        return conflicts

    def start_processing(self, settings):
        """Start processing by launching remwm.py (or the dummy simulator)."""
        if self.is_running:
            return {'error': 'Already running'}
        input_path = settings.get('input', '')
        output_path = settings.get('output', '')

        if not input_path:
            return {'error': 'No input specified'}

        if not output_path:
            if os.path.isfile(input_path):
                output_path = os.path.dirname(input_path)
            else:
                output_path = input_path

        overwrite = settings.get('overwrite', False)
        if self._would_overwrite_input(input_path, output_path):
            return {'error': 'Would overwrite input'}

        conflicts = self._check_file_conflicts(input_path, output_path)
        if conflicts and not overwrite:
            return {'error': f'Conflicting files: {conflicts[:3]}'}

        # Save config (optional)
        self.save_config({
            'input': input_path,
            'output': output_path,
            'overwrite': overwrite
        })

        # Build command: use remwm.py in cwd
        cmd = [sys.executable, 'remwm.py', input_path, output_path]
        if overwrite:
            cmd.append('--overwrite')

        # Launch in background
        self.is_running = True
        threading.Thread(target=self._run_process, args=(cmd,), daemon=True).start()
        return {'status': 'started'}

    def _run_process(self, cmd):
        """Run cmd and stream output (keeps original behaviour but headless)."""
        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            working_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(working_dir, 'remwm.py')
            if not os.path.exists(script_path):
                print("ERROR: remwm.py not found")
                self.is_running = False
                return
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=working_dir
            )
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                line = line.rstrip()
                print("[remwm]", line)
            self.process.wait()
        except Exception as e:
            print("Processing error:", e)
        finally:
            self.is_running = False
            self.process = None

# Helper: create a dummy remwm.py simulator
DUMMY_SCRIPT = r'''#!/usr/bin/env python3
import sys, time, shutil, os, json
def main():
    args = sys.argv[1:]
    if not args:
        print("error: no args")
        sys.exit(1)
    inp = args[0]
    out = args[1] if len(args)>1 else "."
    out_is_dir = os.path.isdir(out) or out.endswith(os.sep) or not out.lower().endswith(('.png','.jpg','.jpeg','.webp','.bmp','.gif'))
    if out_is_dir:
        os.makedirs(out, exist_ok=True)
        base = os.path.basename(inp)
        target = os.path.join(out, base)
    else:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        target = out
    # simulate progress
    for p in [0, 20, 50, 80, 100]:
        print(f"overall_progress: {p}%")
        time.sleep(0.2)
    try:
        # try to copy if file exists; otherwise create dummy file
        if os.path.exists(inp):
            shutil.copyfile(inp, target)
        else:
            with open(target, "wb") as f:
                f.write(b"processed")
        print(f"saved: {target}")
        print("Done")
        sys.exit(0)
    except Exception as e:
        print("error:", e)
        sys.exit(2)

if __name__ == '__main__':
    main()
'''

def make_dummy_environment(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    # create remwm.py in base_dir
    remwm_path = os.path.join(base_dir, 'remwm.py')
    with open(remwm_path, 'w', encoding='utf-8') as f:
        f.write(DUMMY_SCRIPT)
    os.chmod(remwm_path, 0o755)

    uploads_dir = os.path.join(base_dir, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    out_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    # create a sample input file
    sample_in = os.path.join(uploads_dir, 'sample_input.png')
    with open(sample_in, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + b'PNGDATA')
    return remwm_path, sample_in, out_dir

def run_headless_test():
    print("Streamlit not available. Running headless test with dummy remwm.py ...")
    base = os.path.dirname(os.path.abspath(__file__))
    make_dummy_environment(base)
    remwm_path = os.path.join(base, 'remwm.py')
    uploads_dir = os.path.join(base, 'uploads')
    outputs_dir = os.path.join(base, 'outputs')
    sample_in = os.path.join(uploads_dir, 'sample_input.png')

    api = Api()
    settings = {
        'input': sample_in,
        'output': outputs_dir,
        'overwrite': True
    }
    res = api.start_processing(settings)
    if 'error' in res:
        print("Failed to start processing:", res)
        return

    # Wait for processing to end (with timeout)
    timeout = 30
    start = time.time()
    while api.is_running and (time.time() - start) < timeout:
        time.sleep(0.1)
    if api.is_running:
        print("Processing did not finish in time; terminating.")
    else:
        print("Processing finished. Listing output files:")
        for fn in sorted(os.listdir(outputs_dir)):
            fp = os.path.join(outputs_dir, fn)
            print(" -", fp)

    # Start HTTP server to serve outputs_dir for downloads
    os.chdir(outputs_dir)
    port = 8000
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    print(f"Serving '{outputs_dir}' at http://localhost:{port}/")
    print("Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopping HTTP server.")
    finally:
        httpd.server_close()

def start_streamlit_ui():
    # This function assumes streamlit is installed (STREAMLIT_AVAILABLE True)
    api = Api()
    st.title("WatermarkRemover AI - Настройки (Streamlit with upload/download)")

    uploaded = st.file_uploader("Загрузить файл для обработки", type=['png','jpg','jpeg','webp','bmp','gif'])
    saved_path = ""
    if uploaded is not None:
        base = os.path.dirname(os.path.abspath(__file__))
        uploads = os.path.join(base, 'uploads')
        os.makedirs(uploads, exist_ok=True)
        saved_path = os.path.join(uploads, uploaded.name)
        with open(saved_path, 'wb') as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved: {saved_path}")

    input_path = st.text_input("Путь к файлу или папке для обработки", value=saved_path)
    output_path = st.text_input("Путь для сохранения результата (leave empty to use input folder)")
    overwrite = st.checkbox("Перезаписать файлы", value=False)

    if st.button("Запустить обработку"):
        if not input_path:
            st.error("Укажите путь к файлу или загрузите файл.")
        else:
            settings = {
                'input': input_path,
                'output': output_path,
                'overwrite': overwrite
            }
            res = api.start_processing(settings)
            if 'error' in res:
                st.error(res['error'])
            else:
                st.success("Processing started.")
                # Wait & show simple spinner
                with st.spinner("Processing..."):
                    while api.is_running:
                        time.sleep(0.3)
                st.success("Processing finished.")
                # Offer download buttons
                out_dir = settings.get('output') or (os.path.dirname(settings.get('input')) if os.path.isfile(settings.get('input')) else settings.get('input'))
                files = []
                if os.path.isdir(out_dir):
                    for fn in sorted(os.listdir(out_dir)):
                        fp = os.path.join(out_dir, fn)
                        if os.path.isfile(fp):
                            files.append(fp)
                elif os.path.isfile(out_dir):
                    files.append(out_dir)
                if not files:
                    st.info("No output files to download.")
                else:
                    st.markdown("### Downloads")
                    for fp in files:
                        name = os.path.basename(fp)
                        with open(fp, 'rb') as fh:
                            data = fh.read()
                        st.download_button(f"Download {name}", data=data, file_name=name)

if __name__ == '__main__':
    if STREAMLIT_AVAILABLE:
        start_streamlit_ui()
    else:
        run_headless_test()
