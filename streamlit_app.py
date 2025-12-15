import streamlit as st
"""
WatermarkRemover-AI GUI - Ohio Edition
PyWebview frontend with brainrot HTML UI
Интеграция с Streamlit для управления
"""

import logging
# Suppress noisy pywebview WebView2 COM warnings (thread safety noise, doesn't
# affect functionality)
class PyWebviewFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if 'Error while processing window.native' in msg:
            return False
        if 'CoreWebView2 members can only be accessed' in msg:
            return False
        return True
logging.getLogger('pywebview').addFilter(PyWebviewFilter())

import webview
import threading
import subprocess
import sys
import os
import json
import yaml
import base64
from pathlib import Path
import time

# Only psutil for system info (lightweight)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Вариант запуска: True — через Streamlit, False — через Webview
USE_STREAMLIT = True

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.yml")

# ===========================
# API класс
# ===========================

class Api:
    """Python API, доступный из фронтенда"""
    def __init__(self):
        self.window = None
        self.process = None
        self.is_running = False
        self.config = self._load_config()

    def set_window(self, window):
        self.window = window

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
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
            print(f"Failed to save config: {e}")

    def get_config(self):
        return self.config

    def save_config(self, config):
        self.config = config
        self._save_config(config)

    # Простые реализации недостающих проверок (чтобы модуль был самодостаточным)
    def _would_overwrite_input(self, input_path, output_path):
        try:
            inp = os.path.abspath(input_path)
            out = os.path.abspath(output_path)
            # если точно одинаковые пути - перезапись
            if inp == out:
                return True
            # если input файл и output - директория, и запись пойдёт в тот же файл
            if os.path.isfile(inp) and os.path.isdir(out):
                if os.path.abspath(os.path.join(out, os.path.basename(inp))) == inp:
                    return True
            # если output указывает на файл и совпадает с input
            if os.path.isfile(out) and inp == out:
                return True
        except Exception:
            return False
        return False

    def _check_file_conflicts(self, input_path, output_path):
        """
        Простая проверка конфликтов: если input - файл, проверяем наличие файла с тем же именем в output dir.
        Если input - папка, для каждого файла проверяем наличие с тем же именем в output dir.
        Возвращаем список имён файлов, которые уже существуют в выходной папке и могли бы быть перезаписаны.
        """
        conflicts = []
        try:
            if os.path.isfile(input_path):
                out_dir = output_path if os.path.isdir(output_path) else os.path.dirname(output_path)
                target = os.path.join(out_dir, os.path.basename(input_path))
                if os.path.exists(target) and os.path.abspath(target) != os.path.abspath(input_path):
                    conflicts.append(os.path.basename(target))
            elif os.path.isdir(input_path):
                out_dir = output_path if os.path.isdir(output_path) else output_path
                if not os.path.isdir(out_dir):
                    return conflicts
                for root, _, files in os.walk(input_path):
                    for fn in files:
                        target = os.path.join(out_dir, fn)
                        if os.path.exists(target):
                            conflicts.append(fn)
                    # не углубляемся: рассматриваем только корень входной папки
                    break
        except Exception:
            pass
        return conflicts

    # Основной метод для запуска обработки
    def start_processing(self, settings):
        """Запуск обработки"""
        if self.is_running:
            return {'error': 'Уже запущено'}
        input_path = settings.get('input', '')
        output_path = settings.get('output', '')

        if not input_path:
            return {'error': 'Нет пути к файлу или папке'}

        # Используем папку входных данных как выход, если не указано
        if not output_path:
            if os.path.isfile(input_path):
                output_path = os.path.dirname(input_path)
            else:
                output_path = input_path

        # Проверка перезаписи входных данных
        overwrite = settings.get('overwrite', False)
        if self._would_overwrite_input(input_path, output_path):
            return {'error': 'Невозможно перезаписать входной файл! Выберите другую папку.'}

        # Проверка конфликтов файлов
        conflicts = self._check_file_conflicts(input_path, output_path)
        if conflicts and not overwrite:
            conflict_list = ', '.join(conflicts[:3])
            more = f" (+{len(conflicts)-3} шт.)" if len(conflicts) > 3 else ""
            return {'error': f'Файлы уже существуют: {conflict_list}{more}. Включите перезапись или выберите другую папку.'}

        # Сохраняем настройки
        self.save_config({
            'input': input_path,
            'output': output_path,
            'overwrite': overwrite,
            'transparent': settings.get('transparent', False),
            'max_bbox': settings.get('max_bbox', 15),
            'format': settings.get('format', 'None'),
            'detection_prompt': settings.get('detection_prompt', 'watermark'),
            'detection_skip': settings.get('detection_skip', 1),
            'fade_in': settings.get('fade_in', 0),
            'fade_out': settings.get('fade_out', 0),
            'theme': settings.get('theme', 'brainrot'),
            'lang': settings.get('lang', 'brainrot')
        })

        # Построение команды
        cmd = [sys.executable, 'remwm.py', input_path, output_path]

        if overwrite:
            cmd.append('--overwrite')
        if settings.get('transparent'):
            cmd.append('--transparent')
        max_bbox = settings.get('max_bbox', 15)
        cmd.append(f'--max-bbox-percent={int(max_bbox)}')
        format_opt = settings.get('format', 'None')
        if format_opt and format_opt != 'None':
            cmd.append(f'--force-format={format_opt}')
        detection_prompt = settings.get('detection_prompt', 'watermark')
        if detection_prompt != 'watermark':
            cmd.append(f'--detection-prompt={detection_prompt}')
        detection_skip = settings.get('detection_skip', 1)
        if int(detection_skip) > 1:
            cmd.append(f'--detection-skip={int(detection_skip)}')
        fade_in = settings.get('fade_in', 0)
        if float(fade_in) > 0:
            cmd.append(f'--fade-in={float(fade_in)}')
        fade_out = settings.get('fade_out', 0)
        if float(fade_out) > 0:
            cmd.append(f'--fade-out={float(fade_out)}')

        # Запуск в отдельном потоке
        self.is_running = True
        threading.Thread(target=self._run_process, args=(cmd,), daemon=True).start()
        return {'status': 'started'}

    def _run_process(self, cmd):
        """Запуск процесса и передача вывода"""
        try:
            # Логирование команды
            cli_display = ' '.join(cmd[1:])
            cli_display = cli_display.replace('remwm.py ', 'python remwm.py \\\n    ')
            cli_display = cli_display.replace(' --', ' \\\n    --')
            self._call_js(f'addLog("$ {json.dumps(cli_display)[1:-1]}", "text-neon-cyan")')

            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            working_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(working_dir, 'remwm.py')
            if not os.path.exists(script_path):
                self._call_js(f'addLog("ERROR: remwm.py не найден", "text-error")')
                self._call_js('processingComplete()')
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
                if not self.is_running:
                    break
                line = line.strip()
                if not line:
                    continue

                # Обработка прогресса
                if 'overall_progress:' in line:
                    try:
                        progress_str = line.split('overall_progress:')[1].strip()
                        progress = int(progress_str.replace('%', ''))
                        self._call_js(f'updateProgress({progress})')
                    except:
                        pass

                # Логирование
                escaped = json.dumps(line)
                if 'error' in line.lower() or 'failed' in line.lower():
                    color = 'text-error'
                elif 'warning' in line.lower():
                    color = 'text-yellow-400'
                elif 'success' in line.lower() or 'done' in line.lower() or 'saved' in line.lower():
                    color = 'text-neon-green'
                else:
                    color = 'text-gray-400'
                self._call_js(f'addLog({escaped}, "{color}")')
            self.process.wait()
            self._call_js('processingComplete()')
        except Exception as e:
            import traceback
            self._call_js(f'addLog({json.dumps("Ошибка: " + str(e))}, "text-error")')
            self._call_js(f'addLog({json.dumps(traceback.format_exc())}, "text-gray-500")')
            self._call_js('processingComplete()')
        finally:
            self.is_running = False
            self.process = None

    def _call_js(self, js_code):
        if self.window:
            try:
                self.window.evaluate_js(js_code)
            except:
                pass

    def stop_processing(self):
        self.is_running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=0.5)
            except:
                pass
        return {'status': 'stopped'}

    def preview_detection(self, settings):
        """Предварительный просмотр detection"""
        input_path = settings.get('input', '')
        detection_prompt = settings.get('detection_prompt', 'watermark')
        max_bbox = settings.get('max_bbox', 15)
        if not input_path:
            return {'error': 'Нет пути к файлу'}
        try:
            cmd = [
                sys.executable, 'remwm.py',
                input_path, '--preview',
                '--max-bbox-percent', str(int(max_bbox)),
                '--detection-prompt', detection_prompt
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=os.path.dirname(os.path.abspath(__file__)))
            if result.returncode != 0:
                return {'error': result.stderr or 'Предпросмотр не удался'}
            for line in result.stdout.splitlines():
                if line.startswith('{'):
                    return json.loads(line)
            return {'error': 'Нет данных для предпросмотра'}
        except subprocess.TimeoutExpired:
            return {'error': 'Время ожидания вышло'}
        except Exception as e:
            return {'error': str(e)}

# ===========================
# Запуск Webview
# ===========================

def start_webview():
    import webview
    api = Api()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(script_dir, 'ui', 'index.html')
    window = webview.create_window(
        'WatermarkRemover AI - Ohio Edition',
        ui_path,
        js_api=api,
        width=950,
        height=860,
        min_size=(800, 600),
        background_color='#050505'
    )
    api.set_window(window)
    webview.start()

# ===========================
# Запуск Streamlit (с загрузкой и скачиванием)
# ===========================

def start_streamlit():
    global api
    api = Api()

    st.title("WatermarkRemover AI - Настройки (Streamlit с загрузкой/скачиванием)")

    # File uploader
    uploaded = st.file_uploader(
        "Загрузить файл для обработки",
        type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif']
    )
    saved_path = ""
    if uploaded is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(script_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        # Сохраняем загруженный файл в папку uploads
        saved_path = os.path.join(upload_dir, uploaded.name)
        try:
            with open(saved_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Файл сохранён: {saved_path}")
        except Exception as e:
            st.error(f"Не удалось сохранить файл: {e}")
            saved_path = ""

    # Поля для ручного ввода (или заполнятся автоматически если был upload)
    input_path = st.text_input("Путь к файлу или папке для обработки", value=saved_path)
    output_path = st.text_input("Путь для сохранения результата (оставьте пустым — использовать входную папку)")
    overwrite = st.checkbox("Перезаписать файлы", value=False)
    transparent = st.checkbox("Прозрачность", value=False)
    max_bbox = st.slider("Максимальный процент области", 0, 100, 15)
    format_opt = st.selectbox("Формат принудительной обработки", ["None", "JPEG", "PNG"])
    detection_prompt = st.selectbox("Подсказка для обнаружения", ["watermark", "logo", "текст"])
    detection_skip = st.slider("Шаг обнаружения", 1, 10, 1)
    fade_in = st.slider("Плавное появление", 0, 10, 0)
    fade_out = st.slider("Плавное исчезновение", 0, 10, 0)

    if st.button("Запустить обработку"):
        if not input_path:
            st.error("Укажите путь к файлу или загрузите файл.")
        else:
            settings = {
                'input': input_path,
                'output': output_path,
                'overwrite': overwrite,
                'transparent': transparent,
                'max_bbox': max_bbox,
                'format': format_opt,
                'detection_prompt': detection_prompt,
                'detection_skip': detection_skip,
                'fade_in': fade_in,
                'fade_out': fade_out
            }

            def run_and_wait():
                res = api.start_processing(settings)
                if 'error' in res:
                    st.error(res['error'])
                    return
                # Ждём пока процесс завершится (api._run_process в отдельном потоке)
                with st.spinner("Обработка... (ожидание завершения процесса)"):
                    while api.is_running:
                        time.sleep(0.5)
                st.success("Обработка завершена.")

                # Определяем папку с результатами
                inp = settings.get('input')
                out_dir = settings.get('output') or (os.path.dirname(inp) if os.path.isfile(inp) else inp)
                files_to_offer = []
                if os.path.isdir(out_dir):
                    for fn in sorted(os.listdir(out_dir)):
                        full = os.path.join(out_dir, fn)
                        if os.path.isfile(full):
                            files_to_offer.append(full)
                elif os.path.isfile(out_dir):
                    files_to_offer.append(out_dir)

                if not files_to_offer:
                    st.info("Нет файлов для скачивания в указанной папке.")
                else:
                    st.markdown("### Скачать результаты")
                    for fp in files_to_offer:
                        name = os.path.basename(fp)
                        try:
                            with open(fp, "rb") as fh:
                                data = fh.read()
                            st.download_button(label=f"Скачать {name}", data=data, file_name=name)
                        except Exception as e:
                            st.write(f"Не удалось подготовить {name}: {e}")

            # Запускаем в отдельном потоке, чтобы интерфейс не блокировался (Streamlit всё равно обновит вид)
            threading.Thread(target=run_and_wait, daemon=True).start()

# ===========================
# Основной запуск
# ===========================

if __name__ == '__main__':
    if USE_STREAMLIT:
        start_streamlit()
    else:
        start_webview()
