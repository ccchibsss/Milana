# !/usr/bin/env python3
"""
Улучшенная версия скрипта WatermarkRemover GUI.

Особенности и улучшения:
- Отложенный импорт streamlit и pyyaml (работает корректно, если их нет).
- Если pyyaml отсутствует — сохраняем конфиг в JSON (избегаем ошибки ModuleNotFoundError).
- Если streamlit отсутствует — запускаем headless тестовый режим с эмулятором remwm.py
  и простым HTTP-сервером для скачивания результатов.
- Более корректная обработка путей (Pathlib), логирование и обработка ошибок.
- Русскоязычные комментарии и подсказки.
- Кнопка загрузки и скачивания при наличии Streamlit.
"""

from __future__ import annotations
import sys
import os
import time
import logging
import threading
import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import http.server
import socketserver
import signal

# Настройка логирования (вывод в консоль)
LOG = logging.getLogger("wm_gui")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Попытка импортировать streamlit (лениво)
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False
    st = None  # type: ignore

# Попытка импортировать pyyaml; если нет — используем JSON как запасной вариант
try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False
    LOG.info("pyyaml не установлен — конфигурация будет сохраняться в JSON в файле ui.yml")

# Константы директорий
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "ui.yml"  # если yaml недоступен — сохраним JSON туда же
UPLOADS_DIR = BASE_DIR / "uploads"
DEFAULT_OUTPUTS_DIR = BASE_DIR / "outputs"

# Эмулятор remwm.py (используется в headless режиме, если реального remwm.py
# нет)
DUMMY_REMWM = r'''#!/usr/bin/env python3
import sys, time, shutil, os
def main():
    args = sys.argv[1:]
    if not args:
        print("error: no args")
        sys.exit(1)
    inp = args[0]
    out = args[1] if len(args)>1 else "."
    # Рассчёт целевого файла
    out_is_dir = os.path.isdir(out) or not out.lower().endswith(('.png','.jpg','.jpeg','.webp','.bmp','.gif'))
    if out_is_dir:
        os.makedirs(out, exist_ok=True)
        base = os.path.basename(inp) or "output.bin"
        target = os.path.join(out, base)
    else:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        target = out
    # Симуляция прогресса
    for p in [0, 20, 50, 80, 100]:
        print(f"overall_progress: {p}%")
        time.sleep(0.15)
    try:
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

if __name__ == "__main__":
    main()
'''

# ---------------------------
# Работа с конфигурацией
# ---------------------------

def load_config() -> Dict[str, Any]:
    """Загрузить конфиг из файла. Если pyyaml нет — пытаемся читать JSON."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        text = CONFIG_FILE.read_text(encoding="utf-8")
        if YAML_AVAILABLE:
            return yaml.safe_load(text) or {}
        else:
            # Попробуем JSON как запасной план
            return json.loads(text)
    except Exception as e:
        LOG.warning("Не удалось загрузить конфиг: %s", e)
        return {}

def save_config(cfg: Dict[str, Any]) -> None:
    """Сохранить конфиг. Используем YAML, если есть, иначе JSON."""
    try:
        if YAML_AVAILABLE:
            with CONFIG_FILE.open("w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        else:
            with CONFIG_FILE.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        LOG.error("Не удалось сохранить конфиг: %s", e)

# ---------------------------
# Класс API управления
# ---------------------------

class Api:
    """
    Класс Api содержит логику запуска внешнего процесса (remwm.py),
    проверки конфликтов, сохранения настроек и чтения прогресса.
    """
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.is_running: bool = False
        self.config = load_config()
        self._lock = threading.Lock()

    # --- Утилиты проверки путей ---
    def _would_overwrite_input(self, input_path: Path, output_path: Path) -> bool:
        """Проверяет, приведёт ли запись в output к перезаписи исходного файла."""
        try:
            inp = input_path.resolve()
            out = output_path.resolve()
            if inp == out:
                return True
            if inp.is_file() and out.is_dir():
                if (out / inp.name).resolve() == inp:
                    return True
            if out.is_file() and out.resolve() == inp:
                return True
        except Exception:
            return False
        return False

    def _check_file_conflicts(self, input_path: Path, output_path: Path) -> List[str]:
        """
        Ищет конфликты по именам файлов: если вход - файл, проверяем наличие в выходной папке;
        если вход - папка, проверяем пересечения имён в корне.
        """
        conflicts: List[str] = []
        try:
            if input_path.is_file():
                out_dir = output_path if output_path.is_dir() else output_path.parent
                target = out_dir / input_path.name
                if target.exists() and target.resolve() != input_path.resolve():
                    conflicts.append(str(target.name))
            elif input_path.is_dir():
                out_dir = output_path if output_path.is_dir() else output_path
                if not out_dir.exists() or not out_dir.is_dir():
                    return conflicts
                for child in input_path.iterdir():
                    if child.is_file():
                        target = out_dir / child.name
                        if target.exists():
                            conflicts.append(child.name)
        except Exception as e:
            LOG.debug("Ошибка проверки конфликтов: %s", e)
        return conflicts

    # --- Основной запуск обработки ---
    def start_processing(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подготавливает команду и запускает remwm.py в отдельном потоке.
        Возвращает словарь с {'status':'started'} или {'error': '...'}.
        """
        with self._lock:
            if self.is_running:
                return {"error": "Уже выполняется процесс."}

            input_str = settings.get("input", "")
            if not input_str:
                return {"error": "Не указан путь к файлу или папке."}
            input_path = Path(input_str)

            output_str = settings.get("output", "")
            if not output_str:
                output_path = (input_path.parent if input_path.is_file() else input_path) or DEFAULT_OUTPUTS_DIR
            else:
                output_path = Path(output_str)

            # Нормализуем
            input_path = input_path.expanduser()
            output_path = output_path.expanduser()

            # Проверки на перезапись и конфликты
            overwrite = bool(settings.get("overwrite", False))
            if self._would_overwrite_input(input_path, output_path):
                return {"error": "Невозможно перезаписать входной файл! Выберите другую папку."}

            conflicts = self._check_file_conflicts(input_path, output_path)
            if conflicts and not overwrite:
                sample = ", ".join(conflicts[:3])
                more = f" (+{len(conflicts)-3} файлов)" if len(conflicts) > 3 else ""
                return {"error": f"Файлы уже существуют: {sample}{more}. Включите перезапись или выберите другую папку."}

            # Сохраняем конфиг (полезно для UI)
            cfg_to_save = {
                "input": str(input_path),
                "output": str(output_path),
                "overwrite": overwrite,
            }
            save_config(cfg_to_save)
            LOG.info("Сохранены настройки: %s", cfg_to_save)

            # Подготовка команды: remwm.py должен находиться в той же папке, что и скрипт
            remwm_script = BASE_DIR / "remwm.py"
            if not remwm_script.exists():
                LOG.warning("remwm.py не найден в %s — создаём временный эмулятор.", remwm_script)
                remwm_script.write_text(DUMMY_REMWM, encoding="utf-8")
                remwm_script.chmod(0o755)

            cmd = [sys.executable, str(remwm_script), str(input_path), str(output_path)]
            if overwrite:
                cmd.append("--overwrite")

            # Запуск процесса в фоне
            self.is_running = True
            thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
            thread.start()
            LOG.info("Процесс запущен: %s", " ".join(cmd))
            return {"status": "started"}

    def _run_process(self, cmd: List[str]) -> None:
        """Функция, выполняемая в отдельном потоке: запускает subprocess и логирует вывод."""
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(BASE_DIR),
            )
            self.process = proc
            # Читаем stdout построчно
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n")
                if line:
                    LOG.info("[remwm] %s", line)
            proc.wait()
            LOG.info("remwm завершил работу с кодом %s", proc.returncode)
        except Exception as e:
            LOG.exception("Ошибка в процессе обработки: %s", e)
        finally:
            self.is_running = False
            self.process = None

    def stop_processing(self) -> Dict[str, Any]:
        """Прекращает запущенный процесс (если есть)."""
        with self._lock:
            self.is_running = False
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=1.0)
                    LOG.info("Процесс принудительно остановлен.")
                except Exception:
                    LOG.warning("Не удалось корректно завершить процесс; пробуем kill.")
                    try:
                        self.process.kill()
                    except Exception:
                        pass
            return {"status": "stopped"}

    def preview_detection(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Заглушка для preview_detection (в оригинале вызывалось remwm.py --preview).
        Здесь выполняется синхронный вызов и возвращается JSON-строка, если она была в выводе.
        """
        input_str = settings.get("input", "")
        if not input_str:
            return {"error": "Нет пути к файлу"}
        remwm_script = BASE_DIR / "remwm.py"
        if not remwm_script.exists():
            return {"error": "remwm.py не найден"}
        cmd = [sys.executable, str(remwm_script), input_str, "--preview"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR), timeout=30)
            if res.returncode != 0:
                return {"error": res.stderr.strip() or "Предпросмотр не удался"}
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        return json.loads(line)
                    except Exception:
                        pass
            return {"error": "Нет данных для предпросмотра"}
        except subprocess.TimeoutExpired:
            return {"error": "Время ожидания вышло"}
        except Exception as e:
            return {"error": str(e)}

# ---------------------------
# Streamlit UI (если доступен)
# ---------------------------

def start_streamlit_ui() -> None:
    """Запускает интерфейс Streamlit (при наличии установленного пакета)."""
    if not STREAMLIT_AVAILABLE:
        raise RuntimeError("Streamlit не установлен")

    api = Api()
    st.title("WatermarkRemover AI — Настройки (Streamlit)")

    # Загрузчик файлов
    uploaded = st.file_uploader("Загрузить файл для обработки", type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif'])
    saved_path = ""
    if uploaded is not None:
        UP = UPLOADS_DIR
        UP.mkdir(parents=True, exist_ok=True)
        saved_path = str(UP / uploaded.name)
        try:
            with open(saved_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Файл сохранён: {saved_path}")
        except Exception as e:
            st.error(f"Не удалось сохранить файл: {e}")
            saved_path = ""

    # Поля ввода параметров
    input_path = st.text_input("Путь к файлу или папке для обработки", value=saved_path)
    output_path = st.text_input("Путь для сохранения результата (оставьте пустым — использовать входную папку)")
    overwrite = st.checkbox("Перезаписать файлы", value=False)
    # Дополнительные параметры (из исходного интерфейса)
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
                "input": input_path,
                "output": output_path,
                "overwrite": overwrite,
                "transparent": transparent,
                "max_bbox": max_bbox,
                "format": format_opt,
                "detection_prompt": detection_prompt,
                "detection_skip": detection_skip,
                "fade_in": fade_in,
                "fade_out": fade_out,
            }
            res = api.start_processing(settings)
            if "error" in res:
                st.error(res["error"])
            else:
                st.success("Обработка запущена.")
                # Ожидаем завершения и показываем спиннер
                with st.spinner("Ожидание завершения..."):
                    while api.is_running:
                        time.sleep(0.3)
                st.success("Обработка завершена.")

                # Предлагаем скачать результаты
                inp = settings["input"]
                out_dir = settings["output"] or (str(Path(inp).parent) if Path(inp).is_file() else settings["input"])
                out_path = Path(out_dir)
                files_to_offer: List[Path] = []
                if out_path.is_dir():
                    files_to_offer = sorted([p for p in out_path.iterdir() if p.is_file()])
                elif out_path.is_file():
                    files_to_offer = [out_path]

                if not files_to_offer:
                    st.info("Нет файлов для скачивания в указанной папке.")
                else:
                    st.markdown("### Скачать результаты")
                    for fp in files_to_offer:
                        try:
                            with fp.open("rb") as fh:
                                data = fh.read()
                            st.download_button(label=f"Скачать {fp.name}", data=data, file_name=fp.name)
                        except Exception as e:
                            st.warning(f"Не удалось подготовить {fp.name}: {e}")

# ---------------------------
# Headless тест (если Streamlit нет)
# ---------------------------

def run_headless_test() -> None:
    """
    Headless режим:
    - Создаёт папки uploads/ и outputs/
    - Копирует пример файла в uploads/
    - Запускает Api.start_processing с эмулятором remwm.py (если нужно)
    - После завершения стартует HTTP-сервер для скачивания результатов
    """
    LOG.info("Запуск headless теста (Streamlit не установлен).")
    # Подготовка окружения
    UP = UPLOADS_DIR
    OUT = DEFAULT_OUTPUTS_DIR
    UP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    sample = UP / "sample_input.png"
    if not sample.exists():
        sample.write_bytes(b"\x89PNG\r\n\x1a\n" + b"PNGDATA")  # простой фиктивный файл

    api = Api()
    settings = {"input": str(sample), "output": str(OUT), "overwrite": True}
    res = api.start_processing(settings)
    if "error" in res:
        LOG.error("Не удалось запустить обработку: %s", res["error"])
        return

    # Ждём завершения обработки с таймаутом
    timeout = 30  # секунд
    start = time.time()
    while api.is_running and (time.time() - start) < timeout:
        time.sleep(0.1)
    if api.is_running:
        LOG.warning("Обработка не завершилась за %ss, пытаемся остановить.", timeout)
        api.stop_processing()
    else:
        LOG.info("Обработка завершена. Содержимое каталога %s:", OUT)
        for p in sorted(OUT.iterdir()):
            LOG.info(" - %s", p)

    # Запускаем HTTP-сервер для скачивания результатов
    os.chdir(str(OUT))
    port = 8000
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    LOG.info("Сервер раздачи результатов запущен на http://localhost:%d/  (Ctrl+C для остановки)", port)

    # Обрабатываем Ctrl+C корректно
    def _signal_handler(sig, frame):
        LOG.info("Получен сигнал завершения, останавливаем сервер...")
        httpd.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        LOG.info("HTTP сервер остановлен.")

# ---------------------------
# Точка входа
# ---------------------------

def main() -> None:
    if STREAMLIT_AVAILABLE:
        # Если streamlit установлен — запустим UI.
        # Заметьте: обычно Streamlit запускается командой `streamlit run this_script.py`,
        # и в таком случае функция main() вызывается в контексте Streamlit.
        LOG.info("Streamlit доступен — запускаем UI.")
        start_streamlit_ui()
    else:
        # Иначе — headless тест
        LOG.info("Streamlit не найден — запускаем headless тест.")
        run_headless_test()

if __name__ == "__main__":
    main()
