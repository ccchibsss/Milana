# Базовый образ Python
FROM python:3.11-slim

# Обновляем пакеты и устанавливаем необходимые системные библиотеки
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ваш скрипт в контейнер
COPY app.py /app/app.py

# Указываем команду запуска
CMD ["streamlit", "run", "app.py"]
