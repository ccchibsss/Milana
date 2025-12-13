FROM python:3.13-slim

# Обновляем пакеты и устанавливаем необходимые системные библиотеки
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        build-essential \
        python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение
COPY app.py /app/app.py

# Запуск Streamlit
CMD ["streamlit", "run", "app.py"]
