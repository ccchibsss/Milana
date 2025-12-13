FROM python:3.13-slim

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        build-essential \
        python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY . .

# Запуск Streamlit
CMD ["streamlit", "run", "streamlit_app.py"]
