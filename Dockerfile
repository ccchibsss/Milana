FROM python:3.13-slim

# Установка системных зависимостей для сборки пакетов
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        build-essential \
        python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Обновляем pip
RUN pip install --upgrade pip

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY . .

# Запуск приложения
CMD ["streamlit", "run", "streamlit_app.py"]
