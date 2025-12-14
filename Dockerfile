FROM python:3.10

WORKDIR /app

# Установка системных зависимостей для сборки Pillow и opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libopencv-dev

COPY requirements.txt .

# Установка зависимостей из requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
