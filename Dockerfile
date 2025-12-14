# Используем официальный образ Python с нужной версией (например, 3.10)
FROM python:3.10

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем все файлы проекта в контейнер
COPY . .

# Устанавливаем необходимые библиотеки, включая streamlit
RUN pip install --upgrade pip
RUN pip install streamlit

# Открываем порт, который использует Streamlit (обычно 8501)
EXPOSE 8501

# Команда запуска Streamlit с вашим приложением
CMD ["streamlit", "run", "streamlit_app.py"]
