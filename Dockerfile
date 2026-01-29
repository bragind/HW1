FROM python:3.11-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Устанавливаем системные зависимости для сборки scikit-surprise и научных пакетов
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Сначала копируем только зависимости (для кеширования слоёв)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Затем копируем остальной проект
COPY . .

# Точка входа: по умолчанию полный эксперимент; аргументы можно переопределить при docker run
ENTRYPOINT ["python", "main.py"]
CMD []

