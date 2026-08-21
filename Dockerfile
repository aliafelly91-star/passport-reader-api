FROM python:3.11-slim

# نثبّت Tesseract والمكتبات النظامية المطلوبة لـ OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render يمرر رقم المنفذ بمتغير البيئة PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
