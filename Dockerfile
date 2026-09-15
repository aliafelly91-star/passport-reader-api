FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import cv2, numpy, pytesseract, fastapi; from mrz.checker.td3 import TD3CodeChecker; print('imports OK')"
COPY . .
RUN python -c "import main; print('main.py OK')"
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
