FROM python:3.11-slim

# ⚠ الإضافة الأهم: بدونها Python يخزّن المخرجات بذاكرة مؤقتة، وسجلّات
# Render تطلع فاضية حتى لو الخدمة تنهار — ولهذا ما تعرف ليش ما تقلع.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Tesseract والمكتبات النظامية المطلوبة لـ OpenCV
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

# فحص إن الاستيرادات تشتغل وقت البناء، مو وقت التشغيل.
# لو مكتبة انكسرت، البناء يفشل هنا برسالة واضحة بدل ما تنشر
# صورة تنهار عند الإقلاع وترجع 503 بلا تفسير.
RUN python -c "import cv2, numpy, pytesseract, fastapi; \
from mrz.checker.td3 import TD3CodeChecker; \
print('✓ كل الاستيرادات سليمة')"

COPY . .

# نفس الفحص على الملف نفسه — يكشف أي خطأ إملائي قبل النشر
RUN python -c "import main; print('✓ main.py يُستورد بنجاح')"

# Render يمرر رقم المنفذ بمتغير البيئة PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
