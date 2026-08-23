"""
main.py — خدمة قراءة الجوازات (FastAPI) - النسخة المصححة
=========================================
نفس منطق السكربت الأصلي، مع إضافة دعم الصورة المقلوبة، وإصلاح استخراج الأسماء.
"""

import re
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from mrz.checker.td3 import TD3CodeChecker

app = FastAPI(title="خدمة قراءة الجوازات")

# ============================================================================
# نفس إعدادات السكربت الأصلي بالضبط
# ============================================================================
TESS_CONFIG = (
    "--oem 1 --psm 6 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< "
    "-c load_system_dawg=0 -c load_freq_dawg=0"
)

COUNTRY_NAMES = {
    "IRQ": "IRAQ", "PAK": "PAKISTAN", "IND": "INDIA", "AFG": "AFGHANISTAN",
    "IRN": "IRAN", "SYR": "SYRIA", "EGY": "EGYPT", "JOR": "JORDAN",
    "LBN": "LEBANON", "SAU": "SAUDI ARABIA", "ARE": "UNITED ARAB EMIRATES",
    "KWT": "KUWAIT", "QAT": "QATAR", "BHR": "BAHRAIN", "OMN": "OMAN",
    "YEM": "YEMEN", "TUR": "TURKEY", "PSE": "PALESTINE", "BGD": "BANGLADESH",
    "PHL": "PHILIPPINES", "LKA": "SRI LANKA", "NPL": "NEPAL", "ETH": "ETHIOPIA",
    "SDN": "SUDAN", "SOM": "SOMALIA", "MAR": "MOROCCO", "DZA": "ALGERIA",
    "TUN": "TUNISIA", "LBY": "LIBYA", "USA": "UNITED STATES",
    "GBR": "UNITED KINGDOM", "CAN": "CANADA", "FRA": "FRANCE", "DEU": "GERMANY",
}

NATIONALITY_NAMES = {
    "IRQ": "IRAQI", "PAK": "PAKISTANI", "IND": "INDIAN", "AFG": "AFGHAN",
    "IRN": "IRANIAN", "SYR": "SYRIAN", "EGY": "EGYPTIAN", "JOR": "JORDANIAN",
    "LBN": "LEBANESE", "SAU": "SAUDI", "ARE": "EMIRATI", "KWT": "KUWAITI",
    "QAT": "QATARI", "BHR": "BAHRAINI", "OMN": "OMANI", "YEM": "YEMENI",
    "TUR": "TURKISH", "PSE": "PALESTINIAN", "BGD": "BANGLADESHI",
    "PHL": "FILIPINO", "LKA": "SRI LANKAN", "NPL": "NEPALESE",
    "ETH": "ETHIOPIAN", "SDN": "SUDANESE", "SOM": "SOMALI", "MAR": "MOROCCAN",
    "DZA": "ALGERIAN", "TUN": "TUNISIAN", "LBY": "LIBYAN", "USA": "AMERICAN",
    "GBR": "BRITISH", "CAN": "CANADIAN", "FRA": "FRENCH", "DEU": "GERMAN",
}

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def preprocess_variants(img_bgr):
    """نولّد عدة نسخ معالجة مختلفة من الصورة"""
    variants = []
    h, w = img_bgr.shape[:2]

    for crop_ratio in [0.30, 0.25, 0.35, 0.40]:
        y_start = int(h * (1 - crop_ratio))
        crop = img_bgr[y_start:h, 0:w]

        # نكبّر لو صغير
        if crop.shape[1] < 1400:
            scale = 1400 / crop.shape[1]
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # نسخة 1: عتبة تكيّفية
        variants.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15))
        # نسخة 2: Otsu
        _, v2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(v2)
        # نسخة 3: تحسين تباين
        variants.append(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray))

    return variants


def extract_mrz_lines(text):
    """نستخرج سطري منطقة القراءة الآلية من النص الخام"""
    lines = [re.sub(r"[^A-Z0-9<]", "", ln.upper()) for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 30]

    for i in range(len(lines) - 1):
        l1, l2 = lines[i], lines[i + 1]
        if l1.startswith("P") and "<" in l1:
            return l1[:44].ljust(44, "<"), l2[:44].ljust(44, "<")
    return None, None


def format_date(yymmdd, is_birth=True):
    """نحوّل تاريخ MRZ لصيغة: السنة-الشهر(مختصر)-اليوم"""
    if not yymmdd or len(yymmdd) != 6:
        return ""
    try:
        yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
        year = (1900 + yy if yy > 30 else 2000 + yy) if is_birth else 2000 + yy
        return f"{year}-{MONTHS[mm - 1]}-{dd:02d}"
    except Exception:
        return ""


def process_image(img_bgr):
    """دالة مساعدة لمعالجة صورة واحدة وإرجاع أفضل نتيجة"""
    best_score = -1
    best_data = None

    for variant in preprocess_variants(img_bgr):
        try:
            text = pytesseract.image_to_string(variant, config=TESS_CONFIG, lang='eng')
            l1, l2 = extract_mrz_lines(text)
            if not l1 or not l2:
                continue

            checker = TD3CodeChecker(f"{l1}\n{l2}", check_expiry=False)
            fields = checker.fields()

            score = sum([
                bool(checker.report.warnings == []),
                bool(fields.document_number),
                bool(fields.birth_date),
                bool(fields.expiry_date),
            ])

            if score > best_score:
                best_score = score
                country_code = (fields.country or "").upper()
                
                # استخراج الأسماء بشكل صحيح من الـ MRZ
                # الاسم بالـ MRZ يكون بهذا الشكل: SURNAME<<GIVEN<NAMES
                raw_name = fields.name or ""
                name_parts = raw_name.split("<<")
                surname = name_parts[0].replace("<", " ").strip() if len(name_parts) > 0 else ""
                given_names = name_parts[1].replace("<", " ").strip() if len(name_parts) > 1 else ""

                best_data = {
                    "success": True,
                    "given_name_en": given_names,
                    "surname_en": surname,
                    "passport_number": (fields.document_number or "").strip(),
                    "nationality": NATIONALITY_NAMES.get(country_code, country_code),
                    "residence_country": COUNTRY_NAMES.get(country_code, country_code),
                    "birth_date": format_date(fields.birth_date, True),
                    "expiry_date": format_date(fields.expiry_date, False),
                    "sex": fields.sex or "",
                    "score": score,
                    "is_verified": score >= 3, # خففنا الشرط لـ 3 عشان يقبل لو رقم تحقق وحد غلط
                }

            if score >= 4:
                break
        except Exception:
            continue
            
    return best_data


def read_passport_from_bytes(image_bytes: bytes) -> dict:
    """الدالة الرئيسية - نجرب الصورة الأصلية ثم المعكوسة"""
    np_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {"success": False, "error": "تعذر فك ترميز الصورة"}

    # 1. نجرب الصورة الأصلية
    best_data = process_image(img_bgr)

    # 2. لو ما لقينا شي، نجرب الصورة مقلوبة (180 درجة)
    if best_data is None:
        img_flipped = cv2.rotate(img_bgr, cv2.ROTATE_180)
        best_data = process_image(img_flipped)

    if best_data is None:
        return {"success": False, "error": "ما قدرنا نلقى منطقة قراءة آلية واضحة بالصورة"}

    return best_data


# ============================================================================
# نقاط الوصول (Endpoints)
# ============================================================================

@app.get("/")
def health_check():
    """للتأكد إن الخدمة شغالة"""
    return {"status": "الخدمة شغالة ✓"}


@app.post("/read-passport")
async def read_passport_endpoint(file: UploadFile = File(...)):
    """نقطة الوصول الرئيسية — نستقبل صورة، نرجع البيانات"""
    try:
        image_bytes = await file.read()
        result = read_passport_from_bytes(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"خطأ داخلي: {str(e)}"}
        )
