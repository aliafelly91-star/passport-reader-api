"""
main.py — خدمة قراءة الجوازات (FastAPI)
=========================================
نسخة محسنة للقراءة السحابية.

- نفس Endpoint: /read-passport
- نفس أسماء الحقول التي ينتظرها Flutter
- لا يوجد أي تغيير مطلوب في تطبيق Flutter
- يدعم الصورة الأصلية + 180 درجة
- يجرب عدة طرق لمعالجة صورة الـ MRZ
- يبحث عن أفضل سطرين MRZ بدل الاعتماد على قراءة واحدة
- يستخدم MRZ checksum لاختيار النتيجة الأقوى
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
# إعداد Tesseract
# ============================================================================

TESS_CONFIGS = [
    (
        "--oem 1 --psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< "
        "-c load_system_dawg=0 -c load_freq_dawg=0"
    ),
    (
        "--oem 1 --psm 11 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< "
        "-c load_system_dawg=0 -c load_freq_dawg=0"
    ),
]


# ============================================================================
# الدول
# ============================================================================

COUNTRY_NAMES = {
    "IRQ": "IRAQ",
    "PAK": "PAKISTAN",
    "IND": "INDIA",
    "AFG": "AFGHANISTAN",
    "IRN": "IRAN",
    "SYR": "SYRIA",
    "EGY": "EGYPT",
    "JOR": "JORDAN",
    "LBN": "LEBANON",
    "SAU": "SAUDI ARABIA",
    "ARE": "UNITED ARAB EMIRATES",
    "KWT": "KUWAIT",
    "QAT": "QATAR",
    "BHR": "BAHRAIN",
    "OMN": "OMAN",
    "YEM": "YEMEN",
    "TUR": "TURKEY",
    "PSE": "PALESTINE",
    "BGD": "BANGLADESH",
    "PHL": "PHILIPPINES",
    "LKA": "SRI LANKA",
    "NPL": "NEPAL",
    "ETH": "ETHIOPIA",
    "SDN": "SUDAN",
    "SOM": "SOMALIA",
    "MAR": "MOROCCO",
    "DZA": "ALGERIA",
    "TUN": "TUNISIA",
    "LBY": "LIBYA",
    "USA": "UNITED STATES",
    "GBR": "UNITED KINGDOM",
    "CAN": "CANADA",
    "FRA": "FRANCE",
    "DEU": "GERMANY",
}


NATIONALITY_NAMES = {
    "IRQ": "IRAQI",
    "PAK": "PAKISTANI",
    "IND": "INDIAN",
    "AFG": "AFGHAN",
    "IRN": "IRANIAN",
    "SYR": "SYRIAN",
    "EGY": "EGYPTIAN",
    "JOR": "JORDANIAN",
    "LBN": "LEBANESE",
    "SAU": "SAUDI",
    "ARE": "EMIRATI",
    "KWT": "KUWAITI",
    "QAT": "QATARI",
    "BHR": "BAHRAINI",
    "OMN": "OMANI",
    "YEM": "YEMENI",
    "TUR": "TURKISH",
    "PSE": "PALESTINIAN",
    "BGD": "BANGLADESHI",
    "PHL": "FILIPINO",
    "LKA": "SRI LANKAN",
    "NPL": "NEPALESE",
    "ETH": "ETHIOPIAN",
    "SDN": "SUDANESE",
    "SOM": "SOMALI",
    "MAR": "MOROCCAN",
    "DZA": "ALGERIAN",
    "TUN": "TUNISIAN",
    "LBY": "LIBYAN",
    "USA": "AMERICAN",
    "GBR": "BRITISH",
    "CAN": "CANADIAN",
    "FRA": "FRENCH",
    "DEU": "GERMAN",
}


MONTHS = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]


# ============================================================================
# تنظيف نص Tesseract
# ============================================================================

def clean_ocr_line(line: str) -> str:
    """
    ننظف السطر ونبقي فقط أحرف MRZ.
    """

    line = line.upper().strip()

    # أخطاء OCR الشائعة
    replacements = {
        "«": "<",
        "‹": "<",
        "≤": "<",
        "—": "<",
        "_": "<",
        "|": "<",
        " ": "",
    }

    for old, new in replacements.items():
        line = line.replace(old, new)

    line = re.sub(r"[^A-Z0-9<]", "", line)

    return line


# ============================================================================
# استخراج مرشحي MRZ
# ============================================================================

def extract_mrz_candidates(text: str):
    """
    نستخرج جميع الأسطر المحتملة لمنطقة MRZ.

    ما نعتمد فقط على أن السطر يبدأ P،
    لأن Tesseract أحياناً يخطئ أول حرف.
    """

    raw_lines = text.splitlines()

    lines = []

    for raw in raw_lines:
        cleaned = clean_ocr_line(raw)

        if len(cleaned) >= 30:
            lines.append(cleaned)

    candidates = []

    for i in range(len(lines) - 1):

        l1 = lines[i]
        l2 = lines[i + 1]

        # نحاول فقط الأسطر القريبة من طول TD3
        if len(l1) < 30 or len(l2) < 30:
            continue

        # نأخذ أول 44 حرف
        l1_44 = l1[:44].ljust(44, "<")
        l2_44 = l2[:44].ljust(44, "<")

        # السطر الأول لجواز TD3 غالباً يبدأ P
        first_ok = (
            l1_44.startswith("P")
            or l1_44.startswith("<<P")
            or "P<" in l1_44[:5]
        )

        # السطر الثاني عادة يحتوي تواريخ وأرقام تحقق
        second_has_digits = sum(c.isdigit() for c in l2_44) >= 8

        if first_ok and second_has_digits:
            candidates.append((l1_44, l2_44))

    return candidates


# ============================================================================
# استخراج إضافي إذا Tesseract دمج الأسطر
# ============================================================================

def extract_mrz_from_full_text(text: str):
    """
    محاولة ثانية لاستخراج MRZ حتى لو Tesseract دمج الأسطر.
    """

    cleaned = clean_ocr_line(text)

    candidates = []

    # نبحث عن أي مقطع بطول 88 تقريباً
    for start in range(max(0, len(cleaned) - 100)):
        chunk = cleaned[start:start + 88]

        if len(chunk) < 80:
            continue

        l1 = chunk[:44]
        l2 = chunk[44:88]

        if (
            len(l1) == 44
            and len(l2) == 44
            and sum(c.isdigit() for c in l2) >= 8
        ):
            candidates.append((l1, l2))

    return candidates


# ============================================================================
# التاريخ
# ============================================================================

def format_date(yymmdd, is_birth=True):
    """
    نحول تاريخ MRZ إلى:

    YYYY-MON-DD
    """

    if not yymmdd:
        return ""

    yymmdd = str(yymmdd).strip()

    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return ""

    try:
        yy = int(yymmdd[:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])

        if not 1 <= mm <= 12:
            return ""

        if not 1 <= dd <= 31:
            return ""

        if is_birth:
            year = 1900 + yy if yy > 30 else 2000 + yy
        else:
            year = 2000 + yy

        return f"{year}-{MONTHS[mm - 1]}-{dd:02d}"

    except Exception:
        return ""


# ============================================================================
# استخراج الأسماء
# ============================================================================

def extract_names(fields):
    """
    MRZ:

    SURNAME<<GIVEN<NAMES

    نرجع:
    surname
    given_names
    """

    raw_name = str(fields.name or "").strip()

    if not raw_name:
        return "", ""

    raw_name = raw_name.replace(" ", "<")

    parts = raw_name.split("<<", 1)

    surname = ""
    given_names = ""

    if len(parts) >= 1:
        surname = parts[0].replace("<", " ").strip()

    if len(parts) >= 2:
        given_names = parts[1].replace("<", " ").strip()

    # تنظيف المسافات المتكررة
    surname = re.sub(r"\s+", " ", surname)
    given_names = re.sub(r"\s+", " ", given_names)

    return surname, given_names


# ============================================================================
# حساب قوة النتيجة
# ============================================================================

def calculate_score(checker, fields, l1, l2):
    """
    نعطي نقاط للنتيجة حسب صحة البيانات.

    الهدف:
    إذا OCR أعطى أكثر من نتيجة، نختار الأقوى.
    """

    score = 0

    # checksum
    try:
        if checker.report.warnings == []:
            score += 5
    except Exception:
        pass

    # رقم الجواز
    if fields.document_number:
        score += 2

    # تاريخ الميلاد
    if fields.birth_date:
        score += 2

    # تاريخ الانتهاء
    if fields.expiry_date:
        score += 2

    # الجنسية
    if fields.country:
        score += 1

    # الاسم
    if fields.name:
        score += 1

    # طول MRZ الصحيح
    if len(l1) == 44:
        score += 1

    if len(l2) == 44:
        score += 1

    return score


# ============================================================================
# تجربة مرشح MRZ
# ============================================================================

def try_mrz_candidate(l1, l2):
    try:

        # إصلاح بسيط لبعض أخطاء OCR
        l1 = clean_ocr_line(l1)[:44].ljust(44, "<")
        l2 = clean_ocr_line(l2)[:44].ljust(44, "<")

        checker = TD3CodeChecker(
            f"{l1}\n{l2}",
            check_expiry=False
        )

        fields = checker.fields()

        score = calculate_score(
            checker,
            fields,
            l1,
            l2
        )

        country_code = (
            fields.country or ""
        ).upper().strip()

        surname, given_names = extract_names(fields)

        result = {
            "success": True,

            "given_name_en": given_names,

            "surname_en": surname,

            "passport_number": (
                fields.document_number or ""
            ).strip(),

            "nationality": NATIONALITY_NAMES.get(
                country_code,
                country_code
            ),

            "residence_country": COUNTRY_NAMES.get(
                country_code,
                country_code
            ),

            "birth_date": format_date(
                fields.birth_date,
                True
            ),

            "expiry_date": format_date(
                fields.expiry_date,
                False
            ),

            "sex": fields.sex or "",

            "score": score,

            # تأكيد القراءة فقط إذا checksum صحيح
            "is_verified": (
                checker.report.warnings == []
            ),
        }

        return score, result

    except Exception:
        return -1, None


# ============================================================================
# معالجة صورة واحدة
# ============================================================================

def preprocess_variants(img_bgr):
    """
    نولد نسخ متعددة من الصورة.

    نستخدم:
    - القص السفلي
    - grayscale
    - CLAHE
    - OTSU
    - adaptive threshold
    - resize
    """

    variants = []

    h, w = img_bgr.shape[:2]

    # ------------------------------------------------------------------------
    # النسخة الكاملة أيضاً
    # ------------------------------------------------------------------------

    full = img_bgr.copy()

    if full.shape[1] < 1600:
        scale = 1600 / full.shape[1]

        full = cv2.resize(
            full,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    gray_full = cv2.cvtColor(
        full,
        cv2.COLOR_BGR2GRAY
    )

    variants.append(gray_full)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    variants.append(
        clahe.apply(gray_full)
    )

    # OTSU
    _, otsu_full = cv2.threshold(
        gray_full,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(otsu_full)

    # Adaptive
    adaptive_full = cv2.adaptiveThreshold(
        gray_full,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    variants.append(adaptive_full)

    # ------------------------------------------------------------------------
    # قص المنطقة السفلية بعدة نسب
    # ------------------------------------------------------------------------

    for crop_ratio in [
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]:

        y_start = int(
            h * (1 - crop_ratio)
        )

        crop = img_bgr[
            y_start:h,
            0:w
        ]

        if crop.size == 0:
            continue

        # تكبير
        if crop.shape[1] < 1600:

            scale = 1600 / crop.shape[1]

            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        # grayscale
        variants.append(gray)

        # CLAHE
        variants.append(
            clahe.apply(gray)
        )

        # OTSU
        _, otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        variants.append(otsu)

        # Adaptive
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

        variants.append(adaptive)

    return variants


# ============================================================================
# معالجة الصورة واختيار أفضل نتيجة
# ============================================================================

def process_image(img_bgr):

    best_score = -1
    best_data = None

    variants = preprocess_variants(img_bgr)

    for variant in variants:

        for config in TESS_CONFIGS:

            try:

                text = pytesseract.image_to_string(
                    variant,
                    config=config,
                    lang="eng"
                )

                # ------------------------------------------------------------
                # الطريقة الأولى
                # ------------------------------------------------------------

                candidates = extract_mrz_candidates(text)

                # ------------------------------------------------------------
                # الطريقة الثانية
                # ------------------------------------------------------------

                candidates.extend(
                    extract_mrz_from_full_text(text)
                )

                # ------------------------------------------------------------
                # تجربة كل المرشحين
                # ------------------------------------------------------------

                for l1, l2 in candidates:

                    score, data = try_mrz_candidate(
                        l1,
                        l2
                    )

                    if data is None:
                        continue

                    if score > best_score:

                        best_score = score
                        best_data = data

                    # إذا حصلنا على checksum صحيح
                    # مع بيانات أساسية كاملة، ما نحتاج
                    # نضيع وقت على باقي النسخ.
                    if (
                        data["is_verified"]
                        and data["passport_number"]
                        and data["birth_date"]
                        and data["expiry_date"]
                    ):
                        return data

            except Exception:
                continue

    return best_data


# ============================================================================
# قراءة الجواز من bytes
# ============================================================================

def read_passport_from_bytes(image_bytes: bytes) -> dict:

    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    img_bgr = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if img_bgr is None:

        return {
            "success": False,
            "error": "تعذر فك ترميز الصورة"
        }

    # ========================================================================
    # 1. الصورة الأصلية
    # ========================================================================

    best_data = process_image(
        img_bgr
    )

    if best_data is not None:

        # إذا القراءة موثوقة نرجعها مباشرة
        if best_data.get("is_verified"):

            return best_data

    # ========================================================================
    # 2. الصورة مقلوبة 180 درجة
    # ========================================================================

    img_flipped = cv2.rotate(
        img_bgr,
        cv2.ROTATE_180
    )

    flipped_data = process_image(
        img_flipped
    )

    # إذا المقلوبة أفضل
    if flipped_data is not None:

        if (
            best_data is None
            or flipped_data.get("score", 0)
            > best_data.get("score", 0)
        ):
            best_data = flipped_data

    # ========================================================================
    # لا توجد نتيجة
    # ========================================================================

    if best_data is None:

        return {
            "success": False,
            "error": (
                "ما قدرنا نلقى منطقة قراءة آلية واضحة بالصورة"
            )
        }

    return best_data


# ============================================================================
# Endpoint فحص الخدمة
# ============================================================================

@app.get("/")
def health_check():

    return {
        "status": "الخدمة شغالة ✓"
    }


# ============================================================================
# Endpoint قراءة الجواز
# ============================================================================

@app.post("/read-passport")
async def read_passport_endpoint(
    file: UploadFile = File(...)
):

    try:

        image_bytes = await file.read()

        if not image_bytes:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "الصورة فارغة"
                }
            )

        result = read_passport_from_bytes(
            image_bytes
        )

        return JSONResponse(
            content=result
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"خطأ داخلي: {str(e)}"
            }
        )
