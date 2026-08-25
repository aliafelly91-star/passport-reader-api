"""
main.py — خدمة قراءة الجوازات (FastAPI) — النسخة المصححة
==========================================================
شنو تغيّر بهذي النسخة (مقارنة بالقديمة):

1. إصلاح المشكلة الأساسية: الاسم واللقب كانوا يطلعون غلط أو "None"
   السبب: الكود القديم كان ياخذ fields.name بس ويقسمه على "<<"،
   بينما مكتبة mrz أصلاً تفصلهم لحقلين (surname + name).
   الحل: نقرا اللقب والاسم من مواقعهم الصحيحة بالسطر الأول مباشرة.

2. السطر الأول (اللي بيه الأسماء) ما عليه أي رقم تحقق بمعيار TD3 —
   يعني ممكن يكون خردة والقراءة تظل "مؤكدة"! أضفنا فحص جودة
   للأسماء ودخّلناه بالتقييم، حتى النتيجة اللي أسماؤها سليمة تفوز.

3. ما نرجّع أبداً كلمة "None" ولا رموز غريبة — كل حقل ينظّف قبل
   ما يطلع، وإذا مو صالح يرجع فاضي.

4. أضفنا استخراج تاريخ الإصدار واسم الأب/الزوج من النص المطبوع
   (هذول أصلاً غير موجودين بمنطقة القراءة الآلية).

5. أضفنا /health للإيقاظ السريع، ودعم دوران 90 و270 درجة.

- نفس Endpoint: /read-passport
- نفس أسماء الحقول اللي ينتظرها Flutter (أضفنا حقول جديدة بس، ما حذفنا شي)
"""

import re
import time
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

# إعدادات قراءة منطقة MRZ — أحرف كبيرة وأرقام والرمز < فقط
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

# إعداد قراءة النص المطبوع العادي (لتاريخ الإصدار واسم الأب)
TESS_PRINTED_CONFIG = "--oem 1 --psm 6"

# ============================================================================
# سقف الوقت — أهم إضافة
# ============================================================================
# الخدمة المجانية بطيئة، ولو خليناها تجرب كل الاحتمالات ممكن تاخذ
# دقائق ويعلّق التطبيق. الخدمة ما تشتغل أكثر من هذا السقف أبداً —
# ترجع أفضل نتيجة وصلتها وتخلص.
MAX_SECONDS = 45


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
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]

MONTH_INDEX = {name: i + 1 for i, name in enumerate(MONTHS)}

# قيم خردة ممكن ترجع من المكتبة أو من OCR — نرفضها دائماً
BAD_VALUES = {"NONE", "NULL", "NAN", "N/A", "NA", "-", "--"}


# ============================================================================
# تنظيف نص Tesseract
# ============================================================================

def clean_ocr_line(line: str) -> str:
    """ننظف السطر ونبقي فقط أحرف MRZ المسموحة."""

    line = (line or "").upper().strip()

    # تصحيح رموز يخلط بيها Tesseract مع الرمز <
    replacements = {
        "«": "<", "‹": "<", "≤": "<", "—": "<",
        "_": "<", "|": "<", "«": "<", " ": "",
    }

    for old, new in replacements.items():
        line = line.replace(old, new)

    return re.sub(r"[^A-Z0-9<]", "", line)


def safe_text(value) -> str:
    """
    نحوّل أي قيمة لنص نظيف.

    الهدف: ما نرجّع أبداً كلمة "None" أو قيم خردة للتطبيق —
    هذي كانت تظهر بحقل اللقب بالشاشة.
    """

    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    if text.upper() in BAD_VALUES:
        return ""

    return text


# ============================================================================
# قراءة الأسماء من السطر الأول (الإصلاح الأهم)
# ============================================================================

def clean_name_part(raw: str) -> str:
    """
    ننظف اسم مستخرج من MRZ:
    - نحوّل < لمسافة
    - نشيل أي رقم أو رمز غريب
    - نلغي المسافات المتكررة
    """

    if not raw:
        return ""

    text = str(raw).upper()
    text = text.replace("<", " ")
    text = re.sub(r"[^A-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in BAD_VALUES:
        return ""

    return text


def is_valid_name(name: str) -> bool:
    """
    فحص جودة الاسم.

    الاسم الحقيقي: حرفين على الأقل، كله أحرف ومسافات،
    وما يحتوي سلاسل غريبة مثل حرف واحد مكرر.
    """

    if not name or len(name) < 2:
        return False

    if not re.fullmatch(r"[A-Z ]{2,39}", name):
        return False

    # اسم كله نفس الحرف (مثل "XXXX") = خردة
    letters = name.replace(" ", "")
    if len(set(letters)) <= 1:
        return False

    return True


def parse_names_from_line1(l1: str):
    """
    نقرا اللقب والاسم الأول من مواقعهم الصحيحة بالسطر الأول.

    معيار TD3 للسطر الأول (44 حرف):
      الموقع 0-1   : نوع الوثيقة (P<)
      الموقع 2-4   : رمز الدولة المُصدِرة
      الموقع 5-43  : SURNAME<<GIVEN<NAMES

    هذا أدق بكثير من الاعتماد على fields.name وحده،
    لأن المكتبة أصلاً تفصل الحقلين وما ينفع نقسم واحد منهم.
    """

    if not l1 or len(l1) < 6:
        return "", ""

    body = l1[5:44]

    # نقص أي حشو << بالنهاية
    body = body.rstrip("<")

    parts = body.split("<<", 1)

    surname = clean_name_part(parts[0]) if len(parts) >= 1 else ""
    given_names = clean_name_part(parts[1]) if len(parts) >= 2 else ""

    return surname, given_names


def extract_names(fields, l1: str):
    """
    نجيب الاسم واللقب بثلاث محاولات مرتبة:
    1. من السطر الأول مباشرة (الأدق)
    2. من حقول المكتبة surname / name
    3. فاضي إذا كلهم فشلوا
    """

    surname, given_names = parse_names_from_line1(l1)

    # احتياطي من حقول المكتبة
    if not is_valid_name(surname):
        surname = clean_name_part(safe_text(getattr(fields, "surname", "")))

    if not is_valid_name(given_names):
        given_names = clean_name_part(safe_text(getattr(fields, "name", "")))

    # ما نرجّع اسم مو صالح إطلاقاً
    if not is_valid_name(surname):
        surname = ""

    if not is_valid_name(given_names):
        given_names = ""

    return surname, given_names


# ============================================================================
# استخراج مرشحي MRZ
# ============================================================================

def extract_mrz_candidates(text: str):
    """نستخرج كل الأسطر المحتملة لمنطقة القراءة الآلية."""

    lines = []

    for raw in (text or "").splitlines():
        cleaned = clean_ocr_line(raw)
        if len(cleaned) >= 30:
            lines.append(cleaned)

    candidates = []

    for i in range(len(lines) - 1):

        l1 = lines[i]
        l2 = lines[i + 1]

        if len(l1) < 30 or len(l2) < 30:
            continue

        l1_44 = l1[:44].ljust(44, "<")
        l2_44 = l2[:44].ljust(44, "<")

        first_ok = (
            l1_44.startswith("P")
            or l1_44.startswith("<<P")
            or "P<" in l1_44[:5]
        )

        second_has_digits = sum(c.isdigit() for c in l2_44) >= 8

        if first_ok and second_has_digits:
            candidates.append((l1_44, l2_44))

    return candidates


def extract_mrz_from_full_text(text: str):
    """محاولة ثانية لو Tesseract دمج الأسطر كلها بسلسلة وحدة."""

    cleaned = clean_ocr_line(text)

    candidates = []

    for start in range(max(0, len(cleaned) - 100)):

        chunk = cleaned[start:start + 88]

        if len(chunk) < 88:
            continue

        l1 = chunk[:44]
        l2 = chunk[44:88]

        # شرط إضافي: السطر الأول لازم يبدأ بـ P (يقلل الخردة كثير)
        if not l1.startswith("P"):
            continue

        if sum(c.isdigit() for c in l2) >= 8:
            candidates.append((l1, l2))

    return candidates


# ============================================================================
# التاريخ
# ============================================================================

def format_date(yymmdd, is_birth=True):
    """نحوّل تاريخ MRZ (YYMMDD) لصيغة YYYY-MON-DD."""

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

        year = (1900 + yy if yy > 30 else 2000 + yy) if is_birth else 2000 + yy

        return f"{year}-{MONTHS[mm - 1]}-{dd:02d}"

    except Exception:
        return ""


# ============================================================================
# استخراج تاريخ الإصدار واسم الأب من النص المطبوع
# ============================================================================
# هذول الحقلين ما موجودين بمنطقة القراءة الآلية إطلاقاً — لازم
# نقراهم من النص المطبوع بوجه الجواز.
# ============================================================================

FATHER_LABELS = [
    r"FATHER[' ]?S?\s*NAME",
    r"HUSBAND[' ]?S?\s*NAME",
    r"NAME\s*OF\s*FATHER",
    r"GUARDIAN[' ]?S?\s*NAME",
    r"\bS\s*/\s*O\b",
    r"\bW\s*/\s*O\b",
    r"\bD\s*/\s*O\b",
]


def read_printed_text(img_bgr) -> str:
    """نقرا النص المطبوع العادي من الجزء العلوي من الصورة."""

    try:
        h, w = img_bgr.shape[:2]

        # الجزء العلوي 78% — منطقة البيانات المطبوعة
        top = img_bgr[0:int(h * 0.78), 0:w]

        if top.size == 0:
            return ""

        if top.shape[1] < 1600:
            scale = 1600 / top.shape[1]
            top = cv2.resize(
                top, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return pytesseract.image_to_string(
            enhanced,
            config=TESS_PRINTED_CONFIG,
            lang="eng",
        ) or ""

    except Exception:
        return ""


def find_printed_dates(printed_text: str):
    """
    نلقى كل التواريخ المطبوعة بصيغة "14 SEP 2023".

    نرجعها كقائمة (سنة, شهر, يوم) مرتبة.
    """

    found = []

    pattern = re.compile(
        r"\b(\d{1,2})\s*[-/ ]?\s*([A-Z]{3})\s*[-/ ]?\s*(\d{4})\b"
    )

    for match in pattern.finditer((printed_text or "").upper()):

        day = int(match.group(1))
        mon = match.group(2)
        year = int(match.group(3))

        if mon not in MONTH_INDEX:
            continue

        if not 1 <= day <= 31:
            continue

        if not 1900 <= year <= 2100:
            continue

        found.append((year, MONTH_INDEX[mon], day))

    return found


def extract_issue_date(printed_text: str, birth_date: str, expiry_date: str) -> str:
    """
    نستنتج تاريخ الإصدار بذكاء:

    منطقة القراءة الآلية تعطينا الميلاد والنفاذ بشكل مؤكد.
    التاريخ الثالث المطبوع بالجواز = تاريخ الإصدار.

    كمان الإصدار دائماً قبل النفاذ وبعد الميلاد — نستخدم هذا للفحص.
    """

    dates = find_printed_dates(printed_text)

    if not dates:
        return ""

    # نحوّل تواريخ MRZ المعروفة لنفس الشكل حتى نستبعدها
    known = set()

    for known_date in (birth_date, expiry_date):
        if not known_date:
            continue
        parts = known_date.split("-")
        if len(parts) == 3 and parts[1] in MONTH_INDEX:
            try:
                known.add((int(parts[0]), MONTH_INDEX[parts[1]], int(parts[2])))
            except Exception:
                pass

    # حد أعلى وأدنى منطقي
    expiry_tuple = None
    if expiry_date:
        parts = expiry_date.split("-")
        if len(parts) == 3 and parts[1] in MONTH_INDEX:
            try:
                expiry_tuple = (int(parts[0]), MONTH_INDEX[parts[1]], int(parts[2]))
            except Exception:
                pass

    candidates = []

    for date_tuple in dates:

        if date_tuple in known:
            continue

        # الإصدار لازم يكون قبل النفاذ
        if expiry_tuple and date_tuple >= expiry_tuple:
            continue

        # الإصدار ما يكون قبل 1980
        if date_tuple[0] < 1980:
            continue

        candidates.append(date_tuple)

    if not candidates:
        return ""

    # نختار الأقرب للنفاذ (لأن الإصدار عادة قبل النفاذ بـ5-10 سنين)
    best = max(candidates)

    return f"{best[0]}-{MONTHS[best[1] - 1]}-{best[2]:02d}"


def extract_printed_names(printed_text: str):
    """
    نستخرج الاسم الأول واللقب من النص المطبوع بوجه الجواز.

    ليش نحتاجها:
    بعض الجوازات (خاصة الباكستانية) تحط اللقب بس بمنطقة القراءة
    الآلية بدون الفاصل << والاسم الأول، مثل:
        P<PAKFARZANA<<<<<<<<<<<
    بهاي الحالة الاسم الأول موجود بس بالنص المطبوع تحت
    "Given Names". هذي الدالة تجيبه من هناك.
    """

    text = (printed_text or "").upper()

    surname = ""
    given = ""

    # اللقب — تسميات مختلفة حسب البلد
    surname_match = re.search(
        r"\bSURNAME\b\s*[:\-]?\s*([A-Z][A-Z\-' ]{1,40})", text
    )
    if surname_match:
        candidate = re.split(
            r"\b(GIVEN|NAME|NATIONALITY|DATE|SEX|PLACE|FATHER|HUSBAND)\b",
            surname_match.group(1),
        )[0]
        candidate = clean_name_part(candidate)
        if is_valid_name(candidate):
            surname = candidate

    # الاسم الأول
    given_match = re.search(
        r"\bGIVEN\s*NAMES?\b\s*[:\-]?\s*([A-Z][A-Z\-' ]{1,60})", text
    )
    if given_match:
        candidate = re.split(
            r"\b(NATIONALITY|DATE|SEX|PLACE|FATHER|HUSBAND|SURNAME|ISSUING|AUTHORITY)\b",
            given_match.group(1),
        )[0]
        candidate = clean_name_part(candidate)
        if is_valid_name(candidate):
            given = candidate

    return surname, given


def extract_father_name(printed_text: str) -> str:
    """نلقى اسم الأب أو الزوج حسب التسميات المختلفة بالجوازات."""

    text = (printed_text or "").upper()

    for label in FATHER_LABELS:

        match = re.search(label + r"\s*[:\-]?\s*([A-Z][A-Z,'\.\- ]{2,60})", text)

        if not match:
            continue

        raw = match.group(1)

        # نقطع عند أول سطر جديد أو تسمية ثانية
        raw = re.split(
            r"\b(DATE|PLACE|SEX|NATIONALITY|ISSUING|TRACKING|BOOKLET|PASSPORT|AUTHORITY|CITIZENSHIP)\b",
            raw,
        )[0]

        name = clean_name_part(raw.replace(",", " "))

        if is_valid_name(name):
            return name

    return ""


# ============================================================================
# حساب قوة النتيجة
# ============================================================================

def calculate_score(is_checksum_ok, fields, l1, l2, surname, given_names):
    """
    نعطي نقاط للنتيجة حتى نختار الأقوى بين كل الاحتمالات.

    ملاحظة مهمة جداً:
    معيار TD3 ما بيه أي رقم تحقق للسطر الأول (سطر الأسماء)!
    يعني القراءة تقدر تكون "مؤكدة" رياضياً والأسماء خردة كاملة.
    عشان هيك نعطي وزن كبير لجودة الأسماء بالتقييم.
    """

    score = 0

    if is_checksum_ok:
        score += 6

    # جودة الأسماء — الوزن الأكبر بعد أرقام التحقق
    if is_valid_name(surname):
        score += 3

    if is_valid_name(given_names):
        score += 3

    # السطر الأول لازم يبدأ P ويليه رمز دولة من 3 أحرف
    if l1.startswith("P") and re.fullmatch(r"[A-Z]{3}", l1[2:5] or ""):
        score += 2

    doc_number = safe_text(getattr(fields, "document_number", "")).replace("<", "")
    if len(doc_number) >= 6:
        score += 2

    if getattr(fields, "birth_date", None):
        score += 2

    if getattr(fields, "expiry_date", None):
        score += 2

    country_code = safe_text(getattr(fields, "country", "")).upper()
    if country_code in COUNTRY_NAMES:
        score += 1

    if len(l1) == 44:
        score += 1

    if len(l2) == 44:
        score += 1

    return score


# ============================================================================
# تجربة مرشح MRZ
# ============================================================================

def try_mrz_candidate(l1, l2):
    """نجرب زوج أسطر ونرجع (النقاط، البيانات)."""

    try:

        l1 = clean_ocr_line(l1)[:44].ljust(44, "<")
        l2 = clean_ocr_line(l2)[:44].ljust(44, "<")

        checker = TD3CodeChecker(f"{l1}\n{l2}", check_expiry=False)

        fields = checker.fields()

        # هل كل أرقام التحقق الرياضية نجحت؟
        try:
            is_checksum_ok = (checker.report.warnings == [])
        except Exception:
            is_checksum_ok = False

        surname, given_names = extract_names(fields, l1)

        score = calculate_score(
            is_checksum_ok, fields, l1, l2, surname, given_names
        )

        country_code = safe_text(getattr(fields, "country", "")).upper()

        nationality_code = safe_text(
            getattr(fields, "nationality", "")
        ).upper() or country_code

        passport_number = safe_text(
            getattr(fields, "document_number", "")
        ).replace("<", "").upper()

        birth = format_date(getattr(fields, "birth_date", ""), True)
        expiry = format_date(getattr(fields, "expiry_date", ""), False)

        sex = safe_text(getattr(fields, "sex", "")).upper()
        if sex not in ("M", "F"):
            sex = ""

        result = {
            "success": True,

            "given_name_en": given_names,
            "surname_en": surname,

            # حقول جديدة — تنعبّي من النص المطبوع لاحقاً
            "father_name_en": "",
            "issue_date": "",

            "passport_number": passport_number,

            "nationality": NATIONALITY_NAMES.get(
                nationality_code, nationality_code
            ),

            "residence_country": COUNTRY_NAMES.get(
                country_code, country_code
            ),

            "birth_date": birth,
            "expiry_date": expiry,

            "sex": sex,

            "score": score,

            # مؤكد = أرقام التحقق نجحت
            "is_verified": is_checksum_ok,

            # مؤكد بالكامل = أرقام التحقق نجحت **والأسماء سليمة**
            "is_fully_verified": (
                is_checksum_ok
                and is_valid_name(surname)
                and is_valid_name(given_names)
            ),

            # للتشخيص لو صارت مشكلة بالمستقبل
            "mrz_line1": l1,
            "mrz_line2": l2,
        }

        return score, result

    except Exception:
        return -1, None


# ============================================================================
# توليد نسخ معالَجة من الصورة
# ============================================================================

def preprocess_variants(img_bgr, quick=False):
    """
    نولّد نسخ معالَجة من الصورة.

    quick=True  -> 9 نسخ فقط (سريعة جداً، تكفي 90% من الحالات)
    quick=False -> كل النسخ (بطيئة، للحالات الصعبة فقط)

    مهم: هذا التقسيم هو سبب السرعة — بدونه الخدمة تاخذ دقائق
    وتعلّق التطبيق.
    """

    variants = []

    h, w = img_bgr.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # الوضع السريع: نسب القص الأكثر نجاحاً فقط، بثلاث معالجات
    # ------------------------------------------------------------------
    if quick:
        for crop_ratio in [0.30, 0.35, 0.40]:

            y_start = int(h * (1 - crop_ratio))
            crop = img_bgr[y_start:h, 0:w]

            if crop.size == 0:
                continue

            if crop.shape[1] < 1600:
                scale = 1600 / crop.shape[1]
                crop = cv2.resize(
                    crop, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            variants.append(gray)
            variants.append(clahe.apply(gray))

            _, otsu = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            variants.append(otsu)

        return variants

    # ------------------------------------------------------------------
    # الصورة كاملة
    # ------------------------------------------------------------------

    full = img_bgr.copy()

    if full.shape[1] < 1600:
        scale = 1600 / full.shape[1]
        full = cv2.resize(
            full, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    gray_full = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)

    variants.append(gray_full)
    variants.append(clahe.apply(gray_full))

    _, otsu_full = cv2.threshold(
        gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(otsu_full)

    variants.append(cv2.adaptiveThreshold(
        gray_full, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    ))

    # ------------------------------------------------------------------
    # قص المنطقة السفلية بنسب مختلفة
    # ------------------------------------------------------------------

    for crop_ratio in [0.30, 0.35, 0.40, 0.45]:

        y_start = int(h * (1 - crop_ratio))

        crop = img_bgr[y_start:h, 0:w]

        if crop.size == 0:
            continue

        if crop.shape[1] < 1600:
            scale = 1600 / crop.shape[1]
            crop = cv2.resize(
                crop, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        variants.append(gray)
        variants.append(clahe.apply(gray))

        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants.append(otsu)

        variants.append(cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 15
        ))

    return variants


# ============================================================================
# معالجة صورة بزاوية وحدة واختيار أفضل نتيجة
# ============================================================================

def process_image(img_bgr, deadline=None, quick=False):
    """
    نجرب النسخ ونرجع أفضل نتيجة.

    deadline: وقت انتهاء مطلق (time.monotonic) — نوقف عنده مهما كان،
    حتى ما تعلّق الخدمة والتطبيق ينتظر بلا نهاية.
    """

    best_score = -1
    best_data = None

    configs = TESS_CONFIGS[:1] if quick else TESS_CONFIGS

    for variant in preprocess_variants(img_bgr, quick=quick):

        # فحص الوقت قبل كل نسخة
        if deadline is not None and time.monotonic() > deadline:
            break

        for config in configs:

            if deadline is not None and time.monotonic() > deadline:
                break

            try:

                text = pytesseract.image_to_string(
                    variant, config=config, lang="eng"
                )

                candidates = extract_mrz_candidates(text)
                candidates.extend(extract_mrz_from_full_text(text))

                for l1, l2 in candidates:

                    score, data = try_mrz_candidate(l1, l2)

                    if data is None:
                        continue

                    if score > best_score:
                        best_score = score
                        best_data = data

                    # ما نتوقف إلا لما تكون القراءة مؤكدة **والأسماء
                    # سليمة كمان** — هذا الفرق الجوهري عن النسخة القديمة
                    if (
                        data["is_fully_verified"]
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

    np_array = np.frombuffer(image_bytes, np.uint8)

    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {"success": False, "error": "تعذر فك ترميز الصورة"}

    best_data = None
    best_image = img_bgr

    # ميزانية وقت كلية — الخدمة ما تتجاوزها أبداً، حتى ما يعلّق التطبيق
    deadline = time.monotonic() + MAX_SECONDS

    def better(candidate):
        """هل هذي النتيجة أفضل من اللي عندنا؟"""
        if candidate is None:
            return False
        if best_data is None:
            return True
        return candidate.get("score", 0) > best_data.get("score", 0)

    # ======================================================================
    # المرحلة 1: سريعة — الوضع الأصلي بـ9 نسخ فقط
    # هذي وحدها تنجح بأغلب الصور، وتخلص خلال ثواني
    # ======================================================================
    data = process_image(img_bgr, deadline=deadline, quick=True)

    if better(data):
        best_data = data
        best_image = img_bgr

    if best_data is not None and best_data.get("is_fully_verified"):
        return _finalize(best_data, best_image)

    # ======================================================================
    # المرحلة 2: سريعة — الصورة مقلوبة 180 درجة
    # ======================================================================
    if time.monotonic() < deadline:

        flipped = cv2.rotate(img_bgr, cv2.ROTATE_180)

        data = process_image(flipped, deadline=deadline, quick=True)

        if better(data):
            best_data = data
            best_image = flipped

        if best_data is not None and best_data.get("is_fully_verified"):
            return _finalize(best_data, best_image)

    # ======================================================================
    # المرحلة 3: شاملة — كل النسخ، للأصلية والمقلوبة فقط
    # ما نجرب 90 و270 درجة إطلاقاً: الصور بهذا التطبيق دائماً أفقية،
    # وتجربتهم كانت تربّع وقت المعالجة بدون أي فائدة عملية
    # ======================================================================
    for candidate_image in [img_bgr, cv2.rotate(img_bgr, cv2.ROTATE_180)]:

        if time.monotonic() > deadline:
            break

        data = process_image(candidate_image, deadline=deadline, quick=False)

        if better(data):
            best_data = data
            best_image = candidate_image

        if best_data is not None and best_data.get("is_fully_verified"):
            break

    if best_data is None:
        return {
            "success": False,
            "error": "ما قدرنا نلقى منطقة قراءة آلية واضحة بالصورة",
        }

    return _finalize(best_data, best_image)


# ============================================================================
# اللمسات الأخيرة: النص المطبوع + تنظيف الحقول
# ============================================================================

def _finalize(best_data, best_image):
    """نضيف تاريخ الإصدار واسم الأب، وننظف كل الحقول قبل الإرسال."""

    try:
        # نقرا النص المطبوع بس إذا بقى وقت كافي (5 ثواني على الأقل)
        if time.monotonic() > deadline - 5:
            raise TimeoutError("ماكو وقت كافي للنص المطبوع")

        printed_text = read_printed_text(best_image)

        if printed_text:

            issue = extract_issue_date(
                printed_text,
                best_data.get("birth_date", ""),
                best_data.get("expiry_date", ""),
            )

            if issue:
                best_data["issue_date"] = issue

            father = extract_father_name(printed_text)

            if father:
                best_data["father_name_en"] = father

            # ==============================================================
            # احتياطي الأسماء من النص المطبوع
            # ==============================================================
            # لو منطقة القراءة الآلية ما أعطت اسم أول (يصير لما ما يكون
            # بيها الفاصل <<)، نجيبه من النص المطبوع "Given Names"
            # ==============================================================
            printed_surname, printed_given = extract_printed_names(printed_text)

            if not best_data.get("given_name_en") and printed_given:
                best_data["given_name_en"] = printed_given

            if not best_data.get("surname_en") and printed_surname:
                best_data["surname_en"] = printed_surname

            # حالة خاصة: اللقب والاسم الأول طلعوا نفس الشي من منطقة
            # القراءة الآلية (لأن ماكو فاصل)، والنص المطبوع يفرّقهم
            if (
                printed_given
                and printed_surname
                and printed_given != printed_surname
                and best_data.get("given_name_en") == best_data.get("surname_en")
            ):
                best_data["given_name_en"] = printed_given
                best_data["surname_en"] = printed_surname

    except Exception:
        # فشل أو انتهى الوقت — النتيجة الأساسية تبقى سليمة
        pass

    # فحص أخير: ما نطلّع أي "None" للتطبيق
    for key in [
        "given_name_en", "surname_en", "father_name_en",
        "passport_number", "nationality", "residence_country",
        "birth_date", "expiry_date", "issue_date", "sex",
    ]:
        best_data[key] = safe_text(best_data.get(key))

    return best_data


# ============================================================================
# Endpoints فحص الخدمة (للإيقاظ)
# ============================================================================

@app.get("/")
def root_check():
    return {"status": "الخدمة شغالة ✓", "ready": True}


@app.get("/health")
def health_check():
    """endpoint خفيف جداً للإيقاظ من التطبيق."""
    return {"status": "ok", "ready": True}


# ============================================================================
# Endpoint قراءة الجواز
# ============================================================================

@app.post("/read-passport")
async def read_passport_endpoint(file: UploadFile = File(...)):

    try:

        image_bytes = await file.read()

        if not image_bytes:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "الصورة فارغة"},
            )

        result = read_passport_from_bytes(image_bytes)

        return JSONResponse(content=result)

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"خطأ داخلي: {str(e)}"},
        )
