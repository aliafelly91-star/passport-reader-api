"""
main.py — خدمة قراءة الجوازات (FastAPI) — النسخة المصححة
==========================================================

⚠⚠ الإصلاح الأهم بهذي النسخة: اسم الأب وتاريخ الإصدار ما كانوا
    يشتغلون **إطلاقاً** — ولا مرة وحدة.

السبب: دالة _finalize كانت تستخدم المتغير `deadline`، بس هذا
المتغير محلي داخل read_passport_from_bytes وما انمرر لها:

    def _finalize(best_data, best_image):
        try:
            if time.monotonic() > deadline - 5:   ← deadline مو معرّف!
                ...
            printed_text = read_printed_text(best_image)
            ... استخراج اسم الأب وتاريخ الإصدار ...
        except Exception:
            pass                                   ← يبلع الخطأ بصمت

بايثون يرمي NameError بأول سطر، و`except Exception` يبلعه بصمت
تام، فينقفز **البلوك كله**. يعني: ولا صورة انقرا منها اسم الأب،
ولا تاريخ إصدار، ولا احتياطي الأسماء المطبوعة. والخدمة ترجّع
النتيجة "ناجحة" بدون أي إشارة إن شي فشل.

الحل: نمرر deadline كمعامل صريح، ونسجّل سبب أي فشل بدل ما نبلعه.

--------------------------------------------------------------------------
باقي الإصلاحات بهذي النسخة:

2. الأرقام داخل مقطع الأسماء بمنطقة القراءة الآلية:
   "P<PAKBUGHIO" تنقرا "P<PAKBUGHI0" (صفر بدل O)، وبعدها التنظيف
   يحذف الرقم فيطلع اللقب "BUGHI" ناقص حرف — وأرقام التحقق تنجح
   فتطلع القراءة "مؤكدة" واللقب غلط! بمعيار ICAO ما ينفع يجي أي
   رقم بمقطع الأسماء، فأي رقم هناك = خطأ قراءة مؤكد.

3. حرف الحشو "<" ينقرا "X" فيطلع "KAZMIXXSYEDXALI" — نصلّحه
   بدون ما نخرّب أسماء فيها X حقيقي (ALEX، MAX).

4. اسم الأب بنمط "اللقب، الأسماء": الجواز الباكستاني يطبع اسم
   الأب بصيغة "BUGHIO, MAZHAR HUSSAIN" — نفس لقب صاحب الجواز.
   هذي تشتغل حتى لو تسمية "Father Name" ما انقرت أصلاً (وهي
   مطبوعة بخط رمادي صغير وTesseract يضيّعها كثير).

5. حارس اسم الأب: كان يطلع "PAK" (رمز الدولة) و"COUNTY COD"
   (تسمية Country Code منقرية غلط) كأسماء أب.

6. تاريخ الإصدار: نستخدم إن الجواز صلاحيته 5 أو 10 سنين، فنختار
   المرشح الأقرب لـ(النفاذ − 5 سنين) أو (النفاذ − 10 سنين).

7. النص المطبوع ينقرا بقراءتين (psm 6 و psm 4) وكل وحدة **تنحلل
   لحالها**. ما ندمج النصين أبداً: المحلل يشتغل بمنطق "التسمية
   بسطر والقيمة بالسطر اللي بعده"، فدمج النصين يخلي آخر سطر
   بالقراءة الأولى جار أول سطر بالقراءة الثانية — وهما من مكانين
   مختلفين تماماً بالصورة.

8. الجوازات باسم واحد (مثل P<PAKFARZANA<<<) ما بيها اسم أول
   أصلاً — كانت الخدمة تضل تجرب كل الاحتمالات لين ينتهي الوقت
   بلا فايدة. الحين نتعرّف عليها ونخلص بسرعة.

- نفس Endpoint: /read-passport
- نفس أسماء الحقول اللي ينتظرها Flutter (أضفنا حقول بس، ما حذفنا شي)
"""

import re
import time
from datetime import date

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

# إعدادات قراءة النص المطبوع العادي (لتاريخ الإصدار واسم الأب).
# psm 6 = كتلة نص موحّدة، psm 4 = أعمدة بأحجام مختلفة.
# الجواز مقسّم أعمدة، فأحياناً psm 4 يرتّب التسميات مع قيمها أحسن
TESS_PRINTED_CONFIGS = ["--oem 1 --psm 6", "--oem 1 --psm 4"]

# ============================================================================
# سقف الوقت
# ============================================================================
MAX_SECONDS = 45

# نحجز هذي الثواني بالآخر للنص المطبوع (اسم الأب + تاريخ الإصدار)
PRINTED_TEXT_BUDGET = 8


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
MONTH_INDEX["SEPT"] = 9  # خطأ قراءة شائع

# قيم خردة ممكن ترجع من المكتبة أو من OCR — نرفضها دائماً
BAD_VALUES = {"NONE", "NULL", "NAN", "N/A", "NA", "-", "--"}

# ============================================================================
# ⚠ حارس اسم الأب — يرفض القيم اللي مو أسماء بشر
# ============================================================================
# حالات واقعية انمسكت من التطبيق: طلع اسم الأب "PAK" (رمز الدولة)،
# وطلع "COUNTY COD" (تسمية Country Code منقرية غلط). هذي كلمات من
# هيكل الجواز نفسه، مو أسماء بشر
DOCUMENT_WORDS = [
    "COUNTRY", "COUNTY", "CODE", "COD", "NUMBER", "AUTHORITY",
    "BOOKLET", "TRACKING", "CITIZENSHIP", "REPUBLIC", "ISLAMIC",
    "PASSPORT", "NATIONALITY", "PLACE", "BIRTH", "ISSUE", "EXPIRY",
    "SURNAME", "HOLDER", "SIGNATURE", "OBSERVATIONS", "TYPE",
    "GIVEN", "FATHER", "HUSBAND", "GUARDIAN",
]


# ============================================================================
# تنظيف نص Tesseract
# ============================================================================

def clean_ocr_line(line: str) -> str:
    """ننظف السطر ونبقي فقط أحرف MRZ المسموحة."""

    line = (line or "").upper().strip()

    # تصحيح رموز يخلط بيها Tesseract مع الرمز <
    replacements = {
        "«": "<", "‹": "<", "≤": "<", "—": "<",
        "_": "<", "|": "<", "〈": "<", " ": "",
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
# ⚠ إصلاح الأرقام وحروف الحشو داخل مقطع الأسماء
# ============================================================================

# بخط OCR-B المستخدم بالجوازات، هذي الأرقام تشبه هذي الحروف
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
}


def fix_digits_in_letters(text: str) -> str:
    """
    نحوّل الأرقام لحروف داخل المقاطع اللي المفروض كلها حروف.

    ليش هذا صحيح؟ بمعيار ICAO ما ينفع يجي **أي رقم** بمقطع الأسماء
    ولا برمز الدولة — كلهم حروف بس. فأي رقم يطلع هناك = خطأ قراءة
    مؤكد 100%.

    حالة واقعية: "P<PAKBUGHIO" انقرت "P<PAKBUGHI0"، وبعدها التنظيف
    يحذف الرقم فيطلع اللقب "BUGHI" ناقص حرف. والأسوأ إن أرقام
    التحقق (اللي كلها بالسطر الثاني) تنجح، فتطلع القراءة "مؤكدة"
    واللقب غلط بدون أي تحذير!
    """

    return "".join(DIGIT_TO_LETTER.get(char, char) for char in (text or ""))


def normalize_name_fillers(names_section: str) -> str:
    """
    ⚠ حرف الحشو "<" غالباً ينقرا "X".

    مثال واقعي: "KAZMI<<SYED<ALI<<<<<<<<" تنقرا
                "KAZMIXXSYEDXALIXXXXXXXX"
    فيطلع اللقب "KAZMIXXSYEDXALIXXXXXXXX" بدل "KAZMI".

    المشكلة إن X حرف شرعي بأسماء حقيقية (ALEX، MAX، XAVIER)، فما
    نقدر نحذفه بالجملة. القاعدة:
      • المقطع ما بيه ولا "<" إطلاقاً → كل X حشو أكيد (خانة الأسماء
        طولها 39 خانة ودائماً بيها حشو بالآخر)
      • تتابع 3 حروف X فأكثر → حشو أكيد
      • تتابع حرفين XX → حشو بس لو ماكو فاصل "<<" حقيقي، أو لو كل
        اللي بعده حشو (يعني بذيل السطر)
    """

    section = names_section or ""

    if "X" not in section:
        return section

    # ⚠ لازم نحسب هذول على النص **الأصلي** قبل أي تعديل — لأن خطوة
    # التحويل نفسها تنتج "<<" جديدة، ولو حسبناهم بعدها ينقلب المنطق
    had_no_filler = "<" not in section
    has_real_separator = "<<" in section

    if had_no_filler:
        return section.replace("X", "<")

    # 1. تتابع 3 فأكثر: حشو أكيد
    section = re.sub(r"X{3,}", lambda m: "<" * len(m.group(0)), section)

    # 2. تتابع حرفين: حسب السياق
    work = section

    def replace_pair(match):
        rest = work[match.end():]
        tail_is_filler = (rest == "") or bool(re.fullmatch(r"[<X]*", rest))
        if (not has_real_separator) or tail_is_filler:
            return "<<"
        return match.group(0)

    section = re.sub(r"X{2}", replace_pair, work)

    # 3. X ملتصق بحشو بآخر السطر: "…ALI<XX" أو "…ALI<X"
    tail = re.search(r"<(X+)$", section)
    if tail:
        section = section[:tail.start() + 1] + "<" * len(tail.group(1))

    return section


def normalize_mrz_line1(l1: str) -> str:
    """
    نصلّح السطر الأول قبل التحليل:
      • "PXPAK" → "P<PAK"
      • رمز الدولة (خانات 2-4) حروف بس
      • مقطع الأسماء: أرقام → حروف، وبعدين X → حشو

    آمن تماماً: أرقام التحقق بمعيار TD3 كلها محسوبة من **السطر
    الثاني** بس، فتعديل السطر الأول ما يأثر عليها إطلاقاً.
    """

    if not l1 or len(l1) <= 5:
        return l1 or ""

    head = l1[:5]

    if len(head) >= 2 and head[1] == "X":
        head = head[0] + "<" + head[2:]

    head = head[:2] + fix_digits_in_letters(head[2:])

    names = fix_digits_in_letters(l1[5:])

    return head + normalize_name_fillers(names)


# ============================================================================
# قراءة الأسماء من السطر الأول
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


def is_plausible_person_name(name: str) -> bool:
    """
    حارس أقوى من is_valid_name — نستخدمه لاسم الأب تحديداً.

    يرفض رموز الدول وكلمات هيكل الجواز، لأن هذي كانت تطلع كأسماء
    أب بالتطبيق ("PAK"، "COUNTY COD")
    """

    if not name:
        return False

    clean = name.strip().upper()

    if len(clean) < 3:
        return False

    # رمز دولة مثل PAK / IRQ
    if clean.replace(" ", "") in COUNTRY_NAMES:
        return False

    for word in DOCUMENT_WORDS:
        if word in clean:
            return False

    return is_valid_name(clean)


def parse_names_from_line1(l1: str):
    """
    نقرا اللقب والاسم الأول من مواقعهم الصحيحة بالسطر الأول.

    معيار TD3 للسطر الأول (44 حرف):
      الموقع 0-1   : نوع الوثيقة (P<)
      الموقع 2-4   : رمز الدولة المُصدِرة
      الموقع 5-43  : SURNAME<<GIVEN<NAMES

    نرجّع كمان has_separator: هل فيه فاصل "<<" أصلاً؟ لأن بعض
    الجوازات باسم واحد بس (P<PAKFARZANA<<<) وما بيها اسم أول
    إطلاقاً — وهذا مو خطأ قراءة، هيك الجواز فعلاً
    """

    if not l1 or len(l1) < 6:
        return "", "", False

    body = l1[5:44]

    has_separator = "<<" in body.rstrip("<")

    # نقص أي حشو << بالنهاية
    body = body.rstrip("<")

    parts = body.split("<<", 1)

    surname = clean_name_part(parts[0]) if len(parts) >= 1 else ""
    given_names = clean_name_part(parts[1]) if len(parts) >= 2 else ""

    return surname, given_names, has_separator


def extract_names(fields, l1: str):
    """
    نجيب الاسم واللقب بثلاث محاولات مرتبة:
    1. من السطر الأول مباشرة (الأدق)
    2. من حقول المكتبة surname / name
    3. فاضي إذا كلهم فشلوا
    """

    surname, given_names, has_separator = parse_names_from_line1(l1)

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

    return surname, given_names, has_separator


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
            # حالة الحشو المقروء X: "PXPAK…"
            or bool(re.match(r"[A-Z]X[A-Z]{3}", l1_44))
        )

        second_has_digits = sum(c.isdigit() for c in l2_44) >= 8

        if first_ok and second_has_digits:
            candidates.append((l1_44, l2_44))

    return candidates


def extract_mrz_from_full_text(text: str):
    """محاولة ثانية لو Tesseract دمج الأسطر كلها بسلسلة وحدة."""

    cleaned = clean_ocr_line(text)

    candidates = []

    # ⚠ كان range(max(0, len(cleaned) - 100)) — يعني آخر 12 موقع
    # محتمل ما تنفحص أبداً. الصح: نفحص لين آخر نافذة كاملة بطول 88
    for start in range(max(0, len(cleaned) - 87)):

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


def parse_formatted_date(text: str):
    """نحوّل "2023-SEP-14" لـdate — نستخدمها بمقارنات تاريخ الإصدار."""

    if not text:
        return None

    parts = str(text).split("-")

    if len(parts) != 3 or parts[1] not in MONTH_INDEX:
        return None

    try:
        return date(int(parts[0]), MONTH_INDEX[parts[1]], int(parts[2]))
    except Exception:
        return None


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


def read_printed_texts(img_bgr):
    """
    نقرا النص المطبوع من الجزء العلوي من الصورة.

    ⚠ نرجّع **قائمة نصوص منفصلة**، ما ندمجهم بنص واحد أبداً!

    ليش؟ لأن المحلل يشتغل بمنطق "التسمية بسطر والقيمة بالسطر اللي
    بعده". لو لصقنا نصين، آخر سطر بالقراءة الأولى يصير جار أول سطر
    بالقراءة الثانية — وهما من مكانين مختلفين تماماً بالصورة.
    (هذا بالضبط اللي طلّع اسم أب "COUNTY COD" بالتطبيق)
    """

    texts = []

    try:
        h, w = img_bgr.shape[:2]

        # الجزء العلوي 78% — منطقة البيانات المطبوعة
        top = img_bgr[0:int(h * 0.78), 0:w]

        if top.size == 0:
            return texts

        if top.shape[1] < 1600:
            scale = 1600 / top.shape[1]
            top = cv2.resize(
                top, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        for config in TESS_PRINTED_CONFIGS:
            try:
                text = pytesseract.image_to_string(
                    enhanced, config=config, lang="eng"
                )
                if text and text.strip():
                    texts.append(text)
            except Exception:
                continue

    except Exception:
        pass

    return texts


def find_printed_dates(printed_text: str):
    """
    نلقى كل التواريخ المطبوعة.

    ندعم: "14 SEP 2023" و "SEP 14 2023" و "14/09/2023" و "2023-09-14"
    نرجعها كقائمة كائنات date.
    """

    found = []
    text = (printed_text or "").upper()

    def add(year, month, day):
        if not 1 <= month <= 12:
            return
        if not 1 <= day <= 31:
            return
        if not 1900 <= year <= 2100:
            return
        try:
            value = date(year, month, day)
        except Exception:
            return
        if value not in found:
            found.append(value)

    # 14 SEP 2023
    for match in re.finditer(
        r"\b(\d{1,2})\s*[-/ ]?\s*([A-Z]{3,4})\s*[-/ ]?\s*(\d{4})\b", text
    ):
        month = MONTH_INDEX.get(match.group(2))
        if month:
            add(int(match.group(3)), month, int(match.group(1)))

    # SEP 14 2023
    for match in re.finditer(
        r"\b([A-Z]{3,4})\s*[-/ ]?\s*(\d{1,2})\s*[-/ ]?\s*(\d{4})\b", text
    ):
        month = MONTH_INDEX.get(match.group(1))
        if month:
            add(int(match.group(3)), month, int(match.group(2)))

    # 2023-09-14
    for match in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text):
        add(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # 14/09/2023
    for match in re.finditer(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b", text):
        add(int(match.group(3)), int(match.group(2)), int(match.group(1)))

    return found


def extract_issue_date(printed_text: str, birth_date: str, expiry_date: str) -> str:
    """
    نستنتج تاريخ الإصدار بذكاء.

    منطقة القراءة الآلية تعطينا الميلاد والنفاذ بشكل مؤكد رياضياً.
    التاريخ الثالث المطبوع بالجواز = تاريخ الإصدار.

    ⚠ التحسين: الجوازات تنصدر بصلاحية **5 أو 10 سنين**. فبدل ما
    ناخذ "الأحدث" وخلاص، نختار المرشح اللي الفرق بينه وبين النفاذ
    أقرب لوحدة من هالمدتين. هذا يمسك الحالات اللي فيها تواريخ
    مطبوعة زايدة (تاريخ إصدار الهوية، تاريخ الطباعة، إلخ)
    """

    dates = find_printed_dates(printed_text)

    if not dates:
        return ""

    birth = parse_formatted_date(birth_date)
    expiry = parse_formatted_date(expiry_date)
    today = date.today()

    candidates = []

    for value in dates:

        if birth and value == birth:
            continue

        if expiry and value == expiry:
            continue

        # الإصدار لازم يكون قبل النفاذ
        if expiry and value >= expiry:
            continue

        # الإصدار لازم يكون بعد الميلاد
        if birth and value <= birth:
            continue

        # الجوازات الحديثة
        if value.year < 1980:
            continue

        # ما يكون بالمستقبل
        if value > today:
            continue

        candidates.append(value)

    if not candidates:
        return ""

    if expiry:

        five_years = 5 * 365
        ten_years = 10 * 365
        tolerance = 120  # هامش يغطي فروقات الأيام والسنة الكبيسة

        def distance_score(value):
            days = (expiry - value).days
            return min(abs(days - five_years), abs(days - ten_years))

        best = min(candidates, key=distance_score)

        if distance_score(best) <= tolerance:
            return f"{best.year}-{MONTHS[best.month - 1]}-{best.day:02d}"

    # غير هيك: الأحدث (الإصدار عادة أقرب تاريخ للنفاذ)
    best = max(candidates)

    return f"{best.year}-{MONTHS[best.month - 1]}-{best.day:02d}"


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


def extract_father_by_surname_pattern(printed_text: str, holder_surname: str) -> str:
    """
    ⚠ أقوى طريقة لاسم الأب بالجواز الباكستاني: نمط "اللقب، الأسماء"

    الجواز الباكستاني يطبع اسم الأب بصيغة SURNAME, GIVEN NAMES
    ونفس لقب صاحب الجواز:
      • صاحب الجواز BUGHIO  →  "BUGHIO, MAZHAR HUSSAIN"
      • صاحب الجواز SHAMSI  →  "SHAMSI, SYED NAYYAR TOUQIR IRTAZA"

    فايدة هالطريقة: تشتغل **حتى لو تسمية "Father Name" ما انقرت
    أصلاً** — وهذي بالضبط الحالة اللي كانت تخلي الحقل فاضي، لأن
    التسمية مطبوعة بخط رمادي صغير وTesseract يضيّعها كثير.

    نرجّع الجزء اللي بعد الفاصلة (أسماء الأب الأولى)، لأن اللقب
    موجود أصلاً بحقل SURNAME
    """

    surname = (holder_surname or "").strip().upper()

    if len(surname) < 3:
        return ""

    for line in (printed_text or "").upper().splitlines():

        line = re.sub(r"\s+", " ", line).strip()

        comma_index = line.find(",")

        if comma_index <= 0:
            continue

        # لازم يكون اللي قبل الفاصلة هو نفسه لقب صاحب الجواز
        if line[:comma_index].strip() != surname:
            continue

        candidate = clean_name_part(line[comma_index + 1:])

        if is_plausible_person_name(candidate):
            return candidate

    return ""


def extract_father_name(printed_text: str, holder_surname: str = "") -> str:
    """نلقى اسم الأب أو الزوج — بالنمط أول، وبعدها بالتسميات."""

    # 1. النمط "اللقب، الأسماء" (الأدق، وما يحتاج تسمية أصلاً)
    by_pattern = extract_father_by_surname_pattern(printed_text, holder_surname)

    if by_pattern:
        return by_pattern

    # 2. البحث بالتسميات
    text = (printed_text or "").upper()

    for label in FATHER_LABELS:

        match = re.search(label + r"\s*[:\-]?\s*([A-Z][A-Z,'\.\- ]{2,60})", text)

        if not match:
            continue

        raw = match.group(1)

        # نقطع عند أول تسمية ثانية
        raw = re.split(
            r"\b(DATE|PLACE|SEX|NATIONALITY|ISSUING|TRACKING|BOOKLET"
            r"|PASSPORT|AUTHORITY|CITIZENSHIP|COUNTRY|COUNTY|CODE|TYPE)\b",
            raw,
        )[0]

        # لو فيه فاصلة ولقب صاحب الجواز قبلها، ناخذ اللي بعدها بس
        surname = (holder_surname or "").strip().upper()
        if "," in raw and surname:
            before, after = raw.split(",", 1)
            if before.strip() == surname:
                raw = after

        name = clean_name_part(raw.replace(",", " "))

        # ⚠ الحارس ضروري: بدونه طلعت قيم مثل "PAK" و"COUNTY COD"
        if is_plausible_person_name(name):
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

        # ⚠ نصلّح الأرقام وحروف الحشو بالسطر الأول قبل أي تحليل.
        # آمن: كل أرقام التحقق محسوبة من السطر الثاني
        l1 = normalize_mrz_line1(l1)[:44].ljust(44, "<")

        checker = TD3CodeChecker(f"{l1}\n{l2}", check_expiry=False)

        fields = checker.fields()

        # هل كل أرقام التحقق الرياضية نجحت؟
        try:
            is_checksum_ok = (checker.report.warnings == [])
        except Exception:
            is_checksum_ok = False

        surname, given_names, has_separator = extract_names(fields, l1)

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

        # ⚠ الجوازات باسم واحد: بعض الجوازات الباكستانية ما بيها
        # فاصل "<<" إطلاقاً (P<PAKFARZANA<<<) — يعني ماكو اسم أول
        # أصلاً، وهذا **مو خطأ قراءة**. بدون هالاستثناء الخدمة تضل
        # تجرب كل الاحتمالات لين ينتهي الوقت بلا أي فايدة
        names_are_complete = is_valid_name(surname) and (
            is_valid_name(given_names) or not has_separator
        )

        result = {
            "success": True,

            "given_name_en": given_names,
            "surname_en": surname,

            # حقول تنعبّي من النص المطبوع بـ_finalize
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
            "is_fully_verified": is_checksum_ok and names_are_complete,

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

    # ⚠ نحجز آخر ثواني للنص المطبوع (اسم الأب + تاريخ الإصدار).
    # بدون هالحجز، البحث عن MRZ ياكل كل الوقت وما يبقى شي لهم
    mrz_deadline = deadline - PRINTED_TEXT_BUDGET

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
    data = process_image(img_bgr, deadline=mrz_deadline, quick=True)

    if better(data):
        best_data = data
        best_image = img_bgr

    if best_data is not None and best_data.get("is_fully_verified"):
        return _finalize(best_data, best_image, deadline)

    # ======================================================================
    # المرحلة 2: سريعة — الصورة مقلوبة 180 درجة
    # ======================================================================
    if time.monotonic() < mrz_deadline:

        flipped = cv2.rotate(img_bgr, cv2.ROTATE_180)

        data = process_image(flipped, deadline=mrz_deadline, quick=True)

        if better(data):
            best_data = data
            best_image = flipped

        if best_data is not None and best_data.get("is_fully_verified"):
            return _finalize(best_data, best_image, deadline)

    # ======================================================================
    # المرحلة 3: شاملة — كل النسخ، للأصلية والمقلوبة فقط
    # ما نجرب 90 و270 درجة إطلاقاً: الصور بهذا التطبيق دائماً أفقية،
    # وتجربتهم كانت تربّع وقت المعالجة بدون أي فائدة عملية
    # ======================================================================
    for candidate_image in [img_bgr, cv2.rotate(img_bgr, cv2.ROTATE_180)]:

        if time.monotonic() > mrz_deadline:
            break

        data = process_image(candidate_image, deadline=mrz_deadline, quick=False)

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

    return _finalize(best_data, best_image, deadline)


# ============================================================================
# اللمسات الأخيرة: النص المطبوع + تنظيف الحقول
# ============================================================================

def _finalize(best_data, best_image, deadline):
    """
    نضيف تاريخ الإصدار واسم الأب، وننظف كل الحقول قبل الإرسال.

    ⚠⚠ هنا كان البگ الأكبر بالخدمة كلها:

        def _finalize(best_data, best_image):     ← بلا deadline
            try:
                if time.monotonic() > deadline - 5:   ← NameError!
                ...
            except Exception:
                pass                                   ← يبلعه بصمت

    `deadline` كان متغير محلي بـread_passport_from_bytes وما انمرر
    لهنا. بايثون يرمي NameError بأول سطر، وexcept يبلعه، فينقفز
    البلوك كله — يعني **ولا صورة** انقرا منها اسم أب ولا تاريخ
    إصدار، والخدمة ترجّع "success: True" بدون أي إشارة إن شي فشل.

    الحل: deadline صار معامل صريح، وأضفنا حقل "printed_text_note"
    يقول شنو صار بالضبط — عشان ما يتكرر فشل صامت مرة ثانية
    """

    best_data["printed_text_note"] = ""

    try:
        # نقرا النص المطبوع بس إذا بقى وقت كافي
        remaining = deadline - time.monotonic()

        if remaining < 3:
            best_data["printed_text_note"] = (
                f"ماكو وقت كافي للنص المطبوع (بقى {remaining:.1f} ثانية)"
            )
        else:
            printed_texts = read_printed_texts(best_image)

            if not printed_texts:
                best_data["printed_text_note"] = "النص المطبوع طلع فاضي"
            else:
                holder_surname = best_data.get("surname_en", "")

                # ⚠ نحلل **كل قراءة لحالها** — ما ندمج النصوص أبداً
                for printed_text in printed_texts:

                    if not best_data.get("issue_date"):
                        issue = extract_issue_date(
                            printed_text,
                            best_data.get("birth_date", ""),
                            best_data.get("expiry_date", ""),
                        )
                        if issue:
                            best_data["issue_date"] = issue

                    if not best_data.get("father_name_en"):
                        father = extract_father_name(printed_text, holder_surname)
                        if father:
                            best_data["father_name_en"] = father

                    # ==================================================
                    # احتياطي الأسماء من النص المطبوع
                    # ==================================================
                    # لو منطقة القراءة الآلية ما أعطت اسم أول (يصير
                    # لما ما يكون بيها الفاصل <<)، نجيبه من النص
                    # المطبوع "Given Names"
                    # ==================================================
                    printed_surname, printed_given = extract_printed_names(
                        printed_text
                    )

                    if not best_data.get("given_name_en") and printed_given:
                        best_data["given_name_en"] = printed_given

                    if not best_data.get("surname_en") and printed_surname:
                        best_data["surname_en"] = printed_surname

                    # حالة خاصة: اللقب والاسم الأول طلعوا نفس الشي من
                    # منطقة القراءة الآلية (لأن ماكو فاصل)، والنص
                    # المطبوع يفرّقهم
                    if (
                        printed_given
                        and printed_surname
                        and printed_given != printed_surname
                        and best_data.get("given_name_en")
                        == best_data.get("surname_en")
                    ):
                        best_data["given_name_en"] = printed_given
                        best_data["surname_en"] = printed_surname

                notes = []
                if not best_data.get("father_name_en"):
                    notes.append("ما لقينا اسم الأب")
                if not best_data.get("issue_date"):
                    notes.append("ما لقينا تاريخ الإصدار")

                best_data["printed_text_note"] = (
                    " و".join(notes) if notes else "تمام"
                )

    except Exception as error:
        # ⚠ ما نبلع الخطأ بصمت بعد اليوم — نسجّله بالنتيجة
        best_data["printed_text_note"] = (
            f"فشل النص المطبوع: {type(error).__name__}: {error}"
        )

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
