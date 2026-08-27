"""
main.py — خدمة قراءة الجوازات (FastAPI)
==========================================================

إصلاحات هذي النسخة (بناءً على حالات فشل حقيقية من التطبيق):

9.  حرف الحشو "<" ينقرا **"K"** مو بس "X". حالة واقعية:
    "ABBAS<<TASSAWAR<K<K<K<KK<KKKK" طلع منها الاسم الأول
    "TASSAWAR K K K KK KKKK". أضفنا _collapse_filler_letter
    تشتغل على X و K سوا، + trim_filler_tokens تشيل أي مقاطع
    ذيلية من حرف واحد مكرر (حزام أمان أخير).

10. اسم الأب كان يطلع **مقلوب**: الجواز يطبع "HUSSAIN, GHULAM"
    (لقب، أسماء) والكود كان يشيل الفاصلة بس فيطلع
    "HUSSAIN GHULAM" بدل "GHULAM HUSSAIN". الحين نعيد الترتيب صح.

11. اسم الأب وتاريخ الإصدار كانوا يفشلون **سوا** بصور كاملة
    وواضحة. السبب: read_printed_texts كانت قراءتين بس على منطقة
    وحدة. لو التسمية ضاعت بهالقراءتين ينتهي كل شي بصمت. الحين:
      • منطقتين (أعلى 78% + الصورة كاملة) × 3 إعدادات
      • قراءة كسولة مع توقف فوري أول ما نلقى الحقلين
      • لو ما لقينا شي، نجرب الصورة مقلوبة 180°
      • مرساة "مكان الولادة" لاسم الأب — تشتغل حتى لو تسمية
        Father/Husband ما انقرت إطلاقاً
      • تاريخ الإصدار: بحث بالتسمية أول، وبعدها الاستنتاج

12. ⚠⚠ الجنس كان يطلع فاضي **دائماً** — ولا مرة وحدة اشتغل!
    مكتبة mrz ترجّع الجنس ككائن مو حرف، فـstr() يعطي شي مثل
    "Sex.MALE" أو "male"، والفحص `if sex not in ("M","F")` يفشل
    كل مرة ويمسح القيمة. الحين نقراه من موقعه المباشر بمعيار
    TD3 (الخانة 20 بالسطر الثاني). هذا مهم جداً لأن منطق الحالة
    بالتطبيق (حملة دار / طباخ) ما يشتغل بدون الجنس.

13. تحذير رقم الجواز: لو الرقم ما يطابق نمط بلده المعروف
    (باكستان: حرفين + 7 أرقام) نحط تنبيه بالنتيجة.

14. ⚠ تكبير النص المطبوع صار 2200 بكسل بدل 1600. التسميات
    (Father Name / Date of Issue) خطها رمادي ورفيع جداً — أصغر
    بكثير من حروف منطقة القراءة الآلية، فتحتاج تكبير أقوى.
    وكمان رفعنا CLAHE لـ2.5 عشان يطلّع الخط الرمادي من خلفية
    الجواز الملوّنة.

- نفس Endpoint: /read-passport
- نفس أسماء الحقول اللي ينتظرها Flutter (أضفنا حقول بس)
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
# psm 6 = كتلة نص موحّدة، psm 4 = أعمدة بأحجام مختلفة،
# psm 11 = نص متفرق (ينفع لما التسميات مبعثرة بأماكن مختلفة).
# الجواز مقسّم أعمدة، فكل إعداد يرتّب التسميات مع قيمها بشكل مختلف
TESS_PRINTED_CONFIGS = ["--oem 1 --psm 6", "--oem 1 --psm 4", "--oem 1 --psm 11"]

# ============================================================================
# سقف الوقت
# ============================================================================
MAX_SECONDS = 45

# نحجز هذي الثواني بالآخر للنص المطبوع (اسم الأب + تاريخ الإصدار).
# ⚠ رفعناها من 8 لـ14: صرنا نقرا منطقتين بثلاث إعدادات بدل قراءتين
PRINTED_TEXT_BUDGET = 14


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

BAD_VALUES = {"NONE", "NULL", "NAN", "N/A", "NA", "-", "--"}

DOCUMENT_WORDS = [
    "COUNTRY", "COUNTY", "CODE", "COD", "NUMBER", "AUTHORITY",
    "BOOKLET", "TRACKING", "CITIZENSHIP", "REPUBLIC", "ISLAMIC",
    "PASSPORT", "NATIONALITY", "PLACE", "BIRTH", "ISSUE", "EXPIRY",
    "SURNAME", "HOLDER", "SIGNATURE", "OBSERVATIONS", "TYPE",
    "GIVEN", "FATHER", "HUSBAND", "GUARDIAN",
]

# ============================================================================
# ⚠ حروف Tesseract يخلطها مع حرف الحشو "<"
# ============================================================================
# بخط OCR-B، حرف "<" ضيّق ومدبّب. لما الصورة مضغوطة (واتساب/تلغرام)
# ينقرا "X" أو "K". حالتين واقعيتين انمسكوا:
#   "KAZMI<<SYED<ALI<<<<"   →  "KAZMIXXSYEDXALIXXXX"
#   "ABBAS<<TASSAWAR<K<KK"  →  "ABBAS<<TASSAWAR<K<KK"
FILLER_LETTERS = ("X", "K")


# ============================================================================
# تنظيف نص Tesseract
# ============================================================================

def clean_ocr_line(line: str) -> str:
    """ننظف السطر ونبقي فقط أحرف MRZ المسموحة."""

    line = (line or "").upper().strip()

    replacements = {
        "«": "<", "‹": "<", "≤": "<", "—": "<",
        "_": "<", "|": "<", "〈": "<", " ": "",
    }

    for old, new in replacements.items():
        line = line.replace(old, new)

    return re.sub(r"[^A-Z0-9<]", "", line)


def safe_text(value) -> str:
    """نحوّل أي قيمة لنص نظيف — ما نرجّع أبداً كلمة "None" للتطبيق."""

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

DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
}


def fix_digits_in_letters(text: str) -> str:
    """
    نحوّل الأرقام لحروف داخل المقاطع اللي المفروض كلها حروف.

    بمعيار ICAO ما ينفع يجي **أي رقم** بمقطع الأسماء ولا برمز الدولة،
    فأي رقم يطلع هناك = خطأ قراءة مؤكد 100%.
    """

    return "".join(DIGIT_TO_LETTER.get(char, char) for char in (text or ""))


def _collapse_filler_letter(section: str, letter: str) -> str:
    """
    ⚠ نحوّل حرف معيّن (X أو K) لحشو "<" حسب السياق.

    الحرف شرعي بأسماء حقيقية (ALEX، MAX، KHAN، KASHAF)، فما نقدر
    نحذفه بالجملة. القواعد الآمنة:
      • تتابع 3 فأكثر → حشو أكيد (ماكو اسم بشري بثلاث K متتالية)
      • حرف/حرفين **محاصرين بحشو من الجهتين** ("<K<") → حشو
      • حرف/حرفين ملتصقين بحشو بآخر المقطع ("…ALI<KK") → حشو

    نكرر لين يستقر النص، لأن "<K<K<K<" متداخلة وجولة وحدة ما تكفيها
    """

    if letter not in section:
        return section

    for _ in range(6):
        before = section

        # 1. تتابع 3 فأكثر
        section = re.sub(
            letter + r"{3,}",
            lambda m: "<" * len(m.group(0)),
            section,
        )

        # 2. محاصر بحشو من الجهتين
        section = re.sub(
            r"(?<=<)" + letter + r"{1,2}(?=<)",
            lambda m: "<" * len(m.group(0)),
            section,
        )

        # 3. بذيل المقطع بعد حشو
        section = re.sub(
            r"(?<=<)" + letter + r"{1,2}$",
            lambda m: "<" * len(m.group(0)),
            section,
        )

        if section == before:
            break

    return section


def normalize_name_fillers(names_section: str) -> str:
    """
    نصلّح حروف الحشو المقروءة غلط بمقطع الأسماء.

    نبدأ بـX (الحالة الأشيع وأخطر، لأن ممكن ما يبقى ولا "<" بالمقطع
    إطلاقاً)، وبعدها K.
    """

    section = names_section or ""

    if not any(letter in section for letter in FILLER_LETTERS):
        return section

    # ------------------------------------------------------------------
    # حالة X القصوى: المقطع ما بيه ولا حرف "<" إطلاقاً
    # ------------------------------------------------------------------
    # ⚠ لازم نحسبها على النص **الأصلي** قبل أي تعديل — لأن خطوة
    # التحويل نفسها تنتج "<<" جديدة، ولو حسبناها بعدها ينقلب المنطق.
    # خانة الأسماء بمعيار TD3 طولها 39 خانة ودائماً بيها حشو بالآخر،
    # فلو ماكو ولا "<" معناها كل الحشو انقرا X
    if "<" not in section and "X" in section:
        section = section.replace("X", "<")

    for letter in FILLER_LETTERS:
        section = _collapse_filler_letter(section, letter)

    return section


def normalize_mrz_line1(l1: str) -> str:
    """
    نصلّح السطر الأول قبل التحليل:
      • "PXPAK" → "P<PAK"
      • رمز الدولة (خانات 2-4) حروف بس
      • مقطع الأسماء: أرقام → حروف، وبعدين X/K → حشو

    آمن تماماً: أرقام التحقق بمعيار TD3 كلها محسوبة من **السطر
    الثاني** بس، فتعديل السطر الأول ما يأثر عليها إطلاقاً.
    """

    if not l1 or len(l1) <= 5:
        return l1 or ""

    head = l1[:5]

    if len(head) >= 2 and head[1] in ("X", "K"):
        head = head[0] + "<" + head[2:]

    head = head[:2] + fix_digits_in_letters(head[2:])

    names = fix_digits_in_letters(l1[5:])

    return head + normalize_name_fillers(names)


# ============================================================================
# قراءة الأسماء من السطر الأول
# ============================================================================

def trim_filler_tokens(name: str) -> str:
    """
    ⚠ حزام أمان أخير: نشيل المقاطع الذيلية اللي كلها حرف واحد مكرر.

    حالة واقعية: "TASSAWAR K K K KK KKKK" — حروف حشو نجت من كل
    التصحيحات اللي فوق (لأن الصورة مضغوطة والحشو انقرا متقطّع).

    ليش من الذيل بس؟ لأن الأسماء الحقيقية أحياناً بيها مقطع حرف
    واحد **بالوسط** ("SHUHDA E FATIMA" — الـE جزء من الاسم)، بس
    ما تنتهي بمقطع مكرر. وأول ما نلقى مقطع حقيقي نوقف
    """

    tokens = (name or "").split()

    while tokens:
        last = tokens[-1]

        # كله نفس الحرف؟ (K، KK، KKKK، XX…)
        if len(set(last)) != 1:
            break

        # مقطع من حرف واحد: نشيله بس لو الحرف من حروف الحشو المعروفة
        # (نحمي "SHUHDA E" لو انقلبت للذيل)
        if len(last) == 1 and last not in FILLER_LETTERS:
            break

        tokens.pop()

    return " ".join(tokens)


def clean_name_part(raw: str) -> str:
    """ننظف اسم مستخرج: < لمسافة، بدون أرقام ولا رموز."""

    if not raw:
        return ""

    text = str(raw).upper()
    text = text.replace("<", " ")
    text = re.sub(r"[^A-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = trim_filler_tokens(text)

    if text in BAD_VALUES:
        return ""

    return text


def _collapse_doubles(text: str) -> str:
    """
    ⚠ جديد: نلغي أي حرف مكرر متتالي — نوحّد "ABBAS" و"ABAAS" لنفس
    الشكل ("ABAS").

    خطأ OCR شائع جداً: يبلع حرف مكرر (BB→B) أو العكس. هذا كان يكسر
    مطابقة اللقب بـextract_father_by_surname_pattern: لقب صاحب
    الجواز ينقرا "ABAAS" بمكان، و"ABBAS" بمكان ثاني (نفس الجواز!)،
    فالمطابقة الحرفية تفشل وتضيع اسم الأب رغم إنه موجود بالصورة.
    """

    return re.sub(r"(.)\1+", r"\1", text or "")


def is_valid_name(name: str) -> bool:
    """فحص جودة الاسم: حرفين على الأقل، أحرف ومسافات بس، مو حرف مكرر."""

    if not name or len(name) < 2:
        return False

    if not re.fullmatch(r"[A-Z ]{2,39}", name):
        return False

    letters = name.replace(" ", "")
    if len(set(letters)) <= 1:
        return False

    return True


def is_plausible_person_name(name: str) -> bool:
    """
    حارس أقوى من is_valid_name — لاسم الأب تحديداً.

    يرفض رموز الدول وكلمات هيكل الجواز، لأن هذي كانت تطلع كأسماء
    أب بالتطبيق ("PAK"، "COUNTY COD")
    """

    if not name:
        return False

    clean = name.strip().upper()

    if len(clean) < 3:
        return False

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

    نرجّع كمان has_separator: بعض الجوازات باسم واحد بس
    (P<PAKFARZANA<<<) وما بيها اسم أول — وهذا مو خطأ قراءة
    """

    if not l1 or len(l1) < 6:
        return "", "", False

    body = l1[5:44]

    has_separator = "<<" in body.rstrip("<")

    body = body.rstrip("<")

    parts = body.split("<<", 1)

    surname = clean_name_part(parts[0]) if len(parts) >= 1 else ""
    given_names = clean_name_part(parts[1]) if len(parts) >= 2 else ""

    # ------------------------------------------------------------------
    # ⚠ خطة احتياطية: الفاصل "<<" انقرا حرف
    # ------------------------------------------------------------------
    # حالة واقعية: "P<PAKABBASIK<ZAHIDA<PARVEEN" — الفاصل بعد ABBASI
    # انقرا "K<"، فاللقب أخذ كل شي والاسم الأول طلع فاضي.
    # لو صار هيك ننقسم على "<" المفرد: أول مقطع لقب والباقي أسماء أولى
    if not given_names and "<" in parts[0]:
        tokens = [t for t in parts[0].split("<") if t]
        if len(tokens) >= 2:
            surname = clean_name_part(tokens[0])
            given_names = clean_name_part(" ".join(tokens[1:]))

    return surname, given_names, has_separator


def extract_names(fields, l1: str):
    """نجيب الاسم واللقب: من السطر الأول أول، وبعدها من حقول المكتبة."""

    surname, given_names, has_separator = parse_names_from_line1(l1)

    if not is_valid_name(surname):
        surname = clean_name_part(safe_text(getattr(fields, "surname", "")))

    if not is_valid_name(given_names):
        given_names = clean_name_part(safe_text(getattr(fields, "name", "")))

    if not is_valid_name(surname):
        surname = ""

    if not is_valid_name(given_names):
        given_names = ""

    return surname, given_names, has_separator


# ============================================================================
# ⚠ الجنس — من موقعه المباشر بالسطر الثاني
# ============================================================================

def extract_sex_from_line2(l2: str, fields) -> str:
    """
    ⚠⚠ الجنس كان يطلع فاضي **دائماً** — ولا مرة وحدة اشتغل!

    السبب: مكتبة mrz ترجّع الجنس ككائن مو حرف، فـstr() يعطي شي
    مثل "Sex.MALE" أو "male" — والفحص القديم:

        sex = safe_text(getattr(fields, "sex", "")).upper()
        if sex not in ("M", "F"):
            sex = ""              ← يمسحه كل مرة

    كان يفشل دائماً ويمسح القيمة. وهذا خطير: منطق "الحالة"
    بالتطبيق (حملة دار / طباخ للرجال بس) ما يشتغل بدون الجنس.

    الحل: نقراه من موقعه المباشر بمعيار TD3 — الخانة رقم 20
    بالسطر الثاني (M / F / < للغير محدد). هذا أوثق من المكتبة
    أصلاً، ونستخدم المكتبة كاحتياطي بس
    """

    # 1. الموقع المباشر بمعيار TD3 (الأوثق)
    if l2 and len(l2) > 20:
        char = l2[20].upper()
        if char in ("M", "F"):
            return char

    # 2. احتياطي من المكتبة — نفحص النص مهما كان شكله
    raw = str(getattr(fields, "sex", "") or "").upper()

    if "FEMALE" in raw:
        return "F"
    if "MALE" in raw:
        return "M"
    if raw.endswith("F") or raw == "F":
        return "F"
    if raw.endswith("M") or raw == "M":
        return "M"

    return ""


# ============================================================================
# ⚠ إصلاح طول السطر — حرف OCR زايد أو ناقص يزيح كل شي بعده
# ============================================================================
# حالة واقعية: خدش أو ضغط بالصورة يخلي Tesseract يقرا حرف زيادة
# (أو يبلع حرف) بمنتصف السطر الثاني. القص القديم `[:44]` كان يقص
# من الآخر بعمى — فيبقى الحرف الزايد بمكانه ويطيح آخر حرف حقيقي،
# فتنزاح كل الحقول اللي بعد موقع الخلل: تاريخ الميلاد، الجنس،
# تاريخ النفاذ، وكل أرقام التحقق. النتيجة: is_verified/is_checksum_ok
# يطلع False بدون سبب واضح، والجنس يطلع فاضي حتى إن
# extract_sex_from_line2 نفسها صحيحة 100% — لأن الحرف اللي تحسبه
# "موقع 20" مو فعلياً موقع 20 الحقيقي بعد الانزياح.
#
# ⚠ ملاحظة: هذا غير شكل "🔴 غير صالح" اللي يطلع بشاشة المراجعة —
# هذاك عن صلاحية سفر الجواز (أقل من 6 أشهر متبقية)، مو عن صحة
# القراءة. هذا الإصلاح يتكلم عن is_verified/الجنس اللي يرجعهم main.py
#
# ما نقدر نعرف وين بالضبط صار الخلل، فنولّد كل الاحتمالات المعقولة
# (نحذف/نضيف حرف بكل موقع) ونخلي calculate_score يختار الأصح —
# نفس فلسفة "جرب كذا نسخة وخذ الأفضل" المستخدمة بكل الملف.

def _length_fix_candidates(raw: str, expected: int = 44, max_variants: int = 40):
    """نرجّع قائمة احتمالات لسطر طوله مو 44 بالضبط بعد التنظيف.

    أول احتمال دائماً هو نفس السلوك القديم (قص/تبطين بسيط) — لو
    الإصلاح ما فاد، نضل بنفس النتيجة القديمة، ما نخسر شي.
    """

    raw = raw or ""
    variants = [raw[:expected].ljust(expected, "<")]

    diff = len(raw) - expected

    # فرق أكبر من حرفين غالباً خطأ قراءة أعمق (سطر غلط تماماً) —
    # التجربة العشوائية بهالحالة تضيع وقت أكثر مما تفيد
    if diff == 0 or abs(diff) > 2:
        return variants

    if diff > 0:
        # حرف/حرفين زيادة: نجرب حذفهم من مواقع مختلفة
        if diff == 1:
            for i in range(len(raw)):
                candidate = (raw[:i] + raw[i + 1:])[:expected].ljust(expected, "<")
                if candidate not in variants:
                    variants.append(candidate)
                if len(variants) >= max_variants:
                    break
        else:  # diff == 2
            for i in range(len(raw)):
                if len(variants) >= max_variants:
                    break
                for j in range(i + 1, len(raw)):
                    candidate = (raw[:i] + raw[i + 1:j] + raw[j + 1:])
                    candidate = candidate[:expected].ljust(expected, "<")
                    if candidate not in variants:
                        variants.append(candidate)
                    if len(variants) >= max_variants:
                        break
    else:
        # حرف/حرفين ناقصة (غالباً حشو "<" ضاع): نجرب نضيفه بمواقع مختلفة
        missing = -diff
        for i in range(len(raw) + 1):
            candidate = (raw[:i] + ("<" * missing) + raw[i:])
            candidate = candidate[:expected].ljust(expected, "<")
            if candidate not in variants:
                variants.append(candidate)
            if len(variants) >= max_variants:
                break

    return variants


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

        l1_raw = lines[i]
        l2_raw = lines[i + 1]

        if len(l1_raw) < 30 or len(l2_raw) < 30:
            continue

        l1_44 = l1_raw[:44].ljust(44, "<")

        first_ok = (
            l1_44.startswith("P")
            or l1_44.startswith("<<P")
            or "P<" in l1_44[:5]
            # حالة الحشو المقروء X أو K: "PXPAK…" / "PKPAK…"
            or bool(re.match(r"[A-Z][XK][A-Z]{3}", l1_44))
        )

        if not first_ok:
            continue

        # ⚠ كذا احتمال للسطر الثاني لو طوله انحرف عن 44 — كل واحد
        # يروح لـtry_mrz_candidate عادي ويتنافس بالنقاط مع البقية
        for l2_44 in _length_fix_candidates(l2_raw):
            second_has_digits = sum(c.isdigit() for c in l2_44) >= 8
            if second_has_digits:
                candidates.append((l1_44, l2_44))

    return candidates


def extract_mrz_from_full_text(text: str):
    """محاولة ثانية لو Tesseract دمج الأسطر كلها بسلسلة وحدة."""

    cleaned = clean_ocr_line(text)

    candidates = []

    for start in range(max(0, len(cleaned) - 87)):

        chunk = cleaned[start:start + 88]

        if len(chunk) < 88:
            continue

        l1 = chunk[:44]
        l2 = chunk[44:88]

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
    """نحوّل "2023-SEP-14" لـdate."""

    if not text:
        return None

    parts = str(text).split("-")

    if len(parts) != 3 or parts[1] not in MONTH_INDEX:
        return None

    try:
        return date(int(parts[0]), MONTH_INDEX[parts[1]], int(parts[2]))
    except Exception:
        return None


def date_to_text(value) -> str:
    return f"{value.year}-{MONTHS[value.month - 1]}-{value.day:02d}"


# ============================================================================
# قراءة النص المطبوع
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


def _prepare_printed_region(img_bgr, top_ratio):
    """
    نجهّز منطقة من الصورة لقراءة النص المطبوع.

    ⚠ التسميات المطبوعة (Father Name / Date of Issue) خطها رمادي
    ورفيع جداً — أصغر بكثير من حروف منطقة القراءة الآلية. فتحتاج
    تكبير أقوى: 2200 بكسل بدل 1600.

    وكمان: نكبّر **دائماً** لو الصورة أصغر من الهدف، وما نصغّرها
    أبداً لو أكبر — أي تصغير يمحي الخطوط الرفيعة نهائياً
    """

    h, w = img_bgr.shape[:2]

    region = img_bgr[0:int(h * top_ratio), 0:w]

    if region.size == 0:
        return None

    # 2200 بكسل — التسميات المطبوعة تحتاج تكبير أكثر من الـMRZ
    target_width = 2200

    if region.shape[1] < target_width:
        scale = target_width / region.shape[1]
        region = cv2.resize(
            region, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # CLAHE يرفع تباين المناطق المحلية — يطلّع الخط الرمادي الرفيع
    # من الخلفية الملونة للجواز
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    return clahe.apply(gray)


def iter_printed_texts(img_bgr, deadline=None):
    """
    ⚠ مولّد (generator) يرجّع نصوص النص المطبوع **وحدة وحدة**.

    نرجّع كل قراءة لحالها، ما ندمجهم بنص واحد أبداً! لأن المحلل
    يشتغل بمنطق "التسمية بسطر والقيمة بالسطر اللي بعده". لو لصقنا
    نصين، آخر سطر بالقراءة الأولى يصير جار أول سطر بالقراءة الثانية
    — وهما من مكانين مختلفين تماماً بالصورة. (هذا بالضبط اللي طلّع
    اسم أب "COUNTY COD" بالتطبيق)

    ⚠ ليش مولّد مو قائمة؟ عشان نوقف فوراً أول ما نلقى اللي نريده،
    بدل ما نستهلك كل القراءات الست ونضيع الوقت. القراءة الوحدة
    تاخذ ثانيتين تقريباً
    """

    # منطقتين: أعلى 78% (منطقة البيانات)، والصورة كاملة (احتياطي —
    # بعض الصور مقصوصة أو الجواز مايل، فالنسبة الثابتة تقطع سطور)
    for top_ratio in (0.78, 1.0):

        try:
            prepared = _prepare_printed_region(img_bgr, top_ratio)
        except Exception:
            continue

        if prepared is None:
            continue

        for config in TESS_PRINTED_CONFIGS:

            if deadline is not None and time.monotonic() > deadline:
                return

            try:
                text = pytesseract.image_to_string(
                    prepared, config=config, lang="eng"
                )
            except Exception:
                continue

            if text and text.strip():
                yield text


def find_printed_dates(printed_text: str):
    """نلقى كل التواريخ المطبوعة بكل الصيغ الشائعة."""

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


def extract_issue_date_by_label(printed_text: str, birth_date: str, expiry_date: str) -> str:
    """
    ⚠ جديد: نبحث عن تاريخ الإصدار **بتسميته مباشرة** قبل أي استنتاج.

    "Date of Issue" ثم "08 FEB 2023" — أوضح وأدق بكثير من الاستنتاج،
    وتشتغل حتى لو التواريخ المطبوعة ناقصة أو زايدة.

    ⚠ ننتبه: "Place of Issue" مو "Date of Issue"! فنستبعد أي سطر
    فيه PLACE — وإلا نجيب تاريخ من مكان غلط
    """

    lines = [
        re.sub(r"\s+", " ", line).strip().upper()
        for line in (printed_text or "").splitlines()
        if line.strip()
    ]

    birth = parse_formatted_date(birth_date)
    expiry = parse_formatted_date(expiry_date)

    for i, line in enumerate(lines):

        if "ISSUE" not in line and "ISSUS" not in line:
            continue

        if "PLACE" in line:
            continue

        # نفحص نفس السطر والسطرين اللي بعده
        for j in range(i, min(i + 3, len(lines))):

            for value in find_printed_dates(lines[j]):

                if birth and value == birth:
                    continue
                if expiry and value == expiry:
                    continue
                if expiry and value >= expiry:
                    continue
                if value.year < 1980:
                    continue
                if value > date.today():
                    continue

                return date_to_text(value)

    return ""


def extract_issue_date(printed_text: str, birth_date: str, expiry_date: str) -> str:
    """
    نستنتج تاريخ الإصدار.

    منطقة القراءة الآلية تعطينا الميلاد والنفاذ مؤكدين رياضياً،
    فالتاريخ الثالث المطبوع = تاريخ الإصدار.

    الترتيب: البحث بالتسمية أول (الأدق)، وبعدها الاستنتاج.

    ⚠ الاستنتاج يستخدم إن الجوازات تنصدر بصلاحية **5 أو 10 سنين**،
    فنختار المرشح اللي الفرق بينه وبين النفاذ أقرب لوحدة من هالمدتين
    """

    by_label = extract_issue_date_by_label(printed_text, birth_date, expiry_date)
    if by_label:
        return by_label

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

        if expiry and value >= expiry:
            continue

        if birth and value <= birth:
            continue

        if value.year < 1980:
            continue

        if value > today:
            continue

        candidates.append(value)

    if not candidates:
        return ""

    if expiry:

        five_years = 5 * 365
        ten_years = 10 * 365
        tolerance = 120

        def distance_score(value):
            days = (expiry - value).days
            return min(abs(days - five_years), abs(days - ten_years))

        best = min(candidates, key=distance_score)

        if distance_score(best) <= tolerance:
            return date_to_text(best)

    best = max(candidates)

    return date_to_text(best)


def extract_printed_names(printed_text: str):
    """
    نستخرج الاسم الأول واللقب من النص المطبوع بوجه الجواز.

    بعض الجوازات تحط اللقب بس بمنطقة القراءة الآلية بدون الفاصل 
    والاسم الأول (P<PAKFARZANA<<<<) — بهاي الحالة الاسم الأول موجود
    بس بالنص المطبوع تحت "Given Names"
    """

    text = (printed_text or "").upper()

    surname = ""
    given = ""

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


def reorder_comma_name(raw: str, holder_surname: str) -> str:
    """
    ⚠ إصلاح مهم: نمط "اللقب، الأسماء" لازم ينقلب.

    الجواز الباكستاني يطبع اسم الأب/الزوج بصيغة SURNAME, GIVEN NAMES:
        "HUSSAIN, GHULAM"      →  الاسم الحقيقي: GHULAM HUSSAIN
        "SYED, AIJAZ HUSSAIN"  →  الاسم الحقيقي: AIJAZ HUSSAIN SYED

    الكود القديم كان يشيل الفاصلة بس ويترك الترتيب، فيطلع
    "HUSSAIN GHULAM" — مقلوب تماماً.

    استثناء: لو اللقب قبل الفاصلة هو **نفسه لقب صاحب الجواز**،
    نرجّع اللي بعد الفاصلة بس — لأن اللقب موجود أصلاً بحقل SURNAME
    وتكراره حشو بلا فايدة
    """

    text = (raw or "").strip()

    if "," not in text:
        return clean_name_part(text)

    before, after = text.split(",", 1)

    before_clean = clean_name_part(before)
    after_clean = clean_name_part(after)

    if not after_clean:
        return before_clean

    if not before_clean:
        return after_clean

    surname = (holder_surname or "").strip().upper()

    # نفس لقب صاحب الجواز → ناخذ الأسماء بس
    # (مقارنة متسامحة مع الحروف المكررة، نفس سبب _collapse_doubles أعلاه)
    if surname and _collapse_doubles(before_clean) == _collapse_doubles(surname):
        return after_clean

    # غير هيك → الأسماء الأولى أول، وبعدها اللقب
    return f"{after_clean} {before_clean}".strip()


def extract_father_by_surname_pattern(printed_text: str, holder_surname: str) -> str:
    """
    نمط "لقب صاحب الجواز، الأسماء" — الحالة الأوضح.

      • صاحب الجواز BUGHIO  →  "BUGHIO, MAZHAR HUSSAIN"
      • صاحب الجواز SHAMSI  →  "SHAMSI, SYED NAYYAR TOUQIR IRTAZA"

    تشتغل **حتى لو تسمية "Father Name" ما انقرت أصلاً** — وهذي
    بالضبط الحالة اللي كانت تخلي الحقل فاضي، لأن التسمية مطبوعة
    بخط رمادي صغير وTesseract يضيّعها كثير.

    ⚠ خفّفنا الشرط: كان يطلب تطابق **تام** لكل اللي قبل الفاصلة مع
    اللقب. بس Tesseract غالباً يلصق التسمية بنفس السطر
    ("Husband Name SYED, AIJAZ HUSSAIN")، فالتطابق التام يفشل.
    الحين يكفي إن اللقب يكون **آخر كلمة** قبل الفاصلة
    """

    surname = (holder_surname or "").strip().upper()

    if len(surname) < 3:
        return ""

    # ⚠ نقارن بعد إلغاء الحروف المكررة: نفس الجواز ممكن يعطينا اللقب
    # "ABAAS" بمكان (من منطقة القراءة الآلية) و"ABBAS" بمكان ثاني
    # (من النص المطبوع) — نفس الاسم، خطأ OCR بس. مطابقة حرفية تفشل
    # وتضيع اسم الأب رغم وجوده بوضوح بالصورة
    surname_key = _collapse_doubles(surname)

    for line in (printed_text or "").upper().splitlines():

        line = re.sub(r"\s+", " ", line).strip()

        comma_index = line.find(",")

        if comma_index <= 0:
            continue

        before = line[:comma_index].strip()
        before_words = re.sub(r"[^A-Z ]", " ", before).split()

        if not before_words or _collapse_doubles(before_words[-1]) != surname_key:
            continue

        candidate = clean_name_part(line[comma_index + 1:])

        if is_plausible_person_name(candidate):
            return candidate

    return ""


def extract_father_by_place_anchor(printed_text: str) -> str:
    """
    ⚠ جديد: مرساة "مكان الولادة".

    ترتيب حقول الجواز الباكستاني ثابت ومطبوع بنفس العمود:

        Sex          Place of Birth
        F            LAHORE, PAK          ← المرساة
        Husband Name
        SYED, AIJAZ HUSSAIN               ← القيمة اللي نريدها
        Date of Issue
        15 MAY 2017                       ← نوقف هنا

    فايدتها: تشتغل حتى لو تسمية Father/Husband ما انقرت **إطلاقاً**،
    وكمان لو اللقب قبل الفاصلة مو لقب صاحب الجواز (لقب الزوج).
    هذي كانت الحالة بجوازين من أربعة انفحصوا
    """

    lines = [
        re.sub(r"\s+", " ", line).strip().upper()
        for line in (printed_text or "").splitlines()
        if line.strip()
    ]

    place_pattern = re.compile(r"^([A-Z][A-Z .]{1,28}),\s*([A-Z]{3,12})$")

    for i, raw_line in enumerate(lines):

        line = raw_line

        # خانة الجنس (M/F) تنطبع بنفس السطر أحياناً — نشيلها
        line = re.sub(r"^[MF]\s+", "", line)
        # وكمان التسمية لو انلصقت
        line = re.sub(r"^PLACE\s+OF\s+BIRTH\s*:?\s*", "", line)

        match = place_pattern.match(line)

        if not match:
            continue

        # لازم اللي بعد الفاصلة يكون دولة، مو اسم شخص
        tail = match.group(2).replace(" ", "")
        if tail not in COUNTRY_NAMES and tail not in COUNTRY_NAMES.values():
            continue

        # ندور على القيمة بالسطور الثلاثة اللي بعده
        for j in range(i + 1, min(i + 4, len(lines))):

            candidate_line = lines[j]

            # وصلنا لتاريخ (الإصدار) → خلصت منطقة اسم الأب
            if find_printed_dates(candidate_line):
                break

            # نشيل بادئة التسمية لو انلصقت بنفس السطر
            candidate_line = re.sub(
                r"^(FATHER|HUSBAND|GUARDIAN|MOTHER|SPOUSE)[^A-Z]*(NAME)?[^A-Z]*",
                "",
                candidate_line,
            )

            name = reorder_comma_name(candidate_line, "")

            if is_plausible_person_name(name):
                return name

    return ""


def extract_father_name(printed_text: str, holder_surname: str = "") -> str:
    """نلقى اسم الأب أو الزوج — بثلاث طرق مرتبة حسب الدقة."""

    # 1. النمط "لقب صاحب الجواز، الأسماء" (الأدق)
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

        # ⚠ هنا كان يطلع الاسم مقلوب — reorder_comma_name تصلّحه
        name = reorder_comma_name(raw, holder_surname)

        if is_plausible_person_name(name):
            return name

    # 3. مرساة مكان الولادة (تشتغل بدون أي تسمية)
    return extract_father_by_place_anchor(printed_text)


# ============================================================================
# حساب قوة النتيجة
# ============================================================================

def calculate_score(is_checksum_ok, fields, l1, l2, surname, given_names):
    """
    نعطي نقاط للنتيجة حتى نختار الأقوى بين كل الاحتمالات.

    ملاحظة مهمة: معيار TD3 ما بيه أي رقم تحقق للسطر الأول (سطر
    الأسماء)! يعني القراءة تقدر تكون "مؤكدة" رياضياً والأسماء خردة
    كاملة. عشان هيك نعطي وزن كبير لجودة الأسماء
    """

    score = 0

    if is_checksum_ok:
        score += 6

    if is_valid_name(surname):
        score += 3

    if is_valid_name(given_names):
        score += 3

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
# ⚠ فحص شكل رقم الجواز
# ============================================================================
# حالة واقعية: رقم "JK5173291" انقرا "JKS473291" — وأرقام التحقق
# نجحت عليه (لأن السطر الثاني كله انقرا غلط بشكل متسق)، فطلعت
# القراءة "مؤكدة" برقم غلط بدون أي إشارة.
#
# ما نقدر "نصلّحه" برمجياً: أي تعديل يكسر التحقق الرياضي، وما عندنا
# طريقة نعرف الصح. بس نقدر **ننبّه** الموظف يراجعه بعينه
PASSPORT_PATTERNS = {
    "PAK": re.compile(r"^[A-Z]{2}\d{7}$"),
    "IRQ": re.compile(r"^[A-Z]\d{7,8}$"),
}


def passport_number_warning(number: str, country_code: str) -> str:
    pattern = PASSPORT_PATTERNS.get((country_code or "").upper())

    if pattern is None or not number:
        return ""

    if pattern.match(number):
        return ""

    return f"شكل رقم الجواز مو معتاد لـ{country_code} — راجعه بعينك"


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

        # ⚠ الجنس — من موقعه المباشر بالسطر الثاني، مو من المكتبة
        sex = extract_sex_from_line2(l2, fields)

        # ⚠ الجوازات باسم واحد: بعض الجوازات الباكستانية ما بيها
        # فاصل "<<" إطلاقاً (P<PAKFARZANA<<<) — يعني ماكو اسم أول
        # أصلاً، وهذا **مو خطأ قراءة**
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

            "is_verified": is_checksum_ok,
            "is_fully_verified": is_checksum_ok and names_are_complete,

            "number_warning": passport_number_warning(
                passport_number, nationality_code
            ),

            # للتشخيص
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
    """

    variants = []

    h, w = img_bgr.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
    """نجرب النسخ ونرجع أفضل نتيجة، مع احترام سقف الوقت."""

    best_score = -1
    best_data = None

    configs = TESS_CONFIGS[:1] if quick else TESS_CONFIGS

    for variant in preprocess_variants(img_bgr, quick=quick):

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
                    # سليمة كمان**
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

    deadline = time.monotonic() + MAX_SECONDS

    # ⚠ نحجز آخر ثواني للنص المطبوع (اسم الأب + تاريخ الإصدار).
    # بدون هالحجز، البحث عن MRZ ياكل كل الوقت وما يبقى شي لهم
    mrz_deadline = deadline - PRINTED_TEXT_BUDGET

    def better(candidate):
        if candidate is None:
            return False
        if best_data is None:
            return True
        return candidate.get("score", 0) > best_data.get("score", 0)

    # المرحلة 1: سريعة — تنجح بأغلب الصور وتخلص خلال ثواني
    data = process_image(img_bgr, deadline=mrz_deadline, quick=True)

    if better(data):
        best_data = data
        best_image = img_bgr

    if best_data is not None and best_data.get("is_fully_verified"):
        return _finalize(best_data, best_image, deadline)

    # المرحلة 2: سريعة — الصورة مقلوبة 180 درجة
    if time.monotonic() < mrz_deadline:

        flipped = cv2.rotate(img_bgr, cv2.ROTATE_180)

        data = process_image(flipped, deadline=mrz_deadline, quick=True)

        if better(data):
            best_data = data
            best_image = flipped

        if best_data is not None and best_data.get("is_fully_verified"):
            return _finalize(best_data, best_image, deadline)

    # المرحلة 3: شاملة — للأصلية والمقلوبة فقط.
    # ما نجرب 90 و270 درجة: الصور بهذا التطبيق دائماً أفقية
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

def _scan_printed_text(best_data, printed_text, holder_surname):
    """نمرر قراءة وحدة على كل المستخرجات، ونعبّي الفاضي بس."""

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

    # احتياطي الأسماء: لو منطقة القراءة الآلية ما أعطت اسم أول
    # (يصير لما ما يكون بيها الفاصل <<)، نجيبه من "Given Names"
    printed_surname, printed_given = extract_printed_names(printed_text)

    if not best_data.get("given_name_en") and printed_given:
        best_data["given_name_en"] = printed_given

    if not best_data.get("surname_en") and printed_surname:
        best_data["surname_en"] = printed_surname

    # حالة خاصة: اللقب والاسم الأول طلعوا نفس الشي من منطقة القراءة
    # الآلية (لأن ماكو فاصل)، والنص المطبوع يفرّقهم
    if (
        printed_given
        and printed_surname
        and printed_given != printed_surname
        and best_data.get("given_name_en") == best_data.get("surname_en")
    ):
        best_data["given_name_en"] = printed_given
        best_data["surname_en"] = printed_surname


def _finalize(best_data, best_image, deadline):
    """
    نضيف تاريخ الإصدار واسم الأب، وننظف كل الحقول قبل الإرسال.

    ⚠ ملاحظة تاريخية: كان هنا بگ إن `deadline` مو ممرر كمعامل،
    فبايثون يرمي NameError وexcept يبلعه بصمت — يعني **ولا صورة**
    انقرا منها اسم أب ولا تاريخ إصدار. انصلح، وخلّينا حقل
    "printed_text_note" يقول شنو صار بالضبط عشان ما يتكرر فشل صامت.

    ⚠ الإضافة الجديدة: نقرا لحد 6 نصوص (منطقتين × 3 إعدادات)،
    ونوقف فوراً أول ما نلقى الحقلين. ولو ما لقينا شي إطلاقاً، نجرب
    الصورة مقلوبة 180° — أحياناً أفضل نتيجة MRZ تجي من نسخة مقلوبة،
    وقتها "أعلى الصورة" يكون فعلياً أسفل الجواز وماكو بيه تسميات
    """

    best_data["printed_text_note"] = ""
    best_data["printed_reads"] = 0

    try:
        remaining = deadline - time.monotonic()

        if remaining < 3:
            best_data["printed_text_note"] = (
                f"ماكو وقت كافي للنص المطبوع (بقى {remaining:.1f} ثانية)"
            )
        else:
            holder_surname = best_data.get("surname_en", "")
            reads = 0

            # نجرب الصورة كما هي، وبعدها مقلوبة لو ما نجحنا
            images = [best_image, cv2.rotate(best_image, cv2.ROTATE_180)]

            for image in images:

                for printed_text in iter_printed_texts(image, deadline=deadline):

                    reads += 1

                    _scan_printed_text(best_data, printed_text, holder_surname)

                    # لقينا الاثنين؟ خلاص، ما نضيع وقت
                    if best_data.get("father_name_en") and best_data.get("issue_date"):
                        break

                    if time.monotonic() > deadline - 1:
                        break

                if best_data.get("father_name_en") or best_data.get("issue_date"):
                    break

                if time.monotonic() > deadline - 3:
                    break

            best_data["printed_reads"] = reads

            if reads == 0:
                best_data["printed_text_note"] = "النص المطبوع طلع فاضي"
            else:
                notes = []
                if not best_data.get("father_name_en"):
                    notes.append("ما لقينا اسم الأب")
                if not best_data.get("issue_date"):
                    notes.append("ما لقينا تاريخ الإصدار")

                best_data["printed_text_note"] = (
                    " و".join(notes) if notes else "تمام"
                )

    except Exception as error:
        # ⚠ ما نبلع الخطأ بصمت — نسجّله بالنتيجة
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

    # حزام أمان أخير على الأسماء: أي بقايا حشو ذيلية تنشال
    best_data["given_name_en"] = trim_filler_tokens(best_data["given_name_en"])
    best_data["surname_en"] = trim_filler_tokens(best_data["surname_en"])

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
