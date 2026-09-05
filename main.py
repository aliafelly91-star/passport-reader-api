"""
main.py — خدمة قراءة الجوازات (FastAPI) — Label/Crop OCR v2
==========================================================

إصلاحات هذي النسخة (بناءً على حالات فشل حقيقية من التطبيق):

15. ⚠ حرف الحشو "<" ينقرا كمان **"G"**. حالة واقعية:
    "ABBAS<<NAZAR<<<<<<<<" طلع منها الاسم الأول
    "NAZAR G GGGGGGGGGGG". أضفنا G لحروف الحشو — بس بالقواعد
    السياقية الآمنة بس (تتابع 3+، أو محاصر بحشو)، ما نطبّق عليه
    قاعدة X القصوى لأن G حرف شرعي جداً بالأسماء (GHULAM).

16. ⚠ حرف حشو **ملتصق** بآخر الاسم بدون "<" قبله. حالة واقعية:
    "TASSAWAR" طلع "TASSAWARK SE S" — الـK التصقت بالاسم مباشرة
    فما انطبقت عليها أي قاعدة سياقية. الحل: لو انشالت مقاطع خردة
    من الذيل، وآخر مقطع باقي ينتهي بحرف حشو، نشيل الحرف.

17. ⚠ تسمية منقرية غلط تطلع كاسم أب. حالة واقعية:
    "DATE OF EXPIRY" انقرت "DSTE OF EPRY" فطلعت كاسم أب — لأن
    الفحص القديم يقارن الكلمات **حرفياً** بقائمة DOCUMENT_WORDS،
    و"EPRY" مو "EXPIRY". الحين نقارن بمسافة تحرير (Levenshtein).

18. ⚠ مطابقة الاسم مع النص المطبوع. الاسم المطبوع بوجه الجواز
    خطه أكبر وأوضح من منطقة القراءة الآلية، فلو الاثنين يختلفون
    بحروف زايدة بالذيل بس، نثق بالمطبوع.

- نفس Endpoint: /read-passport
- نفس أسماء الحقول اللي ينتظرها Flutter
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
SERVER_VERSION = "cloud-label-crop-v2"


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

TESS_PRINTED_CONFIGS = ["--oem 1 --psm 11 -c preserve_interword_spaces=1", "--oem 1 --psm 6 -c preserve_interword_spaces=1"]

MAX_SECONDS = 35
PRINTED_TEXT_BUDGET = 9


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
MONTH_INDEX["SEPT"] = 9

BAD_VALUES = {"NONE", "NULL", "NAN", "N/A", "NA", "-", "--"}

DOCUMENT_WORDS = [
    "COUNTRY", "COUNTY", "CODE", "COD", "NUMBER", "AUTHORITY",
    "BOOKLET", "TRACKING", "CITIZENSHIP", "REPUBLIC", "ISLAMIC",
    "PASSPORT", "NATIONALITY", "PLACE", "BIRTH", "ISSUE", "EXPIRY",
    "SURNAME", "HOLDER", "SIGNATURE", "OBSERVATIONS", "TYPE",
    "GIVEN", "FATHER", "HUSBAND", "GUARDIAN",
]

# ⚠ كلمات نقارنها بمسافة تحرير — تمسك القراءة المشوّهة كمان
# ("EXPIRY" ← "EPRY"، "DATE" ← "DSTE")
FUZZY_LABEL_WORDS = [
    "DATE", "OF", "EXPIRY", "ISSUE", "BIRTH", "PLACE", "NAME",
    "NAMES", "SURNAME", "GIVEN", "FATHER", "HUSBAND", "GUARDIAN",
    "SEX", "TYPE", "CODE", "COUNTRY", "NUMBER", "NATIONALITY",
    "AUTHORITY", "TRACKING", "BOOKLET", "CITIZENSHIP", "PASSPORT",
    "HOLDER", "SIGNATURE", "VALID", "UNTIL", "ISSUING",
]

# ============================================================================
# ⚠ حروف Tesseract يخلطها مع حرف الحشو "<"
# ============================================================================
# بخط OCR-B حرف "<" ضيّق ومدبّب. لما الصورة مضغوطة ينقرا X أو K أو G.
# ثلاث حالات واقعية انمسكوا:
#   "KAZMI<<SYED<ALI<<<<"      →  "KAZMIXXSYEDXALIXXXX"
#   "ABBAS<<TASSAWAR<<<<"      →  "ABBAS<<TASSAWARK<SE<S"
#   "ABBAS<<NAZAR<<<<<<<<"     →  "ABBAS<<NAZAR<G<GGGGGGG"
FILLER_LETTERS = ("X", "K", "G")


# ============================================================================
# ⚠ مسافة التحرير — لكشف التسميات المنقرية غلط
# ============================================================================

def edit_distance(a: str, b: str) -> int:
    """مسافة Levenshtein بين نصين."""

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(
                previous[j] + 1,        # حذف
                current[j - 1] + 1,     # إضافة
                previous[j - 1] + cost  # استبدال
            ))
        previous = current

    return previous[-1]


def looks_like_label_word(word: str) -> bool:
    """
    ⚠ هل هذي الكلمة تسمية بالجواز (حتى لو منقرية غلط)؟

    الفحص القديم كان حرفياً: "EXPIRY" in text. بس Tesseract قرا
    "EPRY" و"DSTE"، فما انمسكوا وطلعوا كاسم أب.

    نسمح بحرف واحد غلط للكلمات القصيرة، وحرفين للطويلة (6+).
    """

    word = (word or "").strip().upper()

    if len(word) < 2:
        return False

    for label in FUZZY_LABEL_WORDS:
        if word == label:
            return True

        # فرق الطول كبير → مو نفس الكلمة أصلاً
        if abs(len(word) - len(label)) > 2:
            continue

        tolerance = 2 if len(label) >= 6 else 1

        if edit_distance(word, label) <= tolerance:
            return True

    return False


def is_label_noise(value: str) -> bool:
    """كل كلمات القيمة تسميات → إحنا ماسكين تسمية مو قيمة."""

    text = re.sub(r"\s+", " ", (value or "").strip().upper())

    if not text:
        return True

    words = [w for w in text.split(" ") if w]

    if not words:
        return True

    return all(looks_like_label_word(w) for w in words)


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
    ⚠ نحوّل حرف معيّن (X أو K أو G) لحشو "<" حسب السياق.

    الحرف شرعي بأسماء حقيقية (ALEX، KHAN، GHULAM)، فما نقدر نحذفه
    بالجملة. القواعد الآمنة:
      • تتابع 3 فأكثر → حشو أكيد (ماكو اسم بشري بثلاث G متتالية)
      • حرف/حرفين **محاصرين بحشو من الجهتين** ("<G<") → حشو
      • حرف/حرفين ملتصقين بحشو بآخر المقطع ("…ALI<KK") → حشو

    نكرر لين يستقر النص، لأن "<G<G<G<" متداخلة وجولة وحدة ما تكفيها
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

    ⚠ قاعدة X القصوى (استبدال كل X بحشو) تنطبق على **X بس** —
    لأن X نادر جداً بأسماء الجوازات اللي نشتغل عليها. أما K وG
    فشرعيين تماماً (KHAN، GHULAM)، فما ناخذ إلا القواعد السياقية.
    """

    section = names_section or ""

    if not any(letter in section for letter in FILLER_LETTERS):
        return section

    # ------------------------------------------------------------------
    # حالة X القصوى: المقطع ما بيه ولا حرف "<" إطلاقاً
    # ------------------------------------------------------------------
    # ⚠ لازم نحسبها على النص **الأصلي** قبل أي تعديل — لأن خطوة
    # التحويل نفسها تنتج "<<" جديدة، ولو حسبناها بعدها ينقلب المنطق.
    # خانة الأسماء بمعيار TD3 طولها 39 خانة ودائماً بيها حشو بالآخر
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
      • مقطع الأسماء: أرقام → حروف، وبعدين X/K/G → حشو

    آمن تماماً: أرقام التحقق بمعيار TD3 كلها محسوبة من **السطر
    الثاني** بس، فتعديل السطر الأول ما يأثر عليها إطلاقاً.
    """

    if not l1 or len(l1) <= 5:
        return l1 or ""

    head = l1[:5]

    if len(head) >= 2 and head[1] in FILLER_LETTERS:
        head = head[0] + "<" + head[2:]

    head = head[:2] + fix_digits_in_letters(head[2:])

    names = fix_digits_in_letters(l1[5:])

    return head + normalize_name_fillers(names)


# ============================================================================
# قراءة الأسماء من السطر الأول
# ============================================================================

def trim_filler_tokens(name: str) -> str:
    """
    ⚠ حزام أمان: نشيل المقاطع الذيلية الخردة.

    حالتين واقعيتين:
      "NAZAR G GGGGGGGGGGG"  ← حشو انقرا G متقطّع
      "TASSAWARK SE S"       ← حشو انقرا K وS بأشكال مختلفة

    نشيل من الذيل:
      • أي مقطع كله نفس الحرف وهذا الحرف من حروف الحشو (GGGG، KK)
      • أي مقطع من حرف واحد وهو حرف حشو (G، K، X)
      • مقطع من حرفين أو أقل **بشرط** إنه جا بعد مقطع خردة (يعني
        إحنا أصلاً بمنطقة حشو) — هيك ما نأذي "SHUHDA E FATIMA"

    ⚠ وبالآخر: لو انشال أي شي، وآخر مقطع باقي **ينتهي بحرف حشو**
    وطوله 5 فأكثر، نشيل الحرف — هذي حالة "TASSAWARK" اللي الـK
    التصقت بالاسم مباشرة بدون "<" قبلها فما انطبقت عليها أي قاعدة
    سياقية بـ_collapse_filler_letter
    """

    tokens = (name or "").split()
    removed_any = False

    while tokens:
        last = tokens[-1]

        # مقطع كله نفس الحرف
        if len(set(last)) == 1:
            # ⚠ حرف واحد لحاله بالذيل = خردة دايماً، بغض النظر عن
            # الحرف. هذا الشرط ناقص كان يمنع حالة "TASSAWARK SE S"
            # من الانصلاح فعلياً — "S" مو بحروف الحشو (X/K/G)
            # فالحلقة كانت توقف بأول مقطع ولا تشيل شي إطلاقاً
            if last[0] in FILLER_LETTERS or len(last) >= 3 or len(last) == 1:
                tokens.pop()
                removed_any = True
                continue
            break

        # مقطع قصير جداً بعد ما شلنا خردة → غالباً بقايا حشو
        if removed_any and len(last) <= 2:
            tokens.pop()
            continue

        break

    # ⚠ حرف حشو ملتصق بآخر الاسم (TASSAWARK → TASSAWAR)
    if removed_any and tokens:
        last = tokens[-1]
        if len(last) >= 5 and last[-1] in FILLER_LETTERS:
            tokens[-1] = last[:-1]

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
    نلغي أي حرف مكرر متتالي — نوحّد "ABBAS" و"ABAAS" لنفس الشكل.

    خطأ OCR شائع جداً: يبلع حرف مكرر (BB→B) أو العكس. هذا كان يكسر
    مطابقة اللقب: لقب صاحب الجواز ينقرا "ABAAS" بمكان و"ABBAS"
    بمكان ثاني (نفس الجواز!)، فالمطابقة الحرفية تفشل
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

    ⚠ الإضافة: فحص التسميات بمسافة تحرير. الفحص القديم كان يقارن
    حرفياً بـDOCUMENT_WORDS، فـ"DSTE OF EPRY" (قراءة مشوّهة لـ
    "DATE OF EXPIRY") عبرت وطلعت كاسم أب بالتطبيق
    """

    if not name:
        return False

    clean = name.strip().upper()

    if len(clean) < 3:
        return False

    if clean.replace(" ", "") in COUNTRY_NAMES:
        return False

    # مطابقة حرفية (سريعة)
    for word in DOCUMENT_WORDS:
        if word in clean:
            return False

    # ⚠ مطابقة متسامحة — تمسك التسميات المنقرية غلط
    if is_label_noise(clean):
        return False

    # ⚠ كلمة "OF" وسط القيمة = تسمية شبه أكيدة ("DATE OF EXPIRY")
    words = clean.split()
    if len(words) >= 2 and any(looks_like_label_word(w) for w in words):
        # لو **أكثر من نص** الكلمات تسميات → مو اسم شخص
        label_count = sum(1 for w in words if looks_like_label_word(w))
        if label_count * 2 >= len(words):
            return False

    # ⚠ 19. مقاطع خردة من تداخل عمودين متجاورين بصفحة البيانات
    # (مثلاً عمود "الجنسية/محل الإقامة" يندمج مع عمود "محل
    # الولادة" بنفس السطر). حالة واقعية: "PAK IE I AA F KHAIRPUR"
    # طلعت كاسم أب — لا اسم بشر حقيقي فيه مقطع من حرف واحد لحاله
    # ("I"، "F")، ولا فيه مقطعين أو أكثر من حرف/حرفين
    if any(len(w) == 1 for w in words):
        return False

    if sum(1 for w in words if len(w) <= 2) >= 2:
        return False

    # ⚠ رمز دولة كمقطع مستقل (مو الكلمة كلها) — نفس منطق مطابقة
    # الدولة الكاملة أعلاه، بس على مستوى الكلمة المفردة
    if any(w in COUNTRY_NAMES for w in words):
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
    # انقرا "K<"، فاللقب أخذ كل شي والاسم الأول طلع فاضي
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
# ⚠ مطابقة اسم الـMRZ مع الاسم المطبوع
# ============================================================================

def reconcile_name(from_mrz: str, from_printed: str) -> str:
    """
    ⚠ الاسم المطبوع بوجه الجواز خطه أكبر وأوضح بكثير من منطقة
    القراءة الآلية. فلو الاثنين يختلفون بحروف زايدة بالذيل بس،
    نثق بالمطبوع.

    حالة واقعية: MRZ أعطى "TASSAWARK" والمطبوع "TASSAWAR" —
    الحرف الزايد من الفاصل المنقري غلط.

    ما نستبدل إلا بشروط ضيقة، عشان ما نخرب اسم صحيح باسم مطبوع
    منقري غلط
    """

    mrz = (from_mrz or "").strip().upper()
    printed = (from_printed or "").strip().upper()

    if not printed or not is_valid_name(printed) or is_label_noise(printed):
        return mrz

    if not mrz:
        return printed

    if mrz == printed:
        return mrz

    # المطبوع بادئة للـMRZ وفرق حرف أو حرفين → حشو زايد بالـMRZ
    if mrz.startswith(printed) and len(mrz) - len(printed) <= 2:
        return printed

    # فرق حرف واحد والمطبوع مو أطول → نثق بالمطبوع
    if len(printed) <= len(mrz) and edit_distance(mrz, printed) <= 1:
        return printed

    return mrz


# ============================================================================
# ⚠ الجنس — من موقعه المباشر بالسطر الثاني
# ============================================================================

def extract_sex_from_line2(l2: str, fields) -> str:
    """
    الجنس من موقعه المباشر بمعيار TD3 — الخانة رقم 20 بالسطر الثاني
    (M / F / < للغير محدد). أوثق من المكتبة، ونستخدمها كاحتياطي بس.

    منطق "الحالة" بالتطبيق (حملة دار / طباخ للرجال) ما يشتغل بدونه
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

def _length_fix_candidates(raw: str, expected: int = 44, max_variants: int = 40):
    """نرجّع قائمة احتمالات لسطر طوله مو 44 بالضبط بعد التنظيف."""

    raw = raw or ""
    variants = [raw[:expected].ljust(expected, "<")]

    diff = len(raw) - expected

    if diff == 0 or abs(diff) > 2:
        return variants

    if diff > 0:
        if diff == 1:
            for i in range(len(raw)):
                candidate = (raw[:i] + raw[i + 1:])[:expected].ljust(expected, "<")
                if candidate not in variants:
                    variants.append(candidate)
                if len(variants) >= max_variants:
                    break
        else:
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
            # حالة الحشو المقروء X/K/G: "PXPAK…" / "PKPAK…" / "PGPAK…"
            or bool(re.match(r"[A-Z][XKG][A-Z]{3}", l1_44))
        )

        if not first_ok:
            continue

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

    التسميات المطبوعة (Father Name / Date of Issue) خطها رمادي
    ورفيع جداً — أصغر بكثير من حروف منطقة القراءة الآلية، فتحتاج
    تكبير أقوى: 2200 بكسل. ونكبّر دائماً لو أصغر، وما نصغّر أبداً
    """

    h, w = img_bgr.shape[:2]

    region = img_bgr[0:int(h * top_ratio), 0:w]

    if region.size == 0:
        return None

    target_width = 2000

    if region.shape[1] < target_width:
        scale = target_width / region.shape[1]
        region = cv2.resize(
            region, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    return clahe.apply(gray)


def _split_words_into_columns(words):
    """
    ⚠ جديد: نقسم كلمات المنطقة لعمود وحد أو عمودين، حسب الإحداثي
    الأفقي — بدل ما نعتمد على ترتيب الأسطر اللي يقرره Tesseract
    نفسه (وهذا بالضبط سبب حالة "PAK IE I AA F KHAIRPUR": عمود
    الجنسية اندمج مع عمود محل الولادة بنفس السطر لأن الصفين ما
    كانا بنفس الارتفاع بالضبط).

    الفكرة: نلقى أكبر فجوة أفقية بمنتصف الصفحة (25%-75% من العرض).
    فجوة حقيقية بين عمودين تكون أوسع بكثير من مجرد مسافة بين
    كلمتين بنفس السطر. لو ما لقينا فجوة واضحة، نرجع عمود وحد
    (يعني الصفحة أصلاً عمود وحد ومو داعي نقسمها).
    """

    if not words:
        return [words]

    page_left = min(w["left"] for w in words)
    page_right = max(w["left"] + w["width"] for w in words)
    page_width = page_right - page_left

    if page_width <= 0:
        return [words]

    xs = sorted(w["left"] for w in words)

    best_gap = 0
    split_x = None

    for i in range(1, len(xs)):
        gap = xs[i] - xs[i - 1]
        mid = (xs[i] + xs[i - 1]) / 2
        rel = (mid - page_left) / page_width
        if 0.25 <= rel <= 0.75 and gap > best_gap:
            best_gap = gap
            split_x = mid

    min_gap = page_width * 0.06

    if split_x is None or best_gap < min_gap:
        return [words]

    left_col = [w for w in words if w["left"] < split_x]
    right_col = [w for w in words if w["left"] >= split_x]

    # عمود فيه كلمتين أو أقل مو عمود حقيقي (احتمال ضجيج)
    if len(left_col) < 3 or len(right_col) < 3:
        return [words]

    return [left_col, right_col]


def _words_to_text_lines(words):
    """
    نرتب كلمات عمود واحد لسطور حسب تقارب الإحداثي العمودي (بدل
    الاعتماد على رقم السطر اللي يعطيه Tesseract، لأنه محسوب على
    كامل عرض المنطقة مو على العمود لحاله).
    """

    if not words:
        return []

    ws = sorted(words, key=lambda w: w["top"])

    lines = [[ws[0]]]
    line_top = ws[0]["top"]
    line_height = ws[0]["height"] or 20

    for w in ws[1:]:
        tolerance = max(line_height, w["height"] or 20) * 0.6
        if abs(w["top"] - line_top) <= tolerance:
            lines[-1].append(w)
            line_top = min(line_top, w["top"])
        else:
            lines.append([w])
            line_top = w["top"]
            line_height = w["height"] or 20

    text_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["left"])
        text = " ".join(w["text"] for w in line_sorted if w["text"].strip())
        if text.strip():
            text_lines.append(text)

    return text_lines


def _extract_words_from_data(data, min_conf=35):
    """نحوّل خرج pytesseract.image_to_data لقائمة كلمات نظيفة."""

    words = []
    n = len(data.get("text", []))

    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if conf < min_conf:
            continue
        words.append({
            "text": text,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
            "conf": conf,
            "block_num": data.get("block_num", [0] * n)[i],
            "par_num": data.get("par_num", [0] * n)[i],
            "line_num": data.get("line_num", [0] * n)[i],
        })

    return words


# ============================================================================
# قراءة موجّهة للنص المطبوع: Label -> Crop -> Enhance -> OCR
# ============================================================================
TARGET_LABEL_PHRASES = {
    "father_name_en": [
        ("FATHER", "NAME"), ("FATHERS", "NAME"),
        ("HUSBAND", "NAME"), ("HUSBANDS", "NAME"),
        ("GUARDIAN", "NAME"),
    ],
    "issue_date": [
        ("DATE", "OF", "ISSUE"), ("ISSUE", "DATE"), ("DATE", "ISSUE"),
    ],
    "given_name_en": [("GIVEN", "NAME"), ("GIVEN", "NAMES")],
    "surname_en": [("SURNAME",)],
}


def _label_token(text: str) -> str:
    return re.sub(r"[^A-Z]", "", (text or "").upper())


def _label_token_matches(observed: str, expected: str) -> bool:
    a = _label_token(observed)
    b = _label_token(expected)
    if not a or not b:
        return False
    if a == b:
        return True
    if abs(len(a) - len(b)) > 2:
        return False
    return edit_distance(a, b) <= (2 if len(b) >= 7 else 1)


def _group_words_by_line(words):
    groups = {}
    for word in words:
        key = (word.get("block_num", 0), word.get("par_num", 0), word.get("line_num", 0))
        groups.setdefault(key, []).append(word)
    return [sorted(line, key=lambda w: w["left"]) for line in groups.values()]


def _bbox_for_words(items):
    return (
        min(w["left"] for w in items),
        min(w["top"] for w in items),
        max(w["left"] + w["width"] for w in items),
        max(w["top"] + w["height"] for w in items),
    )


def _find_label_box(words, phrases):
    best_box = None
    best_score = -1.0
    for line in _group_words_by_line(words):
        for phrase in phrases:
            size = len(phrase)
            for i in range(max(0, len(line) - size + 1)):
                chunk = line[i:i + size]
                if len(chunk) != size:
                    continue
                if not all(_label_token_matches(chunk[j]["text"], phrase[j]) for j in range(size)):
                    continue
                score = sum(float(w.get("conf", 0)) for w in chunk) / size
                if score > best_score:
                    best_score = score
                    best_box = _bbox_for_words(chunk)
    return best_box


def _crop_value_region(prepared, label_box, field_name):
    if prepared is None or label_box is None:
        return None
    h, w = prepared.shape[:2]
    x1, y1, x2, y2 = label_box
    label_h = max(12, y2 - y1)
    label_w = max(20, x2 - x1)
    left = max(0, int(x1 - 0.025 * w))
    top = max(0, int(y2 + 0.05 * label_h))
    if field_name in ("father_name_en", "given_name_en"):
        crop_w = max(int(label_w * 5.2), int(w * 0.36))
        crop_h = int(label_h * 4.4)
    elif field_name == "surname_en":
        crop_w = max(int(label_w * 4.0), int(w * 0.28))
        crop_h = int(label_h * 4.0)
    else:
        crop_w = max(int(label_w * 3.4), int(w * 0.23))
        crop_h = int(label_h * 4.0)
    right = min(w, left + crop_w)
    bottom = min(h, top + crop_h)
    if right - left < 40 or bottom - top < 20:
        return None
    return prepared[top:bottom, left:right]


def _enhance_field_crop(crop):
    if crop is None or crop.size == 0:
        return []
    if crop.shape[1] < 900:
        scale = 900 / max(1, crop.shape[1])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.55, blur, -0.55, 0)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [sharp, otsu]


def _ocr_field_crop(crop, deadline=None):
    """نقرأ القصاصة الصغيرة ونوقف من أول سطر معقول لتقليل الزمن."""
    for variant in _enhance_field_crop(crop):
        if deadline is not None and time.monotonic() > deadline:
            return []
        try:
            data = pytesseract.image_to_data(
                variant,
                config="--oem 1 --psm 7 -c preserve_interword_spaces=1",
                lang="eng",
                output_type=pytesseract.Output.DICT,
            )
            words = _extract_words_from_data(data, min_conf=20)
            lines = _words_to_text_lines(words)
            lines = [re.sub(r"\s+", " ", x).strip() for x in lines if x.strip()]
            if lines:
                return lines
        except Exception:
            pass
    return []


def _best_name_crop_candidate(candidates, holder_surname=""):
    for raw in candidates:
        cleaned = re.sub(
            r"^(FATHER|FATHERS|HUSBAND|HUSBANDS|GUARDIAN)\s*(NAME)?\s*[:\-]?\s*",
            "", raw.upper(),
        ).strip()
        if find_printed_dates(cleaned):
            continue
        name = reorder_comma_name(cleaned, holder_surname)
        if is_plausible_person_name(name):
            return name
    return ""


def _best_date_crop_candidate(candidates, birth_date="", expiry_date=""):
    birth = parse_formatted_date(birth_date)
    expiry = parse_formatted_date(expiry_date)
    for raw in candidates:
        for value in find_printed_dates(raw):
            if birth and value == birth:
                continue
            if expiry and value == expiry:
                continue
            if expiry and value >= expiry:
                continue
            if value.year < 1980 or value > date.today():
                continue
            return date_to_text(value)
    return ""


def read_targeted_printed_fields(img_bgr, best_data, deadline=None):
    """
    1) تحسين الصفحة مرة واحدة.
    2) تحديد labels مرة واحدة بـ image_to_data.
    3) قص قيمة الأب/تاريخ الإصدار فقط.
    4) تحسين القصاصة وقراءتها.
    """
    result, sources = {}, {}
    reads = 0
    try:
        prepared = _prepare_printed_region(img_bgr, 0.78)
        if prepared is None:
            return result, sources, reads, "تعذر تجهيز النص المطبوع"
        if deadline is not None and time.monotonic() > deadline:
            return result, sources, reads, "انتهى وقت النص المطبوع"
        data = pytesseract.image_to_data(
            prepared,
            config="--oem 1 --psm 11 -c preserve_interword_spaces=1",
            lang="eng",
            output_type=pytesseract.Output.DICT,
        )
        reads += 1
        words = _extract_words_from_data(data, min_conf=18)
        if not words:
            return result, sources, reads, "ما لقينا كلمات مطبوعة"
        holder_surname = best_data.get("surname_en", "")

        # الأب وتاريخ الإصدار فقط؛ باقي الحقول الموثوقة ناخذها من MRZ.
        for field_name in ("father_name_en", "issue_date"):
            if best_data.get(field_name):
                continue
            if deadline is not None and time.monotonic() > deadline - 0.5:
                break
            label_box = _find_label_box(words, TARGET_LABEL_PHRASES[field_name])
            if label_box is None:
                continue
            crop = _crop_value_region(prepared, label_box, field_name)
            candidates = _ocr_field_crop(crop, deadline=deadline)
            if candidates:
                reads += 1
            if field_name == "father_name_en":
                value = _best_name_crop_candidate(candidates, holder_surname)
            else:
                value = _best_date_crop_candidate(
                    candidates,
                    best_data.get("birth_date", ""),
                    best_data.get("expiry_date", ""),
                )
            if value:
                result[field_name] = value
                sources[field_name] = "printed_crop"

        # fallback بدون أي OCR جديد: نستفيد من نفس كلمات تحديد labels.
        for col in _split_words_into_columns(words):
            printed_text = "\n".join(_words_to_text_lines(col))
            if not printed_text:
                continue
            if "father_name_en" not in result:
                father = extract_father_name(printed_text, holder_surname)
                if father:
                    result["father_name_en"] = father
                    sources["father_name_en"] = "layout_words"
            if "issue_date" not in result:
                issue = extract_issue_date(
                    printed_text,
                    best_data.get("birth_date", ""),
                    best_data.get("expiry_date", ""),
                )
                if issue:
                    result["issue_date"] = issue
                    sources["issue_date"] = "layout_words"
            if result.get("father_name_en") and result.get("issue_date"):
                break
        return result, sources, reads, ("تمام" if result else "القص الموجّه ما لقى الحقول")
    except Exception as error:
        return result, sources, reads, f"فشل القص الموجّه: {type(error).__name__}: {error}"


def iter_printed_texts(img_bgr, deadline=None):
    """
    مولّد يرجّع نصوص النص المطبوع **وحدة وحدة**.

    ما ندمجهم بنص واحد أبداً! لأن المحلل يشتغل بمنطق "التسمية بسطر
    والقيمة بالسطر اللي بعده". لو لصقنا نصين، آخر سطر بالقراءة
    الأولى يصير جار أول سطر بالقراءة الثانية — وهما من مكانين
    مختلفين تماماً بالصورة

    ⚠ جديد: قبل ما نبني النص، نقرا الكلمات بإحداثياتها
    (image_to_data) ونقسمها لعمود أو عمودين بالموقع الأفقي الفعلي،
    بدل ما نثق بترتيب الأسطر اللي يقرره Tesseract على عرض الصفحة
    كامل. هذا يمنع تداخل عمودين متجاورين بسطر وحد من الأساس —
    بدل ما نصيد الخردة الناتجة بعدين بفحوصات is_plausible_person_name.
    نرجّع النص القديم (image_to_string) كمان كخط رجعة أخير، لأن
    بعض الصور عمود وحد فعلاً وما تحتاج كل هذا
    """

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

            # ------------------------------------------------------
            # 1. الطريقة الجديدة: كلمات بإحداثياتها → أعمدة → سطور
            # ------------------------------------------------------
            try:
                data = pytesseract.image_to_data(
                    prepared, config=config, lang="eng",
                    output_type=pytesseract.Output.DICT,
                )
                words = _extract_words_from_data(data)
                columns = _split_words_into_columns(words)

                for column_words in columns:
                    lines = _words_to_text_lines(column_words)
                    if lines:
                        yield "\n".join(lines)

            except Exception:
                pass

            if deadline is not None and time.monotonic() > deadline:
                return

            # ------------------------------------------------------
            # 2. خط رجعة أخير: النص الخام القديم (image_to_string)
            # ------------------------------------------------------
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

    for match in re.finditer(
        r"\b(\d{1,2})\s*[-/ ]?\s*([A-Z]{3,4})\s*[-/ ]?\s*(\d{4})\b", text
    ):
        month = MONTH_INDEX.get(match.group(2))
        if month:
            add(int(match.group(3)), month, int(match.group(1)))

    for match in re.finditer(
        r"\b([A-Z]{3,4})\s*[-/ ]?\s*(\d{1,2})\s*[-/ ]?\s*(\d{4})\b", text
    ):
        month = MONTH_INDEX.get(match.group(1))
        if month:
            add(int(match.group(3)), month, int(match.group(2)))

    for match in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text):
        add(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    for match in re.finditer(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b", text):
        add(int(match.group(3)), int(match.group(2)), int(match.group(1)))

    return found


def extract_issue_date_by_label(printed_text: str, birth_date: str, expiry_date: str) -> str:
    """
    نبحث عن تاريخ الإصدار **بتسميته مباشرة** قبل أي استنتاج.

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

    الترتيب: البحث بالتسمية أول (الأدق)، وبعدها الاستنتاج
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

    فايدتها مزدوجة:
      • بعض الجوازات تحط اللقب بس بمنطقة القراءة الآلية بدون الفاصل
      • ونستخدمها كمان لمطابقة اسم الـMRZ (reconcile_name)
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
        if is_valid_name(candidate) and not is_label_noise(candidate):
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
        if is_valid_name(candidate) and not is_label_noise(candidate):
            given = candidate

    return surname, given


def reorder_comma_name(raw: str, holder_surname: str = "") -> str:
    """
    الجواز الباكستاني يطبع بعض أسماء الأب/الزوج بصيغة:
        SURNAME, GIVEN NAMES

    نعيدها بالترتيب الطبيعي **بدون حذف أي جزء**:
        NAWAB, WAZER    -> WAZER NAWAB
        HUSSAIN, GHULAM -> GHULAM HUSSAIN

    holder_surname موجود فقط للتوافق مع الاستدعاءات القديمة.
    """
    value = (raw or "").strip()
    if "," not in value:
        return clean_name_part(value)
    before, after = value.split(",", 1)
    before_clean = clean_name_part(before)
    after_clean = clean_name_part(after)
    if not after_clean:
        return before_clean
    if not before_clean:
        return after_clean
    return f"{after_clean} {before_clean}".strip()

def extract_father_by_surname_pattern(printed_text: str, holder_surname: str) -> str:
    """
    نمط "لقب صاحب الجواز، الأسماء" — الحالة الأوضح.

      • صاحب الجواز BUGHIO  →  "BUGHIO, MAZHAR HUSSAIN"
      • صاحب الجواز ABBAS   →  "AHMED, NABI"

    تشتغل **حتى لو تسمية "Father Name" ما انقرت أصلاً** — وهذي
    بالضبط الحالة اللي كانت تخلي الحقل فاضي
    """

    surname = (holder_surname or "").strip().upper()

    if len(surname) < 3:
        return ""

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

        # نحافظ على الجزأين: NAWAB, WAZER -> WAZER NAWAB
        candidate = reorder_comma_name(line, holder_surname)

        if is_plausible_person_name(candidate):
            return candidate

    return ""


def extract_father_by_comma_line(printed_text: str, holder_surname: str) -> str:
    """
    ⚠ جديد: أي سطر بصيغة "كلمة، كلمات" — حتى لو اللقب قبل الفاصلة
    مو لقب صاحب الجواز.

    حالة واقعية: صاحبة جواز لقبها ZAHRA وخانة الزوج فيها
    "SHAMSI, SYED NAYYAR" — لقب الزوج مختلف تماماً.

    الحارس: نستبعد السطور اللي فيها تاريخ أو تسميات، ونستبعد
    "مكان الولادة" (لأن اللي بعد فاصلته دولة مو اسم)
    """

    lines = [
        re.sub(r"\s+", " ", line).strip().upper()
        for line in (printed_text or "").splitlines()
        if line.strip()
    ]

    for line in lines:

        if "," not in line:
            continue

        # سطر فيه تاريخ → مو سطر اسم
        if find_printed_dates(line):
            continue

        before, after = line.split(",", 1)

        after_clean = clean_name_part(after)

        # اللي بعد الفاصلة دولة → هذا مكان الولادة مو اسم أب
        if after_clean.replace(" ", "") in COUNTRY_NAMES:
            continue
        if after_clean.replace(" ", "") in COUNTRY_NAMES.values():
            continue

        name = reorder_comma_name(line, holder_surname)

        if is_plausible_person_name(name):
            return name

    return ""


def extract_father_by_place_anchor(printed_text: str) -> str:
    """
    مرساة "مكان الولادة".

    ترتيب حقول الجواز الباكستاني ثابت ومطبوع بنفس العمود:

        Sex          Place of Birth
        M            SIALKOT, PAK         ← المرساة
        Father Name
        AHMED, NABI                       ← القيمة اللي نريدها
        Date of Issue
        16 AUG 2022                       ← نوقف هنا

    فايدتها: تشتغل حتى لو تسمية Father/Husband ما انقرت **إطلاقاً**
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

        tail = match.group(2).replace(" ", "")
        if tail not in COUNTRY_NAMES and tail not in COUNTRY_NAMES.values():
            continue

        for j in range(i + 1, min(i + 4, len(lines))):

            candidate_line = lines[j]

            # وصلنا لتاريخ (الإصدار) → خلصت منطقة اسم الأب
            if find_printed_dates(candidate_line):
                break

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
    """نلقى اسم الأب أو الزوج — بأربع طرق مرتبة حسب الدقة."""

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

        raw = re.split(
            r"\b(DATE|PLACE|SEX|NATIONALITY|ISSUING|TRACKING|BOOKLET"
            r"|PASSPORT|AUTHORITY|CITIZENSHIP|COUNTRY|COUNTY|CODE|TYPE)\b",
            raw,
        )[0]

        name = reorder_comma_name(raw, holder_surname)

        if is_plausible_person_name(name):
            return name

    # 3. مرساة مكان الولادة (تشتغل بدون أي تسمية)
    by_anchor = extract_father_by_place_anchor(printed_text)

    if by_anchor:
        return by_anchor

    # 4. أي سطر "لقب، أسماء" (لقب الزوج المختلف)
    return extract_father_by_comma_line(printed_text, holder_surname)


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

    # ⚠ عقوبة: أسماء فيها بقايا حشو (مقاطع من حرف واحد) — نفضّل
    # النسخة اللي أسماؤها نظيفة
    for name in (surname, given_names):
        tokens = (name or "").split()
        junk = sum(1 for t in tokens if len(t) <= 2 and len(set(t)) == 1)
        score -= junk

    return score


# ============================================================================
# فحص شكل رقم الجواز
# ============================================================================

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

        # نصلّح الأرقام وحروف الحشو بالسطر الأول قبل أي تحليل.
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

        sex = extract_sex_from_line2(l2, fields)

        # الجوازات باسم واحد: بعض الجوازات الباكستانية ما بيها
        # فاصل "<<" إطلاقاً (P<PAKFARZANA<<<) — وهذا **مو خطأ قراءة**
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
            "field_sources": {
                "given_name_en": "mrz",
                "surname_en": "mrz",
                "passport_number": "mrz",
                "nationality": "mrz",
                "residence_country": "mrz",
                "birth_date": "mrz",
                "expiry_date": "mrz",
                "sex": "mrz",
            },
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
        # المسار السريع: منطقتان من أسفل الجواز × نسختان فقط.
        # إذا ما نجحن، المسار الشامل يبقى fallback.
        for crop_ratio in [0.33, 0.40]:
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
            enhanced = clahe.apply(gray)
            variants.append(enhanced)
            _, otsu = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
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
    request_started = time.monotonic()

    def finish(data):
        if isinstance(data, dict):
            data["ocr_time_ms"] = int((time.monotonic() - request_started) * 1000)
            data["server_version"] = SERVER_VERSION
        return data

    np_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return finish({"success": False, "error": "تعذر فك ترميز الصورة"})

    best_data = None
    best_image = img_bgr
    deadline = time.monotonic() + MAX_SECONDS
    mrz_deadline = deadline - PRINTED_TEXT_BUDGET

    def better(candidate):
        if candidate is None:
            return False
        if best_data is None:
            return True
        return candidate.get("score", 0) > best_data.get("score", 0)

    # 1) الأصلية - مسار سريع.
    data = process_image(img_bgr, deadline=mrz_deadline, quick=True)
    if better(data):
        best_data = data
        best_image = img_bgr
    if best_data is not None and best_data.get("is_fully_verified"):
        return finish(_finalize(best_data, best_image, deadline))

    # 2) المقلوبة 180 فقط إذا فشل الأصل.
    if time.monotonic() < mrz_deadline:
        flipped = cv2.rotate(img_bgr, cv2.ROTATE_180)
        data = process_image(flipped, deadline=mrz_deadline, quick=True)
        if better(data):
            best_data = data
            best_image = flipped
        if best_data is not None and best_data.get("is_fully_verified"):
            return finish(_finalize(best_data, best_image, deadline))

    # 3) شامل كحل أخير.
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
        return finish({
            "success": False,
            "error": "ما قدرنا نلقى منطقة قراءة آلية واضحة بالصورة",
        })

    return finish(_finalize(best_data, best_image, deadline))


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

    printed_surname, printed_given = extract_printed_names(printed_text)

    # احتياطي الأسماء: لو منطقة القراءة الآلية ما أعطت اسم أول
    if not best_data.get("given_name_en") and printed_given:
        best_data["given_name_en"] = printed_given

    if not best_data.get("surname_en") and printed_surname:
        best_data["surname_en"] = printed_surname

    # ⚠ مطابقة: الاسم المطبوع أوضح من الـMRZ، فلو الفرق حروف زايدة
    # بالذيل بس (TASSAWARK ← TASSAWAR) نثق بالمطبوع
    if printed_given:
        best_data["given_name_en"] = reconcile_name(
            best_data.get("given_name_en", ""), printed_given
        )

    if printed_surname:
        best_data["surname_en"] = reconcile_name(
            best_data.get("surname_en", ""), printed_surname
        )

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
    """النص المطبوع: قص موجّه أولاً، ثم fallback عام واحد فقط."""
    best_data["printed_text_note"] = ""
    best_data["printed_reads"] = 0

    try:
        remaining = deadline - time.monotonic()
        if remaining < 2:
            best_data["printed_text_note"] = (
                f"ماكو وقت كافي للنص المطبوع (بقى {remaining:.1f} ثانية)"
            )
        else:
            targeted, sources, reads, note = read_targeted_printed_fields(
                best_image, best_data, deadline=deadline
            )
            best_data["printed_reads"] += reads
            for key, value in targeted.items():
                if value and not best_data.get(key):
                    best_data[key] = value
            best_data.setdefault("field_sources", {}).update(sources)

            # fallback عام واحد فقط إذا بقي حقل ناقص.
            if (
                (not best_data.get("father_name_en") or not best_data.get("issue_date"))
                and time.monotonic() < deadline - 3
            ):
                holder_surname = best_data.get("surname_en", "")
                for printed_text in iter_printed_texts(best_image, deadline=deadline - 1):
                    best_data["printed_reads"] += 1
                    _scan_printed_text(best_data, printed_text, holder_surname)
                    break
                if best_data.get("father_name_en"):
                    best_data.setdefault("field_sources", {}).setdefault(
                        "father_name_en", "printed_fallback"
                    )
                if best_data.get("issue_date"):
                    best_data.setdefault("field_sources", {}).setdefault(
                        "issue_date", "printed_fallback"
                    )

            notes = []
            if not best_data.get("father_name_en"):
                notes.append("ما لقينا اسم الأب")
            if not best_data.get("issue_date"):
                notes.append("ما لقينا تاريخ الإصدار")
            best_data["printed_text_note"] = " و".join(notes) if notes else note

    except Exception as error:
        best_data["printed_text_note"] = (
            f"فشل النص المطبوع: {type(error).__name__}: {error}"
        )

    # ⚠ حارس أخير على اسم الأب: لو طلعت تسمية منقرية غلط
    # ("DSTE OF EPRY")، نمسحها بدل ما نرسلها للتطبيق
    father = best_data.get("father_name_en", "")
    if father and not is_plausible_person_name(father):
        best_data["father_name_en"] = ""
        note = best_data.get("printed_text_note", "")
        best_data["printed_text_note"] = (
            (note + " · " if note and note != "تمام" else "")
            + "اسم الأب انرفض (تسمية مو قيمة)"
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
    best_data["father_name_en"] = trim_filler_tokens(best_data["father_name_en"])

    # ========================================================================
    # ⚠ 20. حقول جديدة فقط — ما نلمس ولا نحذف أي حقل موجود، فما
    # يتعارض مع أي Parsing موجود بتطبيق Flutter. الفكرة: بيانات
    # الـMRZ (الاسم، رقم الجواز، الميلاد، الانتهاء، الجنسية) مؤكدة
    # رياضياً برقم تحقق checksum — ما فيها مجال غلط عملياً. بيانات
    # النص المطبوع (اسم الأب، تاريخ الإصدار) ماكو لها أي رقم تحقق
    # بمعيار الجواز، فمستحيل تكون مؤكدة 100% مهما قوينا القراءة —
    # هذا حد فيزيائي مو تقصير كود. `needs_review` تعلّم أي حقل
    # يستاهل عين بشر قبل الاعتماد، بدل ما نطارد "0% غلط" ماكو ممكن
    # ========================================================================

    best_data["needs_review"] = bool(
        not best_data.get("father_name_en")
        or not best_data.get("issue_date")
    )

    best_data["fields_confidence"] = {
        "given_name_en": "verified" if best_data.get("is_verified") else "unverified",
        "surname_en": "verified" if best_data.get("is_verified") else "unverified",
        "passport_number": "verified" if best_data.get("is_verified") else "unverified",
        "birth_date": "verified" if best_data.get("is_verified") else "unverified",
        "expiry_date": "verified" if best_data.get("is_verified") else "unverified",
        "nationality": "verified" if best_data.get("is_verified") else "unverified",
        # ماكو رقم تحقق لهذولا بمعيار الجواز إطلاقاً — دايماً "غير مؤكد"
        "father_name_en": "unverified",
        "issue_date": "unverified",
    }

    return best_data



# ============================================================================
# Endpoints فحص الخدمة (للإيقاظ)
# ============================================================================

@app.get("/")
def root_check():
    return {"status": "الخدمة شغالة ✓", "ready": True, "version": SERVER_VERSION}


@app.get("/health")
def health_check():
    """endpoint خفيف جداً للإيقاظ من التطبيق."""
    return {"status": "ok", "ready": True, "version": SERVER_VERSION}


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
