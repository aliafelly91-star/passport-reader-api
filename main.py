"""
main.py — خدمة قراءة الجوازات (FastAPI)
Label -> Crop -> Enhance -> OCR + MRZ reconciliation

الإصدار: cloud-label-crop-v3
التعديل الرئيسي:
- GIVEN NAME و SURNAME صاروا يدخلون بمسار القص الموجّه فعلياً.
- اسم الأب وتاريخ الإصدار كذلك.
- الاسم المطبوع يُستخدم لتصحيح MRZ إذا كان MRZ يحتوي ذيل خردة.
- القص صار أضيق لتقليل دخول الحقول المجاورة وتسريع Tesseract.
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
SERVER_VERSION = "cloud-label-crop-v3"

# ============================================================================
# Tesseract
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

TESS_PRINTED_CONFIGS = [
    "--oem 1 --psm 11 -c preserve_interword_spaces=1",
    "--oem 1 --psm 6 -c preserve_interword_spaces=1",
]

MAX_SECONDS = 35
PRINTED_TEXT_BUDGET = 11

# ============================================================================
# ثوابت
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

FUZZY_LABEL_WORDS = [
    "DATE", "OF", "EXPIRY", "ISSUE", "BIRTH", "PLACE", "NAME",
    "NAMES", "SURNAME", "GIVEN", "FATHER", "HUSBAND", "GUARDIAN",
    "SEX", "TYPE", "CODE", "COUNTRY", "NUMBER", "NATIONALITY",
    "AUTHORITY", "TRACKING", "BOOKLET", "CITIZENSHIP", "PASSPORT",
    "HOLDER", "SIGNATURE", "VALID", "UNTIL", "ISSUING",
]

FILLER_LETTERS = ("X", "K", "G")

DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
}

FATHER_LABELS = [
    r"FATHER[' ]?S?\s*NAME",
    r"HUSBAND[' ]?S?\s*NAME",
    r"NAME\s*OF\s*FATHER",
    r"GUARDIAN[' ]?S?\s*NAME",
    r"\bS\s*/\s*O\b",
    r"\bW\s*/\s*O\b",
    r"\bD\s*/\s*O\b",
]

TARGET_LABEL_PHRASES = {
    "given_name_en": [
        ("GIVEN", "NAME"),
        ("GIVEN", "NAMES"),
    ],
    "surname_en": [
        ("SURNAME",),
    ],
    "father_name_en": [
        ("FATHER", "NAME"),
        ("FATHERS", "NAME"),
        ("HUSBAND", "NAME"),
        ("HUSBANDS", "NAME"),
        ("GUARDIAN", "NAME"),
    ],
    "issue_date": [
        ("DATE", "OF", "ISSUE"),
        ("ISSUE", "DATE"),
        ("DATE", "ISSUE"),
    ],
}

PASSPORT_PATTERNS = {
    "PAK": re.compile(r"^[A-Z]{2}\d{7}$"),
    "IRQ": re.compile(r"^[A-Z]\d{7,8}$"),
}

# ============================================================================
# Helpers
# ============================================================================

def edit_distance(a: str, b: str) -> int:
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
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
        previous = current
    return previous[-1]


def safe_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.upper() in BAD_VALUES:
        return ""
    return text


def looks_like_label_word(word: str) -> bool:
    word = (word or "").strip().upper()
    if len(word) < 2:
        return False

    for label in FUZZY_LABEL_WORDS:
        if word == label:
            return True
        if abs(len(word) - len(label)) > 2:
            continue
        tolerance = 2 if len(label) >= 6 else 1
        if edit_distance(word, label) <= tolerance:
            return True
    return False


def is_label_noise(value: str) -> bool:
    text = re.sub(r"\s+", " ", (value or "").strip().upper())
    if not text:
        return True
    words = [w for w in text.split(" ") if w]
    return bool(words) and all(looks_like_label_word(w) for w in words)


def clean_ocr_line(line: str) -> str:
    line = (line or "").upper().strip()
    replacements = {
        "«": "<", "‹": "<", "≤": "<", "—": "<",
        "_": "<", "|": "<", "〈": "<", " ": "",
    }
    for old, new in replacements.items():
        line = line.replace(old, new)
    return re.sub(r"[^A-Z0-9<]", "", line)


def fix_digits_in_letters(text: str) -> str:
    return "".join(DIGIT_TO_LETTER.get(char, char) for char in (text or ""))


def _collapse_filler_letter(section: str, letter: str) -> str:
    if letter not in section:
        return section

    for _ in range(6):
        before = section
        section = re.sub(
            letter + r"{3,}",
            lambda m: "<" * len(m.group(0)),
            section,
        )
        section = re.sub(
            r"(?<=<)" + letter + r"{1,2}(?=<)",
            lambda m: "<" * len(m.group(0)),
            section,
        )
        section = re.sub(
            r"(?<=<)" + letter + r"{1,2}$",
            lambda m: "<" * len(m.group(0)),
            section,
        )
        if section == before:
            break
    return section


def normalize_name_fillers(names_section: str) -> str:
    section = names_section or ""

    if not any(letter in section for letter in FILLER_LETTERS):
        return section

    if "<" not in section and "X" in section:
        section = section.replace("X", "<")

    for letter in FILLER_LETTERS:
        section = _collapse_filler_letter(section, letter)

    return section


def normalize_mrz_line1(l1: str) -> str:
    if not l1 or len(l1) <= 5:
        return l1 or ""

    head = l1[:5]

    if len(head) >= 2 and head[1] in FILLER_LETTERS:
        head = head[0] + "<" + head[2:]

    head = head[:2] + fix_digits_in_letters(head[2:])
    names = fix_digits_in_letters(l1[5:])

    return head + normalize_name_fillers(names)


def trim_filler_tokens(name: str) -> str:
    tokens = (name or "").split()
    removed_any = False

    while tokens:
        last = tokens[-1]

        if len(set(last)) == 1:
            if last[0] in FILLER_LETTERS or len(last) >= 3 or len(last) == 1:
                tokens.pop()
                removed_any = True
                continue
            break

        if removed_any and len(last) <= 2:
            tokens.pop()
            continue

        break

    if removed_any and tokens:
        last = tokens[-1]
        if len(last) >= 5 and last[-1] in FILLER_LETTERS:
            tokens[-1] = last[:-1]

    return " ".join(tokens)


def clean_name_part(raw: str) -> str:
    if not raw:
        return ""

    text = str(raw).upper()
    text = text.replace("<", " ")
    text = re.sub(r"[^A-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = trim_filler_tokens(text)

    if text in BAD_VALUES:
        return ""

    return text


def _collapse_doubles(text: str) -> str:
    return re.sub(r"(.)\1+", r"\1", text or "")


def is_valid_name(name: str) -> bool:
    if not name or len(name) < 2:
        return False

    if not re.fullmatch(r"[A-Z ]{2,60}", name):
        return False

    letters = name.replace(" ", "")
    if len(set(letters)) <= 1:
        return False

    return True


def is_plausible_person_name(name: str) -> bool:
    if not name:
        return False

    clean = re.sub(r"\s+", " ", name.strip().upper())

    if len(clean) < 3:
        return False

    if clean.replace(" ", "") in COUNTRY_NAMES:
        return False

    if clean in COUNTRY_NAMES.values():
        return False

    for word in DOCUMENT_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", clean):
            return False

    if is_label_noise(clean):
        return False

    words = clean.split()

    if len(words) >= 2:
        label_count = sum(1 for w in words if looks_like_label_word(w))
        if label_count * 2 >= len(words):
            return False

    if any(len(w) == 1 for w in words):
        return False

    if sum(1 for w in words if len(w) <= 2) >= 2:
        return False

    if any(w in COUNTRY_NAMES for w in words):
        return False

    return is_valid_name(clean)


def parse_names_from_line1(l1: str):
    if not l1 or len(l1) < 6:
        return "", "", False

    body = l1[5:44]
    has_separator = "<<" in body.rstrip("<")
    body = body.rstrip("<")
    parts = body.split("<<", 1)

    surname = clean_name_part(parts[0]) if len(parts) >= 1 else ""
    given_names = clean_name_part(parts[1]) if len(parts) >= 2 else ""

    if not given_names and "<" in parts[0]:
        tokens = [t for t in parts[0].split("<") if t]
        if len(tokens) >= 2:
            surname = clean_name_part(tokens[0])
            given_names = clean_name_part(" ".join(tokens[1:]))

    return surname, given_names, has_separator


def extract_names(fields, l1: str):
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


def _word_prefix_match(longer: str, shorter: str) -> bool:
    longer_words = longer.split()
    shorter_words = shorter.split()
    if not shorter_words or len(shorter_words) > len(longer_words):
        return False
    return longer_words[:len(shorter_words)] == shorter_words


def reconcile_name(from_mrz: str, from_printed: str) -> str:
    """
    نثق بالمطبوع إذا:
    - MRZ فارغ.
    - النصان متساويان.
    - الفرق حرف/حرفين.
    - المطبوع يطابق بداية MRZ على مستوى الكلمات والـMRZ عنده
      مقطع/مقطعين زائدات بطول صغير (حالة CKEEE مثلاً).
    """
    mrz = clean_name_part(from_mrz)
    printed = clean_name_part(from_printed)

    if not printed or not is_plausible_person_name(printed):
        return mrz

    if not mrz:
        return printed

    if mrz == printed:
        return printed

    if edit_distance(mrz, printed) <= 2:
        return printed

    if mrz.startswith(printed) and len(mrz) - len(printed) <= 8:
        return printed

    if _word_prefix_match(mrz, printed):
        extra = mrz.split()[len(printed.split()):]
        if len(extra) <= 2 and sum(len(x) for x in extra) <= 8:
            return printed

    return mrz


def extract_sex_from_line2(l2: str, fields) -> str:
    if l2 and len(l2) > 20:
        char = l2[20].upper()
        if char in ("M", "F"):
            return char

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
# MRZ
# ============================================================================

def _length_fix_candidates(raw: str, expected: int = 44, max_variants: int = 40):
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
                    candidate = raw[:i] + raw[i + 1:j] + raw[j + 1:]
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


def extract_mrz_candidates(text: str):
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
            or bool(re.match(r"[A-Z][XKG][A-Z]{3}", l1_44))
        )

        if not first_ok:
            continue

        for l2_44 in _length_fix_candidates(l2_raw):
            if sum(c.isdigit() for c in l2_44) >= 8:
                candidates.append((l1_44, l2_44))

    return candidates


def extract_mrz_from_full_text(text: str):
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


def format_date(yymmdd, is_birth=True):
    if not yymmdd:
        return ""

    yymmdd = str(yymmdd).strip()

    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return ""

    try:
        yy = int(yymmdd[:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])

        if not 1 <= mm <= 12 or not 1 <= dd <= 31:
            return ""

        year = (1900 + yy if yy > 30 else 2000 + yy) if is_birth else 2000 + yy
        return f"{year}-{MONTHS[mm - 1]}-{dd:02d}"
    except Exception:
        return ""


def parse_formatted_date(text: str):
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


def calculate_score(is_checksum_ok, fields, l1, l2, surname, given_names):
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

    for name in (surname, given_names):
        tokens = (name or "").split()
        junk = sum(1 for t in tokens if len(t) <= 2 and len(set(t)) == 1)
        score -= junk

    return score


def passport_number_warning(number: str, country_code: str) -> str:
    pattern = PASSPORT_PATTERNS.get((country_code or "").upper())

    if pattern is None or not number:
        return ""

    if pattern.match(number):
        return ""

    return f"شكل رقم الجواز مو معتاد لـ{country_code} — راجعه بعينك"


def try_mrz_candidate(l1, l2):
    try:
        l1 = clean_ocr_line(l1)[:44].ljust(44, "<")
        l2 = clean_ocr_line(l2)[:44].ljust(44, "<")
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

        names_are_complete = is_valid_name(surname) and (
            is_valid_name(given_names) or not has_separator
        )

        result = {
            "success": True,
            "given_name_en": given_names,
            "surname_en": surname,
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
# تجهيز النص المطبوع + Label detection
# ============================================================================

def _prepare_printed_region(img_bgr, top_ratio=0.78):
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


def _extract_words_from_data(data, min_conf=35):
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
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "conf": conf,
            "block_num": data.get("block_num", [0] * n)[i],
            "par_num": data.get("par_num", [0] * n)[i],
            "line_num": data.get("line_num", [0] * n)[i],
        })

    return words


def _words_to_text_lines(words):
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
        text = " ".join(
            w["text"] for w in line_sorted if w["text"].strip()
        )
        if text.strip():
            text_lines.append(text)

    return text_lines


def _split_words_into_columns(words):
    if not words:
        return [words]

    page_left = min(w["left"] for w in words)
    page_right = max(w["left"] + w["width"] for w in words)
    page_width = page_right - page_left

    if page_width <= 0:
        return [words]

    centers = sorted(
        (w["left"] + w["width"] / 2) for w in words
    )

    best_gap = 0
    split_x = None

    for i in range(1, len(centers)):
        gap = centers[i] - centers[i - 1]
        mid = (centers[i] + centers[i - 1]) / 2
        rel = (mid - page_left) / page_width

        if 0.25 <= rel <= 0.75 and gap > best_gap:
            best_gap = gap
            split_x = mid

    if split_x is None or best_gap < page_width * 0.06:
        return [words]

    left_col = [
        w for w in words
        if (w["left"] + w["width"] / 2) < split_x
    ]
    right_col = [
        w for w in words
        if (w["left"] + w["width"] / 2) >= split_x
    ]

    if len(left_col) < 3 or len(right_col) < 3:
        return [words]

    return [left_col, right_col]


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
    """
    Tesseract line_num مو دائماً مثالي، لذلك نجمع بالكود
    حسب top أولاً ثم نفحص عبارات label داخل كل مجموعة.
    """
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w["top"], w["left"]))
    groups = []

    for word in sorted_words:
        placed = False

        for group in groups[-4:]:
            avg_top = sum(x["top"] for x in group) / len(group)
            avg_h = max(8, sum(x["height"] for x in group) / len(group))
            if abs(word["top"] - avg_top) <= avg_h * 0.75:
                group.append(word)
                placed = True
                break

        if not placed:
            groups.append([word])

    return [sorted(g, key=lambda w: w["left"]) for g in groups]


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

                if not all(
                    _label_token_matches(chunk[j]["text"], phrase[j])
                    for j in range(size)
                ):
                    continue

                score = sum(float(w.get("conf", 0)) for w in chunk) / size

                if score > best_score:
                    best_score = score
                    best_box = _bbox_for_words(chunk)

    return best_box


def _crop_value_region(prepared, label_box, field_name):
    """
    القص تحت الـlabel مباشرة.
    القيم أضيق من النسخة القديمة حتى لا تدخل الحقول المجاورة.
    """
    if prepared is None or label_box is None:
        return None

    h, w = prepared.shape[:2]
    x1, y1, x2, y2 = label_box

    label_h = max(12, y2 - y1)
    label_w = max(20, x2 - x1)

    # نبدأ قريب جداً من تحت التسمية.
    left = max(0, int(x1 - 0.018 * w))
    top = max(0, int(y2 + 0.02 * label_h))

    if field_name == "given_name_en":
        crop_w = max(int(label_w * 4.1), int(w * 0.30))
        crop_h = int(label_h * 2.9)

    elif field_name == "father_name_en":
        crop_w = max(int(label_w * 4.6), int(w * 0.33))
        crop_h = int(label_h * 3.0)

    elif field_name == "surname_en":
        crop_w = max(int(label_w * 3.4), int(w * 0.24))
        crop_h = int(label_h * 2.8)

    else:  # issue_date
        crop_w = max(int(label_w * 3.2), int(w * 0.22))
        crop_h = int(label_h * 2.8)

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
        crop = cv2.resize(
            crop, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    gray = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if len(crop.shape) == 3 else crop
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.2, tileGridSize=(8, 8)
    ).apply(gray)

    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.55, blur, -0.55, 0)

    _, otsu = cv2.threshold(
        sharp, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return [sharp, otsu]


def _ocr_field_crop(crop, field_name, deadline=None):
    if field_name in ("given_name_en", "surname_en", "father_name_en"):
        config = (
            "--oem 1 --psm 7 "
            "-c preserve_interword_spaces=1 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
        )
    else:
        config = (
            "--oem 1 --psm 7 "
            "-c preserve_interword_spaces=1 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/ "
        )

    all_lines = []

    for variant in _enhance_field_crop(crop):
        if deadline is not None and time.monotonic() > deadline:
            break

        try:
            data = pytesseract.image_to_data(
                variant,
                config=config,
                lang="eng",
                output_type=pytesseract.Output.DICT,
            )

            words = _extract_words_from_data(data, min_conf=18)
            lines = _words_to_text_lines(words)

            for line in lines:
                line = re.sub(r"\s+", " ", line).strip()
                if line and line not in all_lines:
                    all_lines.append(line)

        except Exception:
            continue

    return all_lines


def _strip_leading_label_from_name(raw: str) -> str:
    text = (raw or "").upper().strip()
    text = re.sub(
        r"^(GIVEN\s+NAMES?|SURNAME|FATHER(?:S)?\s+NAME|"
        r"HUSBAND(?:S)?\s+NAME|GUARDIAN\s+NAME)\s*[:\-]?\s*",
        "",
        text,
    )
    return text.strip()


def reorder_comma_name(raw: str, holder_surname: str = "") -> str:
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


def _best_name_crop_candidate(candidates, holder_surname=""):
    best = ""

    for raw in candidates:
        cleaned = _strip_leading_label_from_name(raw)

        if find_printed_dates(cleaned):
            continue

        name = reorder_comma_name(cleaned, holder_surname)

        if not is_plausible_person_name(name):
            continue

        # نفضّل الاسم الأطول المعقول بدل أول ضجيج قصير.
        if len(name) > len(best):
            best = name

    return best


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
    1) تجهيز الصفحة مرة واحدة.
    2) كشف labels مرة واحدة.
    3) قص GIVEN NAME / SURNAME / FATHER NAME / DATE OF ISSUE.
    4) تحسين كل قصاصة ثم OCR.
    5) مطابقة الاسم واللقب مع MRZ.
    """
    result = {}
    sources = {}
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

        for field_name in (
            "given_name_en",
            "surname_en",
            "father_name_en",
            "issue_date",
        ):
            # الاسم واللقب لازم نقرأهم دائماً للمقارنة مع MRZ.
            # الأب/الإصدار فقط إذا ناقصين.
            if (
                field_name in ("father_name_en", "issue_date")
                and best_data.get(field_name)
            ):
                continue

            if deadline is not None and time.monotonic() > deadline - 0.5:
                break

            label_box = _find_label_box(
                words,
                TARGET_LABEL_PHRASES[field_name],
            )

            if label_box is None:
                continue

            crop = _crop_value_region(
                prepared,
                label_box,
                field_name,
            )

            candidates = _ocr_field_crop(
                crop,
                field_name,
                deadline=deadline,
            )

            if candidates:
                reads += 1

            if field_name in (
                "given_name_en",
                "surname_en",
                "father_name_en",
            ):
                value = _best_name_crop_candidate(
                    candidates,
                    holder_surname,
                )
            else:
                value = _best_date_crop_candidate(
                    candidates,
                    best_data.get("birth_date", ""),
                    best_data.get("expiry_date", ""),
                )

            if not value:
                continue

            if field_name == "given_name_en":
                result[field_name] = reconcile_name(
                    best_data.get("given_name_en", ""),
                    value,
                )
                sources[field_name] = "mrz+printed_crop"

            elif field_name == "surname_en":
                result[field_name] = reconcile_name(
                    best_data.get("surname_en", ""),
                    value,
                )
                sources[field_name] = "mrz+printed_crop"

            else:
                result[field_name] = value
                sources[field_name] = "printed_crop"

        # fallback من نفس words بدون OCR إضافي
        for col in _split_words_into_columns(words):
            printed_text = "\n".join(
                _words_to_text_lines(col)
            )

            if not printed_text:
                continue

            if "father_name_en" not in result:
                father = extract_father_name(
                    printed_text,
                    holder_surname,
                )
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

            if (
                result.get("father_name_en")
                and result.get("issue_date")
            ):
                break

        return (
            result,
            sources,
            reads,
            "تمام" if result else "القص الموجّه ما لقى الحقول",
        )

    except Exception as error:
        return (
            result,
            sources,
            reads,
            f"فشل القص الموجّه: {type(error).__name__}: {error}",
        )


# ============================================================================
# Printed text fallback
# ============================================================================

def iter_printed_texts(img_bgr, deadline=None):
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
                data = pytesseract.image_to_data(
                    prepared,
                    config=config,
                    lang="eng",
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

            try:
                text = pytesseract.image_to_string(
                    prepared,
                    config=config,
                    lang="eng",
                )
            except Exception:
                continue

            if text and text.strip():
                yield text


def find_printed_dates(printed_text: str):
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
        r"\b(\d{1,2})\s*[-/ ]?\s*([A-Z]{3,4})\s*[-/ ]?\s*(\d{4})\b",
        text,
    ):
        month = MONTH_INDEX.get(match.group(2))
        if month:
            add(int(match.group(3)), month, int(match.group(1)))

    for match in re.finditer(
        r"\b([A-Z]{3,4})\s*[-/ ]?\s*(\d{1,2})\s*[-/ ]?\s*(\d{4})\b",
        text,
    ):
        month = MONTH_INDEX.get(match.group(1))
        if month:
            add(int(match.group(3)), month, int(match.group(2)))

    for match in re.finditer(
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
        text,
    ):
        add(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )

    for match in re.finditer(
        r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b",
        text,
    ):
        add(
            int(match.group(3)),
            int(match.group(2)),
            int(match.group(1)),
        )

    return found


def extract_issue_date_by_label(
    printed_text: str,
    birth_date: str,
    expiry_date: str,
) -> str:
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


def extract_issue_date(
    printed_text: str,
    birth_date: str,
    expiry_date: str,
) -> str:
    by_label = extract_issue_date_by_label(
        printed_text,
        birth_date,
        expiry_date,
    )

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
            return min(
                abs(days - five_years),
                abs(days - ten_years),
            )

        best = min(candidates, key=distance_score)

        if distance_score(best) <= tolerance:
            return date_to_text(best)

    return date_to_text(max(candidates))


def extract_printed_names(printed_text: str):
    text = (printed_text or "").upper()

    surname = ""
    given = ""

    surname_match = re.search(
        r"\bSURNAME\b\s*[:\-]?\s*([A-Z][A-Z\-' ]{1,40})",
        text,
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
        r"\bGIVEN\s*NAMES?\b\s*[:\-]?\s*([A-Z][A-Z\-' ]{1,60})",
        text,
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


def extract_father_by_surname_pattern(
    printed_text: str,
    holder_surname: str,
) -> str:
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

        if (
            not before_words
            or _collapse_doubles(before_words[-1]) != surname_key
        ):
            continue

        candidate = reorder_comma_name(
            line,
            holder_surname,
        )

        if is_plausible_person_name(candidate):
            return candidate

    return ""


def extract_father_by_comma_line(
    printed_text: str,
    holder_surname: str,
) -> str:
    lines = [
        re.sub(r"\s+", " ", line).strip().upper()
        for line in (printed_text or "").splitlines()
        if line.strip()
    ]

    for line in lines:
        if "," not in line:
            continue

        if find_printed_dates(line):
            continue

        before, after = line.split(",", 1)
        after_clean = clean_name_part(after)

        if after_clean.replace(" ", "") in COUNTRY_NAMES:
            continue

        if after_clean in COUNTRY_NAMES.values():
            continue

        name = reorder_comma_name(
            line,
            holder_surname,
        )

        if is_plausible_person_name(name):
            return name

    return ""


def extract_father_by_place_anchor(printed_text: str) -> str:
    lines = [
        re.sub(r"\s+", " ", line).strip().upper()
        for line in (printed_text or "").splitlines()
        if line.strip()
    ]

    place_pattern = re.compile(
        r"^([A-Z][A-Z .]{1,28}),\s*([A-Z]{3,20})$"
    )

    for i, raw_line in enumerate(lines):
        line = re.sub(r"^[MF]\s+", "", raw_line)
        line = re.sub(
            r"^PLACE\s+OF\s+BIRTH\s*:?\s*",
            "",
            line,
        )

        match = place_pattern.match(line)

        if not match:
            continue

        tail = match.group(2).strip()

        if (
            tail not in COUNTRY_NAMES
            and tail not in COUNTRY_NAMES.values()
        ):
            continue

        for j in range(i + 1, min(i + 4, len(lines))):
            candidate_line = lines[j]

            if find_printed_dates(candidate_line):
                break

            candidate_line = re.sub(
                r"^(FATHER|HUSBAND|GUARDIAN|MOTHER|SPOUSE)"
                r"[^A-Z]*(NAME)?[^A-Z]*",
                "",
                candidate_line,
            )

            name = reorder_comma_name(
                candidate_line,
                "",
            )

            if is_plausible_person_name(name):
                return name

    return ""


def extract_father_name(
    printed_text: str,
    holder_surname: str = "",
) -> str:
    by_pattern = extract_father_by_surname_pattern(
        printed_text,
        holder_surname,
    )

    if by_pattern:
        return by_pattern

    text = (printed_text or "").upper()

    for label in FATHER_LABELS:
        match = re.search(
            label + r"\s*[:\-]?\s*([A-Z][A-Z,'\.\- ]{2,60})",
            text,
        )

        if not match:
            continue

        raw = match.group(1)

        raw = re.split(
            r"\b(DATE|PLACE|SEX|NATIONALITY|ISSUING|TRACKING|BOOKLET"
            r"|PASSPORT|AUTHORITY|CITIZENSHIP|COUNTRY|COUNTY|CODE|TYPE)\b",
            raw,
        )[0]

        name = reorder_comma_name(
            raw,
            holder_surname,
        )

        if is_plausible_person_name(name):
            return name

    by_anchor = extract_father_by_place_anchor(printed_text)

    if by_anchor:
        return by_anchor

    return extract_father_by_comma_line(
        printed_text,
        holder_surname,
    )


# ============================================================================
# MRZ preprocessing
# ============================================================================

def preprocess_variants(img_bgr, quick=False):
    variants = []
    h, w = img_bgr.shape[:2]
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    if quick:
        for crop_ratio in [0.33, 0.40]:
            y_start = int(h * (1 - crop_ratio))
            crop = img_bgr[y_start:h, 0:w]

            if crop.size == 0:
                continue

            if crop.shape[1] < 1600:
                scale = 1600 / crop.shape[1]
                crop = cv2.resize(
                    crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            gray = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY,
            )

            enhanced = clahe.apply(gray)
            variants.append(enhanced)

            _, otsu = cv2.threshold(
                enhanced,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            variants.append(otsu)

        return variants

    full = img_bgr.copy()

    if full.shape[1] < 1600:
        scale = 1600 / full.shape[1]
        full = cv2.resize(
            full,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray_full = cv2.cvtColor(
        full,
        cv2.COLOR_BGR2GRAY,
    )

    variants.append(gray_full)
    variants.append(clahe.apply(gray_full))

    _, otsu_full = cv2.threshold(
        gray_full,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    variants.append(otsu_full)

    variants.append(
        cv2.adaptiveThreshold(
            gray_full,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )
    )

    for crop_ratio in [0.30, 0.35, 0.40, 0.45]:
        y_start = int(h * (1 - crop_ratio))
        crop = img_bgr[y_start:h, 0:w]

        if crop.size == 0:
            continue

        if crop.shape[1] < 1600:
            scale = 1600 / crop.shape[1]
            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        variants.append(gray)
        variants.append(clahe.apply(gray))

        _, otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        variants.append(otsu)

    return variants


def process_image(img_bgr, deadline=None, quick=False):
    best_score = -1
    best_data = None
    configs = TESS_CONFIGS[:1] if quick else TESS_CONFIGS

    for variant in preprocess_variants(
        img_bgr,
        quick=quick,
    ):
        if deadline is not None and time.monotonic() > deadline:
            break

        for config in configs:
            if deadline is not None and time.monotonic() > deadline:
                break

            try:
                text = pytesseract.image_to_string(
                    variant,
                    config=config,
                    lang="eng",
                )

                candidates = extract_mrz_candidates(text)
                candidates.extend(
                    extract_mrz_from_full_text(text)
                )

                for l1, l2 in candidates:
                    score, data = try_mrz_candidate(
                        l1,
                        l2,
                    )

                    if data is None:
                        continue

                    if score > best_score:
                        best_score = score
                        best_data = data

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
# Finalization
# ============================================================================

def _scan_printed_text(
    best_data,
    printed_text,
    holder_surname,
):
    if not best_data.get("issue_date"):
        issue = extract_issue_date(
            printed_text,
            best_data.get("birth_date", ""),
            best_data.get("expiry_date", ""),
        )

        if issue:
            best_data["issue_date"] = issue

    if not best_data.get("father_name_en"):
        father = extract_father_name(
            printed_text,
            holder_surname,
        )

        if father:
            best_data["father_name_en"] = father

    printed_surname, printed_given = extract_printed_names(
        printed_text
    )

    if printed_given:
        best_data["given_name_en"] = reconcile_name(
            best_data.get("given_name_en", ""),
            printed_given,
        )

    if printed_surname:
        best_data["surname_en"] = reconcile_name(
            best_data.get("surname_en", ""),
            printed_surname,
        )


def _finalize(best_data, best_image, deadline):
    best_data["printed_text_note"] = ""
    best_data["printed_reads"] = 0

    try:
        remaining = deadline - time.monotonic()

        if remaining < 2:
            best_data["printed_text_note"] = (
                f"ماكو وقت كافي للنص المطبوع "
                f"(بقى {remaining:.1f} ثانية)"
            )

        else:
            targeted, sources, reads, note = (
                read_targeted_printed_fields(
                    best_image,
                    best_data,
                    deadline=deadline,
                )
            )

            best_data["printed_reads"] += reads

            # مهم:
            # given_name_en و surname_en نسمح لهم يصححون MRZ.
            # الأب/الإصدار نعبّيهم إذا كانوا فارغين.
            for key, value in targeted.items():
                if not value:
                    continue

                if key in ("given_name_en", "surname_en"):
                    best_data[key] = value
                elif not best_data.get(key):
                    best_data[key] = value

            best_data.setdefault(
                "field_sources",
                {},
            ).update(sources)

            if (
                (
                    not best_data.get("father_name_en")
                    or not best_data.get("issue_date")
                )
                and time.monotonic() < deadline - 3
            ):
                holder_surname = best_data.get(
                    "surname_en",
                    "",
                )

                for printed_text in iter_printed_texts(
                    best_image,
                    deadline=deadline - 1,
                ):
                    best_data["printed_reads"] += 1

                    _scan_printed_text(
                        best_data,
                        printed_text,
                        holder_surname,
                    )

                    break

                if best_data.get("father_name_en"):
                    best_data.setdefault(
                        "field_sources",
                        {},
                    ).setdefault(
                        "father_name_en",
                        "printed_fallback",
                    )

                if best_data.get("issue_date"):
                    best_data.setdefault(
                        "field_sources",
                        {},
                    ).setdefault(
                        "issue_date",
                        "printed_fallback",
                    )

            notes = []

            if not best_data.get("father_name_en"):
                notes.append("ما لقينا اسم الأب")

            if not best_data.get("issue_date"):
                notes.append("ما لقينا تاريخ الإصدار")

            best_data["printed_text_note"] = (
                " و".join(notes) if notes else note
            )

    except Exception as error:
        best_data["printed_text_note"] = (
            f"فشل النص المطبوع: "
            f"{type(error).__name__}: {error}"
        )

    father = best_data.get("father_name_en", "")

    if father and not is_plausible_person_name(father):
        best_data["father_name_en"] = ""

        old_note = best_data.get(
            "printed_text_note",
            "",
        )

        best_data["printed_text_note"] = (
            (old_note + " · " if old_note and old_note != "تمام" else "")
            + "اسم الأب انرفض (تسمية مو قيمة)"
        )

    for key in [
        "given_name_en",
        "surname_en",
        "father_name_en",
        "passport_number",
        "nationality",
        "residence_country",
        "birth_date",
        "expiry_date",
        "issue_date",
        "sex",
    ]:
        best_data[key] = safe_text(
            best_data.get(key)
        )

    best_data["given_name_en"] = trim_filler_tokens(
        best_data["given_name_en"]
    )

    best_data["surname_en"] = trim_filler_tokens(
        best_data["surname_en"]
    )

    best_data["father_name_en"] = trim_filler_tokens(
        best_data["father_name_en"]
    )

    best_data["needs_review"] = bool(
        not best_data.get("father_name_en")
        or not best_data.get("issue_date")
    )

    best_data["fields_confidence"] = {
        "given_name_en": (
            "verified"
            if best_data.get("is_verified")
            else "unverified"
        ),
        "surname_en": (
            "verified"
            if best_data.get("is_verified")
            else "unverified"
        ),
        "passport_number": (
            "verified"
            if best_data.get("is_verified")
            else "unverified"
        ),
        "birth_date": (
            "verified"
            if best_data.get("is_verified")
            else "unverified"
        ),
        "expiry_date": (
            "verified"
            if best_data.get("is_verified")
            else "unverified"
        ),
        "nationality": (
            "verified"
            if best_data.get("is_verified")
            else "unverified"
        ),
        "father_name_en": (
            "printed"
            if best_data.get("father_name_en")
            else "unverified"
        ),
        "issue_date": (
            "printed"
            if best_data.get("issue_date")
            else "unverified"
        ),
    }

    return best_data


def read_passport_from_bytes(image_bytes: bytes) -> dict:
    request_started = time.monotonic()

    def finish(data):
        if isinstance(data, dict):
            data["ocr_time_ms"] = int(
                (time.monotonic() - request_started) * 1000
            )
            data["server_version"] = SERVER_VERSION
        return data

    np_array = np.frombuffer(
        image_bytes,
        np.uint8,
    )

    img_bgr = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR,
    )

    if img_bgr is None:
        return finish({
            "success": False,
            "error": "تعذر فك ترميز الصورة",
        })

    best_data = None
    best_image = img_bgr

    deadline = time.monotonic() + MAX_SECONDS
    mrz_deadline = deadline - PRINTED_TEXT_BUDGET

    def better(candidate):
        if candidate is None:
            return False
        if best_data is None:
            return True
        return (
            candidate.get("score", 0)
            > best_data.get("score", 0)
        )

    # 1) الأصلية سريع
    data = process_image(
        img_bgr,
        deadline=mrz_deadline,
        quick=True,
    )

    if better(data):
        best_data = data
        best_image = img_bgr

    if best_data is not None and best_data.get("is_fully_verified"):
        return finish(
            _finalize(
                best_data,
                best_image,
                deadline,
            )
        )

    # 2) 180°
    if time.monotonic() < mrz_deadline:
        flipped = cv2.rotate(
            img_bgr,
            cv2.ROTATE_180,
        )

        data = process_image(
            flipped,
            deadline=mrz_deadline,
            quick=True,
        )

        if better(data):
            best_data = data
            best_image = flipped

        if best_data is not None and best_data.get("is_fully_verified"):
            return finish(
                _finalize(
                    best_data,
                    best_image,
                    deadline,
                )
            )

    # 3) شامل
    for candidate_image in [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_180),
    ]:
        if time.monotonic() > mrz_deadline:
            break

        data = process_image(
            candidate_image,
            deadline=mrz_deadline,
            quick=False,
        )

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

    return finish(
        _finalize(
            best_data,
            best_image,
            deadline,
        )
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
def root_check():
    return {
        "status": "الخدمة شغالة ✓",
        "ready": True,
        "version": SERVER_VERSION,
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "ready": True,
        "version": SERVER_VERSION,
    }


@app.post("/read-passport")
async def read_passport_endpoint(
    file: UploadFile = File(...),
):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "الصورة فارغة",
                },
            )

        result = read_passport_from_bytes(
            image_bytes
        )

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"خطأ داخلي: {str(e)}",
            },
        )
