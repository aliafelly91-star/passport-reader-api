"""FastAPI passport OCR service - crop-first optimized reader.

This version keeps the app's successful strategy: Flutter detects/crops each
printed field locally, then Python reads the small value crops. MRZ is used for
structural fields and as a cross-check, not as the only name source.
"""

import logging
import re
import time
from datetime import date
from typing import Optional

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from mrz.checker.td3 import TD3CodeChecker

app = FastAPI(title="Passport Reader API")
SERVER_VERSION = "cloud-app-crop-v13"
logger = logging.getLogger(__name__)

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
MONTH_INDEX = {m: i + 1 for i, m in enumerate(MONTHS)}
MONTH_INDEX["SEPT"] = 9
_MONTH_ABBR = set(MONTHS) | {"SEPT"}

COUNTRY_NAMES = {
    "IRQ":"IRAQ","PAK":"PAKISTAN","IND":"INDIA","AFG":"AFGHANISTAN",
    "IRN":"IRAN","SYR":"SYRIA","EGY":"EGYPT","JOR":"JORDAN",
    "LBN":"LEBANON","SAU":"SAUDI ARABIA","ARE":"UNITED ARAB EMIRATES",
    "KWT":"KUWAIT","QAT":"QATAR","BHR":"BAHRAIN","OMN":"OMAN",
    "YEM":"YEMEN","TUR":"TURKEY","PSE":"PALESTINE","BGD":"BANGLADESH",
    "PHL":"PHILIPPINES","LKA":"SRI LANKA","NPL":"NEPAL","ETH":"ETHIOPIA",
    "SDN":"SUDAN","SOM":"SOMALIA","MAR":"MOROCCO","DZA":"ALGERIA",
    "TUN":"TUNISIA","LBY":"LIBYA","USA":"UNITED STATES","GBR":"UNITED KINGDOM",
    "CAN":"CANADA","FRA":"FRANCE","DEU":"GERMANY",
}
NATIONALITY_NAMES = {
    "IRQ":"IRAQI","PAK":"PAKISTANI","IND":"INDIAN","AFG":"AFGHAN",
    "IRN":"IRANIAN","SYR":"SYRIAN","EGY":"EGYPTIAN","JOR":"JORDANIAN",
    "LBN":"LEBANESE","SAU":"SAUDI","ARE":"EMIRATI","KWT":"KUWAITI",
    "QAT":"QATARI","BHR":"BAHRAINI","OMN":"OMANI","YEM":"YEMENI",
    "TUR":"TURKISH","PSE":"PALESTINIAN","BGD":"BANGLADESHI","PHL":"FILIPINO",
    "LKA":"SRI LANKAN","NPL":"NEPALESE","ETH":"ETHIOPIAN","SDN":"SUDANESE",
    "SOM":"SOMALI","MAR":"MOROCCAN","DZA":"ALGERIAN","TUN":"TUNISIAN",
    "LBY":"LIBYAN","USA":"AMERICAN","GBR":"BRITISH","CAN":"CANADIAN",
    "FRA":"FRENCH","DEU":"GERMAN",
}

LABEL_WORDS = {
    "DATE","OF","EXPIRY","EXPIRATION","ISSUE","ISSUANCE","BIRTH","PLACE",
    "NAME","NAMES","SURNAME","GIVEN","FATHER","HUSBAND","GUARDIAN","SEX",
    "TYPE","CODE","COUNTRY","NUMBER","NO","NATIONALITY","AUTHORITY",
    "PASSPORT","HOLDER","SIGNATURE","VALID","UNTIL","ISSUING",
}


def _safe(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NONE", "NULL", "NAN", "N/A", "NA", "-", "--"} else s


def _decode(data: bytes, gray=True):
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(arr, flag)
    return img if img is not None and img.size else None


def _resize_min_width(gray, width):
    if gray is None:
        return None
    if gray.shape[1] >= width:
        return gray
    scale = width / max(1, gray.shape[1])
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _field_variants(gray, width=900):
    """Return complementary variants without over-upscaling faint text."""
    if gray is None:
        return []

    # Medium 2x image preserves the original stroke shapes. Some very faint
    # Pakistani scans read worse when expanded straight to 900px.
    medium = gray
    if gray.shape[1] < 450:
        medium = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    medium_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(medium)
    _, medium_strong_otsu = cv2.threshold(
        medium_strong, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    img = _resize_min_width(gray, width)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(img)
    strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(img)
    blur = cv2.GaussianBlur(clahe, (0, 0), 0.7)
    sharp = cv2.addWeighted(clahe, 1.40, blur, -0.40, 0)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, strong_otsu = cv2.threshold(strong, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [img, sharp, otsu, strong_otsu, medium_strong_otsu]

def _ocr(img, whitelist, psm=7, timeout=6):
    if img is None:
        return ""
    cfg = (
        f'--oem 1 --psm {psm} '
        f'-c preserve_interword_spaces=1 '
        f'-c tessedit_char_whitelist="{whitelist}"'
    )
    try:
        text = pytesseract.image_to_string(img, config=cfg, lang="eng", timeout=timeout)
        return re.sub(r"\s+", " ", text).strip().upper()
    except Exception as exc:
        logger.warning("Field OCR failed (%s)", type(exc).__name__)
        return ""


def _strip_label(text, kind):
    s = re.sub(r"\s+", " ", (text or "").upper()).strip()
    patterns = {
        "given": r"^(?:GIVEN\s+NAMES?|FIRST\s+NAME)\s*[:\-]?\s*",
        "surname": r"^(?:SURNAME|FAMILY\s+NAME|LAST\s+NAME)\s*[:\-]?\s*",
        "father": r"^(?:FATHER(?:'?S)?\s+NAME|HUSBAND(?:'?S)?\s+NAME|GUARDIAN\s+NAME)\s*[:\-]?\s*",
        "birth": r"^(?:DATE\s+OF\s+BIRTH|BIRTH\s+DATE)\s*[:\-]?\s*",
        "issue": r"^(?:DATE\s+OF\s+ISSUE|ISSUE\s+DATE|DATE\s+OF\s+ISSUANCE)\s*[:\-]?\s*",
        "expiry": r"^(?:DATE\s+OF\s+EXPIRY|EXPIRY\s+DATE|DATE\s+OF\s+EXPIRATION)\s*[:\-]?\s*",
        "nationality": r"^NATIONALITY\s*[:\-]?\s*",
        "sex": r"^SEX\s*[:\-]?\s*",
        "passport": r"^(?:PASSPORT|DOCUMENT)\s+(?:NO\.?|NUMBER)\s*[:\-]?\s*",
    }
    p = patterns.get(kind)
    return re.sub(p, "", s).strip() if p else s


def _clean_name(text):
    s = re.sub(r"[^A-Z ,'\-]", " ", (text or "").upper())
    s = re.sub(r"\s+", " ", s).strip(" ,-'")
    if "," in s:
        a, b = s.split(",", 1)
        a = re.sub(r"[^A-Z ]", " ", a).strip()
        b = re.sub(r"[^A-Z ]", " ", b).strip()
        s = f"{b} {a}".strip() if a and b else (b or a)
    s = re.sub(r"[^A-Z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _plausible_name(name):
    if not name or len(name) < 2 or not re.fullmatch(r"[A-Z ]{2,60}", name):
        return False
    words = name.split()
    if not words or any(len(w) == 1 for w in words):
        return False
    if sum(1 for w in words if w in LABEL_WORDS) >= max(1, len(words) // 2):
        return False
    if any(w in COUNTRY_NAMES or w in COUNTRY_NAMES.values() for w in words):
        return False
    letters = name.replace(" ", "")
    if len(set(letters)) <= 1:
        return False
    # Reject the exact MRZ filler pattern seen in the bad reads: K K K / KKKK...
    if sum(1 for ch in letters if ch == "K") / max(1, len(letters)) > 0.65:
        return False
    return True


def _name_has_mrz_noise(name):
    n = _clean_name(name)
    if not _plausible_name(n):
        return True
    words = n.split()
    if len(words) > 5:
        return True

    for w in words:
        if len(w) == 1:
            return True
        # OCR frequently turns MRZ filler '<' into K/X/S and sometimes one
        # stray E/C. Catch tails like KKKKKSKSEKK instead of accepting them
        # as a real second name.
        if len(w) >= 4:
            filler_ratio = sum(ch in "KXS" for ch in w) / len(w)
            if filler_ratio >= 0.55:
                return True
        if re.fullmatch(r"[KXS]{2,}", w):
            return True
    return False


def _strip_name_noise_tokens(name):
    """Drop tiny OCR remnants of printed labels without changing real name words."""
    n = _clean_name(name)
    if not n:
        return ""
    words = n.split()
    # Common remnants from labels: GIVEN NAME -> G N, FATHER NAME -> F N,
    # PASSPORT -> P.  Remove only isolated one-letter tokens.
    words = [w for w in words if not (len(w) == 1 and w in {"G", "N", "F", "P"})]
    return " ".join(words).strip()


def _name_candidate_score(raw, kind):
    stripped = _strip_label(raw, kind)
    clean = _strip_name_noise_tokens(stripped)
    if not _plausible_name(clean):
        return (-999, clean)
    words = clean.split()
    score = 20 + min(len(clean), 30)
    if len(words) <= 4:
        score += 5
    if any(re.fullmatch(r"[KXS]{2,}", w) for w in words):
        score -= 20
    if sum(ch == "K" for ch in clean.replace(" ", "")) > 4:
        score -= 12
    return (score, clean)


def _choose_ocr_name_candidate(reads, kind):
    candidates = []
    for raw in reads:
        if not raw:
            continue
        score, clean = _name_candidate_score(raw, kind)
        if score <= -999 or not clean:
            continue
        candidates.append((clean, score))
    if not candidates:
        return ""

    # Exact agreement between independent OCR passes is the strongest signal.
    counts = {}
    best_score = {}
    for clean, score in candidates:
        counts[clean] = counts.get(clean, 0) + 1
        best_score[clean] = max(best_score.get(clean, -999), score)
    repeated = [name for name, count in counts.items() if count >= 2]
    if repeated:
        return max(repeated, key=lambda name: (counts[name], best_score[name], len(name)))

    # Otherwise prefer the candidate closest to the others, not simply the
    # longest string (which often contains label noise).
    unique = list(counts)
    def rank(name):
        distance = sum(_edit(name, other) for other in unique if other != name)
        return (-distance, best_score[name], len(name))
    return max(unique, key=rank)


def _edit(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _choose_name(mrz_name, printed_name, mrz_verified=False):
    """
    The focused printed crop is authoritative for names.
    MRZ is only a fallback when the crop failed.
    """
    m = _clean_name(mrz_name)
    p = _clean_name(printed_name)

    if _plausible_name(p):
        return p
    return m


def _find_dates(text):
    s = (text or "").upper()
    out = []

    def add(y, m, d):
        try:
            v = date(y, m, d)
            if 1900 <= y <= 2100 and v not in out:
                out.append(v)
        except Exception:
            pass

    for m in re.finditer(r"\b(\d{1,2})\s*[-/ ]?\s*([A-Z]{3,4})\s*[-/ ]?\s*(\d{4})\b", s):
        mm = MONTH_INDEX.get(m.group(2))
        if mm:
            add(int(m.group(3)), mm, int(m.group(1)))
    for m in re.finditer(r"\b([A-Z]{3,4})\s*[-/ ]?\s*(\d{1,2})\s*[-/ ]?\s*(\d{4})\b", s):
        mm = MONTH_INDEX.get(m.group(1))
        if mm:
            add(int(m.group(3)), mm, int(m.group(2)))
    for m in re.finditer(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", s):
        add(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in re.finditer(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b", s):
        add(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return out


def _date_text(v):
    return f"{v.year}-{MONTHS[v.month - 1]}-{v.day:02d}"


def _format_mrz_date(v, birth):
    s = _safe(v)
    if len(s) != 6 or not s.isdigit():
        return ""
    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
    if not 1 <= mm <= 12 or not 1 <= dd <= 31:
        return ""
    year = (1900 + yy if yy > 30 else 2000 + yy) if birth else 2000 + yy
    try:
        return _date_text(date(year, mm, dd))
    except Exception:
        return ""


def _value_band(gray, kind):
    """Keep the vertical band where the requested value lives inside app crop."""
    if gray is None or gray.size == 0:
        return gray
    h = gray.shape[0]
    ranges = {
        "given": (0.00, 0.62),
        "surname": (0.00, 0.62),
        "father": (0.00, 0.72),
        "passport": (0.00, 0.68),
        "nationality": (0.00, 0.68),
        "sex": (0.00, 0.68),
    }
    if kind not in ranges:
        return gray
    y0, y1 = ranges[kind]
    a = max(0, int(h * y0))
    b = min(h, max(a + 4, int(h * y1)))
    return gray[a:b, :]


def _quick_variant(gray, width=900, clahe=False, otsu=False):
    img = _resize_min_width(gray, width)
    if clahe:
        img = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(img)
    if otsu:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def _ocr_crop(data: bytes, kind: str):
    """Fast crop OCR: 1 primary pass, one fallback only when needed."""
    gray = _decode(data)
    if gray is None:
        return ""

    if kind in {"given", "surname", "father"}:
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ,'- "
        band = _value_band(gray, kind)

        first_psm = 7 if kind in {"given", "surname"} else 6
        first = _ocr(_quick_variant(band, 1000), wl, psm=first_psm, timeout=4)
        first_clean = _strip_name_noise_tokens(_strip_label(first, kind))
        if _plausible_name(first_clean):
            return first_clean

        second = _ocr(
            _quick_variant(band, 1100, clahe=True),
            wl,
            psm=11 if kind == "father" else 7,
            timeout=4,
        )
        second_clean = _strip_name_noise_tokens(_strip_label(second, kind))
        if _plausible_name(second_clean):
            return second_clean

        return ""

    if kind in {"birth", "issue", "expiry"}:
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/."
        # Full crop first: on clear scans it keeps enough context for Tesseract.
        first = _ocr(_quick_variant(gray, 900), wl, psm=11, timeout=4)
        if _find_dates(_strip_label(first, kind)):
            return first

        # Faint green Pakistani scans improve strongly with local contrast.
        second = _ocr(_quick_variant(gray, 900, clahe=True), wl, psm=11, timeout=4)
        if _find_dates(_strip_label(second, kind)):
            return second

        # Last date-only fallback: remove the lower neighbouring field/noise.
        top = gray[:max(4, int(gray.shape[0] * 0.78)), :]
        third = _ocr(_quick_variant(top, 900, otsu=True), wl, psm=11, timeout=4)
        return third or second or first

    if kind == "passport":
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
        band = _value_band(gray, kind)

        first = _ocr(_quick_variant(band, 1000), wl, psm=7, timeout=4)
        flat = re.sub(r"[^A-Z0-9]", "", first.upper())
        if re.fullmatch(r"[A-Z0-9]{6,12}", flat) and any(ch.isdigit() for ch in flat):
            return flat

        second = _ocr(
            _quick_variant(band, 1200, clahe=True),
            wl,
            psm=7,
            timeout=4,
        )
        flat2 = re.sub(r"[^A-Z0-9]", "", second.upper())
        if re.fullmatch(r"[A-Z0-9]{6,12}", flat2) and any(ch.isdigit() for ch in flat2):
            return flat2
        return ""

    if kind == "nationality":
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        band = _value_band(gray, kind)
        return _ocr(_quick_variant(band, 800), wl, psm=11, timeout=4)

    if kind == "sex":
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        band = _value_band(gray, kind)
        return _ocr(_quick_variant(band, 500), wl, psm=11, timeout=3)

    return ""


def _clean_mrz_line(line):
    s = (line or "").upper()
    s = s.replace("«", "<").replace("‹", "<").replace("_", "<").replace("|", "<")
    return re.sub(r"[^A-Z0-9<]", "", s)


def _mrz_names(l1):
    body = l1[5:44].rstrip("<")
    parts = body.split("<<", 1)
    sur = _clean_name(parts[0].replace("<", " "))
    given_block = parts[1].split("<<", 1)[0] if len(parts) > 1 else ""
    giv = _clean_name(given_block.replace("<", " "))
    return sur, giv


def _looks_like_td3(l1, l2):
    if len(l1) != 44 or len(l2) != 44:
        return False
    if not re.match(r"^P[A-Z<]", l1):
        return False
    code = l1[2:5]
    if not re.fullmatch(r"[A-Z]{3}", code) or code in _MONTH_ABBR:
        return False
    if "<<" not in l1[5:]:
        return False
    if not re.fullmatch(r"[A-Z]{3}", l2[10:13]) or l2[10:13] in _MONTH_ABBR:
        return False
    if not re.fullmatch(r"[0-9]{6}", l2[13:19]):
        return False
    if l2[20] not in "MF<":
        return False
    if not re.fullmatch(r"[0-9]{6}", l2[21:27]):
        return False
    return True


def _digits_only_ocr(text):
    """Normalize common OCR confusions only where a numeric MRZ field is expected."""
    table = str.maketrans({"O":"0", "Q":"0", "D":"0", "I":"1", "L":"1", "Z":"2", "S":"5", "B":"8", "G":"6"})
    return (text or "").translate(table)


def _mrz_name_block(block):
    """Decode < separated MRZ name words and stop when OCR filler noise starts."""
    parts = [p for p in re.split(r"<+", block or "") if p]
    words = []
    for token in parts:
        t = re.sub(r"[^A-Z]", "", token.upper())
        if not t:
            continue
        # Padding '<' is commonly misread as K/X/S. Once such a token appears
        # after a real name word, the rest is padding noise.
        if words and (re.fullmatch(r"[KXS]{1,}", t) or len(t) > 12 and set(t) <= set("KXS")):
            break
        if len(t) == 1 and t in {"K", "X", "S"} and words:
            break
        words.append(t)
    return _strip_name_noise_tokens(" ".join(words))


def _parse_mrz_fuzzy(l1, l2):
    """Best-effort TD3 parser when one OCR error makes TD3CodeChecker reject the pair."""
    a = _clean_mrz_line(l1)
    b = _clean_mrz_line(l2)
    if len(a) < 20 or not a.startswith("P"):
        return None

    # Find country code from the first line. Pakistan is the common case, but
    # keep the parser generic for known three-letter codes.
    country = ""
    country_pos = -1
    for code in COUNTRY_NAMES:
        idx = a.find(code, 1, 8)
        if idx >= 0:
            country, country_pos = code, idx
            break
    if not country:
        m = re.search(r"^P<?([A-Z]{3})", a)
        if m:
            country = m.group(1)
            country_pos = m.start(1)
    if not country:
        return None

    name_start = country_pos + 3
    name_body = a[name_start:]
    # OCR may produce <<< instead of <<. One or more extra fillers are harmless.
    parts = re.split(r"<{2,}", name_body, maxsplit=1)
    if len(parts) < 2:
        return None
    sur = _mrz_name_block(parts[0])
    giv = _mrz_name_block(parts[1])
    if not _plausible_name(sur):
        sur = ""
    if not _plausible_name(giv):
        giv = ""

    # Second line: locate nationality code instead of trusting exact columns.
    nat_pos = -1
    nat = ""
    for code in NATIONALITY_NAMES:
        idx = b.find(code)
        if 7 <= idx <= 14:
            nat, nat_pos = code, idx
            break
    if nat_pos < 0:
        # Country code is usually also the nationality code.
        idx = b.find(country)
        if 7 <= idx <= 14:
            nat, nat_pos = country, idx
    if nat_pos < 0:
        return None

    prefix = b[:nat_pos]
    # Standard TD3 = 9-char document number + one check digit before nationality.
    number = re.sub(r"<", "", prefix[:9]).upper()
    if len(number) < 6:
        return None

    tail = b[nat_pos + 3:]
    # Birth date is the first six-character numeric-like block after nationality.
    m_birth = re.search(r"[0-9OQDILZSBG]{6}", tail)
    if not m_birth:
        return None
    birth6 = _digits_only_ocr(m_birth.group(0))
    after_birth = tail[m_birth.end():]

    sex = ""
    sex_index = -1
    for i, ch in enumerate(after_birth[:4]):
        if ch in "MF<":
            sex = "" if ch == "<" else ch
            sex_index = i
            break
    if sex_index < 0:
        # If check digit was swallowed, sex may immediately follow the date.
        sex_index = 0
    after_sex = after_birth[sex_index + 1:]
    m_exp = re.search(r"[0-9OQDILZSBG]{6}", after_sex)
    expiry6 = _digits_only_ocr(m_exp.group(0)) if m_exp else ""

    birth_date = _format_mrz_date(birth6, True)
    expiry_date = _format_mrz_date(expiry6, False) if expiry6 else ""
    if not birth_date:
        return None

    return {
        "success": True,
        "given_name_en": giv,
        "surname_en": sur,
        "father_name_en": "",
        "passport_number": number,
        "nationality": NATIONALITY_NAMES.get(nat or country, nat or country),
        "residence_country": COUNTRY_NAMES.get(country, country),
        "birth_date": birth_date,
        "issue_date": "",
        "expiry_date": expiry_date,
        "sex": sex,
        "is_verified": False,
        "is_fully_verified": False,
        "score": 5 + sum(bool(x) for x in [sur, giv, number, birth_date, expiry_date, sex]),
        "mrz_line1": a[:44].ljust(44, "<"),
        "mrz_line2": b[:44].ljust(44, "<"),
        "mrz_fuzzy": True,
    }


def _prefer_mrz_candidate(current, candidate):
    if current is None:
        return candidate
    if candidate is None:
        return current

    # Prefer the shorter name when the only extra tail is likely filler K/X/S.
    for key in ("given_name_en", "surname_en"):
        a = _clean_name(current.get(key, ""))
        b = _clean_name(candidate.get(key, ""))
        if a and b and a != b:
            if len(a) > len(b) and a.startswith(b) and re.fullmatch(r"[KXS]+", a[len(b):]):
                return candidate
            if len(b) > len(a) and b.startswith(a) and re.fullmatch(r"[KXS]+", b[len(a):]):
                return current

    return candidate if candidate.get("score", 0) > current.get("score", 0) else current


def _try_mrz(l1, l2):
    l1 = _clean_mrz_line(l1)[:44].ljust(44, "<")
    l2 = _clean_mrz_line(l2)[:44].ljust(44, "<")
    if not _looks_like_td3(l1, l2):
        return None
    try:
        checker = TD3CodeChecker(f"{l1}\n{l2}", check_expiry=False)
        fields = checker.fields()
        try:
            verified = bool(checker)
        except Exception:
            verified = False

        sur, giv = _mrz_names(l1)
        country = _safe(getattr(fields, "country", "")).upper()
        nat = _safe(getattr(fields, "nationality", "")).upper() or country
        number = _safe(getattr(fields, "document_number", "")).replace("<", "").upper()
        sex = l2[20] if len(l2) > 20 and l2[20] in "MF" else ""

        score = (8 if verified else 0)
        score += sum(bool(x) for x in [sur, giv, number, sex])
        if sur and not _name_has_mrz_noise(sur):
            score += 2
        if giv and not _name_has_mrz_noise(giv):
            score += 2

        return {
            "success": True,
            "given_name_en": giv,
            "surname_en": sur,
            "father_name_en": "",
            "passport_number": number,
            "nationality": NATIONALITY_NAMES.get(nat, nat),
            "residence_country": COUNTRY_NAMES.get(country, country),
            "birth_date": _format_mrz_date(getattr(fields, "birth_date", ""), True),
            "issue_date": "",
            "expiry_date": _format_mrz_date(getattr(fields, "expiry_date", ""), False),
            "sex": sex,
            "is_verified": verified,
            "is_fully_verified": verified and bool(sur) and bool(number),
            "score": score,
            "mrz_line1": l1,
            "mrz_line2": l2,
        }
    except Exception:
        return None


def _mrz_from_bytes(data: bytes):
    gray = _decode(data)
    if gray is None:
        return None, ""

    gray = _resize_min_width(gray, 1600)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Two bounded passes only. The older version could spend ~28 seconds on
    # MRZ alone, then run 15+ field OCR passes and hit Flutter's 60s timeout.
    passes = [
        (clahe, 11),
        (otsu, 6),
    ]
    best = None
    debug = ""

    for image_variant, psm in passes:
        cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
        try:
            text = pytesseract.image_to_string(
                image_variant, config=cfg, lang="eng", timeout=6
            )
        except Exception as exc:
            logger.warning("MRZ OCR failed (%s)", type(exc).__name__)
            continue

        if text and not debug:
            debug = text.strip()

        lines = [_clean_mrz_line(x) for x in text.splitlines()]
        lines = [x for x in lines if len(x) >= 20]
        for i in range(len(lines) - 1):
            strict = _try_mrz(lines[i], lines[i + 1])
            fuzzy = _parse_mrz_fuzzy(lines[i], lines[i + 1])
            best = _prefer_mrz_candidate(best, strict)
            best = _prefer_mrz_candidate(best, fuzzy)

        # Names do not need to be perfect here: focused printed crops correct
        # them later. Stop as soon as the structural MRZ data is usable.
        if (
            best
            and best.get("passport_number")
            and best.get("birth_date")
            and best.get("expiry_date")
            and best.get("nationality")
        ):
            return best, debug

    return best, debug


def _blank():
    return {
        "success": True,
        "given_name_en": "",
        "surname_en": "",
        "father_name_en": "",
        "passport_number": "",
        "nationality": "",
        "residence_country": "",
        "birth_date": "",
        "issue_date": "",
        "expiry_date": "",
        "sex": "",
        "score": 0,
        "is_verified": False,
        "is_fully_verified": False,
        "field_sources": {},
    }


async def _bytes(upload: Optional[UploadFile]):
    if upload is None:
        return b""
    try:
        return await upload.read()
    except Exception:
        return b""


@app.get("/")
def root():
    return {"status": "الخدمة شغالة ✓", "ready": True, "version": SERVER_VERSION}


@app.get("/health")
def health():
    return {"status": "ok", "ready": True, "version": SERVER_VERSION}


@app.post("/read-passport-fields")
async def read_passport_fields(
    client_crop_mode: str = Form(""),
    given_name_crop: Optional[UploadFile] = File(None),
    surname_crop: Optional[UploadFile] = File(None),
    father_name_crop: Optional[UploadFile] = File(None),
    passport_number_crop: Optional[UploadFile] = File(None),
    birth_date_crop: Optional[UploadFile] = File(None),
    issue_date_crop: Optional[UploadFile] = File(None),
    expiry_date_crop: Optional[UploadFile] = File(None),
    nationality_crop: Optional[UploadFile] = File(None),
    sex_crop: Optional[UploadFile] = File(None),
    mrz_crop: Optional[UploadFile] = File(None),
):
    started = time.monotonic()
    uploads = {
        "given_name_crop": given_name_crop,
        "surname_crop": surname_crop,
        "father_name_crop": father_name_crop,
        "passport_number_crop": passport_number_crop,
        "birth_date_crop": birth_date_crop,
        "issue_date_crop": issue_date_crop,
        "expiry_date_crop": expiry_date_crop,
        "nationality_crop": nationality_crop,
        "sex_crop": sex_crop,
        "mrz_crop": mrz_crop,
    }
    data = {k: await _bytes(v) for k, v in uploads.items()}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": "ما وصلت أي قصاصة",
            "server_version": SERVER_VERSION,
        })

    result = _blank()
    src = result["field_sources"]
    debug = {}

    # MRZ first: one successful pass gives passport/date/nationality/sex quickly.
    if data.get("mrz_crop"):
        mrz, raw = _mrz_from_bytes(data["mrz_crop"])
        if raw:
            debug["mrz_crop"] = re.sub(r"\s+", " ", raw)[:350]
        if mrz:
            for k in ("given_name_en", "surname_en", "passport_number", "nationality", "residence_country", "birth_date", "expiry_date", "sex"):
                if mrz.get(k):
                    result[k] = mrz[k]
                    src[k] = "mrz_crop"
            for k in ("score", "is_verified", "is_fully_verified", "mrz_line1", "mrz_line2"):
                if k in mrz:
                    result[k] = mrz[k]

    # Printed name crops are used as correction/cross-check. Father name always
    # comes from printed crop because it is not part of TD3 MRZ.
    for field, kind, outkey in (
        ("given_name_crop", "given", "given_name_en"),
        ("surname_crop", "surname", "surname_en"),
        ("father_name_crop", "father", "father_name_en"),
    ):
        if not data.get(field):
            continue
        raw = _strip_label(_ocr_crop(data[field], kind), kind)
        if raw:
            debug[field] = raw[:140]
        val = _clean_name(raw)
        if not _plausible_name(val):
            continue

        if outkey in ("given_name_en", "surname_en"):
            old = result.get(outkey, "")
            chosen = _choose_name(old, val, bool(result.get("is_verified")))
            result[outkey] = chosen
            if chosen == val and chosen != old:
                src[outkey] = "app_crop"
            elif old:
                src[outkey] = "mrz+app_crop"
            else:
                src[outkey] = "app_crop"
        else:
            result[outkey] = val
            src[outkey] = "app_crop"

    for field, kind, outkey in (
        ("issue_date_crop", "issue", "issue_date"),
        ("birth_date_crop", "birth", "birth_date"),
        ("expiry_date_crop", "expiry", "expiry_date"),
    ):
        if not data.get(field):
            continue
        # Issue date is absent from TD3. Birth/expiry use crop only if MRZ missed.
        if outkey != "issue_date" and result.get(outkey):
            continue
        raw = _strip_label(_ocr_crop(data[field], kind), kind)
        if raw:
            debug[field] = raw[:120]
        dates = _find_dates(raw)
        if dates:
            result[outkey] = _date_text(dates[0])
            src[outkey] = "app_crop"

    if data.get("passport_number_crop") and not result.get("passport_number"):
        raw = _strip_label(_ocr_crop(data["passport_number_crop"], "passport"), "passport")
        if raw:
            debug["passport_number_crop"] = raw[:100]
        flat = re.sub(r"[^A-Z0-9]", "", raw.upper())
        tokens = re.findall(r"[A-Z0-9]{6,12}", flat)
        tokens = [t for t in tokens if any(c.isdigit() for c in t)]
        if tokens:
            result["passport_number"] = min(tokens, key=lambda x: abs(len(x) - 9))
            src["passport_number"] = "app_crop"

    if data.get("sex_crop") and not result.get("sex"):
        raw = _strip_label(_ocr_crop(data["sex_crop"], "sex"), "sex")
        if raw:
            debug["sex_crop"] = raw[:80]
        if "FEMALE" in raw or re.search(r"\bF\b", raw):
            result["sex"] = "F"
        elif "MALE" in raw or re.search(r"\bM\b", raw):
            result["sex"] = "M"
        if result["sex"]:
            src["sex"] = "app_crop"

    if data.get("nationality_crop") and not result.get("nationality"):
        raw = _strip_label(_ocr_crop(data["nationality_crop"], "nationality"), "nationality")
        raw = re.sub(r"[^A-Z ]", " ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        if raw:
            debug["nationality_crop"] = raw[:100]
        if raw in NATIONALITY_NAMES:
            raw = NATIONALITY_NAMES[raw]
        elif raw in COUNTRY_NAMES.values():
            for code, country in COUNTRY_NAMES.items():
                if raw == country:
                    raw = NATIONALITY_NAMES.get(code, raw)
                    break
        if raw and not any(w in LABEL_WORDS or w in _MONTH_ABBR for w in raw.split()):
            result["nationality"] = raw
            src["nationality"] = "app_crop"

    if not result.get("residence_country"):
        nat_value = (result.get("nationality") or "").strip().upper()
        if nat_value == "PAKISTANI":
            result["residence_country"] = "PAKISTAN"
            src["residence_country"] = "nationality_fallback"

    useful = [result.get(k) for k in (
        "given_name_en", "surname_en", "father_name_en", "passport_number",
        "birth_date", "issue_date", "expiry_date", "nationality", "sex",
    )]
    if not any(useful):
        return JSONResponse(status_code=422, content={
            "success": False,
            "error": "وصلت القصاصات لكن OCR ما استخرج بيانات مفيدة",
            "mode": "app_field_crops",
            "server_version": SERVER_VERSION,
            "crop_debug": debug,
        })

    result.update({
        "success": True,
        "mode": "app_field_crops",
        "client_crop_mode": client_crop_mode or "unknown",
        "server_version": SERVER_VERSION,
        "ocr_time_ms": int((time.monotonic() - started) * 1000),
        "received_crops": sorted(data.keys()),
        "crop_debug": debug,
        "needs_review": not bool(result.get("father_name_en")) or not bool(result.get("issue_date")),
    })

    result["fields_confidence"] = {
        k: ("verified" if src.get(k) == "mrz_crop" and result.get("is_verified") else "printed_crop")
        for k in (
            "given_name_en", "surname_en", "father_name_en", "passport_number",
            "nationality", "birth_date", "issue_date", "expiry_date", "sex",
        )
        if result.get(k)
    }
    return JSONResponse(content=result)


@app.post("/read-passport")
async def read_passport(file: UploadFile = File(...)):
    started = time.monotonic()
    data = await file.read()
    color = _decode(data, gray=False)
    if color is None:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": "الصورة فارغة أو غير صالحة",
        })

    best = None
    for image_variant in (color, cv2.rotate(color, cv2.ROTATE_180)):
        hh, ww = image_variant.shape[:2]
        # MRZ is normally in the lower 45%; no need to OCR the full passport.
        crop = image_variant[int(hh * 0.55):hh, 0:ww]
        ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            continue
        parsed, _ = _mrz_from_bytes(enc.tobytes())
        if parsed and (best is None or parsed["score"] > best["score"]):
            best = parsed
        if parsed and parsed.get("passport_number") and parsed.get("birth_date") and parsed.get("expiry_date"):
            break

    if best is None:
        return JSONResponse(status_code=422, content={
            "success": False,
            "error": "ما قدرنا نلقى MRZ واضح بالصورة",
            "server_version": SERVER_VERSION,
        })

    best["server_version"] = SERVER_VERSION
    best["mode"] = "full_image_mrz_fallback"
    best["ocr_time_ms"] = int((time.monotonic() - started) * 1000)
    best["needs_review"] = True
    best.setdefault("field_sources", {})
    for k in ("given_name_en", "surname_en", "passport_number", "nationality", "residence_country", "birth_date", "expiry_date", "sex"):
        if best.get(k):
            best["field_sources"][k] = "mrz"
    return JSONResponse(content=best)
