"""FastAPI passport OCR service - crop-first optimized reader.

This version keeps the app's successful strategy: Flutter detects/crops each
printed field locally, then Python reads the small value crops. MRZ is used for
structural fields and as a cross-check, not as the only name source.
"""

import logging
import os
import re
import time
import tempfile
from pathlib import Path
from datetime import date
from typing import Optional

# Render has a small/burstable CPU. Tesseract/OpenMP trying to use several
# threads is fast on a desktop but can get heavily throttled in the cloud.
# Keep every OCR subprocess single-threaded so the first passport and the
# fifth passport behave the same instead of timing out after the CPU burst.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from mrz.checker.td3 import TD3CodeChecker

app = FastAPI(title="Passport Reader API")
SERVER_VERSION = "cloud-app-crop-v39"
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

_JOINED_NAME_LABELS = {
    "GIVENNAME", "GIVENNAMES", "FIRSTNAME", "SURNAME", "FAMILYNAME",
    "LASTNAME", "FATHERNAME", "HUSBANDNAME", "GUARDIANNAME",
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


def _ocr(img, whitelist, psm=7, timeout=10):
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
    except RuntimeError as exc:
        # pytesseract raises RuntimeError when its per-call timeout expires.
        # Log the actual reason; the previous code only printed the exception
        # type, which hid the real Render slowdown.
        logger.warning("Field OCR RuntimeError: %s", exc)
        return ""
    except Exception as exc:
        logger.warning("Field OCR failed (%s): %s", type(exc).__name__, exc)
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


def _looks_like_name_label_token(token, kind=""):
    """Recognise field labels even when OCR joins/misspells their letters."""
    token = re.sub(r"[^A-Z]", "", (token or "").upper())
    if not token:
        return False
    if token in _JOINED_NAME_LABELS or token in {"NAME", "NAMES", "ENGLISH"}:
        return True

    targets = {
        "given": ("GIVEN", "GIVENNAME", "GIVENNAMES", "FIRSTNAME"),
        "surname": ("SURNAME", "FAMILYNAME", "LASTNAME"),
        "father": ("FATHER", "FATHERNAME", "HUSBAND", "HUSBANDNAME", "GUARDIAN"),
    }.get(kind, ("GIVENNAME", "GIVENNAMES", "SURNAME", "FATHERNAME", "HUSBANDNAME"))

    for target in targets:
        # Do not fuzzy-match short NAME-like tokens: real names can be close.
        if len(target) < 6:
            continue
        if abs(len(token) - len(target)) <= 2 and _edit(token, target) <= 2:
            return True
    return False


def _strip_name_field_label(text, kind):
    """Remove only a leading field title and report whether one was found."""
    s = re.sub(r"\s+", " ", (text or "").upper()).strip()
    if not s:
        return "", False

    tokens = s.split()
    removed = False

    if tokens and _looks_like_name_label_token(tokens[0], kind):
        tokens.pop(0)
        removed = True

    # Handles split titles such as GIVEN NAME / FAMILY NAME.
    if tokens and removed and tokens[0] in {"NAME", "NAMES", "ENGLISH"}:
        tokens.pop(0)

    # Exact multi-token fallback when OCR kept punctuation between words.
    cleaned = " ".join(tokens) if removed else s
    patterns = {
        "given": r"^(?:GIVEN\s*NAMES?|FIRST\s*NAME)\s*[:\-]?\s*",
        "surname": r"^(?:SURNAME|FAMILY\s*NAME|LAST\s*NAME)\s*[:\-]?\s*",
        "father": r"^(?:FATHER(?:'?S)?\s*NAME|HUSBAND(?:'?S)?\s*NAME|GUARDIAN\s*NAME)\s*[:\-]?\s*",
    }
    pattern = patterns.get(kind)
    if pattern:
        exact = re.sub(pattern, "", cleaned).strip()
        if exact != cleaned:
            cleaned = exact
            removed = True

    return cleaned, removed


def _contains_name_label_noise(text, kind=""):
    words = re.findall(r"[A-Z]+", (text or "").upper())
    if any(_looks_like_name_label_token(word, kind) for word in words):
        return True
    joined = "".join(words[:2])
    return bool(joined and _looks_like_name_label_token(joined, kind))


def _looks_like_issuing_authority(text):
    """Reject the right-column label, including common broken OCR spellings."""
    words = re.findall(r"[A-Z]{4,}", (text or "").upper())
    return any(_edit(word, "AUTHORITY") <= 2 for word in words if len(word) >= 7)


def _without_issuing_authority(text):
    """Keep father text to the left of the separate issuing-authority column."""
    text = (text or "").upper()
    words = list(re.finditer(r"[A-Z]{2,}", text))
    for i, match in enumerate(words):
        word = match.group(0)
        if len(word) >= 7 and _edit(word, "AUTHORITY") <= 2:
            begin = match.start()
            if i and _edit(words[i - 1].group(0), "ISSUING") <= 2:
                begin = words[i - 1].start()
            return text[:begin].strip(" ,-|")
    return text


def _plausible_name(name):
    if not name or len(name) < 2 or not re.fullmatch(r"[A-Z ]{2,60}", name):
        return False
    words = name.split()
    if not words or any(len(w) == 1 for w in words):
        return False
    if _looks_like_issuing_authority(name):
        return False
    if _contains_name_label_noise(name):
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
    """Use printed crop only for unmistakable MRZ filler corrections.

    A clean MRZ name must NOT be replaced by a one-character OCR mistake such
    as SOOMRO -> SGOMRO. Printed OCR is allowed to fix only the known filler
    patterns caused by '<' becoming K/X/S.
    """
    m = _clean_name(mrz_name)
    p = _clean_name(printed_name)

    if not _plausible_name(m):
        return p if _plausible_name(p) else m
    if not _plausible_name(p):
        return m
    if m == p:
        return m

    def filler_tail(word):
        if len(word) < 2:
            return False
        ratio = sum(ch in "KXS" for ch in word) / len(word)
        return ratio >= 0.55

    mw = m.split()
    while len(mw) > 1 and filler_tail(mw[-1]):
        mw.pop()
    m_trim = " ".join(mw)

    # Whole filler tail removed.
    if p == m_trim:
        return p

    mt = m_trim.split()
    pt = p.split()
    if len(mt) != len(pt):
        return m_trim if m_trim != m else m

    changed = False
    for idx, (a, b) in enumerate(zip(mt, pt)):
        if a == b:
            continue

        # TASSAWARK -> TASSAWAR
        if len(a) == len(b) + 1 and a[-1] in "KXS" and a[:-1] == b:
            changed = True
            continue

        # JAWAD KALI -> JAWAD ALI; never do this to the first word so
        # KASHAF can never become ASHAF.
        if (
            idx > 0
            and len(a) == len(b) + 1
            and a[0] in "KXS"
            and a[1:] == b
        ):
            changed = True
            continue

        # Any ordinary OCR substitution (SOOMRO -> SGOMRO etc.) is rejected.
        return m_trim if m_trim != m else m

    return p if changed else (m_trim if m_trim != m else m)


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


def _quick_variant(gray, width=900, clahe=False):
    img = _resize_min_width(gray, width)
    if clahe:
        img = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(img)
    return img


def _ocr_crop(data: bytes, kind: str):
    """Read one label+value crop.

    v23 receives the small field block itself (printed label + printed value),
    so do not cut a fixed percentage from the crop. Let Tesseract see both;
    downstream parsers remove the label and keep the requested value.
    """
    gray = _decode(data)
    if gray is None:
        return ""

    if kind in {"given", "surname", "father"}:
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ,'- "
        return _ocr(_quick_variant(gray, 850, clahe=True), wl, psm=6, timeout=9)

    if kind in {"birth", "issue", "expiry"}:
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/. "
        return _ocr(_quick_variant(gray, 800, clahe=False), wl, psm=6, timeout=9)

    if kind == "passport":
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<.:- "
        return _ocr(_quick_variant(gray, 800, clahe=False), wl, psm=6, timeout=9)

    if kind == "nationality":
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        return _ocr(_quick_variant(gray, 700, clahe=True), wl, psm=6, timeout=9)

    if kind == "sex":
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return _ocr(_quick_variant(gray, 550, clahe=True), wl, psm=6, timeout=7)

    return ""



def _batch_prepare_crop(data: bytes, kind: str, target_h: int = 170):
    """Prepare one crop while preserving the printed strokes.

    The previous stacked-batch path applied CLAHE to every name crop and then
    squeezed all fields into one large page. On pale Pakistani passports that
    destroyed thin letters (IJAZ -> ATA, FAIZ MUHAMMAD -> random fragments).
    Keep each crop as its own Tesseract page instead.
    """
    gray = _decode(data)
    if gray is None or gray.size == 0:
        return None

    # Keep the original grayscale for names/dates/passport. It tested better on
    # the faint green passport stock. Use only a very mild contrast lift for
    # nationality/sex, where the value is short and often low contrast.
    if kind in {"nationality", "sex"}:
        gray = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(gray)

    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return None

    scale = target_h / max(1, h)
    new_w = max(120, min(1800, int(w * scale)))
    resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Mild sharpening helps thin printed letters without the halo/noise CLAHE
    # created in the old implementation.
    if kind in {"given", "surname", "father", "passport"}:
        blur = cv2.GaussianBlur(resized, (0, 0), 0.55)
        resized = cv2.addWeighted(resized, 1.22, blur, -0.22, 0)

    # White border prevents edge characters from being clipped by Tesseract.
    return cv2.copyMakeBorder(
        resized, 12, 12, 18, 18,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def _batch_ocr_printed(data: dict):
    """Read all printed crops with ONE Tesseract process, one page per crop.

    Tesseract accepts a .txt list of image paths as a multi-page input. This
    keeps each crop independently segmented (unlike the old stacked canvas)
    while still spawning only one OCR process, so Render remains stable after
    repeated passports.
    """
    specs = [
        ("given_name_crop", "given"),
        ("surname_crop", "surname"),
        ("father_name_crop", "father"),
        ("passport_number_crop", "passport"),
        ("birth_date_crop", "birth"),
        ("issue_date_crop", "issue"),
        ("expiry_date_crop", "expiry"),
        ("nationality_crop", "nationality"),
        ("sex_crop", "sex"),
    ]

    prepared = []
    for field, kind in specs:
        if not data.get(field):
            continue
        im = _batch_prepare_crop(data[field], kind)
        if im is not None and im.size:
            prepared.append((field, kind, im))

            # Father/Husband name and issue date are absent from MRZ.
            # Add a gentle CLAHE alternative for both without changing glyphs.
            if kind in ("father", "issue"):
                alt = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(im)
                prepared.append((field, kind, alt))

    if not prepared:
        return {}, ""

    cfg = (
        "--oem 1 --psm 6 "
        "-c preserve_interword_spaces=1 "
        '-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.\'/- "'
    )

    try:
        with tempfile.TemporaryDirectory(prefix="passport_ocr_") as tmp:
            tmp_path = Path(tmp)
            image_paths = []
            page_to_field = {}

            for page_num, (field, kind, im) in enumerate(prepared, start=1):
                image_path = tmp_path / f"{page_num:02d}_{kind}.png"
                if not cv2.imwrite(str(image_path), im):
                    continue
                image_paths.append(str(image_path))
                page_to_field[len(image_paths)] = field

            if not image_paths:
                return {}, ""

            list_path = tmp_path / "input_list.txt"
            list_path.write_text("\n".join(image_paths), encoding="utf-8")

            ocr = pytesseract.image_to_data(
                str(list_path),
                config=cfg,
                lang="eng",
                output_type=pytesseract.Output.DICT,
                timeout=26,
            )
    except RuntimeError as exc:
        logger.warning("Printed multipage OCR RuntimeError: %s", exc)
        return {}, ""
    except Exception as exc:
        logger.warning("Printed multipage OCR failed (%s): %s", type(exc).__name__, exc)
        return {}, ""

    # Preserve Tesseract line boundaries. Father/husband crops often also
    # contain a small piece of the next label; picking the strongest text line
    # avoids turning "FAIZ MUHAMMAD" into random words from "Date of Issue".
    lines_by_field = {field: {} for field, _, _ in prepared}
    for i, token in enumerate(ocr.get("text", [])):
        token = (token or "").strip()
        if not token:
            continue
        try:
            page_num = int(ocr.get("page_num", [])[i])
            block_num = int(ocr.get("block_num", [])[i])
            par_num = int(ocr.get("par_num", [])[i])
            line_num = int(ocr.get("line_num", [])[i])
            conf = float(ocr.get("conf", [])[i])
        except Exception:
            continue

        field = page_to_field.get(page_num)
        if not field:
            continue
        key = (page_num, block_num, par_num, line_num)
        bucket = lines_by_field.setdefault(field, {}).setdefault(
            key, {"tokens": [], "conf": []}
        )
        bucket["tokens"].append(token)
        if conf >= 0:
            bucket["conf"].append(conf)

    raw = {}
    for field, line_map in lines_by_field.items():
        line_rows = []
        for bucket in line_map.values():
            line_text = re.sub(r"\s+", " ", " ".join(bucket["tokens"])).strip().upper()
            if not line_text:
                continue
            confs = bucket["conf"]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            alpha_len = sum(len(w) for w in re.findall(r"[A-Z]{2,}", line_text))
            line_rows.append((line_text, avg_conf, alpha_len))

        if not line_rows:
            raw[field] = ""
            continue

        if field == "father_name_crop":
            def father_line_score(row):
                line_text, avg_conf, alpha_len = row
                line_text = _without_issuing_authority(line_text)
                if not line_text:
                    return -10000
                words = re.findall(r"[A-Z]{2,}", line_text)
                label_hits = sum(1 for w in words if w in LABEL_WORDS)
                label_penalty = 35 * label_hits
                comma_bonus = 24 if "," in line_text else 0
                count_bonus = 22 if 2 <= len(words) <= 4 else (-15 if len(words) > 5 else 0)
                short_noise = sum(1 for w in words if len(w) <= 2)
                noise_penalty = short_noise * 8
                return avg_conf + alpha_len * 3.5 + comma_bonus + count_bonus - label_penalty - noise_penalty
            chosen = max(line_rows, key=father_line_score)
            raw[field] = (
                _without_issuing_authority(chosen[0])
                if father_line_score(chosen) > -10000 else ""
            )
        else:
            # Keep all lines for dates/given/surname; downstream parsers use MRZ
            # hints and date patterns to pick the requested value.
            raw[field] = " ".join(row[0] for row in line_rows)

    debug_text = " | ".join(f"{k}={v}" for k, v in raw.items() if v)
    return raw, debug_text[:1200]

def _words_only(text):
    return re.findall(r"[A-Z]{2,}", (text or "").upper())


def _best_printed_name(raw, kind, mrz_hint=""):
    """Extract a name from one batch-OCR band, using MRZ only as a hint."""
    s = re.sub(r"\s+", " ", (raw or "").upper()).strip()
    if not s:
        return ""

    # Labels are sometimes returned glued or misspelled, for example
    # GIVENNAMES GHULAM / SUMAME MURTAZA. Remove the title before scoring so
    # it can never become part of the person's name.
    s, leading_label_removed = _strip_name_field_label(s, kind)
    if not s:
        return ""

    if kind == "father":
        s = _without_issuing_authority(s)
        if not s:
            return ""

    # Father/Husband often appears as SURNAME, GIVEN on Pakistani passports.
    # A comma pair is a very strong signal and _clean_name reverses it.
    if kind == "father":
        comma = re.search(
            r"\b([A-Z]{2,}(?:[- ]+[A-Z]{2,}){0,2})\s*,\s*"
            r"([A-Z]{2,}(?:[- ]+[A-Z]{2,}){0,3})\b",
            s,
        )
        if comma:
            left = comma.group(1)
            right_words = _words_only(comma.group(2))
            kept = []
            for w in right_words:
                # Stop at obvious OCR crumbs such as OO / TR that often come
                # from the next printed label. Real Pakistani name words here
                # are normally at least three letters.
                if len(w) < 3:
                    break
                kept.append(w)
                if len(kept) >= 2:
                    break
            if kept:
                candidate = _clean_name(f"{left}, {' '.join(kept)}")
                if _plausible_name(candidate):
                    return candidate

        # Capture words after Father/Husband Name, tolerating OCR's NAME typo.
        m = re.search(
            r"(?:FATHER|HUSBAND|GUARDIAN)\s+[A-Z]{2,8}\s+(.+)", s
        )
        if m:
            tail = re.split(
                r"\b(?:DATE|NATIONALITY|ISSUING|AUTHORITY|PASSPORT|SEX|PLACE)\b",
                m.group(1), maxsplit=1,
            )[0]
            words = [w for w in _words_only(tail) if w not in LABEL_WORDS][:4]
            candidate = _clean_name(" ".join(words))
            if _plausible_name(candidate):
                return candidate

    # If a printed label survived OCR, text after it is normally the value.
    label_patterns = {
        "given": r"\b(?:GIVEN|GIVFN|GIVEM)\s+[A-Z]{2,8}\b",
        "surname": r"\b(?:SURNAME|SURNAMF|SUR[A-Z]{3,8})\b",
    }
    pat = label_patterns.get(kind)
    if pat:
        m = re.search(pat, s)
        if m:
            tail = s[m.end():]
            words = [w for w in _words_only(tail) if w not in LABEL_WORDS][:4]
            candidate = _clean_name(" ".join(words))
            if _plausible_name(candidate):
                return candidate

    if kind == "father":
        # In a tight father/husband crop the actual value is normally the first
        # 2-4 meaningful words. Stop when the next printed field label begins.
        tail = re.split(
            r"\b(?:DATE|ISSUE|EXPIRY|NATIONALITY|ISSUING|AUTHORITY|PASSPORT|SEX|PLACE)\b",
            s,
            maxsplit=1,
        )[0]
        fw = [w for w in _words_only(tail) if w not in LABEL_WORDS and len(w) >= 2]
        if fw:
            # Prefer up to 4 leading words, but drop obvious OCR crumbs at the end.
            kept = []
            for w in fw:
                if len(kept) >= 2 and len(w) <= 2:
                    break
                kept.append(w)
                if len(kept) >= 4:
                    break
            # Try longest-to-shortest so MUHAMMAD HASSAN SHAH remains intact,
            # while FAIZ MUHAMMAD + trailing noise can fall back to two words.
            for n in range(min(4, len(kept)), 1, -1):
                candidate = _clean_name(" ".join(kept[:n]))
                if _plausible_name(candidate):
                    # If the last word looks like a short OCR fragment, retry
                    # without it before accepting.
                    if n > 2 and len(kept[n - 1]) <= 3:
                        shorter = _clean_name(" ".join(kept[:n - 1]))
                        if _plausible_name(shorter):
                            return shorter
                    return candidate

    words = [
        w for w in _words_only(s)
        if w not in LABEL_WORDS and not _looks_like_name_label_token(w, kind)
    ]
    if not words:
        return ""

    # With an MRZ hint, choose the 1-4 word sequence closest to it. This turns
    # raw 'ABBAS ... TASSAWAR' + hint TASSAWARK into printed TASSAWAR.
    hint = _clean_name(mrz_hint)
    candidates = []
    for size in range(1, min(4, len(words)) + 1):
        for i in range(0, len(words) - size + 1):
            candidate = _clean_name(" ".join(words[i:i + size]))
            if not _plausible_name(candidate):
                continue
            if any(w in COUNTRY_NAMES.values() or w in NATIONALITY_NAMES.values() for w in candidate.split()):
                continue
            candidates.append(candidate)

    if not candidates:
        return ""
    if hint:
        expanded = set(candidates)
        for candidate in list(candidates):
            words_c = candidate.split()
            if words_c:
                if len(words_c[0]) >= 5:
                    trimmed = words_c.copy()
                    trimmed[0] = trimmed[0][1:]
                    v = _clean_name(" ".join(trimmed))
                    if _plausible_name(v):
                        expanded.add(v)
                if len(words_c[-1]) >= 5:
                    trimmed = words_c.copy()
                    trimmed[-1] = trimmed[-1][:-1]
                    v = _clean_name(" ".join(trimmed))
                    if _plausible_name(v):
                        expanded.add(v)
                for wi, word in enumerate(words_c):
                    if len(word) >= 5 and word[-1] in "KXS":
                        trimmed = words_c.copy()
                        trimmed[wi] = word[:-1]
                        v = _clean_name(" ".join(trimmed))
                        if _plausible_name(v):
                            expanded.add(v)
        return min(expanded, key=lambda c: (_edit(c, hint), abs(len(c) - len(hint)), len(c)))

    # A surname crop that visibly contained its label but produced several
    # unrelated OCR words is unsafe. Accept only one complete-looking trailing
    # value; otherwise leave it empty so MRZ/full-image fallback can decide.
    if kind == "surname" and leading_label_removed:
        last = _clean_name(words[-1])
        return last if len(last) >= 5 and _plausible_name(last) else ""

    # Without an MRZ hint, prefer a compact plausible value rather than all
    # surrounding label/context words.
    return max(candidates, key=lambda c: (min(len(c.split()), 3), len(c)))


def _parse_printed_batch(raw_map, mrz=None):
    out = {}
    debug = {}
    mrz = mrz or {}

    for field, kind, outkey in (
        ("given_name_crop", "given", "given_name_en"),
        ("surname_crop", "surname", "surname_en"),
        ("father_name_crop", "father", "father_name_en"),
    ):
        raw = raw_map.get(field, "")
        if raw:
            debug[field] = raw[:180]
        hint = mrz.get(outkey, "") if kind != "father" else ""
        val = _best_printed_name(raw, kind, hint)
        # Without an MRZ match, short broken fragments such as SE AE are not
        # enough evidence for a two-word surname. Leave the field for review.
        if kind == "surname" and len(val.split()) >= 2 and all(
            len(word) <= 2 for word in val.split()
        ):
            val = ""
        if val:
            out[outkey] = val

    for field, kind, outkey in (
        ("birth_date_crop", "birth", "birth_date"),
        ("issue_date_crop", "issue", "issue_date"),
        ("expiry_date_crop", "expiry", "expiry_date"),
    ):
        raw = raw_map.get(field, "")
        if raw:
            debug[field] = raw[:180]
        dates = _find_dates(raw)
        if dates:
            out[outkey] = _date_text(dates[0])

    raw = raw_map.get("passport_number_crop", "")
    if raw:
        debug["passport_number_crop"] = raw[:180]
        up = re.sub(r"\b(?:PASSPORT|DOCUMENT|NUMBER|NO)\b", " ", raw.upper())
        tokens = re.findall(r"[A-Z0-9]{6,12}", re.sub(r"[^A-Z0-9]", " ", up))
        tokens = [t for t in tokens if any(ch.isdigit() for ch in t)]
        if tokens:
            out["passport_number"] = min(tokens, key=lambda x: abs(len(x) - 9))

    raw = raw_map.get("nationality_crop", "")
    if raw:
        debug["nationality_crop"] = raw[:180]
        up = raw.upper()
        for value in sorted(set(NATIONALITY_NAMES.values()), key=len, reverse=True):
            if value in up:
                out["nationality"] = value
                break
        if "nationality" not in out:
            for code, value in NATIONALITY_NAMES.items():
                if re.search(rf"\b{re.escape(code)}\b", up):
                    out["nationality"] = value
                    break

    raw = raw_map.get("sex_crop", "")
    if raw:
        debug["sex_crop"] = raw[:180]
        up = raw.upper()
        if "FEMALE" in up or re.search(r"\bF\b", up):
            out["sex"] = "F"
        elif "MALE" in up or re.search(r"\bM\b", up):
            out["sex"] = "M"

    return out, debug


def _prefer_name_variant(original, enhanced, mrz_hint=""):
    """Prefer the crop variant closest to MRZ; otherwise preserve original."""
    a = _clean_name(original)
    b = _clean_name(enhanced)
    if not _plausible_name(a):
        return b if _plausible_name(b) else ""
    if not _plausible_name(b):
        return a
    if a == b:
        return a

    hint = _clean_name(mrz_hint)
    if _plausible_name(hint):
        da = _edit(a, hint)
        db = _edit(b, hint)
        if db < da:
            return b
    return a


def _prefer_passport_variant(original, enhanced, mrz_hint=""):
    a = re.sub(r"[^A-Z0-9]", "", (original or "").upper())
    b = re.sub(r"[^A-Z0-9]", "", (enhanced or "").upper())
    if not a:
        return b
    if not b or a == b:
        return a
    hint = re.sub(r"[^A-Z0-9]", "", (mrz_hint or "").upper())
    if hint and _edit(b, hint) < _edit(a, hint):
        return b
    return a

def _clean_mrz_line(line):
    s = (line or "").upper()
    s = s.replace("«", "<").replace("‹", "<").replace("_", "<").replace("|", "<")
    return re.sub(r"[^A-Z0-9<]", "", s)


def _mrz_names(l1):
    body = l1[5:44].rstrip("<")
    parts = body.split("<<", 1)
    sur = _mrz_name_block(parts[0])
    given_block = parts[1] if len(parts) > 1 else ""
    giv = _mrz_name_block(given_block)
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
    """Decode an MRZ name block without letting filler OCR become a name.

    Pakistani MRZ fillers '<' are often read as K/X/S/C/B/O.  The old parser
    only stopped K/X/S, which allowed values such as SHAGUFTABOCSSCCCCCC.
    """
    raw = (block or "").upper()

    # If OCR changed a long run of fillers into letters, cut that suspicious
    # suffix before tokenizing.  Require at least 4 filler-like chars so real
    # names such as ABBAS / BUKHARI are not touched.
    raw = re.sub(r"(?<=[A-Z])[KXSCBO]{4,}$", "", raw)

    parts = [p for p in re.split(r"<+", raw) if p]
    words = []

    for token in parts:
        t = re.sub(r"[^A-Z]", "", token)
        if not t:
            continue

        # Cut a filler-like suffix stuck onto a real word:
        # SHAGUFTA + BOCSSCCCC -> SHAGUFTA.
        m = re.match(r"^([A-Z]{2,}?)([KXSCBO]{4,})$", t)
        if m:
            prefix = m.group(1)
            if prefix:
                words.append(prefix)
            break

        filler_ratio = sum(ch in "KXSCBO" for ch in t) / max(1, len(t))
        if words and (
            re.fullmatch(r"[KXSCBO]{1,}", t)
            or (len(t) >= 5 and filler_ratio >= 0.72)
        ):
            break

        if len(t) == 1 and t in {"K", "X", "S", "C"} and words:
            break

        words.append(t)

    return _strip_name_noise_tokens(" ".join(words))


def _repair_mrz_name_line(line):
    """Repair only the TD3 document/country prefix when OCR drops '<' or P."""
    s = _clean_mrz_line(line)
    if not s:
        return ""

    # Real example: P<PAK... was read as <PAK... while the names stayed clear.
    if len(s) >= 4 and s.startswith("<") and s[1:4] in COUNTRY_NAMES:
        return "P" + s
    if len(s) >= 3 and s[:3] in COUNTRY_NAMES:
        return "P<" + s
    if len(s) >= 4 and s.startswith("P") and s[1:4] in COUNTRY_NAMES:
        return "P<" + s[1:]
    return s


def _parse_mrz_name_only(line):
    """Recover surname/given names even when MRZ line 2 is split or unreadable."""
    s = _repair_mrz_name_line(line)
    if len(s) < 12 or not s.startswith("P<"):
        return None
    country = s[2:5]
    if country not in COUNTRY_NAMES or "<<" not in s[5:]:
        return None

    sur, giv = _mrz_names(s[:44].ljust(44, "<"))
    if not _plausible_name(sur) or not _plausible_name(giv):
        return None

    return {
        "success": True,
        "given_name_en": giv,
        "surname_en": sur,
        "father_name_en": "",
        "passport_number": "",
        "nationality": NATIONALITY_NAMES.get(country, country),
        "residence_country": COUNTRY_NAMES.get(country, country),
        "birth_date": "",
        "issue_date": "",
        "expiry_date": "",
        "sex": "",
        "is_verified": False,
        "is_fully_verified": False,
        "score": 6,
        "mrz_line1": s[:44].ljust(44, "<"),
        "mrz_line2": "<" * 44,
        "mrz_name_only": True,
    }


def _parse_mrz_fuzzy(l1, l2):
    """Best-effort TD3 parser when one OCR error makes TD3CodeChecker reject the pair."""
    a = _repair_mrz_name_line(l1)
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
    l1 = _repair_mrz_name_line(l1)[:44].ljust(44, "<")
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
    """Read MRZ once, with a realistic timeout for Render's slower CPU.

    The previous 6-second timeout was the direct source of repeated
    `MRZ OCR failed (RuntimeError)` messages. Two timed-out passes alone cost
    about 12 seconds and then all field OCR started from scratch.
    """
    gray = _decode(data)
    if gray is None:
        return None, ""

    gray = _resize_min_width(gray, 1100)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    cfg = "--oem 1 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

    try:
        text = pytesseract.image_to_string(
            clahe, config=cfg, lang="eng", timeout=18
        )
    except RuntimeError as exc:
        logger.warning("MRZ OCR RuntimeError: %s", exc)
        return None, ""
    except Exception as exc:
        logger.warning("MRZ OCR failed (%s): %s", type(exc).__name__, exc)
        return None, ""

    debug = text.strip() if text else ""
    all_lines = [_clean_mrz_line(x) for x in (text or "").splitlines()]
    all_lines = [x for x in all_lines if x]
    lines = [x for x in all_lines if len(x) >= 20]

    # Names live completely in MRZ line 1. Do not throw them away merely
    # because Tesseract split line 2 into several fragments.
    name_only = None
    for line in all_lines:
        candidate = _parse_mrz_name_only(line)
        if candidate and (
            name_only is None
            or len(candidate["given_name_en"] + candidate["surname_en"])
               > len(name_only["given_name_en"] + name_only["surname_en"])
        ):
            name_only = candidate

    best = None
    for i in range(len(lines) - 1):
        best = _prefer_mrz_candidate(best, _try_mrz(lines[i], lines[i + 1]))
        best = _prefer_mrz_candidate(best, _parse_mrz_fuzzy(lines[i], lines[i + 1]))

    if name_only is not None:
        if best is None:
            best = name_only
        else:
            for key in ("given_name_en", "surname_en"):
                if not best.get(key) or _name_has_mrz_noise(best.get(key, "")):
                    best[key] = name_only[key]
            if not best.get("mrz_line1"):
                best["mrz_line1"] = name_only["mrz_line1"]
            best["score"] = max(best.get("score", 0), name_only["score"])

    return best, debug


def _choose_passport_number(mrz_number, printed_number, country="", mrz_verified=False):
    """Reconcile passport number conservatively.

    For Pakistan the expected layout is two letters + seven digits. OCR often
    confuses G/6 and Z/2 in the two-letter prefix. Normalize by position before
    comparing, instead of blindly trusting either source.
    """
    m = re.sub(r"[^A-Z0-9]", "", (mrz_number or "").upper())
    p = re.sub(r"[^A-Z0-9]", "", (printed_number or "").upper())

    country_up = (country or "").upper()

    def normalize_pak(v):
        if len(v) != 9:
            return v

        prefix_map = {
            "0": "O", "1": "I", "2": "Z", "5": "S",
            "6": "G", "8": "B",
        }
        digit_map = {
            "O": "0", "Q": "0", "D": "0",
            "I": "1", "L": "1", "Z": "2",
            "S": "5", "G": "6", "B": "8",
        }

        first = "".join(prefix_map.get(ch, ch) for ch in v[:2])
        rest = "".join(digit_map.get(ch, ch) for ch in v[2:])
        return first + rest

    if country_up == "PAKISTAN":
        m = normalize_pak(m)
        p = normalize_pak(p)
        pak_pat = re.compile(r"^[A-Z]{2}[0-9]{7}$")
        m_ok = bool(pak_pat.fullmatch(m))
        p_ok = bool(pak_pat.fullmatch(p))

        if mrz_verified and m_ok:
            return m
        if p_ok and not m_ok:
            return p
        if m_ok and not p_ok:
            return m
        if p_ok and m_ok:
            # If both normalize to valid Pakistani numbers, agreement/near
            # agreement is enough; otherwise keep MRZ rather than inventing.
            if p == m:
                return m
            return p if _edit(p, m) <= 2 and not mrz_verified else m

    if p and m:
        if len(p) == len(m) and _edit(p, m) <= 1:
            return p if not mrz_verified else m

        def score(v):
            return (
                int(7 <= len(v) <= 10) * 3
                + int(any(ch.isalpha() for ch in v)) * 2
                + int(any(ch.isdigit() for ch in v)) * 2
            )

        return p if score(p) > score(m) else m

    return p or m


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


def _same_person_name(a, b):
    aa = re.sub(r"[^A-Z]", "", _clean_name(a))
    bb = re.sub(r"[^A-Z]", "", _clean_name(b))
    return bool(aa and bb and aa == bb)


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
    given_name_enhanced_crop: Optional[UploadFile] = File(None),
    surname_enhanced_crop: Optional[UploadFile] = File(None),
    father_name_enhanced_crop: Optional[UploadFile] = File(None),
    passport_number_enhanced_crop: Optional[UploadFile] = File(None),
    mrz_enhanced_crop: Optional[UploadFile] = File(None),
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
        "given_name_enhanced_crop": given_name_enhanced_crop,
        "surname_enhanced_crop": surname_enhanced_crop,
        "father_name_enhanced_crop": father_name_enhanced_crop,
        "passport_number_enhanced_crop": passport_number_enhanced_crop,
        "mrz_enhanced_crop": mrz_enhanced_crop,
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

    # 1) ONE OCR process for every printed crop. This includes the two fields
    #    MRZ can never supply: father/husband name + issue date.
    batch_raw, batch_debug = _batch_ocr_printed(data)
    if batch_debug:
        debug["printed_batch"] = batch_debug

    # 2) ONE MRZ process for structural fields and name hints.
    mrz = None
    mrz_source = "mrz_crop"
    if data.get("mrz_crop"):
        mrz, mrz_raw = _mrz_from_bytes(data["mrz_crop"])
        if mrz_raw:
            debug["mrz_crop"] = re.sub(r"\s+", " ", mrz_raw)[:350]

    # The preview button supplies an independent, color-preserving enhancement.
    # Retry MRZ only if the original failed to yield its essential fields.
    if data.get("mrz_enhanced_crop") and (
        not mrz or not mrz.get("surname_en") or
        not mrz.get("given_name_en") or not mrz.get("passport_number")
    ):
        alt_mrz, alt_mrz_raw = _mrz_from_bytes(data["mrz_enhanced_crop"])
        if alt_mrz_raw:
            debug["mrz_enhanced_crop"] = re.sub(r"\s+", " ", alt_mrz_raw)[:350]
        chosen = _prefer_mrz_candidate(mrz, alt_mrz)
        if chosen is alt_mrz and alt_mrz is not None:
            mrz_source = "mrz_enhanced_crop"
        mrz = chosen

    if mrz:
        for k in (
            "given_name_en", "surname_en", "passport_number", "nationality",
            "residence_country", "birth_date", "expiry_date", "sex",
        ):
            if mrz.get(k):
                result[k] = mrz[k]
                src[k] = mrz_source
        for k in ("score", "is_verified", "is_fully_verified", "mrz_line1", "mrz_line2"):
            if k in mrz:
                result[k] = mrz[k]

    printed, printed_debug = _parse_printed_batch(batch_raw, mrz)
    debug.update(printed_debug)
    printed_source = {key: "printed_batch" for key in printed}

    # Compare the enhanced read to the original, retaining valid original
    # values. Extra pages are read only when the user enabled the preview.
    enhanced_fields = {
        "given_name_crop": "given_name_enhanced_crop",
        "surname_crop": "surname_enhanced_crop",
        "father_name_crop": "father_name_enhanced_crop",
        "passport_number_crop": "passport_number_enhanced_crop",
    }
    enhanced_data = {
        field: data[alt_key] for field, alt_key in enhanced_fields.items()
        if data.get(alt_key)
    }
    if enhanced_data:
        enhanced_raw, enhanced_debug = _batch_ocr_printed(enhanced_data)
        if enhanced_debug:
            debug["enhanced_batch"] = enhanced_debug
        enhanced_printed, enhanced_printed_debug = _parse_printed_batch(
            enhanced_raw, mrz
        )
        debug.update({"enhanced_" + k: v for k, v in enhanced_printed_debug.items()})
        for key in ("given_name_en", "surname_en"):
            chosen = _prefer_name_variant(
                printed.get(key, ""),
                enhanced_printed.get(key, ""),
                (mrz or {}).get(key, ""),
            )
            if chosen:
                if chosen == enhanced_printed.get(key) and chosen != printed.get(key):
                    printed_source[key] = "printed_batch_enhanced"
                printed[key] = chosen

        # Father has no MRZ reference, so enhanced may fill a miss but never
        # replace an already plausible original reading.
        if not printed.get("father_name_en") and enhanced_printed.get("father_name_en"):
            printed["father_name_en"] = enhanced_printed["father_name_en"]
            printed_source["father_name_en"] = "printed_batch_enhanced"

        chosen_passport = _prefer_passport_variant(
            printed.get("passport_number", ""),
            enhanced_printed.get("passport_number", ""),
            (mrz or {}).get("passport_number", ""),
        )
        if chosen_passport:
            if (chosen_passport == enhanced_printed.get("passport_number") and
                    chosen_passport != printed.get("passport_number")):
                printed_source["passport_number"] = "printed_batch_enhanced"
            printed["passport_number"] = chosen_passport

    # Father/Husband + Issue Date are not in MRZ. If the single batch OCR
    # missed either one, retry ONLY those crops. This is at most two small OCR
    # calls, not ten calls, so it does not recreate the old Render collapse.
    if not printed.get("father_name_en") and data.get("father_name_crop"):
        father_retry_raw = _ocr_crop(data["father_name_crop"], "father")
        if father_retry_raw:
            debug["father_retry"] = father_retry_raw[:180]
            father_retry = _best_printed_name(father_retry_raw, "father", "")
            if father_retry:
                printed["father_name_en"] = father_retry
                printed_source["father_name_en"] = "father_retry"

    if not printed.get("issue_date") and data.get("issue_date_crop"):
        issue_retry_raw = _ocr_crop(data["issue_date_crop"], "issue")
        if issue_retry_raw:
            debug["issue_retry"] = issue_retry_raw[:180]
            issue_dates = _find_dates(issue_retry_raw)
            if issue_dates:
                printed["issue_date"] = _date_text(issue_dates[0])

    # Printed names can correct only explicit MRZ filler noise.
    # If MRZ was read and the surname zone is explicitly blank (for example
    # P<PAK<<FOZIA...), do NOT invent a surname from OCR noise.
    for key in ("given_name_en", "surname_en"):
        p = printed.get(key, "")
        if not p:
            continue

        if key == "surname_en" and mrz is not None and not mrz.get("surname_en"):
            debug["surname_printed_ignored_mrz_blank"] = p
            continue

        previous = result.get(key, "")
        result[key] = _choose_name(
            previous,
            p,
            bool(result.get("is_verified")),
        )
        if not previous or result[key] != previous:
            src[key] = printed_source.get(key, "printed_batch")

    # Father/Husband and Issue Date exist only in printed fields.
    # Never accept the holder's own GIVEN/SURNAME as Father; that was the
    # exact MURTAZA -> GHULAM failure caused by a wrong crop.
    father_candidate = _clean_name(printed.get("father_name_en", ""))
    if father_candidate:
        father_raw = (batch_raw.get("father_name_crop", "") or "").upper()
        father_words = father_candidate.split()
        explicit_father_label = bool(
            re.search(r"\b(?:FATHER|HUSBAND|GUARDIAN)\b", father_raw)
        )

        reject_father = (
            _same_person_name(father_candidate, result.get("given_name_en", ""))
            or _same_person_name(father_candidate, result.get("surname_en", ""))
            or (len(father_words) == 1 and not explicit_father_label)
        )

        if reject_father:
            debug["father_rejected"] = father_candidate
        else:
            result["father_name_en"] = father_candidate
            src["father_name_en"] = printed_source.get("father_name_en", "printed_batch")

    issue_candidate = printed.get("issue_date")
    if issue_candidate:
        # Issue date cannot equal DOB or expiry. Reject a crop that accidentally
        # landed on one of those rows instead of writing a wrong date.
        if issue_candidate in {
            result.get("birth_date"),
            result.get("expiry_date"),
        }:
            debug["issue_rejected"] = issue_candidate
        else:
            result["issue_date"] = issue_candidate
            src["issue_date"] = "printed_batch"

    # Reconcile the focused printed number with MRZ. On Pakistani passports
    # this prevents both GZ... -> 6Z... and the opposite failure where the crop
    # loses the first two letters entirely.
    chosen_number = _choose_passport_number(
        result.get("passport_number", ""),
        printed.get("passport_number", ""),
        result.get("residence_country", ""),
        bool(result.get("is_verified")),
    )
    if chosen_number:
        if chosen_number == printed.get("passport_number"):
            src["passport_number"] = printed_source.get("passport_number", "printed_batch")
        result["passport_number"] = chosen_number

    # For the remaining structural fields keep MRZ when available; use printed
    # values as fallback. Issue date was already handled above because MRZ does
    # not contain it.
    for key in ("birth_date", "expiry_date", "nationality", "sex"):
        if not result.get(key) and printed.get(key):
            result[key] = printed[key]
            src[key] = "printed_batch"

    if not result.get("residence_country") and (result.get("nationality") or "").upper() == "PAKISTANI":
        result["residence_country"] = "PAKISTAN"
        src["residence_country"] = "nationality_fallback"

    # Last gate before the response: a field title must never reach the form as
    # a person's name. Keep the field empty for fallback/manual review instead.
    for key, kind in (
        ("given_name_en", "given"),
        ("surname_en", "surname"),
        ("father_name_en", "father"),
    ):
        value = _clean_name(result.get(key, ""))
        if value and (
            not _plausible_name(value)
            or _contains_name_label_noise(value, kind)
        ):
            debug[f"{key}_label_noise_rejected"] = value
            result[key] = ""
            src.pop(key, None)

    useful = [result.get(k) for k in (
        "given_name_en", "surname_en", "father_name_en", "passport_number",
        "birth_date", "issue_date", "expiry_date", "nationality", "sex",
    )]
    elapsed_ms = int((time.monotonic() - started) * 1000)

    if not any(useful):
        return JSONResponse(status_code=422, content={
            "success": False,
            "error": "وصلت القصاصات لكن OCR ما استخرج بيانات مفيدة",
            "mode": "pak_label_rescue_v39",
            "server_version": SERVER_VERSION,
            "ocr_time_ms": elapsed_ms,
            "crop_debug": debug,
        })

    result.update({
        "success": True,
        "mode": "pak_label_rescue_v39",
        "client_crop_mode": client_crop_mode or "pak_label_rescue_v39",
        "server_version": SERVER_VERSION,
        "ocr_time_ms": elapsed_ms,
        "received_crops": sorted(data.keys()),
        "crop_debug": debug,
        "ocr_processes": 1 + int(bool(data.get("mrz_crop")))
                         + int(bool(data.get("mrz_enhanced_crop") and mrz_source == "mrz_enhanced_crop"))
                         + int(bool(enhanced_data)),
        "needs_review": (
            not bool(result.get("given_name_en"))
            or not bool(result.get("surname_en"))
            or not bool(result.get("father_name_en"))
            or not bool(result.get("passport_number"))
            or not bool(result.get("issue_date"))
        ),
    })

    result["fields_confidence"] = {
        k: (
            "printed_crop" if src.get(k) in ("printed_batch", "printed_batch_enhanced", "father_retry")
            else "verified" if src.get(k) in ("mrz_crop", "mrz_enhanced_crop") and result.get("is_verified")
            else "mrz"
        )
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
