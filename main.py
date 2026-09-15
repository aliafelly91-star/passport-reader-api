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
SERVER_VERSION = "cloud-app-crop-v10"
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
    img = _resize_min_width(gray, width)
    if img is None:
        return []
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(img)
    blur = cv2.GaussianBlur(clahe, (0, 0), 0.7)
    sharp = cv2.addWeighted(clahe, 1.35, blur, -0.35, 0)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [sharp, otsu]


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
    if any(re.fullmatch(r"[KXS]{2,}", w) for w in words):
        return True
    if any(len(w) == 1 for w in words):
        return True
    return False


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
    m = _clean_name(mrz_name)
    p = _clean_name(printed_name)
    p_ok = _plausible_name(p)
    m_ok = _plausible_name(m) and not _name_has_mrz_noise(m)

    if not p_ok:
        return m
    if not m:
        return p
    if m == p:
        return m

    # Typical filler error: TASSAWARK -> TASSAWAR.
    if len(m) > len(p) and m.startswith(p):
        extra = m[len(p):].replace(" ", "")
        if extra and re.fullmatch(r"[KXS]{1,8}", extra):
            return p

    if not m_ok:
        return p

    if mrz_verified:
        # Do NOT replace a clean MRZ name just because crop OCR differs by one
        # character. This fixes SAMEER ALI -> SAMEER KALI.
        return m

    if _edit(m, p) <= 2:
        return p
    return p if len(p) >= 2 and not m_ok else m


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


def _ocr_crop(data: bytes, kind: str):
    gray = _decode(data)
    if gray is None:
        return ""

    if kind in {"given", "surname", "father"}:
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ,'- "
    elif kind in {"birth", "issue", "expiry"}:
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/."
    elif kind == "sex":
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    elif kind == "nationality":
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    else:
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

    variants = _field_variants(gray, 850 if kind != "sex" else 550)
    if not variants:
        return ""
    sharp, binary = variants[0], variants[-1]

    if kind in {"given", "surname", "father"}:
        # A value crop is usually one short text block. Start with PSM 6 because
        # it is more tolerant of faint Pakistani passport printing.
        first = _ocr(sharp, wl, psm=6)
        if _plausible_name(_clean_name(_strip_label(first, kind))):
            return first
        second = _ocr(sharp, wl, psm=7)
        if _plausible_name(_clean_name(_strip_label(second, kind))):
            return second
        # Binary retry only when both normal reads failed.
        third = _ocr(binary, wl, psm=6)
        candidates = [x for x in (first, second, third) if x]
        valid = [x for x in candidates if _plausible_name(_clean_name(_strip_label(x, kind)))]
        return max(valid or candidates, key=len) if candidates else ""

    if kind in {"birth", "issue", "expiry"}:
        first = _ocr(sharp, wl, psm=7)
        if _find_dates(_strip_label(first, kind)):
            return first
        return _ocr(binary, wl, psm=7) or first

    if kind == "passport":
        first = _ocr(sharp, wl, psm=7)
        flat = re.sub(r"[^A-Z0-9]", "", first)
        if re.search(r"[A-Z0-9]{6,12}", flat) and any(ch.isdigit() for ch in flat):
            return first
        return _ocr(binary, wl, psm=7) or first

    # Nationality/sex should not spend two OCR processes unless the first is empty.
    first = _ocr(sharp, wl, psm=7)
    return first or _ocr(binary, wl, psm=7)


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

    gray = _resize_min_width(gray, 1500)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    passes = [
        (clahe, 11),
        (clahe, 6),
        (otsu, 11),
    ]
    best = None
    best_score = -999
    debug = ""

    for image_variant, psm in passes:
        cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
        try:
            text = pytesseract.image_to_string(image_variant, config=cfg, lang="eng", timeout=7)
        except Exception as exc:
            logger.warning("MRZ OCR failed (%s)", type(exc).__name__)
            continue

        if text and not debug:
            debug = text.strip()

        lines = [_clean_mrz_line(x) for x in text.splitlines()]
        lines = [x for x in lines if len(x) >= 30]
        for i in range(len(lines) - 1):
            parsed = _try_mrz(lines[i], lines[i + 1])
            if parsed and parsed["score"] > best_score:
                best, best_score = parsed, parsed["score"]

        # Important performance fix: if one pass already produced a structurally
        # valid MRZ with document number and dates, stop. Printed value crops will
        # correct names if filler '<' was misread as K.
        if best and best.get("passport_number") and best.get("birth_date") and best.get("expiry_date"):
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
