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
from PIL import Image

# Render has a small/burstable CPU. Tesseract/OpenMP trying to use several
# threads is fast on a desktop but can get heavily throttled in the cloud.
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
SERVER_VERSION = "cloud-app-crop-v34"
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
        if len(w) >= 4:
            filler_ratio = sum(ch in "KXS" for ch in w) / len(w)
            if filler_ratio >= 0.55:
                return True
        if re.fullmatch(r"[KXS]{2,}", w):
            return True
    return False


def _strip_name_noise_tokens(name):
    n = _clean_name(name)
    if not n:
        return ""
    words = n.split()
    words = [w for w in words if not (len(w) == 1 and w in {"G", "N", "F", "P"})]
    return " ".join(words).strip()


def _choose_name(mrz_name, printed_name, mrz_verified=False):
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

        if len(a) == len(b) + 1 and a[-1] in "KXS" and a[:-1] == b:
            changed = True
            continue

        if (
            idx > 0
            and len(a) == len(b) + 1
            and a[0] in "KXS"
            and a[1:] == b
        ):
            changed = True
            continue

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


def _quick_variant(gray, width=900, clahe=False):
    img = _resize_min_width(gray, width)
    if clahe:
        img = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(img)
    return img


def _ocr_crop(data: bytes, kind: str):
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
    gray = _decode(data)
    if gray is None or gray.size == 0:
        return None

    if kind in {"nationality", "sex"}:
        gray = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(gray)

    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return None

    scale = target_h / max(1, h)
    new_w = max(120, min(1800, int(w * scale)))
    resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    if kind in {"given", "surname", "father", "passport"}:
        blur = cv2.GaussianBlur(resized, (0, 0), 0.55)
        resized = cv2.addWeighted(resized, 1.22, blur, -0.22, 0)

    return cv2.copyMakeBorder(
        resized, 12, 12, 18, 18,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def _ocr_crop_fallback(im, kind):
    if im is None or im.size == 0:
        return ""
    cfg = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    try:
        text = pytesseract.image_to_string(im, config=cfg, lang="eng", timeout=8)
        return re.sub(r"\s+", " ", text).strip().upper()
    except Exception:
        return ""


def _batch_ocr_printed(data: dict):
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

            if kind == "father":
                alt = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(im)
                prepared.append((field, kind, alt))
            elif kind == "issue":
                _, alt = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                prepared.append((field, kind, alt))

    if not prepared:
        return {}, ""

    cfg = (
        "--oem 1 --psm 6 "
        "-c preserve_interword_spaces=1 "
        '-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.\'/- "'
    )

    ocr = None
    page_to_field = {}

    try:
        with tempfile.TemporaryDirectory(prefix="passport_ocr_") as tmp:
            tmp_path = Path(tmp)
            pil_images = []

            for page_num, (field, kind, im) in enumerate(prepared, start=1):
                pil_images.append(Image.fromarray(im))
                page_to_field[page_num] = field

            if not pil_images:
                return {}, ""

            tif_path = tmp_path / "multipage.tif"
            pil_images[0].save(
                str(tif_path),
                save_all=True,
                append_images=pil_images[1:],
                format="TIFF",
            )

            ocr = pytesseract.image_to_data(
                str(tif_path),
                config=cfg,
                lang="eng",
                output_type=pytesseract.Output.DICT,
                timeout=26,
            )
    except Exception as exc:
        logger.warning("Printed multipage OCR failed (%s): %s", type(exc).__name__, exc)
        ocr = None

    # Fallback to individual crop OCR if multi-page batch OCR returned no results
    if not ocr or not ocr.get("text"):
        raw = {}
        for field, kind, im in prepared:
            if field in raw and raw[field]:
                continue
            txt = _ocr_crop_fallback(im, kind)
            if txt:
                raw[field] = txt
        debug_text = " | ".join(f"{k}={v}" for k, v in raw.items() if v)
        return raw, debug_text[:1200]

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
                words = re.findall(r"[A-Z]{2,}", line_text)
                label_hits = sum(1 for w in words if w in LABEL_WORDS)
                label_penalty = 35 * label_hits
                comma_bonus = 24 if "," in line_text else 0
                count_bonus = 22 if 2 <= len(words) <= 4 else (-15 if len(words) > 5 else 0)
                short_noise = sum(1 for w in words if len(w) <= 2)
                noise_penalty = short_noise * 8
                return avg_conf + alpha_len * 3.5 + comma_bonus + count_bonus - label_penalty - noise_penalty
            raw[field] = max(line_rows, key=father_line_score)[0]
        else:
            raw[field] = " ".join(row[0] for row in line_rows)

    debug_text = " | ".join(f"{k}={v}" for k, v in raw.items() if v)
    return raw, debug_text[:1200]


def _words_only(text):
    return re.findall(r"[A-Z]{2,}", (text or "").upper())


def _best_printed_name(raw, kind, mrz_hint=""):
    s = re.sub(r"\s+", " ", (raw or "").upper()).strip()
    if not s:
        return ""

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
                if len(w) < 3:
                    break
                kept.append(w)
                if len(kept) >= 2:
                    break
            if kept:
                candidate = _clean_name(f"{left}, {' '.join(kept)}")
                if _plausible_name(candidate):
                    return candidate

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
        tail = re.split(
            r"\b(?:DATE|ISSUE|EXPIRY|NATIONALITY|ISSUING|AUTHORITY|PASSPORT|SEX|PLACE)\b",
            s,
            maxsplit=1,
        )[0]
        fw = [w for w in _words_only(tail) if w not in LABEL_WORDS and len(w) >= 2]
        if fw:
            kept = []
            for w in fw:
                if len(kept) >= 2 and len(w) <= 2:
                    break
                kept.append(w)
                if len(kept) >= 4:
                    break
            for n in range(min(4, len(kept)), 1, -1):
                candidate = _clean_name(" ".join(kept[:n]))
                if _plausible_name(candidate):
                    if n > 2 and len(kept[n - 1]) <= 3:
                        shorter = _clean_name(" ".join(kept[:n - 1]))
                        if _plausible_name(shorter):
                            return shorter
                    return candidate

    words = [w for w in _words_only(s) if w not in LABEL_WORDS]
    if not words:
        return ""

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

        def edit_dist(a, b):
            if a == b: return 0
            p = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                c = [i]
                for j, cb in enumerate(b, 1):
                    c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
                p = c
            return p[-1]

        return min(expanded, key=lambda c: (edit_dist(c, hint), abs(len(c) - len(hint)), len(c)))

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
        tokens = [t for t in tokens if t not in LABEL_WORDS and any(ch.isdigit() for ch in t)]
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
    table = str.maketrans({"O":"0", "Q":"0", "D":"0", "I":"1", "L":"1", "Z":"2", "S":"5", "B":"8", "G":"6"})
    return (text or "").translate(table)


def _mrz_name_block(block):
    raw = (block or "").upper()
    raw = re.sub(r"(?<=[A-Z])[KXSCBO]{4,}$", "", raw)
    parts = [p for p in re.split(r"<+", raw) if p]
    words = []

    for token in parts:
        t = re.sub(r"[^A-Z]", "", token)
        if not t:
            continue

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


def _parse_mrz_fuzzy(l1, l2):
    a = _clean_mrz_line(l1)
    b = _clean_mrz_line(l2)
    if len(a) < 20 or not a.startswith("P"):
        return None

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
    parts = re.split(r"<{2,}", name_body, maxsplit=1)
    if len(parts) < 2:
        return None
    sur = _mrz_name_block(parts[0])
    giv = _mrz_name_block(parts[1])
    if not _plausible_name(sur):
        sur = ""
    if not _plausible_name(giv):
        giv = ""

    nat_pos = -1
    nat = ""
    for code in NATIONALITY_NAMES:
        idx = b.find(code, 7, 15)
        if idx >= 0:
            nat, nat_pos = code, idx
            break
    if nat_pos < 0:
        idx = b.find(country, 7, 15)
        if idx >= 0:
            nat, nat_pos = country, idx
    if nat_pos < 0:
        return None

    prefix = b[:nat_pos]
    number = re.sub(r"<", "", prefix[:9]).upper()
    if len(number) < 6:
        return None

    tail = b[nat_pos + 3:]
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
    lines = [_clean_mrz_line(x) for x in (text or "").splitlines()]
    lines = [x for x in lines if len(x) >= 20]

    best = None
    for i in range(len(lines) - 1):
        best = _prefer_mrz_candidate(best, _try_mrz(lines[i], lines[i + 1]))
        best = _prefer_mrz_candidate(best, _parse_mrz_fuzzy(lines[i], lines[i + 1]))

    return best, debug


def _choose_passport_number(mrz_number, printed_number, country="", mrz_verified=False):
    m = re.sub(r"[^A-Z0-9]", "", (mrz_number or "").upper())
    p = re.sub(r"[^A-Z0-9]", "", (printed_number or "").upper())
    country_up = (country or "").upper()

    def normalize_pak(v):
        if len(v) != 9:
            return v
        prefix_map = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
        digit_map = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
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
            if p == m:
                return m
            def edit_dist(a, b):
                p_arr = list(range(len(b) + 1))
                for i, ca in enumerate(a, 1):
                    c = [i]
                    for j, cb in enumerate(b, 1):
                        c.append(min(p_arr[j] + 1, c[j - 1] + 1, p_arr[j - 1] + (ca != cb)))
                    p_arr = c
                return p_arr[-1]
            return p if edit_dist(p, m) <= 2 and not mrz_verified else m

    if p and m:
        def edit_dist(a, b):
            p_arr = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                c = [i]
                for j, cb in enumerate(b, 1):
                    c.append(min(p_arr[j] + 1, c[j - 1] + 1, p_arr[j - 1] + (ca != cb)))
                p_arr = c
            return p_arr[-1]

        if len(p) == len(m) and edit_dist(p, m) <= 1:
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

    batch_raw, batch_debug = _batch_ocr_printed(data)
    if batch_debug:
        debug["printed_batch"] = batch_debug

    mrz = None
    if data.get("mrz_crop"):
        mrz, mrz_raw = _mrz_from_bytes(data["mrz_crop"])
        if mrz_raw:
            debug["mrz_crop"] = re.sub(r"\s+", " ", mrz_raw)[:350]

    if mrz:
        for k in (
            "given_name_en", "surname_en", "passport_number", "nationality",
            "residence_country", "birth_date", "expiry_date", "sex",
        ):
            if mrz.get(k):
                result[k] = mrz[k]
                src[k] = "mrz_crop"
        for k in ("score", "is_verified", "is_fully_verified", "mrz_line1", "mrz_line2"):
            if k in mrz:
                result[k] = mrz[k]

    printed, printed_debug = _parse_printed_batch(batch_raw, mrz)
    debug.update(printed_debug)

    if not printed.get("father_name_en") and data.get("father_name_crop"):
        father_retry_raw = _ocr_crop(data["father_name_crop"], "father")
        if father_retry_raw:
            debug["father_retry"] = father_retry_raw[:180]
            father_retry = _best_printed_name(father_retry_raw, "father", "")
            if father_retry:
                printed["father_name_en"] = father_retry

    if not printed.get("issue_date") and data.get("issue_date_crop"):
        issue_retry_raw = _ocr_crop(data["issue_date_crop"], "issue")
        if issue_retry_raw:
            debug["issue_retry"] = issue_retry_raw[:180]
            issue_dates = _find_dates(issue_retry_raw)
            if issue_dates:
                printed["issue_date"] = _date_text(issue_dates[0])

    for key in ("given_name_en", "surname_en"):
        p = printed.get(key, "")
        if not p:
            continue

        if key == "surname_en" and mrz is not None and not mrz.get("surname_en"):
            debug["surname_printed_ignored_mrz_blank"] = p
            continue

        chosen = _choose_name(
            result.get(key, ""),
            p,
            bool(result.get("is_verified")),
        )
        if chosen != result.get(key, "") or not result.get(key):
            result[key] = chosen
            src[key] = "printed_batch"

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
            src["father_name_en"] = "printed_batch"

    issue_candidate = printed.get("issue_date")
    if issue_candidate:
        if issue_candidate in {
            result.get("birth_date"),
            result.get("expiry_date"),
        }:
            debug["issue_rejected"] = issue_candidate
        else:
            result["issue_date"] = issue_candidate
            src["issue_date"] = "printed_batch"

    chosen_number = _choose_passport_number(
        result.get("passport_number", ""),
        printed.get("passport_number", ""),
        result.get("residence_country", ""),
        bool(result.get("is_verified")),
    )
    if chosen_number:
        if chosen_number == printed.get("passport_number"):
            src["passport_number"] = "printed_batch"
        result["passport_number"] = chosen_number

    for key in ("birth_date", "expiry_date", "nationality", "sex"):
        if not result.get(key) and printed.get(key):
            result[key] = printed[key]
            src[key] = "printed_batch"

    if not result.get("residence_country") and (result.get("nationality") or "").upper() == "PAKISTANI":
        result["residence_country"] = "PAKISTAN"
        src["residence_country"] = "nationality_fallback"

    useful = [result.get(k) for k in (
        "given_name_en", "surname_en", "father_name_en", "passport_number",
        "birth_date", "issue_date", "expiry_date", "nationality", "sex",
    )]
    elapsed_ms = int((time.monotonic() - started) * 1000)

    if not any(useful):
        return JSONResponse(status_code=422, content={
            "success": False,
            "error": "وصلت القصاصات لكن OCR ما استخرج بيانات مفيدة",
            "mode": "pak_row_model_10_v34",
            "server_version": SERVER_VERSION,
            "ocr_time_ms": elapsed_ms,
            "crop_debug": debug,
        })

    result.update({
        "success": True,
        "mode": "pak_row_model_10_v34",
        "client_crop_mode": client_crop_mode or "pak_row_model_10_v34",
        "server_version": SERVER_VERSION,
        "ocr_time_ms": elapsed_ms,
        "received_crops": sorted(data.keys()),
        "crop_debug": debug,
        "ocr_processes": 2 if data.get("mrz_crop") else 1,
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
            "printed_crop" if src.get(k) == "printed_batch"
            else "verified" if src.get(k) == "mrz_crop" and result.get("is_verified")
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
