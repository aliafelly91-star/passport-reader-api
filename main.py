"""
FastAPI passport OCR service.
=====  v7  =====

تغييرات v7 مقابل v4 الأصلية:
  1. _ocr_crop        — كان يحتفظ بأقصر نتيجة OCR، فيضيّع الأسماء.
  2. _mrz_from_bytes  — كان يخرج فوراً عند is_verified حتى لو الأسماء فاضية.
  3. _looks_like_td3  — 🔴 الأهم: حارس جديد يرفض السطور اللي مو MRZ.
                        بدونه كان سطر تاريخ مطبوع مثل "12 AUG 1964"
                        يُقرأ كـMRZ، فتطلع الجنسية "AUG".
  4. GET /debug/{file_name} — نقطة تشخيص مؤقتة (تُحذف بعد الانتهاء).
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
SERVER_VERSION = "cloud-app-crop-v9"
logger = logging.getLogger(__name__)

MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
MONTH_INDEX = {m: i + 1 for i, m in enumerate(MONTHS)}
MONTH_INDEX["SEPT"] = 9

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
    return "" if s.upper() in {"","NONE","NULL","NAN","N/A","NA","-","--"} else s

def _decode(data: bytes, gray=True):
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(arr, flag)
    return img if img is not None and img.size else None

def _enhance(gray, width=1000):
    if gray is None:
        return []
    img = gray
    if img.shape[1] < width:
        scale = width / max(1, img.shape[1])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
    blur = cv2.GaussianBlur(clahe, (0,0), 0.8)
    sharp = cv2.addWeighted(clahe, 1.45, blur, -0.45, 0)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [sharp, otsu]

def _ocr(img, whitelist, psm=7):
    if img is None:
        return ""
    # pytesseract parses config with shlex: the apostrophe in name whitelists
    # must be inside double quotes, or every name read fails before OCR starts.
    cfg = f'--oem 1 --psm {psm} -c preserve_interword_spaces=1 -c tessedit_char_whitelist="{whitelist}"'
    try:
        return re.sub(r"\s+", " ", pytesseract.image_to_string(img, config=cfg, lang="eng")).strip().upper()
    except Exception as exc:
        logger.warning("Field OCR failed (%s)", type(exc).__name__)
        return ""

def _ocr_crop(data: bytes, kind: str):
    gray = _decode(data)
    if gray is None:
        return ""
    if kind in {"given","surname","father"}:
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ,'- "
    elif kind in {"birth","issue","expiry"}:
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/."
    elif kind == "sex":
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    elif kind == "nationality":
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    else:
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

    # ==================================================================
    # 🔴 الإصلاح الأول
    # قبل كان يحتفظ بـ "أقصر" نتيجة بين نسختَي المعالجة:
    #     if raw and (not best or len(raw) < len(best)): best = raw
    # وهذا مقلوب تماماً للأسماء — النسخة اللي قرأت حرفاً واحداً ("S")
    # تفوز على اللي قرأت الاسم كامل ("SYED")، وبعدها _plausible_name
    # يرفض "S" لأن طوله أقل من 2، فيطلع الحقل فاضي.
    # الحين نجمع كل المرشحين ونختار الأنسب حسب نوع الحقل.
    # ==================================================================
    candidates = []
    for v in _enhance(gray, 900 if kind != "sex" else 600):
        raw = _ocr(v, wl, 7)
        if raw:
            candidates.append(raw)

    if not candidates:
        return ""

    if kind in {"given", "surname", "father"}:
        # نفضّل المرشّح الذي يعطي اسماً مقبولاً؛ وإن تعادلا نأخذ الأطول
        valid = [
            c for c in candidates
            if _plausible_name(_clean_name(_strip_label(c, kind)))
        ]
        return max(valid or candidates, key=len)

    if kind in {"birth", "issue", "expiry"}:
        # نفضّل المرشّح الذي يحوي تاريخاً فعلياً
        dated = [c for c in candidates if _find_dates(_strip_label(c, kind))]
        return dated[0] if dated else min(candidates, key=len)

    if kind == "passport":
        # نفضّل المرشّح الذي يحوي رمزاً بطول معقول لرقم جواز
        def _has_number(c):
            flat = re.sub(r"[^A-Z0-9]", "", c.upper())
            return any(
                any(ch.isdigit() for ch in t)
                for t in re.findall(r"[A-Z0-9]{6,12}", flat)
            )
        numbered = [c for c in candidates if _has_number(c)]
        return numbered[0] if numbered else min(candidates, key=len)

    # sex / nationality: الأقصر يبقى مناسباً (قيمها قصيرة أصلاً)
    return min(candidates, key=len)

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
    return len(set(name.replace(" ",""))) > 1

def _edit(a, b):
    if a == b: return 0
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca != cb)))
        prev = cur
    return prev[-1]

def _reconcile(mrz_name, printed_name):
    m = _clean_name(mrz_name)
    p = _clean_name(printed_name)
    if not _plausible_name(p): return m
    if not m: return p
    if m == p or _edit(m, p) <= 2: return p
    if m.startswith(p) and len(m) - len(p) <= 8: return p
    mw, pw = m.split(), p.split()
    if mw[:len(pw)] == pw:
        extra = mw[len(pw):]
        if len(extra) <= 2 and sum(map(len, extra)) <= 8:
            return p
    return m

def _find_dates(text):
    s = (text or "").upper()
    out = []
    def add(y,m,d):
        try:
            v = date(y,m,d)
            if 1900 <= y <= 2100 and v not in out:
                out.append(v)
        except Exception:
            pass
    for m in re.finditer(r"\b(\d{1,2})\s*[-/ ]?\s*([A-Z]{3,4})\s*[-/ ]?\s*(\d{4})\b", s):
        mm = MONTH_INDEX.get(m.group(2))
        if mm: add(int(m.group(3)), mm, int(m.group(1)))
    for m in re.finditer(r"\b([A-Z]{3,4})\s*[-/ ]?\s*(\d{1,2})\s*[-/ ]?\s*(\d{4})\b", s):
        mm = MONTH_INDEX.get(m.group(1))
        if mm: add(int(m.group(3)), mm, int(m.group(2)))
    for m in re.finditer(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", s):
        add(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in re.finditer(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b", s):
        add(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return out

def _date_text(v):
    return f"{v.year}-{MONTHS[v.month-1]}-{v.day:02d}"

def _format_mrz_date(v, birth):
    s = _safe(v)
    if len(s) != 6 or not s.isdigit(): return ""
    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
    if not 1 <= mm <= 12 or not 1 <= dd <= 31: return ""
    year = (1900 + yy if yy > 30 else 2000 + yy) if birth else 2000 + yy
    try:
        return _date_text(date(year, mm, dd))
    except Exception:
        return ""

def _clean_mrz_line(line):
    s = (line or "").upper()
    s = s.replace("«","<").replace("‹","<").replace("_","<").replace("|","<")
    return re.sub(r"[^A-Z0-9<]", "", s)

def _mrz_names(l1):
    body = l1[5:44].rstrip("<")
    parts = body.split("<<", 1)
    sur = _clean_name(parts[0].replace("<"," "))
    # Single '<' separates given names; '<<' starts the trailing padding.
    # OCR may turn later padding into K/S/X. Ignore that tail, not real names.
    given_block = parts[1].split("<<", 1)[0] if len(parts) > 1 else ""
    giv = _clean_name(given_block.replace("<", " "))
    return sur, giv

_MONTH_ABBR = set(MONTHS) | {"SEPT"}

def _looks_like_td3(l1, l2):
    """
    🔴 الإصلاح الثالث — الأهم.
    قبل كان _try_mrz يبلع أي سطرين طولهما 30+ حرفاً ويسلّمهما لـ
    TD3CodeChecker. فلو التقط OCR سطر تاريخ مطبوع مثل "12 AUG 1964"
    صار l1[2:5] = "AUG" ويُعتبر رمز دولة، فتطلع الجنسية "AUG" وبلد
    الإقامة "AUG" — وهذا اللي ظهر بالتطبيق فعلاً.
    الحين نتحقق أن السطرين يطابقان بنية TD3 قبل الوثوق بهما.
    """
    if len(l1) != 44 or len(l2) != 44:
        return False
    if not re.match(r"^P[A-Z<]", l1):
        return False
    code = l1[2:5]
    if not re.fullmatch(r"[A-Z]{3}", code):
        return False
    if code in _MONTH_ABBR:          # "AUG" / "MAY" ليست دولاً
        return False
    if "<<" not in l1[5:]:           # الفاصل بين اللقب والاسم إلزامي
        return False
    if not re.fullmatch(r"[A-Z]{3}", l2[10:13]):
        return False
    if l2[10:13] in _MONTH_ABBR:
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
        f = checker.fields()
        try:
            verified = bool(checker)
        except Exception:
            verified = False
        sur, giv = _mrz_names(l1)
        country = _safe(getattr(f, "country", "")).upper()
        nat = _safe(getattr(f, "nationality", "")).upper() or country
        number = _safe(getattr(f, "document_number", "")).replace("<","").upper()
        sex = l2[20] if len(l2) > 20 and l2[20] in "MF" else ""
        result = {
            "success": True,
            "given_name_en": giv,
            "surname_en": sur,
            "father_name_en": "",
            "passport_number": number,
            "nationality": NATIONALITY_NAMES.get(nat, nat),
            "residence_country": COUNTRY_NAMES.get(country, country),
            "birth_date": _format_mrz_date(getattr(f,"birth_date",""), True),
            "issue_date": "",
            "expiry_date": _format_mrz_date(getattr(f,"expiry_date",""), False),
            "sex": sex,
            "is_verified": verified,
            "is_fully_verified": verified and bool(sur) and bool(number),
            "score": (8 if verified else 0) + sum(bool(x) for x in [sur,giv,number,sex]),
            "mrz_line1": l1,
            "mrz_line2": l2,
        }
        return result
    except Exception:
        return None

def _mrz_from_bytes(data: bytes):
    gray = _decode(data)
    if gray is None: return None, ""
    if gray.shape[1] < 1600:
        scale = 1600 / max(1, gray.shape[1])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best, best_score, debug = None, -1, ""
    cfgs = [
        "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
        "--oem 1 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
    ]
    for img in (clahe, otsu):
        for cfg in cfgs:
            try:
                text = pytesseract.image_to_string(img, config=cfg, lang="eng")
            except Exception as exc:
                logger.warning("MRZ OCR failed (%s)", type(exc).__name__)
                continue
            if text and not debug: debug = text.strip()
            lines = [_clean_mrz_line(x) for x in text.splitlines()]
            lines = [x for x in lines if len(x) >= 30]
            for i in range(len(lines)-1):
                r = _try_mrz(lines[i], lines[i+1])
                if r and r["score"] > best_score:
                    best, best_score = r, r["score"]
                    # ==========================================================
                    # 🔴 الإصلاح الثاني
                    # قبل كان يخرج فوراً عند is_verified وحدها.
                    # لكن معيار ICAO 9303 ما يحط رقم تحقق على الأسماء —
                    # فالسطر ممكن "يُوثَّق" بنجاح والأسماء فاضية، والكود
                    # يرجع بيها فوراً ويترك نسخ المعالجة الباقية بدون
                    # تجريب. الحين ما نخرج مبكراً إلا إذا كانت القراءة
                    # موثّقة *و* اللقب ورقم الجواز موجودَين فعلاً.
                    # ==========================================================
                    if (r["is_verified"]
                            and r.get("surname_en")
                            and r.get("given_name_en")
                            and r.get("passport_number")):
                        return r, debug
    return best, debug

def _blank():
    return {
        "success": True, "given_name_en":"", "surname_en":"", "father_name_en":"",
        "passport_number":"", "nationality":"", "residence_country":"",
        "birth_date":"", "issue_date":"", "expiry_date":"", "sex":"",
        "score":0, "is_verified":False, "is_fully_verified":False, "field_sources":{},
    }

async def _bytes(f: Optional[UploadFile]):
    if f is None: return b""
    try: return await f.read()
    except Exception: return b""

@app.get("/")
def root():
    return {"status":"الخدمة شغالة ✓","ready":True,"version":SERVER_VERSION}

@app.get("/health")
def health():
    return {"status":"ok","ready":True,"version":SERVER_VERSION}

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
        "given_name_crop":given_name_crop, "surname_crop":surname_crop,
        "father_name_crop":father_name_crop, "passport_number_crop":passport_number_crop,
        "birth_date_crop":birth_date_crop, "issue_date_crop":issue_date_crop,
        "expiry_date_crop":expiry_date_crop, "nationality_crop":nationality_crop,
        "sex_crop":sex_crop, "mrz_crop":mrz_crop,
    }
    data = {k: await _bytes(v) for k,v in uploads.items()}
    data = {k:v for k,v in data.items() if v}
    if not data:
        return JSONResponse(status_code=400, content={"success":False,"error":"ما وصلت أي قصاصة","server_version":SERVER_VERSION})

    result = _blank()
    src = result["field_sources"]
    debug = {}

    if data.get("mrz_crop"):
        mrz, raw = _mrz_from_bytes(data["mrz_crop"])
        if raw: debug["mrz_crop"] = re.sub(r"\s+"," ",raw)[:350]
        if mrz:
            for k in ("given_name_en","surname_en","passport_number","nationality","residence_country","birth_date","expiry_date","sex"):
                if mrz.get(k):
                    result[k] = mrz[k]
                    src[k] = "mrz_crop"
            for k in ("score","is_verified","is_fully_verified","mrz_line1","mrz_line2"):
                if k in mrz: result[k] = mrz[k]

    for field, kind, outkey in (
        ("given_name_crop","given","given_name_en"),
        ("surname_crop","surname","surname_en"),
        ("father_name_crop","father","father_name_en"),
    ):
        if data.get(field):
            raw = _ocr_crop(data[field], kind)
            raw = _strip_label(raw, kind)
            if raw: debug[field] = raw[:140]
            val = _clean_name(raw)
            if _plausible_name(val):
                if outkey in ("given_name_en","surname_en"):
                    result[outkey] = _reconcile(result.get(outkey,""), val)
                    src[outkey] = "mrz+app_crop" if result.get("is_verified") else "app_crop"
                else:
                    result[outkey] = val
                    src[outkey] = "app_crop"

    for field, kind, outkey in (
        ("issue_date_crop","issue","issue_date"),
        ("birth_date_crop","birth","birth_date"),
        ("expiry_date_crop","expiry","expiry_date"),
    ):
        if data.get(field) and (outkey == "issue_date" or not result.get(outkey)):
            raw = _strip_label(_ocr_crop(data[field], kind), kind)
            if raw: debug[field] = raw[:120]
            ds = _find_dates(raw)
            if ds:
                result[outkey] = _date_text(ds[0])
                src[outkey] = "app_crop"

    if data.get("passport_number_crop") and not result.get("passport_number"):
        raw = _strip_label(_ocr_crop(data["passport_number_crop"], "passport"), "passport")
        if raw: debug["passport_number_crop"] = raw[:100]
        tokens = re.findall(r"[A-Z0-9]{6,12}", re.sub(r"[^A-Z0-9]","",raw))
        tokens = [t for t in tokens if any(c.isdigit() for c in t)]
        if tokens:
            result["passport_number"] = min(tokens, key=lambda x: abs(len(x)-9))
            src["passport_number"] = "app_crop"

    if data.get("sex_crop") and not result.get("sex"):
        raw = _strip_label(_ocr_crop(data["sex_crop"], "sex"), "sex")
        if raw: debug["sex_crop"] = raw[:80]
        if "FEMALE" in raw or re.search(r"\bF\b", raw): result["sex"] = "F"
        elif "MALE" in raw or re.search(r"\bM\b", raw): result["sex"] = "M"
        if result["sex"]: src["sex"] = "app_crop"

    if data.get("nationality_crop") and not result.get("nationality"):
        raw = _strip_label(_ocr_crop(data["nationality_crop"], "nationality"), "nationality")
        raw = re.sub(r"[^A-Z ]"," ",raw)
        raw = re.sub(r"\s+"," ",raw).strip()
        if raw: debug["nationality_crop"] = raw[:100]
        if raw in NATIONALITY_NAMES: raw = NATIONALITY_NAMES[raw]
        elif raw in COUNTRY_NAMES.values():
            for code,country in COUNTRY_NAMES.items():
                if raw == country:
                    raw = NATIONALITY_NAMES.get(code, raw)
                    break
        if (raw and not any(w in LABEL_WORDS or w in _MONTH_ABBR
                            for w in raw.split())):
            result["nationality"] = raw
            src["nationality"] = "app_crop"

    father = result.get("father_name_en","")
    if father and not _plausible_name(father):
        result["father_name_en"] = ""
        src.pop("father_name_en", None)

    useful = [result.get(k) for k in ("given_name_en","surname_en","passport_number","birth_date","expiry_date","father_name_en","issue_date")]
    if not any(useful):
        return JSONResponse(status_code=422, content={
            "success":False,"error":"وصلت القصاصات لكن OCR ما استخرج بيانات مفيدة",
            "mode":"app_field_crops","server_version":SERVER_VERSION,"crop_debug":debug,
        })

    result.update({
        "success":True,
        "mode":"app_field_crops",
        "client_crop_mode":client_crop_mode or "unknown",
        "server_version":SERVER_VERSION,
        "ocr_time_ms":int((time.monotonic()-started)*1000),
        "received_crops":sorted(data.keys()),
        "crop_debug":debug,
        "needs_review":not bool(result.get("father_name_en")) or not bool(result.get("issue_date")),
    })
    result["fields_confidence"] = {
        k: ("verified" if src.get(k) == "mrz_crop" and result.get("is_verified") else "printed_crop")
        for k in ("given_name_en","surname_en","father_name_en","passport_number","nationality","birth_date","issue_date","expiry_date","sex")
        if result.get(k)
    }
    return JSONResponse(content=result)

@app.post("/read-passport")
async def read_passport(file: UploadFile = File(...)):
    """Backward-compatible fallback. Fast path reads MRZ from the bottom of the passport."""
    started = time.monotonic()
    b = await file.read()
    color = _decode(b, gray=False)
    if color is None:
        return JSONResponse(status_code=400, content={"success":False,"error":"الصورة فارغة أو غير صالحة"})
    h,w = color.shape[:2]
    best = None
    for img in (color, cv2.rotate(color, cv2.ROTATE_180)):
        hh,ww = img.shape[:2]
        crop = img[int(hh*0.55):hh, 0:ww]
        ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok: continue
        r,_ = _mrz_from_bytes(enc.tobytes())
        if r and (best is None or r["score"] > best["score"]):
            best = r
        if r and r.get("is_verified"): break
    if best is None:
        return JSONResponse(status_code=422, content={"success":False,"error":"ما قدرنا نلقى MRZ واضح بالصورة","server_version":SERVER_VERSION})
    best["server_version"] = SERVER_VERSION
    best["mode"] = "full_image_mrz_fallback"
    best["ocr_time_ms"] = int((time.monotonic()-started)*1000)
    best["needs_review"] = True
    best.setdefault("field_sources", {})
    for k in ("given_name_en","surname_en","passport_number","nationality","residence_country","birth_date","expiry_date","sex"):
        if best.get(k): best["field_sources"][k] = "mrz"
    return JSONResponse(content=best)


# ══════════════════════════════════════════════════════════════════════
# 🔬 نقطة تشخيص مؤقتة — احذفها بعد ما نخلّص
# ══════════════════════════════════════════════════════════════════════
# تنزّل صورة جواز من رابط عام وتشغّل عليها نفس مسار MRZ، وترجّع
# نص OCR الخام. الهدف: نشوف بأم العين شنو يقرأه Tesseract بدل ما
# نخمّن. مقيّدة بمضيف Supabase حقك فقط حتى ما تنفتح كبوابة تنزيل.
#
#   GET /debug-read?url=https://hifkuvyvhrxmcgkbvgqo.supabase.co/...jpg
# ══════════════════════════════════════════════════════════════════════

import urllib.request
from urllib.parse import urlparse

ALLOWED_IMAGE_HOST = "hifkuvyvhrxmcgkbvgqo.supabase.co"


def _debug_region(color, y_from: float, y_to: float, label: str):
    """يقص منطقة عمودية من الصورة ويرجّع قراءة MRZ منها."""
    hh, ww = color.shape[:2]
    crop = color[int(hh * y_from):int(hh * y_to), 0:ww]
    ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        return {"region": label, "error": "تعذر الترميز"}
    r, raw = _mrz_from_bytes(enc.tobytes())
    out = {
        "region": label,
        "crop_size": [crop.shape[1], crop.shape[0]],
        "raw_ocr": re.sub(r"\s+", " ", raw)[:600],
    }
    if r:
        out["parsed"] = {
            k: r.get(k) for k in (
                "surname_en", "given_name_en", "passport_number",
                "nationality", "residence_country", "birth_date",
                "expiry_date", "sex", "is_verified", "score",
                "mrz_line1", "mrz_line2",
            )
        }
    else:
        out["parsed"] = None
    return out


@app.get("/debug/{file_name}")
def debug_read(file_name: str):
    """
    تشخيص: تنزّل صورة جواز من مخزن Supabase وتشغّل مسار MRZ على ثلاث
    مناطق، وترجّع نص OCR الخام. ترجع 200 دائماً حتى نقرأ سبب أي فشل.
    مثال:  /debug/1787927303912-2.jpg
    """
    out = {"server_version": SERVER_VERSION, "file": file_name}

    if "/" in file_name or ".." in file_name:
        out["error"] = "اسم ملف غير صالح"
        return JSONResponse(content=out)

    url = (
        f"https://{ALLOWED_IMAGE_HOST}"
        f"/storage/v1/object/public/passport-files/{file_name}"
    )
    out["url"] = url

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "passport-reader-debug/1.0"})
        with urllib.request.urlopen(req, timeout=40) as resp:
            b = resp.read()
    except Exception as e:
        out["error"] = f"تعذر التنزيل: {type(e).__name__}: {e}"
        return JSONResponse(content=out)

    out["bytes"] = len(b)
    color = _decode(b, gray=False)
    if color is None:
        out["error"] = "تعذر فك ترميز الصورة"
        return JSONResponse(content=out)

    hh, ww = color.shape[:2]
    out["image_size"] = [ww, hh]
    try:
        out["tesseract"] = str(pytesseract.get_tesseract_version())
    except Exception as e:
        out["tesseract"] = f"غير متاح: {e}"

    out["regions"] = [
        _debug_region(color, 0.55, 1.00, "bottom_45"),
        _debug_region(color, 0.69, 1.00, "bottom_31"),
        _debug_region(color, 0.80, 1.00, "bottom_20"),
    ]
    return JSONResponse(content=out)
