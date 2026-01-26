import os
import re
import io
import json
import hashlib
import logging
import threading
import sqlite3
import unicodedata
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from difflib import SequenceMatcher

import requests
from PIL import Image
import imagehash
from zoneinfo import ZoneInfo

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# =========================
# Config
# =========================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")

TZ_CDMX = ZoneInfo("America/Mexico_City")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lacra")

# =========================
# Render dummy server
# =========================
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

# =========================
# Time formatting
# =========================
_ES_MONTHS = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

def format_cdmx(dt_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
    except Exception:
        return dt_iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(TZ_CDMX)
    hour = local.hour
    ampm = "AM" if hour < 12 else "PM"
    hour12 = hour % 12
    if hour12 == 0:
        hour12 = 12
    return f"{local.day:02d} {_ES_MONTHS[local.month]} {local.year}, {hour12}:{local.minute:02d} {ampm} (CDMX)"

# =========================
# Image utilities
# =========================
def to_real_jpeg_bytes(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=92, optimize=True)
        return out.getvalue()

def compute_image_phash_hex(jpeg_bytes: bytes) -> str:
    with Image.open(io.BytesIO(jpeg_bytes)) as im:
        im = im.convert("RGB")
        h = imagehash.phash(im)
        return str(h)

def phash_distance(hex_a: str, hex_b: str) -> int:
    try:
        return imagehash.hex_to_hash(hex_a) - imagehash.hex_to_hash(hex_b)
    except Exception:
        return 999

# =========================
# Text normalization
# =========================
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def normalize_text_for_hash(s: str) -> str:
    s = (s or "").upper()
    s = strip_accents(s)
    s = re.sub(r"[^A-Z0-9/\n ]+", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    return s

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# =========================
# OCR regex basics
# =========================
CURP_RE = re.compile(r"\b[A-Z]{4}\d{6}[A-Z]{6}\d{2}\b")
CLAVE_ELECTOR_RE = re.compile(r"\b[A-Z0-9]{18}\b")
DOB_RE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")

LABELS = [
    "DOMICILIO", "CLAVE DE ELECTOR", "CURP", "FECHA DE NACIMIENTO",
    "SECCION", "VIGENCIA", "ANO DE REGISTRO", "SEXO", "NOMBRE"
]

def _is_label_line(ln: str) -> bool:
    return any(ln == lab or ln.startswith(lab + " ") for lab in LABELS)

def extract_block_after_label(text_norm: str, label: str, max_lines: int = 6) -> str:
    lines = [ln.strip() for ln in text_norm.splitlines() if ln.strip()]
    grab = False
    out = []
    for ln in lines:
        if ln == label or ln.startswith(label + " "):
            rest = ln[len(label):].strip()
            if rest:
                out.append(rest)
            grab = True
            continue
        if grab:
            if _is_label_line(ln):
                break
            if len(ln) >= 2:
                out.append(ln)
            if len(out) >= max_lines:
                break
    return "\n".join(out).strip()

def extract_name_block_ine(text_norm: str) -> str:
    block = extract_block_after_label(text_norm, "NOMBRE", max_lines=4)
    cleaned = []
    for ln in block.splitlines():
        if re.fullmatch(r"[A-Z ]{2,}", ln):
            cleaned.append(ln)
    return " ".join(cleaned).strip()

# =========================
# Document profiles (Option 2)
# =========================
DOC_PROFILES = {
    "INE_MX": {
        "keywords": ["CREDENCIAL PARA VOTAR", "INSTITUTO NACIONAL ELECTORAL", "ELECTOR", "SECCION", "VIGENCIA", "INE"],
        "id_fields_priority": ["curp", "clave_elector"],
        "fields": {
            "name": {"label": "NOMBRE", "max_lines": 4},
            "domicilio": {"label": "DOMICILIO", "max_lines": 6},
            "curp": {"regex": CURP_RE},
            "clave_elector": {"label": "CLAVE DE ELECTOR", "regex": CLAVE_ELECTOR_RE},
            "dob": {"label": "FECHA DE NACIMIENTO", "regex": DOB_RE},
            "sexo": {"label": "SEXO", "max_lines": 1},
            "seccion": {"label": "SECCION", "max_lines": 2},
            "vigencia": {"label": "VIGENCIA", "max_lines": 2},
            "ano_registro": {"label": "ANO DE REGISTRO", "max_lines": 2},
        },
        "diff_fields": ["domicilio", "vigencia", "seccion", "sexo", "ano_registro"],
    },

    "PASSPORT_MX": {
        "keywords": ["PASAPORTE", "ESTADOS UNIDOS MEXICANOS", "MEXICO", "PASSPORT", "NATIONALITY", "NACIONALIDAD"],
        "id_fields_priority": ["passport_no", "curp"],
        "fields": {
            "passport_no": {"regex": re.compile(r"\b([A-Z]\d{7,9}|\d{8,10})\b")},
            "dob": {"regex": re.compile(r"\b(\d{2}/\d{2}/\d{4}|\d{2}\s?[A-Z]{3}\s?\d{4})\b")},
            "nationality": {"regex": re.compile(r"\b(MEXICANA|MEXICAN|MEXICO)\b")},
            "name": {"regex": re.compile(r"\b[A-Z]{2,}\s+[A-Z]{2,}(\s+[A-Z]{2,})+\b")},
            "sex": {"regex": re.compile(r"\b(M|F)\b")},
            "expiry": {"regex": re.compile(r"\b(\d{2}/\d{2}/\d{4}|\d{2}\s?[A-Z]{3}\s?\d{4})\b")},
        },
        "diff_fields": ["passport_no", "expiry", "nationality"],
    },

    "LICENSE_MX": {
        "keywords": ["LICENCIA", "CONDUCIR", "DRIVER", "VIGENCIA"],
        "id_fields_priority": ["license_no", "curp"],
        "fields": {
            "name": {"label": "NOMBRE", "max_lines": 3},
            "license_no": {"regex": re.compile(r"\b([A-Z0-9]{8,18})\b")},
            "dob": {"regex": DOB_RE},
            "expiry": {"regex": re.compile(r"\b(VIGENCIA|EXPIRA|EXPIRY)\b")},
            "address": {"regex": re.compile(r"\b(DOMICILIO|DIRECCION|ADDRESS)\b")},
        },
        "diff_fields": ["license_no", "expiry", "address"],
    },

    "UNKNOWN": {
        "keywords": [],
        "id_fields_priority": [],
        "fields": {},
        "diff_fields": [],
    }
}

def detect_doc_type(text_norm: str) -> str:
    # Heurística fuerte para INE aunque falten keywords exactas
    ine_signals = 0
    if "CREDENCIAL PARA VOTAR" in text_norm:
        ine_signals += 2
    if "INSTITUTO NACIONAL ELECTORAL" in text_norm or "INSTITUTO FEDERAL ELECTORAL" in text_norm:
        ine_signals += 2
    if "CLAVE DE ELECTOR" in text_norm:
        ine_signals += 2
    if "DOMICILIO" in text_norm:
        ine_signals += 1
    if "SECCION" in text_norm:
        ine_signals += 1
    if CURP_RE.search(text_norm):
        ine_signals += 2

    if ine_signals >= 4:
        return "INE_MX"

    best = ("UNKNOWN", 0)
    for dt, prof in DOC_PROFILES.items():
        if dt == "UNKNOWN":
            continue
        hits = sum(1 for kw in prof["keywords"] if kw in text_norm)
        if hits > best[1]:
            best = (dt, hits)
    return best[0] if best[1] > 0 else "UNKNOWN"

def extract_by_profile(text_norm: str, doc_type: str) -> dict:
    prof = DOC_PROFILES.get(doc_type, DOC_PROFILES["UNKNOWN"])
    out = {}

    for field, spec in prof.get("fields", {}).items():
        val = ""

        if doc_type == "INE_MX" and field == "name":
            val = extract_name_block_ine(text_norm)

        if "label" in spec and not val:
            val = extract_block_after_label(text_norm, spec["label"], max_lines=spec.get("max_lines", 4))

        if "regex" in spec and not val:
            rx = spec["regex"]
            if "label" in spec and spec["label"] in text_norm:
                idx = text_norm.find(spec["label"])
                window = text_norm[idx: idx + 350]
                m = rx.search(window)
            else:
                m = rx.search(text_norm)
            if m:
                val = m.group(0)

        val = (val or "").strip()
        if val:
            out[field] = val

    return out

def build_person_key(doc_type: str, fields: dict) -> tuple[str, str]:
    prof = DOC_PROFILES.get(doc_type, DOC_PROFILES["UNKNOWN"])
    for k in prof.get("id_fields_priority", []):
        v = (fields.get(k) or "").strip()
        if v:
            return v, f"{doc_type}:{k}"

    name = (fields.get("name") or "").strip()
    dob = (fields.get("dob") or "").strip()
    if name and dob:
        return sha256_hex(f"{name}|{dob}")[:24], f"{doc_type}:NAME_DOB"

    return "", ""

# =========================
# OCR.space
# =========================
def ocr_space_bytes(jpeg_bytes: bytes) -> str:
    if not OCR_SPACE_API_KEY:
        raise RuntimeError("Falta OCR_SPACE_API_KEY en Environment Variables")

    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.jpg", jpeg_bytes)}
    data = {
        "apikey": OCR_SPACE_API_KEY,
        "language": "spa",
        "OCREngine": "2",
        "scale": "true",
        "detectOrientation": "true",
        "isOverlayRequired": "false",
        "filetype": "JPG",
    }

    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    payload = r.json()

    log.info("OCR RAW RESPONSE >>> %s", json.dumps(payload)[:1200])

    if payload.get("IsErroredOnProcessing"):
        msg = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Error OCR desconocido"
        raise RuntimeError(str(msg))

    results = payload.get("ParsedResults") or []
    if not results:
        return ""

    return (results[0].get("ParsedText") or "").strip()

# =========================
# DB schema + migrations
# =========================
def ensure_column(cur, table: str, column: str, ddl: str):
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if column not in cols:
        log.info("DB migration: adding column %s.%s", table, column)
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

def init_db():
    os.makedirs(os.path.dirname(DBPATH), exist_ok=True)
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ocr_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            user_id INTEGER,
            message_id INTEGER,
            file_unique_id TEXT,
            created_at TEXT,
            ocr_text TEXT
        )
    """)

    ensure_column(cur, "ocr_texts", "ocr_status", "ocr_status TEXT")
    ensure_column(cur, "ocr_texts", "ocr_error", "ocr_error TEXT")
    ensure_column(cur, "ocr_texts", "image_hash", "image_hash TEXT")
    ensure_column(cur, "ocr_texts", "text_norm", "text_norm TEXT")
    ensure_column(cur, "ocr_texts", "text_hash", "text_hash TEXT")

    ensure_column(cur, "ocr_texts", "doc_type", "doc_type TEXT")
    ensure_column(cur, "ocr_texts", "fields_json", "fields_json TEXT")

    ensure_column(cur, "ocr_texts", "person_key", "person_key TEXT")
    ensure_column(cur, "ocr_texts", "person_key_type", "person_key_type TEXT")

    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_chat_created ON ocr_texts(chat_id, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_text_hash ON ocr_texts(text_hash)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_image_hash ON ocr_texts(image_hash)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_person_key ON ocr_texts(person_key)")
    except Exception as e:
        log.warning("Index creation warning: %s", e)

    conn.commit()
    conn.close()
    log.info("DB inicializada en: %s", DBPATH)

def db_conn():
    return sqlite3.connect(DBPATH)

# =========================
# DB queries
# =========================
def find_exact_duplicate(cur, chat_id: int, text_hash: str, image_hash: str):
    if text_hash:
        cur.execute("""
            SELECT id, created_at, message_id, text_hash, image_hash
            FROM ocr_texts
            WHERE chat_id=? AND text_hash=?
            ORDER BY id ASC LIMIT 1
        """, (chat_id, text_hash))
        row = cur.fetchone()
        if row:
            return ("TEXT", row)

    if image_hash and text_hash:
        cur.execute("""
            SELECT id, created_at, message_id, text_hash, image_hash
            FROM ocr_texts
            WHERE chat_id=? AND image_hash=?
            ORDER BY id ASC LIMIT 1
        """, (chat_id, image_hash))
        row = cur.fetchone()
        if row:
            prev_text_hash = row[3]
            if prev_text_hash == text_hash:
                return ("IMAGE", row)

    return None

def find_first_by_person(cur, chat_id: int, person_key: str):
    if not person_key:
        return None
    cur.execute("""
        SELECT id, created_at, message_id, doc_type, person_key_type
        FROM ocr_texts
        WHERE chat_id=? AND person_key=?
        ORDER BY id ASC LIMIT 1
    """, (chat_id, person_key))
    return cur.fetchone()

def find_latest_by_person(cur, chat_id: int, person_key: str):
    if not person_key:
        return None
    cur.execute("""
        SELECT id, created_at, message_id, doc_type, fields_json
        FROM ocr_texts
        WHERE chat_id=? AND person_key=?
        ORDER BY id DESC LIMIT 1
    """, (chat_id, person_key))
    return cur.fetchone()

def find_fuzzy_suggestions(cur, chat_id: int, name_now: str, image_hash_now: str, limit: int = 150):
    cur.execute("""
        SELECT id, created_at, message_id, image_hash, doc_type, fields_json
        FROM ocr_texts
        WHERE chat_id=?
        ORDER BY id DESC
        LIMIT ?
    """, (chat_id, limit))
    rows = cur.fetchall()

    scored = []
    for (rid, created_at, mid, img_hash, doc_type, fields_json) in rows:
        img_score = 0.0
        name_score = 0.0

        try:
            f = json.loads(fields_json) if fields_json else {}
        except Exception:
            f = {}
        name_prev = (f.get("name") or "").strip()

        if image_hash_now and img_hash:
            d = phash_distance(image_hash_now, img_hash)
            img_score = max(0.0, 1.0 - (d / 14.0))

        if name_now and name_prev:
            name_score = similarity_ratio(name_now, name_prev)

        combined = (0.6 * img_score) + (0.4 * name_score)

        if combined >= 0.72 or img_score >= 0.82 or name_score >= 0.88:
            scored.append((combined, img_score, name_score, rid, created_at, mid, doc_type, name_prev))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:3]

def insert_record(
    chat_id: int,
    user_id: int,
    message_id: int,
    file_unique_id: str,
    created_at_iso: str,
    ocr_text: str,
    ocr_status: str,
    ocr_error: str,
    image_hash: str,
    text_norm: str,
    text_hash: str,
    doc_type: str,
    fields: dict,
    person_key: str,
    person_key_type: str,
):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ocr_texts (
            chat_id, user_id, message_id, file_unique_id, created_at,
            ocr_text, ocr_status, ocr_error,
            image_hash, text_norm, text_hash,
            doc_type, fields_json,
            person_key, person_key_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        chat_id, user_id, message_id, file_unique_id, created_at_iso,
        ocr_text, ocr_status, ocr_error,
        image_hash, text_norm, text_hash,
        doc_type, json.dumps(fields, ensure_ascii=False),
        person_key, person_key_type
    ))
    conn.commit()
    conn.close()

# =========================
# Diff logic per profile
# =========================
def diff_fields_by_profile(doc_type: str, prev_fields: dict, now_fields: dict):
    prof = DOC_PROFILES.get(doc_type, DOC_PROFILES["UNKNOWN"])
    keys = prof.get("diff_fields", [])

    diffs = []
    for k in keys:
        a = (prev_fields.get(k) or "").strip()
        b = (now_fields.get(k) or "").strip()
        if a and b and a != b:
            diffs.append((k, a, b))
        elif a == "" and b != "":
            diffs.append((k, "(vacío)", b))
        elif a != "" and b == "":
            diffs.append((k, a, "(vacío)"))
    return diffs

def pretty_label(k: str) -> str:
    m = {
        "domicilio": "🏠 Domicilio",
        "vigencia": "📅 Vigencia",
        "seccion": "🧩 Sección",
        "sexo": "👤 Sexo",
        "ano_registro": "🗓️ Año de registro",
        "passport_no": "🛂 Pasaporte No.",
        "expiry": "📅 Expira",
        "nationality": "🌎 Nacionalidad",
        "license_no": "🪪 Licencia No.",
        "address": "🏠 Dirección",
        "dob": "🎂 Nacimiento",
        "name": "🧑 Nombre",
        "clave_elector": "🧾 Clave elector",
        "curp": "🧬 CURP",
    }
    return m.get(k, k)

# =========================
# Telegram handlers
# =========================
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Mándame una foto.")

def photo_received(update, context):
    try:
        msg = update.message
        if not msg or not msg.photo:
            return

        chat_id = msg.chat_id
        user_id = msg.from_user.id
        message_id = msg.message_id

        photo = msg.photo[-1]
        file_unique_id = photo.file_unique_id

        log.info("PHOTO_RECEIVED chat=%s msg=%s user=%s file_unique_id=%s", chat_id, message_id, user_id, file_unique_id)

        tg_file = context.bot.get_file(photo.file_id)
        raw_bytes = tg_file.download_as_bytearray()
        raw_bytes = bytes(raw_bytes)  # <- importante

        # JPEG + pHash
        try:
            jpeg_bytes = to_real_jpeg_bytes(raw_bytes)
            img_hash = compute_image_phash_hex(jpeg_bytes)
        except Exception as e:
            jpeg_bytes = raw_bytes
            img_hash = ""
            log.warning("JPEG conversion/hash failed: %s", e)

        # OCR
        ocr_text = ""
        ocr_status = "ok"
        ocr_error = ""
        try:
            ocr_text = ocr_space_bytes(jpeg_bytes)
            if not ocr_text.strip():
                ocr_status = "empty"
        except Exception as e:
            ocr_status = "error"
            ocr_error = str(e)
            log.warning("OCR error: %s", ocr_error)

        # Normalize + hashes
        text_norm = normalize_text_for_hash(ocr_text)
        text_hash = sha256_hex(text_norm) if text_norm else ""

        # Doc type + fields
        doc_type = detect_doc_type(text_norm)
        fields = extract_by_profile(text_norm, doc_type)

        # Ensure INE pulls curp/clave/name better
        if doc_type == "INE_MX":
            if "name" not in fields:
                fields["name"] = extract_name_block_ine(text_norm)
            if "curp" not in fields:
                m = CURP_RE.search(text_norm)
                if m:
                    fields["curp"] = m.group(0)
            if "clave_elector" not in fields:
                if "CLAVE DE ELECTOR" in text_norm:
                    idx = text_norm.find("CLAVE DE ELECTOR")
                    window = text_norm[idx: idx + 350]
                    m2 = CLAVE_ELECTOR_RE.search(window)
                    if m2:
                        fields["clave_elector"] = m2.group(0)
                if "clave_elector" not in fields:
                    m3 = CLAVE_ELECTOR_RE.search(text_norm)
                    if m3:
                        fields["clave_elector"] = m3.group(0)
            if "dob" not in fields:
                m4 = DOB_RE.search(text_norm)
                if m4:
                    fields["dob"] = m4.group(0)

        # person key (ID fuerte por perfil)
        person_key, person_key_type = build_person_key(doc_type, fields)

        created_at_iso = datetime.now(timezone.utc).isoformat()

        conn = db_conn()
        cur = conn.cursor()

        exact = find_exact_duplicate(cur, chat_id, text_hash, img_hash)

        first_person = None
        latest_person = None
        if person_key:
            first_person = find_first_by_person(cur, chat_id, person_key)
            latest_person = find_latest_by_person(cur, chat_id, person_key)

        name_now = (fields.get("name") or "").strip()
        suggestions = []
        if not first_person:
            try:
                suggestions = find_fuzzy_suggestions(cur, chat_id, name_now, img_hash)
            except Exception as e:
                log.warning("Fuzzy suggestion error: %s", e)

        conn.close()

        # If exact duplicate -> reply and do not insert
        if exact:
            kind, row = exact
            # row = (id, created_at, message_id, text_hash, image_hash)
            first_seen_iso = row[1]
            when = format_cdmx(first_seen_iso)
            update.message.reply_text(
                f"✅ Ya estaba registrada.\n"
                f"🕒 Primera vez: {when}\n"
                f"🔎 Coincidencia: {'TEXTO' if kind=='TEXT' else 'IMAGEN'}"
            )
            return

        # If same person -> compute changes vs latest record
        person_note = ""
        changes_note = ""

        if first_person and latest_person:
            _id1, first_seen_iso, _mid1, _doc_first, pk_type = first_person
            _idl, _last_seen_iso, _midl, _doc_last, prev_fields_json = latest_person

            person_note = f"\n🟡 Misma persona detectada ({pk_type}).\n🕒 Primera vez: {format_cdmx(first_seen_iso)}"

            try:
                prev_fields = json.loads(prev_fields_json) if prev_fields_json else {}
            except Exception:
                prev_fields = {}

            diffs = diff_fields_by_profile(doc_type, prev_fields, fields)

            if diffs:
                pretty = []
                for k, a, b in diffs:
                    pretty.append(f"{pretty_label(k)}\nANTES:\n{a}\nAHORA:\n{b}")
                changes_note = "\n\n🧾 Cambios detectados:\n\n" + "\n\n".join(pretty)
            else:
                changes_note = "\n\n🧾 Sin cambios detectables (con lo que leyó el OCR)."

        fuzzy_note = ""
        if suggestions:
            lines = []
            for (combined, img_s, name_s, rid, created_at, mid, dtyp, name_prev) in suggestions:
                when = format_cdmx(created_at)
                lines.append(f"• {int(combined*100)}% (img {int(img_s*100)}%, nombre {int(name_s*100)}%) — {when} — {dtyp}")
            fuzzy_note = "\n\n🟠 Sugerencias (no seguro):\n" + "\n".join(lines)

        insert_record(
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            file_unique_id=file_unique_id,
            created_at_iso=created_at_iso,
            ocr_text=ocr_text,
            ocr_status=ocr_status,
            ocr_error=ocr_error,
            image_hash=img_hash,
            text_norm=text_norm,
            text_hash=text_hash,
            doc_type=doc_type,
            fields=fields,
            person_key=person_key,
            person_key_type=person_key_type,
        )

        if ocr_status == "ok" and ocr_text.strip():
            shown = ocr_text.strip()
            if len(shown) > 2500:
                shown = shown[:2500] + "\n…(recortado)"
            update.message.reply_text(
                f"📄 Texto detectado y guardado ({doc_type}).\n\n{shown}"
                + person_note + changes_note + fuzzy_note
            )
        elif ocr_status == "empty":
            update.message.reply_text(
                f"📸 Foto guardada (sin texto legible) ({doc_type})."
                + person_note + changes_note + fuzzy_note
            )
        else:
            update.message.reply_text(
                f"📸 Foto guardada, pero el OCR falló ({doc_type})."
                + person_note + changes_note + fuzzy_note
            )

    except Exception:
        log.exception("Error en photo_received")
        try:
            update.message.reply_text("⚠️ Se cayó algo procesando tu foto. Revisa logs en Render.")
        except Exception:
            pass

def error_handler(update, context):
    log.exception("Unhandled exception:", exc_info=context.error)
    try:
        if update and getattr(update, "message", None):
            update.message.reply_text("⚠️ Error interno. Ya quedó en logs.")
    except Exception:
        pass

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN")

    init_db()

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, photo_received))
    dp.add_error_handler(error_handler)

    log.info("Bot arrancando polling...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

