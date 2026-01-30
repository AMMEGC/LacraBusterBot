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
# Document profiles
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS person_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            person_key TEXT,
            status_code INTEGER,
            note TEXT,
            tagged_at TEXT,
            tagged_by_user_id INTEGER
        )
    """)

    try:
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_person_tags_unique ON person_tags(chat_id, person_key)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_person_tags_status ON person_tags(chat_id, status_code)")
    except Exception as e:
        log.warning("Index creation warning (person_tags): %s", e)

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

def upsert_person_tag(cur, chat_id: int, person_key: str, status_code: int, note: str, tagged_by_user_id: int):
    now_iso = datetime.now(timezone.utc).isoformat()
    cur.execute("""
        INSERT INTO person_tags (chat_id, person_key, status_code, note, tagged_at, tagged_by_user_id)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(chat_id, person_key)
        DO UPDATE SET
            status_code=excluded.status_code,
            note=excluded.note,
            tagged_at=excluded.tagged_at,
            tagged_by_user_id=excluded.tagged_by_user_id
    """, (chat_id, person_key, int(status_code), (note or "").strip(), now_iso, int(tagged_by_user_id)))


def get_person_tag(cur, chat_id: int, person_key: str):
    if not person_key:
        return None
    cur.execute("""
        SELECT status_code, note, tagged_at, tagged_by_user_id
        FROM person_tags
        WHERE chat_id=? AND person_key=?
        LIMIT 1
    """, (chat_id, person_key))
    return cur.fetchone()


def delete_person_tag(cur, chat_id: int, person_key: str):
    cur.execute("DELETE FROM person_tags WHERE chat_id=? AND person_key=?", (chat_id, person_key))

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

def safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}

def get_last_records(cur, chat_id: int, limit: int = 10):
    cur.execute("""
        SELECT id, created_at, doc_type, fields_json, ocr_status
        FROM ocr_texts
        WHERE chat_id=?
        ORDER BY id DESC
        LIMIT ?
    """, (chat_id, limit))
    return cur.fetchall()

def get_record_by_id(cur, chat_id: int, rid: int):
    cur.execute("""
        SELECT id, created_at, message_id, doc_type, ocr_status, ocr_error,
               image_hash, text_hash, fields_json, person_key, person_key_type
        FROM ocr_texts
        WHERE chat_id=? AND id=?
        LIMIT 1
    """, (chat_id, rid))
    return cur.fetchone()

def get_person_records(cur, chat_id: int, person_key: str, limit: int = 30):
    cur.execute("""
        SELECT id, created_at, message_id, doc_type, fields_json
        FROM ocr_texts
        WHERE chat_id=? AND person_key=?
        ORDER BY id DESC
        LIMIT ?
    """, (chat_id, person_key, limit))
    return cur.fetchall()

def find_person_key_by_identifier(cur, chat_id: int, token: str):
    """
    token puede ser:
      - person_key directo (hash)
      - CURP / CLAVE_ELECTOR / PASAPORTE etc. (porque tú guardas person_key = ese valor cuando existe)
    """
    t = (token or "").strip().upper()
    if not t:
        return None

    # Primero: exact match en person_key (cuando person_key es CURP/CLAVE/PASAPORTE)
    cur.execute("""
        SELECT person_key, person_key_type, doc_type, created_at
        FROM ocr_texts
        WHERE chat_id=? AND person_key=?
        ORDER BY id DESC
        LIMIT 1
    """, (chat_id, t))
    row = cur.fetchone()
    if row:
        return row  # (person_key, person_key_type, doc_type, created_at)

    # Segundo: buscar dentro de fields_json (por si luego cambias lógica)
    # Nota: LIKE es "barato" pero sirve. Para serio, luego metemos columnas indexadas.
    cur.execute("""
        SELECT person_key, person_key_type, doc_type, created_at, fields_json
        FROM ocr_texts
        WHERE chat_id=? AND fields_json LIKE ?
        ORDER BY id DESC
        LIMIT 1
    """, (chat_id, f'%{t}%'))
    row2 = cur.fetchone()
    if row2 and row2[0]:
        return (row2[0], row2[1], row2[2], row2[3])

    return None

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
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid


# =========================
# Diff logic
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

def diff_fields_generic(prev_fields: dict, now_fields: dict):
    # Compara todo lo que exista en cualquiera de los dos
    keys = sorted(set(prev_fields.keys()) | set(now_fields.keys()))
    diffs = []
    for k in keys:
        if k in ("name",):
            continue
        a = (prev_fields.get(k) or "").strip()
        b = (now_fields.get(k) or "").strip()
        if a == b:
            continue
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
        raw_bytes = bytes(raw_bytes)

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
        tag_alert = ""
        if person_key:
            conn = db_conn()
            cur = conn.cursor()
            tagrow = get_person_tag(cur, chat_id, person_key)
            conn.close()

            if tagrow:
                status_code, note, tagged_at, tagged_by = tagrow
                if int(status_code) == 110:
                        tag_alert = "🚨 ALERTA: ESTA PERSONA ESTÁ MARCADA COMO 110\n" + (f"📝 {note}\n" if note else "")

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
            if not diffs:
                # Si el doc_type no tiene diff_fields (ej: UNKNOWN), intenta comparación genérica
                diffs = diff_fields_generic(prev_fields, fields)

            if diffs:
                pretty = []
                for k, a, b in diffs:
                    pretty.append(f"{pretty_label(k)}\nANTES:\n{a}\nAHORA:\n{b}")
                changes_note = "\n\n🧾 Cambios detectados:\n\n" + "\n\n".join(pretty)
            else:
                changes_note = "\n\n🧾 Sin cambios detectables (con lo que leyó el OCR)."

        # Suggestions note (más legible)
        fuzzy_note = ""
        if suggestions:
            lines = []
            for (combined, img_s, name_s, rid, created_at, mid, dtyp, name_prev) in suggestions:
                when = format_cdmx(created_at)

                parts = []
                if img_s > 0:
                    parts.append(f"📸 foto parecida {int(img_s*100)}%")
                else:
                    parts.append("📸 sin comparación de foto")

                if name_s > 0:
                    parts.append(f"🧑 nombre parecido {int(name_s*100)}%")
                else:
                    parts.append("🧑 nombre no detectado / no comparable")

                lines.append(f"• Posible match — {', '.join(parts)}\n  🕒 {when} — {dtyp}")

            fuzzy_note = "\n\n🟠 Sugerencias (no confirmadas):\n" + "\n".join(lines)

        # Insert record
        rid = insert_record(
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

        # Reply
        if ocr_status == "ok" and ocr_text.strip():
            shown = ocr_text.strip()
            if len(shown) > 2500:
                shown = shown[:2500] + "\n…(recortado)"
            update.message.reply_text(
                tag_alert + f"🆔 Registro #{rid}\n📄 Texto detectado y guardado ({doc_type}).\n\n{shown}"
                + person_note + changes_note + fuzzy_note
            )
        elif ocr_status == "empty":
            update.message.reply_text(
                tag_alert + f"🆔 Registro #{rid}\n📸 Foto guardada (sin texto legible) ({doc_type})."
                + person_note + changes_note + fuzzy_note
            )
        else:
            update.message.reply_text(
                tag_alert + f"🆔 Registro #{rid}\n📸 Foto guardada, pero el OCR falló ({doc_type})."
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


def historial(update, context):
    msg = update.message
    chat_id = msg.chat_id

    n = 10
    try:
        if context.args and context.args[0].isdigit():
            n = max(1, min(30, int(context.args[0])))
    except Exception:
        pass

    conn = db_conn()
    cur = conn.cursor()
    rows = get_last_records(cur, chat_id, limit=n)
    conn.close()

    if not rows:
        msg.reply_text("📭 No hay registros todavía en este chat.")
        return

    lines = [f"📚 Últimos {len(rows)} registros:"]
    for (rid, created_at, doc_type, fields_json, ocr_status) in rows:
        when = format_cdmx(created_at)
        f = safe_json_loads(fields_json)
        name = (f.get("name") or "").strip()
        name_part = f" — 🧑 {name}" if name else ""
        lines.append(f"• #{rid} — {when} — {doc_type} — OCR:{ocr_status}{name_part}")

    msg.reply_text("\n".join(lines))


def ver(update, context):
    msg = update.message
    chat_id = msg.chat_id

    if not context.args or not context.args[0].isdigit():
        msg.reply_text("Uso: /ver <id>\nEj: /ver 12")
        return

    rid = int(context.args[0])

    conn = db_conn()
    cur = conn.cursor()
    row = get_record_by_id(cur, chat_id, rid)
    conn.close()

    if not row:
        msg.reply_text("No encontré ese ID en este chat.")
        return

    (rid, created_at, message_id, doc_type, ocr_status, ocr_error,
     image_hash, text_hash, fields_json, person_key, person_key_type) = row

    f = safe_json_loads(fields_json)
    name = (f.get("name") or "").strip()

    parts = [
        f"🧾 Registro #{rid}",
        f"🕒 {format_cdmx(created_at)}",
        f"📄 Tipo: {doc_type}",
        f"👤 Nombre: {name if name else '(no detectado)'}",
        f"🔑 person_key: {person_key if person_key else '(vacío)'} ({person_key_type if person_key_type else '-'})",
        f"🧠 OCR: {ocr_status}",
        f"🧬 text_hash: {text_hash if text_hash else '(vacío)'}",
        f"📸 image_hash: {image_hash if image_hash else '(vacío)'}",
    ]

    if ocr_error:
        parts.append(f"⚠️ OCR error: {ocr_error}")

    if f:
        pretty = []
        for k, v in f.items():
            if not v:
                continue
            pretty.append(f"{pretty_label(k)}: {v}")
        if pretty:
            parts.append("\n📌 Campos:\n" + "\n".join(pretty))

    msg.reply_text("\n".join(parts))


def persona(update, context):
    msg = update.message
    chat_id = msg.chat_id

    if not context.args:
        msg.reply_text("Uso: /persona <identificador>\nEj: /persona GODE900101HDFXXX00 (CURP)")
        return

    token = " ".join(context.args).strip().upper()

    conn = db_conn()
    cur = conn.cursor()

    pk_row = find_person_key_by_identifier(cur, chat_id, token)
    if not pk_row:
        conn.close()
        msg.reply_text("No encontré a esa persona / identificador en este chat.")
        return

    person_key, person_key_type, doc_type, created_at = pk_row

    rows = get_person_records(cur, chat_id, person_key, limit=30)
    conn.close()

    if not rows:
        msg.reply_text("Encontré la llave de persona, pero no hay registros (raro).")
        return

    last = rows[0]
    first = rows[-1]

    def row_name(fields_json: str) -> str:
        f = safe_json_loads(fields_json)
        return (f.get("name") or "").strip()

    lines = []
    lines.append("🟡 Persona encontrada")
    lines.append(f"🔑 person_key: {person_key} ({person_key_type})")
    lines.append(f"📄 Tipo: {doc_type}")
    lines.append(f"🕒 Primer registro: #{first[0]} — {format_cdmx(first[1])} — {first[3]} — 🧑 {row_name(first[4]) or '(sin nombre)'}")
    lines.append(f"🕒 Último registro:  #{last[0]} — {format_cdmx(last[1])} — {last[3]} — 🧑 {row_name(last[4]) or '(sin nombre)'}")
    lines.append("")
    lines.append("📚 Registros recientes:")
    for (rid, created_at, mid, dtyp, fields_json) in rows[:10]:
        nm = row_name(fields_json)
        nm_part = f" — 🧑 {nm}" if nm else ""
        lines.append(f"• #{rid} — {format_cdmx(created_at)} — {dtyp}{nm_part}")

    msg.reply_text("\n".join(lines))

def tag(update, context):
    msg = update.message
    chat_id = msg.chat_id
    user_id = msg.from_user.id

    if len(context.args) < 2 or not context.args[0].isdigit():
        msg.reply_text("Uso: /tag <id_registro> <codigo> [nota]\nEj: /tag 12 110 ratero confirmado")
        return

    rid = int(context.args[0])
    code = int(context.args[1])
    note = " ".join(context.args[2:]).strip() if len(context.args) > 2 else ""

    conn = db_conn()
    cur = conn.cursor()

    row = get_record_by_id(cur, chat_id, rid)
    if not row:
        conn.close()
        msg.reply_text("No encontré ese ID en este chat.")
        return

    person_key = row[-2]  # tu get_record_by_id deja person_key como penúltimo
    if not person_key:
        conn.close()
        msg.reply_text("Ese registro no tiene person_key (no pude identificar persona).")
        return

    upsert_person_tag(cur, chat_id, person_key, code, note, user_id)
    conn.commit()
    conn.close()

    msg.reply_text(f"✅ Marcado como {code}.\n🔑 person_key: {person_key}" + (f"\n📝 Nota: {note}" if note else ""))


def untag(update, context):
    msg = update.message
    chat_id = msg.chat_id

    if not context.args or not context.args[0].isdigit():
        msg.reply_text("Uso: /untag <id_registro>\nEj: /untag 12")
        return

    rid = int(context.args[0])

    conn = db_conn()
    cur = conn.cursor()

    row = get_record_by_id(cur, chat_id, rid)
    if not row:
        conn.close()
        msg.reply_text("No encontré ese ID en este chat.")
        return

    person_key = row[-2]
    if not person_key:
        conn.close()
        msg.reply_text("Ese registro no tiene person_key.")
        return

    delete_person_tag(cur, chat_id, person_key)
    conn.commit()
    conn.close()

    msg.reply_text(f"🧽 Marca eliminada.\n🔑 person_key: {person_key}")

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN")

    init_db()

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("historial", historial))
    dp.add_handler(CommandHandler("ver", ver))
    dp.add_handler(CommandHandler("persona", persona))
    dp.add_handler(CommandHandler("tag", tag))
    dp.add_handler(CommandHandler("untag", untag))

    dp.add_handler(MessageHandler(Filters.photo, photo_received))
    dp.add_error_handler(error_handler)

    log.info("Bot arrancando polling...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()


