import os
import re
import io
import json
import time
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
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
# Helpers: time formatting
# =========================
_ES_MONTHS = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

def format_cdmx(dt_iso: str) -> str:
    """dt_iso: ISO string in UTC (or with tz). Returns pretty CDMX string."""
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
# Image: ensure real JPEG + hashes
# =========================
def to_real_jpeg_bytes(image_bytes: bytes) -> bytes:
    """
    Telegram sometimes gives weird bytes; convert to real JPEG.
    """
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=92, optimize=True)
        return out.getvalue()


def compute_image_phash_hex(jpeg_bytes: bytes) -> str:
    with Image.open(io.BytesIO(jpeg_bytes)) as im:
        im = im.convert("RGB")
        h = imagehash.phash(im)
        return str(h)  # hex string


def phash_distance(hex_a: str, hex_b: str) -> int:
    try:
        return imagehash.hex_to_hash(hex_a) - imagehash.hex_to_hash(hex_b)
    except Exception:
        return 999


# =========================
# Text normalization + identity extraction
# =========================
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def normalize_text_for_hash(s: str) -> str:
    """
    Make OCR text stable for hashing:
    - uppercase
    - remove accents
    - keep letters/numbers/basic punctuation as spaces
    - collapse whitespace
    """
    s = (s or "").upper()
    s = strip_accents(s)
    s = re.sub(r"[^A-Z0-9/\n ]+", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    return s

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

CURP_RE = re.compile(r"\b[A-Z]{4}\d{6}[A-Z]{6}\d{2}\b")
# CLAVE ELECTOR suele ser 18 alfanum (a veces 18 exacto, a veces 18+)
CLAVE_ELECTOR_RE = re.compile(r"\b[A-Z0-9]{18}\b")
DOB_RE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")

def extract_name_block(text_norm: str) -> str:
    """
    Extract likely name lines after 'NOMBRE'.
    """
    lines = [ln.strip() for ln in text_norm.splitlines() if ln.strip()]
    name_lines = []
    grab = False
    for ln in lines:
        if "NOMBRE" == ln or ln.startswith("NOMBRE "):
            grab = True
            continue
        if grab:
            # stop if hits another known field label
            if any(k in ln for k in [
                "DOMICILIO", "CURP", "CLAVE DE ELECTOR", "FECHA DE NACIMIENTO",
                "SECCION", "VIGENCIA", "ANO DE REGISTRO", "SEXO"
            ]):
                break
            # take up to 4 lines (AP + AM + NOMBRES suele ser 3)
            if re.fullmatch(r"[A-Z ]{2,}", ln):
                name_lines.append(ln)
            if len(name_lines) >= 4:
                break
    # Join
    return " ".join(name_lines).strip()

def extract_identity(text_norm: str) -> dict:
    curp = None
    clave = None
    dob = None

    m = CURP_RE.search(text_norm)
    if m:
        curp = m.group(0)

    # If there are many 18-char tokens, prefer the one near 'CLAVE DE ELECTOR'
    if "CLAVE DE ELECTOR" in text_norm:
        idx = text_norm.find("CLAVE DE ELECTOR")
        window = text_norm[idx: idx + 250]
        m2 = CLAVE_ELECTOR_RE.search(window)
        if m2:
            clave = m2.group(0)

    if not clave:
        m3 = CLAVE_ELECTOR_RE.search(text_norm)
        if m3:
            clave = m3.group(0)

    m4 = DOB_RE.search(text_norm)
    if m4:
        dob = m4.group(0)

    name = extract_name_block(text_norm)

    # Person key priority: CURP > CLAVE ELECTOR > name+dob
    person_key = None
    person_key_type = None
    if curp:
        person_key = curp
        person_key_type = "CURP"
    elif clave:
        person_key = clave
        person_key_type = "CLAVE_ELECTOR"
    elif name and dob:
        person_key = sha256_hex(f"{name}|{dob}")[:24]
        person_key_type = "NAME_DOB"

    return {
        "curp": curp,
        "clave_elector": clave,
        "dob": dob,
        "name": name,
        "person_key": person_key,
        "person_key_type": person_key_type,
    }

def similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


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
# DB: schema + safe migrations
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

    # Main table
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

    # Add new columns safely
    ensure_column(cur, "ocr_texts", "ocr_status", "ocr_status TEXT")
    ensure_column(cur, "ocr_texts", "ocr_error", "ocr_error TEXT")
    ensure_column(cur, "ocr_texts", "image_hash", "image_hash TEXT")
    ensure_column(cur, "ocr_texts", "text_norm", "text_norm TEXT")
    ensure_column(cur, "ocr_texts", "text_hash", "text_hash TEXT")
    ensure_column(cur, "ocr_texts", "person_key", "person_key TEXT")
    ensure_column(cur, "ocr_texts", "person_key_type", "person_key_type TEXT")
    ensure_column(cur, "ocr_texts", "name_extracted", "name_extracted TEXT")
    ensure_column(cur, "ocr_texts", "curp", "curp TEXT")
    ensure_column(cur, "ocr_texts", "clave_elector", "clave_elector TEXT")
    ensure_column(cur, "ocr_texts", "dob", "dob TEXT")

    # Indexes (only if columns exist)
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


def find_exact_duplicate(cur, chat_id: int, text_hash: str, image_hash: str):
    """
    Exact duplicate if same text_hash OR same image_hash.
    Returns row or None.
    """
    if text_hash:
        cur.execute("""
            SELECT id, created_at, message_id FROM ocr_texts
            WHERE chat_id=? AND text_hash=?
            ORDER BY id ASC LIMIT 1
        """, (chat_id, text_hash))
        row = cur.fetchone()
        if row:
            return ("TEXT", row)

    if image_hash:
        cur.execute("""
            SELECT id, created_at, message_id FROM ocr_texts
            WHERE chat_id=? AND image_hash=?
            ORDER BY id ASC LIMIT 1
        """, (chat_id, image_hash))
        row = cur.fetchone()
        if row:
            return ("IMAGE", row)

    return None


def find_same_person(cur, chat_id: int, person_key: str):
    if not person_key:
        return None
    cur.execute("""
        SELECT id, created_at, message_id, name_extracted, person_key_type
        FROM ocr_texts
        WHERE chat_id=? AND person_key=?
        ORDER BY id ASC LIMIT 1
    """, (chat_id, person_key))
    return cur.fetchone()


def find_fuzzy_suggestions(cur, chat_id: int, name_now: str, image_hash_now: str, limit: int = 120):
    """
    Suggest possible matches based on:
    - close image phash distance
    - name similarity
    Returns top 3 suggestions.
    """
    cur.execute("""
        SELECT id, created_at, message_id, image_hash, name_extracted, curp, clave_elector
        FROM ocr_texts
        WHERE chat_id=?
        ORDER BY id DESC
        LIMIT ?
    """, (chat_id, limit))
    rows = cur.fetchall()

    scored = []
    for (rid, created_at, msg_id, img_hash, name_prev, curp, clave) in rows:
        img_score = 0.0
        name_score = 0.0

        if image_hash_now and img_hash:
            d = phash_distance(image_hash_now, img_hash)
            # distance 0..64-ish. Convert to 0..1 (rough)
            img_score = max(0.0, 1.0 - (d / 14.0))  # <=14 still somewhat close

        if name_now and name_prev:
            name_score = similarity_ratio(name_now, name_prev)

        # Weighted combined
        combined = (0.6 * img_score) + (0.4 * name_score)

        # keep if looks somewhat meaningful
        if combined >= 0.72 or (img_score >= 0.82) or (name_score >= 0.88):
            scored.append((combined, img_score, name_score, rid, created_at, msg_id, name_prev, curp, clave))

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
    ident: dict,
):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ocr_texts (
            chat_id, user_id, message_id, file_unique_id, created_at,
            ocr_text, ocr_status, ocr_error,
            image_hash, text_norm, text_hash,
            person_key, person_key_type,
            name_extracted, curp, clave_elector, dob
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        chat_id, user_id, message_id, file_unique_id, created_at_iso,
        ocr_text, ocr_status, ocr_error,
        image_hash, text_norm, text_hash,
        ident.get("person_key"), ident.get("person_key_type"),
        ident.get("name"), ident.get("curp"), ident.get("clave_elector"), ident.get("dob")
    ))
    conn.commit()
    conn.close()


# =========================
# Telegram handlers
# =========================
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Mándame una foto.")


def photo_received(update, context):
    msg = update.message
    if not msg or not msg.photo:
        return

    chat_id = msg.chat_id
    user_id = msg.from_user.id
    message_id = msg.message_id

    photo = msg.photo[-1]
    file_unique_id = photo.file_unique_id

    log.info("PHOTO_RECEIVED chat=%s msg=%s user=%s file_unique_id=%s",
             chat_id, message_id, user_id, file_unique_id)

    # Download bytes
    tg_file = context.bot.get_file(photo.file_id)
    raw_bytes = tg_file.download_as_bytearray()

    # Convert to real JPEG + image hash
    try:
        jpeg_bytes = to_real_jpeg_bytes(raw_bytes)
        img_hash = compute_image_phash_hex(jpeg_bytes)
    except Exception as e:
        jpeg_bytes = bytes(raw_bytes)
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

    # Normalize + hashes + identity
    text_norm = normalize_text_for_hash(ocr_text)
    text_hash = sha256_hex(text_norm) if text_norm else ""
    ident = extract_identity(text_norm)

    created_at_iso = datetime.now(timezone.utc).isoformat()

    # DB checks
    conn = db_conn()
    cur = conn.cursor()

    exact = find_exact_duplicate(cur, chat_id, text_hash, img_hash)
    same_person = None
    if ident.get("person_key"):
        same_person = find_same_person(cur, chat_id, ident["person_key"])

    # Close conn early, we will insert after decisions
    conn.close()

    # 1) Exact duplicate => no insert, reply with first seen
    if exact:
        kind, row = exact
        _, first_seen_iso, first_msg_id = row
        when = format_cdmx(first_seen_iso)
        update.message.reply_text(
            f"✅ Esa foto ya estaba registrada.\n"
            f"🕒 Primera vez: {when}\n"
            f"🔎 Coincidencia: {'TEXTO' if kind=='TEXT' else 'IMAGEN'}"
        )
        log.info("DUPLICATE exact (%s) ignored. first_seen=%s", kind, first_seen_iso)
        return

    # 2) Not exact, but same person key => insert + reply as “relacionado”
    # (CURP/Clave/NOMBRE+FECHA)
    person_note = ""
    if same_person:
        _, first_seen_iso, first_msg_id, prev_name, pk_type = same_person
        when = format_cdmx(first_seen_iso)
        person_note = (
            f"\n🟡 Posible MISMA PERSONA ({pk_type}).\n"
            f"🕒 Primera vez: {when}"
        )

    # 3) Fuzzy suggestions (optional) – only if no strong person key match
    fuzzy_note = ""
    if not same_person:
        try:
            conn = db_conn()
            cur = conn.cursor()
            suggestions = find_fuzzy_suggestions(cur, chat_id, ident.get("name") or "", img_hash)
            conn.close()
        except Exception as e:
            suggestions = []
            log.warning("Fuzzy suggestion error: %s", e)

        if suggestions:
            lines = []
            for (combined, img_s, name_s, rid, created_at, mid, name_prev, curp, clave) in suggestions:
                when = format_cdmx(created_at)
                lines.append(
                    f"• {int(combined*100)}% (img {int(img_s*100)}%, nombre {int(name_s*100)}%) — {when}"
                )
            fuzzy_note = "\n🟠 Sugerencias (no es seguro):\n" + "\n".join(lines)

    # Insert new record
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
        ident=ident,
    )

    # Reply: show OCR text if any
    if ocr_status == "ok" and ocr_text.strip():
        # limit message length (Telegram)
        max_len = 3000
        shown = ocr_text.strip()
        if len(shown) > max_len:
            shown = shown[:max_len] + "\n…(recortado)"
        update.message.reply_text(
            "📄 Texto detectado y guardado:\n\n" + shown + person_note + fuzzy_note
        )
    elif ocr_status == "empty":
        update.message.reply_text("📸 Foto guardada (sin texto legible)." + person_note + fuzzy_note)
    else:
        update.message.reply_text("📸 Foto guardada, pero el OCR falló." + person_note + fuzzy_note)


def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN")

    init_db()

    # Render dummy server
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, photo_received))

    log.info("Bot arrancando polling...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
