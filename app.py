import os
import threading
import sqlite3
import logging
import hashlib
from io import BytesIO
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Pillow
from PIL import Image

# Zoneinfo (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# (Opcional) perceptual hash para detectar duplicados "parecidos"
try:
    import imagehash
except Exception:
    imagehash = None


# ============ CONFIG ============
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")
PORT = int(os.environ.get("PORT", 10000))

# Logging bonito en Render
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============ RENDER DUMMY SERVER ============
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run_server():
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


# ============ HELPERS ============
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def to_cdmx_pretty(iso_utc: str) -> str:
    """
    Recibe ISO UTC (ej: 2026-01-26T17:05:29.46+00:00) y regresa string bonito CDMX.
    """
    dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    if ZoneInfo:
        dt_local = dt.astimezone(ZoneInfo("America/Mexico_City"))
        return dt_local.strftime("%d/%m/%Y %I:%M %p") + " (CDMX)"
    # fallback UTC
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%d/%m/%Y %I:%M %p") + " (UTC)"

def normalize_to_jpeg(image_bytes: bytes) -> bytes:
    """
    Convierte cualquier cosa que mande Telegram a JPEG real.
    Esto evita errores tipo 'Input file corrupted?' en OCR.space.
    """
    with Image.open(BytesIO(image_bytes)) as im:
        # RGB para JPG
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        elif im.mode == "L":
            # gris -> RGB (opcional) pero ayuda a consistencia
            im = im.convert("RGB")

        out = BytesIO()
        im.save(out, format="JPEG", quality=92, optimize=True)
        return out.getvalue()

def compute_image_hash(jpeg_bytes: bytes) -> str:
    """
    Si ImageHash está disponible: perceptual hash (pHash) -> detecta duplicados similares.
    Si no: SHA256 del JPG -> detecta duplicados exactos.
    """
    if imagehash is not None:
        with Image.open(BytesIO(jpeg_bytes)) as im:
            ph = imagehash.phash(im)  # perceptual hash
            return str(ph)
    # fallback exacto
    return hashlib.sha256(jpeg_bytes).hexdigest()

def safe_preview(text: str, limit: int = 3500) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...\n(Recortado)"


# ============ DB ============
def get_conn():
    os.makedirs(os.path.dirname(DBPATH), exist_ok=True)
    return sqlite3.connect(DBPATH)

def ensure_column(cur, table: str, col: str, coltype: str):
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]  # name at index 1
    if col not in cols:
        logging.info(f"DB migration: agregando columna {table}.{col}")
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Tabla principal
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ocr_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            user_id INTEGER,
            message_id INTEGER,
            file_unique_id TEXT,
            created_at TEXT,
            image_hash TEXT,
            ocr_status TEXT,
            ocr_error TEXT,
            ocr_text TEXT
        )
    """)

    # Si vienes de una versión vieja: agrega columnas faltantes sin romper
    ensure_column(cur, "ocr_texts", "chat_id", "INTEGER")
    ensure_column(cur, "ocr_texts", "user_id", "INTEGER")
    ensure_column(cur, "ocr_texts", "message_id", "INTEGER")
    ensure_column(cur, "ocr_texts", "file_unique_id", "TEXT")
    ensure_column(cur, "ocr_texts", "created_at", "TEXT")
    ensure_column(cur, "ocr_texts", "image_hash", "TEXT")
    ensure_column(cur, "ocr_texts", "ocr_status", "TEXT")
    ensure_column(cur, "ocr_texts", "ocr_error", "TEXT")
    ensure_column(cur, "ocr_texts", "ocr_text", "TEXT")

    # Índices
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_hash ON ocr_texts(image_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_file_unique ON ocr_texts(file_unique_id)")

    conn.commit()
    conn.close()
    logging.info(f"DB inicializada en: {DBPATH}")

def find_existing_by_hash(image_hash: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT created_at, ocr_text
        FROM ocr_texts
        WHERE image_hash = ?
        ORDER BY id ASC
        LIMIT 1
    """, (image_hash,))
    row = cur.fetchone()

    cur.execute("SELECT COUNT(*) FROM ocr_texts WHERE image_hash = ?", (image_hash,))
    count = cur.fetchone()[0] or 0

    conn.close()
    return row, count

def insert_record(chat_id, user_id, message_id, file_unique_id, image_hash, status, ocr_error, ocr_text):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ocr_texts (chat_id, user_id, message_id, file_unique_id, created_at, image_hash, ocr_status, ocr_error, ocr_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        chat_id,
        user_id,
        message_id,
        file_unique_id,
        utcnow_iso(),
        image_hash,
        status,
        ocr_error,
        ocr_text
    ))
    conn.commit()
    conn.close()


# ============ OCR (OCR.space) ============
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
        "filetype": "JPG"
    }

    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    payload = r.json()

    logging.info(f"OCR RAW RESPONSE >>> {payload}")

    if payload.get("IsErroredOnProcessing"):
        msg = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Error OCR desconocido"
        raise RuntimeError(str(msg))

    results = payload.get("ParsedResults") or []
    if not results:
        return ""

    return (results[0].get("ParsedText") or "").strip()


# ============ BOT ============
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Mándame una foto y te regreso el texto.")

def photo_received(update, context):
    message = update.message
    if not message or not message.photo:
        return

    # Mejor calidad
    photo = message.photo[-1]
    file_unique_id = photo.file_unique_id

    logging.info(f"PHOTO_RECEIVED chat={message.chat_id} msg={message.message_id} user={message.from_user.id} file_unique_id={file_unique_id}")

    # Descargar bytes
    tg_file = context.bot.get_file(photo.file_id)
    raw_bytes = bytes(tg_file.download_as_bytearray())

    # Convertir a JPG real + hash
    try:
        jpeg_bytes = normalize_to_jpeg(raw_bytes)
    except Exception as e:
        logging.exception("JPG convert failed")
        update.message.reply_text("❌ No pude convertir la imagen a JPG. Intenta con otra foto.")
        return

    image_hash = compute_image_hash(jpeg_bytes)

    # Duplicado por hash (más confiable que file_unique_id si reenvían/descargan)
    existing, count = find_existing_by_hash(image_hash)
    if existing:
        created_at, _old_text = existing
        when_pretty = to_cdmx_pretty(created_at)
        times = count  # cuántas veces ya existe en DB

        update.message.reply_text(
            f"✅ Esa foto ya estaba registrada.\n"
            f"🕒 Primera vez: {when_pretty}\n"
            f"🔁 Veces registrada: {times}"
        )
        logging.info(f"DUPLICATE ignored: hash={image_hash} first={created_at} count={times}")
        return

    # Si no es duplicado, corre OCR
    try:
        text = ocr_space_bytes(jpeg_bytes)
        status = "ok" if text.strip() else "empty"
        err = None
    except Exception as e:
        text = ""
        status = "error"
        err = str(e)
        logging.warning(f"OCR error: {err}")

    # Guardar en DB
    insert_record(
        chat_id=message.chat_id,
        user_id=message.from_user.id,
        message_id=message.message_id,
        file_unique_id=file_unique_id,
        image_hash=image_hash,
        status=status,
        ocr_error=err,
        ocr_text=text
    )

    # Responder con preview
    if text.strip():
        preview = safe_preview(text, limit=3500)
        update.message.reply_text("📄 Texto detectado y guardado:\n\n" + preview)
    else:
        if status == "error":
            update.message.reply_text("📸 Foto guardada, pero OCR falló (sin texto legible).")
        else:
            update.message.reply_text("📸 Foto guardada (sin texto legible).")


def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN")

    init_db()

    # Dummy server para Render (healthcheck)
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, photo_received))

    logging.info("Bot arrancando polling...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()

