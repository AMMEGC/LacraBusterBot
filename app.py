import os
import threading
import sqlite3
import logging
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import requests
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Pillow (para convertir a JPG real)
from PIL import Image

# (Opcional) ImageHash para “similitud” futura. Si no está, no rompe nada.
try:
    import imagehash
except Exception:
    imagehash = None


# =========================
# Config
# =========================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")

# Logging “tipo servidor” (sale en Render logs)
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

    # quita ruido en logs
    def log_message(self, format, *args):
        return


def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


# =========================
# DB
# =========================
def init_db():
    os.makedirs(os.path.dirname(DBPATH), exist_ok=True)
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()

    # Tabla única y consistente
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ocr_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            user_id INTEGER,
            message_id INTEGER,
            file_unique_id TEXT UNIQUE,
            image_hash TEXT,
            created_at TEXT,
            ocr_text TEXT,
            ocr_status TEXT,
            ocr_error TEXT
        )
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_created_at ON ocr_texts(created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_image_hash ON ocr_texts(image_hash)")

    conn.commit()
    conn.close()
    log.info("DB inicializada en: %s", DBPATH)


def get_existing_created_at(file_unique_id: str):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("SELECT created_at FROM ocr_texts WHERE file_unique_id = ? LIMIT 1", (file_unique_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def insert_record(chat_id, user_id, message_id, file_unique_id, image_hash, created_at, ocr_text, ocr_status, ocr_error):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO ocr_texts
        (chat_id, user_id, message_id, file_unique_id, image_hash, created_at, ocr_text, ocr_status, ocr_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (chat_id, user_id, message_id, file_unique_id, image_hash, created_at, ocr_text, ocr_status, ocr_error),
    )
    conn.commit()
    conn.close()


# =========================
# Imagen -> JPG REAL (baseline)
# =========================
def to_real_jpeg_bytes(src_bytes: bytes) -> bytes:
    """
    Convierte cualquier cosa que mande Telegram (webp, jpg raro, etc.)
    a un JPG baseline REAL (RGB, no progressive).
    """
    with Image.open(BytesIO(src_bytes)) as im:
        if im.mode not in ("RGB",):
            im = im.convert("RGB")

        out = BytesIO()
        im.save(
            out,
            format="JPEG",
            quality=90,
            optimize=True,
            progressive=False,  # <-- CLAVE: baseline real
        )
        return out.getvalue()


def compute_hash(jpeg_bytes: bytes) -> str:
    if not imagehash:
        return None
    try:
        with Image.open(BytesIO(jpeg_bytes)) as im:
            return str(imagehash.phash(im))
    except Exception:
        return None


# =========================
# OCR.space
# =========================
def ocr_space_bytes(image_bytes: bytes) -> str:
    if not OCR_SPACE_API_KEY:
        raise RuntimeError("Falta OCR_SPACE_API_KEY en Environment Variables (Render)")

    url = "https://api.ocr.space/parse/image"

    # OCR.space es MUY sensible al file “de verdad”
    files = {
        "file": ("image.jpg", image_bytes, "image/jpeg")
    }

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

    # Log completo (como lo pedías)
    log.info("OCR RAW RESPONSE >>> %s", payload)

    if payload.get("IsErroredOnProcessing"):
        msg = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Error OCR desconocido"
        # a veces ErrorMessage viene como lista
        if isinstance(msg, list):
            msg = " | ".join([str(x) for x in msg])
        raise RuntimeError(str(msg))

    results = payload.get("ParsedResults") or []
    if not results:
        return ""

    return (results[0].get("ParsedText") or "").strip()


# =========================
# Bot handlers
# =========================
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto.")


def format_when(iso_utc: str) -> str:
    """
    Guarda todo en UTC. Al usuario le mostramos claro “UTC”.
    """
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return iso_utc


def photo_received(update, context):
    message = update.message
    if not message or not message.photo:
        return

    # Mejor calidad
    photo = message.photo[-1]
    file_unique_id = photo.file_unique_id

    log.info("PHOTO_RECEIVED chat=%s msg=%s user=%s file_unique_id=%s",
             message.chat_id, message.message_id, message.from_user.id, file_unique_id)

    # 1) Duplicado exacto por file_unique_id (rápido y fiable)
    existing_created_at = get_existing_created_at(file_unique_id)
    if existing_created_at:
        when = format_when(existing_created_at)
        update.message.reply_text(f"✅ Esa foto ya estaba registrada.\n🕒 Primera vez: {when}")
        log.info("DUPLICATE ignored: file_unique_id=%s first_seen=%s", file_unique_id, existing_created_at)
        return

    # 2) Descargar bytes reales desde Telegram
    tg_file = context.bot.get_file(photo.file_id)
    src_bytes = tg_file.download_as_bytearray()

    # 3) Convertir a JPG REAL antes de OCR (CLAVE)
    try:
        jpeg_bytes = to_real_jpeg_bytes(src_bytes)
    except Exception as e:
        log.exception("JPEG conversion failed: %s", e)
        # guardamos registro igual, para no perder evidencia
        created_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        insert_record(
            message.chat_id, message.from_user.id, message.message_id,
            file_unique_id, None, created_at,
            "", "jpeg_failed", str(e)
        )
        update.message.reply_text("⚠️ No pude convertir la imagen a JPG. Intenta con otra foto.")
        return

    img_hash = compute_hash(jpeg_bytes)

    # 4) OCR
    created_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    ocr_text = ""
    ocr_status = "ok"
    ocr_error = None

    try:
        ocr_text = ocr_space_bytes(jpeg_bytes)
        if not ocr_text.strip():
            ocr_status = "empty"
    except Exception as e:
        ocr_status = "error"
        ocr_error = str(e)
        log.warning("OCR error: %s", ocr_error)

    # 5) Guardar en DB (aunque falle OCR, guardamos)
    insert_record(
        message.chat_id,
        message.from_user.id,
        message.message_id,
        file_unique_id,
        img_hash,
        created_at,
        ocr_text,
        ocr_status,
        ocr_error,
    )

    # 6) Respuesta al usuario
    if ocr_status == "ok" and ocr_text.strip():
        update.message.reply_text("📄 Texto detectado y guardado.")
    elif ocr_status == "empty":
        update.message.reply_text("📸 Foto guardada (sin texto legible).")
    else:
        # si OCR falló, por lo menos lo decimos
        update.message.reply_text("📸 Foto guardada (OCR falló / sin texto legible).")


# =========================
# Main
# =========================
def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN en Environment Variables")

    init_db()

    # Dummy server para Render
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Identidad del bot (para que NO haya dudas de token)
    try:
        me = updater.bot.get_me()
        log.info("Bot conectado: @%s (id=%s)", me.username, me.id)
    except Exception as e:
        log.warning("No pude obtener get_me(): %s", e)

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, photo_received))

    log.info("Bot arrancando polling...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()

