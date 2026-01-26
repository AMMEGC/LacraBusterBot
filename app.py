import os
import io
import threading
import sqlite3
import logging
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from PIL import Image

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# =========================
# Config / Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BOT_TOKEN = os.environ.get("BOT_TOKEN")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")
DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")


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
# DB
# =========================
def init_db():
    os.makedirs(os.path.dirname(DBPATH), exist_ok=True)
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()

    # Tabla única
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

    # Índice para duplicados
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ocr_texts_file_unique_id
        ON ocr_texts(file_unique_id)
    """)

    conn.commit()
    conn.close()
    logging.info(f"DB inicializada en: {DBPATH}")


def is_duplicate(file_unique_id: str) -> bool:
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM ocr_texts WHERE file_unique_id = ? LIMIT 1", (file_unique_id,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def get_first_seen(file_unique_id: str):
    """Regresa created_at del primer registro (ASC) para ese file_unique_id."""
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT created_at FROM ocr_texts WHERE file_unique_id = ? ORDER BY id ASC LIMIT 1",
        (file_unique_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def save_ocr(chat_id, user_id, message_id, file_unique_id, ocr_text):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ocr_texts (chat_id, user_id, message_id, file_unique_id, created_at, ocr_text)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        chat_id,
        user_id,
        message_id,
        file_unique_id,
        datetime.now(timezone.utc).isoformat(),
        ocr_text
    ))
    conn.commit()
    conn.close()


# =========================
# Image -> Valid JPEG bytes
# =========================
def to_valid_jpeg_bytes(image_bytes: bytes) -> bytes:
    """
    Telegram a veces manda bytes que OCR.space interpreta mal.
    Aquí abrimos con Pillow y re-guardamos como JPEG real.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92, optimize=True)
    return out.getvalue()


# =========================
# OCR.space call
# =========================
def ocr_space_bytes(image_bytes: bytes) -> str:
    if not OCR_SPACE_API_KEY:
        raise RuntimeError("Falta OCR_SPACE_API_KEY en Environment Variables (Render)")

    # Forzar JPEG válido
    jpeg_bytes = to_valid_jpeg_bytes(image_bytes)

    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.jpg", jpeg_bytes, "image/jpeg")}
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

    logging.info(f"OCR RAW RESPONSE: {payload}")

    if payload.get("IsErroredOnProcessing"):
        msg = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Error OCR desconocido"
        raise RuntimeError(str(msg))

    results = payload.get("ParsedResults") or []
    if not results:
        return ""

    return (results[0].get("ParsedText") or "").strip()


# =========================
# Telegram bot handlers
# =========================
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto.")


def photo_received(update, context):
    message = update.message
    if not message or not message.photo:
        return

    photo = message.photo[-1]  # mejor calidad
    file_unique_id = photo.file_unique_id

    logging.info(f"PHOTO_RECEIVED chat={message.chat_id} msg={message.message_id} user={message.from_user.id}")

    # Duplicado
    if is_duplicate(file_unique_id):
        first_seen = get_first_seen(file_unique_id)
        if first_seen:
            update.message.reply_text(f"✅ Esa foto ya estaba registrada.\n🕒 Primera vez: {first_seen} (UTC)")
        else:
            update.message.reply_text("✅ Esa foto ya estaba registrada.")
        logging.info(f"DUPLICATE ignored: {file_unique_id} first_seen={first_seen}")
        return

    # Descargar bytes desde Telegram
    file = context.bot.get_file(photo.file_id)
    image_bytes = file.download_as_bytearray()

    # OCR
    text = ""
    try:
        text = ocr_space_bytes(image_bytes)
    except Exception as e:
        logging.warning(f"OCR error: {e}")
        text = ""

    # Guardar DB (aunque no haya texto)
    save_ocr(
        chat_id=message.chat_id,
        user_id=message.from_user.id,
        message_id=message.message_id,
        file_unique_id=file_unique_id,
        ocr_text=text
    )

    if text.strip():
        update.message.reply_text("📄 Texto detectado y guardado.")
    else:
        update.message.reply_text("📸 Foto guardada (sin texto legible).")


def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN")

    init_db()

    # Server dummy (Render)
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
