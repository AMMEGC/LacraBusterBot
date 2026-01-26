import os
import threading
import sqlite3
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

BOT_TOKEN = os.environ.get("BOT_TOKEN")

DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")

def ocr_space_extract_text(image_bytes: bytes) -> str:
    if not OCR_SPACE_API_KEY:
        return ""

    url = "https://api.ocr.space/parse/image"
    files = {"filename": ("image.jpg", image_bytes)}
    data = {
        "apikey": OCR_SPACE_API_KEY,
        "language": "spa",
        "isOverlayRequired": "false",
        "OCREngine": "2",
    }

    try:
        r = requests.post(url, files=files, data=data, timeout=60)
        r.raise_for_status()
        payload = r.json()

        if payload.get("IsErroredOnProcessing"):
            return ""

        parsed = payload.get("ParsedResults", [])
        if not parsed:
            return ""

        return (parsed[0].get("ParsedText") or "").strip()
    except Exception:
        return ""


# --- Servidor dummy para Render (si lo sigues usando) ---
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


# --- DB ---
def init_db():
    os.makedirs(os.path.dirname(DBPATH), exist_ok=True)
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ocr_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            user_id INTEGER,
            message_id INTEGER,
            file_unique_id TEXT,
            created_at TEXT,
            ocr_text TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_ocr(chat_id, user_id, message_id, file_unique_id, ocr_text):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("""
INSERT INTO ocr_texts (chat_id, user_id, message_id, file_unique_id, created_at, ocr_text)
VALUES (?, ?, ?, ?, ?, ?)
""", (chat_id, user_id, message_id, file_unique_id, datetime.utcnow().isoformat(), ocr_text))
    conn.commit()
    conn.close()


# --- OCR.space ---
def ocr_space_bytes(image_bytes: bytes) -> str:
    if not OCR_SPACE_API_KEY:
        raise RuntimeError("Falta OCR_SPACE_API_KEY en Environment Variables (Render)")

    # Endpoint oficial /parse/image :contentReference[oaicite:2]{index=2}
    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.jpg", image_bytes)}
    data = {
    "apikey": OCR_SPACE_API_KEY,
    "language": "spa",
    "OCREngine": "2",
    "scale": "true",
    "detectOrientation": "true"
    "isOverlayRequired": "false",
    "filetype": "JPG"
}


    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    payload = r.json()
    print("OCR RAW RESPONSE >>>", payload)

    if payload.get("IsErroredOnProcessing"):
        msg = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Error OCR desconocido"
        raise RuntimeError(str(msg))

    results = payload.get("ParsedResults") or []
    if not results:
        return ""

    text = (results[0].get("ParsedText") or "").strip()
    return text


# --- Bot ---
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto.")


def photo_received(update, context):
    message = update.message

    if not message.photo:
        return

    photo = message.photo[-1]  # mejor calidad
    file_unique_id = photo.file_unique_id
    file = context.bot.get_file(photo.file_id)
    image_bytes = file.download_as_bytearray()

    text = ocr_space_extract_text(image_bytes)

    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()

    cur.execute("""
INSERT INTO ocr_texts (chat_id, user_id, message_id, file_unique_id, created_at, ocr_text)
VALUES (?, ?, ?, ?, ?, ?)
""", (
    message.chat_id,
    message.from_user.id,
    message.message_id,
    file_unique_id,
    datetime.utcnow().isoformat(),
    text
))


    conn.commit()
    conn.close()

    if text.strip():
        update.message.reply_text("📄 Texto detectado y guardado.")
    else:
        update.message.reply_text("📸 Foto guardada (sin texto legible).")


def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN")

    init_db()

    # Server dummy (si lo necesitas)
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, photo_received))

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()

