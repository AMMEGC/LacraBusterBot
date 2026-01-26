import os
import threading
import sqlite3
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# ---------------- CONFIG ----------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BOT_TOKEN = os.environ.get("BOT_TOKEN")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")
DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")

# ---------------- UTIL ----------------

def iso_to_pretty(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_str

# ---------------- OCR ----------------

def ocr_space_bytes(image_bytes: bytes) -> str:
    if not OCR_SPACE_API_KEY:
        logging.warning("OCR_SPACE_API_KEY no configurada")
        return ""

    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.jpg", image_bytes)}
    data = {
        "apikey": OCR_SPACE_API_KEY,
        "language": "spa",
        "OCREngine": "2",
        "scale": "true",
        "detectOrientation": "true",
        "isOverlayRequired": "false",
        "filetype": "JPG",
    }

    try:
        r = requests.post(url, files=files, data=data, timeout=60)
        r.raise_for_status()
        payload = r.json()

        logging.info("OCR RAW RESPONSE: %s", payload)

        if payload.get("IsErroredOnProcessing"):
            logging.warning("OCR error: %s", payload.get("ErrorMessage"))
            return ""

        results = payload.get("ParsedResults") or []
        if not results:
            return ""

        return (results[0].get("ParsedText") or "").strip()

    except Exception as e:
        logging.exception("OCR exception")
        return ""

# ---------------- DB ----------------

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
            file_unique_id TEXT UNIQUE,
            created_at TEXT,
            ocr_text TEXT
        )
    """)

    conn.commit()
    conn.close()
    logging.info("DB inicializada en %s", DBPATH)


def get_existing_by_file_unique_id(file_unique_id: str):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT created_at, chat_id, user_id, message_id
        FROM ocr_texts
        WHERE file_unique_id = ?
    """, (file_unique_id,))
    row = cur.fetchone()
    conn.close()
    return row


def save_ocr(chat_id, user_id, message_id, file_unique_id, ocr_text):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ocr_texts (
            chat_id, user_id, message_id, file_unique_id, created_at, ocr_text
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        chat_id,
        user_id,
        message_id,
        file_unique_id,
        datetime.utcnow().isoformat(),
        ocr_text
    ))
    conn.commit()
    conn.close()

# ---------------- RENDER DUMMY SERVER ----------------

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

# ---------------- BOT ----------------

def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto.")


def photo_received(update, context):
    message = update.message
    if not message or not message.photo:
        return

    logging.info(
        "PHOTO_RECEIVED chat=%s msg=%s user=%s",
        message.chat_id,
        message.message_id,
        message.from_user.id
    )

    photo = message.photo[-1]
    file_unique_id = photo.file_unique_id

    existing = get_existing_by_file_unique_id(file_unique_id)
    if existing:
        created_at, chat_id_old, user_id_old, msg_id_old = existing
        update.message.reply_text(
            f"✅ Esa foto ya estaba registrada.\n"
            f"📅 Registrada: {iso_to_pretty(created_at)}\n"
            f"🧾 Ref: chat={chat_id_old}, msg={msg_id_old}"
        )
        logging.info("DUPLICATE ignored: %s", file_unique_id)
        return

    file = context.bot.get_file(photo.file_id)
    image_bytes = file.download_as_bytearray()

    text = ocr_space_bytes(image_bytes)

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

# ---------------- MAIN ----------------

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

    logging.info("Bot arrancando polling...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
