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
        INSERT INTO ocr_logs (chat_id, user_id, message_id, file_unique_id, created_at, ocr_text)
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
        "language": "spa",      # español :contentReference[oaicite:3]{index=3}
        "isOverlayRequired": "false",
        "OCREngine": "2",       # Engine 2 suele ir mejor en textos raros/rotados :contentReference[oaicite:4]{index=4}
        "scale": "true",
        "detectOrientation": "true",
    }

    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    payload = r.json()

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
    try:
        # foto más grande
        photo = update.message.photo[-1]
        file_unique_id = photo.file_unique_id

        tg_file = photo.get_file()
        image_bytes = tg_file.download_as_bytearray()

        text = ocr_space_bytes(image_bytes)

        save_ocr(
            chat_id=update.message.chat_id,
            user_id=update.message.from_user.id,
            message_id=update.message.message_id,
            file_unique_id=file_unique_id,
            ocr_text=text
        )

        preview = text[:300] + ("..." if len(text) > 300 else "")
        if preview.strip():
            update.message.reply_text("✅ OCR guardado en la base.\n\n🧾 Texto detectado:\n" + preview)
        else:
            update.message.reply_text("✅ OCR guardado en la base, pero no detecté texto legible en esta foto.")

    except Exception as e:
        update.message.reply_text(f"❌ Error OCR/DB: {e}")


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

