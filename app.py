import os
import threading
import sqlite3
from datetime import datetime
from io import BytesIO
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from PIL import Image
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# ================== ENV ==================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")
DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")

# ================== HTTP SERVER (Render) ==================
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

# ================== DB ==================
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
    conn.commit()
    conn.close()
    print("[INFO] DB inicializada en:", DBPATH)

def is_duplicate(file_unique_id):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM ocr_texts WHERE file_unique_id = ?", (file_unique_id,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

def save_ocr(chat_id, user_id, message_id, file_unique_id, text):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ocr_texts
        (chat_id, user_id, message_id, file_unique_id, created_at, ocr_text)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        chat_id,
        user_id,
        message_id,
        file_unique_id,
        datetime.utcnow().isoformat(),
        text
    ))
    conn.commit()
    conn.close()

# ================== IMAGE FIX (CLAVE) ==================
def normalize_to_jpeg(image_bytes: bytes) -> bytes:
    """
    Convierte cualquier imagen recibida de Telegram
    a JPEG real compatible con OCR.space
    """
    img = Image.open(BytesIO(image_bytes))
    img = img.convert("RGB")

    out = BytesIO()
    img.save(out, format="JPEG", quality=95)
    return out.getvalue()

# ================== OCR ==================
def ocr_space(image_bytes: bytes) -> str:
    jpeg_bytes = normalize_to_jpeg(image_bytes)

    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.jpg", jpeg_bytes)}
    data = {
        "apikey": OCR_SPACE_API_KEY,
        "language": "spa",
        "OCREngine": "2",
        "scale": "true",
        "detectOrientation": "true",
        "isOverlayRequired": "false"
    }

    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    payload = r.json()

    print("[INFO] OCR RAW RESPONSE:", payload)

    if payload.get("IsErroredOnProcessing"):
        raise RuntimeError(payload.get("ErrorMessage") or payload.get("ErrorDetails"))

    results = payload.get("ParsedResults") or []
    if not results:
        return ""

    return (results[0].get("ParsedText") or "").strip()

# ================== BOT ==================
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto.")

def photo_received(update, context):
    message = update.message
    photo = message.photo[-1]

    print(f"[INFO] PHOTO_RECEIVED chat={message.chat_id} msg={message.message_id}")

    if is_duplicate(photo.file_unique_id):
        update.message.reply_text("✅ Esa foto ya estaba registrada.")
        print("[INFO] DUPLICATE ignored:", photo.file_unique_id)
        return

    file = context.bot.get_file(photo.file_id)
    image_bytes = file.download_as_bytearray()

    try:
        text = ocr_space(image_bytes)
    except Exception as e:
        print("[WARNING] OCR error:", e)
        update.message.reply_text("📸 Foto guardada (sin texto legible).")
        save_ocr(
            message.chat_id,
            message.from_user.id,
            message.message_id,
            photo.file_unique_id,
            ""
        )
        return

    save_ocr(
        message.chat_id,
        message.from_user.id,
        message.message_id,
        photo.file_unique_id,
        text
    )

    if text:
        update.message.reply_text("📄 Texto detectado y guardado.")
    else:
        update.message.reply_text("📸 Foto guardada (sin texto legible).")

# ================== MAIN ==================
def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN")

    if not OCR_SPACE_API_KEY:
        raise RuntimeError("Falta OCR_SPACE_API_KEY")

    init_db()

    threading.Thread(target=run_server, daemon=True).start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, photo_received))

    print("[INFO] Bot arrancando polling...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
