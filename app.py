import os
import io
import threading
import sqlite3
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from PIL import Image
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

BOT_TOKEN = os.environ.get("BOT_TOKEN")
DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")


# -------------------------
# Util: timestamp (ISO)
# -------------------------
def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# -------------------------
# Render dummy server
# -------------------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


# -------------------------
# DB
# -------------------------
def init_db():
    os.makedirs(os.path.dirname(DBPATH), exist_ok=True)

    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()

    # Tabla REAL usada por tu bot (porque INSERT va a ocr_texts)
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

    # Índice para duplicados por file_unique_id
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ocr_texts_file_unique_id
        ON ocr_texts(file_unique_id)
    """)

    conn.commit()
    conn.close()

    print(f"{now_iso()} [INFO] DB inicializada en: {DBPATH}")


def is_duplicate(file_unique_id: str) -> tuple[bool, str]:
    """Regresa (True, created_at) si ya existe ese file_unique_id."""
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT created_at FROM ocr_texts WHERE file_unique_id=? LIMIT 1",
        (file_unique_id,)
    )
    row = cur.fetchone()
    conn.close()

    if row:
        return True, row[0]
    return False, ""


def save_ocr(chat_id, user_id, message_id, file_unique_id, created_at, ocr_text):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ocr_texts (chat_id, user_id, message_id, file_unique_id, created_at, ocr_text)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (chat_id, user_id, message_id, file_unique_id, created_at, ocr_text))
    conn.commit()
    conn.close()


# -------------------------
# OCR helpers
# -------------------------
def to_jpeg_bytes(image_bytes: bytes) -> bytes:
    """
    Fuerza conversión a JPEG real (evita E301 'Input file corrupted').
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def ocr_space_bytes(image_bytes: bytes) -> str:
    if not OCR_SPACE_API_KEY:
        raise RuntimeError("Falta OCR_SPACE_API_KEY en Environment Variables (Render)")

    # Fuerza JPEG real sí o sí
    jpeg_bytes = to_jpeg_bytes(bytes(image_bytes))

    url = "https://api.ocr.space/parse/image"

    # ✅ Campo correcto: "file"
    files = {"file": ("image.jpg", jpeg_bytes, "image/jpeg")}

    data = {
        "apikey": OCR_SPACE_API_KEY,
        "language": "spa",
        "OCREngine": "2",
        "scale": "true",
        "detectOrientation": "true",
        "isOverlayRequired": "false",
    }

    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    payload = r.json()

    print(f"{now_iso()} [INFO] OCR RAW RESPONSE >>> {payload}")

    if payload.get("IsErroredOnProcessing"):
        msg = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Error OCR desconocido"
        raise RuntimeError(str(msg))

    results = payload.get("ParsedResults") or []
    if not results:
        return ""

    return (results[0].get("ParsedText") or "").strip()


# -------------------------
# Bot handlers
# -------------------------
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto.")


def photo_received(update, context):
    message = update.message
    if not message or not message.photo:
        return

    # mejor calidad
    photo = message.photo[-1]
    file_unique_id = photo.file_unique_id

    print(f"{now_iso()} [INFO] PHOTO_RECEIVED chat={message.chat_id} msg={message.message_id} user={message.from_user.id}")

    # Duplicado
    dup, created_at = is_duplicate(file_unique_id)
    if dup:
        print(f"{now_iso()} [INFO] DUPLICATE ignored: file_unique_id={file_unique_id} first_seen={created_at}")
        update.message.reply_text(f"✅ Esa foto ya estaba registrada.\n🕒 Primera vez: {created_at}")
        return

    # Descargar imagen
    tg_file = context.bot.get_file(photo.file_id)
    image_bytes = tg_file.download_as_bytearray()

    # OCR
    text = ""
    try:
        text = ocr_space_bytes(image_bytes)
    except Exception as e:
        print(f"{now_iso()} [WARNING] OCR error: {repr(e)}")
        text = ""

    created_at_now = now_iso()
    save_ocr(
        message.chat_id,
        message.from_user.id,
        message.message_id,
        file_unique_id,
        created_at_now,
        text
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

    print(f"{now_iso()} [INFO] Bot arrancando polling...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
