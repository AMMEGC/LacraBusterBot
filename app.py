import os
import threading
import sqlite3
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# -------------------------
# Config / Env
# -------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("lacra-bot")


# -------------------------
# Render dummy server (health)
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
    # Si DBPATH no tiene carpeta (raro), evita crash
    folder = os.path.dirname(DBPATH)
    if folder:
        os.makedirs(folder, exist_ok=True)

    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()

    # TABLA CORRECTA: ocr_texts (es la que usamos en inserts)
    # dedupe: file_unique_id UNIQUE para que la misma foto no se guarde 2 veces
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
    log.info("DB inicializada en: %s", DBPATH)


def already_processed(file_unique_id: str) -> bool:
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM ocr_texts WHERE file_unique_id = ? LIMIT 1", (file_unique_id,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def save_ocr(chat_id, user_id, message_id, file_unique_id, ocr_text) -> bool:
    """
    Devuelve True si insertó, False si ya existía (por UNIQUE).
    """
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO ocr_texts
        (chat_id, user_id, message_id, file_unique_id, created_at, ocr_text)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        chat_id,
        user_id,
        message_id,
        file_unique_id,
        datetime.utcnow().isoformat(),
        ocr_text
    ))
    inserted = (cur.rowcount == 1)
    conn.commit()
    conn.close()
    return inserted


# -------------------------
# OCR.space
# -------------------------
def ocr_space(image_bytes: bytes) -> str:
    if not OCR_SPACE_API_KEY:
        log.warning("Falta OCR_SPACE_API_KEY en env vars.")
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

        # Log del payload para debug (se ve en Render logs)
        log.info("OCR RAW RESPONSE: %s", payload)

        if payload.get("IsErroredOnProcessing"):
            err = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Error OCR desconocido"
            log.warning("OCR errored: %s", err)
            return ""

        results = payload.get("ParsedResults") or []
        if not results:
            return ""

        return (results[0].get("ParsedText") or "").strip()

    except Exception as e:
        log.exception("Fallo OCR request: %s", e)
        return ""


# -------------------------
# Telegram Bot handlers
# -------------------------
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto.")


def photo_received(update, context):
    message = update.message
    if not message:
        return

    # Log de entrada (esto DEBE aparecer cada que mandas foto)
    log.info("PHOTO_RECEIVED chat=%s msg=%s user=%s",
             getattr(message, "chat_id", None),
             getattr(message, "message_id", None),
             getattr(getattr(message, "from_user", None), "id", None))

    if not message.photo:
        return

    photo = message.photo[-1]  # mejor calidad
    file_unique_id = photo.file_unique_id

    # Dedupe rápido (antes de gastar OCR)
    if already_processed(file_unique_id):
        update.message.reply_text("✅ Esa foto ya estaba registrada.")
        log.info("DUPLICATE ignored: file_unique_id=%s", file_unique_id)
        return

    # Descargar bytes
    try:
        tg_file = context.bot.get_file(photo.file_id)
        image_bytes = bytes(tg_file.download_as_bytearray())
    except Exception as e:
        log.exception("No pude descargar imagen de Telegram: %s", e)
        update.message.reply_text("❌ No pude descargar la foto. Intenta otra vez.")
        return

    # OCR
    text = ocr_space(image_bytes)

    # Guardar
    inserted = save_ocr(
        chat_id=message.chat_id,
        user_id=message.from_user.id if message.from_user else None,
        message_id=message.message_id,
        file_unique_id=file_unique_id,
        ocr_text=text
    )

    if not inserted:
        update.message.reply_text("✅ Esa foto ya estaba registrada.")
        return

    if text.strip():
        update.message.reply_text("📄 Texto detectado y guardado.")
    else:
        update.message.reply_text("📸 Foto guardada (sin texto legible).")


def on_error(update, context):
    # Para que no salga “No error handlers…”
    log.exception("TELEGRAM ERROR: %s", context.error)


def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN en Environment Variables")

    init_db()

    # Server dummy para Render
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, photo_received))
    dp.add_error_handler(on_error)

    log.info("Bot arrancando polling…")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
