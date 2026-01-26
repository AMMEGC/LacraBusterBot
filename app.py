import os
import threading
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import requests
from PIL import Image
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

BOT_TOKEN = os.environ.get("BOT_TOKEN")
DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")
OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")


# -------------------------
# Render dummy web server
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
# DB helpers
# -------------------------
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    os.makedirs(os.path.dirname(DBPATH), exist_ok=True)
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()

    # 1) Tabla base
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

    # 2) Migraciones: agrega columnas si faltan (sin borrar DB)
    cur.execute("PRAGMA table_info(ocr_texts)")
    existing_cols = {row[1] for row in cur.fetchall()}

    def add_col(name, ddl):
        if name not in existing_cols:
            cur.execute(f"ALTER TABLE ocr_texts ADD COLUMN {ddl}")

    # columnas extra (por si las usas hoy o mañana)
    add_col("ocr_status", "ocr_status TEXT")
    add_col("error_message", "error_message TEXT")
    add_col("image_hash", "image_hash TEXT")
    add_col("duplicate_of_id", "duplicate_of_id INTEGER")
    add_col("similarity", "similarity REAL")

    # índice para dedupe por file_unique_id (si no existe)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ocr_texts_file_unique_id ON ocr_texts(file_unique_id)")

    conn.commit()
    conn.close()
    print(f"[INFO] DB inicializada en: {DBPATH}")


def get_existing_by_file_unique_id(file_unique_id: str):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, created_at
        FROM ocr_texts
        WHERE file_unique_id = ?
        ORDER BY id ASC
        LIMIT 1
    """, (file_unique_id,))
    row = cur.fetchone()
    conn.close()
    return row  # (id, created_at) o None


def insert_record(chat_id, user_id, message_id, file_unique_id, created_at, ocr_text, ocr_status="ok", error_message=None):
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ocr_texts (chat_id, user_id, message_id, file_unique_id, created_at, ocr_text, ocr_status, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (chat_id, user_id, message_id, file_unique_id, created_at, ocr_text, ocr_status, error_message))
    conn.commit()
    conn.close()


# -------------------------
# Image -> Real JPEG bytes
# -------------------------
def to_real_jpeg(image_bytes: bytes) -> bytes:
    """
    Convierte bytes (PNG/WebP/lo que sea) a JPEG real.
    Esto evita el error E301 de OCR.space ("Input file corrupted?").
    """
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        out = BytesIO()
        img.save(out, format="JPEG", quality=92, optimize=True)
        return out.getvalue()


# -------------------------
# OCR.space
# -------------------------
def ocr_space(image_bytes: bytes) -> str:
    if not OCR_SPACE_API_KEY:
        raise RuntimeError("Falta OCR_SPACE_API_KEY en Environment Variables")

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

    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    payload = r.json()
    print("[INFO] OCR RAW RESPONSE >>>", payload)

    if payload.get("IsErroredOnProcessing"):
        msg = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Error OCR desconocido"
        raise RuntimeError(str(msg))

    results = payload.get("ParsedResults") or []
    if not results:
        return ""

    return (results[0].get("ParsedText") or "").strip()


# -------------------------
# Telegram bot handlers
# -------------------------
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto.")


def photo_received(update, context):
    message = update.message
    if not message or not message.photo:
        return

    photo = message.photo[-1]  # mejor calidad
    file_unique_id = photo.file_unique_id

    print(f"[INFO] PHOTO_RECEIVED chat={message.chat_id} msg={message.message_id} user={message.from_user.id} file_unique_id={file_unique_id}")

    # 1) Dedupe por file_unique_id
    existing = get_existing_by_file_unique_id(file_unique_id)
    if existing:
        existing_id, existing_created_at = existing
        when = existing_created_at or "(sin fecha)"
        update.message.reply_text(f"✅ Esa foto ya estaba registrada.\n🕒 Primera vez: {when}")
        print(f"[INFO] DUPLICATE ignored: file_unique_id={file_unique_id} first_seen={when} row_id={existing_id}")
        return

    # 2) Descargar bytes reales desde Telegram
    tg_file = context.bot.get_file(photo.file_id)
    raw_bytes = tg_file.download_as_bytearray()

    # 3) Convertir a JPEG real antes del OCR
    try:
        jpeg_bytes = to_real_jpeg(raw_bytes)
    except Exception as e:
        err = f"convert_to_jpeg_failed: {repr(e)}"
        created_at = utc_now_iso()
        insert_record(message.chat_id, message.from_user.id, message.message_id, file_unique_id, created_at, "", ocr_status="error", error_message=err)
        update.message.reply_text("📸 Foto guardada, pero falló la conversión a JPEG.")
        print("[ERROR]", err)
        return

    # 4) OCR
    created_at = utc_now_iso()
    try:
        text = ocr_space(jpeg_bytes)
        insert_record(message.chat_id, message.from_user.id, message.message_id, file_unique_id, created_at, text, ocr_status="ok", error_message=None)

        if text.strip():
            update.message.reply_text("📄 Texto detectado y guardado.")
        else:
            update.message.reply_text("📸 Foto guardada (sin texto legible).")

    except Exception as e:
        err = repr(e)
        insert_record(message.chat_id, message.from_user.id, message.message_id, file_unique_id, created_at, "", ocr_status="error", error_message=err)
        update.message.reply_text("📸 Foto guardada, pero el OCR falló.")
        print("[WARN] OCR error:", err)


def on_error(update, context):
    # Esto evita “No error handlers are registered…”
    print("[ERROR_HANDLER]", repr(context.error))


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
    dp.add_error_handler(on_error)

    print("[INFO] Bot arrancando polling...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()


