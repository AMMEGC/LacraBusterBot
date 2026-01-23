import os
import threading
import sqlite3
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

BOT_TOKEN = os.environ.get("BOT_TOKEN")
DB_PATH = os.environ.get("DB_PATH", "/var/data/lacra.sqlite")


# --- Servidor dummy para Render (mantener vivo el worker) ---
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


# --- DB helpers ---
def db_connect():
    # Ojo: check_same_thread=False para evitar broncas con threads del bot
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def db_init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = db_connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            user_id INTEGER,
            username TEXT,
            file_id TEXT,
            file_unique_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_photos_chat_unique ON photos(chat_id, file_unique_id)"
    )
    conn.commit()
    conn.close()


def db_count_photos():
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM photos")
    n = cur.fetchone()[0]
    conn.close()
    return n


def db_insert_photo(chat_id, message_id, user_id, username, file_id, file_unique_id):
    conn = db_connect()
    cur = conn.cursor()

    # ¿ya existe esta foto en este chat?
    cur.execute(
        "SELECT id, message_id, created_at FROM photos WHERE chat_id=? AND file_unique_id=? LIMIT 1",
        (chat_id, file_unique_id),
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return False, row  # duplicado

    cur.execute(
        """
        INSERT INTO photos (chat_id, message_id, user_id, username, file_id, file_unique_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chat_id,
            message_id,
            user_id,
            username,
            file_id,
            file_unique_id,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return True, None


# --- Bot handlers ---
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto.\nUsa /dbstatus para ver la base.")


def dbstatus(update, context):
    # reporta estado de DB
    exists = os.path.exists(DB_PATH)
    count = db_count_photos() if exists else 0
    update.message.reply_text(
        "🗄️ DB STATUS\n"
        f"- DB_PATH: {DB_PATH}\n"
        f"- Existe: {'SI' if exists else 'NO'}\n"
        f"- Fotos guardadas: {count}"
    )


def photo_received(update, context):
    msg = update.message
    chat_id = msg.chat_id
    message_id = msg.message_id
    user_id = msg.from_user.id if msg.from_user else None
    username = msg.from_user.username if msg.from_user else None

    # Telegram manda lista de tamaños; agarramos el más grande
    photo = msg.photo[-1]
    file_id = photo.file_id
    file_unique_id = photo.file_unique_id

    saved, dup = db_insert_photo(chat_id, message_id, user_id, username, file_id, file_unique_id)

    if saved:
        update.message.reply_text("💾 Foto guardada en la base.")
    else:
        dup_id, dup_msg_id, dup_date = dup
        update.message.reply_text(
            "⚠️ Ojo: esta foto (misma imagen) ya se había mandado antes en este grupo.\n"
            f"- Registro ID: {dup_id}\n"
            f"- Message ID: {dup_msg_id}\n"
            f"- Fecha (UTC): {dup_date}"
        )


def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN")

    # Inicializa DB al arrancar
    db_init()
    print(f"[BOOT] DB_PATH={DB_PATH} (exists={os.path.exists(DB_PATH)})")

    # servidor dummy en hilo
    threading.Thread(target=run_server, daemon=True).start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("dbstatus", dbstatus))
    dp.add_handler(MessageHandler(Filters.photo, photo_received))

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
