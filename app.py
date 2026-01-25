import os
import io
import re
import time
import hashlib
import sqlite3
import threading
from datetime import datetime

from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image
import imagehash
import pytesseract

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

BOT_TOKEN = os.environ.get("BOT_TOKEN")
DBPATH = os.environ.get("DBPATH", "/var/data/lacra.sqlite")

# ---- Servidor dummy para Render (si lo necesitas para healthcheck) ----
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

# ---- DB ----
def db_connect():
    os.makedirs(os.path.dirname(DBPATH), exist_ok=True)
    return sqlite3.connect(DBPATH)

def init_db():
    with db_connect() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            user_id INTEGER,
            username TEXT,
            message_id INTEGER,
            file_id TEXT,
            file_unique_id TEXT,
            sha256 TEXT,
            phash TEXT,
            ocr_text TEXT,
            created_at TEXT
        )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_photos_sha256 ON photos(sha256)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_photos_chat ON photos(chat_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_photos_created ON photos(created_at)")

# ---- Utilidades ----
def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def compute_phash(data: bytes) -> str:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return str(imagehash.phash(img))  # hex string

def do_ocr(data: bytes) -> str:
    # OCR en español; si no encuentra, igual regresa algo.
    img = Image.open(io.BytesIO(data)).convert("RGB")
    text = pytesseract.image_to_string(img, lang="spa")
    # limpia un poco
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text

def now_iso():
    return datetime.utcnow().isoformat()

def find_exact_match(chat_id: int, sha256: str):
    with db_connect() as con:
        cur = con.execute(
            "SELECT message_id, username, created_at FROM photos WHERE chat_id=? AND sha256=? ORDER BY id DESC LIMIT 1",
            (chat_id, sha256)
        )
        return cur.fetchone()

def find_similar_match(chat_id: int, phash_hex: str, max_distance: int = 8):
    target = imagehash.hex_to_hash(phash_hex)
    with db_connect() as con:
        cur = con.execute(
            "SELECT message_id, username, created_at, phash FROM photos WHERE chat_id=? ORDER BY id DESC LIMIT 200",
            (chat_id,)
        )
        rows = cur.fetchall()

    best = None
    best_dist = 999
    for message_id, username, created_at, phash_db in rows:
        try:
            h = imagehash.hex_to_hash(phash_db)
        except Exception:
            continue
        dist = (target - h)
        if dist < best_dist:
            best_dist = dist
            best = (message_id, username, created_at, phash_db, dist)

    if best and best[4] <= max_distance:
        return best  # (message_id, username, created_at, phash_db, dist)
    return None

def store_photo(chat_id, user_id, username, message_id, file_id, file_unique_id, sha256, phash, ocr_text):
    with db_connect() as con:
        con.execute("""
            INSERT INTO photos (chat_id, user_id, username, message_id, file_id, file_unique_id, sha256, phash, ocr_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (chat_id, user_id, username, message_id, file_id, file_unique_id, sha256, phash, ocr_text, now_iso()))

def extract_tokens(ocr_text: str):
    # Tokens útiles (números largos tipo licencia/INE, etc.)
    nums = re.findall(r"\b\d{6,}\b", ocr_text)
    # CURP (aprox) — opcional
    curp = re.findall(r"\b[A-Z]{4}\d{6}[HM][A-Z]{5}\d{2}\b", ocr_text.upper())
    return list(set(nums + curp))

def find_text_hits(chat_id: int, tokens):
    if not tokens:
        return []
    hits = []
    with db_connect() as con:
        for t in tokens[:6]:
            cur = con.execute(
                "SELECT message_id, username, created_at FROM photos WHERE chat_id=? AND ocr_text LIKE ? ORDER BY id DESC LIMIT 3",
                (chat_id, f"%{t}%")
            )
            for row in cur.fetchall():
                hits.append((t, row[0], row[1], row[2]))
    return hits[:6]

# ---- Bot handlers ----
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto (ID/INE/licencia/pasaporte).")

def buscar(update, context):
    chat_id = update.effective_chat.id
    query = " ".join(context.args).strip()
    if not query:
        update.message.reply_text("Uso: /buscar <texto o número>")
        return
    with db_connect() as con:
        cur = con.execute(
            "SELECT message_id, username, created_at FROM photos WHERE chat_id=? AND ocr_text LIKE ? ORDER BY id DESC LIMIT 5",
            (chat_id, f"%{query}%")
        )
        rows = cur.fetchall()

    if not rows:
        update.message.reply_text("No encontré coincidencias en la base.")
        return

    msg = "🔎 Coincidencias:\n"
    for message_id, username, created_at in rows:
        msg += f"- MsgID {message_id} | @{username or 'sin_user'} | {created_at}\n"
    update.message.reply_text(msg)

def photo_received(update, context):
    chat_id = update.effective_chat.id
    user = update.effective_user
    username = user.username if user else None

    # toma la mejor resolución
    photo = update.message.photo[-1]
    file_obj = context.bot.getFile(photo.file_id)
    data = file_obj.download_as_bytearray()

    sha = compute_sha256(data)
    ph = compute_phash(data)

    # 1) exacta
    exact = find_exact_match(chat_id, sha)
    if exact:
        message_id, u, created_at = exact
        update.message.reply_text(f"⚠️ Ojo: esta imagen ya había salido.\n- MsgID {message_id}\n- @{u or 'sin_user'}\n- {created_at}")
        return

    # 2) similar (pHash)
    similar = find_similar_match(chat_id, ph, max_distance=8)
    if similar:
        message_id, u, created_at, _, dist = similar
        update.message.reply_text(f"👀 Se parece a una foto anterior (distancia {dist}).\n- MsgID {message_id}\n- @{u or 'sin_user'}\n- {created_at}")

    # 3) OCR
    ocr_text = ""
    try:
        ocr_text = do_ocr(data)
    except Exception:
        ocr_text = ""

    store_photo(
        chat_id=chat_id,
        user_id=user.id if user else None,
        username=username,
        message_id=update.message.message_id,
        file_id=photo.file_id,
        file_unique_id=photo.file_unique_id,
        sha256=sha,
        phash=ph,
        ocr_text=ocr_text
    )

    # 4) buscar tokens dentro del OCR para avisar repetidos
    tokens = extract_tokens(ocr_text) if ocr_text else []
    hits = find_text_hits(chat_id, tokens)

    if hits:
        msg = "💾 Foto guardada. Y ojo, encontré coincidencias por texto:\n"
        for t, mid, u, created_at in hits:
            msg += f"- Token {t} → MsgID {mid} | @{u or 'sin_user'} | {created_at}\n"
        update.message.reply_text(msg)
    else:
        update.message.reply_text("💾 Foto guardada en la base.")

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN")

    init_db()

    # server dummy (opcional, no estorba)
    threading.Thread(target=run_server, daemon=True).start()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("buscar", buscar))
    dp.add_handler(MessageHandler(Filters.photo, photo_received))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
