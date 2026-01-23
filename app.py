import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

BOT_TOKEN = os.environ.get("BOT_TOKEN")

# --- Servidor dummy para Render ---
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

# --- Bot ---
def start(update, context):
    update.message.reply_text("🤖 Bot activo. Manda una foto.")

def photo_received(update, context):
    update.message.reply_text("📸 Foto recibida.")

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN")

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, photo_received))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    main()
