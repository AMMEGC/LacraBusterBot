import os
import threading
from telegram.ext import Updater, CommandHandler

from fastapi import FastAPI
import uvicorn

BOT_TOKEN = os.environ.get("BOT_TOKEN")
app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

def start(update, context):
    update.message.reply_text("👋 Ya quedó. Estoy vivo.")

def run_bot():
    if not BOT_TOKEN:
        raise ValueError("Falta BOT_TOKEN en Environment Variables")

    updater = Updater(token=BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    # corre el bot en otro hilo
    threading.Thread(target=run_bot, daemon=True).start()

    # abre el puerto que Render necesita
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
