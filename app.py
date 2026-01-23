import os
from telegram.ext import Updater, CommandHandler

BOT_TOKEN = os.environ.get("BOT_TOKEN")

def start(update, context):
    update.message.reply_text("👋 Ya quedó. Estoy vivo.")

def main():
    if not BOT_TOKEN:
        raise ValueError("Falta BOT_TOKEN en Environment Variables")

    updater = Updater(token=BOT_TOKEN, use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
