import os
import asyncio
from fastapi import FastAPI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")

app = FastAPI()
tg_app = None

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Ya quedé prendido 😄")

@app.get("/")
def root():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    global tg_app
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN en Render")

    tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
    tg_app.add_handler(CommandHandler("start", start_cmd))

    await tg_app.initialize()
    await tg_app.start()
    asyncio.create_task(tg_app.updater.start_polling())

@app.on_event("shutdown")
async def shutdown_event():
    if tg_app:
        await tg_app.updater.stop()
        await tg_app.stop()
        await tg_app.shutdown()
