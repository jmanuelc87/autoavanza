import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


API_TOKEN = "7928982487:AAGXFhX5Rh4td6Ehg4allA04V4wc2L7j_gs"


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am a simple bot.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! again")


if __name__ == "__main__":
    application = ApplicationBuilder().token(API_TOKEN).build()

    start_handler = CommandHandler("start", start)
    application.add_handler(start_handler)

    handle_message_handler = MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message
    )
    application.add_handler(handle_message_handler)

    application.run_polling()
