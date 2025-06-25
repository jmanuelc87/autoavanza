import os
import locale
import logging
import dotenv

from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
)

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from application.validation import get_validation_pipeline

dotenv.load_dotenv(dotenv.find_dotenv())


locale.setlocale(locale.LC_ALL, "")

API_TOKEN = os.environ.get("TELEGRAM_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Initialize the LLM and state machines
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, streaming=False)
system_prompt = "Eres un asistente servicial llamado AutoMate y te refieres al usuario por su nombre."
chat_map = {}
user_state = {}
user_values = {}
chat_prompts = {
    "START": f"{system_prompt} pidele la foto frontal de su INE",
    "INE_FRONT": f"{system_prompt} pidele la foto frontal de su tarjeta de circulacion",
    "TARJETA_CIRCULACION_FRONT": f"{system_prompt} pidele la foto de la factura",
    "FACTURA_FRONT": f"{system_prompt} pidele el kilometraje de su vehiculo",
    "ODOMETRO": f"{system_prompt} pidele que espere unos minutos en lo que se calcula su oferta",
}
chat_states = {
    "START": "INE_FRONT",
    "INE_FRONT": "TARJETA_CIRCULACION_FRONT",
    "TARJETA_CIRCULACION_FRONT": "FACTURA_FRONT",
    "FACTURA_FRONT": "ODOMETRO",
    "ODOMETRO": "END",
}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]


def get_llm_on_user_id(user_id):
    state = user_state[user_id]
    pipeline = None
    if state != "ODOMETRO":
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(chat_prompts[state]),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )
        pipeline = RunnableWithMessageHistory(
            prompt_template | llm,
            get_session_history=get_chat_history,
            input_messages_key="query",
            history_messages_key="history",
        )
    else:
        pipeline = get_validation_pipeline()

    # advance to the next state
    user_state[user_id] = chat_states[state]
    return pipeline


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Reset any prior history for this user
    uid = update.effective_user.id
    if uid in chat_map:
        del chat_map[uid]
    user_state[uid] = "START"
    user_values[uid] = defaultdict(str)
    await update.message.reply_text(
        "¡Bienvenido al simulador AutoAvanza!\nPor favor dime tu nombre!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    pipeline = get_llm_on_user_id(uid)

    if user_state[uid] == "END":
        await update.message.reply_text(
            "Por favor espera un minuto en lo que calculamos tu oferta."
        )

        result = pipeline.invoke(
            input={
                "factura": user_values[uid]["FACTURA_FRONT"],
                "tc": user_values[uid]["TARJETA_CIRCULACION_FRONT"],
                "ine": user_values[uid]["INE_FRONT"],
                "odometro": int(update.message.text),
            }
        )
        await update.message.reply_text(
            f"Tu dictamen fue {result['dictamen_documental']}"
        )

        if result["dictamen_documental"] == "Aprobado":
            amount = locale.currency(result["value"], grouping=True)
            await update.message.reply_text(
                f"Te prestamos hasta {amount} del valor de tu coche."
            )
    else:
        result = pipeline.invoke(
            {"query": update.message.text},
            config={"session_id": uid},
        )
        await update.message.reply_text(result.content)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    # Check if the message contains photo(s)
    if update.message.photo:
        # Get the highest-resolution photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        path = os.path.join(
            "./UI/telegram_bot/received_images/", str(uid), user_state[uid] + ".jpg"
        )
        os.makedirs(
            os.path.join("./UI/telegram_bot/received_images/", str(uid)), exist_ok=True
        )
        await file.download_to_drive(path)

        user_values[uid][user_state[uid]] = path
        llm = get_llm_on_user_id(uid)
        result = llm.invoke(
            {"query": "ok"},
            config={"session_id": uid},
        )
        await update.message.reply_text(result.content)
    else:
        await update.message.reply_text("No image found in your message.")


if __name__ == "__main__":
    application = ApplicationBuilder().token(API_TOKEN).build()

    # /start command
    application.add_handler(CommandHandler("start", start))
    # Text messages
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    # Photo attachments
    application.add_handler(
        MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo)
    )

    application.run_polling()
