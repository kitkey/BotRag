import logging
import os
import re
import httpx
from telegram import Update, Chat
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


async def give_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat.type not in [Chat.GROUP, Chat.SUPERGROUP]:
        return

    text = " ".join(context.args) if context.args else "(нет текста)"
    pattern = r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]{11}"
    matches = re.findall(pattern, text)

    async with httpx.AsyncClient(timeout=httpx.Timeout(360.0)) as client:
        await client.post('http://rag_service:8000/get_text', json=matches)

    await update.message.reply_text(f"Принято: {matches}")

async def le_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat.type not in [Chat.GROUP, Chat.SUPERGROUP]:
        return

    text = " ".join(context.args) if context.args else "нет текста"
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        response = await client.post('http://rag_service:8000/retrieve_information', json=text)

    if response.status_code != 200:
        await update.message.reply_text("Ошибка")
        return

    await update.message.reply_text(response.text)


def main() -> None:
    application = Application.builder().token(os.getenv("BOT_TOKEN")).build()
    application.add_handler(CommandHandler("give", give_handler))
    application.add_handler(CommandHandler("le", le_handler))
    application.run_polling()

if __name__ == "__main__":
    main()
