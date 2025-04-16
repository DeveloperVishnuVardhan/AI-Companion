"""
This module implements the FastAPI controller for the Alice AI system.
It handles Telegram webhook integration, message processing, and response generation.
The controller manages the communication between Telegram and the AI system's graph-based processing pipeline.
"""

# fmt: off
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)

from Alice.settings import settings
from Alice.settings import logger
from Alice.interfaces.fastApi.services import ProcessAudioService, ProcessImageService
from Alice.graph import graph_builder
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from fastapi import FastAPI, Request, HTTPException
import requests
import logging
# fmt: on

app = FastAPI(title="telegram bot")

logger = logging.getLogger(__name__)

BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
WEBHOOK_URL = "https://f743-2601-19b-200-210-bc69-9a12-ed57-76d4.ngrok-free.app/webhook"
TELEGRAM_URL = settings.TELEGRAM_URL + BOT_TOKEN
TELEGRAM_USER_ID = settings.TELEGRAM_USER_ID


@app.get("/set-webhook")
def set_webhook_endpoint():
    """
    Endpoint to setup the webhook for the bot.

    This endpoint configures Telegram to send updates to our webhook URL.
    It's used during the initial setup of the bot.

    Returns:
        dict: The response from Telegram's API
    """
    url = f"{TELEGRAM_URL}/setWebhook"
    payload = {"url": WEBHOOK_URL}
    response = requests.post(url, data=payload)
    if response.ok:
        logger.info(f"Webhook set successfully")
    else:
        logger.error(f"Failed to set webhook: {response.json()}")
    return response.json()


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """
    Endpoint to receive updates from Telegram.
    It processes incoming messages and sends responses using the Agentic graph system.

    This endpoint:
    1. Validates the sender
    2. Processes different message types (text, photo, voice)
    3. Routes the message through the AI system
    4. Sends appropriate responses back to Telegram

    Args:
        request (Request): The incoming webhook request from Telegram

    Returns:
        dict: Status of the operation
    """
    update = await request.json()
    logger.debug(f"Received update: {update}")

    if "message" not in update:
        return {"ok": True}

    # Check if the message is from the authorized user
    user_id = str(update["message"]["from"]["id"])
    if user_id != TELEGRAM_USER_ID:
        logger.warning(f"Unauthorized access attempt from user ID: {user_id}")
        raise HTTPException(status_code=403, detail="Unauthorized access")

    chat_id = update["message"]["chat"]["id"]
    message_content = ""

    if "text" in update["message"]:
        logger.info(f"Received text message: {update['message']['text']}")
        message_content += update["message"]["text"]

    if "photo" in update["message"]:
        logger.info(f"Received photo message: {update['message']['photo']}")
        # Get the latest photo (last in the array)
        photo = update["message"]["photo"][-1]
        image_analysis = await ProcessImageService().process_media(
            chat_id=chat_id, media_id=photo["file_id"]
        )
        if image_analysis:
            message_content += image_analysis

    # Handle both voice messages and audio file messages
    if "voice" in update["message"]:
        logger.info(f"Received voice message: {update['message']['voice']}")
        voice = update["message"]["voice"]
        transcription = await ProcessAudioService().process_media(
            chat_id=chat_id, media_id=voice["file_id"]
        )
        if transcription:
            message_content += transcription

    if not message_content:
        return {"ok": True}

    # Process through graph with message content
    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        graph = graph_builder.compile(checkpointer=short_term_memory)
        output_state = await graph.ainvoke(
            {"messages": [HumanMessage(content=message_content)]},
            {"configurable": {"thread_id": chat_id}},
        )

    # Extract workflow type and response message
    workflow = output_state.get("workflow", "conversation")
    response_message = output_state["messages"][-1].content

    logger.info(f"Sending response with workflow type: {workflow}")

    try:
        # Handle different response types based on workflow
        if workflow == "audio":
            if "audio_buffer" in output_state:
                await send_audio_response(chat_id, response_message, output_state["audio_buffer"])
            else:
                logger.warning(
                    "Audio workflow but no audio_buffer found in output state")
                await send_text_response(chat_id, response_message)

        elif workflow == "image":
            if "image_path" in output_state:
                with open(output_state["image_path"], "rb") as img_file:
                    await send_image_response(chat_id, response_message, img_file.read())
            else:
                logger.warning(
                    "Image workflow but no image_path found in output state")
                await send_text_response(chat_id, response_message)

        else:  # Default to conversation workflow (text response)
            await send_text_response(chat_id, response_message)

        return {"ok": True}

    except Exception as e:
        logger.error(f"Error sending response to Telegram: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


async def send_text_response(chat_id: int, message: str) -> bool:
    """
    Send a text message to a Telegram chat.

    Args:
        chat_id (int): The Telegram chat ID to send the message to
        message (str): The text message to send

    Returns:
        bool: True if the message was sent successfully, False otherwise
    """
    try:
        payload = {"chat_id": chat_id, "text": message}
        response = requests.post(f"{TELEGRAM_URL}/sendMessage", json=payload)
        if not response.ok:
            logger.error(f"Failed to send text message: {response.json()}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error sending text message: {e}")
        return False


async def send_image_response(chat_id: int, caption: str, image_data: bytes) -> bool:
    """
    Send an image with caption to a Telegram chat.

    Args:
        chat_id (int): The Telegram chat ID to send the image to
        caption (str): The caption to accompany the image
        image_data (bytes): The binary image data

    Returns:
        bool: True if the image was sent successfully, False otherwise
    """
    try:
        files = {"photo": image_data}
        payload = {"chat_id": chat_id, "caption": caption}
        response = requests.post(
            f"{TELEGRAM_URL}/sendPhoto", data=payload, files=files)
        if not response.ok:
            logger.error(f"Failed to send image: {response.json()}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error sending image: {e}")
        return False


async def send_audio_response(chat_id: int, caption: str, audio_data: bytes) -> bool:
    """
    Send an audio message with caption to a Telegram chat.

    Args:
        chat_id (int): The Telegram chat ID to send the audio to
        caption (str): The caption to accompany the audio
        audio_data (bytes): The binary audio data

    Returns:
        bool: True if the audio was sent successfully, False otherwise
    """
    try:
        files = {"voice": audio_data}
        payload = {"chat_id": chat_id, "caption": caption}
        response = requests.post(
            f"{TELEGRAM_URL}/sendVoice", data=payload, files=files)
        if not response.ok:
            logger.error(f"Failed to send audio: {response.json()}")
            # Send caption as text if audio sending fails
            await send_text_response(chat_id, caption)
            return False
        return True
    except Exception as e:
        logger.error(f"Error sending audio: {e}")
        # Send caption as text if audio sending fails
        await send_text_response(chat_id, caption)
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("controller:app", host="0.0.0.0", port=8000)
