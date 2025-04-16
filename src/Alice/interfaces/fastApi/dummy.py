# fmt: off
import sys
import os
import io
import base64

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)

from Alice.settings import settings
from Alice.graph import graph_builder
from Alice.modules.image import ImageToText
from Alice.modules.speech import TextToSpeech, SpeechToText
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from fastapi import FastAPI, Request
import requests

# fmt: on

app = FastAPI(title="telegram bot")

BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
WEBHOOK_URL = "https://12d2-2601-19b-200-210-c035-5e8b-4aff-c3d8.ngrok-free.app/webhook"
TELEGRAM_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Initialize global modules
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()


def set_webhook(bot_token: str, webhook_url: str):
    """Set the webhook for the bot
    Args:
        bot_token: The token for the bot
        webhook_url: The url for the webhook
    """
    url = f"{TELEGRAM_URL}/setWebhook"
    payload = {"url": webhook_url}
    response = requests.post(url, data=payload)
    if response.ok:
        print("Webhook set successfully")
    else:
        print(f"Failed to set webhook: {response.json()}")
    return response.json()


@app.get("/set-webhook")
def set_webhook_endpoint():
    """
    Endpoint to trigger the webhook setup.
    When you access this route, it calls the set_webhook() function.
    """
    result = set_webhook(BOT_TOKEN, WEBHOOK_URL)
    return result


async def process_image(chat_id: int, file_id: str):
    """Process an image sent by the user"""
    # Get the file path from Telegram
    file_info = requests.get(
        f"{TELEGRAM_URL}/getFile", params={"file_id": file_id}).json()
    if not file_info.get("ok"):
        return None

    file_path = file_info["result"]["file_path"]
    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

    # Download the image
    image_response = requests.get(file_url)
    if not image_response.ok:
        return None

    # Analyze the image
    try:
        description = await image_to_text.analyze_image(
            image_response.content,
            "Please describe what you see in this image in the context of our conversation.",
        )
        return f"\n[Image Analysis: {description}]"
    except Exception as e:
        print(f"Failed to analyze image: {e}")
        return None


async def process_audio(chat_id: int, file_id: str):
    """Process an audio message sent by the user"""
    # Get the file path from Telegram
    file_info = requests.get(
        f"{TELEGRAM_URL}/getFile", params={"file_id": file_id}).json()
    if not file_info.get("ok"):
        return None

    file_path = file_info["result"]["file_path"]
    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

    # Download the audio
    audio_response = requests.get(file_url)
    if not audio_response.ok:
        return None

    # Transcribe the audio
    try:
        transcription = await speech_to_text.transcribe(audio_response.content)
        return transcription
    except Exception as e:
        print(f"Failed to transcribe audio: {e}")
        return None


async def send_audio_response(chat_id: int, text: str):
    """Send an audio response to the user"""
    try:
        # Generate audio from text
        audio_buffer = await text_to_speech.synthesize(text)

        # Send the audio file
        files = {'audio': ('response.mp3', audio_buffer, 'audio/mpeg')}
        payload = {'chat_id': chat_id}
        response = requests.post(
            f"{TELEGRAM_URL}/sendAudio", data=payload, files=files)
        return response.ok
    except Exception as e:
        print(f"Failed to send audio response: {e}")
        return False


async def send_image_response(chat_id: int, image_path: str):
    """Send an image response to the user"""
    try:
        with open(image_path, 'rb') as image_file:
            files = {'photo': image_file}
            payload = {'chat_id': chat_id}
            response = requests.post(
                f"{TELEGRAM_URL}/sendPhoto", data=payload, files=files)
            return response.ok
    except Exception as e:
        print(f"Failed to send image response: {e}")
        return False


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """
    Endpoint to receive updates from Telegram.
    It processes incoming messages and sends responses using the graph system.
    """
    update = await request.json()
    print("Received update:", update)

    if "message" not in update:
        return {"ok": True}

    chat_id = update["message"]["chat"]["id"]
    message_content = ""

    # Process text message
    if "text" in update["message"]:
        message_content = update["message"]["text"]

    # Process image
    elif "photo" in update["message"]:
        # Get the largest photo (last in the array)
        photo = update["message"]["photo"][-1]
        image_analysis = await process_image(chat_id, photo["file_id"])
        if image_analysis:
            message_content = image_analysis

    # Process audio/voice message
    elif "voice" in update["message"]:
        voice = update["message"]["voice"]
        transcription = await process_audio(chat_id, voice["file_id"])
        if transcription:
            message_content = transcription

    if not message_content:
        return {"ok": True}

    # Process through graph with message content
    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        graph = graph_builder.compile(checkpointer=short_term_memory)
        output_state = await graph.ainvoke(
            {"messages": [HumanMessage(content=message_content)]},
            {"configurable": {"thread_id": chat_id}},
        )

    # Get the response from the graph output
    response_text = output_state["messages"][-1].content
    workflow = output_state.get("workflow", "conversation")
    print(f"Workflow: {workflow}")

    # Handle different types of responses based on workflow
    if workflow == "image":
        # Send image response
        if "image_path" in output_state:
            await send_image_response(chat_id, output_state["image_path"])
        # Also send the text response
        payload = {"chat_id": chat_id, "text": response_text}
        requests.post(f"{TELEGRAM_URL}/sendMessage", json=payload)
    elif workflow == "audio":
        # Send audio response
        await send_audio_response(chat_id, response_text)
    else:  # conversation workflow
        # Send text response
        payload = {"chat_id": chat_id, "text": response_text}
        response = requests.post(f"{TELEGRAM_URL}/sendMessage", json=payload)
        print("Sent message, Telegram API response:", response.text)

    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
