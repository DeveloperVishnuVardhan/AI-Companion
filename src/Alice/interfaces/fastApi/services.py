"""
This module provides services for processing media content (images and audio) received through Telegram.
It handles downloading, analyzing, and transcribing media files using the appropriate AI models.
"""

# fmt: off
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)

from abc import ABC, abstractmethod
import requests
import logging
from Alice.settings import logger
from Alice.settings import settings
from Alice.modules.image import ImageToText
from Alice.modules.speech import SpeechToText
# fmt: on
BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
TELEGRAM_URL = settings.TELEGRAM_URL + BOT_TOKEN

image_to_text = ImageToText()
speech_to_text = SpeechToText()

logger = logging.getLogger(__name__)


class ProcessMediaService(ABC):
    """
    Abstract base class for media processing services.
    Defines the interface for processing different types of media content.
    """
    @abstractmethod
    async def process_media(self, chat_id: int, media_id: str) -> str:
        """
        Process media content received from Telegram.

        Args:
            chat_id (int): The Telegram chat ID
            media_id (str): The Telegram file ID of the media

        Returns:
            str: Processed content (e.g., image description or audio transcription)
        """
        pass


class ProcessImageService(ProcessMediaService):
    """
    Service for processing images received through Telegram.
    Downloads the image and uses AI to analyze its content.
    """

    async def process_media(self, chat_id: int, media_id: str) -> str:
        """
        Process an image sent by the user.

        This method:
        1. Downloads the image from Telegram
        2. Uses AI to analyze the image content
        3. Returns a description of the image

        Args:
            chat_id (int): The Telegram chat ID
            media_id (str): The Telegram file ID of the image

        Returns:
            str: A description of the image content, or None if processing fails
        """
        try:
            # Get the file path from Telegram
            file_info = requests.get(
                f"{TELEGRAM_URL}/getFile", params={"file_id": media_id}).json()
            logger.debug(f"File info response: {file_info}")

            if not file_info.get("ok"):
                logger.error(f"Failed to get file info: {file_info}")
                return None

            file_path = file_info["result"]["file_path"]
            # Correct URL format for downloading files from Telegram
            file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
            logger.debug(f"Downloading image from: {file_url}")

            # Download the image
            image_response = requests.get(file_url)
            if not image_response.ok:
                logger.error(
                    f"Failed to download image: {image_response.status_code}")
                return None

            # Analyze the image
            logger.info(f"Analyzing image")
            description = await image_to_text.analyze_image(
                image_data=image_response.content,
                prompt="please describe what you see in this image in the context of our conversation",
            )
            return f"\n[Image Analysis: {description}]"
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None


class ProcessAudioService(ProcessMediaService):
    """
    Service for processing audio messages received through Telegram.
    Downloads the audio and transcribes it using speech-to-text AI.
    """

    async def process_media(self, chat_id: int, media_id: str) -> str:
        """
        Process an audio message sent by the user.

        This method:
        1. Downloads the audio from Telegram
        2. Uses AI to transcribe the audio content
        3. Returns the transcription

        Args:
            chat_id (int): The Telegram chat ID
            media_id (str): The Telegram file ID of the audio

        Returns:
            str: The transcribed text, or None if processing fails
        """
        try:
            file_info = requests.get(
                f"{TELEGRAM_URL}/getFile", params={"file_id": media_id}).json()
            logger.debug(f"File info response: {file_info}")

            if not file_info.get("ok"):
                logger.error(f"Failed to get file info: {file_info}")
                return None

            file_path = file_info["result"]["file_path"]
            # Correct URL format for downloading files from Telegram
            file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
            logger.debug(f"Downloading audio from: {file_url}")

            # Download the audio
            audio_response = requests.get(file_url)
            if not audio_response.ok:
                logger.error(
                    f"Failed to download audio: {audio_response.status_code}")
                return None

            # Transcribe the audio
            logger.info(f"Transcribing audio")
            transcription = await speech_to_text.transcribe(audio_response.content)
            logger.info(f"Transcription result: {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return None
