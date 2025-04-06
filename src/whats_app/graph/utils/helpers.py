# fmt: off
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from whats_app.modules.image.image_to_text import ImageToText
from whats_app.modules.image.text_to_image import TextToImage
from whats_app.modules.speech.text_to_speech import TextToSpeech
from whats_app.settings import settings

# fmt: on


def get_chat_model(temperature: float = 0.7):
    return ChatGroq(
        model_name=settings.TEXT_MODEL_NAME,
        temperature=temperature,
        api_key=settings.GROQ_API_KEY,
    )


def get_text_to_speech_model():
    return TextToSpeech()


def get_text_to_image_model():
    return TextToImage()


def get_image_to_text_model():
    return ImageToText()


def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text):
        return remove_asterisk_content(super().parse(text))
