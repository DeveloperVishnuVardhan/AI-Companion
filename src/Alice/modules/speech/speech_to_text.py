"""Speech-to-text conversion module using Groq's Whisper model.

This module provides functionality to convert speech audio data to text using Groq's Whisper model.
It handles audio file processing, API communication, and error handling for speech recognition tasks.
"""

# fmt: off
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)

import os
import tempfile
from typing import Optional

from Alice.core.exceptions import SpeechToTextError
from Alice.settings import settings
from groq import Groq

# fmt: on


class SpeechToText:
    """A class to handle speech-to-text conversion using Groq's Whisper model.

    This class provides methods to convert audio data to text using Groq's Whisper model.
    It handles environment variable validation, client initialization, and audio processing.
    The class implements a singleton pattern for the Groq client to optimize resource usage.

    Attributes:
        REQUIRED_ENV_VARS (list): List of required environment variables
        _client (Optional[Groq]): Groq client instance

    Methods:
        transcribe: Convert audio data to text
        _validate_env_vars: Validate required environment variables
        client: Get or create Groq client instance
    """

    # Required environment variables
    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

    def __init__(self):
        """Initialize the SpeechToText class and validate environment variables."""
        self._validate_env_vars()
        self._client: Optional[Groq] = None

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [
            var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def client(self) -> Groq:
        """Get or create Groq client instance using singleton pattern."""
        if self._client is None:
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        return self._client

    async def transcribe(self, audio_data: bytes) -> str:
        """Convert speech to text using Groq's Whisper model.

        Args:
            audio_data: Binary audio data

        Returns:
            str: Transcribed text

        Raises:
            ValueError: If the audio file is empty or invalid
            RuntimeError: If the transcription fails
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")

        try:
            # Create a temporary file with .wav extension
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            try:
                # Open the temporary file for the API request
                with open(temp_file_path, "rb") as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3-turbo",
                        language="en",
                        response_format="text",
                    )

                if not transcription:
                    raise SpeechToTextError("Transcription result is empty")

                return transcription

            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)

        except Exception as e:
            raise SpeechToTextError(
                f"Speech-to-text conversion failed: {str(e)}") from e
