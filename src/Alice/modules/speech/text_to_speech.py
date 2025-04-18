"""Text-to-speech conversion module using ElevenLabs.

This module provides functionality to convert text to speech using ElevenLabs' API.
It handles text processing, voice configuration, and audio generation for speech synthesis tasks.
"""

# fmt: off
import os
from typing import Optional

from Alice.core.exceptions import TextToSpeechError
from Alice.settings import settings
from elevenlabs import ElevenLabs, Voice, VoiceSettings

# fmt: on


class TextToSpeech:
    """A class to handle text-to-speech conversion using ElevenLabs.

    This class provides methods to convert text to speech using ElevenLabs' API.
    It handles environment variable validation, client initialization, and voice configuration.
    The class implements a singleton pattern for the ElevenLabs client to optimize resource usage.

    Attributes:
        REQUIRED_ENV_VARS (list): List of required environment variables
        _client (Optional[ElevenLabs]): ElevenLabs client instance

    Methods:
        synthesize: Convert text to speech
        _validate_env_vars: Validate required environment variables
        client: Get or create ElevenLabs client instance
    """

    # Required environment variables
    REQUIRED_ENV_VARS = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]

    def __init__(self):
        """Initialize the TextToSpeech class and validate environment variables."""
        self._validate_env_vars()
        self._client: Optional[ElevenLabs] = None

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [
            var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def client(self) -> ElevenLabs:
        """Get or create ElevenLabs client instance using singleton pattern."""
        if self._client is None:
            self._client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        return self._client

    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech using ElevenLabs.

        Args:
            text: Text to convert to speech

        Returns:
            bytes: Audio data

        Raises:
            ValueError: If the input text is empty or too long
            TextToSpeechError: If the text-to-speech conversion fails
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        if len(text) > 5000:  # ElevenLabs typical limit
            raise ValueError(
                "Input text exceeds maximum length of 5000 characters")

        try:
            audio_generator = self.client.generate(
                text=text,
                voice=Voice(
                    voice_id=settings.ELEVENLABS_VOICE_ID,
                    settings=VoiceSettings(
                        stability=0.5, similarity_boost=0.5),
                ),
                model=settings.TTS_MODEL_NAME,
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)
            if not audio_bytes:
                raise TextToSpeechError("Generated audio is empty")

            return audio_bytes

        except Exception as e:
            raise TextToSpeechError(
                f"Text-to-speech conversion failed: {str(e)}") from e
