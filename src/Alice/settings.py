from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import logging
from pathlib import Path

# Configure logging


def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "alice.log"),
            logging.StreamHandler()
        ]
    )

    # Set up logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")

    return logger


# Initialize logging
logger = setup_logging()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_file_encoding="utf-8")

    GROQ_API_KEY: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str
    TOGETHER_API_KEY: str

    QDRANT_API_KEY: str | None
    QDRANT_URL: str
    QDRANT_PORT: str = "6333"
    QDRANT_HOST: str | None = None

    TEXT_MODEL_NAME: str = "llama-3.3-70b-versatile"
    SMALL_TEXT_MODEL_NAME: str = "gemma2-9b-it"
    STT_MODEL_NAME: str = "whisper-large-v3-turbo"
    TTS_MODEL_NAME: str = "eleven_flash_v2_5"
    TTI_MODEL_NAME: str = "black-forest-labs/FLUX.1-schnell-Free"
    ITT_MODEL_NAME: str = "meta-llama/llama-4-maverick-17b-128e-instruct"

    MEMORY_TOP_K: int = 3
    ROUTER_MESSAGES_TO_ANALYZE: int = 3
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5

    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_USER_ID: str
    TELEGRAM_URL: str = "https://api.telegram.org/bot"

    SHORT_TERM_MEMORY_DB_PATH: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../data/memory.db")


settings = Settings()
