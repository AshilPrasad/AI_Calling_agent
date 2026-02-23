import os
import tempfile
from typing import List, Tuple

import dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

dotenv.load_dotenv()

class Settings(BaseSettings):
    """Application configuration with validation."""

    HF_TOKEN: str
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    DEVICE: str = "cpu"
    MODEL_NAME: str = "large-v3"
    COMPUTE_TYPE: str = "int8"
    MAX_FILE_SIZE_MB: int = 500
    ALLOWED_AUDIO_FORMATS: Tuple[str, ...] = ("mp3","mp4","wav", "m4a", "ogg", "flac","amr")
    TEMP_DIR: str = tempfile.gettempdir()
    CORS_ORIGINS: List[str] = ["*"]
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"
    
    # Performance optimization settings
    BATCH_SIZE: int = 8
    ENABLE_TRANSLATION: bool = True
    TRANSLATION_BATCH_SIZE: int = 4
    MAX_WORKERS: int = 1

    model_config = ConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()