import uvicorn
from app.config import settings
from app.logger import logger

if not settings.HF_TOKEN:
    logger.error("HF_TOKEN not set — diarization and transcription will fail.")
    raise ValueError("Missing HF_TOKEN in environment variables. App cannot start.")

if not settings.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set — summarization and LLM features will fail.")
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

logger.info(f"App starting in '{settings.ENVIRONMENT}' mode")
logger.info(f"Using device: {settings.DEVICE}, model: {settings.MODEL_NAME}")
logger.info(f"Temporary directory: {settings.TEMP_DIR}")
logger.info(f"Log level set to: {settings.LOG_LEVEL.upper()}")
logger.info("Configuration loaded successfully.")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1
    )