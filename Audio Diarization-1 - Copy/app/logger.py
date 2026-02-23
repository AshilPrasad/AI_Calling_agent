import sys
import logging
from logging.handlers import RotatingFileHandler

from app.config import settings

def setup_logging():
    """Configure application logging."""
    numeric_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    file_handler = RotatingFileHandler(
        "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler
        ],
    )

    logger = logging.getLogger(__name__)

    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger

logger = setup_logging()