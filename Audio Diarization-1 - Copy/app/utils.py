import os
from pathlib import Path
from fastapi import HTTPException

from app.config import settings
from app.logger import logger

def validate_audio_file(filename: str, file_size: int) -> None:
    """Validate audio file format and size."""
    file_ext = Path(filename).suffix.lower().lstrip(".")
    
    if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
        logger.warning(f"Invalid audio format: {file_ext} (file: {filename})")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(settings.ALLOWED_AUDIO_FORMATS)}"
        )
    
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        logger.warning(
            f"File too large: {filename} ({file_size / (1024*1024):.2f}MB > {settings.MAX_FILE_SIZE_MB}MB)"
        )
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    logger.info(f"Validated audio file format: {filename} ({file_size / (1024*1024):.2f}MB)")

def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary files."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {file_path}: {e}")