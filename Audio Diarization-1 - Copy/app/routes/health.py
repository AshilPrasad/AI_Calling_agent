from fastapi import APIRouter

from app.schemas import HealthResponse
from app.config import settings
from app.models_cache import models_cache

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        models_loaded=models_cache.transcription_model is not None
    )