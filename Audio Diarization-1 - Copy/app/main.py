from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logger import logger
from app.services.transcription import load_models, cleanup_models
from app.models_cache import models_cache
from app.error_handlers import setup_error_handlers
from app.routes import health, audio

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown."""
    logger.info("Starting application...")
    try:
        logger.info("Preloading models...")
        load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload models: {e}")
    
    yield
    
    logger.info("Shutting down application...")
    cleanup_models()

def create_app():
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Audio Diarization & Summarization API",
        description="Production-ready API for audio transcription, diarization, and summarization",
        version="1.0.0",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    setup_error_handlers(app)

    app.include_router(health.router)
    app.include_router(audio.router)

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Audio Diarization & Summarization API",
            "docs": "/docs",
            "health": "/health"
        }

    return app

app = create_app()

