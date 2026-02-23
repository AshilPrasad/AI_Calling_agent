from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.logger import logger

def setup_error_handlers(app: FastAPI):
    """Configure error handlers."""
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        logger.error(f"Validation error: {str(exc)}")
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)}
        )