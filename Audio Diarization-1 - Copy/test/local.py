# ========================== STANDARD LIBRARIES ==========================
import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

# ========================== THIRD-PARTY LIBRARIES ==========================
import dotenv
import whisperx
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ConfigDict
from pydantic_settings import BaseSettings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence



# ========================== LOAD ENV ==========================


dotenv.load_dotenv()  # Load environment variables from .env file

# ========================== CONFIG ==========================


class Settings(BaseSettings):
    """Application configuration with validation."""

    HF_TOKEN: str
    OPENAI_API_KEY: str
    DEVICE: str = "cpu"
    MODEL_NAME: str = "small"
    COMPUTE_TYPE: str = "int8"
    MAX_FILE_SIZE_MB: int = 500
    ALLOWED_AUDIO_FORMATS: Tuple[str, ...] = ("mp3", "wav", "m4a", "ogg", "flac")
    TEMP_DIR: str = tempfile.gettempdir()
    CORS_ORIGINS: List[str] = ["*"]
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"  # Read from .env for dynamic logging

    # Pydantic V2 style config
    model_config = ConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()


# ========================== LOGGING SETUP ==========================


# Convert string log level from Settings to numeric
numeric_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

# Create rotating file handler
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")
file_handler = RotatingFileHandler(log_file,mode="w", maxBytes=5_000_000, backupCount=0, encoding="utf-8")

# Common log format
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configure global logging
logging.basicConfig(
    level=numeric_level,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        file_handler
    ],
)

# Main logger
logger = logging.getLogger(__name__)

# Quiet noisy libraries
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ========================== VALIDATE REQUIRED ENV ==========================
if not settings.HF_TOKEN:
    logger.error("HF_TOKEN not set — diarization and transcription will fail.")
    raise ValueError("Missing HF_TOKEN in environment variables. App cannot start.")

if not settings.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set — summarization and LLM features will fail.")
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

# ========================== STARTUP INFO ==========================
logger.info(f"App starting in '{settings.ENVIRONMENT}' mode")
logger.info(f"Using device: {settings.DEVICE}, model: {settings.MODEL_NAME}")
logger.info(f"Temporary directory: {settings.TEMP_DIR}")
logger.info(f"Log level set to: {settings.LOG_LEVEL.upper()}")
logger.info("Configuration loaded successfully.")


# ========================== SCHEMAS ==========================
class DialogueSegment(BaseModel):
    speaker: str = Field(..., min_length=1)
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    
    @field_validator("end")
    def validate_end_time(cls, v, info: ValidationInfo):
        start_value = info.data.get("start")
        if start_value is not None and v <= start_value:
            raise ValueError("end time must be greater than start time")
        return v

class SummaryResponse(BaseModel):
    summary_paragraph: str = Field(..., min_length=20)
    key_topics: List[str] = Field(default_factory=list, max_items=20)
    tone: str = Field(..., min_length=2)

class ConversationResponse(BaseModel):
    dialogue: List[DialogueSegment]
    summary: SummaryResponse

class HealthResponse(BaseModel):
    status: str
    environment: str
    models_loaded: bool


# ========================== MODELS CACHE ==========================
class ModelsCache:
    """Cache loaded models to avoid reloading."""
    def __init__(self):
        self.transcription_model = None
        self.diarization_model = None
        self.align_model = None
        self.align_metadata = None

models_cache = ModelsCache()

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

def load_models():
    """Preload models at startup."""
    try:
        if models_cache.transcription_model is None:
            logger.info(f"Loading WhisperX model '{settings.MODEL_NAME}'...")
            models_cache.transcription_model = whisperx.load_model(
                settings.MODEL_NAME,
                settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE
            )
    except Exception as e:
        logger.error(f"Failed to load transcription model: {e}")
        raise

def cleanup_models():
    """Clean up model resources."""
    try:
        models_cache.transcription_model = None
        models_cache.diarization_model = None
        models_cache.align_model = None
        models_cache.align_metadata = None
        logger.info("Models cleaned up")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# ========================== FASTAPI APP ==========================
app = FastAPI(
    title="Audio Diarization & Summarization API",
    description="Production-ready API for audio transcription, diarization, and summarization",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================== UTILS ==========================
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



def transcribe_and_diarize(audio_path: str) -> List[Dict[str, Any]]:
    """Perform transcription + diarization using WhisperX."""
    try:
        if models_cache.transcription_model is None:
            load_models()
        
        logger.info(f"Loading audio from {audio_path}")
        audio = whisperx.load_audio(audio_path)

        logger.info("Transcribing audio...")
        result = models_cache.transcription_model.transcribe(audio, batch_size=8)
        
        if not result.get("segments"):
            logger.warning("No speech detected in audio")
            return []

        logger.info("Aligning words...")
        try:
            align_model, metadata = whisperx.load_align_model(result["language"], settings.DEVICE)
            aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, settings.DEVICE)
        except Exception as e:
            logger.warning(f"Alignment failed, using raw segments. Exception: {e}")
            aligned_result = result

        logger.info("Performing diarization...")
        try:
            if models_cache.diarization_model is None:
                models_cache.diarization_model = whisperx.diarize.DiarizationPipeline(
                    use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None,
                    device=settings.DEVICE
                )
            diarize_segments = models_cache.diarization_model(audio, min_speakers=2, max_speakers=5)
            final = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        except Exception as e:
            logger.warning(f"Diarization failed, using single speaker. Exception: {e}")
            final = aligned_result
            for seg in final.get("segments", []):
                seg["speaker"] = "Speaker 1"

        dialogue = [
            {
                "speaker": seg.get("speaker", "Unknown"),
                "start": round(float(seg.get("start", 0)), 2),
                "end": round(float(seg.get("end", 0)), 2),
                "text": str(seg.get("text", "")).strip()
            }
            for seg in final.get("segments", [])
            if seg.get("text", "").strip()
        ]

        dialogue = sorted(dialogue, key=lambda x: x["start"])
        logger.info(f"✅ Transcription complete: {len(dialogue)} segments")
        return dialogue

    except Exception as e:
        logger.error(f"WhisperX processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")




def summarize_with_langchain(dialogue_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate structured conversation summary using LLM."""
    if not dialogue_segments:
        return {
            "summary_paragraph": "No dialogue detected in the audio.",
            "key_topics": [],
            "tone": "N/A"
        }

    conversation_text = "\n".join(
        [f"{seg['speaker']}: {seg['text']}" for seg in dialogue_segments]
    )

    prompt_template = """Your task is to read the following multi-speaker conversation and produce a clear, human-like summary.

                        Please provide:
                        1. Write one clear, cohesive paragraph that naturally summarizes the entire multi-speaker conversation, capturing its main points and flow.
                        2. A concise list of up to five key topics discussed in the conversation. These should be the main *subjects* or *themes*, not tones or opinions.
                        3. The overall tone or mood of the conversation, described in a few words.

                        Format response as:

                        SUMMARY:
                        <paragraph>

                        KEY TOPICS:
                        - Topic 1
                        - Topic 2

                        TONE:
                        <tone or mood>

                        Conversation:
                        {conversation}
                        """

    try:
        prompt = PromptTemplate(
            input_variables=["conversation"],
            template=prompt_template
        )

        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0.4,
            timeout=30
        )

        chain = prompt | llm
        logger.info("Generating summary with LLM...")
        output_text = chain.invoke({"conversation": conversation_text}).content

        # Parse structured fields
        summary_paragraph = ""
        key_topics = []
        tone = ""

        lines = output_text.strip().splitlines()
        mode = None
        for line in lines:
            if line.startswith("SUMMARY:"):
                mode = "summary"
                continue
            elif line.startswith("KEY TOPICS:"):
                mode = "topics"
                continue
            elif line.startswith("TONE:"):
                mode = "tone"
                continue

            if mode == "summary" and line.strip():
                summary_paragraph += line.strip() + " "
            elif mode == "topics" and line.strip().startswith("-"):
                topic = line.strip("- ").strip()
                if topic:
                    key_topics.append(topic)
            elif mode == "tone" and line.strip():
                tone += line.strip() + " "

        return {
            "summary_paragraph": summary_paragraph.strip() or "Unable to generate summary.",
            "key_topics": key_topics[:5],
            "tone": tone.strip() or "Neutral"
        }

    except Exception as e:
        logger.error(f"LLM summarization failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary files."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {file_path}: {e}")



# ========================== API ROUTES ==========================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        models_loaded=models_cache.transcription_model is not None
    )

@app.post("/process-audio", response_model=ConversationResponse)
async def process_audio(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload an audio file → diarization + summary.
    
    Supported formats: mp3, wav, m4a, ogg, flac
    Max file size: 500MB
    """
    tmp_path = None
    try:
        # Validate file
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        validate_audio_file(file.filename, file_size)
        logger.info(f"Processing audio file: {file.filename} ({file_size / 1024 / 1024:.2f}MB)")

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Process audio
        logger.info("Starting transcription and diarization...")
        dialogue = transcribe_and_diarize(tmp_path)
        
        if not dialogue:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in audio file"
            )

        logger.info("Starting summarization...")
        summary = summarize_with_langchain(dialogue)

        response = ConversationResponse(
            dialogue=[DialogueSegment(**seg) for seg in dialogue],
            summary=SummaryResponse(**summary)
        )
        
        logger.info(f"Successfully processed audio: {len(dialogue)} segments")
        return JSONResponse(content=response.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if tmp_path:
            background_tasks.add_task(cleanup_temp_file, tmp_path)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Audio Diarization & Summarization API",
        "docs": "/docs",
        "health": "/health"
    }

# ========================== ERROR HANDLERS ==========================
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )

# ========================== MAIN ENTRY ==========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)




