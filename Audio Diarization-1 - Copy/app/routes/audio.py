import tempfile
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.schemas import ConversationResponse, DialogueSegment, SummaryResponse
from app.utils import validate_audio_file, cleanup_temp_file
from app.services.transcription import transcribe_and_diarize
from app.services.summarization import summarize_with_langchain
from app.logger import logger

router = APIRouter()

@router.post("/process-audio", response_model=ConversationResponse)
async def process_audio(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload an audio file → diarization + translation + summary.
    
    Supported formats: mp3, wav, m4a, ogg, flac
    Max file size: 500MB
    
    Response includes:
    - text_source: Original transcribed text
    - text_english: Translated to English (if source language differs)
    """
    tmp_path = None
    start_time = time.time()
    
    try:
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        validate_audio_file(file.filename, file_size)
        logger.info(f"Processing audio file: {file.filename} ({file_size / 1024 / 1024:.2f}MB)")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        logger.info("Starting transcription, diarization, and translation...")
        dialogue = transcribe_and_diarize(tmp_path)
        
        if not dialogue:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in audio file"
            )

        logger.info("Starting summarization...")
        summary = summarize_with_langchain(dialogue)

        processing_time = time.time() - start_time
        
        response = ConversationResponse(
            dialogue=[DialogueSegment(**seg) for seg in dialogue],
            summary=SummaryResponse(**summary),
            processing_time_seconds=round(processing_time, 2)
        )
        
        logger.info(f"Successfully processed audio: {len(dialogue)} segments in {processing_time:.2f}s")
        return JSONResponse(content=response.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if tmp_path:
            background_tasks.add_task(cleanup_temp_file, tmp_path)