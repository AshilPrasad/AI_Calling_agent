from typing import List, Dict, Any, Optional
from fastapi import HTTPException
import assemblyai as aai
from langdetect import detect_langs, DetectorFactory, LangDetectException
import logging
import time
from openai import OpenAI
 
from app.config import settings
from app.logger import logger
from app.models_cache import models_cache
 
logger = logging.getLogger(__name__)
DetectorFactory.seed = 0
 
# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)
 
def detect_language(text: str) -> str:
    """Detect primary language from text."""
    try:
        if not text or not text.strip():
            return "en"
        detected = detect_langs(text)
        if detected and detected[0].prob > 0.5:
            return detected[0].lang
        return "en"
    except LangDetectException:
        return "en"
 
def translate_text_to_english(text: str) -> str:
    """
    Uses OpenAI GPT-4 to translate text into natural, fluent English.
    """
    if not text or not text.strip():
        return text
   
    system_prompt = (
        "You are a professional translation assistant. "
        "Translate the user's message into clear, natural, and fluent English, "
        "preserving meaning and context accurately. "
        "Automatically detect the source language. "
        "Do not add explanations or extra commentary—return only the translated text."
    )
 
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=1000
        )
 
        translated = response.choices[0].message.content.strip()
        logger.info(f"✅ Translation: {text[:50]}... → {translated[:50]}...")
        return translated
 
    except Exception as e:
        logger.warning(f"OpenAI translation error: {e}. Using original text.")
        return text
 
def load_models():
    """Initialize AssemblyAI configuration at startup."""
    try:
        if not hasattr(settings, 'ASSEMBLYAI_API_KEY') or not settings.ASSEMBLYAI_API_KEY:
            logger.warning("AssemblyAI API key not found in settings")
            return
       
        # Initialize AssemblyAI
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        logger.info("✅ AssemblyAI initialized successfully")
       
    except Exception as e:
        logger.error(f"Failed to initialize AssemblyAI: {e}", exc_info=True)
        raise
 
def cleanup_models():
    """Clean up model resources."""
    try:
        models_cache.transcription_model = None
        models_cache.diarization_model = None
        logger.info("Models cleaned up")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
 
def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default
 
def assign_speakers_by_turns(
    segments: List[Dict[str, Any]],
    silence_threshold: float = 1.5
) -> List[Dict[str, Any]]:
    """
    Assign speakers based on conversation turns (alternating pattern).
    This works better than unreliable diarization for some audio files.
   
    Args:
        segments: List of transcribed segments
        silence_threshold: Gap in seconds to consider a speaker change
    """
    if not segments:
        return segments
   
    # Start with Speaker 1
    current_speaker = "Speaker 1"
    speaker_num = 1
   
    for i, seg in enumerate(segments):
        if i == 0:
            seg["speaker"] = current_speaker
            continue
       
        # Calculate gap from previous segment
        prev_end = segments[i-1]["end"]
        current_start = seg["start"]
        gap = current_start - prev_end
       
        # If there's a significant pause, likely a speaker change
        if gap >= silence_threshold:
            speaker_num = 2 if speaker_num == 1 else 1
            current_speaker = f"Speaker {speaker_num}"
       
        seg["speaker"] = current_speaker
   
    return segments
 
def calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate overlap duration between two time segments."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start)
 
def find_best_speaker_diarization(
    seg_start: float,
    seg_end: float,
    speaker_segments: List[Dict[str, Any]],
    time_tolerance: float = 0.3
) -> Optional[str]:
    """
    Find the best matching speaker from diarization results.
    Returns None if confidence is too low.
    """
    best_speaker = None
    max_overlap = 0.0
    seg_duration = seg_end - seg_start
   
    for dia_seg in speaker_segments:
        overlap = calculate_overlap(
            seg_start - time_tolerance,
            seg_end + time_tolerance,
            dia_seg["start"],
            dia_seg["end"]
        )
       
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = dia_seg["speaker"]
   
    # Only return if we have good confidence (>60% overlap)
    confidence = max_overlap / seg_duration if seg_duration > 0 else 0
    return best_speaker if confidence > 0.6 else None
 
def transcribe_and_diarize(audio_path: str, use_simple_turns: bool = False) -> List[Dict[str, Any]]:
    """
    Perform transcription + speaker identification using AssemblyAI.
   
    Args:
        audio_path: Path to audio file or URL
        use_simple_turns: If True, use simple turn-taking logic instead of AssemblyAI diarization
    """
    try:
        # Initialize AssemblyAI if not already done
        if not aai.settings.api_key:
            load_models()
       
        if not aai.settings.api_key:
            raise HTTPException(
                status_code=400,
                detail="AssemblyAI API key not configured. Please set ASSEMBLYAI_API_KEY in settings."
            )
       
        logger.info(f"Loading audio from {audio_path}")
       
        # Configure transcription settings with optimal parameters for speaker detection
        config = aai.TranscriptionConfig(
            speaker_labels=True,  # Always enable speaker diarization
            speakers_expected=2,  # Expect 2 speakers (adjust if needed)
            language_detection=True,  # Auto-detect language
            punctuate=True,
            format_text=True,
            # Use words for better granularity
            speech_model=aai.SpeechModel.best,
        )
       
        # Create transcriber
        transcriber = aai.Transcriber()
       
        # Transcribe audio
        logger.info("Transcribing audio with AssemblyAI...")
        transcript = transcriber.transcribe(audio_path, config=config)
       
        # Check for errors
        if transcript.status == aai.TranscriptStatus.error:
            logger.error(f"Transcription failed: {transcript.error}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")
       
        # Wait for completion if still processing
        while transcript.status in [aai.TranscriptStatus.queued, aai.TranscriptStatus.processing]:
            logger.info("Waiting for transcription to complete...")
            time.sleep(3)
            transcript = transcriber.get_transcript(transcript.id)
            if transcript.status == aai.TranscriptStatus.error:
                raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")
       
        detected_lang = transcript.language_code if hasattr(transcript, 'language_code') else 'en'
        logger.info(f"🔤 Detected language: {detected_lang}")
        logger.info(f"Confidence: {transcript.confidence if hasattr(transcript, 'confidence') else 'N/A'}")
        logger.info(f"Audio duration: {transcript.audio_duration if hasattr(transcript, 'audio_duration') else 'N/A'}s")
       
        if not transcript.utterances:
            logger.warning("No speech detected in audio")
            return []
       
        logger.info(f"✅ Transcription complete: {len(transcript.utterances)} segments")
       
        # Build dialogue list
        dialogue = []
        translation_cache = {}
       
        for utterance in transcript.utterances:
            source_text = utterance.text.strip()
           
            if not source_text:
                continue
           
            seg_start = safe_float(utterance.start / 1000)  # Convert ms to seconds
            seg_end = safe_float(utterance.end / 1000)
           
            # Get speaker from AssemblyAI
            speaker = utterance.speaker if hasattr(utterance, 'speaker') else "Unknown"
           
            # Detect language per segment
            segment_lang = detect_language(source_text)
            if segment_lang == "en" and detected_lang != "en":
                segment_lang = detected_lang
           
            # Translate if needed
            if segment_lang.lower() not in ["en", "english"]:
                cache_key = source_text.lower().strip()
               
                if cache_key in translation_cache:
                    translated_text = translation_cache[cache_key]
                else:
                    logger.info(f"Translating from {segment_lang}: {source_text[:50]}...")
                    translated_text = translate_text_to_english(source_text)
                    translation_cache[cache_key] = translated_text
            else:
                translated_text = source_text
           
            dialogue.append({
                "speaker": speaker,
                "start": round(seg_start, 2),
                "end": round(seg_end, 2),
                "text_source": source_text,
                "text_english": translated_text,
                "detected_language": segment_lang,
                "confidence": round(float(utterance.confidence), 2) if hasattr(utterance, 'confidence') else None
            })
       
        # Sort by start time
        dialogue = sorted(dialogue, key=lambda x: x["start"])
       
        # Apply simple turn-taking if requested
        if use_simple_turns:
            logger.info("Using simple turn-taking logic for speaker assignment")
            dialogue = assign_speakers_by_turns(dialogue, silence_threshold=1.5)
        else:
            # Normalize speaker labels (A, B, C -> Speaker 1, Speaker 2, Speaker 3)
            speaker_map = {}
            speaker_counter = 1
           
            for item in dialogue:
                original_speaker = item["speaker"]
                if original_speaker not in speaker_map:
                    speaker_map[original_speaker] = f"Speaker {speaker_counter}"
                    speaker_counter += 1
                item["speaker"] = speaker_map[original_speaker]
       
        # Log final statistics
        speaker_summary = {}
        for item in dialogue:
            speaker = item["speaker"]
            speaker_summary[speaker] = speaker_summary.get(speaker, 0) + 1
       
        logger.info(f"✅ Final output: {len(dialogue)} segments")
        logger.info(f"Speaker distribution: {speaker_summary}")
       
        return dialogue
   
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AssemblyAI processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
 
 
# Additional function for advanced features (optional)
def transcribe_with_advanced_features(
    audio_path: str,
    enable_sentiment: bool = False,
    enable_chapters: bool = False,
    enable_entities: bool = False
) -> Dict[str, Any]:
    """
    Advanced transcription with additional AssemblyAI features.
   
    Args:
        audio_path: Path to audio file or URL
        enable_sentiment: Analyze sentiment of each utterance
        enable_chapters: Auto-generate chapter summaries
        enable_entities: Detect named entities (people, organizations, etc.)
   
    Returns:
        Dictionary with transcription and additional analysis
    """
    try:
        if not aai.settings.api_key:
            load_models()
       
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            language_detection=True,
            punctuate=True,
            format_text=True,
            sentiment_analysis=enable_sentiment,
            auto_chapters=enable_chapters,
            entity_detection=enable_entities,
        )
       
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)
       
        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")
       
        # Wait for completion
        while transcript.status in [aai.TranscriptStatus.queued, aai.TranscriptStatus.processing]:
            time.sleep(3)
            transcript = transcriber.get_transcript(transcript.id)
       
        result = {
            "text": transcript.text,
            "language": transcript.language_code if hasattr(transcript, 'language_code') else 'en',
            "confidence": transcript.confidence if hasattr(transcript, 'confidence') else None,
            "utterances": []
        }
       
        # Add utterances with optional features
        for utterance in transcript.utterances:
            utt_data = {
                "speaker": utterance.speaker,
                "start": round(utterance.start / 1000, 2),
                "end": round(utterance.end / 1000, 2),
                "text": utterance.text,
                "confidence": round(utterance.confidence, 2) if hasattr(utterance, 'confidence') else None
            }
           
            if enable_sentiment and hasattr(utterance, 'sentiment'):
                utt_data["sentiment"] = utterance.sentiment
           
            result["utterances"].append(utt_data)
       
        # Add chapters if enabled
        if enable_chapters and hasattr(transcript, 'chapters'):
            result["chapters"] = [
                {
                    "start": round(chapter.start / 1000, 2),
                    "end": round(chapter.end / 1000, 2),
                    "summary": chapter.summary,
                    "headline": chapter.headline
                }
                for chapter in transcript.chapters
            ]
       
        # Add entities if enabled
        if enable_entities and hasattr(transcript, 'entities'):
            result["entities"] = [
                {
                    "text": entity.text,
                    "type": entity.entity_type,
                    "start": round(entity.start / 1000, 2),
                    "end": round(entity.end / 1000, 2)
                }
                for entity in transcript.entities
            ]
       
        return result
   
    except Exception as e:
        logger.error(f"Advanced transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
 
 
"""
Installation requirements:
pip install assemblyai
pip install langdetect
pip install openai
 
Configuration in app/config.py (settings):
- ASSEMBLYAI_API_KEY: Your AssemblyAI API key (get free at https://www.assemblyai.com/)
- OPENAI_API_KEY: Your OpenAI API key for translation
 
Get your free AssemblyAI API key:
https://www.assemblyai.com/
100 hours free transcription + diarization
"""