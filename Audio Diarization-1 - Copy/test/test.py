
# FastAPI app with separate endpoints for diarization and summarization

import os
import dotenv
import whisperx
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import uvicorn

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEVICE = "cpu"
MODEL_NAME = "small"
COMPUTE_TYPE = "int8"

app = FastAPI(title="Audio Diarization & Summarization API", version="1.0")

class DialogueSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str

class ConversationSummary(BaseModel):
    summary_paragraph: str = Field(description="Natural paragraph summarizing conversation")
    key_topics: List[str] = Field(description="List of discussion topics")
    tone: str = Field(description="Overall tone of conversation")

def transcribe_and_diarize(audio_path: str) -> List[dict]:
    try:
        print(f"Loading WhisperX model '{MODEL_NAME}' on {DEVICE}...")
        model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)

        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=8)

        align_model, metadata = whisperx.load_align_model(result["language"], DEVICE)
        aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, DEVICE)

        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=5)

        final = whisperx.assign_word_speakers(diarize_segments, aligned_result)

        dialogue = [
            {
                "speaker": seg.get("speaker", "unknown"),
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "text": seg["text"].strip()
            }
            for seg in final["segments"]
        ]
        return sorted(dialogue, key=lambda x: x["start"])
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")

def summarize_with_langchain_structured(dialogue_segments: List[dict]) -> ConversationSummary:
    conversation_text = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in dialogue_segments)

    prompt_template = """
                        You are summarizing a phone conversation between two people.

                        Please provide JSON output with three fields:
                        1. summary_paragraph: A single cohesive paragraph summarizing the full conversation, clearly describing the main topics and information discussed.
                        2. key_topics: A JSON array of key topics discussed.
                        3. tone: A short description of the overall tone.

                        Example output:
                        {
                        "summary_paragraph": "The two participants discussed daily routines and plans for the weekend...",
                        "key_topics": ["Weekend plans", "Daily routines", "Weather"],
                        "tone": "Informal and friendly"
                        }

                        Conversation:
                        {conversation}
                        """

    prompt = PromptTemplate(input_variables=["conversation"], template=prompt_template)

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.4
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(conversation=conversation_text)

    import json
    # Parse the JSON string output from the model to a dict
    try:
        summary_dict = json.loads(output)
    except json.JSONDecodeError:
        raise RuntimeError("Failed to parse summary output as JSON. Raw output: " + output)

    return ConversationSummary(**summary_dict)




@app.post("/transcribe/", response_model=List[DialogueSegment])
async def transcribe_endpoint(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        dialogue = transcribe_and_diarize(tmp_path)
        return JSONResponse(content=dialogue)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)



@app.post("/summarize/", response_model=ConversationSummary)
async def summarize_endpoint(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        dialogue = transcribe_and_diarize(tmp_path)
        summary = summarize_with_langchain_structured(dialogue)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/")
def root():
    return {"message": "Welcome to the Audio Diarization and Summarization API"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)


# FastAPI app with single endpoint for audio diarization and summarization

import os
import tempfile
import dotenv
import whisperx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

# ========================== ENV & CONFIG ==========================
dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEVICE = "cpu"  # or "cuda" if GPU available
MODEL_NAME = "small"
COMPUTE_TYPE = "int8"

app = FastAPI(title="Audio Diarization & Summarization API")

# ========================== SCHEMAS ==========================
class DialogueSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str

class SummaryResponse(BaseModel):
    summary_paragraph: str
    key_topics: List[str]
    tone: str

class ConversationResponse(BaseModel):
    dialogue: List[DialogueSegment]
    summary: SummaryResponse

# ========================== UTILS ==========================
def transcribe_and_diarize(audio_path: str) -> List[Dict[str, Any]]:
    """Perform transcription + diarization using WhisperX."""
    try:
        print(f"🔊 Loading WhisperX model '{MODEL_NAME}' on {DEVICE} ...")
        model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)

        print("🎧 Loading audio ...")
        audio = whisperx.load_audio(audio_path)

        print("🗣️ Transcribing ...")
        result = model.transcribe(audio, batch_size=8)

        print("📏 Aligning words ...")
        align_model, metadata = whisperx.load_align_model(result["language"], DEVICE)
        aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, DEVICE)

        print("👥 Performing diarization ...")
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=5)

        print("🔗 Assigning speakers to segments ...")
        final = whisperx.assign_word_speakers(diarize_segments, aligned_result)

        dialogue = [
            {
                "speaker": seg.get("speaker", "unknown"),
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "text": seg.get("text", "").strip()
            }
            for seg in final["segments"]
        ]

        dialogue = sorted(dialogue, key=lambda x: x["start"])
        print("✅ Transcription & diarization done.")
        return dialogue

    except Exception as e:
        raise RuntimeError(f"WhisperX processing failed: {str(e)}")

def summarize_with_langchain(dialogue_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate structured conversation summary using LLM."""
    conversation_text = "\n".join(
        [f"{seg['speaker']}: {seg['text']}" for seg in dialogue_segments]
    )

    prompt_template = """
You are summarizing a conversation between multiple speakers.

Please provide:
1. A single cohesive paragraph summarizing the entire conversation naturally.
2. A list of the key topics discussed.
3. The overall tone of the conversation.

Format response as:

SUMMARY:
<paragraph>

KEY TOPICS:
- Topic 1
- Topic 2

TONE:
<tone>

Conversation:
{conversation}
"""

    prompt = PromptTemplate(
        input_variables=["conversation"],
        template=prompt_template
    )

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.4
    )

    chain = RunnableSequence(prompt | llm)
    print("🤖 Generating paragraph summary with LLM ...")
    output_text = chain.invoke({"conversation": conversation_text}).content

    # Parse structured fields
    summary_paragraph = ""
    key_topics = []
    tone = ""

    try:
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

            if mode == "summary":
                summary_paragraph += line.strip() + " "
            elif mode == "topics":
                if line.strip().startswith("-"):
                    key_topics.append(line.strip("- ").strip())
            elif mode == "tone":
                tone += line.strip() + " "
    except Exception:
        summary_paragraph = output_text.strip()
        key_topics = []
        tone = "Neutral"

    return {
        "summary_paragraph": summary_paragraph.strip(),
        "key_topics": key_topics,
        "tone": tone.strip()
    }

# ========================== API ROUTES ==========================
@app.post("/process-audio", response_model=ConversationResponse)
async def process_audio(file: UploadFile = File(...)):
    """Upload an audio file → diarization + summary."""
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        dialogue = transcribe_and_diarize(tmp_path)
        summary = summarize_with_langchain(dialogue)

        response = ConversationResponse(
            dialogue=[DialogueSegment(**seg) for seg in dialogue],
            summary=SummaryResponse(**summary)
        )
        return JSONResponse(content=response.dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ========================== MAIN ENTRY ==========================
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)






#Single endpoint contains all code inside that endpoint
import os
import tempfile
import json
import dotenv
import whisperx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

# =========================
# ENVIRONMENT & MODEL SETUP
# =========================
dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEVICE = "cpu"               # or "cuda" if GPU available
MODEL_NAME = "small"         # can be "base", "medium", "large-v2"
COMPUTE_TYPE = "int8"        # good for CPU
BATCH_SIZE = 16

# ================
# FASTAPI INSTANCE
# ================
app = FastAPI(title="Unified Audio Processing API")


@app.get("/")
def root():
    return {"message": "Unified Audio Processing API - Ready!"}


# ===============================
# MAIN ENDPOINT: /process_audio/
# ===============================
@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        print(">>> Starting transcription + diarization...")

        # ------------------------------
        # 1. Save uploaded audio to temp
        # ------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(await file.read())
            tmp_audio_path = tmp_audio.name

        # ------------------------------
        # 2. Load WhisperX model
        # ------------------------------
        model = whisperx.load_model(
            MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE
        )

        # ------------------------------
        # 3. Transcribe Audio
        # ------------------------------
        result = model.transcribe(tmp_audio_path, batch_size=BATCH_SIZE)

        # ------------------------------
        # 4. Align (optional but improves timestamps)
        # ------------------------------
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=DEVICE
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            tmp_audio_path,
            DEVICE,
            return_char_alignments=False,
        )

        # ------------------------------
        # 5. Speaker Diarization
        # ------------------------------
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=HF_TOKEN, device=DEVICE
        )
        diarize_segments = diarize_model(tmp_audio_path)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        # ------------------------------
        # 6. Create dialogue structure
        # ------------------------------
        dialogue_output = []
        for segment in result["segments"]:
            dialogue_output.append({
                "speaker": segment.get("speaker", "unknown"),
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": segment["text"].strip()
            })

        # Combine all text for summary
        conversation_text = "\n".join(
            [f"{d['speaker']}: {d['text']}" for d in dialogue_output]
        )

        # ------------------------------
        # 7. Generate Summary using LLM
        # ------------------------------
        print(">>> Generating summary with LLM...")

        prompt = PromptTemplate(
            input_variables=["conversation"],
            template="""
You are an expert meeting summarizer.
Analyze the following multi-speaker conversation and produce a concise structured summary.

Conversation:
{conversation}

Return the result strictly in valid JSON format like this:
{{
  "summary_paragraph": "<short paragraph summarizing key points>",
  "key_topics": ["<topic1>", "<topic2>"],
  "tone": "<overall tone>"
}}
"""
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        chain = RunnableSequence(prompt | llm)

        summary_result = chain.invoke({"conversation": conversation_text})
        summary_text = summary_result.content.strip()

        # Try to parse JSON safely
        try:
            summary_json = json.loads(summary_text)
        except json.JSONDecodeError:
            # Fallback if model output is not perfectly formatted JSON
            summary_json = {"summary_paragraph": summary_text, "key_topics": [], "tone": ""}

        # ------------------------------
        # 8. Return unified structured output
        # ------------------------------
        final_output = {
            "dialogue": dialogue_output,
            "summary": summary_json
        }

        return JSONResponse(content=final_output)

    except Exception as e:
        # Detailed exception handling for debugging
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


# ===============================
# RUN SERVER
# ===============================


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)