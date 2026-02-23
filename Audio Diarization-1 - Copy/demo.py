import torch
import whisperx
import os


def diarize_and_transcribe(audio_path: str, whisper_model="large-v3", device=None):
    """
    Runs ASR → word alignment → speaker diarization → merges speakers per word.
    Returns a dict with segments, word-level timing, and speaker for each word.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------------------
    # 1. Load Whisper ASR Model
    # ------------------------------------------
    print("Loading WhisperX model...")
    model = whisperx.load_model(
        whisper_model,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )

    # ------------------------------------------
    # 2. Run ASR Transcription (segment-level)
    # ------------------------------------------
    print("Running transcription...")
    asr_result = model.transcribe(audio_path)

    # asr_result contains:
    # {
    #   "segments": [{start, end, text}],
    #   "text": "...",
    #   "language": "xx"
    # }

    language = asr_result.get("language", None)

    # ------------------------------------------
    # 3. Alignment Model (word-level timestamps)
    # ------------------------------------------
    print("Loading alignment model...")
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device
    )

    print("Aligning word-level timestamps...")
    aligned_result = whisperx.align(
        asr_result["segments"],
        align_model,
        metadata,
        audio_path,
        device=device,
        return_char_alignments=False
    )

    # aligned_result contains aligned_result["word_segments"]

    # ------------------------------------------
    # 4. Speaker Diarization Model
    # ------------------------------------------
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=None,
        device=device
    )

    print("Running speaker diarization...")
    diarize_segments = diarize_model(audio_path)

    # ------------------------------------------
    # 5. Assign speaker for each word
    # ------------------------------------------
    print("Assigning speakers to words...")

    word_segments = whisperx.assign_word_speakers(
        diarize_segments,
        aligned_result["word_segments"]
    )

    # Final structured format
    final_output = {
        "segments": asr_result["segments"],
        "words": word_segments,
        "language": language
    }

    return final_output


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    # 👉 CHANGE THIS PATH TO YOUR LOCAL AUDIO FILE
    audio_path = r"C:\Users\ashil.p\OneDrive - Difinity Digital\Projects\Audio Diarization-1\audio\AUD-20251110-WA0012-[AudioTrimmer.com].mp3"

    result = diarize_and_transcribe(audio_path, whisper_model="large-v3")

    print("\n========== WORD-LEVEL OUTPUT ==========\n")
    for w in result["words"]:
        start = w.get("start", 0)
        end = w.get("end", 0)
        text = w.get("text", "")
        speaker = w.get("speaker", "UNKNOWN")

        print(f"{start:.2f}-{end:.2f} | {speaker} | {text}")
