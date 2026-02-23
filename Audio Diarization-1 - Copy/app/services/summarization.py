from typing import List, Dict, Any
from fastapi import HTTPException
import re
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from app.config import settings
from app.logger import logger


def summarize_with_langchain(dialogue_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate structured conversation summary using LLM with robust parsing."""

    if not dialogue_segments:
        return {
            "summary_paragraph": "No dialogue detected in the audio.",
            "key_topics": [],
            "tone": "N/A"
        }

    # Combine dialogue into readable text
    conversation_text = "\n".join(
        [f"{seg['speaker']}: {seg.get('text_english', '')}" for seg in dialogue_segments]
    ).strip()

    # Skip short or empty conversation
    if len(conversation_text) < 30:
        return {
            "summary_paragraph": "Conversation too short to summarize.",
            "key_topics": [],
            "tone": "N/A"
        }

    prompt_template = """
                        You are a professional conversation summarizer. Your task is to read a multi-speaker dialogue and produce a natural, concise summary.

                        Follow these rules carefully:
                        1. DO NOT mention speaker names or identifiers like SPEAKER_00 or SPEAKER_01.
                        2. Focus only on what was discussed — not who said it.
                        3. Combine all contributions into one coherent narrative paragraph.
                        4. Highlight the main subjects, ideas, or problems discussed.
                        5. Be objective, concise, and factual.

                        Return your answer ONLY in the following structured format:

                        SUMMARY:
                        <one clear paragraph summarizing the main ideas — no speaker names>

                        KEY TOPICS:
                        - <topic 1>
                        - <topic 2>
                        - <topic 3>

                        TONE:
                        <overall mood or tone of the conversation, e.g., friendly, professional, tense, etc.>

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
            timeout=40
        )

        chain = prompt | llm
        logger.info("Generating summary with LLM...")

        result = chain.invoke({"conversation": conversation_text})
        output_text = result.content.strip() if hasattr(result, "content") else str(result).strip()

        if not output_text:
            raise ValueError("Empty LLM output.")

        # --- Robust parsing section ---
        summary_match = re.search(r"summary\s*:\s*(.+?)(?=\n\s*key topics:|\Z)", output_text, re.I | re.S)
        topics_match = re.search(r"key topics\s*:\s*(.+?)(?=\n\s*tone:|\Z)", output_text, re.I | re.S)
        tone_match = re.search(r"tone\s*:\s*(.+)", output_text, re.I | re.S)

        summary_paragraph = summary_match.group(1).strip() if summary_match else ""
        tone = tone_match.group(1).strip() if tone_match else ""

        key_topics = []
        if topics_match:
            for line in topics_match.group(1).splitlines():
                topic = line.strip("-• ").strip()
                if topic:
                    key_topics.append(topic)

        # --- Fallbacks ---
        summary_paragraph = summary_paragraph or "Unable to generate summary."
        tone = tone or "Neutral"
        key_topics = key_topics[:5] or ["General Discussion"]

        return {
            "summary_paragraph": summary_paragraph,
            "key_topics": key_topics,
            "tone": tone
        }

    except Exception as e:
        logger.error(f"LLM summarization failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
