from typing import List
from pydantic import BaseModel, Field, field_validator, ValidationInfo

class DialogueSegment(BaseModel):
    """Schema for dialogue segments with source and English translation."""
    speaker: str = Field(..., min_length=1)
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    text_source: str = Field(..., min_length=1)
    text_english: str = Field(..., min_length=1)
    
    @field_validator("end")
    def validate_end_time(cls, v, info: ValidationInfo):
        start_value = info.data.get("start")
        if start_value is not None and v <= start_value:
            raise ValueError("end time must be greater than start time")
        return v

class SummaryResponse(BaseModel):
    """Schema for conversation summary."""
    summary_paragraph: str = Field(..., min_length=20)
    key_topics: List[str] = Field(default_factory=list, max_items=20)
    tone: str = Field(..., min_length=2)

class ConversationResponse(BaseModel):
    """Complete response with dialogue and summary."""
    dialogue: List[DialogueSegment]
    summary: SummaryResponse
    processing_time_seconds: float = Field(default=0, ge=0)

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    environment: str
    models_loaded: bool