"""Pydantic models for data management endpoints."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ImportRequest(BaseModel):
    """Request for importing survey data."""

    replace_existing: bool = Field(
        default=False, description="Replace existing data for the year"
    )


class ImportResult(BaseModel):
    """Result of a data import operation."""

    year: str
    total_responses: int = Field(description="Number of survey responses imported")
    free_responses_extracted: int = Field(
        description="Number of free-text responses extracted"
    )
    replaced_existing: bool = Field(
        description="Whether existing data was replaced"
    )


class SurveyResponseSummary(BaseModel):
    """Summary of a survey response for list views."""

    id: str
    year: str
    respondent_id: str
    school_level: str | None = None
    demographics: dict[str, Any] | None = None
    satisfaction_scores: dict[str, Any] | None = None
    imported_at: datetime


class FreeResponseSummary(BaseModel):
    """Summary of a free-text response."""

    id: str
    year: str
    response_id: str
    question: str
    question_type: str  # "praise" or "improvement"
    level: str | None = None
    response_text: str


class DataStatus(BaseModel):
    """Status of data for a given year."""

    year: str
    survey_response_count: int
    free_response_count: int
    tagging_count: int
    has_data: bool


# Question type mapping for free responses
PRAISE_QUESTIONS = [
    "What makes GVCA good?",
    "What makes",  # Partial match for flexibility
]

IMPROVEMENT_QUESTIONS = [
    "How could GVCA better serve",
    "How could",  # Partial match for flexibility
]


def get_question_type(question: str) -> str:
    """Determine if a question is praise or improvement oriented."""
    question_lower = question.lower()
    for praise_keyword in ["good", "well", "positive", "strength", "like", "best"]:
        if praise_keyword in question_lower:
            return "praise"
    for improve_keyword in ["improve", "better", "serve", "change", "suggestion"]:
        if improve_keyword in question_lower:
            return "improvement"
    # Default to improvement for unknown questions
    return "improvement"
