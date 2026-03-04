"""Pydantic models for chart generation endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class DemographicComparisonRequest(BaseModel):
    """Request for generating demographic comparison chart."""

    segment_a: str = Field(
        description="First segment: all, year1, minority, support, grammar, middle, high"
    )
    segment_b: str = Field(
        description="Second segment to compare against"
    )
    questions: list[str] | None = Field(
        default=None, description="Specific questions to compare (default: all)"
    )


class TrendComparisonRequest(BaseModel):
    """Request for generating cross-year trend chart."""

    years: list[str] = Field(min_length=2, description="Years to compare")
    school_level: str | None = Field(
        default=None, description="Filter by school level: Grammar, Middle, High"
    )
    questions: list[str] | None = Field(
        default=None, description="Specific questions to compare (default: all)"
    )


class SentimentRequest(BaseModel):
    """Request for generating sentiment chart."""

    school_level: str | None = Field(
        default=None, description="Filter by school level"
    )
    demographic: str | None = Field(
        default=None, description="Filter by demographic: year1, minority, support"
    )


class ChartResult(BaseModel):
    """Result of chart generation."""

    chart_id: str
    chart_type: str
    file_path: str | None = Field(
        default=None, description="Path to generated PNG file"
    )
    parameters: dict[str, Any]
    data: dict[str, Any] = Field(
        description="Chart data for frontend rendering"
    )


class SentimentChartResult(BaseModel):
    """Result of sentiment chart generation."""

    chart_id: str
    file_path: str | None = None
    data: list[dict[str, Any]] = Field(
        description="Array of {tag, positive_count, negative_count}"
    )


class SentimentDataPoint(BaseModel):
    """Single data point in sentiment chart."""

    tag: str
    positive_count: int
    negative_count: int
