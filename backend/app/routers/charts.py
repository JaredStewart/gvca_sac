"""Chart generation endpoints for school board presentations."""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.models.charts import (
    ChartResult,
    DemographicComparisonRequest,
    SentimentChartResult,
    SentimentRequest,
    TrendComparisonRequest,
)
from app.services.pipeline_manager import pipeline_manager
from app.services.pocketbase_client import pb_client

logger = logging.getLogger(__name__)

router = APIRouter()


class ChartGenerateRequest(BaseModel):
    include_tags: bool = True


class CompareChartRequest(BaseModel):
    years: list[str]


@router.post("/{year}/generate-all")
async def generate_all_charts(year: str, request: ChartGenerateRequest):
    """
    Generate all presentation charts for a year.

    This generates per-question stacked bar charts, satisfaction summary,
    and tag frequency charts if tagging has been completed.
    """
    from app.core.charts import generate_all_charts

    pipeline = await pipeline_manager.ensure_loaded(year)

    # Get tag distribution if requested
    tag_distribution = None
    if request.include_tags:
        try:
            # Fetch tagging results and compute distribution by question type
            results = await pb_client.get_full_list(
                "tagging_results",
                filter_str=f'year = "{year}"',
            )

            good_choice_tags: dict[str, int] = {}
            better_serve_tags: dict[str, int] = {}

            for result in results:
                question = result.get("question", "")
                tags = result.get("llm_tags", [])

                if "good choice" in question.lower():
                    for tag in tags:
                        good_choice_tags[tag] = good_choice_tags.get(tag, 0) + 1
                elif "better serve" in question.lower():
                    for tag in tags:
                        better_serve_tags[tag] = better_serve_tags.get(tag, 0) + 1

            tag_distribution = {
                "good_choice": good_choice_tags,
                "better_serve": better_serve_tags,
            }
        except Exception:
            # Continue without tags if there's an error
            pass

    try:
        generated = await generate_all_charts(
            year=year,
            df=pipeline.data,
            tag_distribution=tag_distribution,
        )

        return {
            "year": year,
            "status": "completed",
            "charts_generated": len(generated),
            "files": [str(f).split("/")[-1] for f in generated],
        }
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/compare/generate")
async def generate_comparison_charts(request: CompareChartRequest):
    """
    Generate year-over-year comparison charts.

    This generates composite trend charts, level composite charts,
    and per-question YoY charts.
    """
    from app.core.charts import generate_comparison_charts

    if len(request.years) < 2:
        raise HTTPException(
            status_code=400, detail="At least 2 years required for comparison"
        )

    # Load data for all years
    years_data = {}
    for year in request.years:
        pipeline = await pipeline_manager.ensure_loaded(year)
        years_data[year] = pipeline.data

    try:
        generated = await generate_comparison_charts(years_data)

        return {
            "years": request.years,
            "status": "completed",
            "charts_generated": len(generated),
            "files": [str(f).split("/")[-1] for f in generated],
        }
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/list")
async def list_charts(year: str):
    """List generated chart files for a year."""
    settings = get_settings()
    output_dir = settings.artifacts_dir / year

    if not output_dir.exists():
        return {
            "year": year,
            "total": 0,
            "charts": [],
        }

    charts = []
    for filepath in output_dir.glob("*.png"):
        charts.append({
            "filename": filepath.name,
            "size": filepath.stat().st_size,
            "url": f"/api/artifacts/{year}/file/{filepath.name}",
        })

    return {
        "year": year,
        "total": len(charts),
        "charts": sorted(charts, key=lambda x: x["filename"]),
    }


@router.get("/compare/list")
async def list_comparison_charts():
    """List generated comparison chart files."""
    settings = get_settings()
    output_dir = settings.artifacts_dir / "comparison"

    if not output_dir.exists():
        return {
            "total": 0,
            "charts": [],
        }

    charts = []
    for filepath in output_dir.glob("*.png"):
        charts.append({
            "filename": filepath.name,
            "size": filepath.stat().st_size,
            "url": f"/api/artifacts/comparison/file/{filepath.name}",
        })

    return {
        "total": len(charts),
        "charts": sorted(charts, key=lambda x: x["filename"]),
    }


# ================ New Endpoints for FR-007, FR-008, FR-009, FR-010 ================


@router.post("/{year}/demographic-comparison", response_model=ChartResult)
async def generate_demographic_comparison(
    year: str, request: DemographicComparisonRequest
):
    """
    Generate demographic comparison chart (FR-008).

    Compares satisfaction scores between two demographic segments.
    """
    from app.core.charts import generate_demographic_comparison_chart

    try:
        # Fetch survey responses from database
        responses = await pb_client.get_full_list(
            "survey_responses",
            filter_str=f'year = "{year}"',
        )

        if not responses:
            raise HTTPException(
                status_code=404, detail=f"No data found for year {year}"
            )

        # Generate chart data
        chart_data = await generate_demographic_comparison_chart(
            responses=responses,
            segment_a=request.segment_a,
            segment_b=request.segment_b,
            questions=request.questions,
            year=year,
        )

        chart_id = str(uuid.uuid4())[:8]

        return ChartResult(
            chart_id=chart_id,
            chart_type="demographic_comparison",
            file_path=chart_data.get("file_path"),
            parameters={
                "year": year,
                "segment_a": request.segment_a,
                "segment_b": request.segment_b,
            },
            data=chart_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/trend-comparison", response_model=ChartResult)
async def generate_trend_comparison(request: TrendComparisonRequest):
    """
    Generate cross-year trend chart (FR-009).

    Compares satisfaction scores across multiple years.
    """
    from app.core.charts import generate_trend_comparison_chart

    if len(request.years) < 2:
        raise HTTPException(
            status_code=400, detail="At least 2 years required for comparison"
        )

    try:
        # Fetch data for all years
        years_data: dict[str, list[dict[str, Any]]] = {}
        for year in request.years:
            responses = await pb_client.get_full_list(
                "survey_responses",
                filter_str=f'year = "{year}"',
            )
            if responses:
                years_data[year] = responses

        if len(years_data) < 2:
            raise HTTPException(
                status_code=404,
                detail="Not enough years with data for comparison",
            )

        # Generate chart data
        chart_data = await generate_trend_comparison_chart(
            years_data=years_data,
            school_level=request.school_level,
            questions=request.questions,
        )

        chart_id = str(uuid.uuid4())[:8]

        return ChartResult(
            chart_id=chart_id,
            chart_type="trend_comparison",
            file_path=chart_data.get("file_path"),
            parameters={
                "years": request.years,
                "school_level": request.school_level,
            },
            data=chart_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{year}/sentiment", response_model=SentimentChartResult)
async def generate_sentiment_chart(year: str, request: SentimentRequest):
    """
    Generate tag sentiment diverging bar chart (FR-010).

    Shows positive vs negative tag counts based on question type.
    """
    from app.core.charts import generate_sentiment_chart as gen_sentiment

    try:
        # Build filter for tagging results
        filters = [f'year = "{year}"']

        # We need to join with free_responses to get question_type
        # For now, fetch all tagging results and free responses, then join in Python
        tagging_results = await pb_client.get_full_list(
            "tagging_results",
            filter_str=f'year = "{year}"',
        )

        free_responses = await pb_client.get_full_list(
            "free_responses",
            filter_str=f'year = "{year}"',
        )

        if not tagging_results:
            raise HTTPException(
                status_code=404,
                detail=f"No tagging results found for year {year}",
            )

        # Build lookup for question_type by response_id
        response_to_type: dict[str, str] = {}
        response_to_level: dict[str, str | None] = {}
        for fr in free_responses:
            response_to_type[fr["response_id"]] = fr.get("question_type", "improvement")
            response_to_level[fr["response_id"]] = fr.get("level")

        # Filter and process
        filtered_results = []
        for tr in tagging_results:
            rid = tr.get("response_id")
            level = response_to_level.get(rid)

            # Apply school_level filter
            if request.school_level and level != request.school_level:
                continue

            # TODO: Apply demographic filter if needed (requires joining with survey_responses)

            q_type = response_to_type.get(rid, "improvement")
            tr["question_type"] = q_type
            filtered_results.append(tr)

        # Generate sentiment data
        sentiment_data = await gen_sentiment(filtered_results)

        chart_id = str(uuid.uuid4())[:8]

        return SentimentChartResult(
            chart_id=chart_id,
            file_path=None,  # Frontend will render this
            data=sentiment_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
