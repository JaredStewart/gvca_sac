"""Configuration API endpoints."""

from fastapi import APIRouter

from app.core.survey_config import SurveyConfig, get_taxonomy

router = APIRouter()


@router.get("/questions")
async def get_questions():
    """Get survey questions configuration."""
    config = SurveyConfig()
    return {
        "questions": config.questions,
        "levels": config.levels,
    }


@router.get("/taxonomy")
async def get_taxonomy_config():
    """Get taxonomy tags and keywords."""
    taxonomy = get_taxonomy()
    return {
        "tags": [
            {
                "name": name,
                "keywords": keywords,
            }
            for name, keywords in taxonomy.items()
        ]
    }


@router.get("/scales")
async def get_response_scales():
    """Get response scale definitions."""
    config = SurveyConfig()
    return {
        "scales": config.scales,
    }


@router.get("/")
async def get_full_config():
    """Get full configuration."""
    config = SurveyConfig()
    taxonomy = get_taxonomy()
    return {
        "questions": config.questions,
        "levels": config.levels,
        "scales": config.scales,
        "taxonomy": [
            {"name": name, "keywords": keywords}
            for name, keywords in taxonomy.items()
        ],
    }
