"""Deck generation endpoint."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Default template location relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_TEMPLATE = _PROJECT_ROOT / "data" / "templates" / "presentation.pptx"


class DeckGenerateRequest(BaseModel):
    year: str
    years: list[str] | None = None
    template_path: str | None = None


@router.post("/generate")
async def generate_deck(request: DeckGenerateRequest):
    """Generate a PPTX deck from survey data and return it as a download."""
    from app.core.deck_generator import generate_deck_from_pipeline

    template = Path(request.template_path) if request.template_path else _DEFAULT_TEMPLATE
    if not template.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Template not found: {template}. Place a template PPTX at {_DEFAULT_TEMPLATE}",
        )

    try:
        output_path = await generate_deck_from_pipeline(
            year=request.year,
            template_path=template,
            years=request.years,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Deck generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Deck generation failed")

    return FileResponse(
        path=str(output_path),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=output_path.name,
    )
