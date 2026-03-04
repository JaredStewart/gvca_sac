"""Tag management endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.survey_config import get_taxonomy
from app.services.pocketbase_client import pb_client

logger = logging.getLogger(__name__)

router = APIRouter()


class TagOverrideRequest(BaseModel):
    modified_tags: list[str]
    reason: str | None = None


@router.get("/taxonomy")
async def get_taxonomy_tags():
    """Get the taxonomy tags and keywords."""
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


@router.put("/{year}/responses/{response_id}")
async def override_response_tags(
    year: str,
    response_id: str,
    request: TagOverrideRequest,
):
    """Override tags for a specific response."""
    # Get current tags
    filter_str = f'year = "{year}" && response_id = "{response_id}"'
    try:
        existing = await pb_client.get_list("tagging_results", filter_str=filter_str)
        original_tags = []
        if existing.get("items"):
            original_tags = existing["items"][0].get("llm_tags", [])

        # Create or update override
        override_data = {
            "year": year,
            "response_id": response_id,
            "question": existing["items"][0].get("question", "") if existing.get("items") else "",
            "original_tags": original_tags,
            "modified_tags": request.modified_tags,
            "reason": request.reason,
        }

        result = await pb_client.upsert(
            "tag_overrides",
            override_data,
            f'year = "{year}" && response_id = "{response_id}"',
        )

        return {
            "status": "success",
            "response_id": response_id,
            "original_tags": original_tags,
            "modified_tags": request.modified_tags,
        }
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/modifications")
async def get_tag_modifications(year: str):
    """Get all tag modifications for a year."""
    filter_str = f'year = "{year}"'
    try:
        results = await pb_client.get_full_list(
            "tag_overrides",
            filter_str=filter_str,
            sort="-created",
        )
        return {
            "year": year,
            "total": len(results),
            "modifications": results,
        }
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{year}/modifications/{response_id}")
async def delete_tag_modification(year: str, response_id: str):
    """Delete a tag modification (restore original tags)."""
    filter_str = f'year = "{year}" && response_id = "{response_id}"'
    try:
        deleted = await pb_client.delete_by_filter("tag_overrides", filter_str)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Modification not found")
        return {"status": "success", "deleted": deleted}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
