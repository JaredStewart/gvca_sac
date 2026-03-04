"""Cluster metadata endpoints — user-editable names and descriptions."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.pocketbase_client import pb_client

logger = logging.getLogger(__name__)

router = APIRouter()


class ClusterMetadataUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


@router.get("/{year}/metadata")
async def get_cluster_metadata(year: str):
    """Return all cluster metadata for a year."""
    try:
        records = await pb_client.get_full_list(
            "cluster_metadata",
            filter_str=f'year = "{year}"',
            sort="cluster_id",
        )
        return {"year": year, "metadata": records}
    except Exception as e:
        logger.error("Error fetching cluster metadata: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{year}/metadata/{cluster_id}")
async def update_cluster_metadata(
    year: str, cluster_id: int, body: ClusterMetadataUpdate
):
    """Upsert name/description for a cluster."""
    data: dict = {"year": year, "cluster_id": cluster_id}
    if body.name is not None:
        data["name"] = body.name
    if body.description is not None:
        data["description"] = body.description

    try:
        record = await pb_client.upsert(
            "cluster_metadata",
            data,
            filter_str=f'year = "{year}" && cluster_id = {cluster_id}',
        )
        return record
    except Exception as e:
        logger.error("Error upserting cluster metadata: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
