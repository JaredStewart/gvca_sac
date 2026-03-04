"""Clustering operation endpoints."""

import logging
import os

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.job_queue import job_queue, JobType
from app.services.pipeline_manager import pipeline_manager
from app.services.pocketbase_client import pb_client

logger = logging.getLogger(__name__)

router = APIRouter()


class EmbeddingStartRequest(BaseModel):
    embed_model: str = "text-embedding-3-small"
    question: str | None = None


class ClusteringStartRequest(BaseModel):
    embed_model: str = "text-embedding-3-small"
    min_cluster_size: int = 5
    question: str | None = None


class ReclusterRequest(BaseModel):
    response_ids: list[str] = Field(..., min_length=5)
    min_cluster_size: int = 3


class SummarizeRequest(BaseModel):
    response_ids: list[str]
    prompt_context: str | None = None


@router.post("/{year}/embed")
async def start_embedding(year: str, request: EmbeddingStartRequest):
    """
    Generate embeddings for free responses without running clustering.

    Pre-generates and caches OpenAI embeddings to Parquet files.
    Useful for warming the embedding cache before running the full
    clustering pipeline.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY not configured. Set it in the .env file.",
        )

    pipeline = await pipeline_manager.ensure_loaded(year)

    from app.core.clustering import run_embedding_async

    job_id = await job_queue.submit(
        JobType.CLUSTERING,
        year,
        run_embedding_async,
        year,
        data=pipeline.data,
        embed_model=request.embed_model,
        question=request.question,
    )

    return {
        "job_id": job_id,
        "year": year,
        "status": "submitted",
    }


@router.post("/{year}/start")
async def start_clustering(year: str, request: ClusteringStartRequest):
    """Start clustering job for a year."""
    # Validate API key is configured
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY not configured. Set it in the .env file.",
        )

    pipeline = await pipeline_manager.ensure_loaded(year)

    # Import here to avoid circular imports
    from app.core.clustering import run_clustering_async

    # Submit the job
    job_id = await job_queue.submit(
        JobType.CLUSTERING,
        year,
        run_clustering_async,
        year,
        data=pipeline.data,
        embed_model=request.embed_model,
        min_cluster_size=request.min_cluster_size,
        question=request.question,
    )

    return {
        "job_id": job_id,
        "year": year,
        "status": "submitted",
    }


@router.post("/{year}/recluster")
async def recluster_selection(year: str, request: ReclusterRequest):
    """
    Re-cluster a subset of responses using stored embeddings.

    Runs UMAP + HDBSCAN on the selected subset. No new API calls.
    Synchronous endpoint (small subset completes in seconds).
    """
    from app.core.clustering import recluster_subset

    try:
        result = await recluster_subset(
            response_ids=request.response_ids,
            year=year,
            min_cluster_size=request.min_cluster_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Enrich coordinates with tag data
    tagging_map = {}
    try:
        tag_results = await pb_client.get_full_list(
            "tagging_results",
            filter_str=f'year = "{year}"',
        )
        for t in tag_results:
            tagging_map[t["response_id"]] = {
                "level": t.get("level", ""),
                "tags": t.get("llm_tags", []),
            }
    except Exception:
        pass  # Enrichment is optional

    for coord in result["coordinates"]:
        if coord["response_id"] in tagging_map:
            tag_data = tagging_map[coord["response_id"]]
            coord["level"] = tag_data["level"]
            coord["tags"] = tag_data["tags"]

    return result


@router.get("/{year}/results")
async def get_clustering_results(
    year: str,
    question: str | None = None,
    cluster_id: int | None = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    """Get clustering results for a year."""
    filters = [f'year = "{year}"']
    if question:
        filters.append(f'question ~ "{question}"')
    if cluster_id is not None:
        filters.append(f"cluster_id = {cluster_id}")

    filter_str = " && ".join(filters)

    try:
        results = await pb_client.get_list(
            "clustering_results",
            page=page,
            per_page=per_page,
            filter_str=filter_str,
            sort="cluster_id",
        )
        return results
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/coordinates")
async def get_cluster_coordinates(
    year: str,
    question: str | None = None,
    include_tags: bool = False,
):
    """Get UMAP 2D coordinates for cluster visualization."""
    filters = [f'year = "{year}"']
    if question:
        filters.append(f'question ~ "{question}"')

    filter_str = " && ".join(filters)

    try:
        results = await pb_client.get_full_list(
            "clustering_results",
            filter_str=filter_str,
        )

        # If include_tags is True, fetch tagging results to enrich with tags
        tagging_map = {}
        if include_tags:
            tag_results = await pb_client.get_full_list(
                "tagging_results",
                filter_str=f'year = "{year}"',
            )
            for t in tag_results:
                tagging_map[t["response_id"]] = {
                    "tags": t.get("llm_tags", []),
                }

        # Transform to coordinate format
        coordinates = []
        for r in results:
            if r.get("umap_x") is None or r.get("umap_y") is None:
                continue

            coord = {
                "response_id": r["response_id"],
                "x": r.get("umap_x"),
                "y": r.get("umap_y"),
                "cluster_id": r.get("cluster_id"),
                # response_text and level stored directly on clustering_results
                "response_text": r.get("response_text", ""),
                "level": r.get("level", ""),
                "question": r.get("question", ""),
            }

            # Add tags from tagging_results if available
            if include_tags and r["response_id"] in tagging_map:
                coord["tags"] = tagging_map[r["response_id"]]["tags"]

            coordinates.append(coord)

        return {
            "year": year,
            "question": question,
            "total": len(coordinates),
            "coordinates": coordinates,
        }
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/summaries")
async def get_cluster_summaries(year: str, question: str | None = None):
    """Get cluster summaries for a year."""
    filters = [f'year = "{year}"']
    if question:
        filters.append(f'question ~ "{question}"')

    filter_str = " && ".join(filters)

    try:
        results = await pb_client.get_full_list(
            "cluster_summaries",
            filter_str=filter_str,
            sort="cluster_id",
        )
        return {
            "year": year,
            "question": question,
            "clusters": results,
        }
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{year}/summarize")
async def summarize_responses(year: str, request: SummarizeRequest):
    """
    Summarize a selection of survey responses using AI.

    This endpoint takes a list of response IDs and returns an AI-generated
    summary highlighting key themes and sentiment.
    """
    from app.core.summarization import summarize_responses as do_summarize

    if not request.response_ids:
        raise HTTPException(status_code=400, detail="No response IDs provided")

    if len(request.response_ids) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 responses can be summarized at once"
        )

    # Fetch the tagging results for these response IDs
    responses = []
    for response_id in request.response_ids:
        filter_str = f'year = "{year}" && response_id = "{response_id}"'
        try:
            results = await pb_client.get_list(
                "tagging_results",
                filter_str=filter_str,
                per_page=1,
            )
            if results.get("items"):
                item = results["items"][0]
                responses.append({
                    "response_id": item["response_id"],
                    "response_text": item.get("response_text", ""),
                    "level": item.get("level", ""),
                    "question": item.get("question", ""),
                    "llm_tags": item.get("llm_tags", []),
                })
        except Exception as e:
            logger.warning(f"Error fetching response {response_id}: {e}")

    if not responses:
        raise HTTPException(
            status_code=404,
            detail="No responses found for the provided IDs"
        )

    # Call the summarization function
    result = await do_summarize(
        responses=responses,
        prompt_context=request.prompt_context,
    )

    return result
