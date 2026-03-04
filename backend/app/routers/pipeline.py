"""Pipeline management endpoints."""

import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from app.constants import DEFAULT_N_SAMPLES, DEFAULT_TAG_THRESHOLD
from app.services.pipeline_manager import pipeline_manager
from app.services.pocketbase_client import pb_client
from app.services.job_queue import job_queue, JobType

router = APIRouter()


class PipelineInitRequest(BaseModel):
    force_reprocess: bool = False
    weight_by_parents: bool = False


class PipelineRunRequest(BaseModel):
    run_tagging: bool = True
    run_clustering: bool = True
    model: str | None = None
    n_samples: int = DEFAULT_N_SAMPLES
    threshold: int = DEFAULT_TAG_THRESHOLD


@router.get("/status-all")
async def get_all_status():
    """Get status for all available years with PocketBase counts."""
    years = pipeline_manager.get_available_years()
    result = []
    for year in years:
        # Auto-load CSV if available
        try:
            await pipeline_manager.ensure_loaded(year)
        except Exception:
            pass  # No CSV for this year — continue with unloaded state

        status = pipeline_manager.get_status(year)

        # Query PocketBase for persisted counts
        try:
            tagging_count = await pb_client.count(
                "tagging_results", f'year = "{year}"'
            )
        except Exception:
            tagging_count = 0
        try:
            clustering_count = await pb_client.count(
                "clustering_results", f'year = "{year}"'
            )
        except Exception:
            clustering_count = 0

        # Reconcile in-memory flags with PocketBase state
        if status["loaded"] and not status["tagging_complete"] and tagging_count > 0:
            pipeline_manager.set_tagging_complete(year)
            status["tagging_complete"] = True
        if status["loaded"] and not status["clustering_complete"] and clustering_count > 0:
            pipeline_manager.set_clustering_complete(year)
            status["clustering_complete"] = True

        result.append({
            "year": year,
            "loaded": status["loaded"],
            "row_count": status["row_count"],
            "tagging_count": tagging_count,
            "tagging_complete": status["tagging_complete"],
            "clustering_count": clustering_count,
            "clustering_complete": status["clustering_complete"],
            "embeddings_complete": status["embeddings_complete"],
        })
    return {"years": result}


@router.get("/years")
async def list_years():
    """List available survey years."""
    years = pipeline_manager.get_available_years()
    return {"years": years}


@router.post("/init/{year}")
async def init_pipeline(year: str, request: PipelineInitRequest):
    """Initialize pipeline for a specific year."""
    try:
        pipeline = pipeline_manager.init_pipeline(year)

        # Load data
        df = await pipeline_manager.load_data(
            year,
            force_reprocess=request.force_reprocess,
            weight_by_parents=request.weight_by_parents,
        )

        return {
            "year": year,
            "status": "initialized",
            "row_count": len(df),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{year}")
async def get_pipeline_status(year: str):
    """Get pipeline status for a year, auto-loading CSV and reconciling with PocketBase."""
    # Auto-load CSV if available
    try:
        await pipeline_manager.ensure_loaded(year)
    except Exception:
        pass  # No CSV for this year — return unloaded state

    status = pipeline_manager.get_status(year)

    # Reconcile in-memory flags with PocketBase state
    if status["loaded"]:
        if not status["tagging_complete"]:
            try:
                tagging_count = await pb_client.count(
                    "tagging_results", f'year = "{year}"'
                )
                if tagging_count > 0:
                    pipeline_manager.set_tagging_complete(year)
                    status["tagging_complete"] = True
            except Exception:
                pass
        if not status["clustering_complete"]:
            try:
                clustering_count = await pb_client.count(
                    "clustering_results", f'year = "{year}"'
                )
                if clustering_count > 0:
                    pipeline_manager.set_clustering_complete(year)
                    status["clustering_complete"] = True
            except Exception:
                pass

    return status


@router.post("/run/{year}")
async def run_full_pipeline(
    year: str,
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
):
    """Run the full pipeline as a background job."""
    pipeline = await pipeline_manager.ensure_loaded(year)

    # Import here to avoid circular imports
    from app.core.pipeline import run_full_pipeline_async

    # Submit the job
    job_id = await job_queue.submit(
        JobType.FULL_PIPELINE,
        year,
        run_full_pipeline_async,
        year,
        run_tagging=request.run_tagging,
        run_clustering=request.run_clustering,
        model=request.model,
        n_samples=request.n_samples,
        threshold=request.threshold,
    )

    return {
        "job_id": job_id,
        "year": year,
        "status": "started",
    }
