"""Main pipeline orchestration."""

import logging
from typing import Any, Callable, Coroutine

import polars as pl

from app.config import get_settings
from app.constants import DEFAULT_N_SAMPLES, DEFAULT_TAG_THRESHOLD
from app.core.analysis import compute_statistics
from app.core.clustering import run_clustering_async
from app.core.survey_config import SurveyConfig
from app.core.tagging import run_tagging_async
from app.services.pipeline_manager import pipeline_manager

logger = logging.getLogger(__name__)


async def run_full_pipeline_async(
    year: str,
    run_tagging: bool = True,
    run_clustering: bool = True,
    model: str | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    threshold: int = DEFAULT_TAG_THRESHOLD,
    progress_callback: Callable[[int, int], Coroutine[Any, Any, None]] | None = None,
) -> dict[str, Any]:
    """
    Run the full analysis pipeline.

    Args:
        year: Survey year
        run_tagging: Whether to run tagging
        run_clustering: Whether to run clustering
        model: LLM model for tagging
        n_samples: Samples per response for tagging
        threshold: Voting threshold for tagging
        progress_callback: Progress update callback

    Returns:
        Pipeline results summary
    """
    model = model or get_settings().default_llm_model
    results = {"year": year, "steps_completed": []}

    # Get pipeline data
    pipeline = await pipeline_manager.ensure_loaded(year)

    data = pipeline.data

    # Calculate total steps
    total_steps = 1  # statistics always runs
    if run_tagging:
        total_steps += 1
    if run_clustering:
        total_steps += 1

    current_step = 0

    async def update_progress(step: int, total: int):
        if progress_callback:
            overall_progress = current_step + (step / total)
            await progress_callback(int(overall_progress), total_steps)

    # Step 1: Compute statistics
    logger.info(f"Computing statistics for year {year}")
    try:
        stats = compute_statistics(data, pipeline.config)
        results["statistics"] = stats
        results["steps_completed"].append("statistics")
        current_step += 1
        if progress_callback:
            await progress_callback(current_step, total_steps)
    except Exception as e:
        logger.error(f"Error computing statistics: {e}")
        results["statistics_error"] = str(e)

    # Step 2: Tagging
    if run_tagging:
        logger.info(f"Running tagging for year {year}")
        try:
            tagging_result = await run_tagging_async(
                year=year,
                data=data,
                model=model,
                n_samples=n_samples,
                threshold=threshold,
                progress_callback=update_progress,
            )
            results["tagging"] = tagging_result
            results["steps_completed"].append("tagging")
            pipeline_manager.set_tagging_complete(year)
            current_step += 1
            if progress_callback:
                await progress_callback(current_step, total_steps)
        except Exception as e:
            logger.error(f"Error running tagging: {e}")
            results["tagging_error"] = str(e)

    # Step 3: Clustering
    if run_clustering:
        logger.info(f"Running clustering for year {year}")
        try:
            clustering_result = await run_clustering_async(
                year=year,
                data=data,
                progress_callback=update_progress,
            )
            results["clustering"] = clustering_result
            results["steps_completed"].append("clustering")
            pipeline_manager.set_clustering_complete(year)
            current_step += 1
            if progress_callback:
                await progress_callback(current_step, total_steps)
        except Exception as e:
            logger.error(f"Error running clustering: {e}")
            results["clustering_error"] = str(e)

    results["status"] = "completed"
    return results
