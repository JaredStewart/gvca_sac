"""Tagging operation endpoints."""

import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from openai import APIError, AuthenticationError, RateLimitError
from pydantic import BaseModel

from app.config import get_settings
from app.constants import DEFAULT_N_SAMPLES, DEFAULT_TAG_THRESHOLD, STABILITY_THRESHOLD
from app.core.survey_config import get_taxonomy_tags
from app.services.job_queue import job_queue, JobType, batch_polling_manager, batch_orchestrator
from app.services.pipeline_manager import pipeline_manager
from app.services.pocketbase_client import pb_client

logger = logging.getLogger(__name__)

router = APIRouter()


class TaggingStartRequest(BaseModel):
    model: str | None = None
    n_samples: int = DEFAULT_N_SAMPLES
    threshold: int = DEFAULT_TAG_THRESHOLD
    use_batch_api: bool = False
    test_mode: bool = False
    test_size: int = 20


class BatchTaggingRequest(BaseModel):
    """Request for batch tagging via OpenAI Batch API."""
    retag_existing: bool = False
    model: str | None = None


class TagModifyRequest(BaseModel):
    """Request for manually modifying tags."""
    tags: list[str]
    reason: str | None = None


class TagToggleRequest(BaseModel):
    """Request for toggling a single tag on a response (US2)."""
    tag: str
    value: bool  # True to add, False to remove


@router.post("/{year}/start")
async def start_tagging(year: str, request: TaggingStartRequest):
    """Start tagging job for a year."""
    pipeline = await pipeline_manager.ensure_loaded(year)

    # Import here to avoid circular imports
    from app.core.tagging import run_tagging_async

    # Submit the job
    job_id = await job_queue.submit(
        JobType.TAGGING,
        year,
        run_tagging_async,
        data=pipeline.data,
        model=request.model,
        n_samples=request.n_samples,
        threshold=request.threshold,
        use_batch_api=request.use_batch_api,
        test_mode=request.test_mode,
        test_size=request.test_size,
    )

    return {
        "job_id": job_id,
        "year": year,
        "status": "started",
    }


@router.get("/{year}/results")
async def get_tagging_results(
    year: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    level: str | None = None,
    tag: str | None = None,
    question: str | None = None,
):
    """Get tagging results for a year."""
    # Build filter
    filters = [f'year = "{year}"']
    if level:
        filters.append(f'level ~ "{level}"')
    if tag:
        filters.append(f'llm_tags ~ "{tag}"')
    if question:
        filters.append(f'question ~ "{question}"')

    filter_str = " && ".join(filters)

    try:
        results = await pb_client.get_list(
            "tagging_results",
            page=page,
            per_page=per_page,
            filter_str=filter_str,
            sort="-created",
        )
        return results
    except Exception as e:
        logger.error("Error fetching tagging results: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/results/{response_id}")
async def get_response_tags(year: str, response_id: str):
    """Get tagging results for a specific response."""
    filter_str = f'year = "{year}" && response_id = "{response_id}"'

    try:
        results = await pb_client.get_list(
            "tagging_results",
            filter_str=filter_str,
        )
        if not results.get("items"):
            raise HTTPException(status_code=404, detail="Response not found")
        return results["items"]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching response tags: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# Map common question text fragments to stored question identifiers.
# Batch tagging stores question as "Q8"/"Q9" (from PocketBase free_responses),
# while the frontend sends human-readable fragments like "good choice".
_QUESTION_FILTER_MAP = {
    "good choice": "Q8",
    "better serve": "Q9",
}


@router.get("/{year}/distribution")
async def get_tag_distribution(year: str, level: str | None = None, question: str | None = None):
    """Get tag distribution for a year."""
    filters = [f'year = "{year}"']
    if level:
        filters.append(f'level ~ "{level}"')
    if question:
        mapped_q = _QUESTION_FILTER_MAP.get(question.lower().strip())
        if mapped_q:
            # Match both Q-code format (batch path) and full question text (direct path)
            filters.append(f'(question = "{mapped_q}" || question ~ "{question}")')
        else:
            filters.append(f'question ~ "{question}"')
    filter_str = " && ".join(filters)

    try:
        # Get all tagging results for this year
        results = await pb_client.get_full_list(
            "tagging_results",
            filter_str=filter_str,
        )

        # Aggregate tags
        tag_counts: dict[str, int] = {}
        total = 0

        for result in results:
            tags = result.get("llm_tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    total += 1

        # Safety net: filter to valid taxonomy tags only
        valid_tags = set(get_taxonomy_tags())
        invalid_total = sum(c for t, c in tag_counts.items() if t not in valid_tags)
        tag_counts = {tag: count for tag, count in tag_counts.items() if tag in valid_tags}
        total -= invalid_total

        # Sort by count descending
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "year": year,
            "level": level,
            "total_tags": total,
            "unique_tags": len(tag_counts),
            "distribution": [
                {
                    "tag": tag,
                    "count": count,
                    "percentage": round(count / total * 100, 2)
                    if total > 0
                    else 0,
                }
                for tag, count in sorted_tags
            ],
        }
    except Exception as e:
        logger.error("Error fetching tag distribution: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ================ Batch API Tagging (FR-012) ================


@router.post("/{year}/batch")
async def start_batch_tagging(year: str, request: BatchTaggingRequest):
    """
    Start batch AI tagging via OpenAI Batch API (FR-012).

    Submits free responses to the Batch API for cost-effective
    asynchronous processing. Automatically splits into multiple
    sub-batches if the estimated token count exceeds OpenAI limits.
    Responses are processed with 4 samples each for stability scoring.
    """
    from app.services.openai_batch import openai_batch_client

    settings = get_settings()
    model_to_use = request.model or settings.default_llm_model

    try:
        # Get free responses to tag
        if request.retag_existing:
            free_responses = await pb_client.get_full_list(
                "free_responses",
                filter_str=f'year = "{year}"',
            )
        else:
            free_responses = await pb_client.get_untagged_free_responses(year)

        if not free_responses:
            raise HTTPException(
                status_code=404,
                detail=f"No responses to tag for year {year}",
            )

        # Create batch input files (may split into multiple batches)
        batch_files = await openai_batch_client.create_batch_input_files(
            free_responses, model=model_to_use
        )

        # Generate a group ID to link sub-batches together
        batch_group_id = str(uuid4())

        # Use orchestrator for sequential submission with backpressure
        sub_batches = await batch_orchestrator.start(
            year=year,
            batch_files=batch_files,
            batch_group_id=batch_group_id,
            model_used=model_to_use,
        )

        return {
            "job_id": sub_batches[0]["pb_record_id"] if len(sub_batches) == 1 else batch_group_id,
            "job_type": "tagging_batch",
            "status": "queued",
            "total_items": len(free_responses),
            "sub_batches": len(sub_batches),
            "batch_group_id": batch_group_id,
            "model": model_to_use,
            "message": f"Batch tagging queued ({len(sub_batches)} batch(es), {len(free_responses)} responses). Batches will be submitted sequentially.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch tagging for year {year}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/batch/{batch_job_id}")
async def get_batch_status(year: str, batch_job_id: str):
    """
    Get status of a batch tagging job.

    Returns current status from OpenAI and progress information.
    """
    from app.services.openai_batch import openai_batch_client

    try:
        # Get batch job record from PocketBase
        results = await pb_client.get_list(
            "batch_jobs",
            filter_str=f'id = "{batch_job_id}"',
            per_page=1,
        )

        if not results.get("items"):
            raise HTTPException(
                status_code=404,
                detail=f"Batch job {batch_job_id} not found",
            )

        job = results["items"][0]
        openai_batch_id = job.get("openai_batch_id")

        # Check current status from OpenAI
        openai_status = None
        if openai_batch_id:
            try:
                openai_status = await openai_batch_client.check_batch_status(openai_batch_id)
            except Exception as e:
                logger.warning(f"Could not fetch OpenAI batch status: {e}")

        return {
            "job_id": batch_job_id,
            "year": job.get("year"),
            "status": job.get("status"),
            "total_items": job.get("total_items"),
            "processed_items": job.get("processed_items"),
            "failed_items": job.get("failed_items"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "openai_status": openai_status,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ================ Single Response Tagging (FR-013) ================


@router.post("/{year}/single/{response_id}")
async def tag_single_response(
    year: str,
    response_id: str,
    model: str | None = Query(None, description="LLM model to use (defaults to server config)"),
):
    """
    Tag a single response via direct API call (FR-013).

    Makes one OpenAI request with n=4 samples for proper stability scoring.
    """
    from app.core.tagging import tag_response_with_stability, detect_keyword_mismatches

    # Get the free response
    free_response = await pb_client.get_free_response_by_id(response_id)

    if not free_response:
        raise HTTPException(
            status_code=404,
            detail=f"Response {response_id} not found",
        )

    if free_response["year"] != year:
        raise HTTPException(
            status_code=400,
            detail=f"Response {response_id} belongs to year {free_response['year']}, not {year}",
        )

    # Verify OpenAI API key is configured
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured. Set OPENAI_API_KEY in your .env file.",
        )

    model_used = model or settings.default_llm_model

    try:
        result = await tag_response_with_stability(
            response_text=free_response["response_text"],
            question=free_response["question"],
            level=free_response.get("level"),
            model=model_used,
            n_samples=DEFAULT_N_SAMPLES,
        )
    except AuthenticationError:
        raise HTTPException(
            status_code=503,
            detail="Invalid OpenAI API key. Check OPENAI_API_KEY in your .env file.",
        )
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="OpenAI rate limit exceeded. Please wait and try again.",
        )
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error: {e.message}",
        )

    llm_tags = result["llm_tags"]
    tag_votes = result["tag_votes"]
    stability_score = result["stability_score"]
    n_samples = result["n_samples"]

    if n_samples == 0:
        raise HTTPException(
            status_code=502,
            detail="LLM tagging failed: no response received from OpenAI.",
        )

    # Check for keyword mismatches
    mismatches = detect_keyword_mismatches(
        free_response["response_text"],
        llm_tags,
    )

    review_status = "pending" if mismatches else None

    # Save/update tagging result
    tagging_data = {
        "year": year,
        "response_id": response_id,
        "question": free_response["question"],
        "level": free_response.get("level"),
        "response_text": free_response["response_text"],
        "llm_tags": llm_tags,
        "tag_votes": tag_votes,
        "stability_score": stability_score,
        "keyword_mismatches": mismatches,
        "review_status": review_status,
        "model_used": model_used,
        "n_samples": n_samples,
    }

    try:
        await pb_client.upsert(
            "tagging_results",
            tagging_data,
            filter_str=f'year = "{year}" && response_id = "{response_id}"',
        )
    except Exception as e:
        logger.error(f"Error saving tagging result for {response_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save tagging result")

    return {
        "response_id": response_id,
        "llm_tags": llm_tags,
        "stability_score": stability_score,
        "keyword_mismatches": mismatches,
        "review_status": review_status,
    }


# ================ Tag Review Workflow (FR-017, FR-018, FR-019, FR-020) ================


@router.get("/{year}/review")
async def get_review_queue(
    year: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    stability_threshold: float = Query(0.75, ge=0, le=1),
):
    """
    Get responses needing tag review (FR-017).

    Returns responses with low stability or keyword mismatches,
    sorted by review priority.
    """
    try:
        result = await pb_client.get_tagging_results_for_review(
            year=year,
            stability_threshold=stability_threshold,
            page=page,
            per_page=per_page,
        )

        # Add review flags and priority to each item
        for item in result.get("items", []):
            flags = []
            priority = 0

            stability = item.get("stability_score", 1.0) or 1.0
            if stability < stability_threshold:
                flags.append("low_stability")
                priority += (1 - stability) * 100

            mismatches = item.get("keyword_mismatches", []) or []
            if mismatches:
                flags.append("keyword_mismatch")
                priority += len(mismatches) * 10

            item["review_flags"] = flags
            item["review_priority"] = round(priority, 2)

        return result

    except Exception as e:
        logger.error(f"Error fetching review queue for year {year}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{year}/review/{response_id}/approve")
async def approve_tags(year: str, response_id: str):
    """
    Approve tags and hide from review queue (FR-020).
    """
    try:
        # Find the tagging result
        results = await pb_client.get_list(
            "tagging_results",
            filter_str=f'year = "{year}" && response_id = "{response_id}"',
            per_page=1,
        )

        if not results.get("items"):
            raise HTTPException(
                status_code=404,
                detail=f"Tagging result for {response_id} not found",
            )

        record_id = results["items"][0]["id"]
        await pb_client.update_review_status(record_id, "approved")

        return {"status": "approved", "response_id": response_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving tags for {response_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{year}/review/{response_id}/hide")
async def hide_from_review(year: str, response_id: str):
    """
    Hide from review queue without approval (FR-020).
    """
    try:
        results = await pb_client.get_list(
            "tagging_results",
            filter_str=f'year = "{year}" && response_id = "{response_id}"',
            per_page=1,
        )

        if not results.get("items"):
            raise HTTPException(
                status_code=404,
                detail=f"Tagging result for {response_id} not found",
            )

        record_id = results["items"][0]["id"]
        await pb_client.update_review_status(record_id, "hidden")

        return {"status": "hidden", "response_id": response_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error hiding {response_id} from review: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{year}/review/{response_id}/modify")
async def modify_tags(year: str, response_id: str, request: TagModifyRequest):
    """
    Modify tags manually and mark as approved (FR-016, FR-020).
    """
    try:
        # Find the tagging result
        results = await pb_client.get_list(
            "tagging_results",
            filter_str=f'year = "{year}" && response_id = "{response_id}"',
            per_page=1,
        )

        if not results.get("items"):
            raise HTTPException(
                status_code=404,
                detail=f"Tagging result for {response_id} not found",
            )

        tagging_result = results["items"][0]
        original_tags = tagging_result.get("llm_tags", [])

        # Create tag override record
        override_data = {
            "year": year,
            "response_id": response_id,
            "question": tagging_result.get("question"),
            "original_tags": original_tags,
            "modified_tags": request.tags,
            "reason": request.reason,
            "modified_at": datetime.utcnow().isoformat(),
        }
        await pb_client.create("tag_overrides", override_data)

        # Update tagging result with new tags and approved status
        await pb_client.update(
            "tagging_results",
            tagging_result["id"],
            {
                "llm_tags": request.tags,
                "review_status": "approved",
            },
        )

        return {
            "response_id": response_id,
            "original_tags": original_tags,
            "modified_tags": request.tags,
            "reason": request.reason,
            "status": "approved",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error modifying tags for {response_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ================ Tag-by-Tag Workflow ================


@router.get("/{year}/by-tag/{tag}")
async def get_responses_by_tag(
    year: str,
    tag: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    include_dismissed: bool = Query(False, description="Include dismissed responses"),
    sort: str = Query("-stability_score", description="Sort field (prefix with - for desc)"),
):
    """
    Get responses with a specific tag for tag-by-tag review workflow.

    Returns responses sorted by stability (high first by default).
    """
    try:
        # Build filter
        filters = [f'year = "{year}"', f'llm_tags ~ "{tag}"']
        if not include_dismissed:
            filters.append('(dismissed = false || dismissed = null)')

        filter_str = " && ".join(filters)

        results = await pb_client.get_list(
            "tagging_results",
            page=page,
            per_page=per_page,
            filter_str=filter_str,
            sort=sort,
        )

        return results

    except Exception as e:
        logger.error(f"Error fetching responses by tag {tag} for year {year}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{year}/responses/{response_id}/dismiss")
async def dismiss_response(year: str, response_id: str):
    """
    Mark a response as dismissed (hide from active tag review list).
    """
    try:
        # Find the tagging result
        results = await pb_client.get_list(
            "tagging_results",
            filter_str=f'year = "{year}" && response_id = "{response_id}"',
            per_page=1,
        )

        if not results.get("items"):
            raise HTTPException(
                status_code=404,
                detail=f"Tagging result for {response_id} not found",
            )

        record_id = results["items"][0]["id"]
        await pb_client.update(
            "tagging_results",
            record_id,
            {
                "dismissed": True,
                "dismissed_at": datetime.utcnow().isoformat(),
            },
        )

        return {"status": "dismissed", "response_id": response_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error dismissing response {response_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{year}/responses/{response_id}/undismiss")
async def undismiss_response(year: str, response_id: str):
    """
    Un-dismiss a response (show again in active tag review list).
    """
    try:
        # Find the tagging result
        results = await pb_client.get_list(
            "tagging_results",
            filter_str=f'year = "{year}" && response_id = "{response_id}"',
            per_page=1,
        )

        if not results.get("items"):
            raise HTTPException(
                status_code=404,
                detail=f"Tagging result for {response_id} not found",
            )

        record_id = results["items"][0]["id"]
        await pb_client.update(
            "tagging_results",
            record_id,
            {
                "dismissed": False,
                "dismissed_at": None,
            },
        )

        return {"status": "undismissed", "response_id": response_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error undismissing response {response_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ================ Tag Toggle for Inline Editing (US2) ================


@router.put("/{year}/responses/{response_id}/tags")
async def toggle_tag(year: str, response_id: str, request: TagToggleRequest):
    """
    Toggle a single tag on a response (US2 - Edit Tags Inline).

    Adds or removes a tag and creates a tag_override record to preserve
    the original tagging history.
    """
    try:
        # First check if there's an existing tagging result
        tagging_results = await pb_client.get_list(
            "tagging_results",
            filter_str=f'year = "{year}" && response_id = "{response_id}"',
            per_page=1,
        )

        # If no tagging result exists, we need to get the free response
        # and create a tagging result first
        if not tagging_results.get("items"):
            # Get the free response
            free_response = await pb_client.get_free_response_by_id(response_id)
            if not free_response:
                raise HTTPException(
                    status_code=404,
                    detail=f"Response {response_id} not found",
                )

            if free_response["year"] != year:
                raise HTTPException(
                    status_code=400,
                    detail=f"Response {response_id} belongs to year {free_response['year']}, not {year}",
                )

            # Create empty tagging result
            original_tags: list[str] = []
            tagging_data = {
                "year": year,
                "response_id": response_id,
                "question": free_response["question"],
                "level": free_response.get("level") or "",
                "response_text": free_response["response_text"],
                "llm_tags": [],
                "tag_votes": {},
                "stability_score": 0.0,  # Use 0.0 instead of None to avoid null issues
                "keyword_mismatches": [],
                "review_status": "pending",
                "model_used": "manual",
                "n_samples": 0,
            }
            logger.info(f"Creating tagging result for {response_id}: {tagging_data}")
            try:
                tagging_record = await pb_client.create("tagging_results", tagging_data)
            except Exception as e:
                logger.error(f"Error creating tagging result: {e}")
                raise
        else:
            tagging_record = tagging_results["items"][0]
            original_tags = tagging_record.get("llm_tags", []) or []

        # Calculate new tags
        current_tags = list(tagging_record.get("llm_tags", []) or [])
        if request.value:
            # Add tag if not present
            if request.tag not in current_tags:
                current_tags.append(request.tag)
        else:
            # Remove tag if present
            if request.tag in current_tags:
                current_tags.remove(request.tag)

        # Check if there's an existing override
        override_results = await pb_client.get_list(
            "tag_overrides",
            filter_str=f'year = "{year}" && response_id = "{response_id}"',
            per_page=1,
        )

        if override_results.get("items"):
            # Update existing override
            override_record = override_results["items"][0]
            await pb_client.update(
                "tag_overrides",
                override_record["id"],
                {
                    "modified_tags": current_tags,
                    "modified_at": datetime.utcnow().isoformat(),
                },
            )
        else:
            # Create new override record
            override_data = {
                "year": year,
                "response_id": response_id,
                "question": tagging_record.get("question"),
                "original_tags": original_tags,
                "modified_tags": current_tags,
                "reason": "inline_edit",
                "modified_at": datetime.utcnow().isoformat(),
            }
            await pb_client.create("tag_overrides", override_data)

        # Update tagging result with new tags
        await pb_client.update(
            "tagging_results",
            tagging_record["id"],
            {"llm_tags": current_tags},
        )

        return {
            "response_id": response_id,
            "tag": request.tag,
            "value": request.value,
            "tags": current_tags,
            "original_tags": original_tags,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling tag for {response_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{year}/tag-summary")
async def get_tag_summary(year: str):
    """
    Get summary of tags with counts of pending (non-dismissed) and dismissed responses.

    Useful for showing progress in the tag-by-tag workflow.
    """
    try:
        # Get all tagging results for the year
        all_results = await pb_client.get_full_list(
            "tagging_results",
            filter_str=f'year = "{year}"',
        )

        # Aggregate by tag
        tag_summary: dict[str, dict[str, int]] = {}

        for result in all_results:
            tags = result.get("llm_tags", [])
            is_dismissed = result.get("dismissed", False)

            for tag in tags if isinstance(tags, list) else []:
                if tag not in tag_summary:
                    tag_summary[tag] = {"pending": 0, "dismissed": 0, "total": 0}

                tag_summary[tag]["total"] += 1
                if is_dismissed:
                    tag_summary[tag]["dismissed"] += 1
                else:
                    tag_summary[tag]["pending"] += 1

        # Sort by pending count (descending) then by name
        sorted_summary = sorted(
            tag_summary.items(),
            key=lambda x: (-x[1]["pending"], x[0]),
        )

        return {
            "year": year,
            "tags": [
                {
                    "name": tag,
                    "pending": counts["pending"],
                    "dismissed": counts["dismissed"],
                    "total": counts["total"],
                }
                for tag, counts in sorted_summary
            ],
        }

    except Exception as e:
        logger.error(f"Error fetching tag summary for year {year}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
