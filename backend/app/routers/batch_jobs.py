"""Batch job query and management endpoints."""

import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from app.services.pocketbase_client import pb_client

logger = logging.getLogger(__name__)

router = APIRouter()

TERMINAL_STATES = {"completed", "failed", "expired", "cancelled", "superseded"}


@router.get("")
async def list_batch_jobs(
    year: str | None = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    """List batch jobs, newest first."""
    try:
        filters = []
        if year:
            filters.append(f'year = "{year}"')

        filter_str = " && ".join(filters) if filters else None

        results = await pb_client.get_list(
            "batch_jobs",
            page=page,
            per_page=per_page,
            filter_str=filter_str,
            sort="-created",
        )
        return results
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("")
async def clear_all_batch_jobs(year: str | None = None):
    """Force-delete all batch_jobs records, cancelling any active OpenAI batches first."""
    from app.services.openai_batch import openai_batch_client

    try:
        filter_str = f'year = "{year}"' if year else None
        jobs = await pb_client.get_full_list("batch_jobs", filter_str=filter_str)

        if not jobs:
            return {"status": "ok", "deleted": 0, "cancelled": 0}

        deleted = 0
        cancelled = 0

        for job in jobs:
            # Best-effort cancel active OpenAI batches
            if job["status"] not in TERMINAL_STATES:
                openai_batch_id = job.get("openai_batch_id")
                if openai_batch_id:
                    try:
                        await openai_batch_client.cancel_batch(openai_batch_id)
                    except Exception as e:
                        logger.warning("Could not cancel OpenAI batch %s: %s", openai_batch_id, e)
                cancelled += 1

            await pb_client.delete("batch_jobs", job["id"])
            deleted += 1

        return {"status": "ok", "deleted": deleted, "cancelled": cancelled}

    except Exception as e:
        logger.error("Error clearing batch jobs: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{job_id}")
async def get_batch_job(job_id: str):
    """Get single batch job details."""
    try:
        result = await pb_client.get_list(
            "batch_jobs",
            filter_str=f'id = "{job_id}"',
            per_page=1,
        )

        if not result.get("items"):
            raise HTTPException(
                status_code=404,
                detail=f"Batch job {job_id} not found",
            )

        return result["items"][0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/groups/{batch_group_id}")
async def delete_batch_group(batch_group_id: str):
    """Delete all batch_jobs records in a group. All must be in terminal states."""
    try:
        jobs = await pb_client.get_full_list(
            "batch_jobs",
            filter_str=f'batch_group_id = "{batch_group_id}"',
        )

        if not jobs:
            raise HTTPException(status_code=404, detail=f"No jobs found for group {batch_group_id}")

        non_terminal = [j for j in jobs if j["status"] not in TERMINAL_STATES]
        if non_terminal:
            raise HTTPException(
                status_code=400,
                detail=f"{len(non_terminal)} job(s) are still active. Wait for completion or cancel first.",
            )

        deleted = 0
        for job in jobs:
            await pb_client.delete("batch_jobs", job["id"])
            deleted += 1

        return {"status": "deleted", "deleted": deleted}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting batch group %s: %s", batch_group_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{job_id}")
async def delete_batch_job(job_id: str):
    """Delete a single batch_jobs record. Only allowed for terminal states."""
    try:
        result = await pb_client.get_list(
            "batch_jobs",
            filter_str=f'id = "{job_id}"',
            per_page=1,
        )

        if not result.get("items"):
            raise HTTPException(status_code=404, detail=f"Batch job {job_id} not found")

        job = result["items"][0]
        if job["status"] not in TERMINAL_STATES:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete job in '{job['status']}' state. Must be in a terminal state.",
            )

        await pb_client.delete("batch_jobs", job_id)
        return {"status": "deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting batch job %s: %s", job_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{job_id}/cancel")
async def cancel_batch_job(job_id: str):
    """Cancel an in-progress batch via OpenAI API."""
    from app.services.openai_batch import openai_batch_client

    try:
        result = await pb_client.get_list(
            "batch_jobs",
            filter_str=f'id = "{job_id}"',
            per_page=1,
        )

        if not result.get("items"):
            raise HTTPException(status_code=404, detail=f"Batch job {job_id} not found")

        job = result["items"][0]
        if job["status"] in TERMINAL_STATES:
            raise HTTPException(
                status_code=400,
                detail=f"Job already in terminal state: {job['status']}",
            )

        openai_batch_id = job.get("openai_batch_id")
        if not openai_batch_id:
            raise HTTPException(
                status_code=400,
                detail="No OpenAI batch ID associated with this job",
            )

        # Cancel via OpenAI API
        cancel_result = await openai_batch_client.cancel_batch(openai_batch_id)

        # Update PocketBase record
        await pb_client.update("batch_jobs", job_id, {
            "status": "cancelled",
            "completed_at": datetime.utcnow().isoformat(),
        })

        return {
            "status": "cancelled",
            "batch_id": openai_batch_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error cancelling batch job %s: %s", job_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{job_id}/poll")
async def poll_batch_job(job_id: str):
    """Immediately check OpenAI batch status and update PocketBase record."""
    from app.services.openai_batch import openai_batch_client

    try:
        result = await pb_client.get_list(
            "batch_jobs",
            filter_str=f'id = "{job_id}"',
            per_page=1,
        )

        if not result.get("items"):
            raise HTTPException(status_code=404, detail=f"Batch job {job_id} not found")

        job = result["items"][0]
        openai_batch_id = job.get("openai_batch_id")
        if not openai_batch_id:
            raise HTTPException(
                status_code=400,
                detail="No OpenAI batch ID associated with this job",
            )

        # Check current status from OpenAI
        status = await openai_batch_client.check_batch_status(openai_batch_id)
        request_counts = status.get("request_counts", {})

        # Update PocketBase record
        update_data: dict = {
            "status": status["status"],
            "total_items": request_counts.get("total", 0),
            "processed_items": request_counts.get("completed", 0),
            "failed_items": request_counts.get("failed", 0),
        }

        if status.get("output_file_id"):
            update_data["output_file_id"] = status["output_file_id"]
        if status.get("error_file_id"):
            update_data["error_file_id"] = status["error_file_id"]
        if status["status"] in TERMINAL_STATES and not job.get("completed_at"):
            update_data["completed_at"] = datetime.utcnow().isoformat()

        await pb_client.update("batch_jobs", job_id, update_data)

        # Return updated record
        updated = await pb_client.get_list(
            "batch_jobs",
            filter_str=f'id = "{job_id}"',
            per_page=1,
        )
        return updated["items"][0]

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error polling batch job %s: %s", job_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/process-all")
async def process_all_completed_batches(year: str | None = None):
    """Download and store results for all completed batches missing tagging results.

    Recovers orphaned batches that completed while the server was down.
    Safe to call repeatedly — results are upserted idempotently.
    """
    from app.services.openai_batch import openai_batch_client
    from app.services.job_queue import process_completed_batch

    try:
        filters = ['status = "completed"']
        if year:
            filters.append(f'year = "{year}"')
        filter_str = " && ".join(filters)

        jobs = await pb_client.get_full_list("batch_jobs", filter_str=filter_str)

        if not jobs:
            return {"status": "ok", "processed": 0, "message": "No completed batches found"}

        results = []
        for job in jobs:
            openai_batch_id = job.get("openai_batch_id")
            output_file_id = job.get("output_file_id")

            if not output_file_id:
                if not openai_batch_id:
                    results.append({"job_id": job["id"], "status": "skipped", "reason": "no batch ID"})
                    continue
                try:
                    status = await openai_batch_client.check_batch_status(openai_batch_id)
                    output_file_id = status.get("output_file_id")
                    if not output_file_id:
                        results.append({"job_id": job["id"], "status": "skipped", "reason": "no output file"})
                        continue
                    # Save the output_file_id we discovered
                    await pb_client.update("batch_jobs", job["id"], {
                        "output_file_id": output_file_id,
                    })
                except Exception as e:
                    results.append({"job_id": job["id"], "status": "error", "reason": str(e)[:200]})
                    continue

            try:
                stored, failed = await process_completed_batch(
                    output_file_id,
                    job["year"],
                    openai_batch_id or "unknown",
                    job.get("model_used", "unknown"),
                )
                await pb_client.update("batch_jobs", job["id"], {
                    "processed_items": stored,
                    "failed_items": failed,
                })
                results.append({"job_id": job["id"], "status": "processed", "stored": stored, "failed": failed})
            except Exception as e:
                logger.error("Error processing batch %s: %s", job["id"], e, exc_info=True)
                results.append({"job_id": job["id"], "status": "error", "reason": str(e)[:200]})

        total_stored = sum(r.get("stored", 0) for r in results)
        return {
            "status": "ok",
            "processed": len([r for r in results if r["status"] == "processed"]),
            "total_stored": total_stored,
            "details": results,
        }

    except Exception as e:
        logger.error("Error in process-all: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{job_id}/process")
async def process_batch_results(job_id: str):
    """Download and store results for a single completed batch.

    Use this to recover results from a batch that completed while the
    server was down, or to re-process results. Upserts are idempotent.
    """
    from app.services.openai_batch import openai_batch_client
    from app.services.job_queue import process_completed_batch

    try:
        result = await pb_client.get_list(
            "batch_jobs", filter_str=f'id = "{job_id}"', per_page=1
        )
        if not result.get("items"):
            raise HTTPException(status_code=404, detail=f"Batch job {job_id} not found")

        job = result["items"][0]
        openai_batch_id = job.get("openai_batch_id")
        output_file_id = job.get("output_file_id")

        if not output_file_id:
            if not openai_batch_id:
                raise HTTPException(
                    status_code=400, detail="No OpenAI batch ID or output file on record"
                )
            # Fetch output_file_id from OpenAI
            status = await openai_batch_client.check_batch_status(openai_batch_id)
            if status["status"] != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch not completed (status: {status['status']})",
                )
            output_file_id = status.get("output_file_id")
            if not output_file_id:
                raise HTTPException(
                    status_code=400, detail="Batch completed but has no output file"
                )
            # Persist the discovered output_file_id
            await pb_client.update("batch_jobs", job_id, {
                "output_file_id": output_file_id,
            })

        stored, failed = await process_completed_batch(
            output_file_id,
            job["year"],
            openai_batch_id or "unknown",
            job.get("model_used", "unknown"),
        )

        await pb_client.update("batch_jobs", job_id, {
            "status": "completed",
            "processed_items": stored,
            "failed_items": failed,
            "output_file_id": output_file_id,
        })

        return {"status": "processed", "stored": stored, "failed": failed}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing batch job %s: %s", job_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{job_id}/retry")
async def retry_batch_job(job_id: str):
    """
    Retry a failed batch by splitting it in half and resubmitting.

    Downloads the original input file, splits into two halves,
    uploads both, and submits them sequentially via the orchestrator.
    """
    from app.services.openai_batch import openai_batch_client
    from app.services.job_queue import batch_orchestrator

    try:
        result = await pb_client.get_list(
            "batch_jobs",
            filter_str=f'id = "{job_id}"',
            per_page=1,
        )

        if not result.get("items"):
            raise HTTPException(status_code=404, detail=f"Batch job {job_id} not found")

        job = result["items"][0]
        if job["status"] != "failed":
            raise HTTPException(
                status_code=400,
                detail=f"Can only retry failed jobs. Current status: {job['status']}",
            )

        input_file_id = job.get("input_file_id")
        if not input_file_id:
            raise HTTPException(
                status_code=400,
                detail="No input file ID to retry from",
            )

        # Download original JSONL
        try:
            content = await openai_batch_client.client.files.content(input_file_id)
            lines = content.text.strip().split("\n")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not download original input file: {e}",
            )

        if len(lines) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Input file only has {len(lines)} request(s), cannot split further",
            )

        # Split in half
        mid = len(lines) // 2
        halves = [lines[:mid], lines[mid:]]

        # Upload both halves as new files
        import tempfile
        from pathlib import Path

        new_file_ids = []
        for half in halves:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                f.write("\n".join(half))
                temp_path = f.name
            try:
                with open(temp_path, "rb") as f:
                    file_response = await openai_batch_client.client.files.create(
                        file=f, purpose="batch"
                    )
                new_file_ids.append(file_response.id)
            finally:
                try:
                    Path(temp_path).unlink()
                except OSError:
                    pass

        # Mark original as superseded
        await pb_client.update("batch_jobs", job_id, {
            "status": "superseded",
        })

        # Build batch_files tuples for orchestrator
        year = job["year"]
        model_used = job.get("model_used", "unknown")
        batch_group_id = str(uuid4())

        batch_files = []
        for i, fid in enumerate(new_file_ids):
            # Each line is one request = one response (n completions handled via n parameter)
            est_responses = len(halves[i])
            est_tokens = (job.get("estimated_tokens", 0) // 2) or 1
            batch_files.append((fid, est_responses, est_tokens))

        sub_batches = await batch_orchestrator.start(
            year=year,
            batch_files=batch_files,
            batch_group_id=batch_group_id,
            model_used=model_used,
        )

        return {
            "status": "retrying",
            "original_job_id": job_id,
            "new_batches": len(sub_batches),
            "batch_group_id": batch_group_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrying batch job %s: %s", job_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
