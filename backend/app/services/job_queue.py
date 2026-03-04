"""Background job queue for long-running operations."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    TAGGING = "tagging"
    TAGGING_BATCH = "tagging_batch"
    CLUSTERING = "clustering"
    FULL_PIPELINE = "full_pipeline"
    DATA_LOAD = "data_load"


@dataclass
class Job:
    """Represents a background job."""

    id: str
    job_type: JobType
    year: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    total_items: int = 0
    processed_items: int = 0
    error_message: str | None = None
    result: Any = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type.value,
            "year": self.year,
            "status": self.status.value,
            "progress": self.progress,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


# Type alias for progress callback
ProgressCallback = Callable[[int, int], Coroutine[Any, Any, None]]


class JobQueue:
    """In-process async job queue."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    async def submit(
        self,
        job_type: JobType,
        year: str,
        func: Callable[..., Coroutine],
        *args,
        **kwargs,
    ) -> str:
        """Submit a job to the queue."""
        job_id = str(uuid4())
        job = Job(id=job_id, job_type=job_type, year=year)
        self._jobs[job_id] = job

        # Create progress callback for this job
        async def progress_callback(processed: int, total: int):
            await self.update_progress(job_id, processed, total)

        # Wrap the function to handle job lifecycle
        async def run_job():
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            await self._notify_subscribers(job_id)

            try:
                # Try to call with progress_callback first
                try:
                    result = await func(*args, progress_callback=progress_callback, **kwargs)
                except TypeError as te:
                    # Function doesn't accept progress_callback, call without it
                    if "progress_callback" in str(te):
                        result = await func(*args, **kwargs)
                    else:
                        raise
                job.status = JobStatus.COMPLETED
                job.result = result
                job.progress = 100.0
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                raise
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
            finally:
                job.completed_at = datetime.utcnow()
                await self._notify_subscribers(job_id)

        task = asyncio.create_task(run_job())
        self._tasks[job_id] = task

        return job_id

    async def update_progress(self, job_id: str, processed: int, total: int):
        """Update job progress."""
        if job_id not in self._jobs:
            return

        job = self._jobs[job_id]
        job.processed_items = processed
        job.total_items = total
        job.progress = (processed / total * 100) if total > 0 else 0

        await self._notify_subscribers(job_id)

    async def get_status(self, job_id: str) -> Job | None:
        """Get job status."""
        return self._jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id not in self._tasks:
            return False

        task = self._tasks[job_id]
        if not task.done():
            task.cancel()
            return True
        return False

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to job updates."""
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []

        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id].append(queue)

        # Send current state immediately
        if job_id in self._jobs:
            await queue.put(self._jobs[job_id].to_dict())

        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue):
        """Unsubscribe from job updates."""
        if job_id in self._subscribers:
            try:
                self._subscribers[job_id].remove(queue)
            except ValueError:
                pass

    async def _notify_subscribers(self, job_id: str):
        """Notify all subscribers of a job update."""
        if job_id not in self._subscribers:
            return

        job = self._jobs.get(job_id)
        if not job:
            return

        job_dict = job.to_dict()
        for queue in self._subscribers[job_id]:
            try:
                await queue.put(job_dict)
            except Exception:
                pass

    def get_all_jobs(self, year: str | None = None) -> list[Job]:
        """Get all jobs, optionally filtered by year."""
        jobs = list(self._jobs.values())
        if year:
            jobs = [j for j in jobs if j.year == year]
        return sorted(jobs, key=lambda j: j.started_at or datetime.min, reverse=True)

    async def shutdown(self):
        """Cancel all running jobs and clean up."""
        for task in self._tasks.values():
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)


# Singleton instance
job_queue = JobQueue()


async def process_completed_batch(
    output_file_id: str,
    year: str,
    openai_batch_id: str,
    model_used: str = "unknown",
) -> tuple[int, int]:
    """Download and store tagging results from a completed OpenAI batch.

    Standalone function usable from both the polling loop and manual
    recovery endpoints.

    Returns (stored_count, failed_count).
    """
    import logging
    from app.services.openai_batch import openai_batch_client
    from app.services.pocketbase_client import pb_client
    from app.core.tagging import detect_keyword_mismatches

    logger = logging.getLogger(__name__)

    raw_results = await openai_batch_client.download_results(output_file_id)
    aggregated = openai_batch_client.aggregate_batch_results(raw_results)

    logger.info(f"Aggregated {len(aggregated)} responses from batch {openai_batch_id}")

    free_responses = await pb_client.get_full_list(
        "free_responses", filter_str=f'year = "{year}"'
    )
    response_texts = {r["response_id"]: r for r in free_responses}

    stored_count = 0
    failed_count = 0

    for response_id, result in aggregated.items():
        try:
            free_response = response_texts.get(response_id, {})
            response_text = free_response.get("response_text", "")

            mismatches = []
            if response_text:
                mismatches = detect_keyword_mismatches(
                    response_text, result["llm_tags"]
                )

            review_status = None
            if result["stability_score"] < 0.75 or mismatches:
                review_status = "pending"

            tagging_data = {
                "year": year,
                "response_id": response_id,
                "question": free_response.get("question", ""),
                "level": free_response.get("level", ""),
                "response_text": response_text,
                "llm_tags": result["llm_tags"],
                "tag_votes": result["tag_votes"],
                "stability_score": result["stability_score"],
                "keyword_mismatches": mismatches,
                "review_status": review_status,
                "batch_id": openai_batch_id,
                "model_used": model_used,
                "n_samples": result["n_samples"],
            }

            await pb_client.upsert(
                "tagging_results",
                tagging_data,
                filter_str=f'year = "{year}" && response_id = "{response_id}"',
            )
            stored_count += 1

        except Exception as e:
            logger.error(f"Error storing result for {response_id}: {e}")
            failed_count += 1

    logger.info(
        f"Batch {openai_batch_id}: {stored_count} stored, {failed_count} failed"
    )
    return stored_count, failed_count


class BatchPollingManager:
    """Manages polling for OpenAI batch job completion."""

    def __init__(self, polling_interval: int = 10):
        self.polling_interval = polling_interval
        self._polling_tasks: dict[str, asyncio.Task] = {}
        self._active = True

    async def start_polling(
        self,
        job_id: str,
        batch_id: str,
        year: str,
        metadata: dict[str, Any] | None = None,
        on_complete: Callable[[], Any] | None = None,
    ) -> None:
        """
        Start polling for batch job completion.

        Args:
            job_id: Internal job ID (from job_queue)
            batch_id: OpenAI batch ID
            year: Survey year for storing results
            metadata: Additional metadata to attach to the job (e.g. model_used)
            on_complete: Optional callback invoked when polling finishes (success or failure)
        """
        if job_id in self._polling_tasks:
            return

        # Attach metadata to the job
        if metadata:
            job = await job_queue.get_status(job_id)
            if job:
                job.metadata.update(metadata)

        task = asyncio.create_task(
            self._poll_batch(job_id, batch_id, year, on_complete=on_complete)
        )
        self._polling_tasks[job_id] = task

    async def _poll_batch(
        self,
        job_id: str,
        batch_id: str,
        year: str,
        on_complete: Callable[[], Any] | None = None,
    ) -> None:
        """Poll OpenAI batch status until completion."""
        import logging
        from app.services.openai_batch import openai_batch_client
        from app.services.pocketbase_client import pb_client
        from app.core.tagging import detect_keyword_mismatches

        logger = logging.getLogger(__name__)

        try:
            while self._active:
                # Check batch status
                status = await openai_batch_client.check_batch_status(batch_id)
                batch_status = status.get("status")
                request_counts = status.get("request_counts", {})

                logger.info(f"Batch {batch_id} status: {batch_status}")

                # Update job metadata
                job = await job_queue.get_status(job_id)
                if job:
                    job.metadata["openai_status"] = batch_status
                    job.total_items = request_counts.get("total", 0)
                    job.processed_items = request_counts.get("completed", 0)
                    if job.total_items > 0:
                        job.progress = (job.processed_items / job.total_items) * 100

                # Update PocketBase batch_jobs record with current status inline
                try:
                    pb_results = await pb_client.get_list(
                        "batch_jobs", filter_str=f'openai_batch_id = "{batch_id}"', per_page=1
                    )
                    if pb_results.get("items"):
                        record_id = pb_results["items"][0]["id"]
                        await pb_client.update("batch_jobs", record_id, {
                            "status": batch_status,
                            "total_items": request_counts.get("total", 0),
                            "processed_items": request_counts.get("completed", 0),
                            "failed_items": request_counts.get("failed", 0),
                        })
                except Exception as e:
                    logger.warning(f"Could not update batch_jobs inline: {e}")

                if batch_status == "completed":
                    output_file_id = status.get("output_file_id")
                    error_file_id = status.get("error_file_id")
                    total = request_counts.get("total", 0)
                    failed = request_counts.get("failed", 0)

                    if total > 0 and failed == total:
                        # ALL requests failed
                        error_details = await self._download_error_details(error_file_id)
                        error_msg = f"All {total} requests failed. {error_details}"
                        logger.error(f"Batch {batch_id}: {error_msg}")
                        if job:
                            job.status = JobStatus.FAILED
                            job.error_message = error_msg
                            job.completed_at = datetime.utcnow()
                        await self._update_batch_job_record(
                            year, batch_id, "failed",
                            error_file_id=error_file_id,
                            error_message=error_msg,
                            failed=failed,
                        )
                    elif output_file_id:
                        # Normal path: some or all succeeded
                        await self._process_batch_results(
                            job_id, batch_id, output_file_id, year
                        )
                        if error_file_id and failed > 0:
                            error_details = await self._download_error_details(error_file_id)
                            logger.warning(f"Batch {batch_id}: {failed}/{total} failed. {error_details}")
                    else:
                        # Edge case: completed, not all failed, but no output
                        logger.warning(f"Batch {batch_id} completed with no output file")
                        if job:
                            job.status = JobStatus.COMPLETED
                            job.completed_at = datetime.utcnow()
                        await self._update_batch_job_record(
                            year, batch_id, "completed",
                            error_message="No output file produced",
                        )
                    break

                elif batch_status in ("failed", "expired", "cancelled"):
                    # Extract error details from status
                    errors = status.get("errors", [])
                    error_msg = (
                        "; ".join(e["message"] for e in errors if e.get("message"))
                        if errors
                        else f"Batch {batch_status}"
                    )
                    logger.error(f"Batch {batch_id} failed: {error_msg}")

                    # Mark job as failed
                    if job:
                        job.status = JobStatus.FAILED
                        job.error_message = error_msg
                        job.completed_at = datetime.utcnow()

                    # Update PocketBase batch_jobs record
                    await self._update_batch_job_record(
                        year, batch_id, batch_status, status.get("error_file_id"),
                        error_message=error_msg,
                    )
                    break

                # Wait before next poll
                await asyncio.sleep(self.polling_interval)

        except asyncio.CancelledError:
            logger.info(f"Batch polling cancelled for {batch_id}")
            raise
        except Exception as e:
            logger.error(f"Error polling batch {batch_id}: {e}")
            job = await job_queue.get_status(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
        finally:
            self._polling_tasks.pop(job_id, None)
            if on_complete:
                try:
                    result = on_complete()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.warning(f"on_complete callback error: {e}")

    async def _download_error_details(self, error_file_id: str | None) -> str:
        """Download and summarize error details from an OpenAI error file."""
        if not error_file_id:
            return "No error file available."
        try:
            from app.services.openai_batch import openai_batch_client
            content = await openai_batch_client.client.files.content(error_file_id)
            lines = content.text.strip().split("\n")
            error_counts: dict[str, int] = {}
            for line in lines[:100]:
                try:
                    data = json.loads(line)
                    error = data.get("response", {}).get("body", {}).get("error", {})
                    key = f"{error.get('code', 'unknown')}: {error.get('message', 'unknown')[:120]}"
                    error_counts[key] = error_counts.get(key, 0) + 1
                except json.JSONDecodeError:
                    continue
            if error_counts:
                return "; ".join(f"{count}x {msg}" for msg, count in error_counts.items())
            return f"Error file has {len(lines)} entries"
        except Exception as e:
            return f"Could not download error file: {e}"

    async def _process_batch_results(
        self,
        job_id: str,
        batch_id: str,
        output_file_id: str,
        year: str,
    ) -> None:
        """Process completed batch results via the standalone helper."""
        job = await job_queue.get_status(job_id)
        model_used = job.metadata.get("model_used", "unknown") if job else "unknown"

        stored_count, failed_count = await process_completed_batch(
            output_file_id, year, batch_id, model_used
        )

        # Update in-memory job status
        if job:
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.processed_items = stored_count
            job.completed_at = datetime.utcnow()
            job.result = {
                "stored": stored_count,
                "failed": failed_count,
            }

        # Update PocketBase batch_jobs record
        await self._update_batch_job_record(
            year, batch_id, "completed", None, output_file_id, stored_count, failed_count
        )

    async def _update_batch_job_record(
        self,
        year: str,
        batch_id: str,
        status: str,
        error_file_id: str | None = None,
        output_file_id: str | None = None,
        processed: int = 0,
        failed: int = 0,
        error_message: str | None = None,
    ) -> None:
        """Update the batch_jobs record in PocketBase."""
        from app.services.pocketbase_client import pb_client

        try:
            # Find the batch job record
            results = await pb_client.get_list(
                "batch_jobs",
                filter_str=f'openai_batch_id = "{batch_id}"',
                per_page=1,
            )

            if results.get("items"):
                record_id = results["items"][0]["id"]
                update_data = {
                    "status": status,
                    "processed_items": processed,
                    "failed_items": failed,
                    "completed_at": datetime.utcnow().isoformat(),
                }
                if error_file_id:
                    update_data["error_file_id"] = error_file_id
                if output_file_id:
                    update_data["output_file_id"] = output_file_id
                if error_message:
                    update_data["error_message"] = error_message

                await pb_client.update("batch_jobs", record_id, update_data)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                f"Error updating batch_jobs record: {e}"
            )

    def stop_polling(self, job_id: str) -> bool:
        """Stop polling for a specific job."""
        if job_id in self._polling_tasks:
            self._polling_tasks[job_id].cancel()
            return True
        return False

    async def shutdown(self) -> None:
        """Cancel all polling tasks."""
        self._active = False
        for task in self._polling_tasks.values():
            task.cancel()
        if self._polling_tasks:
            await asyncio.gather(
                *self._polling_tasks.values(), return_exceptions=True
            )


class BatchOrchestrator:
    """Orchestrates sequential batch submission with backpressure.

    Submits sub-batches one at a time, waiting for each to complete
    before submitting the next. All sub-batches are created in PocketBase
    immediately with status 'queued' for visibility.
    """

    def __init__(self, polling_manager: BatchPollingManager):
        self.polling_manager = polling_manager
        self._tasks: dict[str, asyncio.Task] = {}

    async def start(
        self,
        year: str,
        batch_files: list[tuple[str, int, int]],
        batch_group_id: str,
        model_used: str,
    ) -> list[dict[str, Any]]:
        """
        Create PocketBase records for all sub-batches and start orchestration.

        Args:
            year: Survey year
            batch_files: List of (file_id, num_responses, estimated_tokens)
            batch_group_id: Group ID linking sub-batches
            model_used: Model used for tagging

        Returns:
            List of sub-batch info dicts (pb_record_id, file_id, etc.)
        """
        import logging
        from app.services.pocketbase_client import pb_client

        logger = logging.getLogger(__name__)

        sub_batches = []
        for idx, (file_id, num_responses, est_tokens) in enumerate(batch_files):
            # Create PB record with 'queued' status
            job_data = {
                "job_type": "tagging_batch",
                "year": year,
                "status": "queued",
                "total_items": num_responses,
                "processed_items": 0,
                "failed_items": 0,
                "input_file_id": file_id,
                "model_used": model_used,
                "batch_group_id": batch_group_id,
                "estimated_tokens": est_tokens,
            }
            job_record = await pb_client.create("batch_jobs", job_data)

            sub_batches.append({
                "pb_record_id": job_record["id"],
                "file_id": file_id,
                "num_responses": num_responses,
                "estimated_tokens": est_tokens,
                "index": idx,
            })

        logger.info(
            f"Created {len(sub_batches)} queued batch records for group {batch_group_id}"
        )

        # Start background orchestration
        task = asyncio.create_task(
            self._orchestrate(year, sub_batches, batch_group_id, model_used)
        )
        self._tasks[batch_group_id] = task

        return sub_batches

    async def _orchestrate(
        self,
        year: str,
        sub_batches: list[dict[str, Any]],
        batch_group_id: str,
        model_used: str,
    ) -> None:
        """Submit batches sequentially, waiting for each to complete."""
        import logging
        from app.services.openai_batch import openai_batch_client
        from app.services.pocketbase_client import pb_client

        logger = logging.getLogger(__name__)

        for sub in sub_batches:
            file_id = sub["file_id"]
            pb_record_id = sub["pb_record_id"]
            idx = sub["index"]

            try:
                # Update status to 'submitting'
                await pb_client.update("batch_jobs", pb_record_id, {
                    "status": "submitting",
                    "started_at": datetime.utcnow().isoformat(),
                })

                # Submit to OpenAI
                batch_result = await openai_batch_client.submit_batch(file_id)
                openai_batch_id = batch_result["batch_id"]

                # Update PB record with OpenAI batch ID
                await pb_client.update("batch_jobs", pb_record_id, {
                    "status": batch_result["status"],
                    "openai_batch_id": openai_batch_id,
                })

                logger.info(
                    f"Sub-batch {idx + 1}/{len(sub_batches)} submitted: {openai_batch_id}"
                )

                # Create internal job for polling
                internal_job_id = await job_queue.submit(
                    JobType.TAGGING_BATCH,
                    year,
                    _noop_batch_job,
                )

                # Use an event to wait for this batch to complete
                done_event = asyncio.Event()

                def on_batch_complete():
                    done_event.set()

                # Start polling with completion callback
                await self.polling_manager.start_polling(
                    job_id=internal_job_id,
                    batch_id=openai_batch_id,
                    year=year,
                    metadata={
                        "model_used": model_used,
                        "batch_group_id": batch_group_id,
                    },
                    on_complete=on_batch_complete,
                )

                # Wait for this batch to finish before submitting next
                await done_event.wait()

                logger.info(f"Sub-batch {idx + 1}/{len(sub_batches)} completed")

            except Exception as e:
                logger.error(
                    f"Sub-batch {idx + 1}/{len(sub_batches)} failed to submit: {e}"
                )
                try:
                    await pb_client.update("batch_jobs", pb_record_id, {
                        "status": "failed",
                        "error_message": str(e)[:500],
                        "completed_at": datetime.utcnow().isoformat(),
                    })
                except Exception:
                    pass
                # Continue to next sub-batch

        self._tasks.pop(batch_group_id, None)
        logger.info(f"Batch group {batch_group_id} orchestration complete")


async def _noop_batch_job(progress_callback=None):
    """Placeholder job for batch tagging - actual work done by polling manager."""
    pass


# Singleton instances
batch_polling_manager = BatchPollingManager()
batch_orchestrator = BatchOrchestrator(batch_polling_manager)
