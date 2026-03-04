"""Job management endpoints."""

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.services.job_queue import job_queue

router = APIRouter()


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a job."""
    job = await job_queue.get_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    job = await job_queue.get_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    success = await job_queue.cancel(job_id)
    if success:
        return {"status": "cancelled", "job_id": job_id}
    else:
        return {"status": "already_completed", "job_id": job_id}


@router.get("/")
async def list_jobs(year: str | None = None):
    """List all jobs, optionally filtered by year."""
    jobs = job_queue.get_all_jobs(year=year)
    return {
        "total": len(jobs),
        "jobs": [j.to_dict() for j in jobs],
    }


@router.websocket("/ws/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    """WebSocket for real-time job updates."""
    await websocket.accept()

    # Check if job exists
    job = await job_queue.get_status(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    # Subscribe to updates
    queue = await job_queue.subscribe(job_id)

    try:
        while True:
            # Wait for update
            try:
                update = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(update)

                # Close if job is complete
                if update.get("status") in ["completed", "failed", "cancelled"]:
                    break
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        await job_queue.unsubscribe(job_id, queue)
