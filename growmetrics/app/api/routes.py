"""
FastAPI route definitions for the Growmetrics AI Chemistry Video Request Service.

Endpoints:
  POST /api/v1/videos/request   — Submit a new video generation job
  GET  /api/v1/videos           — List all jobs
  GET  /api/v1/videos/{job_id}  — Get detailed job status
  GET  /api/v1/videos/{job_id}/artifact — Stream/download the generated MP4
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.api.schemas import (
    VideoRequest,
    VideoRequestResponse,
    VideoJobResponse,
    JobListResponse,
    StageEventResponse,
)
from app.core.state_machine import JobStatus
from app.repository.job_store import job_repository
from app.repository.artifact_store import artifact_store
from app.generator.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/videos", tags=["Videos"])


# ---------------------------------------------------------------------------
# Helper: convert domain model → response schema
# ---------------------------------------------------------------------------

def _to_response(job) -> VideoJobResponse:
    return VideoJobResponse(
        job_id=job.job_id,
        query=job.query,
        status=job.status,
        failure_reason=job.failure_reason,
        failure_detail=job.failure_detail,
        created_at=job.created_at,
        updated_at=job.updated_at,
        video_duration_seconds=job.video_duration_seconds,
        retry_count=job.retry_count,
        used_fallback=job.used_fallback,
        estimated_cost_usd=job.estimated_cost_usd,
        stage_history=[
            StageEventResponse(
                status=e.status.value if hasattr(e.status, "value") else str(e.status),
                timestamp=e.timestamp,
                detail=e.detail,
            )
            for e in job.stage_history
        ],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/request", response_model=VideoRequestResponse, status_code=202)
async def request_video(body: VideoRequest, background_tasks: BackgroundTasks):
    """Submit a chemistry concept query and start an async video generation job.

    The job is processed in the background; poll GET /videos/{job_id} for status.
    """
    job = await job_repository.create(body.query)
    background_tasks.add_task(run_pipeline, job)
    logger.info("Job %s created for query: %r", job.job_id, body.query)
    return VideoRequestResponse(job_id=job.job_id, status=job.status)


@router.get("", response_model=JobListResponse)
async def list_videos():
    """Return all video generation jobs ordered by creation time (newest first)."""
    jobs = await job_repository.list_all()
    return JobListResponse(total=len(jobs), jobs=[_to_response(j) for j in jobs])


@router.get("/{job_id}", response_model=VideoJobResponse)
async def get_video_status(job_id: str):
    """Return the current status and full stage history of a video generation job."""
    job = await job_repository.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _to_response(job)


@router.get("/{job_id}/artifact")
async def download_video(job_id: str):
    """Stream the generated MP4 video artifact to the client.

    Returns 404 if the job does not exist or has not completed yet.
    """
    job = await job_repository.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not yet completed. Current status: {job.status.value}.",
        )
    video_path = Path(job.artifact_path) if job.artifact_path else artifact_store.video_path(job_id)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video artifact file not found on server.")
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"{job_id}_chemistry_video.mp4",
    )
