"""
Pydantic request/response schemas for the Growmetrics video request API.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from app.core.state_machine import JobStatus, FailureReason


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class VideoRequest(BaseModel):
    """Body for POST /api/v1/videos/request."""
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
        examples=["How does the pH scale work?"],
        description="A chemistry concept question to generate a video for.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class StageEventResponse(BaseModel):
    status: str
    timestamp: datetime
    detail: Optional[str] = None


class VideoJobResponse(BaseModel):
    """Returned for individual job queries and in the job list."""
    job_id: str
    query: str
    status: JobStatus
    failure_reason: Optional[FailureReason] = None
    failure_detail: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    video_duration_seconds: Optional[float] = None
    retry_count: int = 0
    used_fallback: bool = False
    estimated_cost_usd: Optional[float] = None
    stage_history: list[StageEventResponse] = []


class VideoRequestResponse(BaseModel):
    """Returned immediately after POST /request."""
    job_id: str
    status: JobStatus
    message: str = "Video generation job queued successfully."


class JobListResponse(BaseModel):
    total: int
    jobs: list[VideoJobResponse]
