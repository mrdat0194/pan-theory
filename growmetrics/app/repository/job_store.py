"""
In-memory, thread-safe job store for the Growmetrics video request service.

Implements a clean repository interface so the persistence layer can be
swapped out for Redis / PostgreSQL without touching application logic.
"""

import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field

from app.core.state_machine import JobStatus, FailureReason, assert_valid_transition


# ---------------------------------------------------------------------------
# Domain Model
# ---------------------------------------------------------------------------

class StageEvent(BaseModel):
    """A single timestamped stage transition recorded for observability."""
    status: JobStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail: Optional[str] = None


class VideoJob(BaseModel):
    """Full representation of a video generation job."""

    job_id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:10]}")
    query: str
    status: JobStatus = JobStatus.PENDING
    failure_reason: Optional[FailureReason] = None
    failure_detail: Optional[str] = None

    # Tracking
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage_history: list[StageEvent] = Field(default_factory=list)

    # Output metadata (populated on COMPLETED)
    artifact_path: Optional[str] = None   # absolute filesystem path to the MP4
    video_duration_seconds: Optional[float] = None
    retry_count: int = 0
    used_fallback: bool = False
    estimated_cost_usd: Optional[float] = None




# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class JobRepository:
    """Thread-safe, async-compatible in-memory store for VideoJob objects.

    Design: backed by a plain dict protected by an asyncio.Lock.
    Swap in a DB-backed implementation by honouring the same method signatures.
    """

    def __init__(self) -> None:
        self._store: dict[str, VideoJob] = {}
        self._lock = asyncio.Lock()

    async def create(self, query: str) -> VideoJob:
        """Create and persist a new job in PENDING state."""
        job = VideoJob(query=query)
        job.stage_history.append(StageEvent(status=JobStatus.PENDING))
        async with self._lock:
            self._store[job.job_id] = job
        return job

    async def get(self, job_id: str) -> Optional[VideoJob]:
        """Return the job with *job_id*, or None if not found."""
        async with self._lock:
            return self._store.get(job_id)

    async def list_all(self) -> list[VideoJob]:
        """Return all jobs ordered by creation time (newest first)."""
        async with self._lock:
            return sorted(self._store.values(), key=lambda j: j.created_at, reverse=True)

    async def transition(
        self,
        job_id: str,
        new_status: JobStatus,
        detail: Optional[str] = None,
        failure_reason: Optional[FailureReason] = None,
        **kwargs,
    ) -> VideoJob:
        """Advance a job to *new_status*, validating the transition.

        Extra keyword args are merged into the job model (e.g. artifact_path,
        retry_count, used_fallback, …).

        Raises ValueError on illegal transitions so the caller can surface a
        clear error instead of silently corrupting job state.
        """
        async with self._lock:
            job = self._store.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found.")

            # Validate transition (raises ValueError on illegal jump)
            assert_valid_transition(job.status, new_status)

            job.status = new_status
            job.updated_at = datetime.now(timezone.utc)
            job.stage_history.append(StageEvent(status=new_status, detail=detail))

            if failure_reason is not None:
                job.failure_reason = failure_reason
                job.failure_detail = detail

            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)

            return job


# ---------------------------------------------------------------------------
# Singleton — imported by all modules that need the store
# ---------------------------------------------------------------------------
job_repository: JobRepository = JobRepository()
