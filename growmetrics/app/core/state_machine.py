"""
Job State Machine for the Growmetrics AI Chemistry Video Request Service.

Defines:
  - JobStatus: the allowed states a video generation job can occupy.
  - FailureReason: typed error codes surfaced in API responses so callers
    always see an understandable failure state instead of a silent 500.
  - VALID_TRANSITIONS: enforces a strict directed acyclic graph of allowed
    state transitions to prevent illegal state mutations.

Design principle (from the challenge brief):
  «define understandable failure states instead of failing silently or half-way»
"""

from enum import Enum


class JobStatus(str, Enum):
    """Ordered lifecycle states of a video generation job."""

    PENDING = "PENDING"
    """Job accepted by the API, queued for background processing."""

    ANALYZING = "ANALYZING"
    """LLM agent is generating the structured scene specification (JSON)."""

    GENERATING_AUDIO = "GENERATING_AUDIO"
    """Edge-TTS is synthesizing the narration audio track."""

    GENERATING_CODE = "GENERATING_CODE"
    """LLM agent is generating the Manim Python animation code."""

    RENDERING = "RENDERING"
    """Manim subprocess is rendering the final MP4 video."""

    VALIDATING = "VALIDATING"
    """All Quality Gates (1–4) are being evaluated."""

    COMPLETED = "COMPLETED"
    """Video artifact is ready; client can download it."""

    FAILED = "FAILED"
    """All retries and fallbacks exhausted; job cannot be completed."""


class FailureReason(str, Enum):
    """Typed error codes exposed in the API response on failure.

    Each code maps to exactly one point-of-failure in the pipeline so
    operators can diagnose issues without reading logs.
    """

    # Gate 1 failures
    ERR_JSON_PARSE_FAILED = "ERR_JSON_PARSE_FAILED"
    """LLM returned output that cannot be parsed as valid JSON."""

    ERR_JSON_SCHEMA_INVALID = "ERR_JSON_SCHEMA_INVALID"
    """JSON parsed but required fields (topic, narration_script, …) are absent."""

    # Gate 2 failures
    ERR_SCRIPT_TOO_SHORT = "ERR_SCRIPT_TOO_SHORT"
    """Narration script has fewer than the required minimum word count."""

    ERR_AUDIO_TOO_SHORT = "ERR_AUDIO_TOO_SHORT"
    """Edge-TTS audio duration is below the minimum threshold (25 s)."""

    ERR_AUDIO_TOO_LONG = "ERR_AUDIO_TOO_LONG"
    """Edge-TTS audio duration exceeds the maximum threshold (90 s)."""

    ERR_TTS_FAILED = "ERR_TTS_FAILED"
    """Edge-TTS could not generate audio (network / engine error)."""

    # Gate 3 failures
    ERR_CODE_SYNTAX = "ERR_CODE_SYNTAX"
    """LLM-generated Manim code has a Python syntax error."""

    ERR_CODE_UNSAFE = "ERR_CODE_UNSAFE"
    """AST linter detected forbidden constructs (os.system, eval, …)."""

    ERR_RENDER_FAILED = "ERR_RENDER_FAILED"
    """Manim subprocess exited with a non-zero return code."""

    ERR_RENDER_TIMEOUT = "ERR_RENDER_TIMEOUT"
    """Manim subprocess exceeded the configured wall-clock time limit."""

    # Gate 4 failures
    ERR_ARTIFACT_MISSING = "ERR_ARTIFACT_MISSING"
    """Expected MP4 output file was not created by Manim."""

    ERR_ARTIFACT_EMPTY = "ERR_ARTIFACT_EMPTY"
    """Output MP4 file exists but has zero bytes."""

    ERR_AV_SYNC = "ERR_AV_SYNC"
    """Audio and video track durations differ by more than the tolerance."""

    # General
    ERR_RETRIES_EXHAUSTED = "ERR_RETRIES_EXHAUSTED"
    """All self-correction retries failed; fallback template also failed."""

    ERR_UNKNOWN = "ERR_UNKNOWN"
    """An unexpected error occurred (see job logs for details)."""


# ---------------------------------------------------------------------------
# Allowed state transitions (strict DAG — no silent jumps allowed)
# ---------------------------------------------------------------------------
VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.ANALYZING, JobStatus.FAILED},
    JobStatus.ANALYZING: {JobStatus.GENERATING_AUDIO, JobStatus.FAILED},
    JobStatus.GENERATING_AUDIO: {JobStatus.GENERATING_CODE, JobStatus.FAILED},
    JobStatus.GENERATING_CODE: {JobStatus.RENDERING, JobStatus.ANALYZING, JobStatus.FAILED},
    JobStatus.RENDERING: {JobStatus.VALIDATING, JobStatus.GENERATING_CODE, JobStatus.FAILED},
    JobStatus.VALIDATING: {JobStatus.COMPLETED, JobStatus.RENDERING, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
}


def assert_valid_transition(current: JobStatus, next_state: JobStatus) -> None:
    """Raise ValueError if *next_state* is not reachable from *current*."""
    allowed = VALID_TRANSITIONS.get(current, set())
    if next_state not in allowed:
        raise ValueError(
            f"Illegal job state transition: {current} → {next_state}. "
            f"Allowed: {', '.join(s.value for s in allowed) or 'none'}"
        )
