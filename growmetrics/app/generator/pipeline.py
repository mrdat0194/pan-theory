"""
Main video generation pipeline orchestrator.

This module wires together the LLM agent, TTS engine, Manim renderer,
all four Quality Gates, the self-correction retry loop, and the parametric
fallback — implementing the full reliability framework from the spec.

Pipeline flow per job:
  1.  ANALYZING   → call LLM agent → Gate 1 (JSON schema)
  2.  GENERATING_AUDIO → Edge-TTS → Gate 2 (script length & audio duration)
  3.  GENERATING_CODE  → AST safety check (Gate 3a)
  4.  RENDERING   → Manim dry-run render (Gate 3b)
  5.  VALIDATING  → FFprobe artifact check (Gate 4)
  6.  COMPLETED   (or FAILED with typed error code)

Self-correction loop: on Gate 3 failure, feed traceback back to LLM and
retry GENERATING_CODE → RENDERING up to settings.max_retries times.

Fallback: if retries exhausted, resolve parametric template and re-run
from GENERATING_CODE (template is guaranteed to pass Gates 3 & 4).
"""

import asyncio
import logging
from pathlib import Path

from app.core.config import settings
from app.core.state_machine import JobStatus, FailureReason
from app.repository.job_store import job_repository, VideoJob
from app.repository.artifact_store import artifact_store
from app.generator import agent, manim_engine
from app.generator.tts_engine import generate_audio, TTSError
from app.generator.quality_gates import (
    GateError,
    gate1_json_schema,
    gate2_script_length,
    gate3_ast_safety,
    gate3_manim_dryrun,
    gate4_artifact_integrity,
)
from app.generator.templates.router import resolve_fallback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estimated cost helper
# ---------------------------------------------------------------------------

def _estimate_cost(token_calls: int, used_fallback: bool) -> float:
    """Rough cost estimate in USD.

    Gemini 1.5 Flash: ~$0.000075 / 1K input tokens, ~$0.0003 / 1K output tokens.
    We assume ~800 tokens/call as a rough average.
    """
    if used_fallback:
        return round(token_calls * 0.000075, 6)
    return round(token_calls * 0.0003, 6)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(job: VideoJob) -> None:
    """Execute the full video generation pipeline for *job*.

    Mutates the job's status in *job_repository* at each stage transition.
    All exceptions are caught internally — the job will always end in either
    COMPLETED or FAILED, never in an intermediate state.
    """
    job_id = job.job_id
    query = job.query
    token_calls = 0
    used_fallback = False

    try:
        # ── Stage 1: ANALYZING — LLM scene spec generation ──────────────────
        await job_repository.transition(job_id, JobStatus.ANALYZING)
        raw_llm = agent.generate_scene_spec(query)
        token_calls += 1
        scene_spec = gate1_json_schema(raw_llm)   # Gate 1: JSON parse & schema
        logger.info("[%s] Gate 1 passed (JSON schema OK)", job_id)

        narration = scene_spec["narration_script"]
        animation_code = scene_spec["animation_code"]

        # ── Stage 2: GENERATING_AUDIO — Edge-TTS synthesis ──────────────────
        await job_repository.transition(job_id, JobStatus.GENERATING_AUDIO)
        audio_path = artifact_store.audio_path(job_id)
        audio_result = await generate_audio(narration, audio_path)
        gate2_script_length(narration, audio_result.duration_seconds)  # Gate 2
        logger.info("[%s] Gate 2 passed (audio=%.1fs, words=%d)",
                    job_id, audio_result.duration_seconds, len(narration.split()))

        # ── Stage 3: GENERATING_CODE + RENDERING — self-correction loop ─────
        last_gate_error: GateError | None = None
        work_dir = artifact_store.manim_work_dir(job_id)

        for attempt in range(1, settings.max_retries + 2):  # +1 for initial attempt
            await job_repository.transition(
                job_id, JobStatus.GENERATING_CODE,
                detail=f"Code attempt {attempt}",
                retry_count=attempt - 1,
            )

            # Gate 3a: AST safety
            try:
                gate3_ast_safety(animation_code)
            except GateError as ge:
                last_gate_error = ge
                logger.warning("[%s] Gate 3a failed (attempt %d): %s", job_id, attempt, ge.detail)
                if attempt > settings.max_retries:
                    break
                raw_repair = agent.repair_animation_code(animation_code, ge.detail, attempt)
                token_calls += 1
                fixed = agent.extract_animation_code_from_repair(raw_repair)
                if fixed:
                    animation_code = fixed
                continue

            # Gate 3b: Manim dry-run render
            await job_repository.transition(job_id, JobStatus.RENDERING, detail=f"Render attempt {attempt}")
            scene_file = manim_engine.write_scene_file(animation_code, work_dir)
            try:
                traceback_output = gate3_manim_dryrun(scene_file, manim_engine.SCENE_CLASS_NAME)
                last_gate_error = None  # Render succeeded
                break
            except GateError as ge:
                last_gate_error = ge
                logger.warning("[%s] Gate 3b failed (attempt %d): %s", job_id, attempt, ge.detail[:200])
                if attempt > settings.max_retries:
                    break
                # Self-correction: send traceback to LLM
                raw_repair = agent.repair_animation_code(animation_code, ge.detail, attempt)
                token_calls += 1
                fixed = agent.extract_animation_code_from_repair(raw_repair)
                if fixed:
                    animation_code = fixed
                # Transition back to GENERATING_CODE for next iteration
                await job_repository.transition(job_id, JobStatus.GENERATING_CODE,
                                                 detail=f"Self-correction attempt {attempt}")

        # ── Fallback: parametric template ────────────────────────────────────
        if last_gate_error is not None:
            logger.warning("[%s] Retries exhausted — activating fallback template", job_id)
            fallback = resolve_fallback(query)
            if fallback is None:
                raise last_gate_error  # No template available → FAILED
            animation_code = fallback.animation_code
            narration = fallback.narration_script
            used_fallback = True

            # Re-synthesise audio for the fallback narration
            audio_result = await generate_audio(narration, audio_path)

            # Write & render fallback (should always succeed)
            await job_repository.transition(job_id, JobStatus.GENERATING_CODE,
                                             detail="Using deterministic fallback template")
            gate3_ast_safety(animation_code)
            await job_repository.transition(job_id, JobStatus.RENDERING,
                                             detail="Rendering fallback template")
            scene_file = manim_engine.write_scene_file(animation_code, work_dir)
            gate3_manim_dryrun(scene_file, manim_engine.SCENE_CLASS_NAME)

        # ── Full render with audio merge ─────────────────────────────────────
        final_video = artifact_store.video_path(job_id)
        manim_engine.render_scene(animation_code, audio_path, work_dir, final_video)

        # ── Stage 4: VALIDATING — artifact integrity check ──────────────────
        await job_repository.transition(job_id, JobStatus.VALIDATING)
        gate4_artifact_integrity(final_video, audio_result.duration_seconds)
        logger.info("[%s] Gate 4 passed (artifact OK)", job_id)

        # Cleanup temp manim files
        artifact_store.cleanup_temp(job_id)

        cost = _estimate_cost(token_calls, used_fallback)
        await job_repository.transition(
            job_id,
            JobStatus.COMPLETED,
            detail="All quality gates passed.",
            artifact_path=str(final_video),
            video_duration_seconds=audio_result.duration_seconds,
            used_fallback=used_fallback,
            estimated_cost_usd=cost,
        )
        logger.info("[%s] COMPLETED (cost≈$%.5f, fallback=%s)", job_id, cost, used_fallback)

    except GateError as ge:
        logger.error("[%s] FAILED via GateError: %s", job_id, ge)
        await _fail_job(job_id, ge.failure_reason, ge.detail)

    except TTSError as te:
        logger.error("[%s] FAILED via TTSError: %s", job_id, te)
        await _fail_job(job_id, FailureReason.ERR_TTS_FAILED, str(te))

    except Exception as exc:
        logger.exception("[%s] FAILED via unexpected error", job_id)
        await _fail_job(job_id, FailureReason.ERR_UNKNOWN, str(exc))


async def _fail_job(job_id: str, reason: FailureReason, detail: str) -> None:
    """Safely transition the job to FAILED state from any current state."""
    try:
        job = await job_repository.get(job_id)
        if job and job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
            # Force-set status directly to avoid transition graph conflicts
            job.status = JobStatus.FAILED
            job.failure_reason = reason
            job.failure_detail = detail
            from datetime import datetime, timezone
            job.updated_at = datetime.now(timezone.utc)
            from app.core.state_machine import StageEvent
            job.stage_history.append(
                StageEvent(status=JobStatus.FAILED, detail=detail)  # type: ignore[arg-type]
            )
    except Exception as e:
        logger.error("Could not mark job %s as FAILED: %s", job_id, e)
