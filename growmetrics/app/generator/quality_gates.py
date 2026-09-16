"""
4-Stage Quality Gate pipeline for the Growmetrics video request service.

Each gate raises a typed GateError whose .failure_reason maps directly to a
FailureReason enum value so the API always returns an understandable failure
state instead of a bare exception traceback.

Stages
------
Gate 1 – JSON Schema & Structure Validation
Gate 2 – Script Word Count & Edge-TTS Audio Duration
Gate 3 – AST Security Linting & Manim Dry-Run
Gate 4 – FFprobe Artifact Integrity Inspection
"""

import ast
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.state_machine import FailureReason


# ---------------------------------------------------------------------------
# Typed result & exception
# ---------------------------------------------------------------------------

@dataclass
class GateError(Exception):
    """Raised by any quality gate on failure.

    Attributes:
        failure_reason: FailureReason enum value for API surfacing.
        detail: Human-readable explanation of the failure.
    """
    failure_reason: FailureReason
    detail: str

    def __str__(self) -> str:
        return f"[{self.failure_reason.value}] {self.detail}"


# ---------------------------------------------------------------------------
# Gate 1: JSON Schema & Structure Validation
# ---------------------------------------------------------------------------

REQUIRED_JSON_FIELDS = {"topic", "title", "narration_script", "animation_code"}


def gate1_json_schema(raw_llm_output: str) -> dict:
    """Parse *raw_llm_output* as JSON and verify required fields.

    Returns:
        The parsed dict on success.

    Raises:
        GateError(ERR_JSON_PARSE_FAILED): If the string is not valid JSON.
        GateError(ERR_JSON_SCHEMA_INVALID): If required fields are missing.
    """
    # Strip markdown code fences if the LLM wrapped the JSON in ```json ... ```
    cleaned = raw_llm_output.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise GateError(
            failure_reason=FailureReason.ERR_JSON_PARSE_FAILED,
            detail=f"LLM output is not valid JSON: {exc}",
        ) from exc

    missing = REQUIRED_JSON_FIELDS - set(data.keys())
    if missing:
        raise GateError(
            failure_reason=FailureReason.ERR_JSON_SCHEMA_INVALID,
            detail=f"LLM JSON is missing required fields: {sorted(missing)}",
        )

    return data


# ---------------------------------------------------------------------------
# Gate 2: Script Word Count & Audio Duration
# ---------------------------------------------------------------------------

def gate2_script_length(narration_script: str, audio_duration: float) -> None:
    """Validate that the narration script and the synthesised audio are long enough.

    Args:
        narration_script: Raw text of the narration.
        audio_duration: Duration of the generated audio in seconds (from FFprobe).

    Raises:
        GateError(ERR_SCRIPT_TOO_SHORT): Word count below threshold.
        GateError(ERR_AUDIO_TOO_SHORT): Audio is shorter than min_audio_seconds.
        GateError(ERR_AUDIO_TOO_LONG): Audio is longer than max_audio_seconds.
    """
    word_count = len(narration_script.split())
    if word_count < settings.min_script_words:
        raise GateError(
            failure_reason=FailureReason.ERR_SCRIPT_TOO_SHORT,
            detail=(
                f"Narration script has {word_count} words; "
                f"minimum required: {settings.min_script_words}."
            ),
        )

    if audio_duration < settings.min_audio_seconds:
        raise GateError(
            failure_reason=FailureReason.ERR_AUDIO_TOO_SHORT,
            detail=(
                f"Audio duration is {audio_duration:.1f}s; "
                f"minimum required: {settings.min_audio_seconds}s."
            ),
        )

    if audio_duration > settings.max_audio_seconds:
        raise GateError(
            failure_reason=FailureReason.ERR_AUDIO_TOO_LONG,
            detail=(
                f"Audio duration is {audio_duration:.1f}s; "
                f"maximum allowed: {settings.max_audio_seconds}s."
            ),
        )


# ---------------------------------------------------------------------------
# Gate 3: AST Security & Manim Dry-Run
# ---------------------------------------------------------------------------

# Constructs that are explicitly forbidden in LLM-generated animation code
_FORBIDDEN_NAMES = {
    "eval", "exec", "__import__", "compile",
    "open", "input",
}
_FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil",
    "importlib", "ctypes", "pickle",
}


def gate3_ast_safety(animation_code: str) -> None:
    """Parse *animation_code* with the Python AST and reject unsafe constructs.

    Raises:
        GateError(ERR_CODE_SYNTAX): Python syntax error.
        GateError(ERR_CODE_UNSAFE): Forbidden builtins or module imports.
    """
    try:
        tree = ast.parse(animation_code)
    except SyntaxError as exc:
        raise GateError(
            failure_reason=FailureReason.ERR_CODE_SYNTAX,
            detail=f"Syntax error in LLM-generated animation code: {exc}",
        ) from exc

    for node in ast.walk(tree):
        # Forbidden function calls / names
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise GateError(
                failure_reason=FailureReason.ERR_CODE_UNSAFE,
                detail=f"Forbidden built-in detected: '{node.id}'",
            )
        # Forbidden imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_MODULES:
                    raise GateError(
                        failure_reason=FailureReason.ERR_CODE_UNSAFE,
                        detail=f"Forbidden import detected: 'import {alias.name}'",
                    )
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _FORBIDDEN_MODULES:
                raise GateError(
                    failure_reason=FailureReason.ERR_CODE_UNSAFE,
                    detail=f"Forbidden import detected: 'from {node.module} import ...'",
                )


def gate3_manim_dryrun(scene_file: Path, scene_class: str) -> str:
    """Execute Manim in a subprocess and collect stdout/stderr.

    Args:
        scene_file: Path to the Python file containing the Manim scene class.
        scene_class: Name of the Manim Scene subclass to render.

    Returns:
        Captured stdout + stderr (useful for self-correction prompts).

    Raises:
        GateError(ERR_RENDER_TIMEOUT): Subprocess exceeded the time limit.
        GateError(ERR_RENDER_FAILED): Subprocess exited with non-zero code.
    """
    cmd = [
        "manim",
        "-q" + settings.manim_quality,
        "--disable_caching",
        str(scene_file),
        scene_class,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.manim_timeout_seconds,
            cwd=str(scene_file.parent),
        )
    except subprocess.TimeoutExpired:
        raise GateError(
            failure_reason=FailureReason.ERR_RENDER_TIMEOUT,
            detail=f"Manim subprocess timed out after {settings.manim_timeout_seconds}s.",
        )

    combined_output = (result.stdout or "") + "\n" + (result.stderr or "")

    if result.returncode != 0:
        raise GateError(
            failure_reason=FailureReason.ERR_RENDER_FAILED,
            detail=f"Manim exited with code {result.returncode}.\n{combined_output.strip()}",
        )

    return combined_output


# ---------------------------------------------------------------------------
# Gate 4: FFprobe Artifact Integrity
# ---------------------------------------------------------------------------

def gate4_artifact_integrity(video_path: Path, expected_audio_duration: float) -> None:
    """Use FFprobe to verify the rendered MP4 is complete and audio-video synced.

    Args:
        video_path: Path to the rendered MP4 file.
        expected_audio_duration: Duration of the Edge-TTS audio in seconds.

    Raises:
        GateError(ERR_ARTIFACT_MISSING): File does not exist.
        GateError(ERR_ARTIFACT_EMPTY): File exists but has zero bytes.
        GateError(ERR_RENDER_FAILED): FFprobe cannot parse the file.
        GateError(ERR_AV_SYNC): Audio and video track durations differ too much.
    """
    if not video_path.exists():
        raise GateError(
            failure_reason=FailureReason.ERR_ARTIFACT_MISSING,
            detail=f"Expected output file not found: {video_path}",
        )

    if video_path.stat().st_size == 0:
        raise GateError(
            failure_reason=FailureReason.ERR_ARTIFACT_EMPTY,
            detail=f"Output file is empty (0 bytes): {video_path}",
        )

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        raise GateError(
            failure_reason=FailureReason.ERR_RENDER_FAILED,
            detail="FFprobe not found. Install FFmpeg and ensure it is on PATH.",
        )

    if result.returncode != 0:
        raise GateError(
            failure_reason=FailureReason.ERR_RENDER_FAILED,
            detail=f"FFprobe could not read video file: {result.stderr.strip()}",
        )

    try:
        info = json.loads(result.stdout)
        streams = info.get("streams", [])
        durations = [float(s["duration"]) for s in streams if "duration" in s]
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise GateError(
            failure_reason=FailureReason.ERR_RENDER_FAILED,
            detail=f"Could not parse FFprobe JSON output: {exc}",
        ) from exc

    if not durations:
        raise GateError(
            failure_reason=FailureReason.ERR_ARTIFACT_EMPTY,
            detail="FFprobe found no streams with duration in the video file.",
        )

    video_duration = max(durations)
    av_diff = abs(video_duration - expected_audio_duration)
    if av_diff > settings.av_sync_tolerance:
        raise GateError(
            failure_reason=FailureReason.ERR_AV_SYNC,
            detail=(
                f"Audio-video duration mismatch: video={video_duration:.2f}s, "
                f"audio={expected_audio_duration:.2f}s, "
                f"diff={av_diff:.2f}s (tolerance={settings.av_sync_tolerance}s)."
            ),
        )
