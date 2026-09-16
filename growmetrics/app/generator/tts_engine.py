"""
Edge-TTS narration audio generator for the Growmetrics video request service.

Design principles applied here:
  - «validate generated output before it reaches the learner»: audio duration
    is measured with FFprobe immediately after synthesis and returned to the
    caller so Quality Gate 2 can act on it.
  - «define understandable failure states»: every exception is caught and
    re-raised as a typed TTSError with a human-readable message.
"""

import asyncio
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass

import edge_tts  # type: ignore

from app.core.config import settings


# ---------------------------------------------------------------------------
# Typed result & exception
# ---------------------------------------------------------------------------

@dataclass
class AudioResult:
    """Return value from generate_audio()."""
    path: Path
    duration_seconds: float


class TTSError(RuntimeError):
    """Raised when Edge-TTS synthesis or duration inspection fails."""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

async def generate_audio(narration_script: str, output_path: Path) -> AudioResult:
    """Synthesise *narration_script* to a WAV file at *output_path*.

    Uses Edge-TTS (Microsoft Neural TTS) — zero API cost, no key required.

    Args:
        narration_script: Plain-text narration to synthesise.
        output_path: Desired destination for the generated audio file.

    Returns:
        AudioResult with the output path and measured duration.

    Raises:
        TTSError: If synthesis fails or FFprobe cannot inspect the file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        communicate = edge_tts.Communicate(
            text=narration_script,
            voice=settings.tts_voice,
        )
        await communicate.save(str(output_path))
    except Exception as exc:
        raise TTSError(f"Edge-TTS synthesis failed: {exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TTSError("Edge-TTS produced an empty or missing audio file.")

    duration = _probe_duration(output_path)
    return AudioResult(path=output_path, duration_seconds=duration)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe_duration(audio_path: Path) -> float:
    """Use FFprobe to get the duration of *audio_path* in seconds.

    Raises:
        TTSError: If FFprobe is unavailable or returns an unexpected result.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(audio_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        raise TTSError("FFprobe not found on PATH. Install FFmpeg and ensure it is on PATH.")
    except subprocess.TimeoutExpired:
        raise TTSError("FFprobe timed out while inspecting audio file.")

    if result.returncode != 0:
        raise TTSError(f"FFprobe error: {result.stderr.strip()}")

    try:
        info = json.loads(result.stdout)
        duration = float(info["streams"][0]["duration"])
        return duration
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        raise TTSError(f"Could not parse FFprobe output: {exc}") from exc
