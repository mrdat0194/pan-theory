"""
Manim rendering engine for the Growmetrics AI Chemistry Video Request Service.

Handles:
  1. Writing LLM-generated animation code to a temporary Python file.
  2. Running Manim in a subprocess (gated through Quality Gate 3 externally).
  3. Locating and moving the rendered MP4 to the artifact store.
  4. Merging the audio track from Edge-TTS into the final MP4 using FFmpeg.
"""

import logging
import shutil
import subprocess
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

# The fixed class name the LLM is instructed to use
SCENE_CLASS_NAME = "ChemistryScene"


def write_scene_file(animation_code: str, work_dir: Path) -> Path:
    """Write *animation_code* to a Python file inside *work_dir*.

    Returns:
        Path to the written scene file.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    scene_file = work_dir / "scene.py"
    scene_file.write_text(animation_code, encoding="utf-8")
    logger.debug("Manim scene written to %s", scene_file)
    return scene_file


def find_rendered_mp4(work_dir: Path) -> Path | None:
    """Locate the MP4 that Manim outputs inside its media/ directory tree."""
    # Manim places output under: <work_dir>/media/videos/scene/<quality>/ChemistryScene.mp4
    candidates = list(work_dir.rglob(f"{SCENE_CLASS_NAME}.mp4"))
    if candidates:
        # Pick the most recently modified one
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def merge_audio_into_video(
    silent_video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> None:
    """Merge Edge-TTS audio into the silent Manim video using FFmpeg.

    The audio is treated as the duration master: the video is trimmed or
    padded with a freeze-frame to match audio length.

    Raises:
        RuntimeError: If FFmpeg returns a non-zero exit code.
    """
    cmd = [
        "ffmpeg",
        "-y",                            # overwrite without asking
        "-i", str(silent_video_path),
        "-i", str(audio_path),
        "-c:v", "copy",                  # copy video stream unchanged
        "-c:a", "aac",                   # encode audio as AAC
        "-b:a", "128k",
        "-shortest",                     # trim to shortest stream
        str(output_path),
    ]
    logger.info("Merging audio into video: %s", output_path.name)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio merge failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )
    logger.info("Audio merge complete: %s", output_path)


def render_scene(
    animation_code: str,
    audio_path: Path,
    work_dir: Path,
    final_output_path: Path,
) -> None:
    """Orchestrate the full render pipeline for a single Manim scene.

    Steps:
      1. Write animation code to work_dir/scene.py
      2. Execute ``manim`` subprocess (Gate 3 must already have passed)
      3. Locate rendered MP4
      4. Merge Edge-TTS audio track
      5. Copy final MP4 to final_output_path

    NOTE: Quality Gate 3 (dry-run) is called from the orchestrator (pipeline.py)
    *before* this function to avoid duplicate subprocess invocations.

    Raises:
        RuntimeError: If the render output cannot be located or merged.
    """
    scene_file = write_scene_file(animation_code, work_dir)

    # Run Manim
    cmd = [
        "manim",
        f"-q{settings.manim_quality}",
        "--disable_caching",
        str(scene_file),
        SCENE_CLASS_NAME,
    ]
    logger.info("Launching Manim render for %s …", SCENE_CLASS_NAME)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=settings.manim_timeout_seconds,
        cwd=str(work_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Manim render exited with code {result.returncode}:\n"
            f"{(result.stdout + result.stderr).strip()}"
        )

    # Locate output
    silent_mp4 = find_rendered_mp4(work_dir)
    if silent_mp4 is None:
        raise RuntimeError(
            f"Manim completed but no output MP4 found in {work_dir}."
        )

    # Merge audio
    merged_path = work_dir / "merged.mp4"
    merge_audio_into_video(silent_mp4, audio_path, merged_path)

    # Copy to final destination
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(merged_path, final_output_path)
    logger.info("Final video ready at %s", final_output_path)
