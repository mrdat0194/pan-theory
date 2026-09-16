"""
Artifact store — manages the filesystem paths for generated MP4 and WAV files.

All paths live under the configured ARTIFACTS_DIR so the boundary between
generation logic and storage is explicit and swappable.
"""

import shutil
from pathlib import Path

from app.core.config import settings


class ArtifactStore:
    """Manages read/write access to video and audio artifact files.

    The structure under ARTIFACTS_DIR is:
        artifacts/
            {job_id}/
                audio.wav       — Edge-TTS narration
                video.mp4       — Final rendered video
    """

    def __init__(self, base_dir: Path = settings.artifacts_dir) -> None:
        self._base = base_dir
        self._base.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        """Return (and create) the per-job artifact directory."""
        d = self._base / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def audio_path(self, job_id: str) -> Path:
        """Return the expected path for the Edge-TTS WAV file."""
        return self.job_dir(job_id) / "audio.wav"

    def video_path(self, job_id: str) -> Path:
        """Return the expected path for the final rendered MP4."""
        return self.job_dir(job_id) / "video.mp4"

    def manim_work_dir(self, job_id: str) -> Path:
        """Return a temporary working directory for Manim scene source code."""
        d = self.job_dir(job_id) / "manim_work"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cleanup_temp(self, job_id: str) -> None:
        """Remove Manim intermediate files, keeping only the final MP4 and WAV."""
        work_dir = self.job_dir(job_id) / "manim_work"
        media_dir = self.job_dir(job_id) / "media"
        for d in (work_dir, media_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

    def exists(self, job_id: str) -> bool:
        """Return True if a completed video artifact exists for *job_id*."""
        return self.video_path(job_id).exists() and self.video_path(job_id).stat().st_size > 0


# Singleton instance
artifact_store: ArtifactStore = ArtifactStore()
