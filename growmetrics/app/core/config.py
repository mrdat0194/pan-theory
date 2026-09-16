"""
Configuration module for the Growmetrics AI Chemistry Video Request Service.

Loads settings from environment variables with sensible defaults.
All paths are resolved relative to this file so the project is portable.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Directory Layout
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]          # growmetrics/
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = ROOT_DIR / "artifacts" / "tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application-wide settings.

    Values can be overridden via environment variables or a .env file.
    """

    # ---- LLM Provider (choose one) ----------------------------------------
    # Set OPENAI_API_KEY or GOOGLE_API_KEY in your environment.
    openai_api_key: str = ""
    google_api_key: str = ""

    # Default model selection: Gemini Flash is cost-optimal; fall back to OpenAI.
    llm_model: str = "gemini-1.5-flash"

    # ---- Generation Reliability Controls -----------------------------------
    # Maximum number of LLM self-correction attempts before falling back
    # to the deterministic parametric template.
    max_retries: int = 3

    # ---- Quality Gate Thresholds -------------------------------------------
    # Gate 2: minimum word count of narration script.
    min_script_words: int = 70
    # Gate 2: minimum/maximum Edge-TTS audio duration in seconds.
    min_audio_seconds: float = 25.0
    max_audio_seconds: float = 90.0
    # Gate 4: maximum allowed difference between video and audio track (seconds).
    av_sync_tolerance: float = 0.5

    # ---- Manim Render Settings ---------------------------------------------
    manim_quality: str = "l"           # l = low (720p fast), h = high (1080p)
    manim_timeout_seconds: int = 120   # max subprocess wall-clock time

    # ---- Edge-TTS Settings -------------------------------------------------
    tts_voice: str = "en-US-JennyNeural"

    # ---- Paths (computed from ROOT_DIR) ------------------------------------
    artifacts_dir: Path = ARTIFACTS_DIR
    temp_dir: Path = TEMP_DIR

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
