"""
LLM Agent for the Growmetrics AI Chemistry Video Request Service.

Responsibilities:
  1. Generate a structured JSON scene specification for a chemistry concept.
  2. Self-correct LLM-generated Manim code when Quality Gate 3 fails,
     using the traceback as feedback (up to max_retries attempts).

Non-determinism strategy:
  - All LLM responses are requested as structured JSON (Gate 1 validates this).
  - Temperature is set to 0.2 for lower output variance.
  - Each retry includes the exact error message so the model can target the fix.
  - After max_retries, the caller falls back to a parametric template.
"""

import re
import json
import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """
You are an expert chemistry educator and Python/Manim animation developer.
Your task is to produce a JSON object that describes a short educational video
about a chemistry concept.

The JSON MUST have exactly these fields:
{
  "topic": "<slug identifying the topic, e.g. ph_scale>",
  "title": "<Short display title of the video>",
  "narration_script": "<Full plain-text narration that takes 30–80 seconds to read aloud. Must be at least 70 words.>",
  "animation_code": "<Complete, runnable Manim Community Edition Python code. Must subclass Scene and be named ChemistryScene.>"
}

Manim code rules:
- Import only: manim, math, numpy
- Class must be: class ChemistryScene(Scene):
- DO NOT use: os, sys, subprocess, eval, exec, open, input
- Produce clear, colourful, vector animations that match the narration
- Use MathTex for chemical formulae (e.g. MathTex(r"H_2O"))
- Keep the total animation under 90 seconds wall-clock time

Return ONLY the raw JSON object. No markdown fences, no explanation.
"""

REPAIR_SYSTEM_PROMPT = """
You are a Manim Python expert. The animation code you previously generated
contains an error. Repair ONLY the `animation_code` field.

Return a JSON object with a single field:
{ "animation_code": "<corrected, complete, runnable Manim code>" }

Keep all other aspects of the scene intact. Return ONLY the raw JSON object.
"""


# ---------------------------------------------------------------------------
# LLM Client (Gemini Flash primary, OpenAI fallback)
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_message: str) -> str:
    """Call the configured LLM and return the raw text response.

    Tries Google Gemini first (cost-optimal), falls back to OpenAI if
    google_api_key is not set.
    """
    model_name = settings.llm_model

    if settings.google_api_key and "gemini" in model_name.lower():
        return _call_gemini(system_prompt, user_message, model_name)
    elif settings.openai_api_key:
        return _call_openai(system_prompt, user_message)
    else:
        raise RuntimeError(
            "No LLM API key configured. Set GOOGLE_API_KEY or OPENAI_API_KEY in .env"
        )


def _call_gemini(system_prompt: str, user_message: str, model: str) -> str:
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=settings.google_api_key)
    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config={"temperature": 0.2},
    )
    response = client.generate_content(user_message)
    return response.text


def _call_openai(system_prompt: str, user_message: str) -> str:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_scene_spec(query: str) -> str:
    """Ask the LLM to generate a full scene specification JSON for *query*.

    Returns:
        Raw string (to be validated by Gate 1 before use).
    """
    user_message = f"Chemistry topic query: {query}"
    logger.info("LLM: generating scene spec for query=%r", query)
    raw = _call_llm(GENERATION_SYSTEM_PROMPT, user_message)
    logger.debug("LLM raw output (first 300 chars): %s", raw[:300])
    return raw


def repair_animation_code(
    original_code: str,
    error_traceback: str,
    attempt: int,
) -> str:
    """Ask the LLM to fix *original_code* given *error_traceback*.

    Args:
        original_code: The broken Manim Python code.
        error_traceback: stdout/stderr captured from the failed render.
        attempt: Current retry attempt number (1-indexed) for logging.

    Returns:
        Raw JSON string containing the corrected animation_code field.
        (Caller must extract and validate the field.)
    """
    user_message = (
        f"Attempt {attempt}/{settings.max_retries}.\n\n"
        f"BROKEN CODE:\n```python\n{original_code}\n```\n\n"
        f"ERROR TRACEBACK:\n{error_traceback.strip()}\n\n"
        "Fix the animation_code. Return only the JSON object."
    )
    logger.info("LLM: repair attempt %d for broken Manim code", attempt)
    raw = _call_llm(REPAIR_SYSTEM_PROMPT, user_message)
    logger.debug("LLM repair raw output (first 300 chars): %s", raw[:300])
    return raw


def extract_animation_code_from_repair(raw_repair: str) -> Optional[str]:
    """Parse the repair response and extract the animation_code string."""
    cleaned = raw_repair.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1])
    try:
        data = json.loads(cleaned)
        return data.get("animation_code")
    except (json.JSONDecodeError, AttributeError):
        # Best-effort: try to extract code block from partial output
        match = re.search(r'"animation_code"\s*:\s*"(.*)"', cleaned, re.DOTALL)
        if match:
            return match.group(1).encode().decode("unicode_escape")
        return None
