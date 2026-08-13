"""
omnistats/modules/simulation/matraix_loader.py
───────────────────────────────────────────────
Phase V: MatrAIx Persona Loader (HuggingFace → OmniStats DataFrame).

Responsibilities
----------------
1. Download and cache the MatrAIx Persona coreset from HuggingFace:
   Dataset: MatrAIx2026/MatrAIx_Persona_1M (or local parquet fallback).

2. Map the raw 1,290-dimensional schema into OmniStats-compatible columns:
   - Demographic columns  → config.DEMOGRAPHIC_COLS
   - Indicator columns    → config.INDICATOR_COLS
   - AB group / metric    → config.AB_GROUP_COL, config.AB_METRIC_COL

3. Return a clean pandas DataFrame ready for omnistats/main.py.

Notes
-----
The dataset uses a 1M-coreset of synthetic personas.
Each row is an AI-simulated agent with rich psychometric + demographic attributes.
References: https://github.com/MatrAIx-ai/MatrAIx-Persona-8B
"""
from __future__ import annotations

import os
import sys
import hashlib
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── resolve omnistats root ────────────────────────────────────────────────────
_OMNI_ROOT = Path(__file__).resolve().parents[3]  # omnistats/
sys.path.insert(0, str(_OMNI_ROOT))

try:
    from config import (
        OUTPUT_DIR,
        INDICATOR_COLS,
        DEMOGRAPHIC_COLS,
        AB_GROUP_COL,
        AB_METRIC_COL,
        AB_CONVERSION_COL,
    )
except ImportError as e:
    raise ImportError(
        "[matraix_loader] Could not import omnistats config. "
        "Run from within the pan-theory workspace."
    ) from e


# ── Constants ─────────────────────────────────────────────────────────────────

# HuggingFace dataset handle
_HF_DATASET_ID = "MatrAIx2026/MatrAIx_Persona_1M"

# Local cache location (avoid re-downloading)
_CACHE_DIR = Path(OUTPUT_DIR) / "matraix_cache"

# ── Schema Mapping ─────────────────────────────────────────────────────────────
# Map canonical MatrAIx field names → OmniStats column names.
# Extend this dict as you discover the full 1,290-dimensional schema.
_SCHEMA_MAP: dict[str, str] = {
    # Psychographic indicators
    "openness":          "openness",
    "conscientiousness": "conscientiousness",
    "extraversion":      "extraversion",
    "agreeableness":     "agreeableness",
    "neuroticism":       "neuroticism",
    "risk_tolerance":    "risk_tolerance",
    "tech_savviness":    "tech_savviness",
    "impulsivity":       "impulsivity",
    # Demographics
    "age":               "age",
    "income":            "income",
    "education_level":   "education_level",
    "location_tier":     "location_tier",   # 1=urban, 2=suburban, 3=rural
}

# Default OmniStats indicator columns we populate from MatrAIx
_DEFAULT_INDICATOR_COLS = [
    "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism", "risk_tolerance",
    "tech_savviness", "impulsivity",
]

# Default OmniStats demographic columns
_DEFAULT_DEMOGRAPHIC_COLS = ["age", "income", "education_level", "location_tier"]


# =============================================================================
# Public API
# =============================================================================

def load_matraix_personas(
    n_personas: int = 5_000,
    feature_variant: str = "control",
    seed: int = 42,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download (or load from cache) a sample of MatrAIx personas and format
    them as a clean OmniStats-compatible DataFrame.

    Parameters
    ----------
    n_personas    : int     Number of personas to sample (default 5,000).
    feature_variant: str   Label for the experimental variant being tested.
    seed          : int     Random seed for reproducible sampling.
    use_cache     : bool    If True, use a local Parquet cache when available.
    verbose       : bool    Print progress messages.

    Returns
    -------
    pd.DataFrame with OmniStats-compatible columns ready for data_manager.py.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{n_personas}_{seed}".encode()).hexdigest()[:8]
    cache_path = _CACHE_DIR / f"personas_{cache_key}.parquet"

    if use_cache and cache_path.exists():
        if verbose:
            print(f"[MatrAIxLoader] Loading cached personas from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    df = _try_hf_download(n_personas=n_personas, seed=seed, verbose=verbose)

    # Apply schema mapping
    df = _map_schema(df, verbose=verbose)

    # Assign AB group and metric columns (treatment/control will be set in bridge)
    if AB_GROUP_COL and AB_GROUP_COL not in df.columns:
        df[AB_GROUP_COL] = feature_variant
    if AB_METRIC_COL and AB_METRIC_COL not in df.columns:
        df[AB_METRIC_COL] = np.nan   # filled in by matraix_bridge.py

    # Save cache
    if use_cache:
        df.to_parquet(cache_path, index=False)
        if verbose:
            print(f"[MatrAIxLoader] Cached {len(df)} personas -> {cache_path}")

    return df


# =============================================================================
# Internal helpers
# =============================================================================

def _try_hf_download(
    n_personas: int,
    seed: int,
    verbose: bool,
) -> pd.DataFrame:
    """
    Attempt to stream from HuggingFace. Falls back to a synthetic mock dataset
    if the HF datasets library is not installed or no internet is available.
    """
    try:
        from datasets import load_dataset  # type: ignore

        if verbose:
            print(f"[MatrAIxLoader] Streaming {n_personas} personas from HuggingFace: {_HF_DATASET_ID}")

        ds = load_dataset(_HF_DATASET_ID, split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= n_personas:
                break
            records.append(row)

        df = pd.DataFrame(records)
        if verbose:
            print(f"[MatrAIxLoader] Downloaded {len(df)} personas. Columns: {list(df.columns)[:10]}...")
        return df

    except (ImportError, Exception) as e:
        warnings.warn(
            f"[MatrAIxLoader] HuggingFace unavailable ({e}). "
            "Generating synthetic mock persona dataset for quickstart.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _generate_mock_personas(n_personas=n_personas, seed=seed)


def _generate_mock_personas(n_personas: int, seed: int) -> pd.DataFrame:
    """
    Generate a synthetic population of MatrAIx-style personas.
    Used as quickstart fallback when the HF dataset is unavailable.

    Each persona is sampled from distributions that approximate real-world
    demographic and psychometric patterns.
    """
    rng = np.random.default_rng(seed)

    n = n_personas
    df = pd.DataFrame({
        # Big-5 personality (0–1 normalized)
        "openness":          rng.beta(2.5, 2.0, n),
        "conscientiousness": rng.beta(2.0, 2.5, n),
        "extraversion":      rng.beta(2.0, 2.0, n),
        "agreeableness":     rng.beta(3.0, 2.0, n),
        "neuroticism":       rng.beta(2.0, 3.0, n),
        # Behavioral traits
        "risk_tolerance":    rng.beta(2.0, 3.0, n),
        "tech_savviness":    rng.beta(2.5, 1.5, n),
        "impulsivity":       rng.beta(1.5, 3.0, n),
        # Demographics
        "age":               rng.integers(18, 75, n).astype(float),
        "income":            rng.lognormal(mean=10.5, sigma=0.8, size=n),   # ~USD annual
        "education_level":   rng.choice([1, 2, 3, 4], p=[0.15, 0.30, 0.35, 0.20], size=n).astype(float),
        "location_tier":     rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20], size=n).astype(float),
    })

    return df


def _map_schema(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply _SCHEMA_MAP to rename MatrAIx columns to OmniStats-compatible names.
    Drops columns not present in the schema map.
    Fills missing mapped columns with sensible defaults.
    """
    rename_map = {k: v for k, v in _SCHEMA_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Ensure all expected columns exist (fill missing ones)
    all_expected = _DEFAULT_INDICATOR_COLS + _DEFAULT_DEMOGRAPHIC_COLS
    for col in all_expected:
        if col not in df.columns:
            df[col] = np.nan
            if verbose:
                print(f"[MatrAIxLoader] Warning: Expected column '{col}' not found. Filled with NaN.")

    # Return only the expected columns to avoid surprises downstream
    available = [c for c in all_expected if c in df.columns]
    return df[available].copy()
