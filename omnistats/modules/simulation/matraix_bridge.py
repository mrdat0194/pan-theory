"""
omnistats/modules/simulation/matraix_bridge.py
───────────────────────────────────────────────
Phase V: MatrAIx Simulation Bridge.

Responsibilities
----------------
1. Take a loaded MatrAIx persona population (from matraix_loader.py).
2. Accept a feature/treatment dict from plan_experiment.py or the loop orchestrator.
3. Randomly assign treatment/control split across the persona population.
4. Apply a Mock Outcome Function (MOF) using persona covariates to simulate
   realistic outcome deltas (AB_METRIC_COL). The MOF is a weighted linear
   combination of persona traits calibrated to produce realistic effect sizes.
5. Write `simulated_experiment_data.csv` to OUTPUT_DIR for omnistats ingestion.

The Mock Outcome Function can be replaced with live LLM calls to MatrAIx-8B
when the full pipeline is operational. The interface is identical.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── resolve omnistats root ────────────────────────────────────────────────────
_OMNI_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_OMNI_ROOT))

from config import (
    OUTPUT_DIR,
    AB_GROUP_COL,
    AB_METRIC_COL,
    AB_CONVERSION_COL,
    INDICATOR_COLS,
    DEMOGRAPHIC_COLS,
)

# Output file consumed by omnistats main.py
SIM_DATA_PATH = os.path.join(OUTPUT_DIR, "simulated_experiment_data.csv")


# =============================================================================
# Mock Outcome Function
# =============================================================================

class MockOutcomeFunction:
    """
    Deterministic (but realistic) outcome simulator based on MatrAIx persona covariates.

    Treatment Effect Model
    ----------------------
    The true Average Treatment Effect (ATE) is controlled by `ate`. On top of
    the global ATE, individual treatment effects are modulated by persona traits:

        tau_i = ate + delta(persona_i)

    where delta(persona_i) is a weighted sum of the persona's latent traits.
    This produces realistic Heterogeneous Treatment Effects (HTEs) that the
    omnistats causal pipeline will then try to recover.

    Args
    ----
    ate           : float  Global average treatment effect (e.g. 0.05 = 5% lift).
    noise_sigma   : float  Idiosyncratic noise std.
    seed          : int    For reproducibility.
    """

    # Weights per persona dimension for heterogeneous treatment effect
    _HTE_WEIGHTS: dict[str, float] = {
        "tech_savviness":    +0.12,   # tech-savvy users respond more positively
        "risk_tolerance":    -0.08,   # risk-averse users convert less under treatment
        "impulsivity":       +0.06,   # impulsive users act faster
        "openness":          +0.04,   # open users are curious about new features
        "income":            +0.03,   # (log-scaled) higher income → higher engagement
        "neuroticism":       -0.05,   # high neuroticism → lower response
    }

    def __init__(
        self,
        ate: float = 0.05,
        noise_sigma: float = 0.02,
        seed: int = 42,
    ):
        self.ate = ate
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        df: pd.DataFrame,
        treatment_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate AB_METRIC_COL outcomes for all personas.

        Parameters
        ----------
        df             : DataFrame of personas (columns per _SCHEMA_MAP).
        treatment_mask : bool array [N] — True for treatment group.

        Returns
        -------
        outcomes : np.ndarray [N]  simulated metric values.
        """
        n = len(df)
        baseline = np.zeros(n)

        # Baseline metric for control group (e.g. baseline conversion rate ~ 0.10)
        baseline_conversion = 0.10

        # Calculate per-persona treatment effect delta
        delta = np.zeros(n)
        for col, weight in self._HTE_WEIGHTS.items():
            if col in df.columns:
                vals = df[col].fillna(df[col].median()).to_numpy()
                # Normalize income (log scale)
                if col == "income":
                    vals = np.log1p(vals) / 12.0
                delta += weight * vals

        # Treatment group gets ATE + persona-specific HTE
        noise = self.rng.normal(0, self.noise_sigma, n)
        outcomes = baseline_conversion + treatment_mask.astype(float) * (self.ate + delta) + noise
        outcomes = np.clip(outcomes, 0.0, 1.0)
        return outcomes


# =============================================================================
# Public API
# =============================================================================

def simulate_experiment(
    personas_df: pd.DataFrame,
    feature_name: str = "feature_A",
    treatment_fraction: float = 0.5,
    ate: float = 0.05,
    noise_sigma: float = 0.02,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Assign treatment/control and simulate outcomes for the MatrAIx persona population.

    Parameters
    ----------
    personas_df        : DataFrame from matraix_loader.load_matraix_personas().
    feature_name       : str   Name of the feature being A/B tested.
    treatment_fraction : float Fraction of personas assigned to treatment.
    ate                : float True Average Treatment Effect to embed (ground truth).
    noise_sigma        : float Individual outcome noise (σ).
    seed               : int   Random seed.
    verbose            : bool  Print progress.

    Returns
    -------
    pd.DataFrame with AB_GROUP_COL and AB_METRIC_COL columns added,
    saved to OUTPUT_DIR/simulated_experiment_data.csv.
    """
    rng = np.random.default_rng(seed)
    n = len(personas_df)

    df = personas_df.copy()

    # ── 1. Assign treatment/control ───────────────────────────────────────────
    shuffled_idx = rng.permutation(n)
    n_treatment = int(n * treatment_fraction)
    treatment_mask = np.zeros(n, dtype=bool)
    treatment_mask[shuffled_idx[:n_treatment]] = True

    if AB_GROUP_COL:
        df[AB_GROUP_COL] = np.where(treatment_mask, "treatment", "control")
    else:
        df["ab_group"] = np.where(treatment_mask, "treatment", "control")

    # ── 2. Simulate outcomes via Mock Outcome Function ────────────────────────
    mof = MockOutcomeFunction(ate=ate, noise_sigma=noise_sigma, seed=seed)
    outcomes = mof.simulate(df, treatment_mask)

    if AB_METRIC_COL:
        df[AB_METRIC_COL] = outcomes
    else:
        df["metric"] = outcomes

    if AB_CONVERSION_COL:
        df[AB_CONVERSION_COL] = (outcomes > 0.10).astype(int)

    # ── 3. Add feature identifier ─────────────────────────────────────────────
    df["feature_name"] = feature_name
    df["true_ate"] = ate   # Ground truth for evaluation later

    # ── 4. Save for omnistats ingestion ──────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(SIM_DATA_PATH, index=False)

    if verbose:
        n_treat = treatment_mask.sum()
        print(
            f"[MatrAIxBridge] Simulated '{feature_name}': "
            f"{n_treat} treatment / {n - n_treat} control. "
            f"ATE={ate:.4f}. Saved -> {SIM_DATA_PATH}"
        )

    return df


def run_multi_feature_sim(
    personas_df: pd.DataFrame,
    features: list[dict[str, Any]],
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Simulate multiple feature variants across the same persona population.
    This is the Phase V "Chạy Sim" entry point.

    Parameters
    ----------
    personas_df : Base persona population.
    features    : List of dicts, each with keys:
                  - 'name'  : str    Feature identifier
                  - 'ate'   : float  Embedded true ATE
                  - 'noise' : float  (optional) noise sigma
    seed        : int   Random seed.
    verbose     : bool

    Returns
    -------
    dict mapping feature_name -> simulated DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    for i, feat in enumerate(features):
        name = feat.get("name", f"feature_{i}")
        ate = feat.get("ate", 0.05)
        noise = feat.get("noise", 0.02)
        df_sim = simulate_experiment(
            personas_df=personas_df,
            feature_name=name,
            ate=ate,
            noise_sigma=noise,
            seed=seed + i,
            verbose=verbose,
        )
        results[name] = df_sim

    return results
