"""
omnistats/modules/simulation/mumo_ab_tester.py
───────────────────────────────────────────────
Phase V: OmniStats A/B Test Runner + Best Feature Selector.

Responsibilities
----------------
1. Accept a dict of {feature_name -> simulated_df} from matraix_bridge.py.
2. For each feature, temporarily redirect DATA_PATH to the simulated CSV,
   then run the full omnistats pipeline (Stages 1–4).
3. Extract the causal estimate (ATT/ATT_DR) from the APA outputs.
4. Return a ranked leaderboard of features by causal lift.
5. Identify and return the "Best" feature and its winning dataset.

This module is the "Feature A/B Test → Test ra Best" stage of the loop.
"""
from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── resolve omnistats root ────────────────────────────────────────────────────
_OMNI_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_OMNI_ROOT))

from config import OUTPUT_DIR, AB_GROUP_COL, AB_METRIC_COL


# =============================================================================
# Public API
# =============================================================================

def run_omnistats_on_sim(
    simulated_df: pd.DataFrame,
    feature_name: str,
    sim_data_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full OmniStats pipeline on a single simulated feature's data.

    Saves the DataFrame to a temporary CSV, then calls main.py logic
    programmatically. Extracts the causal ATT estimate from output CSVs.

    Parameters
    ----------
    simulated_df  : DataFrame produced by matraix_bridge.simulate_experiment().
    feature_name  : str  Name of the feature under test.
    sim_data_path : str  (optional) Override path for temp CSV.
    verbose       : bool

    Returns
    -------
    dict with keys:
        feature_name, att_estimate, cuped_reduction_pct, n_treatment, n_control
    """
    # Write simulated data to disk for omnistats to consume
    if sim_data_path is None:
        sim_data_path = os.path.join(OUTPUT_DIR, f"sim_{feature_name}.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    simulated_df.to_csv(sim_data_path, index=False)

    if verbose:
        print(f"\n[OmniStatsABTest] ── Running pipeline for '{feature_name}' ──")

    # Dynamically patch config.DATA_PATH to point to simulated data
    import config as _cfg
    _orig_data_path = _cfg.DATA_PATH
    _orig_indicator = _cfg.INDICATOR_COLS[:]
    _orig_demo = _cfg.DEMOGRAPHIC_COLS[:]

    try:
        _cfg.DATA_PATH = sim_data_path

        # Inject persona cols if not already in config
        persona_cols = [
            "openness", "conscientiousness", "extraversion",
            "agreeableness", "neuroticism", "risk_tolerance",
            "tech_savviness", "impulsivity",
        ]
        demo_cols = ["age", "income", "education_level", "location_tier"]
        for c in persona_cols:
            if c not in _cfg.INDICATOR_COLS:
                _cfg.INDICATOR_COLS.append(c)
        for c in demo_cols:
            if c not in _cfg.DEMOGRAPHIC_COLS:
                _cfg.DEMOGRAPHIC_COLS.append(c)

        # Run the omnistats data preparation and A/B testing stages
        import data_manager as dm
        importlib.reload(dm)
        df_clean = dm.load_and_prepare(verbose=verbose)

        # Stage 2: CUPED Variance Reduction
        result = {"feature_name": feature_name, "att_estimate": np.nan,
                  "cuped_reduction_pct": np.nan, "n_treatment": np.nan,
                  "n_control": np.nan}

        try:
            from modules.cuped import cuped_variance_reduction
            grp_col = AB_GROUP_COL or "ab_group"
            metric_col = AB_METRIC_COL or "metric"

            if grp_col in df_clean.columns and metric_col in df_clean.columns:
                # Run CUPED using first available indicator as covariate
                indicator_z_cols = [c + "_z" for c in persona_cols if c + "_z" in df_clean.columns]
                covariate_col = indicator_z_cols[0] if indicator_z_cols else None

                if covariate_col:
                    cuped_df, theta, pct_reduction = cuped_variance_reduction(
                        df_clean,
                        group_col=grp_col,
                        metric_col=metric_col,
                        covariate_col=covariate_col,
                        verbose=verbose,
                    )
                    result["cuped_reduction_pct"] = pct_reduction

                    # Stage 3: Simple ATT estimate from CUPED-adjusted means
                    treat_mask = cuped_df[grp_col] == "treatment"
                    ctrl_mask = cuped_df[grp_col] == "control"
                    cuped_metric = metric_col + "_cuped"
                    if cuped_metric in cuped_df.columns:
                        att = cuped_df.loc[treat_mask, cuped_metric].mean() - \
                              cuped_df.loc[ctrl_mask, cuped_metric].mean()
                    else:
                        att = cuped_df.loc[treat_mask, metric_col].mean() - \
                              cuped_df.loc[ctrl_mask, metric_col].mean()

                    result["att_estimate"] = float(att)
                    result["n_treatment"] = int(treat_mask.sum())
                    result["n_control"] = int(ctrl_mask.sum())

        except Exception as e:
            if verbose:
                print(f"[OmniStatsABTest] Warning: causal stage failed: {e}")

    finally:
        # Restore original config
        _cfg.DATA_PATH = _orig_data_path
        _cfg.INDICATOR_COLS = _orig_indicator
        _cfg.DEMOGRAPHIC_COLS = _orig_demo

    if verbose:
        print(
            f"[OmniStatsABTest] '{feature_name}': "
            f"ATT={result['att_estimate']:.4f}, "
            f"CUPED reduction={result['cuped_reduction_pct']:.1f}%"
        )

    return result


def find_best_feature(
    sim_results: dict[str, pd.DataFrame],
    verbose: bool = True,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Run OmniStats on all simulated feature variants and return the "Best" one.

    Parameters
    ----------
    sim_results : dict mapping feature_name -> simulated DataFrame.
    verbose     : bool

    Returns
    -------
    best_name     : str          Name of the winning feature.
    best_df       : pd.DataFrame Simulated data for the winning feature.
    leaderboard   : pd.DataFrame Ranked table of all features by ATT estimate.
    """
    records = []
    for name, df in sim_results.items():
        result = run_omnistats_on_sim(df, feature_name=name, verbose=verbose)
        result["true_ate"] = df["true_ate"].iloc[0] if "true_ate" in df.columns else np.nan
        records.append(result)

    leaderboard = pd.DataFrame(records).sort_values("att_estimate", ascending=False)
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard["rank"] = leaderboard.index + 1

    best_name = leaderboard.iloc[0]["feature_name"]
    best_df = sim_results[best_name]

    if verbose:
        print("\n[OmniStatsABTest] ══ FEATURE LEADERBOARD ══")
        print(leaderboard[["rank", "feature_name", "att_estimate",
                             "cuped_reduction_pct", "true_ate"]].to_string(index=False))
        print(f"\n[OmniStatsABTest] Best feature: '{best_name}' (ATT={leaderboard.iloc[0]['att_estimate']:.4f})")

    return best_name, best_df, leaderboard
