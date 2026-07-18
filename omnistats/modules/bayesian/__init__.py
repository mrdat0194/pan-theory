"""
omnistats/modules/bayesian/__init__.py
---------------------------------------
Sequential Bayesian A/B Testing subpackage — Stage 2 upgrade.

Migrated and adapted from:
  Bayesian/abtesting/abtesting_suite.py
  Bayesian/importance_sampling_bayesian.py
  Bayesian/mcmc_bayesian.py

Exports
-------
bayesian_proportion_test()  — Beta-Binomial conjugate
bayesian_means_test()       — PyMC NUTS StudentT (Mandatory)
sequential_monitor()        — Batch-by-batch SIR stopping rule
expected_loss()             — EVSI decision criterion
run_bayesian_ab_tests()     — Unified DataFrame-based orchestrator
"""

from .beta_binomial import bayesian_proportion_test
from .normal_model  import bayesian_means_test
from .sequential    import sequential_monitor, expected_loss

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config import OUTPUT_DIR


def run_bayesian_ab_tests(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    conversion_col: str = None,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    threshold: float = 0.95,
    loss_thresh: float = 0.01,
    n_samples: int = 2_000,
    tune: int = 1_000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    DataFrame-based orchestrator for Bayesian A/B tests.

    Parameters
    ----------
    df             : DataFrame with group_col and metric_col
    group_col      : column with exactly 2 unique group labels
    metric_col     : continuous outcome metric
    conversion_col : optional binary (0/1) conversion column
    prior_alpha    : Beta prior α (conversion tests)
    prior_beta     : Beta prior β
    threshold      : P(B > A) threshold to declare winner
    loss_thresh    : Expected Loss threshold to stop
    n_samples      : PyMC posterior draws (NUTS)
    tune           : PyMC tuning draws (discarded)
    seed           : random seed

    Returns
    -------
    dict with keys: "means", "proportion" (if applicable)
    Each value is the standardised result dict.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    groups = df[group_col].unique()
    if len(groups) < 2:
        print(f"[Bayesian A/B] Need at least 2 groups in '{group_col}'. Found: {groups}")
        return {}

    g_a, g_b = groups[0], groups[1]
    ctrl = df[df[group_col] == g_a][metric_col].dropna().values
    trt  = df[df[group_col] == g_b][metric_col].dropna().values

    results = {}

    # ── Continuous metric test ─────────────────────────────────────────────
    results["means"] = bayesian_means_test(
        ctrl, trt,
        label_control=str(g_a), label_treatment=str(g_b),
        n_samples=n_samples, tune=tune, seed=seed,
        threshold=threshold, loss_thresh=loss_thresh,
        verbose=verbose,
    )

    # ── Conversion / proportion test ───────────────────────────────────────
    if conversion_col and conversion_col in df.columns:
        n_a    = len(df[df[group_col] == g_a])
        conv_a = int(df[df[group_col] == g_a][conversion_col].sum())
        n_b    = len(df[df[group_col] == g_b])
        conv_b = int(df[df[group_col] == g_b][conversion_col].sum())

        results["proportion"] = bayesian_proportion_test(
            n_a, conv_a, n_b, conv_b,
            prior_alpha=prior_alpha, prior_beta=prior_beta,
            threshold=threshold, loss_thresh=loss_thresh,
            verbose=verbose,
        )

    # ── Save summary CSV ───────────────────────────────────────────────────
    rows = [{"test": k, **v} for k, v in results.items()]
    out_path = os.path.join(OUTPUT_DIR, "bayesian_ab_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    if verbose:
        print(f"[Bayesian A/B] Results saved -> {out_path}")

    return results


__all__ = [
    "bayesian_proportion_test",
    "bayesian_means_test",
    "sequential_monitor",
    "expected_loss",
    "run_bayesian_ab_tests",
]
