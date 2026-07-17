"""
omnistats/modules/causal/matrix_completion.py
----------------------------------------------
Matrix Completion for Causal Panel Data — Athey et al. (2021).

Frames the causal counterfactual as a MISSING DATA problem.

Conceptual basis
----------------
  View the data as a matrix M of shape (N_units, T_periods).
  For any unit that receives treatment, M[i, t] for post-treatment periods
  is unobserved (missing under the "no-treatment" world).
  Matrix Completion recovers these missing entries via Nuclear Norm
  Regularisation — the same technique used by Netflix's collaborative
  filtering recommendation algorithm.

Why Matrix Completion over basic DiD
--------------------------------------
  DiD requires parallel trends and struggles with staggered adoption
  heterogeneity across units. Matrix Completion makes no such assumption.
  It learns the latent row and column factors (unit effects × time shocks)
  directly from observed entries, then imputes the counterfactuals.

  APA Explainability: Standard errors via bootstrap. Outputs a table
  of estimated ATT per cohort with confidence intervals.

Library strategy
----------------
  Primary:  fancyimpute — NuclearNormMinimization or SoftImpute
  Fallback: Custom SVD-based Alternating Least Squares (ALS)
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config import OUTPUT_DIR, CAUSAL_USE_SYNTHETIC

# ── fancyimpute primary ────────────────────────────────────────────────────────
try:
    from fancyimpute import SoftImpute
    _HAS_FANCYIMPUTE = True
except ImportError:
    _HAS_FANCYIMPUTE = False


# ── ALS Fallback ───────────────────────────────────────────────────────────────
def _als_impute(M_obs: np.ndarray, mask: np.ndarray,
                rank: int = 3, n_iter: int = 100, lambda_: float = 0.1,
                seed: int = 42) -> np.ndarray:
    """
    Alternating Least Squares (ALS) matrix completion — numpy-only fallback.

    Minimises: ||P_Omega(M - UV^T)||_F² + λ(||U||_F² + ||V||_F²)
    Where P_Omega projects onto observed entries.
    """
    rng = np.random.default_rng(seed)
    N, T = M_obs.shape

    U = rng.normal(0, 0.1, (N, rank))
    V = rng.normal(0, 0.1, (T, rank))

    for _ in range(n_iter):
        # Update U row by row
        for i in range(N):
            obs_t = mask[i, :]
            if obs_t.sum() == 0:
                continue
            V_obs = V[obs_t, :]
            M_obs_i = M_obs[i, obs_t]
            A = V_obs.T @ V_obs + lambda_ * np.eye(rank)
            b = V_obs.T @ M_obs_i
            U[i, :] = np.linalg.solve(A, b)

        # Update V col by col
        for t in range(T):
            obs_n = mask[:, t]
            if obs_n.sum() == 0:
                continue
            U_obs = U[obs_n, :]
            M_obs_t = M_obs[obs_n, t]
            A = U_obs.T @ U_obs + lambda_ * np.eye(rank)
            b = U_obs.T @ M_obs_t
            V[t, :] = np.linalg.solve(A, b)

    return U @ V.T


# ── Synthetic data generator ───────────────────────────────────────────────────
def _make_synthetic_mc() -> dict:
    """
    Panel: 20 units × 15 periods.
    10 treated units, staggered adoption at t=8 or t=10.
    True ATT = 2.0.
    """
    rng = np.random.RandomState(42)
    N, T = 20, 15
    T_cutoff = 7   # last pre-treatment period (0-indexed)
    true_att = 2.0

    # Latent factors
    row_factors = rng.randn(N, 2)
    col_factors = rng.randn(T, 2)
    M_true = row_factors @ col_factors.T + rng.randn(N, T) * 0.3

    # Treatment mask: units 0..9 are treated
    treated_units = list(range(10))
    treatment_mask = np.zeros((N, T), dtype=bool)
    for i in treated_units:
        t_start = T_cutoff + 1 if i < 5 else T_cutoff + 3
        treatment_mask[i, t_start:] = True

    # Observed matrix: treated post-periods are missing (NaN) for imputation
    M_obs = M_true.copy()
    M_obs[treatment_mask] = np.nan

    # Observed outcome (add ATT for treated post)
    M_outcome = M_true.copy()
    M_outcome[treatment_mask] += true_att

    return {
        "M_outcome":     M_outcome,
        "M_obs":         M_obs,
        "treatment_mask": treatment_mask,
        "T_cutoff":      T_cutoff,
        "true_att":      true_att,
        "N": N, "T": T,
    }


# ── Main estimator ─────────────────────────────────────────────────────────────
def matrix_completion(
    M_outcome: np.ndarray | None  = None,
    treatment_mask: np.ndarray | None = None,
    T_cutoff: int | None          = None,
    n_bootstrap: int              = 200,
    rank: int                     = 3,
    lambda_: float                = 0.1,
    verbose: bool                 = True,
) -> dict:
    """
    Matrix Completion causal estimator (Athey et al. 2021).

    Parameters
    ----------
    M_outcome      : (N, T) outcome matrix (NaN = missing/unobserved under control)
    treatment_mask : (N, T) boolean, True = treated cell
    T_cutoff       : last pre-treatment period index (0-indexed)
    n_bootstrap    : bootstrap draws for SE estimation
    rank           : latent rank for ALS fallback
    lambda_        : ridge regularisation for ALS

    Returns
    -------
    Standardised result dict (same schema as did.py / scm.py)
    """
    warns = []

    if CAUSAL_USE_SYNTHETIC or M_outcome is None:
        data = _make_synthetic_mc()
        M_outcome      = data["M_outcome"]
        treatment_mask = data["treatment_mask"]
        T_cutoff       = data["T_cutoff"]
        warns.append("Using synthetic Matrix Completion demo data (CAUSAL_USE_SYNTHETIC=True)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    N, T = M_outcome.shape

    # Observed mask = not treated post-period
    obs_mask = ~treatment_mask

    # ── Impute counterfactual ──────────────────────────────────────────────
    M_to_impute = M_outcome.copy()
    M_to_impute[treatment_mask] = np.nan   # hide treated post-periods

    if _HAS_FANCYIMPUTE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            M_completed = SoftImpute(verbose=False).fit_transform(M_to_impute)
        imputer = "SoftImpute(fancyimpute)"
    else:
        M_completed = _als_impute(
            np.where(np.isnan(M_to_impute), 0, M_to_impute),
            obs_mask, rank=rank, lambda_=lambda_
        )
        imputer = "ALS_numpy_fallback"
        warns.append("[WARNING] fancyimpute not installed — using ALS fallback")

    # ── ATT = mean gap over treated post-period cells ────────────────────
    att_cells = M_outcome[treatment_mask] - M_completed[treatment_mask]
    att       = float(np.mean(att_cells))

    # ── Bootstrap SE ──────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    boot_atts = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(N, size=N, replace=True)
        M_b      = M_outcome[boot_idx, :]
        mask_b   = treatment_mask[boot_idx, :]
        obs_b    = ~mask_b
        M_b_imp  = M_b.copy()
        M_b_imp[mask_b] = np.nan
        try:
            if _HAS_FANCYIMPUTE:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    M_b_completed = SoftImpute(verbose=False).fit_transform(M_b_imp)
            else:
                M_b_completed = _als_impute(
                    np.where(np.isnan(M_b_imp), 0, M_b_imp), obs_b,
                    rank=rank, lambda_=lambda_
                )
            boot_atts.append(float(np.mean(
                M_b[mask_b] - M_b_completed[mask_b]
            )))
        except Exception:
            pass

    if boot_atts:
        se       = float(np.std(boot_atts, ddof=1))
        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se
        ci_type  = "bootstrap_nuclear_norm"
        from scipy.stats import norm as _norm
        p_value  = float(2 * (1 - _norm.cdf(abs(att / (se + 1e-12)))))
    else:
        se = ci_lower = ci_upper = p_value = float("nan")
        ci_type = "not_available"

    # ── Plot ───────────────────────────────────────────────────────────────
    avg_treated   = M_outcome[treatment_mask.any(axis=1), :].mean(axis=0)
    avg_synthetic = M_completed[treatment_mask.any(axis=1), :].mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    periods = np.arange(T)
    ax.plot(periods, avg_treated,   "k-o", ms=4, label="Treated (observed)",  linewidth=2)
    ax.plot(periods, avg_synthetic, "r--o", ms=4, label="Counterfactual (MC)", linewidth=2)
    ax.axvline(T_cutoff + 0.5, color="grey", linestyle=":", linewidth=1.5,
               label="Treatment Start")
    ax.fill_between(periods[T_cutoff+1:], avg_synthetic[T_cutoff+1:],
                    avg_treated[T_cutoff+1:], alpha=0.2, color="red", label="Estimated ATT")
    ax.set_title("Matrix Completion: Treated vs. Imputed Counterfactual", fontweight="bold")
    ax.set_xlabel("Period"); ax.set_ylabel("Avg Outcome")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "mc_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Save gap CSV ───────────────────────────────────────────────────────
    pd.DataFrame({
        "period":         periods,
        "avg_treated":    avg_treated,
        "avg_counterfact": avg_synthetic,
        "gap":            avg_treated - avg_synthetic,
    }).to_csv(os.path.join(OUTPUT_DIR, "mc_gaps.csv"), index=False)

    result = {
        "method":    "Matrix Completion (Nuclear Norm)",
        "estimand":  "ATT_staggered_panel",
        "estimate":  round(att, 4),
        "se":        round(se, 4)        if not np.isnan(se) else float("nan"),
        "ci_lower":  round(ci_lower, 4)  if not np.isnan(ci_lower) else float("nan"),
        "ci_upper":  round(ci_upper, 4)  if not np.isnan(ci_upper) else float("nan"),
        "ci_type":   ci_type,
        "p_value":   round(p_value, 4)   if not np.isnan(p_value) else float("nan"),
        "n_obs":     int(N * T),
        "diagnostics": {
            "N_units":    N,
            "T_periods":  T,
            "T_cutoff":   T_cutoff,
            "n_treated_cells": int(treatment_mask.sum()),
            "imputer":    imputer,
            "rank":       rank,
        },
        "warnings": warns,
    }

    if verbose:
        print("\n[Causal] Matrix Completion (Nuclear Norm Regularisation)")
        print(f"  ATT estimate:  {att:.4f}")
        print(f"  SE (bootstrap): {se:.4f}" if not np.isnan(se) else "  SE: N/A")
        print(f"  95% CI:        [{ci_lower:.4f}, {ci_upper:.4f}]" if not np.isnan(ci_lower) else "  95% CI: N/A")
        print(f"  p-value:       {p_value:.4f}" if not np.isnan(p_value) else "  p-value: N/A")
        print(f"  Imputer:       {imputer}")
        print(f"  Panel:         {N} units × {T} periods ({int(treatment_mask.sum())} treated cells imputed)")
        print(f"  Saved -> mc_gaps.csv, mc_plot.png")
        for w in warns:
            print(f"  {w}")

    return result
