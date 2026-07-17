"""
omnistats/modules/cuped.py
---------------------------
CUPED — Controlled-experiment Using Pre-Experiment Data (variance reduction).
Stage 2.5 of the OmniStats pipeline.

Migrated from:  Bayesian/mono_casual.ipynb  (CatBoost monotonic regression)

Mathematical basis
------------------
  Y_i_cuped = Y_i - θ̂ · (X_i - X̄)

  Where:
    Y_i      = post-experiment outcome (AB_METRIC_COL)
    X_i      = pre-experiment covariate (profile_prob_max from Stage 1 LPA)
    θ̂        = Cov(Y, X) / Var(X)   [linear approximation]
              OR slope of a monotonic regression (CatBoost / DecisionTree)
    X̄        = mean of covariate across all units

  Why the LPA profile score is the right covariate
  -------------------------------------------------
    profile_prob_max (the posterior probability of the assigned LPA profile)
    satisfies the CUPED independence assumption:
      1. Pre-experiment: profile membership reflects stable behavioural patterns
         that existed before any treatment was applied.
      2. Correlated with outcome: high-profile-certainty users consistently show
         higher AB_METRIC_COL values (revenue, fare, engagement).
    Using AB_METRIC_COL itself as the covariate would be circular.

  Why monotonic constraints (from mono_casual.ipynb)
  --------------------------------------------------
    The profile→outcome relationship is monotone by construction: a user with
    higher certainty in a "high-value" profile almost always has a higher outcome.
    CatBoostRegressor(monotone_constraints=[+1]) enforces this, preventing
    overfitting of the hat matrix and producing a more stable θ̂.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

# ── CatBoost primary backend ───────────────────────────────────────────────────
try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

# ── sklearn Decision Tree fallback ─────────────────────────────────────────────
try:
    from sklearn.tree import DecisionTreeRegressor
    _HAS_SKLEARN_DT = True
except ImportError:
    _HAS_SKLEARN_DT = False


def _fit_monotonic(X: np.ndarray, Y: np.ndarray,
                   monotone_dir: int, use_catboost: bool,
                   seed: int = 42) -> np.ndarray:
    """
    Fit a monotone regression model and return predictions Ŷ.

    Primary:  CatBoostRegressor(monotone_constraints=[monotone_dir])
    Fallback: DecisionTreeRegressor(monotone_cst=[monotone_dir])
    Last:     OLS linear regression (guaranteed monotone for single covariate)
    """
    X2 = X.reshape(-1, 1)

    if use_catboost and _HAS_CATBOOST:
        model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            monotone_constraints=[monotone_dir],
            random_seed=seed,
            verbose=0,
        )
        model.fit(X2, Y)
        return model.predict(X2)

    if _HAS_SKLEARN_DT:
        # sklearn ≥ 1.0 supports monotone_cst
        try:
            model = DecisionTreeRegressor(
                max_depth=5,
                monotone_cst=[monotone_dir],
                random_state=seed,
            )
            model.fit(X2, Y)
            return model.predict(X2)
        except TypeError:
            pass  # older sklearn — fall through to OLS

    # OLS fallback (monotone for single covariate when sign matches monotone_dir)
    theta = np.cov(X, Y)[0, 1] / (np.var(X) + 1e-12)
    return theta * X + (Y.mean() - theta * X.mean())


def run_cuped(
    df: pd.DataFrame,
    outcome_col: str,
    covariate_col: str = "profile_prob_max",
    group_col: str | None = None,
    monotone_dir: int = 1,
    use_catboost: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply CUPED variance reduction to the outcome column.

    Parameters
    ----------
    df            : DataFrame with outcome_col and covariate_col
    outcome_col   : post-experiment outcome (e.g. "Fare", "revenue")
    covariate_col : pre-experiment covariate — defaults to "profile_prob_max"
                    (the LPA posterior probability from Stage 1)
    group_col     : if provided, θ̂ is estimated pooled across groups (recommended)
    monotone_dir  : +1 = non-decreasing, -1 = non-increasing covariate→outcome
    use_catboost  : True = CatBoost, False = sklearn DT / OLS fallback

    Returns
    -------
    df copy with a new column "{outcome_col}_cuped"
    Also saves cuped_variance_reduction.csv to OUTPUT_DIR.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = df.copy()

    # ── Guard: covariate must exist ────────────────────────────────────────
    if covariate_col not in df.columns:
        raise ValueError(
            f"[CUPED] Covariate column '{covariate_col}' not found in DataFrame.\n"
            f"  Available columns: {list(df.columns)}\n"
            f"  Ensure Stage 1 (LPA) has run and added 'profile_prob_max'."
        )
    if outcome_col not in df.columns:
        raise ValueError(f"[CUPED] Outcome column '{outcome_col}' not found.")

    mask = df[[outcome_col, covariate_col]].notna().all(axis=1)
    Y    = df.loc[mask, outcome_col].values.astype(float)
    X    = df.loc[mask, covariate_col].values.astype(float)
    X_mean = X.mean()

    # ── Fit monotonic model ────────────────────────────────────────────────
    Y_hat = _fit_monotonic(X, Y, monotone_dir, use_catboost, seed)

    # ── CUPED adjustment ───────────────────────────────────────────────────
    # θ̂ as the slope of the linear projection of Ŷ onto X
    theta_hat = np.cov(X, Y_hat)[0, 1] / (np.var(X) + 1e-12)
    Y_cuped   = Y - theta_hat * (X - X_mean)

    cuped_col = f"{outcome_col}_cuped"
    df[cuped_col] = np.nan
    df.loc[mask, cuped_col] = Y_cuped

    # ── Variance reduction ratio ───────────────────────────────────────────
    var_raw   = float(np.var(Y,       ddof=1))
    var_cuped = float(np.var(Y_cuped, ddof=1))
    reduction_pct = (1 - var_cuped / var_raw) * 100 if var_raw > 0 else 0.0

    # ── Backend label ──────────────────────────────────────────────────────
    if use_catboost and _HAS_CATBOOST:
        backend = "CatBoostRegressor(monotone)"
    elif _HAS_SKLEARN_DT:
        backend = "DecisionTreeRegressor(monotone)"
    else:
        backend = "OLS_linear_fallback"

    summary = {
        "outcome_col":     outcome_col,
        "covariate_col":   covariate_col,
        "monotone_dir":    monotone_dir,
        "theta_hat":       round(theta_hat, 6),
        "X_mean":          round(X_mean, 6),
        "var_raw":         round(var_raw, 6),
        "var_cuped":       round(var_cuped, 6),
        "variance_reduction_pct": round(reduction_pct, 2),
        "backend":         backend,
        "n_obs":           int(mask.sum()),
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(OUTPUT_DIR, "cuped_variance_reduction.csv"), index=False
    )

    if verbose:
        print("\n[CUPED] Variance Reduction — Stage 2.5")
        print(f"  Outcome:    {outcome_col}")
        print(f"  Covariate:  {covariate_col}  (LPA profile posterior probability)")
        print(f"  θ̂ (slope): {theta_hat:.6f}")
        print(f"  Var(Y_raw):   {var_raw:.4f}")
        print(f"  Var(Y_cuped): {var_cuped:.4f}")
        print(f"  Reduction:    {reduction_pct:.2f}%  [{backend}]")
        print(f"  Saved -> {OUTPUT_DIR}/cuped_variance_reduction.csv")

    return df
