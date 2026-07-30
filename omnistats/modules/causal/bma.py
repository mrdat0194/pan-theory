"""
omnistats/modules/causal/bma.py
--------------------------------
HTE Subgroup Analysis — Doubly Robust Interaction OLS (DR-OLS).

Stage 4 of the OmniStats pipeline.  Fills the BMA slot reserved in
causal_results.csv and the APA report's Table 8.

Why DR-OLS instead of exhaustive Bayesian Model Averaging
----------------------------------------------------------
  Full BMA performs a 3^M model-space search (exponential in the number
  of demographic dummies M).  For M > 4 this is computationally prohibitive
  and numerically unstable on real-world datasets.

  DR-OLS achieves the same goal — quantifying who benefits from treatment —
  by fitting a single OLS model with all Treatment × Demographic interaction
  terms and applying HC3 heteroscedasticity-robust standard errors and
  Bonferroni correction for multiple comparisons.  It requires only
  statsmodels, which is already in requirements.txt.

Algorithm
---------
  1. Load CUPED-adjusted data from outputs/lpa_profiles.csv
     (uses {AB_METRIC_COL}_cuped column if present, else AB_METRIC_COL).
  2. Recode AB_GROUP_COL to binary {0, 1}.
  3. One-hot encode DEMOGRAPHIC_COLS (drop_first=True to avoid collinearity).
  4. Create Treatment × Demographic interaction dummies T * D_j.
  5. Fit one OLS model with intercept, T, all D_j dummies, all T*D_j:
       Y_i = α + β·T_i + Σ_j γ_j·D_ij + Σ_j δ_j·(T_i × D_ij) + ε_i
  6. Extract per-subgroup ATT = δ_j with HC3 SE.
  7. Apply Bonferroni correction to p-values for M simultaneous tests.
  8. Save full subgroup table to outputs/bma_subgroups.csv.
  9. Return marginalized ATT (β coefficient) for causal_results.csv.

Return schema (standardised — shared with DiD, IV, RDD, SCM, MC)
-----------------------------------------------------------------
  method   : "Subgroup HTE (DR-OLS)"
  estimand : "Marginalized ATT (Treatment coefficient, interaction OLS)"
  estimate : β (overall treatment effect, marginalizing over demographics)
  se       : HC3 robust SE of β
  ci_lower : β − 1.96 * SE
  ci_upper : β + 1.96 * SE
  ci_type  : "hc3_robust_bonferroni"
  p_value  : two-sided p-value for β
  n_obs    : number of observations used
  warnings : list of strings (any skipped columns / pruning notices)
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config import (
    OUTPUT_DIR, DATA_PATH,
    AB_GROUP_COL, AB_METRIC_COL,
    DEMOGRAPHIC_COLS,
    CAUSAL_BMA_ENABLED, CAUSAL_BMA_MAX_DUMMIES,
)

try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    _HAS_SM = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data() -> pd.DataFrame:
    """
    Load CUPED-adjusted data.
    Priority: outputs/lpa_profiles.csv (has profile_prob_max + _cuped col).
    Fallback:  DATA_PATH raw CSV.
    """
    cuped_path = os.path.join(OUTPUT_DIR, "lpa_profiles.csv")
    if os.path.exists(cuped_path):
        return pd.read_csv(cuped_path)
    return pd.read_csv(DATA_PATH)


def _recode_treatment(df: pd.DataFrame, group_col: str) -> pd.Series:
    """
    Recode the A/B group column to binary {0, 1}.
    The lexicographically first value → 0 (control), second → 1 (treatment).
    """
    col_data = df[group_col]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
    vals = sorted(col_data.dropna().unique())
    if len(vals) < 2:
        raise ValueError(
            f"[BMA] AB_GROUP_COL='{group_col}' has fewer than 2 unique values: {vals}"
        )
    mapping = {vals[0]: 0, vals[1]: 1}
    return col_data.map(mapping)


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def run_bma(verbose: bool = True) -> dict:
    """
    Run DR-OLS HTE Subgroup Analysis and save outputs/bma_subgroups.csv.

    Returns
    -------
    dict with standardised causal suite schema.
    """
    _warn: list[str] = []

    # ── Guard: statsmodels required ──────────────────────────────────────────
    if not _HAS_SM:
        msg = "statsmodels not installed — skipping HTE Subgroup Analysis."
        warnings.warn(msg)
        return _null_result(msg)

    if not CAUSAL_BMA_ENABLED:
        return _null_result("CAUSAL_BMA_ENABLED = False in config.py")

    # ── Load data ────────────────────────────────────────────────────────────
    try:
        df = _load_data()
    except Exception as exc:
        return _null_result(f"Data load failed: {exc}")

    # ── Select outcome: prefer CUPED-adjusted column ─────────────────────────
    cuped_col = f"{AB_METRIC_COL}_cuped"
    outcome_col = cuped_col if cuped_col in df.columns else AB_METRIC_COL

    # ── Drop rows with missing outcome or group ───────────────────────────────
    required = [outcome_col, AB_GROUP_COL] + [
        c for c in DEMOGRAPHIC_COLS if c in df.columns
    ]
    df = df[required].dropna()
    n_total = len(df)

    if n_total < 30:
        return _null_result(f"Too few observations after dropping NaN: {n_total}")

    # ── Recode treatment to {0, 1} ───────────────────────────────────────────
    try:
        T = _recode_treatment(df, AB_GROUP_COL)
    except ValueError as exc:
        return _null_result(str(exc))

    # ── One-hot encode demographics (drop_first to avoid multicollinearity) ───
    demo_cols_present = [c for c in DEMOGRAPHIC_COLS if c in df.columns]
    if not demo_cols_present:
        _warn.append("No DEMOGRAPHIC_COLS found in data — running treatment-only OLS.")

    if demo_cols_present:
        dummies = pd.get_dummies(
            df[demo_cols_present], drop_first=True, dtype=float
        )
    else:
        dummies = pd.DataFrame(index=df.index)

    # ── Prune if too many dummies (CAUSAL_BMA_MAX_DUMMIES) ───────────────────
    if len(dummies.columns) > CAUSAL_BMA_MAX_DUMMIES:
        _warn.append(
            f"Demographic dummies ({len(dummies.columns)}) exceed "
            f"CAUSAL_BMA_MAX_DUMMIES={CAUSAL_BMA_MAX_DUMMIES}. "
            f"Keeping first {CAUSAL_BMA_MAX_DUMMIES} by variance."
        )
        top_cols = dummies.var().nlargest(CAUSAL_BMA_MAX_DUMMIES).index.tolist()
        dummies = dummies[top_cols]

    dummy_names = list(dummies.columns)
    n_dummies = len(dummy_names)

    # ── Build design matrix ───────────────────────────────────────────────────
    #   Columns: intercept, T, D_1...D_k, T*D_1...T*D_k
    X = pd.DataFrame({"T": T.values}, index=df.index)
    for col in dummy_names:
        X[col] = dummies[col].values
    for col in dummy_names:
        X[f"T_x_{col}"] = T.values * dummies[col].values

    X = sm.add_constant(X, has_constant="add")
    Y = df[outcome_col].values.astype(float)

    # ── Fit OLS with HC3 robust standard errors ───────────────────────────────
    try:
        model = sm.OLS(Y, X).fit(cov_type="HC3")
    except Exception as exc:
        return _null_result(f"OLS fit failed: {exc}")

    # ── Extract per-subgroup ATTs (interaction coefficients T_x_{D_j}) ────────
    interaction_names = [f"T_x_{col}" for col in dummy_names]
    n_tests = max(len(interaction_names), 1)  # Bonferroni denominator

    subgroup_rows = []
    for iname in interaction_names:
        if iname not in model.params.index:
            continue
        coef   = float(model.params[iname])
        se     = float(model.bse[iname])
        ci_lo  = coef - 1.96 * se
        ci_hi  = coef + 1.96 * se
        pval   = float(model.pvalues[iname])
        # Bonferroni correction
        pval_bonf = min(pval * n_tests, 1.0)
        subgroup_rows.append({
            "subgroup":       iname.replace("T_x_", ""),
            "att_delta":      round(coef, 6),
            "se_hc3":         round(se, 6),
            "ci_lower_95":    round(ci_lo, 6),
            "ci_upper_95":    round(ci_hi, 6),
            "p_value_raw":    round(pval, 6),
            "p_value_bonf":   round(pval_bonf, 6),
            "significant_05": pval_bonf < 0.05,
            "n_tests":        n_tests,
        })

    subgroup_df = pd.DataFrame(subgroup_rows)

    # ── Save bma_subgroups.csv ────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subgroup_path = os.path.join(OUTPUT_DIR, "bma_subgroups.csv")
    subgroup_df.to_csv(subgroup_path, index=False)

    # ── Marginalized ATT = β coefficient on T (main effect) ──────────────────
    beta     = float(model.params["T"])
    beta_se  = float(model.bse["T"])
    beta_p   = float(model.pvalues["T"])
    ci_lo_b  = beta - 1.96 * beta_se
    ci_hi_b  = beta + 1.96 * beta_se

    # ── Verbose output ────────────────────────────────────────────────────────
    if verbose:
        print("\n[BMA/HTE] Doubly Robust Interaction OLS — Subgroup Analysis")
        print(f"  Outcome:           {outcome_col}")
        print(f"  Treatment:         {AB_GROUP_COL}")
        print(f"  Demographics:      {demo_cols_present}")
        print(f"  Dummies:           {n_dummies}  |  Observations: {n_total}")
        print(f"  Marginalized ATT:  {beta:.4f}  (SE={beta_se:.4f}, p={beta_p:.4f})")
        if not subgroup_df.empty:
            sig = subgroup_df[subgroup_df["significant_05"]]
            print(f"  Significant subgroups (Bonferroni p<0.05): {len(sig)}/{n_dummies}")
            for _, row in sig.iterrows():
                print(f"    • {row['subgroup']}: Δ={row['att_delta']:.4f} "
                      f"[{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}] "
                      f"p_bonf={row['p_value_bonf']:.4f}")
        print(f"  Saved → {subgroup_path}")

    return {
        "method":   "Subgroup HTE (DR-OLS)",
        "estimand": "Marginalized ATT (Treatment coef, interaction OLS)",
        "estimate": round(beta, 6),
        "se":       round(beta_se, 6),
        "ci_lower": round(ci_lo_b, 6),
        "ci_upper": round(ci_hi_b, 6),
        "ci_type":  "hc3_robust_bonferroni",
        "p_value":  round(beta_p, 6),
        "n_obs":    int(n_total),
        "warnings": _warn,
        "subgroup_path": subgroup_path,
    }


def _null_result(reason: str) -> dict:
    """Return a NaN-filled result dict when the estimator cannot run."""
    return {
        "method":   "Subgroup HTE (DR-OLS)",
        "estimand": "Marginalized ATT (Treatment coef, interaction OLS)",
        "estimate": float("nan"), "se": float("nan"),
        "ci_lower": float("nan"), "ci_upper": float("nan"),
        "ci_type":  "not_available", "p_value": float("nan"),
        "n_obs":    0, "warnings": [reason],
    }
