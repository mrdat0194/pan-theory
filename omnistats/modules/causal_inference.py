"""
omnistats/modules/causal_inference.py
──────────────────────────────────────
Causal Inference module integrated from Bayesian/someMethod/*.py.

Provides three causal methods:
  1. diff_in_diff()            — Difference-in-Differences (DiD)
  2. instrumental_variables()  — IV / 2SLS estimator
  3. regression_discontinuity()— Sharp RDD

Each function uses synthetic data by default (CAUSAL_USE_SYNTHETIC=True in
config.py) and prints an interpretable summary of the estimated causal effect.
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, CAUSAL_USE_SYNTHETIC

_RNG = np.random.RandomState(42)


# ─── 1. Difference-in-Differences ────────────────────────────────────────────

def diff_in_diff(
    df: pd.DataFrame = None,
    outcome_col: str = "outcome",
    group_col: str = "treated",
    time_col: str = "post",
    verbose: bool = True,
) -> dict:
    """
    Estimates the Average Treatment Effect (ATE) using a 2x2 DiD design.

    If df is None or CAUSAL_USE_SYNTHETIC=True, generates synthetic panel data.
    """
    if df is None or CAUSAL_USE_SYNTHETIC:
        n = 500
        treated = _RNG.binomial(1, 0.5, n)
        post    = _RNG.binomial(1, 0.5, n)
        true_ate = 8.0
        outcome = (
            20
            + 5  * treated
            + 3  * post
            + true_ate * treated * post
            + _RNG.normal(0, 3, n)
        )
        df = pd.DataFrame({outcome_col: outcome, group_col: treated, time_col: post})
        if verbose:
            print("[DiD] Using synthetic data  (true ATE = 8.0)")

    # 2x2 DiD formula
    means = df.groupby([group_col, time_col])[outcome_col].mean()
    try:
        did_est = (means[1, 1] - means[1, 0]) - (means[0, 1] - means[0, 0])
    except KeyError:
        did_est = np.nan

    # OLS with interaction for SE + p-value
    try:
        import statsmodels.formula.api as smf
        formula = f"{outcome_col} ~ {group_col} * {time_col}"
        model   = smf.ols(formula, data=df).fit()
        inter_name = f"{group_col}:{time_col}"
        ate_se  = model.bse.get(inter_name, np.nan)
        ate_p   = model.pvalues.get(inter_name, np.nan)
        ate_ci  = model.conf_int().loc[inter_name].tolist() if inter_name in model.conf_int().index else [np.nan, np.nan]
    except Exception:
        ate_se, ate_p, ate_ci = np.nan, np.nan, [np.nan, np.nan]

    result = {
        "method": "Difference-in-Differences",
        "ATE_est": round(float(did_est), 4) if not np.isnan(did_est) else np.nan,
        "SE":      round(float(ate_se), 4)  if not np.isnan(ate_se)  else np.nan,
        "p_value": round(float(ate_p), 4)   if not np.isnan(ate_p)   else np.nan,
        "CI_lower": round(ate_ci[0], 4)     if not np.isnan(ate_ci[0]) else np.nan,
        "CI_upper": round(ate_ci[1], 4)     if not np.isnan(ate_ci[1]) else np.nan,
    }

    if verbose:
        print(f"\n[DiD] ATE = {result['ATE_est']:.4f}  (SE={result['SE']:.4f},  p={result['p_value']:.4f})")
        print(f"  95% CI: [{result['CI_lower']:.4f}, {result['CI_upper']:.4f}]")

    return result


# ─── 2. Instrumental Variables ────────────────────────────────────────────────

def instrumental_variables(
    df: pd.DataFrame = None,
    outcome_col: str = "outcome",
    treatment_col: str = "treatment",
    instrument_col: str = "instrument",
    verbose: bool = True,
) -> dict:
    """
    Two-Stage Least Squares (2SLS) IV estimator.
    If df is None, uses synthetic data with a known true effect of 3.0.
    """
    if df is None or CAUSAL_USE_SYNTHETIC:
        n    = 600
        z    = _RNG.binomial(1, 0.5, n)      # instrument (random assignment)
        u    = _RNG.normal(0, 1, n)           # unobserved confounder
        d    = (0.6 * z + 0.4 * u + _RNG.normal(0, 0.3, n)) > 0   # endogenous treatment
        d    = d.astype(float)
        true_effect = 3.0
        y    = 5 + true_effect * d + 2 * u + _RNG.normal(0, 1, n)
        df   = pd.DataFrame({outcome_col: y, treatment_col: d, instrument_col: z})
        if verbose:
            print("[IV] Using synthetic data  (true effect = 3.0)")

    # First stage: D ~ Z
    d    = df[treatment_col].values
    z    = df[instrument_col].values
    y    = df[outcome_col].values

    first_coef = np.cov(d, z)[0, 1] / np.var(z)   # simple slope D~Z
    d_hat      = first_coef * z

    # Second stage: Y ~ D_hat
    n     = len(y)
    d_hat_mean = d_hat.mean()
    y_mean     = y.mean()
    tsls_slope = np.sum((d_hat - d_hat_mean) * (y - y_mean)) / np.sum((d_hat - d_hat_mean) ** 2)

    result = {
        "method":          "Instrumental Variables (2SLS)",
        "LATE_est":        round(float(tsls_slope), 4),
        "first_stage_F":   round(float((first_coef ** 2 * np.var(z) * n) /
                                       np.var(d - first_coef * z)), 4),
    }

    if verbose:
        print(f"\n[IV] LATE = {result['LATE_est']:.4f}  (First-stage F = {result['first_stage_F']:.2f})")
        if result["first_stage_F"] < 10:
            print("  [WARNING] Weak instrument (F < 10).")

    return result


# ─── 3. Regression Discontinuity ─────────────────────────────────────────────

def regression_discontinuity(
    df: pd.DataFrame = None,
    outcome_col: str = "outcome",
    running_col: str = "score",
    cutoff: float = 50.0,
    bandwidth: float = 20.0,
    verbose: bool = True,
    save_plot: bool = True,
) -> dict:
    """
    Sharp RDD: estimates the treatment effect at the cutoff.
    Fits local linear regressions on each side of the cutoff within bandwidth.
    """
    if df is None or CAUSAL_USE_SYNTHETIC:
        n    = 1000
        x    = _RNG.uniform(0, 100, n)
        treat = (x >= cutoff).astype(float)
        true_effect = 15.0
        y    = 10 + 0.5 * x + true_effect * treat + _RNG.normal(0, 5, n)
        df   = pd.DataFrame({outcome_col: y, running_col: x})
        if verbose:
            print(f"[RDD] Using synthetic data  (true effect = {true_effect}, cutoff = {cutoff})")

    df     = df.copy()
    df["_above"]    = (df[running_col] >= cutoff).astype(float)
    df["_centered"] = df[running_col] - cutoff

    # Restrict to bandwidth
    bw_df = df[np.abs(df["_centered"]) <= bandwidth].copy()

    try:
        import statsmodels.formula.api as smf
        model = smf.ols(f"{outcome_col} ~ _centered * _above", data=bw_df).fit()
        rdd_est = model.params.get("_above", np.nan)
        rdd_se  = model.bse.get("_above", np.nan)
        rdd_p   = model.pvalues.get("_above", np.nan)
    except Exception:
        rdd_est = rdd_se = rdd_p = np.nan

    result = {
        "method":   "Regression Discontinuity",
        "cutoff":   cutoff,
        "bandwidth": bandwidth,
        "RDD_est":  round(float(rdd_est), 4) if not np.isnan(rdd_est) else np.nan,
        "SE":       round(float(rdd_se), 4)  if not np.isnan(rdd_se)  else np.nan,
        "p_value":  round(float(rdd_p), 4)   if not np.isnan(rdd_p)   else np.nan,
    }

    if verbose:
        print(f"\n[RDD] Effect at cutoff = {result['RDD_est']:.4f}  (SE={result['SE']:.4f},  p={result['p_value']:.4f})")

    if save_plot:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        left  = df[df["_centered"] < 0]
        right = df[df["_centered"] >= 0]
        ax.scatter(left[running_col],  left[outcome_col],  alpha=0.3, s=8, color="#2E86AB", label="Control")
        ax.scatter(right[running_col], right[outcome_col], alpha=0.3, s=8, color="#E84855", label="Treatment")
        for side, side_df, color in [(left, left, "#2E86AB"), (right, right, "#E84855")]:
            x_s = side_df[running_col].sort_values()
            if len(x_s) > 2:
                fit   = np.polyfit(x_s, side_df.loc[x_s.index, outcome_col], 1)
                ax.plot(x_s, np.poly1d(fit)(x_s), color=color, linewidth=2)
        ax.axvline(cutoff, color="gray", linestyle="--", linewidth=1.2, label=f"Cutoff={cutoff}")
        ax.set_title("Regression Discontinuity Design", fontsize=12, fontweight="bold")
        ax.set_xlabel(running_col); ax.set_ylabel(outcome_col)
        ax.legend(framealpha=0.9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "rdd_plot.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  Plot saved -> {OUTPUT_DIR}/rdd_plot.png")

    return result


# ─── Unified runner ───────────────────────────────────────────────────────────

def run_causal_suite(verbose: bool = True) -> dict:
    """Run all three causal inference methods and save a combined summary CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    results["did"] = diff_in_diff(verbose=verbose)
    results["iv"]  = instrumental_variables(verbose=verbose)
    results["rdd"] = regression_discontinuity(verbose=verbose)

    rows = []
    for key, res in results.items():
        rows.append(res)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "causal_results.csv"), index=False)
    if verbose:
        print(f"\n[Causal] All results saved -> {OUTPUT_DIR}/causal_results.csv")

    return results
