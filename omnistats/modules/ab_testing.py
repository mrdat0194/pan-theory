"""
omnistats/modules/ab_testing.py
────────────────────────────────
A/B Testing module integrated from Bayesian/abtesting/abtesting_suite.py.

Provides three test types:
  1. proportion_test()  — Two-proportion z-test for binary conversion metrics
  2. means_test()       — Welch's t-test for continuous metrics
  3. distribution_fit() — Chi-square goodness-of-fit against Normal distribution

Each function prints a clear result summary and saves outputs to OUTPUT_DIR.
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR


# ─── 1. Proportion / Conversion Test ─────────────────────────────────────────

def proportion_test(
    n_control: int,
    conv_control: int,
    n_treatment: int,
    conv_treatment: int,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Two-proportion z-test.

    Parameters
    ----------
    n_control / n_treatment : int   — group sample sizes
    conv_control / conv_treatment : int — conversions in each group

    Returns
    -------
    dict with keys: p_ctrl, p_treat, lift, z_stat, p_value, significant
    """
    p_ctrl  = conv_control  / n_control
    p_treat = conv_treatment / n_treatment
    lift    = (p_treat - p_ctrl) / p_ctrl if p_ctrl > 0 else np.nan

    # Pooled proportion
    p_pool = (conv_control + conv_treatment) / (n_control + n_treatment)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z_stat = (p_treat - p_ctrl) / se if se > 0 else np.nan
    p_val  = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan

    result = {
        "p_ctrl": round(p_ctrl, 4),
        "p_treat": round(p_treat, 4),
        "lift_pct": round(lift * 100, 2) if not np.isnan(lift) else np.nan,
        "z_stat": round(z_stat, 4) if not np.isnan(z_stat) else np.nan,
        "p_value": round(p_val, 4) if not np.isnan(p_val) else np.nan,
        "significant": bool(p_val < alpha) if not np.isnan(p_val) else False,
        "alpha": alpha,
    }

    if verbose:
        print("\n[A/B] Proportion Test")
        print(f"  Control:   {p_ctrl:.2%}  (n={n_control})")
        print(f"  Treatment: {p_treat:.2%} (n={n_treatment})")
        print(f"  Lift:      {result['lift_pct']:+.2f}%")
        print(f"  z = {z_stat:.4f},  p = {p_val:.4f}  {'SIGNIFICANT' if result['significant'] else 'not significant'} (alpha={alpha})")

    return result


# ─── 2. Means Comparison Test ─────────────────────────────────────────────────

def means_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    label_control: str = "Control",
    label_treatment: str = "Treatment",
    verbose: bool = True,
) -> dict:
    """
    Welch's t-test for comparing two group means.

    Parameters
    ----------
    control / treatment : array-like — observations for each group

    Returns
    -------
    dict with keys: mean_ctrl, mean_treat, diff, t_stat, p_value, cohen_d, significant
    """
    ctrl = np.asarray(control)
    trt  = np.asarray(treatment)

    t_stat, p_val = scipy_stats.ttest_ind(ctrl, trt, equal_var=False)
    diff    = trt.mean() - ctrl.mean()
    pooled_std = np.sqrt((ctrl.std(ddof=1) ** 2 + trt.std(ddof=1) ** 2) / 2)
    cohen_d = diff / pooled_std if pooled_std > 0 else np.nan

    result = {
        "mean_ctrl":  round(ctrl.mean(), 4),
        "mean_treat": round(trt.mean(), 4),
        "diff":       round(diff, 4),
        "t_stat":     round(t_stat, 4),
        "p_value":    round(p_val, 4),
        "cohen_d":    round(cohen_d, 4) if not np.isnan(cohen_d) else np.nan,
        "significant": bool(p_val < alpha),
        "alpha":      alpha,
    }

    if verbose:
        print(f"\n[A/B] Means Test ({label_control} vs {label_treatment})")
        print(f"  {label_control}:   mean={ctrl.mean():.4f},  sd={ctrl.std(ddof=1):.4f},  n={len(ctrl)}")
        print(f"  {label_treatment}: mean={trt.mean():.4f}, sd={trt.std(ddof=1):.4f},  n={len(trt)}")
        print(f"  Diff: {diff:+.4f}   Cohen's d: {cohen_d:.4f}")
        print(f"  t = {t_stat:.4f},  p = {p_val:.4f}  {'SIGNIFICANT' if result['significant'] else 'not significant'} (alpha={alpha})")

    return result


# ─── 3. Distribution Fit Test ─────────────────────────────────────────────────

def distribution_fit_test(
    data: np.ndarray,
    n_bins: int = 20,
    label: str = "metric",
    save_plot: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Chi-square goodness-of-fit: tests if data follows a Normal distribution.
    Bins the data and compares observed vs expected frequencies.

    Returns
    -------
    dict with keys: chi2_stat, p_value, df, significant
    """
    data    = np.asarray(data).flatten()
    mu, sig = data.mean(), data.std()

    counts, bin_edges = np.histogram(data, bins=n_bins)
    # Expected counts under fitted Normal
    expected = np.array([
        len(data) * (
            scipy_stats.norm.cdf(bin_edges[i + 1], mu, sig)
            - scipy_stats.norm.cdf(bin_edges[i], mu, sig)
        )
        for i in range(n_bins)
    ])
    # Merge bins with expected < 5 to satisfy chi-square assumption
    valid = expected >= 5
    obs_v = counts[valid]
    exp_v = expected[valid]

    chi2_stat = float(np.sum((obs_v - exp_v) ** 2 / exp_v)) if exp_v.sum() > 0 else np.nan
    dof       = max(len(obs_v) - 3, 1)          # -3: estimated mu, sigma, normalise
    p_val     = 1 - scipy_stats.chi2.cdf(chi2_stat, dof) if not np.isnan(chi2_stat) else np.nan

    result = {
        "chi2_stat":   round(chi2_stat, 4) if not np.isnan(chi2_stat) else np.nan,
        "p_value":     round(p_val, 4)     if not np.isnan(p_val)     else np.nan,
        "df":          dof,
        "significant": bool(p_val < 0.05)  if not np.isnan(p_val)     else False,
        "mu":          round(mu, 4),
        "sigma":       round(sig, 4),
    }

    if verbose:
        print(f"\n[A/B] Distribution Fit Test ({label})")
        print(f"  Normal fit: mu={mu:.4f}, sigma={sig:.4f}")
        print(f"  chi2={chi2_stat:.4f},  df={dof},  p={p_val:.4f}  "
              f"{'Rejects normality' if result['significant'] else 'Does not reject normality'} at alpha=0.05")

    if save_plot:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(data, bins=n_bins, density=True, alpha=0.6, color="#2E86AB", label="Observed")
        x_range = np.linspace(data.min(), data.max(), 300)
        ax.plot(x_range, scipy_stats.norm.pdf(x_range, mu, sig),
                color="#E84855", linewidth=2, label=f"Normal fit (μ={mu:.2f}, σ={sig:.2f})")
        ax.set_title(f"Distribution Fit: {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel(label); ax.set_ylabel("Density")
        ax.legend(framealpha=0.9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"dist_fit_{label}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  Plot saved -> {OUTPUT_DIR}/dist_fit_{label}.png")

    return result


# ─── Unified runner for DataFrame-based A/B ───────────────────────────────────

def run_ab_tests(df: pd.DataFrame,
                 group_col: str,
                 metric_col: str,
                 conversion_col: str = None,
                 alpha: float = 0.05,
                 verbose: bool = True) -> dict:
    """
    Convenience wrapper: extracts two groups from df[group_col], then runs
    means_test() and optionally proportion_test() on df[conversion_col].

    Parameters
    ----------
    df             : DataFrame with at least group_col and metric_col
    group_col      : column with exactly 2 unique values (A / B)
    metric_col     : continuous metric column
    conversion_col : optional binary (0/1) column for proportion test
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = df[group_col].unique()
    if len(groups) < 2:
        print(f"[A/B] Need at least 2 groups in '{group_col}'. Found: {groups}")
        return {}

    g_a, g_b = groups[0], groups[1]
    ctrl = df[df[group_col] == g_a][metric_col].dropna().values
    trt  = df[df[group_col] == g_b][metric_col].dropna().values

    results = {}
    results["means"] = means_test(ctrl, trt, alpha=alpha,
                                  label_control=str(g_a), label_treatment=str(g_b),
                                  verbose=verbose)
    results["dist_ctrl"] = distribution_fit_test(ctrl, label=f"{metric_col}_{g_a}",
                                                  save_plot=True, verbose=verbose)
    results["dist_trt"]  = distribution_fit_test(trt,  label=f"{metric_col}_{g_b}",
                                                  save_plot=True, verbose=verbose)

    if conversion_col and conversion_col in df.columns:
        n_a    = len(df[df[group_col] == g_a])
        conv_a = int(df[df[group_col] == g_a][conversion_col].sum())
        n_b    = len(df[df[group_col] == g_b])
        conv_b = int(df[df[group_col] == g_b][conversion_col].sum())
        results["proportion"] = proportion_test(n_a, conv_a, n_b, conv_b,
                                                alpha=alpha, verbose=verbose)

    # Save summary CSV
    rows = []
    for test_name, res in results.items():
        rows.append({"test": test_name, **res})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "ab_test_results.csv"), index=False)
    if verbose:
        print(f"[A/B] Results saved -> {OUTPUT_DIR}/ab_test_results.csv")

    return results
