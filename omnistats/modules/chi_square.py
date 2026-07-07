"""
omnistats/modules/chi_square.py
────────────────────────────────
Chi-square independence test + Cramér's V module.
Migrated from lpa_analysis/step4_test_chi_square.py.

Tests whether latent profile membership is independent of categorical
demographic variables, reports Cramér's V as effect size.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, DEMOGRAPHIC_COLS


def _cramers_v(ct: np.ndarray) -> float:
    """Cramér's V from a contingency table array."""
    chi2, _, _, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.sum()
    k = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * k)) if (n * k) > 0 else np.nan


def run_chi_square(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Profile' column and the DEMOGRAPHIC_COLS.

    Returns
    -------
    result_df   : pd.DataFrame  — one row per demographic variable
    crosstab_df : pd.DataFrame  — observed frequency crosstabs (long format)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_rows  = []
    crosstab_dfs = []

    for demo in DEMOGRAPHIC_COLS:
        if demo not in df.columns:
            if verbose:
                print(f"[Chi2] Column '{demo}' not found — skipping")
            continue

        ct = pd.crosstab(df[demo], df["Profile"])
        chi2_stat, p_val, df_chi, _ = stats.chi2_contingency(ct.values, correction=False)
        v = _cramers_v(ct.values)

        sig = ""
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"

        result_rows.append({
            "Demographic": demo,
            "chi2":        round(chi2_stat, 3),
            "df":          int(df_chi),
            "p":           round(p_val, 4),
            "Cramers_V":   round(v, 4),
            "sig":         sig,
        })
        if verbose:
            print(f"[Chi2] {demo}: chi2={chi2_stat:.3f}, df={df_chi}, p={p_val:.4f}, V={v:.4f}")

        ct_out = ct.reset_index()
        ct_out.insert(0, "Demographic", demo)
        crosstab_dfs.append(ct_out)

    result_df   = pd.DataFrame(result_rows)
    crosstab_df = pd.concat(crosstab_dfs, ignore_index=True) if crosstab_dfs else pd.DataFrame()

    result_df.to_csv(os.path.join(OUTPUT_DIR, "chi_square_results.csv"), index=False)
    crosstab_df.to_csv(os.path.join(OUTPUT_DIR, "chi_square_tables.csv"), index=False)
    if verbose:
        print(f"[Chi2] Results saved -> {OUTPUT_DIR}")

    return result_df, crosstab_df
