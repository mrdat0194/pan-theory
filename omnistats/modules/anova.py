"""
omnistats/modules/anova.py
──────────────────────────
Welch's ANOVA + Games-Howell post-hoc module.
Migrated from lpa_analysis/step3_test_anova.py.

Tests whether latent profiles differ significantly on each indicator
variable using heteroscedasticity-robust Welch ANOVA and pairwise
Games-Howell comparisons.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, INDICATOR_COLS


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _eta_squared(groups: list) -> float:
    grand      = np.concatenate(groups)
    grand_mean = grand.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum((x - grand_mean) ** 2 for g in groups for x in g)
    return ss_between / ss_total if ss_total > 0 else np.nan


def _games_howell(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    """Pure-Python Games-Howell pairwise post-hoc test."""
    groups = df.groupby(group_col)[value_col].apply(np.array)
    labels = list(groups.index)
    rows   = []

    for a, b in combinations(labels, 2):
        g1, g2 = groups[a], groups[b]
        n1, n2 = len(g1), len(g2)
        m1, m2 = g1.mean(), g2.mean()
        v1, v2 = g1.var(ddof=1), g2.var(ddof=1)

        denom = v1 / n1 + v2 / n2
        se      = np.sqrt(denom) if denom > 0 else np.nan
        t_stat  = (m1 - m2) / se if (se and se > 0) else np.nan

        if denom > 0:
            df_w = denom ** 2 / (
                (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            )
        else:
            df_w = np.nan

        p_val    = 2 * stats.t.sf(abs(t_stat), df=df_w) if not np.isnan(t_stat) else np.nan
        cohens_d = (m1 - m2) / np.sqrt((v1 + v2) / 2) if (v1 + v2) > 0 else np.nan

        rows.append({
            "Group_A":  a,   "Group_B":   b,
            "Mean_A":   round(m1, 4),     "Mean_B":   round(m2, 4),
            "Mean_Diff":round(m1 - m2, 4),"SE":       round(se, 4) if not np.isnan(se) else np.nan,
            "t":        round(t_stat, 4)  if not np.isnan(t_stat) else np.nan,
            "df":       round(df_w, 2)    if not np.isnan(df_w)   else np.nan,
            "p":        round(p_val, 4)   if not np.isnan(p_val)  else np.nan,
            "Cohen_d":  round(cohens_d, 4) if not np.isnan(cohens_d) else np.nan,
        })

    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_anova(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Profile' column and the INDICATOR_COLS.

    Returns
    -------
    anova_df : pd.DataFrame   — per-indicator Welch ANOVA results
    posthoc_df : pd.DataFrame — pairwise Games-Howell comparisons
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anova_rows  = []
    posthoc_dfs = []

    for col in INDICATOR_COLS:
        if col not in df.columns:
            continue
        groups = [g[col].dropna().values for _, g in df.groupby("Profile")]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            if verbose:
                print(f"[ANOVA] Skipping {col} — insufficient group data")
            continue

        eta2     = _eta_squared(groups)
        means    = np.array([g.mean() for g in groups])
        vars_    = np.array([g.var(ddof=1) for g in groups])
        ns       = np.array([len(g) for g in groups])
        k        = len(groups)

        if np.any(vars_ == 0):
            if verbose:
                print(f"  [WARNING] {col}: zero-variance group — skipping Welch ANOVA")
            anova_rows.append({
                "Indicator": col, "F_Welch": np.nan, "df1": np.nan,
                "df2": np.nan, "p": np.nan,
                "eta_squared": round(eta2, 4), "sig": "",
            })
            continue

        ws      = ns / vars_
        grand_w = ws.sum()
        w_mean  = (ws * means).sum() / grand_w
        ss_b    = (ws * (means - w_mean) ** 2).sum()
        lam_    = (3 / (k ** 2 - 1)) * sum(
            ((1 - ws[i] / grand_w) ** 2) / (ns[i] - 1) for i in range(k)
        )
        f_welch = (ss_b / (k - 1)) / (1 + (2 * (k - 2) * lam_) / 3)
        df1     = k - 1
        df2     = 1 / lam_ if lam_ > 0 else np.nan
        p_welch = 1 - stats.f.cdf(f_welch, df1, df2) if not np.isnan(df2) else np.nan

        sig = ""
        if not np.isnan(p_welch):
            if p_welch < 0.001: sig = "***"
            elif p_welch < 0.01: sig = "**"
            elif p_welch < 0.05: sig = "*"

        anova_rows.append({
            "Indicator":   col,
            "F_Welch":     round(f_welch, 3),
            "df1":         df1,
            "df2":         round(df2, 2) if not np.isnan(df2) else np.nan,
            "p":           round(p_welch, 4) if not np.isnan(p_welch) else np.nan,
            "eta_squared": round(eta2, 4),
            "sig":         sig,
        })
        if verbose:
            print(f"[ANOVA] {col}: F={f_welch:.3f}, p={p_welch:.4f}, eta2={eta2:.4f}")

        ph = _games_howell(df, col, "Profile")
        ph.insert(0, "Indicator", col)
        posthoc_dfs.append(ph)

    anova_df   = pd.DataFrame(anova_rows)
    posthoc_df = pd.concat(posthoc_dfs, ignore_index=True) if posthoc_dfs else pd.DataFrame()

    anova_df.to_csv(os.path.join(OUTPUT_DIR, "anova_results.csv"), index=False)
    posthoc_df.to_csv(os.path.join(OUTPUT_DIR, "anova_posthoc.csv"), index=False)
    if verbose:
        print(f"[ANOVA] Results saved -> {OUTPUT_DIR}")

    return anova_df, posthoc_df
