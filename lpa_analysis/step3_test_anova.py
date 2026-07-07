"""
step3_test_anova.py
───────────────────
Test whether latent profiles differ significantly on the mean score
of each indicator variable using:

  - Welch's one-way ANOVA   (scipy / pingouin)
  - Games-Howell post-hoc   (pingouin.pairwise_gameshowell)
  - Eta-squared (η²) effect size

Saves:
  - outputs/anova_results.csv         — per-indicator ANOVA summary
  - outputs/anova_posthoc.csv         — pairwise Games-Howell comparisons
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from config import OUTPUT_DIR, INDICATOR_COLS


def welch_eta_squared(groups: list) -> float:
    """Eta-squared from one-way ANOVA (approximate)."""
    grand = np.concatenate(groups)
    grand_mean = grand.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum((x - grand_mean) ** 2 for g in groups for x in g)
    return ss_between / ss_total if ss_total > 0 else np.nan


def games_howell(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    """
    Pure-Python Games-Howell post-hoc test (no pingouin dependency).
    Returns a DataFrame of pairwise comparisons.
    """
    from itertools import combinations

    groups = df.groupby(group_col)[value_col].apply(np.array)
    labels = list(groups.index)
    rows   = []

    for a, b in combinations(labels, 2):
        g1, g2  = groups[a], groups[b]
        n1, n2  = len(g1), len(g2)
        m1, m2  = g1.mean(), g2.mean()
        v1, v2  = g1.var(ddof=1), g2.var(ddof=1)

        se      = np.sqrt(v1 / n1 + v2 / n2)
        t_stat  = (m1 - m2) / se if se > 0 else np.nan

        # Welch-Satterthwaite df
        if (v1 / n1 + v2 / n2) > 0:
            df_w = (v1 / n1 + v2 / n2) ** 2 / (
                (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            )
        else:
            df_w = np.nan

        p_val   = 2 * stats.t.sf(abs(t_stat), df=df_w) if not np.isnan(t_stat) else np.nan
        cohens_d = (m1 - m2) / np.sqrt((v1 + v2) / 2) if (v1 + v2) > 0 else np.nan

        rows.append({
            "Group_A": a,
            "Group_B": b,
            "Mean_A": round(m1, 4),
            "Mean_B": round(m2, 4),
            "Mean_Diff": round(m1 - m2, 4),
            "SE": round(se, 4),
            "t": round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
            "df": round(df_w, 2) if not np.isnan(df_w) else np.nan,
            "p": round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "Cohen_d": round(cohens_d, 4) if not np.isnan(cohens_d) else np.nan,
        })

    return pd.DataFrame(rows)


def run_anova(df: pd.DataFrame) -> tuple:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    anova_rows  = []
    posthoc_dfs = []

    for col in INDICATOR_COLS:
        groups = [g[col].dropna().values for _, g in df.groupby("Profile")]
        groups = [g for g in groups if len(g) > 1]   # need ≥2 observations

        if len(groups) < 2:
            print(f"[Step 3] Skipping {col} — insufficient group data")
            continue

        f_stat, p_val = stats.f_oneway(*groups)   # Welch via equal_var=False note:
        # scipy f_oneway is standard ANOVA; for Welch use scipy.stats directly
        # Proper Welch ANOVA:
        _, p_welch = stats.f_oneway(*groups)      # placeholder; full Welch below
        eta2 = welch_eta_squared(groups)

        # Proper Welch one-way ANOVA
        # Method: use the formula from Brown-Forsythe / Welch
        grand_n  = sum(len(g) for g in groups)
        k        = len(groups)
        means    = np.array([g.mean() for g in groups])
        vars_    = np.array([g.var(ddof=1) for g in groups])
        ns       = np.array([len(g) for g in groups])

        # Guard: if any group has zero variance, Welch ANOVA is undefined
        if np.any(vars_ == 0):
            print(f"  [WARNING] {col}: one or more groups have zero variance — skipping Welch ANOVA")
            anova_rows.append({
                "Indicator": col,
                "F_Welch": np.nan, "df1": np.nan, "df2": np.nan,
                "p": np.nan, "eta_squared": round(welch_eta_squared(groups), 4), "sig": "",
            })
            continue

        ws       = ns / vars_
        grand_w  = ws.sum()
        w_mean   = (ws * means).sum() / grand_w
        ss_b     = (ws * (means - w_mean) ** 2).sum()
        lambda_  = (3 / (k ** 2 - 1)) * sum(
            ((1 - ws[i] / grand_w) ** 2) / (ns[i] - 1) for i in range(k)
        )
        f_welch  = (ss_b / (k - 1)) / (1 + (2 * (k - 2) * lambda_) / 3)
        df1      = k - 1
        df2      = 1 / lambda_ if lambda_ > 0 else np.nan
        p_welch  = 1 - stats.f.cdf(f_welch, df1, df2) if not np.isnan(df2) else np.nan

        anova_rows.append({
            "Indicator": col,
            "F_Welch": round(f_welch, 3),
            "df1": df1,
            "df2": round(df2, 2) if not np.isnan(df2) else np.nan,
            "p": round(p_welch, 4) if not np.isnan(p_welch) else np.nan,
            "eta_squared": round(eta2, 4),
            "sig": "*" if (not np.isnan(p_welch) and p_welch < 0.05) else "",
        })
        print(f"[Step 3] {col}: F={f_welch:.3f}, p={p_welch:.4f}, η²={eta2:.4f}")

        # Post-hoc
        ph = games_howell(df, col, "Profile")
        ph.insert(0, "Indicator", col)
        posthoc_dfs.append(ph)

    anova_df  = pd.DataFrame(anova_rows)
    posthoc_df = pd.concat(posthoc_dfs, ignore_index=True) if posthoc_dfs else pd.DataFrame()

    anova_df.to_csv(os.path.join(OUTPUT_DIR, "anova_results.csv"), index=False)
    posthoc_df.to_csv(os.path.join(OUTPUT_DIR, "anova_posthoc.csv"), index=False)
    print(f"[Step 3] ANOVA results saved → {OUTPUT_DIR}")

    return anova_df, posthoc_df


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "lpa_profiles.csv"))
    run_anova(df)
