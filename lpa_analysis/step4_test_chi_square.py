"""
step4_test_chi_square.py
────────────────────────
Test whether the latent profiles differ significantly on three
categorical demographic variables using:

  - Pearson chi-square test  (scipy.stats.chi2_contingency)
  - Cramér's V effect size

Saves:
  - outputs/chi_square_results.csv    — one row per demographic variable
  - outputs/chi_square_tables.csv     — observed frequency crosstabs
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from config import OUTPUT_DIR, DEMOGRAPHIC_COLS


def cramers_v(contingency_table: np.ndarray) -> float:
    """Cramér's V from a contingency table."""
    chi2, _, _, _ = stats.chi2_contingency(contingency_table, correction=False)
    n   = contingency_table.sum()
    k   = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * k)) if (n * k) > 0 else np.nan


def run_chi_square(df: pd.DataFrame) -> tuple:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result_rows  = []
    crosstab_dfs = []

    for demo in DEMOGRAPHIC_COLS:
        if demo not in df.columns:
            print(f"[Step 4] Column '{demo}' not found — skipping")
            continue

        # Build contingency table: rows = demographic categories, cols = profiles
        ct = pd.crosstab(df[demo], df["Profile"])
        chi2_stat, p_val, df_chi, expected = stats.chi2_contingency(ct.values, correction=False)
        v = cramers_v(ct.values)

        result_rows.append({
            "Demographic": demo,
            "chi2": round(chi2_stat, 3),
            "df": int(df_chi),
            "p": round(p_val, 4),
            "Cramers_V": round(v, 4),
            "sig": "*" if p_val < 0.05 else "",
        })
        print(f"[Step 4] {demo}: χ²={chi2_stat:.3f}, df={df_chi}, p={p_val:.4f}, V={v:.4f}")

        # Save crosstab
        ct_out = ct.reset_index()
        ct_out.insert(0, "Demographic", demo)
        crosstab_dfs.append(ct_out)

    result_df   = pd.DataFrame(result_rows)
    crosstab_df = pd.concat(crosstab_dfs, ignore_index=True) if crosstab_dfs else pd.DataFrame()

    result_df.to_csv(os.path.join(OUTPUT_DIR, "chi_square_results.csv"), index=False)
    crosstab_df.to_csv(os.path.join(OUTPUT_DIR, "chi_square_tables.csv"), index=False)
    print(f"[Step 4] Chi-square results saved → {OUTPUT_DIR}")

    return result_df, crosstab_df


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "lpa_profiles.csv"))
    run_chi_square(df)
