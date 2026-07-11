"""
step1_prepare_data.py
─────────────────────
Load the Titanic CSV, select indicator + demographic columns,
standardise the continuous indicators, and save a clean CSV
ready for LPA.
"""
import os
import pandas as pd
import numpy as np
from config import DATA_PATH, OUTPUT_DIR, INDICATOR_COLS, DEMOGRAPHIC_COLS


def prepare_data() -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"[Step 1] Loaded {len(df)} rows from {DATA_PATH}")

    # ── Select columns ────────────────────────────────────────────────────────
    keep = INDICATOR_COLS + DEMOGRAPHIC_COLS
    df = df[keep].copy()

    # ── Drop rows with any missing values in selected columns ─────────────────
    before = len(df)
    df.dropna(subset=keep, inplace=True)
    print(f"[Step 1] Dropped {before - len(df)} rows with NaN → {len(df)} rows remain")

    # ── Standardise continuous indicators (z-score) ───────────────────────────
    for col in INDICATOR_COLS:
        mu, sigma = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sigma

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "lpa_input.csv")
    df.to_csv(out_path, index=False)
    print(f"[Step 1] Saved prepared data → {out_path}")

    return df


if __name__ == "__main__":
    prepare_data()
