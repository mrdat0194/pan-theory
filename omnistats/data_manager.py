"""
omnistats/data_manager.py
─────────────────────────
Centralised data loading, cleaning, and z-score standardisation.
Replaces lpa_analysis/step1_prepare_data.py with a reusable utility.
"""
import os
import pandas as pd
import numpy as np
from config import DATA_PATH, OUTPUT_DIR, INDICATOR_COLS, DEMOGRAPHIC_COLS


def load_and_prepare(verbose: bool = True) -> pd.DataFrame:
    """
    Load the configured dataset, drop NaN rows, z-score the indicator
    columns, and return a clean DataFrame ready for all pipeline modules.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if verbose:
        print(f"[DataManager] Loaded {len(df)} rows from {DATA_PATH}")

    keep = INDICATOR_COLS + DEMOGRAPHIC_COLS
    df = df[[c for c in keep if c in df.columns]].copy()

    before = len(df)
    df.dropna(inplace=True)
    if verbose:
        print(f"[DataManager] Dropped {before - len(df)} rows with NaN -> {len(df)} rows remain")

    # Z-score standardise continuous indicators
    for col in INDICATOR_COLS:
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std()
            df[f"{col}_z"] = (df[col] - mu) / (sigma if sigma > 0 else 1.0)

    out_path = os.path.join(OUTPUT_DIR, "prepared_data.csv")
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"[DataManager] Saved prepared data -> {out_path}")

    return df
