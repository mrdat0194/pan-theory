"""
omnistats/config.py
───────────────────
Unified configuration for the OmniStats pipeline.
Edit this file to point to your data and choose your analytical parameters.
"""
import os

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "adHoc", "titanic.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ─── LPA SETTINGS ─────────────────────────────────────────────────────────────
# Update N_PROFILES after reviewing outputs/lpa_fit_stats.csv with your co-author.
N_PROFILES    = 3
K_MIN         = 1
K_MAX         = 6

# ─── INDICATOR & DEMOGRAPHIC COLUMNS ──────────────────────────────────────────
INDICATOR_COLS   = ["Age", "Fare", "SibSp", "Parch"]
DEMOGRAPHIC_COLS = ["Sex", "Pclass", "Embarked"]

# ─── A/B TESTING SETTINGS ─────────────────────────────────────────────────────
# Column names in your data file for A/B testing (override as needed)
AB_GROUP_COL       = "Sex"         # column that contains group labels (A / B)
AB_METRIC_COL      = "Fare"        # continuous metric to compare
AB_CONVERSION_COL  = "Survived"    # binary (0/1) conversion metric

# ─── CAUSAL INFERENCE SETTINGS ────────────────────────────────────────────────
# DiD / IV / RDD parameters — these run on synthetic data by default.
# Change to real column names when working with your own observational dataset.
CAUSAL_USE_SYNTHETIC = True   # True = use built-in demo data; False = use DATA_PATH

# ─── BAYESIAN SETTINGS ────────────────────────────────────────────────────────
MCMC_ITERATIONS = 10_000
MCMC_BURNIN     = 2_000
MCMC_SEED       = 42
