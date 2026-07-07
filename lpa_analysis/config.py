"""
config.py — Set N_PROFILES after reviewing outputs/lpa_fit_stats.csv with your co-author.
"""
import os

# ─── USER SETTINGS ────────────────────────────────────────────────────────────
# Update this after reviewing lpa_fit_stats.csv
N_PROFILES = 3

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "adHoc", "titanic.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ─── INDICATOR & DEMOGRAPHIC COLUMNS ──────────────────────────────────────────
INDICATOR_COLS   = ["Age", "Fare", "SibSp", "Parch"]
DEMOGRAPHIC_COLS = ["Sex", "Pclass", "Embarked"]

# ─── LPA SEARCH RANGE ─────────────────────────────────────────────────────────
K_MIN = 1
K_MAX = 6
