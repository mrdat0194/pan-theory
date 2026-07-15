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
# Set CAUSAL_USE_SYNTHETIC = False and fill in the column names below to use
# your own observational dataset instead of the built-in synthetic demos.
CAUSAL_USE_SYNTHETIC = True   # True = use built-in demo data; False = use DATA_PATH

# DiD — Staggered Difference-in-Differences (Callaway & Sant'Anna ATT(g,t))
CAUSAL_DID_OUTCOME_COL = ""        # continuous outcome
CAUSAL_DID_UNIT_COL    = ""        # unit / entity identifier
CAUSAL_DID_TIME_COL    = ""        # integer time period
CAUSAL_DID_COHORT_COL  = ""        # first treatment period (NaN = never treated)
CAUSAL_DID_N_BOOTSTRAP = 999       # bootstrap draws for clustered SEs

# IV — Robust 2SLS (linearmodels) with Anderson-Rubin weak-instrument fallback
CAUSAL_IV_OUTCOME_COL     = ""     # dependent variable
CAUSAL_IV_TREATMENT_COL   = ""     # endogenous regressor
CAUSAL_IV_INSTRUMENT_COLS = []     # list of instrument column names
CAUSAL_IV_COVARIATE_COLS  = []     # exogenous controls (may be empty)

# RDD — Calonico-Cattaneo-Titiunik MSE-optimal bandwidth + rddensity manipulation test
CAUSAL_RDD_RUNNING_COL = ""        # running / forcing variable
CAUSAL_RDD_OUTCOME_COL = ""        # outcome variable
CAUSAL_RDD_CUTOFF      = 0.0       # assignment cutoff value
CAUSAL_RDD_FUZZY_COL   = None      # treatment column for fuzzy RDD; None = sharp

# ─── BAYESIAN SETTINGS ────────────────────────────────────────────────────────
MCMC_ITERATIONS = 10_000
MCMC_BURNIN     = 2_000
MCMC_SEED       = 42
