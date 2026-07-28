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

# ─── CUPED SETTINGS ──────────────────────────────────────────────────────────
# Stage 2.5: variance reduction using LPA profile score as pre-experiment covariate.
# profile_prob_max is the posterior probability from Stage 1 LPA — pre-experiment
# and correlated with outcome; using AB_METRIC_COL itself would be circular.
CUPED_ENABLED        = True
CUPED_COVARIATE_COL  = "profile_prob_max"  # LPA Stage 1 posterior probability output
CUPED_MONOTONE_DIR   = 1                   # +1 non-decreasing, -1 non-increasing
CUPED_USE_CATBOOST   = True                # False = sklearn DecisionTree / OLS fallback

# ─── BAYESIAN A/B SETTINGS ────────────────────────────────────────────────────
# Stage 2: Sequential Bayesian A/B testing.
# Primary: PyMC NUTS (No-U-Turn Sampler).
BAYES_AB_PRIOR_ALPHA  = 1.0       # Beta prior α (conversion tests; 1.0 = uniform)
BAYES_AB_PRIOR_BETA   = 1.0       # Beta prior β
BAYES_AB_THRESHOLD    = 0.95      # P(B > A) threshold to declare a winner
BAYES_AB_LOSS_THRESH  = 0.01      # Expected loss threshold (1% of baseline)
BAYES_AB_N_SAMPLES    = 2_000     # PyMC posterior draws (NUTS)
BAYES_AB_TUNE         = 1_000     # PyMC tuning draws (discarded)
BAYES_AB_SEED         = 42

# ─── CAUSAL STAGE 3 EXTENSIONS ────────────────────────────────────────────────
# Synthetic Control Method and Matrix Completion extend the causal suite.
CAUSAL_SCM_ENABLED         = True
CAUSAL_MATRIX_COMP_ENABLED = True

# ─── TIME-SERIES CAUSAL SETTINGS (Stage 4) ────────────────────────────────────
# CausalImpact: Bayesian Structural Time Series with control series forecasting.
# Primary: Pyro BSTS (on PyTorch).
TS_CAUSALIMPACT_ENABLED  = False        # Set True with real intervention data
TS_DATE_COL              = ""           # Date column (parseable by pd.to_datetime)
TS_METRIC_COL            = ""           # Outcome metric column
TS_INTERVENTION_DATE     = ""           # "YYYY-MM-DD" — treatment start date
TS_CONTROL_COLS          = []           # Control series for BSTS spike-and-slab
TS_SEASONALITY_MODE      = "multiplicative"  # "additive" for roughly-constant amplitude

# ─── EXPERIMENTAL DESIGN (Phase I — SOTA CAR) ────────────────────────────────
# experiment_design.py uses these settings for power analysis and SOTA
# Covariate-Adaptive Randomization (Mahalanobis-distance minimization).
DESIGN_MDE_RELATIVE      = 0.05     # Minimum Detectable Effect (relative, e.g. 0.05 = 5%)
DESIGN_POWER             = 0.80     # Statistical power (1 - Type II error rate)
DESIGN_ALPHA             = 0.05     # Significance level (two-sided)
DESIGN_STRATIFY_COLS     = ["Sex", "Pclass"]   # Covariates to balance via CAR
DESIGN_N_SIMULATIONS     = 1_000   # Bootstrap replications for imbalance simulation check

# ─── HTE SUBGROUP ANALYSIS (Stage 4 — DR-OLS replaces BMA stub) ──────────────
# bma.py implements Doubly Robust Interaction OLS for demographic HTE.
# Outputs bma_subgroups.csv (per-subgroup ATTs) and feeds causal_results.csv.
CAUSAL_BMA_ENABLED       = True    # Enable DR-OLS HTE in Stage 4 causal suite
CAUSAL_BMA_MAX_DUMMIES   = 10      # Maximum dummy variables before pruning
