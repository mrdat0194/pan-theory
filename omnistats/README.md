# OmniStats — Unified Statistical Analysis Pipeline

A single, **100 % Python** pipeline that combines four statistical workflows
into one orchestrated run:

| Stage | Methods | Source |
|---|---|---|
| **1 — LPA** | Latent Profile Analysis (GMM), Welch ANOVA, Games-Howell, Chi-square, Cramér's V | `lpa_analysis/` |
| **2 — A/B Testing** | Proportion z-test, Welch t-test, Distribution fit (χ²) | `Bayesian/abtesting/` |
| **3 — Causal Inference** | **Staggered DiD** (Callaway & Sant'Anna ATT(g,t)), **Robust IV/2SLS** (linearmodels + Anderson-Rubin), **RDD** (rdrobust CCT + rddensity) | `modules/causal/` |
| **4 — APA Report** | APA 7th edition Word document (6 tables) | combined |

---

## Directory Structure

```
omnistats/
├── config.py               ← Edit this file to configure all settings
├── data_manager.py         ← Centralised data loading & z-scoring
├── main.py                 ← Run this to execute the full pipeline
├── requirements.txt
├── modules/
│   ├── lpa.py              ← Gaussian Mixture Model (LPA) fitting
│   ├── anova.py            ← Welch ANOVA + Games-Howell post-hoc
│   ├── chi_square.py       ← Chi-square independence test + Cramér's V
│   ├── ab_testing.py       ← Proportion test, Welch t-test, dist. fit
│   ├── causal/             ← Robust causal inference subpackage
│   │   ├── __init__.py     ←   run_causal_suite() orchestrator
│   │   ├── did.py          ←   Staggered DiD  (Callaway & Sant'Anna ATT(g,t))
│   │   ├── iv.py           ←   Robust IV/2SLS (linearmodels + Anderson-Rubin)
│   │   └── rdd.py          ←   CCT optimal-bandwidth RDD (rdrobust + rddensity)
│   ├── visualisation.py    ← All plots (line, stacked bar, heatmap, mosaic)
│   └── apa_report.py       ← APA 7th edition .docx generator
└── outputs/                ← All CSVs, PNGs, and .docx created here
```

---

## Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Point to your data

Edit `config.py`:

```python
DATA_PATH        = r"path\to\your_data.csv"
INDICATOR_COLS   = ["col1", "col2", "col3"]   # continuous variables for LPA
DEMOGRAPHIC_COLS = ["sex", "group", "region"]  # categorical variables
AB_GROUP_COL     = "group"     # column with exactly 2 values (control / treatment)
AB_METRIC_COL    = "revenue"   # continuous metric for A/B comparison
AB_CONVERSION_COL= "converted" # binary 0/1 conversion column (or None)
```

### 3. Run the pipeline

```powershell
python -X utf8 main.py
```

---

## Workflow

### Stage 1 — Latent Profile Analysis

The pipeline fits **Gaussian Mixture Models** for K = 1 … 6 profiles (configurable via `K_MIN` / `K_MAX`) using diagonal covariance — mathematically equivalent to LPA in Mplus / tidySEM.

**After Step 2**, inspect `outputs/lpa_fit_stats.csv`:

| K | BIC | Entropy | LMR-LRT p | Decision |
|---|---|---|---|---|
| 1 | Highest | — | — | Baseline |
| 2 | Lower | 0.99 | < .001 | ✓ Better |
| **3** | **Lowest** | **0.96** | **< .001** | **✓ Selected** |

Choose the K where BIC stops decreasing significantly AND entropy > 0.80.
Then set `N_PROFILES = K` in `config.py` and re-run.

### Stage 2 — A/B Testing

Runs automatically on your real data using `AB_GROUP_COL` and `AB_METRIC_COL`.

| Test | When to use |
|---|---|
| `proportion_test()` | Binary conversion rates (click-through, sign-ups) |
| `means_test()` | Continuous metrics (revenue, time on page, age) |
| `distribution_fit_test()` | Verify normality assumption of each group |

### Stage 3 — Causal Inference

By default runs on **synthetic demo data** (`CAUSAL_USE_SYNTHETIC = True` in `config.py`). Each estimator uses a methodologically robust implementation replacing the legacy flat module:

#### Why the new implementation is better

| Issue in the old `causal_inference.py` | Fix in `modules/causal/` |
|---|---|
| **DiD** used a plain 2×2 OLS interaction (`treated × post`). Under *staggered adoption* (different units treated at different times) this TWFE estimator is provably biased — treated-early units act as a "contaminated control" for treated-late units. | Replaced with **Callaway & Sant'Anna (2021)** ATT(g,t): estimates one treatment effect per cohort-period pair, then aggregates. Doubly-robust (IPW + outcome regression), so consistent if *either* the propensity or outcome model is correct. |
| **IV** computed a raw numpy Wald ratio with homoscedastic OLS standard errors and no weak-instrument check, giving wrong SEs under heteroscedasticity and invalid inference when the instrument is weak. | Replaced with **`linearmodels` IV2SLS** (HC3 robust SEs). Reports the Kleibergen-Paap rk-F statistic. Automatically switches to an **Anderson-Rubin** identification-robust confidence interval when KP rk-F < 10 — valid even with a weak instrument. |
| **RDD** used an arbitrary fixed bandwidth (`bandwidth=20`) with no principled selection, no manipulation test, and no bias-corrected CI. The fixed bandwidth inflates MSE and the conventional CI ignores the smoothing bias. | Replaced with **`rdrobust`** (Calonico, Cattaneo & Titiunik) MSE-optimal data-driven bandwidth, bias-corrected robust CI, and **`rddensity`** McCrary manipulation density test. Supports both sharp and fuzzy designs. |


**Fallback behaviour:** all three estimators degrade gracefully if optional libraries are missing — they fall back to clean `statsmodels`/`numpy` implementations with clear `[WARNING]` messages and install hints.

**To use your own data:** set `CAUSAL_USE_SYNTHETIC = False` and fill in the column names in `config.py` under the `CAUSAL_DID_*`, `CAUSAL_IV_*`, and `CAUSAL_RDD_*` settings.

#### Standardised output schema

All three estimators return the same dict structure for easy downstream comparison:

```python
{
    "method":    str,    # estimator name
    "estimand":  str,    # "ATT(g,t)" | "LATE" | "LATE_at_cutoff"
    "estimate":  float,
    "se":        float,
    "ci_lower":  float,
    "ci_upper":  float,
    "ci_type":   str,    # "doubly_robust" | "anderson_rubin" | "robust_bc" | fallback
    "p_value":   float,
    "n_obs":     int,
    "diagnostics": dict, # pre-trend p, KP rk-F, CCT bandwidth, manipulation p …
    "warnings":  list,
}
```

### Stage 4 — APA Report

A single Word document (`outputs/apa_report.docx`) is created with:

| Table | Content |
|---|---|
| 1 | LPA model fit statistics (AIC, BIC, aBIC, Entropy, LMR-LRT p) |
| 2 | Profile indicator means (SD) + Welch ANOVA + η² |
| 3 | Chi-square tests + Cramér's V for demographic variables |
| 4 | Profile membership counts, percentages, mean max probability |
| 5 | A/B test results (proportion, means, distribution fit) |
| 6 | Causal inference results — method, estimand, estimate, SE, 95% CI, CI type, p, N |

Table 6 CI types: `doubly_robust` (DiD bootstrap), `anderson_rubin` (IV weak-instrument), `robust_bc` (RDD bias-corrected).

Formatting follows **APA 7th edition**: no vertical borders, three horizontal rules, Times New Roman 12pt.

---

## All Outputs

| File | Description |
|---|---|
| `prepared_data.csv` | Cleaned dataset with z-scored indicator columns |
| `lpa_fit_stats.csv` | Model fit statistics for K = 1 … K_MAX |
| `lpa_profiles.csv` | Dataset with profile labels and posterior probabilities |
| `anova_results.csv` | Welch ANOVA F, df, p, η² per indicator |
| `anova_posthoc.csv` | Games-Howell pairwise comparisons (t, p, Cohen's d) |
| `chi_square_results.csv` | χ², df, p, Cramér's V per demographic |
| `chi_square_tables.csv` | Observed frequency crosstabs (long format) |
| `ab_test_results.csv` | A/B test results summary |
| `causal_results.csv` | Standardised causal results (method, estimand, estimate, SE, CI, p, N) |
| `did_attgt.csv` | Full ATT(g,t) table from Callaway & Sant'Anna |
| `iv_estimates.csv` | IV 2SLS point estimate + diagnostics |
| `rdd_results.csv` | RDD estimate + CCT bandwidth + manipulation p |
| `profiles_lineplot.png` | LPA indicator means with 95% CI (publication quality) |
| `demographics_plot.png` | Stacked bar charts of demographics by profile |
| `posthoc_heatmap.png` | Games-Howell p-value heatmap (indicator × profile pair) |
| `chi_square_mosaic.png` | Tile chart of demographic categories by profile |
| `did_event_study.png` | Pre/post event-study plot (Callaway & Sant'Anna) |
| `rdd_plot.png` | RDD scatter plot with local polynomial fits |
| `rdd_density.png` | Running variable density test for manipulation (rddensity) |
| `dist_fit_*.png` | Distribution fit plots for each A/B group |
| `apa_report.docx` | Full APA 7th edition Word document (6 tables) |

---

## Configuration Reference (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `DATA_PATH` | `../adHoc/titanic.csv` | Path to your dataset |
| `N_PROFILES` | `3` | Number of LPA profiles — **review lpa_fit_stats.csv first** |
| `K_MIN` / `K_MAX` | `1` / `6` | Range of K values to fit |
| `INDICATOR_COLS` | `["Age", "Fare", "SibSp", "Parch"]` | Continuous variables for LPA |
| `DEMOGRAPHIC_COLS` | `["Sex", "Pclass", "Embarked"]` | Categorical variables for chi-square |
| `AB_GROUP_COL` | `"Sex"` | Column with A/B group labels |
| `AB_METRIC_COL` | `"Fare"` | Continuous metric for A/B test |
| `AB_CONVERSION_COL` | `"Survived"` | Binary conversion metric |
| `CAUSAL_USE_SYNTHETIC` | `True` | Use synthetic demo data for causal methods |
| `CAUSAL_DID_OUTCOME_COL` | `""` | DiD outcome column |
| `CAUSAL_DID_UNIT_COL` | `""` | DiD unit/entity identifier |
| `CAUSAL_DID_TIME_COL` | `""` | DiD integer time period |
| `CAUSAL_DID_COHORT_COL` | `""` | First treatment period (NaN = never treated) |
| `CAUSAL_IV_OUTCOME_COL` | `""` | IV dependent variable |
| `CAUSAL_IV_TREATMENT_COL` | `""` | IV endogenous regressor |
| `CAUSAL_IV_INSTRUMENT_COLS` | `[]` | IV excluded instruments (list) |
| `CAUSAL_RDD_RUNNING_COL` | `""` | RDD running/forcing variable |
| `CAUSAL_RDD_OUTCOME_COL` | `""` | RDD outcome variable |
| `CAUSAL_RDD_CUTOFF` | `0.0` | RDD assignment threshold |
| `CAUSAL_RDD_FUZZY_COL` | `None` | Fuzzy RDD treatment column (None = sharp) |
| `MCMC_ITERATIONS` | `10000` | MCMC iterations (Bayesian module, future) |

---

## Dependencies

```
# Core
scikit-learn >= 1.3.0
scipy        >= 1.11.0
pandas       >= 2.0.0
numpy        >= 1.24.0
matplotlib   >= 3.7.0
seaborn      >= 0.12.0
python-docx  >= 1.1.0
statsmodels  >= 0.14.0

# Causal inference (Stage 3 — robust estimators)
differences  >= 1.0.0    # Callaway & Sant'Anna staggered DiD
linearmodels >= 6.0.0    # IV2SLS with KP rk-F and robust SEs
ivmodels     >= 0.4.0    # Anderson-Rubin identification-robust CI (IV fallback)
rdrobust     >= 1.2.0    # CCT optimal-bandwidth RDD
rddensity    >= 2.4.0    # McCrary/Cattaneo manipulation density test
```

---

## Relationship to Legacy Codebases

| Legacy folder | Role in OmniStats |
|---|---|
| `lpa_analysis/` | Stage 1 — fully migrated into `modules/lpa.py`, `anova.py`, `chi_square.py`, `visualisation.py`, `apa_report.py` |
| `Bayesian/abtesting/` | Stage 2 — core tests migrated into `modules/ab_testing.py` |
| `Bayesian/someMethod/` | Stage 3 — DiD, IV, RDD reimplemented as `modules/causal/` (Callaway & Sant'Anna, linearmodels, rdrobust) |
| `Bayesian/` (root) | Theory reference — MCMC / Bayesian module planned for future Stage 5 |

The legacy folders remain **unchanged** — OmniStats is a new, unified layer on top.

---

## Advanced Experimentation Roadmap & Bayesian Integration

To move the current experimentation framework (`modules/ab_testing.py`) from basic statistical tests (Welch's t-test and two-proportion z-tests) toward advanced industry standards (Statsig / Netflix / Stats-tech), the local files in the `Bayesian/` directory serve as a direct integration roadmap:

### 1. Bayesian Model Averaging (BMA)
*   **Asset:** [BMA.ipynb](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/BMA.ipynb)
*   **Advanced Tactic:** Subgroup Analysis & Heterogeneous Treatment Effects (HTE) under uncertainty.
*   **Integration:** Instead of running isolated sub-group tests (which inflates the false-positive rate), integrate the BMA class to model interaction effects (e.g., `Treatment * Demographic`). BMA averages over all plausible covariate structures to output **Posterior Inclusion Probabilities (PIPs)** representing the exact probability that a subgroup response is a true treatment effect.

### 2. MCMC & Importance Sampling
*   **Assets:** [mcmc_bayesian.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/mcmc_bayesian.py) and [importance_sampling_bayesian.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/importance_sampling_bayesian.py)
*   **Advanced Tactic:** Sequential Bayesian A/B Testing (continuous monitoring without peaking inflation).
*   **Integration:** Traditional A/B testing suffers from the "peaking problem" (inflated error rates when checking p-values early). Integrate the Importance Sampling and Metropolis-Hastings MCMC engines to numerically calculate the posterior probability $P(\text{Treatment} > \text{Control} \mid \text{Data})$ dynamically as data streams in. This allows tests to be safely terminated early once a posterior threshold (e.g., 95% probability of improvement) is reached.

### 3. Non-Linear Variance Reduction (CUPED)
*   **Asset:** [mono_casual.ipynb](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/mono_casual.ipynb)
*   **Advanced Tactic:** Covariate-Adjusted Experimentation (variance reduction via machine learning).
*   **Integration:** CUPED reduces variance by regressing out pre-experiment covariates. For complex, non-linear but monotonic relationships (e.g., historical user engagement), integrate the monotonic constraints (e.g. `CatBoostRegressor` or `DecisionTreeRegressor` with `monotone_constraints`) from `mono_casual.ipynb` to predict and adjust post-experiment metrics. This significantly lowers required sample sizes and speeds up test convergence.

### 4. Bayesian Counterfactual Time-Series
*   **Asset:** [prophet.ipynb](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/prophet.ipynb)
*   **Advanced Tactic:** Switchback Experiments & Geo-Testing.
*   **Integration:** When network effects prevent 50/50 allocation (e.g., matching algorithms or platform pricing), users are exposed to time-blocked treatments (switchbacks). Integrate `Prophet` to model the historical time-series baseline. This generates the counterfactual control prediction for treatment windows, allowing evaluation of treatment lift where concurrent controls are impossible.
