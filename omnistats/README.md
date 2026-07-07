# OmniStats — Unified Statistical Analysis Pipeline

A single, **100 % Python** pipeline that combines four statistical workflows
into one orchestrated run:

| Stage | Methods | Source |
|---|---|---|
| **1 — LPA** | Latent Profile Analysis (GMM), Welch ANOVA, Games-Howell, Chi-square, Cramér's V | `lpa_analysis/` |
| **2 — A/B Testing** | Proportion z-test, Welch t-test, Distribution fit (χ²) | `Bayesian/abtesting/` |
| **3 — Causal Inference** | Difference-in-Differences, IV/2SLS, Regression Discontinuity | `Bayesian/someMethod/` |
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
│   ├── causal_inference.py ← DiD, IV/2SLS, RDD estimators
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

By default runs on **synthetic demo data** (set `CAUSAL_USE_SYNTHETIC = True` in `config.py`) to demonstrate each estimator:

| Method | What it estimates | Key assumption |
|---|---|---|
| **DiD** | Average Treatment Effect (ATE) | Parallel trends pre-treatment |
| **IV/2SLS** | Local Average Treatment Effect (LATE) | Strong, exogenous instrument |
| **RDD** | Treatment effect at the cutoff | Continuity of potential outcomes |

To use your own data, set `CAUSAL_USE_SYNTHETIC = False` and configure the column names when calling each function directly.

### Stage 4 — APA Report

A single Word document (`outputs/apa_report.docx`) is created with:

| Table | Content |
|---|---|
| 1 | LPA model fit statistics (AIC, BIC, aBIC, Entropy, LMR-LRT p) |
| 2 | Profile indicator means (SD) + Welch ANOVA + η² |
| 3 | Chi-square tests + Cramér's V for demographic variables |
| 4 | Profile membership counts, percentages, mean max probability |
| 5 | A/B test results (proportion, means, distribution fit) |
| 6 | Causal inference results (DiD ATE, IV LATE, RDD estimate) |

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
| `causal_results.csv` | DiD / IV / RDD causal estimates |
| `profiles_lineplot.png` | LPA indicator means with 95% CI (publication quality) |
| `demographics_plot.png` | Stacked bar charts of demographics by profile |
| `posthoc_heatmap.png` | Games-Howell p-value heatmap (indicator × profile pair) |
| `chi_square_mosaic.png` | Tile chart of demographic categories by profile |
| `rdd_plot.png` | RDD scatter plot with local linear fits |
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
| `MCMC_ITERATIONS` | `10000` | MCMC iterations (Bayesian module, future) |

---

## Dependencies

```
scikit-learn >= 1.3.0
scipy        >= 1.11.0
pandas       >= 2.0.0
numpy        >= 1.24.0
matplotlib   >= 3.7.0
seaborn      >= 0.12.0
python-docx  >= 1.1.0
statsmodels  >= 0.14.0
```

---

## Relationship to Legacy Codebases

| Legacy folder | Role in OmniStats |
|---|---|
| `lpa_analysis/` | Stage 1 — fully migrated into `modules/lpa.py`, `anova.py`, `chi_square.py`, `visualisation.py`, `apa_report.py` |
| `Bayesian/abtesting/` | Stage 2 — core tests migrated into `modules/ab_testing.py` |
| `Bayesian/someMethod/` | Stage 3 — DiD, IV, RDD migrated into `modules/causal_inference.py` |
| `Bayesian/` (root) | Theory reference — MCMC / Bayesian module planned for future Stage 5 |

The legacy folders remain **unchanged** — OmniStats is a new, unified layer on top.
