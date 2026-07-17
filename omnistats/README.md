# OmniStats — Unified Statistical Analysis Pipeline

A single, **100 % Python** pipeline that combines five statistical stages
into one orchestrated run, generating an APA 7th edition Word report.

| Stage | Purpose | Methods |
|---|---|---|
| **1 — LPA** | **DESCRIBE** — segment users | GMM, Welch ANOVA, Games-Howell, Chi-square, Cramér's V |
| **2 — A/B Testing** | **COMPARE** — measure effect size | Frequentist: z-test, Welch t-test; Bayesian Sequential: Beta-Binomial (PyMC / IS), StudentT (PyMC NUTS), Expected Loss |
| **3 — CUPED** | **SHARPEN** — reduce variance before causal attribution | Monotonic CatBoost / DT regression on LPA profile score |
| **4 — Causal** | **ATTRIBUTE** — explain *why* | Staggered DiD (C&S-A), IV/2SLS, RDD, SCM, Matrix Completion, BMA (HTE) |
| **5 — Time-Series + APA** | **PROJECT + CONSOLIDATE** | CausalImpact (BSTS + spike-and-slab); APA 7th edition Word document (Tables 1–8) |

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
│   ├── ab_testing.py       ← Frequentist: proportion z-test, Welch t-test, dist. fit
│   ├── bayesian/           ← Bayesian Sequential A/B subpackage [Stage 2]
│   │   ├── __init__.py     ←   run_bayesian_ab_tests() orchestrator
│   │   ├── beta_binomial.py←   Beta-Binomial conjugate + IS fallback
│   │   ├── normal_model.py ←   PyMC NUTS StudentT (IS fallback)
│   │   └── sequential.py   ←   SIR batch stopping rule + Expected Loss
│   ├── cuped.py            ← CUPED variance reduction (CatBoost monotonic) [Stage 3]
│   ├── causal/             ← Robust causal inference subpackage [Stage 4]
│   │   ├── __init__.py     ←   run_causal_suite() orchestrator
│   │   ├── did.py          ←   Staggered DiD (Callaway & Sant'Anna ATT(g,t))
│   │   ├── iv.py           ←   Robust IV/2SLS (linearmodels + Anderson-Rubin)
│   │   ├── rdd.py          ←   CCT optimal-bandwidth RDD (rdrobust + rddensity)
│   │   ├── scm.py          ←   Synthetic Control Method (cvxpy convex opt.)
│   │   └── matrix_completion.py ← Matrix Completion (SoftImpute / ALS)
│   ├── timeseries/         ← Bayesian time-series causal subpackage [Stage 5]
│   │   ├── __init__.py     ←   run_timeseries_suite() orchestrator
│   │   └── causal_impact.py←   CausalImpact BSTS + Prophet fallback
│   ├── visualisation.py    ← All plots (line, stacked bar, heatmap, mosaic)
│   └── apa_report.py       ← APA 7th edition .docx generator (Tables 1–8)
└── outputs/                ← All CSVs, PNGs, and .docx created here
```

---

## Relationship to `Bayesian/` — Migration Source

`C:\Users\mrdat\PycharmProjects\pan-theory\Bayesian\` is the **direct migration source** for all advanced methods. The mapping is:

| `Bayesian/` asset | Migrated to `omnistats/modules/` |
|---|---|
| `abtesting/abtesting_suite.py` | `bayesian/beta_binomial.py` (Beta-Binomial logic + simulation fixtures) |
| `importance_sampling_bayesian.py` | Archive/reference (unused fallback logic removed from `bayesian/beta_binomial.py`) |
| `mcmc_bayesian.py` | Archive/reference — role superseded by PyMC NUTS |
| `mono_casual.ipynb` | `cuped.py` (CatBoost monotonic regression → Stage 3 CUPED) |
| `prophet.ipynb` | `timeseries/causal_impact.py` (Prophet fallback; primary = CausalImpact BSTS in Stage 5) |
| `BMA.ipynb` | `causal/bma.py` — **Stage 4 migration target** for Heterogeneous Treatment Effects (HTE) / subgroup analysis via Bayesian Model Averaging. BMA models `Treatment × Demographic` interactions and outputs Posterior Inclusion Probabilities (PIPs). This is distinct from CUPED (which reduces variance); BMA *explains* who benefits from treatment. |

The `Bayesian/` directory remains **unchanged** as a reference archive.

---

## Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
# For CausalImpact (Stage 5 primary):
pip install tfcausalimpact
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

## Stage Explanations — Why Each Exists

These are five **distinct inferential modes** following a strict dependency chain.
The output of each stage is the direct input to the next.

```
Stage 1 (LPA)
  └─ outputs profile_prob_max
       │
       ▼
Stage 2 (A/B Testing)
  └─ outputs bayesian_ab_results (comparison: did B beat A?)
       │
       ▼
Stage 3 (CUPED) ← uses profile_prob_max from Stage 1
  └─ outputs df_cuped
       │
       ▼
Stage 4 (Causal Inference) ← operates on df_cuped
  └─ outputs causal_results.csv (DiD, IV, RDD, SCM, MC, BMA)
       │
       ▼
Stage 5 (Time-Series + APA Report)
  └─ outputs apa_report.docx (Tables 1–8)
```

### Stage 1 — Latent Profile Analysis `[DESCRIBE]`

Fits **Gaussian Mixture Models** for K = 1 … 6 profiles. Mathematically equivalent to LPA in Mplus / tidySEM.

After Step 2, inspect `outputs/lpa_fit_stats.csv`. Choose K where BIC stops decreasing AND entropy > 0.80.

**Key output feeding Stage 3:** `profile_prob_max` — the posterior probability of profile membership. This is the CUPED covariate in Stage 3.

### Stage 2 — A/B Testing `[COMPARE]`

Runs both frequentist and Bayesian tests in parallel.

**Frequentist (backward-compatible):**

| Test | When to use |
|---|---|
| `proportion_test()` | Binary conversion rates |
| `means_test()` | Continuous metrics |
| `distribution_fit_test()` | Verify normality |

**Bayesian Sequential — solving the peaking problem:**

Traditional A/B tests suffer from inflated Type-I error when p-values are checked before the pre-specified sample size is reached. The Bayesian framework solves this:

| Test | Method | Engine |
|---|---|---|
| `bayesian_proportion_test()` | Beta-Binomial conjugate | Analytic + IS fallback |
| `bayesian_means_test()` | StudentT likelihood | PyMC NUTS (auto-tuned MCMC) |
| `sequential_monitor()` | Batch SIR stopping rule | Sequential Importance Resampling |

**Decision rule:** Stop when P(B > A) ≥ 0.95 AND Expected Loss ≤ 0.01.

### Stage 3 — CUPED Variance Reduction `[SHARPEN]`

**CUPED** (Controlled-experiment Using Pre-Experiment Data) adjusts the outcome
before passing it to Stage 4 Causal Inference:

$$Y_i^{\text{adj}} = Y_i - \hat\theta (X_i - \bar X)$$

**Why `profile_prob_max` is the covariate** (not `AB_METRIC_COL`):
- `profile_prob_max` is measured *before* any treatment (pre-experiment)
- It correlates with the outcome (high-profile users have higher metrics)
- Using `AB_METRIC_COL` itself would be circular

**Why monotonic constraints:** The profile→outcome relationship is monotone by construction. `CatBoostRegressor(monotone_constraints=[+1])` enforces this, preventing overfitting of the hat matrix.

### Stage 4 — Causal Inference `[ATTRIBUTE]`

Five core estimators plus BMA for HTE, all operating on `df_cuped` from Stage 3:

| Estimator | Identification strategy | Estimand | Key assumption |
|---|---|---|---|
| **DiD** (Callaway & Sant'Anna) | Staggered parallel trends | ATT(g,t) | Parallel trends + no anticipation |
| **IV/2SLS** (linearmodels) | Exclusion restriction | LATE | Instrument relevance + exclusion |
| **RDD** (rdrobust CCT) | Continuity at cutoff | LATE at cutoff | Continuity of potential outcomes |
| **SCM** (cvxpy) | Convex donor matching | ATT treated unit | Pre-period fit quality |
| **Matrix Completion** (SoftImpute) | Missing data / nuclear norm | ATT staggered panel | Low-rank latent factor structure |
| **BMA** | Model averaging over subgroup interactions | HTE / PIP per subgroup | Prior on covariate inclusion |

**Why BMA belongs in Stage 4:**
BMA models `Treatment × Demographic` interaction effects across all plausible covariate structures and outputs **Posterior Inclusion Probabilities (PIPs)** — the probability that each subgroup has a true heterogeneous treatment effect.

### Stage 5 — Bayesian Time-Series Causal + APA Report `[PROJECT + CONSOLIDATE]`

**CausalImpact (BSTS)** vs. Prophet:

| | Prophet | CausalImpact (BSTS) |
|---|---|---|
| Counterfactual source | Historical trend of *treated* series only | Weighted blend of *untreated control* series |
| Shock handling | Cannot separate macro shocks from treatment | Spike-and-slab separates shared shocks |
| Explainability | Trend + seasonality components | Posterior Inclusion Probabilities per control series |

**Relationship to Stage 4 DiD:** CausalImpact is DiD generalised to continuous time. Instead of a binary pre/post comparison with parallel trends, BSTS models the full counterfactual trajectory.

The APA Report (`apa_report.docx`) is generated at the end of Stage 5, after all results are available, producing **Tables 1–8** in a single pass.

---

## Explainable AI (XAI) in OmniStats

This pipeline is designed as an **Explainable AI system** for causal inference. Every result is directly interpretable. No black boxes. Every number in the APA report has a mathematical interpretation expressible in plain language for non-statistician reviewers.

---

## Advanced Experimentation Roadmap

### 1. Bayesian Model Averaging (BMA)
*   **Asset:** `causal/bma.py`
*   **Advanced Tactic:** Subgroup Analysis & Heterogeneous Treatment Effects (HTE) under model uncertainty.

### 2. MCMC & Importance Sampling (Reference Engines)
*   **Role in OmniStats:** The mathematical foundations that PyMC NUTS (Stage 2) and IS fallbacks directly implement at a higher level.

### 3. CUPED via Monotonic Regression
*   **Asset:** `modules/cuped.py`
*   **Status:** Implemented in **Stage 3**.

---

## Future Roadmap — JEPA & Deep Learning Counterfactuals

> **Research direction — significant work required before integration.**

### Why JEPA Is Not in the Current Pipeline

Meta's **Joint Embedding Predictive Architecture (JEPA)** predicts causal dynamics in an *abstract latent space* rather than predicting the raw metric value. While this makes JEPA extremely efficient and generalisable, it is currently incompatible with an APA reporting pipeline for two reasons:

1. **No interpretable effect size:** JEPA predicts that a user's latent state vector shifts from `[0.2, -1.4]` to `[0.5, -0.9]`. This cannot be placed in an APA table — reviewers require dollar amounts, conversion rates, or other real-world quantities.
2. **No standard errors:** Translating a JEPA latent shift back into a real metric with a valid standard error is an unsolved, actively researched problem.

### How OmniStats Enables Future JEPA Integration

By producing **highly interpretable, mathematically proven causal counterfactuals** (SCM weights, BSTS posterior intervals, doubly-robust ATT, BMA PIPs), this pipeline generates **ground-truth training signal** for future latent-space causal models.

A future JEPA architecture for causal inference could be trained using OmniStats as the teacher: JEPA learns to align its abstract latent state transitions with the explainable causal effects calculated by the econometric estimators in Stage 4 and Stage 5. The pipeline is thus the necessary rigorous foundation before deploying opaque latent-space causal models.

### Reinforcement Learning Connection


| OmniStats component | RL equivalent |
|---|---|
| Bayesian Sequential A/B (Stage 2) | Thompson Sampling — Multi-Armed Bandit (Exploration vs. Exploitation) |
| Doubly Robust DiD / IV (Stage 4) | Off-Policy Evaluation (OPE) for safe agent policy comparison |
| CausalImpact BSTS / Kalman Filter (Stage 5) | Belief-state updating in POMDPs (Partially Observable MDPs) |

By building OmniStats, you are constructing the **critic (evaluator) network** for a future Reinforcement Learning agent that automatically allocates user traffic to optimal treatments.

---

## APA Report — Table Summary

| Table | Content |
|---|---|
| 1 | LPA model fit statistics (AIC, BIC, aBIC, Entropy, LMR-LRT p) |
| 2 | Profile indicator means (SD) + Welch ANOVA + η² |
| 3 | Chi-square tests + Cramér's V for demographic variables |
| 4 | Profile membership counts, percentages, mean max probability |
| 5 | Frequentist A/B test results (proportion, means, distribution fit) |
| 6 | Causal inference results — DiD, IV, RDD, SCM (weights), Matrix Completion |
| 7 | Time-Series Causal — CausalImpact BSTS lift, MCMC credible interval |
| 8 | Bayesian A/B — P(B>A), Expected Loss, ESS, R-hat, decision |

Formatting follows **APA 7th edition**: no vertical borders, three horizontal rules, Times New Roman 12pt.

---

## All Outputs

| File | Stage | Description |
|---|---|---|
| `prepared_data.csv` | 1 | Cleaned dataset with z-scored indicators |
| `lpa_fit_stats.csv` | 1 | Model fit for K = 1 … K_MAX |
| `lpa_profiles.csv` | 1 | Dataset with profile labels and posteriors |
| `anova_results.csv` | 1 | Welch ANOVA F, df, p, η² |
| `anova_posthoc.csv` | 1 | Games-Howell comparisons (t, p, Cohen's d) |
| `chi_square_results.csv` | 1 | χ², df, p, Cramér's V |
| `ab_test_results.csv` | 2 | Frequentist A/B summary |
| `bayesian_ab_results.csv` | 2 | P(B>A), Expected Loss, ESS, decision |
| `cuped_variance_reduction.csv` | 3 | θ̂, variance reduction %, backend |
| `causal_results.csv` | 4 | DiD / IV / RDD / SCM / MC standardised schema |
| `did_attgt.csv` | 4 | Full ATT(g,t) table |
| `iv_estimates.csv` | 4 | IV 2SLS estimate + diagnostics |
| `rdd_results.csv` | 4 | RDD estimate + CCT bandwidth |
| `scm_weights.csv` | 4 | SCM donor unit weights |
| `scm_gaps.csv` | 4 | SCM gap series (treated − synthetic) |
| `mc_gaps.csv` | 4 | Matrix Completion counterfactual gaps |
| `timeseries_causal_results.csv` | 5 | CausalImpact / BSTS lift + credible band |
| `ts_causalimpact.png` | 5 | CausalImpact summary plot |
| `ts_counterfactual.png` | 5 | Prophet fallback counterfactual plot |
| `ts_lift.csv` | 5 | Pointwise lift (observed − counterfactual) |
| `apa_report.docx` | 5 | Full APA 7th edition Word document (Tables 1–8) |

---

## Configuration Reference (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `DATA_PATH` | `../adHoc/titanic.csv` | Path to your dataset |
| `N_PROFILES` | `3` | Number of LPA profiles |
| `INDICATOR_COLS` | `["Age","Fare","SibSp","Parch"]` | Continuous vars for LPA |
| `AB_GROUP_COL` | `"Sex"` | A/B group column |
| `AB_METRIC_COL` | `"Fare"` | Continuous outcome metric |
| `AB_CONVERSION_COL` | `"Survived"` | Binary conversion metric |
| `CUPED_ENABLED` | `True` | Enable CUPED Stage 2.5 |
| `CUPED_COVARIATE_COL` | `"profile_prob_max"` | LPA Stage 1 posterior probability |
| `CUPED_USE_CATBOOST` | `True` | False = sklearn DT / OLS fallback |
| `BAYES_AB_THRESHOLD` | `0.95` | P(B>A) stopping threshold |
| `BAYES_AB_LOSS_THRESH` | `0.01` | Expected Loss stopping threshold |
| `BAYES_AB_N_SAMPLES` | `2000` | PyMC NUTS posterior draws |
| `CAUSAL_USE_SYNTHETIC` | `True` | Use built-in demo data for causal |
| `CAUSAL_SCM_ENABLED` | `True` | Include Synthetic Control in suite |
| `CAUSAL_MATRIX_COMP_ENABLED` | `True` | Include Matrix Completion in suite |
| `TS_CAUSALIMPACT_ENABLED` | `False` | Enable Stage 4 with real data |
| `TS_INTERVENTION_DATE` | `""` | `"YYYY-MM-DD"` treatment start |
| `TS_CONTROL_COLS` | `[]` | Control series for BSTS spike-and-slab |

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

# Causal Inference Stage 3 — core estimators
differences  >= 1.0.0    # Callaway & Sant'Anna staggered DiD
linearmodels >= 6.0.0    # IV2SLS with KP rk-F and robust SEs
ivmodels     >= 0.4.0    # Anderson-Rubin identification-robust CI
rdrobust     >= 1.2.0    # CCT optimal-bandwidth RDD
rddensity    >= 2.4.0    # McCrary/Cattaneo manipulation density test

# Causal Inference Stage 3 — advanced panel data
cvxpy        >= 1.3.0    # Synthetic Control Method convex optimisation
fancyimpute  >= 0.7.0    # Matrix Completion (SoftImpute)

# Bayesian A/B Testing Stage 2
pymc         >= 5.0.0    # NUTS sampler (primary Bayesian backend)
arviz        >= 0.17.0   # PyMC diagnostics (R-hat, ESS)

# CUPED Variance Reduction Stage 2.5
catboost     >= 1.2.0    # Monotonic regression (fallback: sklearn DT)

# Stage 4 Time-Series Causal
# Install one:
# pip install tfcausalimpact   # CausalImpact primary (recommended)
# pip install pycausalimpact   # Alternative port
prophet      >= 1.1.0    # Fallback if CausalImpact unavailable
```
