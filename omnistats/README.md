# OmniStats — Unified 4-Stage Statistical & AI Pipeline

A concise **100% Python** 4-stage experimental statistics and world model planning architecture:

---

## 4-Stage Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Pre-Experiment Design  (experiment_design.py)            │
│  Power Analysis, LPA baseline, Covariate-Adaptive Randomization     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  Share randomization schedule with Engineering
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Execution  (Outside OmniStats / Engineering)             │
│  Run live A/B test / field trial on traffic (intentionally empty)   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  Collect experiment data
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Post-Experiment Evaluation  (main.py)                    │
│  Diagnostics, LPA, A/B Testing, CUPED, Causal Suite & APA Report    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  Pass statistical outputs to World Model
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — World Model Planning  (plan_experiment.py)               │
│  JEPA World Model + CEM/MPPI Planner optimizes next experiment plan │
└─────────────────────────────────────────────────────────────────────┘
```

### At a Glance

| Stage | Entry Point | Purpose | Key Output |
|---|---|---|---|
| **Stage 1 — Design** | `experiment_design.py` | Design experiment & balance covariates | `randomization_schedule.csv` |
| **Stage 2 — Execution** | *(Outside OmniStats)* | Run live A/B test / field trial | Raw experiment dataset |
| **Stage 3 — Evaluation** | `main.py` | Run diagnostics, A/B tests, CUPED & Causal Suite | `apa_report.docx` (Tables 1–8) |
| **Stage 4 — Planning** | `plan_experiment.py` | JEPA World Model plans next experiment | `jepa_experiment_plan.csv` |

---

## Directory Structure

```
omnistats/
├── config.py               ← Edit settings and parameters
├── data_manager.py         ← Centralised data loading & preparation
├── experiment_design.py    ← Stage 1: Power analysis + CAR randomization schedule
├── main.py                 ← Stage 3: Post-experiment evaluation & APA report
├── plan_experiment.py      ← Stage 4: JEPA World Model experiment planner
├── requirements.txt
├── modules/
│   ├── diagnostics.py      ← Pre-flight diagnostics (MMD, SVD, Rank, SRM)
│   ├── lpa.py              ← Gaussian Mixture Model (LPA) fitting
│   ├── anova.py            ← Welch ANOVA + Games-Howell post-hoc
│   ├── chi_square.py       ← Chi-square test + Cramér's V
│   ├── ab_testing.py       ← Frequentist z-test & Welch t-test
│   ├── bayesian/           ← Bayesian Sequential A/B (PyMC NUTS, Expected Loss)
│   ├── cuped.py            ← CUPED variance reduction (CatBoost monotonic)
│   ├── causal/             ← Robust Causal Suite (DiD, IV, RDD, SCM, MC, HTE)
│   ├── timeseries/         ← Time-series causal (CausalImpact BSTS)
│   ├── jepa_bridge.py      ← OmniStats ↔ JEPA state encoder & decoder
│   ├── visualisation.py    ← Visualization plots
│   └── apa_report.py       ← APA 7th edition Word report generator
└── outputs/                ← Saved CSVs, PNGs, and .docx report
```
    ├── ...                 ← (existing outputs from Phases I–III)
    ├── jepa_experiment_plan.csv  ← Phase IV: optimal continuous experiment plan
    └── jepa_planning_losses.csv  ← Phase IV: CEM/MPPI cost convergence curve
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
| `prophet.ipynb` | `timeseries/causal_impact.py` (Archive reference; superseded by Pyro BSTS CausalImpact in Stage 5) |
| `BMA.ipynb` | `causal/bma.py` — **Stage 4 migration target** for Heterogeneous Treatment Effects (HTE) / subgroup analysis via Bayesian Model Averaging. BMA models `Treatment × Demographic` interactions and outputs Posterior Inclusion Probabilities (PIPs). This is distinct from CUPED (which reduces variance); BMA *explains* who benefits from treatment. |

The `Bayesian/` directory remains **unchanged** as a reference archive.

---

## Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
# For CausalImpact (Stage 5 primary):
pip install pyro-ppl
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

# Phase I — Experimental Design
DESIGN_MDE_RELATIVE  = 0.05          # 5% Minimum Detectable Effect
DESIGN_STRATIFY_COLS = ["sex", "region"]  # covariates to balance via CAR
```

### 3. Run Pipeline via `main.py`

Run any stage or the entire pipeline using `main.py`:

```powershell
# Stage 1 — Pre-Experiment Design (Power analysis & CAR randomization)
python main.py --mode design

# Stage 3 — Post-Experiment Evaluation (Diagnostics, LPA, A/B, CUPED, Causal Suite, APA Report) [DEFAULT]
python main.py --mode eval

# Stage 4 — World Model Planning (JEPA experiment design search)
python main.py --mode plan --planner cem --epochs 50

# Run all stages end-to-end (Stage 1 -> Stage 3 -> Stage 4)
python main.py --mode all
```

---

## 4-Stage Pipeline Workflow

```
Stage 1: Pre-Experiment Design  (experiment_design.py / main.py --mode design)
  └─ outputs randomization_schedule.csv → Share with Engineering
       │
       ▼
Stage 2: Execution (Outside OmniStats / Field Trial by Engineering)
  └─ Engineering routes traffic and collects experimental dataset
       │
       ▼
Stage 3: Post-Experiment Evaluation (main.py / main.py --mode eval)
  └─ Diagnostics → LPA → A/B Testing → CUPED → Causal Suite → apa_report.docx
       │
       ▼
Stage 4: World Model Planning (plan_experiment.py / main.py --mode plan)
  └─ JEPA Bridge → APADecoder → CEM/MPPI Planner → jepa_experiment_plan.csv
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

### Stage 4 — Causal Inference Suite `[ATTRIBUTE]`

Six core estimators plus DR-OLS for HTE, all operating on `df_cuped` from Stage 3:

| Estimator | Identification strategy | Estimand | Key assumption |
|---|---|---|---|
| **DiD** (Callaway & Sant'Anna) | Staggered parallel trends | ATT(g,t) | Parallel trends + no anticipation |
| **IV/2SLS** (linearmodels) | Exclusion restriction | LATE | Instrument relevance + exclusion |
| **RDD** (rdrobust CCT) | Continuity at cutoff | LATE at cutoff | Continuity of potential outcomes |
| **SCM** (cvxpy) | Convex donor matching | ATT treated unit | Pre-period fit quality |
| **Matrix Completion** (SoftImpute) | Missing data / nuclear norm | ATT staggered panel | Low-rank latent factor structure |
| **CausalImpact** (Pyro BSTS) | State-space Bayesian time-series | Continuous ATT | Unobserved control trajectory |
| **DR-OLS Subgroups** | Doubly robust interaction regression | Subgroup HTE / PIP | Correct propensity or outcome specification |

**CausalImpact (BSTS) Integration:**
CausalImpact is DiD generalized to continuous time. Instead of a binary pre/post comparison with parallel trends, BSTS models the full counterfactual trajectory. It runs within Stage 4 alongside all other causal estimators, saving results to `causal_results.csv`.

### Report Generation — APA 7th Edition `[CONSOLIDATE]`

The APA Report (`apa_report.docx`) is generated after all statistical inferential outputs (Stages 1–4) are completed, consolidating results into Tables 1–8 in a single pass.

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

## Future Roadmap — World Models, JEPA & Beyond RL

> **"A machine cannot be said to be intelligent if it cannot predict the consequences
> of its actions."** — Yann LeCun, *A Path Towards Autonomous Machine Intelligence* (2022)

### 🧠 The Big Idea: JEPA vs. Generative World Models (PPUU)

Traditional Reinforcement Learning learns by **trial and error** — taking millions of actions on real traffic. This is dangerous and slow.

LeCun's **World Model** approach flips this: learn a simulator, then *imagine* the outcomes of actions to plan safely. However, not all world models are the same:

| Feature | Generative World Models (e.g., `pytorch-PPUU`) | JEPA (`eb_jepa` / `le-wm`) |
|---|---|---|
| **Prediction Space** | **Observation Space** (reconstructs raw pixels/metrics via VAEs/Decoders). | **Abstract Latent Space** (joint embedding, no reconstruction needed). |
| **Uncertainty** | Explicit generative sampling (VAE prior/posterior). | **Energy-Based / Anti-collapse Loss** (e.g., VICReg). |
| **Planning Mechanism**| Optimizes actions through a heavy generative decoder. | Plans directly in latent space using fast MPC (CEM/MPPI). |
| **Noise Sensitivity** | Wastes capacity predicting task-irrelevant noise. | Filters out noise, retaining only features necessary for state prediction. |

**JEPA (Joint Embedding Predictive Architecture)** is far more sample-efficient and generalizable because it avoids pixel-perfect reconstruction.

### Integrating OmniStats APA Knowledge into JEPA Planning

How do we bridge OmniStats' rigorous econometrics with JEPA's abstract latent planning? 

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                      OmniStats APA Knowledge (The Ground Truth)         │
 │  • Stage 1: LPA Profiles (User Segments)                                │
 │  • Stage 2: Bayesian Loss / P(B>A) (Risk limits)                        │
 │  • Stage 3: CUPED (Noise-reduced baselines)                             │
 │  • Stage 4: Causal ATT & DR-OLS Subgroups (True Lift & Fairness)        │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                    JEPA Latent Planning Engine                          │
 │                                                                         │
 │  1. State Encoder (z_t): Conditioned on LPA Profile Posterior & CUPED.  │
 │  2. Predictor (P_psi): Unrolls experiment actions in latent space.      │
 │  3. APA Objective: Score trajectories based on ATT Lift, Bayesian Risk, │
 │                    and Subgroup Fairness.                               │
 └─────────────────────────────────────────────────────────────────────────┘
```

#### The Four-Step Integration:
1. **State Space ($z_t$)**: The JEPA Encoder ingests Stage 1 LPA Profile Probabilities and Stage 3 CUPED-adjusted baselines to strip historical noise.
2. **Action Space ($a_t$)**: Interventions are experiment designs (treatment allocation $T$, sample size $n$, subgroup filters).
3. **APA Planning Objective**: In `planning.py`, CEM/MPPI scores unrolled paths:
   $$Cost = -\mathbb{E}[\text{ATT}] + \lambda_1 (\text{Expected Loss}) + \lambda_2 (\text{Subgroup Disparity})$$
4. **Latent-to-APA Decoder**: A trained linear probe maps the latent state back to readable APA formats (Effect Size $\pm$ SE).

---

### Why JEPA Is Not in the Current Pipeline (Yet)

JEPA today predicts that a user's latent state shifts from `z = [0.2, -1.4]` to
`z' = [0.5, -0.9]` after treatment.  There are two open problems:

1. **No interpretable effect size** — an APA table requires "revenue increased by
   $2.30 ± 0.45" — not a latent vector delta.
2. **No valid standard errors** — translating a latent shift into a metric with a
   calibrated confidence interval remains an active research problem (see
   *Conformal Prediction* and *Latent-Space Calibration*).

### The Integration Vision: OmniStats → World Model → Autonomous Experimentation

The key insight is that OmniStats and JEPA are **complementary, not competing**:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TODAY: OmniStats generates interpretable causal ground truth           │
│                                                                        │
│  experiment_design.py  →  [Run Experiment]  →  main.py  →  APA Report  │
│  (power, CAR)              (real traffic)      (DiD, IV,    (Tables 1-8)│
│                                                 RDD, SCM,              │
│                                                 HTE, BSTS)             │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
              OmniStats outputs become TRAINING SIGNAL
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TOMORROW: JEPA World Model learns from OmniStats ground truth         │
│                                                                        │
│  Encoder              Predictor             Decoder (new)              │
│  ┌─────────┐          ┌──────────┐          ┌──────────────┐           │
│  │ User    │  action  │ Predict  │  decode  │ Map latent Δ │           │
│  │ context │ ───────► │ next z'  │ ───────► │ → ΔATT, ΔSE  │           │
│  │ → z     │          │ in latent│          │ (calibrated) │           │
│  └─────────┘          └──────────┘          └──────────────┘           │
│                                                                        │
│  Training loss = ||decoded_ATT − OmniStats_ATT||² + calibration_loss   │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
              World Model enables AUTONOMOUS PLANNING
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FUTURE: Autonomous Experimentation Agent                              │
│                                                                        │
│  1. World Model imagines 1000 possible experiments (zero real traffic)  │
│  2. Picks the experiment with highest expected ATT / lowest risk        │
│  3. Validates with OmniStats on real data (human-in-the-loop)          │
│  4. Feeds real results back → World Model improves                     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Concrete Integration Paths

#### Path A: OmniStats as Teacher → JEPA as Student (Knowledge Distillation)

OmniStats' causal estimates become **supervised labels** for training a JEPA world model:

| OmniStats Output | What JEPA Learns From It |
|---|---|
| `causal_results.csv` (ATT, SE, CI per method) | Ground-truth treatment effects → JEPA decoder targets |
| `bma_subgroups.csv` (per-demographic ATT) | Who benefits from treatment → JEPA learns heterogeneous dynamics |
| `lpa_profiles.csv` (profile assignments) | User archetypes → JEPA encoder initialisation / pre-training |
| `cuped_variance_reduction.csv` (θ, % reduction) | Noise structure → JEPA regularizer calibration |
| `randomization_schedule.csv` (CAR assignments) | Balanced experiment design → JEPA training data quality |
| `power_analysis.csv` (n, power curve) | Sample efficiency targets → JEPA evaluation benchmark |

#### Path B: JEPA as Counterfactual Generator → OmniStats as Validator

Once the world model is trained, it generates **synthetic counterfactuals** that
OmniStats validates against real experiments:

1. JEPA imagines: *"If we apply Treatment B to Profile 2 users, revenue shifts by Δz"*
2. A learned **decoder head** maps Δz → predicted ΔATT = +$1.80 ± $0.35
3. When the real experiment runs, OmniStats computes the true ATT via DiD/IV/RDD
4. **Calibration loss** = |predicted_ATT − true_ATT| feeds back to the decoder

#### Path C: Energy-Based Planning (EB-JEPA → Experiment Design) ✅ **Implemented**

`eb_jepa/` energy-based planning now drives OmniStats experiment design via
`plan_experiment.py`. The Two Rooms environment is replaced with the **APA
experiment design space**:

| EB-JEPA Component | Implementation in OmniStats Phase IV |
|---|---|
| `planning.py` → `APACausalMPCObjective` | Scores action trajectories on ATT Lift, Bayesian Risk & Subgroup Disparity |
| `TabularEncoder` + `TabularPredictor` | Encode user context (LPA + CUPED) → predict outcome in latent space |
| `APADecoder` (joint training) | Map latent Δz → predicted ΔATT ± SE (calibrated against causal_results.csv) |
| `CEMPlanner` / `MPPIPlanner` | Sample 100–500 continuous experiment designs, keep elite top-10%, converge |
| Energy function (APA Cost) | $-w_{att}\cdot\text{ATT}+w_{risk}\cdot\text{Risk}+w_{disp}\cdot\text{Disparity}$ |

### Why This Is Better Than Pure RL

| Dimension | Pure RL Agent | World Model + OmniStats |
|---|---|---|
| **Sample cost** | Millions of real user interactions | Imagines outcomes; validates on small real experiments |
| **Safety** | Deploys bad treatments to discover they're bad | Predicts bad outcomes *before* deployment |
| **Interpretability** | "Policy says do X" (black box) | "ATT = +$2.30 for Profile 2 because DiD estimates show..." |
| **Regulatory** | Cannot explain decisions (GDPR Art. 22 risk) | Full APA audit trail with standard errors and CIs |
| **Speed to insight** | Weeks of exploration | Seconds of imagination + 1 validation experiment |

### Phased Roadmap

All implementation phases (Phases I through IV) are now fully complete, integrated, and verified. Only Phase V (Autonomous Experimentation Loop) remains planned for future development.

```
Phase I  ✅  DONE — OmniStats generates interpretable causal ground truth
  │         experiment_design.py  →  main.py  →  APA Report (Tables 1–8)
  │         Outputs: randomization_schedule.csv, causal_results.csv, apa_report.docx
  │
Phase II ✅  DONE — APADecoder: latent z → (ATT_hat, Risk_hat)
  │         modules/jepa_bridge.py: APADecoder MLP + train_apa_decoder()
  │         Joint training forces JEPA latent space to align with causal effects.
  │         Success criterion met: decoder loss converges (L_ATT < 20 after 30 epochs).
  │
Phase III ✅  DONE — Counterfactual State Context
  │         modules/jepa_bridge.py: load_state_context() encodes LPA profiles,
  │         CUPED baselines, Bayesian posteriors, and historical ATT into a
  │         single latent state tensor [1, D, 1, 1, 1] for the world model.
  │
Phase IV ✅  DONE — Energy-Based Experiment Planning
  │         plan_experiment.py: TabularJEPA + APACausalMPCObjective
  │         CEM/MPPI planner searches the continuous experiment design space:
  │           Action = [treatment_fraction, segment_focus, sample_size, horizon]
  │           Cost   = -w_att·ATT_hat + w_risk·Risk_hat + w_disp·Disparity_hat
  │         Outputs: jepa_experiment_plan.csv, jepa_planning_losses.csv
  │
Phase V  🔲  Autonomous Experimentation Loop (Future)
            World Model proposes → OmniStats validates → human approves
            → real experiment runs → results feed back → World Model improves.
            Human remains in the loop for safety and regulatory compliance.
```

### Reinforcement Learning Connection (OmniStats as Critic Network)

Even though World Models reduce the need for pure RL, the RL interpretation remains
useful — OmniStats functions as the **critic (evaluator) network** in an actor-critic
architecture:

| OmniStats Component | RL / World Model Equivalent |
|---|---|
| Bayesian Sequential A/B (Stage 2) | Thompson Sampling — Multi-Armed Bandit (Exploration vs. Exploitation) |
| LPA Profile Segmentation (Stage 1) | State representation learning — JEPA encoder pre-training |
| CUPED Variance Reduction (Stage 3) | Advantage normalisation — reducing variance of the reward signal |
| Doubly Robust DiD / IV (Stage 4) | Off-Policy Evaluation (OPE) for safe agent policy comparison |
| DR-OLS Subgroup HTE (Stage 4) | Heterogeneous reward modelling — different users get different reward functions |
| CausalImpact BSTS (Stage 5) | Belief-state updating in POMDPs (Partially Observable MDPs) |
| APA Report (Stage 5) | Evaluation report card — the "training log" of the critic |
| JEPA World Model (Future) | The *actor* that proposes experiments; OmniStats is the *critic* that evaluates them |

By building OmniStats, you are constructing the **rigorous statistical foundation**
that any future World Model — JEPA, Genie, IRIS, Dreamer, or whatever comes next —
will need as its ground-truth training signal and safety validator.

---

## APA Report — Table Summary

| Table | Content |
|---|---|
| 1 | LPA model fit statistics (AIC, BIC, aBIC, Entropy, LMR-LRT p) |
| 2 | Profile indicator means (SD) + Welch ANOVA + η² |
| 3 | Chi-Square tests + Cramér's V for demographic variables |
| 4 | Profile membership counts, percentages, mean max probability |
| 5 | Frequentist A/B test results (proportion, means, distribution fit) |
| 6 | Sequential Bayesian A/B Results — P(B>A), Expected Loss, ESS |
| 7 | CUPED Variance Reduction — θ, variance reduction % |
| 8 | Full Causal Suite — DiD, IV, RDD, SCM, MC, DR-OLS HTE, CausalImpact BSTS |

Formatting follows **APA 7th edition**: no vertical borders, three horizontal rules, Times New Roman 12pt.

---

### 💡 Informal Guide: How to Read the APA Report Results

If you want to quickly understand what the tables are telling you in plain English:

* **Table 1 (LPA Model Fit)**: *"How many customer segments make sense?"*
  * **Look for**: Lower **BIC** / **AIC** values and an **Entropy** score closer to $1.0$ (above $0.80$ is great). It tells you if $K=3$ profiles represent your user base better than $K=2$ or $K=4$.
* **Table 2 & 3 (Profile Breakdown & Demographics)**: *"Who is in each segment and how do they behave?"*
  * **Table 2**: Compares metric averages across segments. Small $p$-values ($p < 0.05$) and higher $\eta^2$ mean the segments have distinctly different usage behaviors.
  * **Table 3**: Shows if demographics (e.g. gender, tier, device) are tied to segments. A higher **Cramér's $V$** ($> 0.3$) indicates a strong relationship.
* **Table 4 (Segment Sizes)**: *"How big is each user group?"*
  * Gives you the percentage breakdown ($N, \%$) of your user base across profiles.
* **Table 5 (Frequentist A/B Test)**: *"Did B beat A in traditional testing?"*
  * **Look for**: Is $p < 0.05$? If yes, the difference between Treatment B and Control A is statistically significant under standard standard hypothesis testing.
* **Table 6 (Bayesian A/B Test)**: *"How sure are we that B is better, and what is the risk?"*
  * **$P(B > A)$**: If this is $\ge 95\%$, B is almost certainly better.
  * **Expected Loss**: If you deploy B and it turns out to be wrong, this is the worst-case metric drop. If loss is $<1\%$ of baseline, it's safe to ship!
* **Table 7 (CUPED Variance Reduction)**: *"How much noise did we clean up?"*
  * **Look for**: **Variance Reduction %**. Higher percentage means CUPED removed pre-experiment noise, tightening your confidence intervals so you need fewer users or less time to declare a winner.
* **Table 8 (Causal Inference Suite & Subgroups)**: *"Did the feature ACTUALLY cause the uplift, and for whom?"*
  * **Estimates & CIs**: Summarizes true treatment effects across observational methods (DiD, IV, RDD, SCM, Matrix Completion, CausalImpact BSTS).
  * **DR-OLS Subgroups**: Highlights Heterogeneous Treatment Effects (HTE) — showing *which exact demographic subgroup* benefited the most.

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
