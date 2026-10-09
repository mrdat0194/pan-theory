# Unified XAI-JEPA & XAI-DoE for OmniStats: Concepts, Implementation & Roadmap

> **Location:** `omnistats/XAI_JEPA_ROADMAP.md`  
> **Status:** ✅ Phase 1 & 2 complete — all P0/P1 modules verified. XAI-DoE (Gendex DoE suite) integrated.  
> **Covers:** All concepts from handwritten notes (Aug-2020, Feb-2022, Nov-2022), Tang (2023) dequantization,  
> Nam-Ky Nguyen (VIASM, 2018) Gendex DoE suite (`C:\Users\mrdat\Downloads\workhop`), Incomplete Block Designs (IBD/CIBD),  
> Alpha Designs, Row-Column Designs (RCD/RRCD), Definitive Screening Designs (DSD), Response Surface Methodology (RSM),  
> Mean Field Games (MFG), and Buckmaster PINN verification.

---

## 🎯 The Core Mission: Bridging JEPA World Models & Rigorous DoE with OmniStats

**The core philosophy of OmniStats is to make complex probabilistic profiles and high-dimensional experimental spaces approachable, interpretable, and optimal for human decision-making.**

Traditionally, OmniStats Stage 1 (Latent Profile Analysis) clusters users into interpretable segments for A/B testing and Causal Impact. **XAI-JEPA** and **XAI-DoE** extend this philosophy directly into deep learning world models and multi-factor experimental design:

1. **XAI-JEPA (Explainable Latent World Models):** Maps opaque, continuous JEPA embeddings onto a discrete *Eigenstate Bank* (or *Concept Bank*), converting black-box world models into structured, human-readable **Latent Profiles**. Decision-makers query the model using `bayesian_inverse_score` to see exactly which concepts drive predictions.
2. **XAI-DoE (Explainable Design of Experiments):** Integrates Nam-Ky Nguyen's Gendex algorithms (Alpha designs, Row-Column blocking, Definitive Screening Designs, and Response Surface Optimization) directly into OmniStats:
   - **Screen $N$ factors in $2N+1$ runs** without aliasing main effects with 2-factor interactions (Definitive Screening Designs).
   - **Eliminate 1D and 2D nuisance variability** across user cohorts, server batches, and time windows (Alpha Designs & Row-Column Designs).
   - **Map continuous non-linear response surfaces** to pinpoint optimal feature configurations using minimal experimental units (Second-Order Designs / RSM).
   - **Sample continuous JEPA latent state spaces systematically** by replacing random noise perturbations with orthogonal response surface grids.

Unifying XAI-JEPA continuous world models with XAI-DoE experimental design creates a closed-loop autonomous intelligence pipeline for OmniStats.

---

## 1. Theoretical Foundation (Notes & Workshop Materials → Code)

### 1.1 Probability → Decision → Information Theory (Aug 2020)

| Concept from Notes | Implemented In | Formula |
|---|---|---|
| **Bayes inverse problem** $n \to y$ | `information_theory.py::bayesian_inverse_score()` | ℓ₂ posterior over concept bank |
| **JKL Divergence** | `information_theory.py::jeffreys_kl()` | $J(p,q) = KL(p\|q) + KL(q\|p)$ |
| **Shannon: Uncertainty + KL = Entropy** | `information_theory.py::shannon_entropy()` | $H(p) = -\sum p \log p$ |
| **Forward KL: $q(x)$ known → $p(x)\log\frac{p(x)}{q(x)}$** | `information_theory.py::kl_divergence()` | KL divergence |
| **Beta-Binomial** | `omnistats/modules/bayesian/beta_binomial.py` | Conjugate prior for proportions |
| **WAIC / LOO model selection** ✅ NEW | `bayesian/waic_loo.py` | Replaces AIC/BIC for causal model comparison |

### 1.2 Probability Distributions (Feb 2022)

| Distribution | Formula | Role in XAI Pipeline |
|---|---|---|
| **Geometric** | $P(X) = 1/p$ | Prior for discrete latent modes |
| **Gaussian** | $e^{-x^2/2\sigma^2}/\sqrt{2\pi\sigma^2}$ | Free-particle density matrix $p^{free}$ |
| **Maxwell-Boltzmann** ✅ NEW | $p(v) \propto v^2 e^{-v^2/2\sigma^2}$ | Prior for `energy_hat` in `APADecoder` |
| **Exponential** | $e^{t/\tau}$ | Waiting time between high-energy events |
| **Kinetic Energy** | $E_k = \|z\|^2/\beta$ | `information_theory.py::boltzmann_energy()` |

### 1.3 Classical → Quantum Density Matrix (Nov 2022)

$$\pi(x,n) \propto e^{-\beta E_n} \psi_n(x) \psi_n^*(x) \qquad p(x, x', \beta) = \sum_n e^{-\beta E_n} \psi_n(x) \psi_n^*(x')$$

| Theoretical Property | Implemented In | Status |
|---|---|---|
| **Convolution:** $\int dx' p(x,x',\beta_1)p(x',x'',\beta_2) = p(x,x'',\beta_1+\beta_2)$ | `dequantized_jepa_predictor.py::path_integral_rollout()` | ✅ |
| **Free density matrix** $p^{free}(z,z',\beta) \propto e^{-\|z-z'\|^2/4\sigma^2\beta}$ | `dequantized_jepa_predictor.py::free_density_matrix()` | ✅ |
| **High-Temperature limit** $p \approx e^{-\beta/2 V(x)} p^{free} e^{-\beta/2 V(x')}$ | `DequantizedLatentTransition.forward()` | ✅ |
| **Partition Function** $Z(\beta) = \text{Tr}(\rho) = \sum_n e^{-\beta E_n}$ | `information_theory.py::partition_function()` | ✅ |
| **Quantum Kalman prediction** $p(x_{t+1}\|x_t) \propto p^{free}(x_t, x_{t+1}, \beta)$ | `timeseries/quantum_kalman.py` ✅ NEW | ✅ |

### 1.4 Tang Dequantization (2023 Thesis)

| Tang Concept | Implemented In | Status |
|---|---|---|
| ℓ₂ importance sampling (SQ oracle) | `L2ImportanceSampler.sketch_matrix_vector()` | ✅ |
| Low-rank matrix sketch | `DequantizedLatentTransition` (eigenstate bank, rank `k`) | ✅ |
| **Clenshaw recursion for Chebyshev QSVT** | `datastructure/Lesson/chebyshev_qsvt.py` ✅ NEW | ✅ |
| **Heat-kernel / low-pass spectral filter** | `heat_kernel_chebyshev_coeffs()`, `low_pass_chebyshev_coeffs()` ✅ NEW | ✅ |
| **ChebyshevDequantizedTransition** (SOTA predictor) | `chebyshev_qsvt.py::ChebyshevDequantizedTransition` ✅ NEW | ✅ |

### 1.5 Generative Latent Flow & Causal Inference as Mean Field Games (ODE $\to$ MFG)

Upgrading standard Probability Flow / Neural ODEs ($\frac{dz}{dt} = f_\theta(z,t)$) to coupled **Hamilton-Jacobi-Bellman (HJB)** and **Fokker-Planck (FP)** equations.

| Concept | Implemented In | Formula / Role |
|---|---|---|
| **Schrödinger Bridges & Wasserstein Flow** | `mean_field_jepa.py` | Models `path_integral_rollout` as interacting agents minimizing transport cost via coupled HJB ($-\partial_t u - \frac{\sigma^2}{2}\Delta u + H = 0$) and FP ($\partial_t \rho + \nabla \cdot (\rho v) = \frac{\sigma^2}{2}\Delta \rho$). |
| **Continuous Shapley / Integrated Gradients** | `continuous_shapley.py` | Infinitesimal marginal contribution of each latent dimension over continuous manifolds. |
| **MFG Counterfactual Trajectory Flow** | `causal/mfg_did.py` | Population-level density transport equilibrium replacing linear DiD. |
| **MFG Confounder-Density Matching** | `causal/mfg_iv.py` | Steering treatment density $P(X\|Z)$ to target structural density $P(X)$ via coupled HJB-FP. |
| **MFG Discontinuity Density Flow** | `causal/mfg_rdd.py` | HJB jump interface in running variable density flow at cutoff. |

### 1.6 Buckmaster Fluid Dynamics & Computer-Assisted Verification (2023–2026)

Applying Tristan Buckmaster's PDE discovery & computer-assisted proof methodology to continuous-time latent world models (`latent_ode_jepa.py`).

| Concept | Implemented / Planned In | Role in XAI Pipeline |
|---|---|---|
| **PINNs as Discovery Tool** | `latent_ode_jepa.py::ODEDynamicsNet` | Resolves sharp gradients in latent state dynamics $\frac{dz(t)}{dt} = f_\theta(z(t), t)$. |
| **Gradient-Normalized Residual Re-Weighting** | `latent_ode_jepa.py::LatentODE_JEPA` | Penalizes ODE residual gradients for machine-precision trajectory prediction. |
| **Spectral Basis (FFT / Chebyshev)** | `spectral_ode_solver.py`, `chebyshev_qsvt.py` | Spatial derivatives and Clenshaw polynomial expansions for operator matrices. |
| **Computer-Assisted Proofs (CAP) & Interval Bounds** | `omnistats/modules/simulation/pinn_safety_oracle.py` ✅ NEW | Validates neural trajectories via interval enclosures and Neumann series contraction bounds. |

### 1.7 Advanced Design of Experiments (DoE) & Sequential Experimentation (Nguyen 2018, VIASM)

Derived from Nam-Ky Nguyen's Gendex algorithms (`C:\Users\mrdat\Downloads\workhop`).

#### 1.7.1 Incomplete Block Designs (IBD & Cyclic CIBD)
When treatments $v$ exceed block size $k < v$, full replication per block is impossible.
- **Incidence Matrix ($N$):** $N_{ij} = 1$ if treatment $i$ in block $j$, else $0$.
- **C-Matrix (Information Matrix):** $C = r I - \frac{1}{k} N N^T$.
- **Canonical Efficiency Factor ($E$):** $E = \frac{v-1}{v \cdot \text{Tr}(C^+)}$, used to rank candidate designs in Stage 2.

#### 1.7.2 Resolvable Alpha Designs (Alpha-Modules)
Resolvable designs group $b$ incomplete blocks into $r$ complete replicates of all $v$ treatments ($v = k \cdot s$).
- **Intra-Block & Inter-Block Variance:** $\text{Var}(\hat{\tau}_i - \hat{\tau}_j) = \frac{2\sigma^2}{r E}$, minimizing variance across multi-arm UI variant tests.

#### 1.7.3 Row-Column Designs (RCD & Resolvable RRCD)
Controls **two independent sources of nuisance variability** (e.g., User Cohort $\times$ Time Window):
$$y_{ijk} = \mu + \tau_i + \alpha_j + \beta_k + \varepsilon_{ijk}$$
- Eliminates 2D background noise in batch user allocations via orthogonal swap search.

#### 1.7.4 Definitive Screening Designs (DSD - Jones & Nachtsheim 2011)
3-level factor screening (levels $-1, 0, +1$) requiring only $2m + 1$ runs for $m$ factors.
- **Main Effect Orthogonality:** $\text{Cov}(\hat{\beta}_i, \hat{\beta}_j) = 0$ for $i \neq j$.
- **Zero Aliasing with 2FIs:** $\text{Cov}(\hat{\beta}_i, \hat{\beta}_{jk}) = 0$. Main effects are unconfounded with any 2-factor interaction.
- Fits quadratic curvature $y = \beta_0 + \sum \beta_i x_i + \sum \beta_{ii} x_i^2$ in a single screening pass.

#### 1.7.5 Second-Order Response Surface Methodology (SOD / RSM)
Fits 3-level second-order polynomial response surfaces:
$$y = \beta_0 + \sum_{i=1}^k \beta_i x_i + \sum_{i=1}^k \beta_{ii} x_i^2 + \sum_{i<j}^k \beta_{ij} x_i x_j + \varepsilon$$
Includes Central Composite (CCD), Box-Behnken (BBD), Small Composite (SCD), and Augmented-Pair (APD) designs constructed via swap-permutation search ($f$ and $g$ orthogonality criteria).

#### 1.7.6 Sequential Experimentation (Fisher's 3 R's)
Stage 1: Primary Screening (DSD) $\to$ Stage 2: Augmentation (SCD/APD) $\to$ Stage 3: Response Surface Optimization (CCD/BBD/SOD) to locate the stationary optimum $x^* = -\frac{1}{2} B^{-1} b$.

---

## 2. Complete File Inventory

### 2.1 Phase 1 — Core Modules (Complete ✅ & In Progress 🚀)

| File | Type | Status | Purpose |
|---|---|---|---|
| [`omnistats/modules/information_theory.py`](information_theory.py) | XAI | ✅ Complete | Shannon Entropy, JKL, Bayesian Inverse Score, Boltzmann Energy |
| [`datastructure/Lesson/dequantized_jepa_predictor.py`](../datastructure/Lesson/dequantized_jepa_predictor.py) | JEPA | ✅ Complete | `L2ImportanceSampler`, `DequantizedLatentTransition`, `free_density_matrix()` |
| [`omnistats/modules/xai_visualisation.py`](xai_visualisation.py) | Vis | ✅ Complete | Energy landscape, JKL boundary, and partition function plots |
| [`omnistats/modules/jepa_bridge.py`](jepa_bridge.py) | Bridge | ✅ Complete | `APADecoder` extended: `energy_hat`, `beta_hat`, `entropy` heads |
| [`omnistats/modules/doe/alpha_design.py`](modules/doe/alpha_design.py) | DoE | 🚀 Phase 1 | `AlphaDesignBuilder`, `compute_intra_block_variance()` |
| [`omnistats/modules/doe/incomplete_block.py`](modules/doe/incomplete_block.py) | DoE | 🚀 Phase 1 | `CyclicIBDBuilder`, `compute_c_matrix()`, `calculate_efficiency_factor()` |
| [`omnistats/modules/doe/row_column_design.py`](modules/doe/row_column_design.py) | DoE | 🚀 Phase 1 | `RowColumnDesignBuilder`, `ResolvableRCDBuilder` |
| [`omnistats/modules/doe/definitive_screening.py`](modules/doe/definitive_screening.py) | DoE | 🚀 Phase 1 | `DefinitiveScreeningBuilder`, `fit_screening_model()` |

### 2.2 Phase 2 — Frontier P0/P1 Modules (Complete ✅ & Planned 🟡)

| File | Priority | Purpose |
|---|---|---|
| [`datastructure/Lesson/chebyshev_qsvt.py`](../datastructure/Lesson/chebyshev_qsvt.py) | **P0** | Chebyshev-QSVT via Clenshaw recursion + `ChebyshevDequantizedTransition` |
| [`eb_jepa/eb_jepa/quantum_mppi.py`](../eb_jepa/eb_jepa/quantum_mppi.py) | **P0** | `QuantumMPPIPlanner`: Boltzmann noise + JKL cost + β annealing |
| [`omnistats/modules/bayesian/maxwell_prior.py`](modules/bayesian/maxwell_prior.py) | **P1** | Maxwell-Boltzmann prior for `energy_hat`, `maxwell_prior_loss()` |
| [`omnistats/modules/bayesian/waic_loo.py`](modules/bayesian/waic_loo.py) | **P1** | WAIC + PSIS-LOO + `compare_models()` for causal estimator selection |
| [`omnistats/modules/timeseries/quantum_kalman.py`](modules/timeseries/quantum_kalman.py) | **P1** | `QuantumKalmanFilter` + RTS smoother + ATT estimation |
| [`omnistats/modules/doe/response_surface.py`](modules/doe/response_surface.py) | **P1** | `ResponseSurfaceBuilder` (CCD, BBD, SCD, APD) + stationary point search |
| [`omnistats/modules/doe/sod_optimizer.py`](modules/doe/sod_optimizer.py) | **P1** | `SwapPermuteSOD` for algorithmic 3-level design generation |

### 2.3 Phase 3 — SOTA Latent Games & DoE Sampling (Upcoming 🟣)

| File | Priority | Purpose |
|---|---|---|
| [`omnistats/modules/game_theory/continuous_shapley.py`](modules/game_theory/continuous_shapley.py) | **P1** | Integrated gradients and continuous Shapley approximations |
| [`omnistats/modules/game_theory/mean_field_jepa.py`](modules/game_theory/mean_field_jepa.py) | **P1** | Forward-Backward HJB and Fokker-Planck equations |
| [`omnistats/modules/causal/mfg_did.py`](modules/causal/mfg_did.py) | **P1** | **MFG Counterfactual Trajectory Flow**: Density transport equilibrium |
| [`omnistats/modules/causal/mfg_iv.py`](modules/causal/mfg_iv.py) | **P1** | **MFG Confounder-Density Matching**: HJB-FP velocity field steering |
| [`omnistats/modules/causal/mfg_rdd.py`](modules/causal/mfg_rdd.py) | **P1** | **MFG Discontinuity Density Flow**: HJB jump interface |
| [`omnistats/modules/doe/xai_doe_bridge.py`](modules/doe/xai_doe_bridge.py) | **P1** | **Latent DoE Sampler**: Orthogonal DSD/RSM grids in JEPA latent space |
| [`omnistats/modules/doe/latent_rsms.py`](modules/doe/latent_rsms.py) | **P1** | Maps JEPA energy landscape $V(z)$ via second-order response surface |

---

## 3. Architecture Diagrams

### 3.1 APADecoder (Extended)

```
z [B, D_latent]
      │
   SharedNet (MLP + LayerNorm + GELU)
      │
  ┌───┴──────────────────────────────┐
  ▼          ▼          ▼            ▼
att_hat   risk_hat  energy_hat    beta_hat
[B]        [B]         [B]           [B]
(ATT)   (Softplus)  (Softplus)   (Softplus)
                  ↓ maxwell_prior_loss(energy_hat, beta_hat)
                  + entropy [B]
                  + jkl_from_prior [B]   ← bayesian_inverse_score()
                  + posterior [B, N]     ← ℓ₂ soft concept attribution
```

### 3.2 ChebyshevDequantizedTransition (SOTA Predictor)

```
z_input [B, D]
      │
  PotentialNet → V(z) [B]             "Energy cost of state"
      │
  e^{-β/2 V(z)} weighting             "High-Temperature approximation"
      │
  Clenshaw Recursion on EigenstateBank [k, D]
  (Chebyshev polynomial of A^T A applied to z_weighted)
  Spectral filter:
    'heat'     → f(λ) = exp(-β λ²)   [quantum diffusion]
    'low_pass' → f(λ) = 1_{|λ|≤cut}  [stable attractor isolation]
      │
  Boltzmann re-weighting: w_n = softmax(-β E_n)
      │
  cheb_head: [k → 2D]
      │
  z_next_mean [B,D]  +  z_next_logvar [B,D]
```

### 3.3 QuantumMPPIPlanner (Action Sampling)

```
For each MPPI iteration:
  ┌─ Boltzmann noise sampling ─────────────────────────────┐
  │  beta(t) = beta_0 * (T-t)/T    ← annealing schedule   │
  │  w_n(t) = softmax(-beta_t * E_n)                       │
  │  epsilon_t = sum_n w_n * psi_n * xi_n,  xi_n ~ N(0,1) │
  └────────────────────────────────────────────────────────┘
        │
  actions = mean + std * epsilon              [T, B, A]
        │
  cost = prediction_error
       + lambda_jkl * JKL(q_actions || Boltzmann_prior)    [B]
        │
  Elite selection → update mean, std
```

### 3.4 Quantum Kalman Filter (Causal Impact)

```
Standard Kalman prediction:   x_{t+1} = F x_t + N(0, Q)
Quantum Kalman prediction:    p(x_{t+1}|x_t) ∝ p^{free}(x_t, x_{t+1}, β)
                              Q_quantum = 2σ²β · exp(-β/2 · V(x_t)) · I
                                          ↑ high energy state = more uncertainty

Forward filter → RTS smoother → ATT estimation
```

### 3.5 Sequential DoE Execution Flow

```
Raw Candidate Factors (x_1, x_2, ..., x_m)  [Large m]
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Definitive Screening Design (DSD)               │
│ • Runs: 2m + 1                                          │
│ • Zero main-effect / 2FI aliasing                       │
│ • Identifies top-k significant active factors            │
└─────────────────────────┬───────────────────────────────┘
                          │ Active Factors (k << m)
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Incomplete Block / Row-Column Execution        │
│ • Allocates k factors across blocks/cohorts/time windows │
│ • Resolvable Alpha or Row-Column Design (RCD)           │
│ • Absorbs 1D/2D nuisance variance                        │
└─────────────────────────┬───────────────────────────────┘
                          │ Balanced Multi-Arm Data
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Response Surface Optimization (RSM)            │
│ • Fits Quadratic Model: y = β_0 + Σ β_i x_i + Σ β_ij x_i x_j + Σ β_ii x_i² │
│ • Solves stationary point x* = -1/2 B⁻¹ b               │
│ • Pinpoints optimal feature configuration               │
└─────────────────────────┴───────────────────────────────┘
```

### 3.6 Dual Nuisance Variance Blocking (Row-Column Design)

```
                  Column Blocking Factor (Time Slot / Device Type)
                   Col 1       Col 2       Col 3       Col 4
                 ┌───────────┬───────────┬───────────┬───────────┐
         Row 1   │ Variant A │ Variant B │ Variant C │ Variant D │
Row      ────────┼───────────┼───────────┼───────────┼───────────┤
Blocking Row 2   │ Variant C │ Variant D │ Variant A │ Variant B │
Factor   ────────┼───────────┼───────────┼───────────┼───────────┤
(User    Row 3   │ Variant B │ Variant A │ Variant D │ Variant C │
Cohort)  ────────┼───────────┼───────────┼───────────┼───────────┤
         Row 4   │ Variant D │ Variant C │ Variant B │ Variant A │
                 └───────────┴───────────┴───────────┴───────────┘
```

### 3.7 Unified Closed-Loop Autonomous System Architecture

```
                      ┌───────────────────────────────────────────┐
                      │    JEPA Latent World Model (Predictor)    │
                      │   • Density matrix ρ(z, z', β)            │
                      │   • Energy landscape V(z)                 │
                      └─────────────────────┬─────────────────────┘
                                            │ Latent Space Representation
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │      XAI-DoE Latent Grid Generator        │
                      │   • Definitive Screening (DSD) in Latent  │
                      │   • Orthogonal Concept Grid Sample        │
                      └─────────────────────┬─────────────────────┘
                                            │ Factorial Experiment Plan
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │     OmniStats Stage 2 Execution Engine    │
                      │   • Resolvable Alpha & Row-Column Blocking│
                      │   • CUPED Variance Reduction              │
                      └─────────────────────┬─────────────────────┘
                                            │ Live Trial Data
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │     Response Surface & MFG Causal Fit     │
                      │   • Fit Quadratic Model y(x) & HJB-FP     │
                      │   • Compute Stationary Point x*           │
                      │   • Fine-tune Concept Bank & JEPA Weights │
                      └───────────────────────────────────────────┘
```

---

## 4. How to Use the Integrated Pipeline

```python
import torch, numpy as np
import sys; sys.path.insert(0, 'datastructure/Lesson')

# ── Core XAI & Predictors ───────────────────────────────────────────────────
from omnistats.modules.information_theory import compute_xai_metrics
from chebyshev_qsvt import ChebyshevDequantizedTransition
from omnistats.modules.timeseries.quantum_kalman import QuantumKalmanFilter

# ── DoE Generators ───────────────────────────────────────────────────────────
from omnistats.modules.doe.definitive_screening import DefinitiveScreeningBuilder
from omnistats.modules.doe.row_column_design import ResolvableRCDBuilder
from omnistats.modules.doe.response_surface import ResponseSurfaceBuilder

# Step 1: DSD Screening for 6 candidate feature factors (2m+1 = 13 runs)
dsd_builder = DefinitiveScreeningBuilder(n_factors=6)
dsd_grid = dsd_builder.generate()

# Step 2: Row-Column Nuisance Blocking for trial deployment
rcd_builder = ResolvableRCDBuilder(v_treatments=8, n_rows=4, n_cols=2, n_replicates=3)
schedule = rcd_builder.generate_schedule()

# Step 3: Encode JEPA latents and obtain XAI metrics
z = jepa_encoder(context_frames)
metrics = compute_xai_metrics(z, reference_embeddings=concept_bank, beta=1.0)

# Step 4: Quantum Kalman Causal Impact
qkf = QuantumKalmanFilter(beta=1.0, diffusion_sigma=0.3)
att = qkf.estimate_att(y_tensor, X_tensor, T_treat=40)
print(f"ATT = {att['estimate']:.3f}  [{att['ci_lower']:.3f}, {att['ci_upper']:.3f}]")

# Step 5: Fit Response Surface Methodology (RSM) Quadratic Model
rsm = ResponseSurfaceBuilder()
rsm.fit(X=dsd_grid, y=observed_conversions)
optimum = rsm.find_stationary_point()
print(f"Optimal Factor Setting (x*): {optimum['x_star']}")
```

---

## 5. Next Steps (P2–P3)

### 🟡 P2 — Quantum β-VAE
Replace isotropic Gaussian prior in iVAE with Boltzmann prior `p(z) ∝ exp(-β V(z))`.
- File: `VAE/Beta-VAE/quantum_beta_vae.py`

### 🟡 P2 — Active GP-DoE & Micro-Batch Split-Plot Designs
Use DSD as initial seed design for Gaussian Process surrogate modeling. Implement Split-Plot designs for hard-to-change vs. easy-to-change server parameters.
- Files: `omnistats/modules/doe/active_gp_doe.py`, `omnistats/modules/doe/split_plot.py`

### 🟡 P2 — Hardware Acceleration & GPU Fallback Mitigation
Wrap `chebyshev_qsvt.py` using `torch.compile()` or OpenAI Triton kernels. Migrate Bayesian causal components from PyMC to NumPyro (JAX).

### 🟢 P3 — Heavy-Tailed Robust DoE & Path Integral ODE Solvers
Extend RSM fitting to Generalized Linear Models (GLM) for heavy-tailed metrics (e.g. revenue). Unify `stirling_partition.py` with partition function formalism.

---

## 6. Open Questions

> [!IMPORTANT]
> **Q1 — Rank `k`:** What is the effective intrinsic dimension of the JEPA latent space? Run PCA to set `rank_k` for `ChebyshevDequantizedTransition`.

> [!IMPORTANT]
> **Q2 — Latent DSD Step Size:** When sampling a DSD grid $x_i \in \{-1, 0, +1\}$ along concept vectors $v_i$, what step size $\delta$ maintains valid, in-distribution latent states?

> [!IMPORTANT]
> **Q3 — Block Size $k$ vs. Latent Profile Count $K_{LPA}$:** Should the number of incomplete blocks $b$ in an Alpha design equal $K_{LPA}$?

> [!WARNING]
> **Q4 — Stationary Point Classification:** Second-order response surfaces often yield saddle points. Should automatic ridge analysis be triggered when Hessian $B$ has mixed eigenvalues?

---

## 7. Reference Index

| Source | Key Contribution |
|---|---|
| **Nguyen, N.-K. (2018)** | *Definitive Screening, Response Surface & Incomplete Block Designs*. VIASM Workshop |
| **Jones & Nachtsheim (2011)** | 3-level Definitive Screening Designs (DSD) |
| **Box & Wilson (1951)** | Central Composite Designs (CCD) & Response Surface Methodology |
| **Box & Behnken (1960)** | Box-Behnken 3-level Designs (BBD) |
| **Draper & Lin (1990)** | Small Composite Designs (SCD) |
| **Morris, M.D. (2000)** | Augmented-Pair Designs (APD) |
| **Tang, E. (2023)** | ℓ₂ importance sampling, QSVT dequantization, **Chebyshev/Clenshaw recursion** |
| **Kalman, R.E. (1960)** | Classical linear filter baseline for Quantum Kalman |
| **Buckmaster et al. (2023–2026)** | PINNs for unstable PDE blow-up, machine-precision gradient-normalized re-weighting |

---

## 8. Phase VI — Open Research Directions & Unified Synergy Matrix

### 8.1 Synergy Matrix

| Dimension | XAI-JEPA Component | XAI-DoE Component | Combined Synergy |
|---|---|---|---|
| **Representation** | Continuous Latent Space $z \in \mathbb{R}^D$ | Discrete Factorial Matrix $x \in \{-1, 0, +1\}^m$ | Discrete DoE Grids in Continuous Manifolds |
| **Uncertainty** | Boltzmann Energy $V(z)$, Shannon Entropy | Block / Row-Column Variance Components | Joint Latent-Physical Variance Reduction |
| **Optimization** | Quantum MPPI Planner | RSM Stationary Search ($x^* = -\frac{1}{2} B^{-1} b$) | Physics-Informed Experimental Policy Synthesis |

### 8.2 Real-Traffic Closing the Loop & PINN Safety Gate (`PINNSafetyOracle`)
Route winning feature candidates to real users. Gate World Model actions using `PINNSafetyOracle`: verify `ATT_lower_CI > 0` and Neumann contraction radius $r < 1$ around latent trajectories before deployment.

### 8.3 Causal MFG & Density-Level Feedback
Fit `MFGCounterfactualFlow` on density $\rho(x, t)$ after each iteration and add Wasserstein density consistency $W_2(\hat{\rho}_{t+1}, \rho^*_{t+1})$ to the JEPA loss.

### 8.4 Adaptive Concept Bank & Causal Experiment Graph
Re-discover top-k predictive concept directions online after fine-tuning. Store experimental history as a Causal Knowledge Graph for graph neural network candidate proposal.

---

## 9. Autonomous Harness Engineering for Recursive Self-Improvement

> **Inspired by:** Lilian Weng (2026). *Harness Engineering for Self-Improvement.* https://lilianweng.github.io/posts/2026-07-04-harness/  
> **Status:** 🚀 Architecture Design — Defines how the OmniStats autonomous loop achieves Recursive Self-Improvement (RSI) by treating the harness (workflow + memory + sub-agents + evaluators) as an **optimizable artifact**, not a static script.

---

### 9.1 Core Principle: The Harness IS the Optimization Target

Weng (2026) defines a **harness** as the layer between the raw model and the real world that determines *how* the model thinks, plans, calls tools, perceives context, stores artifacts, and evaluates its own results. In the OmniStats XAI-JEPA context:

> The harness = `run_autonomous_loop.py` + `experiment_design.py` + all Stage 1–5 orchestration code + the DoE schedule generator + the PINN safety oracle + the sub-agent configs.

**Key insight from Weng (2026):** The progression of what gets optimized is:
```
instruction prompts → structured context → workflow → harness code → optimizer code
```

OmniStats is currently at the **workflow** layer. The roadmap below moves it toward **self-optimizing harness code**.

---

### 9.2 Four Harness Design Patterns Applied to OmniStats

#### Pattern 1: Goal-Oriented Workflow Automation

Map the OmniStats 5-stage pipeline onto an explicit goal-directed loop:

```
┌───────────────────────────────────────────────────────────────────┐
│  OmniStats Autonomous Harness Loop (OAHL)                        │
│                                                                   │
│  GOAL: Maximize ATT lift on target metric within safety bounds   │
│                                                                   │
│  PLAN  ──► EXECUTE  ──► OBSERVE / TEST  ──► IMPROVE  ──► REPEAT │
│  (DSD)    (Stage 2)    (QKF ATT + PINN)   (JEPA FT)             │
│                                                  │               │
│              ◄──── human-approved gate ──────────┘               │
└───────────────────────────────────────────────────────────────────┘
```

**Implementation:** `omnistats/modules/harness/oahl_controller.py` *(new)*
- Accepts a structured goal spec: `{target_metric, mde, safety_threshold, max_iterations}`.
- Emits structured status records after every stage to the file system (not just print statements).

#### Pattern 2: File System as Persistent Harness Memory

Weng (2026) emphasizes that long-horizon agents must **never carry full workflow state in context**. Instead, durable state lives as files that outlast any single LLM call. Map this directly to OmniStats:

| Artifact | Current State | Harness-Memory Upgrade |
|---|---|---|
| Experiment leaderboard | `leaderboard_iterN.csv` (flat) | `harness/memory/experiment_graph.json` — full provenance DAG |
| ATT history | `autonomous_loop_log.csv` | `harness/memory/att_timeline.parquet` — queryable by QKF |
| DoE schedule | Generated once, discarded | `harness/memory/doe_schedules/` — indexed by iteration & design type |
| PINN safety oracle results | Not persisted | `harness/memory/safety_certificates/` — interval enclosures per rollout |
| Failure traces | Print logs | `harness/memory/failure_traces/` — structured JSON per failure mode |

**Key rule from Weng (2026):** *"If subagent outputs only live in transient context, they become obsolete. If stored as files, the model can recover after interruptions and reason over its own execution history."*

#### Pattern 3: Sub-Agent Parallelism & Backend Job Management

The harness spawns specialized sub-agents for isolated subtasks without polluting the main experiment context:

```
┌─────────────────────────────────────────────────────────────────┐
│           OmniStats Harness — Sub-Agent Registry               │
├──────────────────┬──────────────────────────────────────────────┤
│ Sub-Agent        │ Role & Integration                           │
├──────────────────┼──────────────────────────────────────────────┤
│ service-agent    │ Google ADK LlmAgent (Gemini 2.0)             │
│ (service-agent/) │ • Queries ChromaDB RAG for domain context    │
│                  │ • Firecrawl MCP: scrapes competitor metrics  │
│                  │ • Tavily MCP: real-time market benchmarks    │
│                  │ → Feeds verified context into DSD factor     │
│                  │   definitions before experiment launch       │
├──────────────────┼──────────────────────────────────────────────┤
│ growmetrics-     │ Manim + Edge-TTS video generation agent      │
│ agent            │ • Receives winning experiment ATT report     │
│ (growmetrics/)   │ • Generates educational animation of XAI    │
│                  │   concept attribution (concept bank → video) │
│                  │ • 4-gate quality validation + self-correction│
│                  │ → Produces human-readable XAI briefing video │
├──────────────────┼──────────────────────────────────────────────┤
│ jepa-trainer     │ Fine-tunes JEPA world model on winning       │
│ (eb_jepa/)       │ population embeddings from Stage 4 causal   │
│                  │ evaluation; updates Concept Bank             │
├──────────────────┼──────────────────────────────────────────────┤
│ doe-optimizer    │ Runs SwapPermuteSOD / DSD generation for the │
│ (modules/doe/)   │ next iteration's experimental design based   │
│                  │ on RSM stationary point x* from current iter │
└──────────────────┴──────────────────────────────────────────────┘
```

**Implementation:** `omnistats/modules/harness/subagent_manager.py` *(new)*
- `launch_subagent(name, task_spec, output_dir)` — spawns job, writes PID + spec to `harness/jobs/`.
- `poll_subagent(job_id)` — checks completion by reading `output_dir/status.json`.
- `merge_results(job_ids)` — aggregates sub-agent outputs into main harness memory.

#### Pattern 4: Agentic Context Engineering (ACE / MCE)

Adapting Weng (2026) ACE (Zhang et al. 2025) and MCE (Ye et al. 2026) to OmniStats:

```python
# harness/context_playbook.json — structured, itemized, de-duplicated
{
  "experiment_goal": "Maximize checkout conversion ATT > 0.02",
  "active_concepts": ["Price Sensitivity", "Trust Signal", "Urgency"],
  "safety_constraints": ["ATT_lower_CI > 0", "pinn_radius < 1.0"],
  "doe_strategy": "DSD(m=6) → Alpha(v=8,b=4) → CCD",
  "failure_patterns": [
    {"id": "FP-001", "pattern": "DSD saddle point", "fix": "ridge_analysis()"},
    {"id": "FP-002", "pattern": "PINN radius > 1", "fix": "reduce beta annealing"}
  ],
  "skill_history": [...]   // MCE outer-loop skill evolution log
}
```

**Key rule:** The curator updates `context_playbook.json` incrementally with `(identifier, description)` bullets — it never rewrites the full blob. Prevents context collapse between long-horizon iterations.

---

### 9.3 Self-Harness Evolution Loop (Weng Pattern — Self-Harness, Zhang et al. 2026)

Adapt the **weakness mining → harness proposal → validation** loop to OmniStats:

```
┌──────────────────────────────────────────────────────────────────────┐
│            OmniStats Self-Harness Evolution Loop                    │
└──────────────────────────────────────────────────────────────────────┘

Step 1: WEAKNESS MINING
  • Run N=k iterations of the autonomous loop
  • Collect failure traces: {stage_failed, root_cause, causal_status}
  • Cluster failure patterns:
      - "DSD identifies non-significant factors" → DoE hyperparameter issue
      - "PINN oracle rejects all rollouts"       → β schedule too aggressive
      - "RSM stationary point is saddle"         → insufficient axial runs
      - "Quantum Kalman ATT CI too wide"         → pre-period too short
      - "service-agent RAG returns empty"        → ChromaDB index stale

Step 2: HARNESS PROPOSAL (bounded edits)
  • Propose targeted harness code edits, each grounded by a failure record
  • Edits scoped to: `oahl_controller.py`, `doe/definitive_screening.py`,
    `timeseries/quantum_kalman.py`, `simulation/pinn_safety_oracle.py`
  • Read-only: verifier logic, base JEPA model weights, `run_autonomous_loop.py` main

Step 3: VALIDATION (regression testing)
  • Held-in: D_in  — does the specific failure pattern resolve?
  • Held-out: D_out — does existing passing behavior regress?
  • Accept: merge edit → h_{t+1}
  • Reject: log in harness/memory/rejected_edits.json without modifying harness
```

**Implementation:** `omnistats/modules/harness/self_harness_evolver.py` *(new)*

**Permission boundaries (critical — Weng 2026 caution):**
- The evolver **cannot** disable the PINN safety oracle.
- The evolver **cannot** swap the base JEPA model.
- The evolver **cannot** modify the human-in-the-loop ATT approval gate.
- All accepted edits are versioned in `harness/versions/h_<timestamp>/`.

---

### 9.4 AlphaEvolve / Darwin Gödel Machine Pattern for DoE Workflow Search

Adapting Novikov et al. (2025) AlphaEvolve and Zhang et al. (2025) Darwin Gödel Machine to automatically discover optimal DoE-JEPA workflow configurations:

```python
# EVOLVE-BLOCK-START: doe_workflow_config
DOE_WORKFLOW = {
    "screening_design": "DSD",      # candidates: DSD, PB, FF
    "blocking_design":  "Alpha",    # candidates: Alpha, RCD, RRCD, CIBD
    "rsm_design":       "CCD",      # candidates: CCD, BBD, SCD, APD
    "sequential_stages": 3,
    "beta_schedule":    "cosine",   # candidates: cosine, linear, step
    "mppi_horizon":     5,
}
# EVOLVE-BLOCK-END
```

The meta-agent evaluates each workflow config by comparing `WAIC` scores across DoE designs and selects configurations on the **Pareto frontier** of {ATT lift, run efficiency, safety certificate pass rate}.

**Implementation:** `omnistats/modules/harness/workflow_evolver.py` *(new)*

---

### 9.5 service-agent Integration Specification

[`service-agent/agent.py`](../service-agent/agent.py) is an **ADK LlmAgent** (Gemini 2.0 Flash) with:
- **Google Drive MCP** — indexes domain PDFs (GDPR, compliance, benchmarks).
- **Firecrawl MCP** — scrapes JS-rendered competitor pages & market reports.
- **Tavily MCP** — real-time search for updated market statistics.
- **ChromaDB RAG** — vector retrieval over all indexed documents.

**Integration into the OmniStats Harness:**

```python
# omnistats/modules/harness/service_agent_bridge.py  (new)

class ServiceAgentBridge:
    """
    Calls the service-agent's RAG tool to fetch domain context
    before DSD factor specification, ensuring experiment factors
    are grounded in verified market intelligence rather than guesswork.
    """

    def enrich_doe_factors(self, candidate_factors: list[str]) -> dict:
        """
        For each candidate DSD factor, query service-agent ChromaDB for:
        - Historical benchmark values (baseline conversion, revenue/user)
        - Regulatory constraints (GDPR relevance flags)
        - Competitor A/B test patterns from scraped reports
        Returns enriched factor spec with recommended level boundaries.
        """
        ...

    def fetch_pre_experiment_context(self, metric: str) -> dict:
        """
        Returns verified baseline_mean, baseline_std, and confounders
        for the target metric — fed into quantum_kalman.estimate_att()
        pre-period calibration.
        """
        ...
```

**Harness memory bridge:** After each experiment iteration, service-agent indexes the `apa_report_iterN.md` APA reports back into ChromaDB — creating a **self-growing RAG knowledge base** of past experiment results that informs future DoE factor specifications.

---

### 9.6 growmetrics Integration Specification

[`growmetrics/`](../growmetrics/) is a **Manim + Edge-TTS AI video generation service** with:
- 4-stage quality gate self-correction loop.
- Self-correcting fallback router for failed Manim codegen.
- Planned A2A Protocol endpoint (`/a2a/v1/tasks`) for harness delegation.

**Integration into the OmniStats Harness:**

```python
# omnistats/modules/harness/growmetrics_bridge.py  (new)

class GrowmetricsBridge:
    """
    Delegates XAI explanation video generation to growmetrics after
    each autonomous loop iteration completes, for human-readable
    briefings on winning concept attributions.
    """

    def generate_xai_briefing(
        self,
        concept_attributions: dict,  # from bayesian_inverse_score()
        att_result: dict,            # from QuantumKalmanFilter.estimate_att()
        rsm_optimum: dict,           # from ResponseSurfaceBuilder.find_stationary_point()
        output_path: str,
    ) -> str:
        """
        Sends a structured scene spec to growmetrics:
        - Scene 1: Energy landscape V(z) with winning concept highlighted
        - Scene 2: RSM response surface contour with x* marked
        - Scene 3: ATT confidence interval bar chart (Quantum Kalman)
        Returns path to generated .mp4 briefing video.
        """
        ...
```

**Planned A2A integration (growmetrics Phase 2 roadmap):**
Once growmetrics exposes `/a2a/v1/tasks`, `oahl_controller.py` will fire-and-forget video generation jobs asynchronously between iterations, so human reviewers have a video briefing ready when approving the next iteration's safety gate.

---

### 9.7 Known Failure Modes & Harness Safeguards

Adapting Trehan & Chopra (2026) and Weng (2026) failure taxonomies to OmniStats:

| Failure Mode | OmniStats Manifestation | Harness Safeguard |
|---|---|---|
| **Training-data default bias** | JEPA predictor anchors to dominant cohort embedding | Adaptive Concept Bank re-discovery after each fine-tune step |
| **Implementation drift** | RSM fit silently falls back to linear model under curvature | `verify_dsd_properties()` assertion + WAIC model comparison forced |
| **Memory & context degradation** | ATT history lost between iterations | File-system harness memory: `harness/memory/att_timeline.parquet` |
| **Over-optimism / p-hacking** | World Model declares lift on noisy pre-period data | PINN Safety Oracle mandatory: `ATT_lower_CI > 0` + Neumann bound $r < 1$ |
| **Reward hacking** | Agent disables safety oracle to pass gate | Safety oracle and verifier are **read-only** surfaces in harness evolver |
| **Diversity collapse** | Evolutionary DoE search converges to CCD always | ShinkaEvolve cosine similarity rejection: reject configs too similar to pool |
| **Weak evaluator** | WAIC scores for all causal models within noise | Expand `compare_models()` to include conformal prediction calibration |

---

### 9.8 Harness File Layout

```
omnistats/
├── modules/
│   └── harness/                          # NEW: Harness Engineering Layer
│       ├── oahl_controller.py            # Goal-directed autonomous loop orchestrator
│       ├── subagent_manager.py           # Sub-agent launch, poll, merge
│       ├── service_agent_bridge.py       # ADK RAG enrichment for DoE factors
│       ├── growmetrics_bridge.py         # A2A video briefing delegation
│       ├── self_harness_evolver.py       # Weakness mining → harness edit → validate
│       └── workflow_evolver.py           # AlphaEvolve-style DoE workflow search
│
├── harness/                              # NEW: Persistent Harness Memory (file-system)
│   ├── memory/
│   │   ├── context_playbook.json         # ACE/MCE structured context (itemized bullets)
│   │   ├── experiment_graph.json         # Provenance DAG of all experiment iterations
│   │   ├── att_timeline.parquet          # Queryable ATT history for QKF calibration
│   │   ├── doe_schedules/               # Per-iteration DoE layouts
│   │   ├── safety_certificates/         # PINN interval enclosures per rollout
│   │   └── failure_traces/             # Structured JSON failure records
│   ├── jobs/                            # Sub-agent job specs & status files
│   ├── versions/                        # Harness edit version history
│   └── rejected_edits.json             # Logged but not applied harness proposals
```

---

### 9.9 Reference Index — Harness Engineering

| Source | Key Contribution |
|---|---|
| **Weng, L. (2026)** | *Harness Engineering for Self-Improvement.* Comprehensive RSI harness taxonomy |
| **Zhang et al. (2026)** | Self-Harness: weakness mining → bounded proposal → validation loop |
| **Lin et al. (2026)** | AHE: component/experience/decision observability triad for harness evolution |
| **Novikov et al. (2025)** | AlphaEvolve: evolutionary search with `# EVOLVE-BLOCK` markers & meta-prompt co-evolution |
| **Zhang et al. (2025)** | Darwin Gödel Machine: LLM coding agent modifies its own harness codebase |
| **Zhang et al. (2025)** | ACE: Agentic Context Engineering — generator, reflector, curator pattern |
| **Ye et al. (2026)** | MCE: Meta Context Engineering — bi-level skill evolution for context management |
| **Zelikman et al. (2023)** | STOP: Self-Taught Optimizer — recursive scaffolding improvement ($I_t = I_{t-1}(\hat{u}, I_{t-1}; M)$) |
| **Hu et al. (2025)** | ADAS: Automated Design of Agentic Systems — meta-agent proposes workflows in code |
| **Trehan & Chopra (2026)** | Six failure modes in LLM auto-research (drift, over-optimism, memory degradation) |
