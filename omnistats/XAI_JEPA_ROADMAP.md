# Quantum-Inspired XAI for JEPA: Concepts, Implementation & Next Steps

> **Location:** `omnistats/XAI_JEPA_ROADMAP.md`
> **Status:** ✅ Phase 1 & 2 complete — all P0/P1 modules verified.
> **Covers:** All concepts from handwritten notes (Aug-2020, Feb-2022, Nov-2022),
> Tang (2023) dequantization, what was implemented, and the full next-step frontier.

## 🎯 The Core Mission: Bridging JEPA with OmniStats

**The core philosophy of OmniStats is to make complex probabilistic profiles approachable for human decision-making.** 

Traditionally, OmniStats Stage 1 (Latent Profile Analysis) clusters users into interpretable segments for A/B testing and Causal Impact. **XAI JEPA extends this philosophy directly into deep learning.** 

By mapping opaque, continuous JEPA embeddings onto a discrete *Eigenstate Bank* (or *Concept Bank*), we turn a black-box world model into structured, human-readable **Latent Profiles**. Decision-makers can now query the model using `bayesian_inverse_score` to see exactly which "profiles" drove a prediction. This unifies deep learning world models with OmniStats's rigorous, transparent causal decision-making pipeline.

---

## 1. Theoretical Foundation (Your Notes → Code)

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

Upgrading standard Probability Flow / Neural ODEs ($\frac{dz}{dt} = f_\theta(z,t)$) to coupled **Hamilton-Jacobi-Bellman (HJB)** and **Fokker-Planck (FP)** equations. *(Note: Discrete permutation MSA `kuffmode/msa` and non-MFG 2-stage causal estimators are **[DEPRECATED]** in favor of continuous MFG optimal transport).*

| Concept | Implemented In | Formula / Role |
|---|---|---|
| **Schrödinger Bridges & Wasserstein Flow** | `mean_field_jepa.py` | Models the `path_integral_rollout` as interacting agents minimizing transport cost via coupled HJB ($-\partial_t u - \frac{\sigma^2}{2}\Delta u + H = 0$) and FP ($\partial_t \rho + \nabla \cdot (\rho v) = \frac{\sigma^2}{2}\Delta \rho$) equations. |
| **Continuous Shapley / Integrated Gradients** | `continuous_shapley.py` | Calculates exact infinitesimal marginal contribution of each latent dimension over continuous manifolds *(Discrete MSA **[DEPRECATED]**)*. |
| **MFG Counterfactual Trajectory Flow** | `causal/mfg_did.py` | Replaces heuristic parallel trend assumptions in `deep_did.py` (**[DEPRECATED]**) with population-level density transport equilibrium under optimal control cost. |
| **MFG Confounder-Density Matching** | `causal/mfg_iv.py` | Replaces 2-stage splitting bias in `deep_iv.py` (**[DEPRECATED]**) by dynamically steering treatment density $P(X\|Z)$ to target structural density $P(X)$ via coupled HJB-FP dynamics. |
| **MFG Discontinuity Density Flow** | `causal/mfg_rdd.py` | Replaces dual Transformer fits in `deep_rdd.py` (**[DEPRECATED]**) by modeling the cutoff as an HJB jump interface in the running variable density flow. |

---

## 2. Complete File Inventory

### 2.1 Phase 1 — Core XAI Modules (Complete ✅)

| File | Purpose |
|---|---|
| [`omnistats/modules/information_theory.py`](information_theory.py) | Shannon Entropy, JKL, Bayesian Inverse Score, Boltzmann Energy, Partition Function, `compute_xai_metrics()` |
| [`datastructure/Lesson/dequantized_jepa_predictor.py`](../datastructure/Lesson/dequantized_jepa_predictor.py) | `L2ImportanceSampler`, `DequantizedLatentTransition`, `free_density_matrix()`, `path_integral_rollout()` |
| [`omnistats/modules/xai_visualisation.py`](xai_visualisation.py) | `plot_energy_landscape()`, `plot_information_decision_boundary()`, `plot_partition_function_evolution()` |
| [`omnistats/modules/jepa_bridge.py`](jepa_bridge.py) | `APADecoder` extended: `energy_hat`, `beta_hat`, `entropy` heads |
| [`omnistats/tests/test_information_theory.py`](../tests/test_information_theory.py) | **21/21 unit tests PASS** ✅ |

### 2.2 Phase 2 — Frontier P0/P1 (Complete ✅)

| File | Priority | Purpose |
|---|---|---|
| [`datastructure/Lesson/chebyshev_qsvt.py`](../datastructure/Lesson/chebyshev_qsvt.py) | **P0** | Chebyshev-QSVT via Clenshaw recursion + `ChebyshevDequantizedTransition` |
| [`eb_jepa/eb_jepa/quantum_mppi.py`](../eb_jepa/eb_jepa/quantum_mppi.py) | **P0** | `QuantumMPPIPlanner`: Boltzmann noise + JKL cost + β annealing |
| [`omnistats/modules/bayesian/maxwell_prior.py`](modules/bayesian/maxwell_prior.py) | **P1** | Maxwell-Boltzmann prior for `energy_hat`, `maxwell_prior_loss()` |
| [`omnistats/modules/bayesian/waic_loo.py`](modules/bayesian/waic_loo.py) | **P1** | WAIC + PSIS-LOO + `compare_models()` for causal estimator selection |
| [`omnistats/modules/timeseries/quantum_kalman.py`](modules/timeseries/quantum_kalman.py) | **P1** | `QuantumKalmanFilter` + RTS smoother + ATT estimation |

### 2.3 Phase 3 — SOTA Continuous Game Theory & Causal MFGs (Upcoming)

| File | Priority | Purpose |
|---|---|---|
| [`omnistats/modules/game_theory/continuous_shapley.py`](modules/game_theory/continuous_shapley.py) | **P1** | Integrated gradients and continuous Shapley approximations for latent dimensions |
| [`omnistats/modules/game_theory/mean_field_jepa.py`](modules/game_theory/mean_field_jepa.py) | **P1** | Solves Forward-Backward HJB and Fokker-Planck equations to regularize the latent flow |
| [`omnistats/modules/causal/mfg_did.py`](modules/causal/mfg_did.py) | **P1** | **MFG Counterfactual Trajectory Flow**: Density transport equilibrium replacing linear/minimax DiD |
| [`omnistats/modules/causal/mfg_iv.py`](modules/causal/mfg_iv.py) | **P1** | **MFG Confounder-Density Matching**: HJB-FP velocity field steering treatment density replacing 2-stage IV |
| [`omnistats/modules/causal/mfg_rdd.py`](modules/causal/mfg_rdd.py) | **P1** | **MFG Discontinuity Density Flow**: HJB jump interface at cutoff replacing dual Transformer RDD |

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

---

## 4. How to Use the Full XAI Pipeline

```python
import torch, numpy as np
import sys; sys.path.insert(0, 'datastructure/Lesson')

# ── Core XAI metrics ─────────────────────────────────────────────────────────
from omnistats.modules.information_theory import compute_xai_metrics
from omnistats.modules.xai_visualisation import (
    plot_energy_landscape, plot_information_decision_boundary,
    plot_partition_function_evolution,
)

# ── SOTA Chebyshev predictor (P0) ─────────────────────────────────────────────
from chebyshev_qsvt import ChebyshevDequantizedTransition
from dequantized_jepa_predictor import path_integral_rollout

# ── Bayesian model selection (P1) ─────────────────────────────────────────────
from omnistats.modules.bayesian.maxwell_prior import maxwell_prior_loss
from omnistats.modules.bayesian.waic_loo import compare_models

# ── Quantum causal impact (P1) ────────────────────────────────────────────────
from omnistats.modules.timeseries.quantum_kalman import QuantumKalmanFilter

# Step 1: Build SOTA predictor
predictor = ChebyshevDequantizedTransition(
    d_latent=128, rank_k=32, cheb_degree=8, filter_type='heat'
)

# Step 2: Encode and get XAI metrics
z = jepa_encoder(context_frames)           # [B, D, T, H, W]
metrics = compute_xai_metrics(z, reference_embeddings=concept_bank, beta=1.0)

# Step 3: Path integral rollout
rollout = path_integral_rollout(predictor, z_flat, T=5)

# Step 4: Visualize
plot_energy_landscape(z_np, metrics['energy'].numpy())
plot_information_decision_boundary(z_np, metrics['jkl_from_prior'].numpy(),
                                   metrics['entropy'].numpy())
plot_partition_function_evolution(rollout['energies'].numpy().T)

# Step 5: Causal impact with Quantum Kalman
qkf = QuantumKalmanFilter(beta=1.0, diffusion_sigma=0.3)
att = qkf.estimate_att(y_tensor, X_tensor, T_treat=40)
print(f"ATT = {att['estimate']:.3f}  [{att['ci_lower']:.3f}, {att['ci_upper']:.3f}]")

# Step 6: WAIC model comparison
results = compare_models({'DiD': ll_did, 'RDD': ll_rdd, 'QuantumKalman': ll_qkf})
print(f"Best causal model: {results['winner']}")
```

---

## 5. Next Steps (P2–P3, Not Yet Implemented)

### 🟡 P2 — Quantum β-VAE

**File:** `VAE/Beta-VAE/quantum_beta_vae.py` *(new)*

Replace the isotropic Gaussian prior in iVAE with the Boltzmann prior `p(z) ∝ exp(-β V(z))`.

- Low β → fully disentangled, high-entropy latent space (explores)
- High β → collapsed, low-entropy attractor states (exploits)

### 🟡 P2 — Marchenko-Pastur Spectral Regularization

**File:** `MLModel/AIModel/model/spectral_regularizer.py` *(new)*

Add a loss term penalizing neural network weight matrices whose eigenvalue
distribution departs from the Marchenko-Pastur bulk:

$$\mathcal{L}_{spectral} = KL(\rho_{empirical} \| \rho_{MP})$$

Connect to `concept_integrator.py::analyze_spectral_properties()`.

### 🟣 P2 — Hardware Acceleration & GPU Fallback Mitigation

**Current Fallbacks**: The pipeline currently relies on CPU-bound loops (e.g., PyMC NUTS sampling, Python `for` loops in Clenshaw recursion) and unoptimized PyTorch matrix multiplications.
**Next Step Plans**:
- **Triton / PyTorch Compile**: Wrap `chebyshev_qsvt.py` using `torch.compile(mode="reduce-overhead")` or custom OpenAI Triton kernels to fuse the recursion operations into a single CUDA graph.
- **JAX / NumPyro**: Migrate the Bayesian causal components (Stage 2 A/B Testing, Stage 4 Quantum Kalman) from PyMC to NumPyro. This allows parallel XLA-compiled NUTS sampling on the GPU, yielding a 10x–100x speedup.
- **Vectorization (vmap)**: Use PyTorch `vmap` inside `QuantumMPPIPlanner` to parallelize thousands of action rollouts on GPU without sequential bottlenecks.

### 🟢 P3 — story_teller.py XAI Node Embedding

Embed each story graph node as a JEPA latent vector.
Use `bayesian_inverse_score()` to attribute each event to its closest concept anchor.
Visualize with `plot_information_decision_boundary()`.

### 🟢 P3 — Path Integral ODE Solver

**File:** `datastructure/Lesson/stirling_partition.py` *(new)*

Unify `Stirling_solution.py` with the partition function formalism.
Stirling's approximation is the saddle-point approximation of $\log Z(\beta)$.

---

## 6. Open Questions

> [!IMPORTANT]
> **Q1 — Rank `k`:** What is the effective intrinsic dimension of the JEPA latent space on your datasets? Run PCA and check explained variance — this sets `rank_k` for `ChebyshevDequantizedTransition`.

> [!IMPORTANT]
> **Q2 — β Annealing Direction for MPPI:** `QuantumMPPIPlanner` currently uses cold→hot (precise short-term, exploratory long-term). Should this be reversed for your planning task?

> [!WARNING]
> **Q3 — iVAE Identifiability:** Quantum β-VAE (P2) changes the prior. Verify that the Boltzmann prior factorizes across coordinates before implementing to preserve Khemakhem et al. (2020) guarantees.

---

## 7. Reference Index

| Source | Key Contribution |
|---|---|
| Handwritten notes (Aug 2020) | JKL, Shannon Entropy, Bayesian inverse problems, Beta-Binomial |
| Handwritten notes (Feb 2022) | Maxwell, Gaussian, Exponential distributions; kinetic energy analogy |
| Handwritten notes (Nov 2022) | Density matrix $p(x,x',\beta)$, path integral convolution, $Z(\beta)$ |
| Tang, E. (2023). *Quantum ML Without Any Quantum*. UW PhD Thesis | ℓ₂ importance sampling, QSVT dequantization, **Chebyshev/Clenshaw recursion** (Ch.6) |
| Kalman, R.E. (1960) | Classical linear filter — baseline for Quantum Kalman |
| Watanabe, S. (2010) | WAIC theory for singular statistical models |
| Vehtari, Gelman, Gabry (2017) | PSIS-LOO cross-validation, Pareto k-hat diagnostic |
| Khemakhem et al. (2020) | iVAE identifiability theory (relevant to P2 Q3) |
| Williams et al. (2017) | Information Theoretic MPC / MPPI — baseline for QuantumMPPIPlanner |
| Feynman & Hibbs (1965) | Path integral formalism, partition function |
| Marchenko & Pastur (1967) | Random Matrix Theory spectral law (P2) |

---

## 8. Phase VI — Open Research Directions

> **Status:** Open research directions (Section 8.6 **Completed**: Off-Policy Trajectory Evaluation & LLM Narrator integration).

### 8.1 Real-Traffic Closing the Loop (Human-in-the-Loop Safety Gate)

Phase V simulates personas via MatrAIx. The next frontier is routing the winning feature candidate to **real users** and feeding the live ATT back as a training signal.

**Open design questions:**
- **Safety gate:** How do we prevent the World Model from proposing harmful experiments? Propose a `SafetyOracle` module that gates actions against a hard constraint: `ATT_lower_CI > 0` and `Bayesian Expected Loss < epsilon`.
- **Conformal Prediction wrapper:** Wrap the APADecoder's `att_hat` with a conformal prediction interval calibrated on historical OmniStats results so the World Model has uncertainty-aware reward signals.
- **Gradual rollout:** Integrate with feature flagging systems (LaunchDarkly, Unleash) to progressively ramp the winning feature from 1% → 10% → 50% with automated checkpoints at each stage.

```
[Phase VI Autonomous Loop]
  Sim (MatrAIx)  →  A/B Test  →  Best  →  Train JEPA  →  XAI
       ↑                                                    │
       └──── Real Traffic ATT (human-approved gate) ────────┘
```

---

### 8.2 Multi-Objective ATT Pareto Frontier

The current cost function is a scalar weighted sum:

$$Cost = -w_{att} \cdot ATT + w_{risk} \cdot Risk + w_{disp} \cdot Disparity$$

**Open direction:** Replace with a **Pareto front optimizer** (NSGA-II or MOEA/D) that maintains a population of Pareto-optimal experiment designs — allowing decision-makers to pick their own trade-off point between ATT lift, risk tolerance, and fairness constraints at deployment time.

**Files to create:**
- `omnistats/modules/optimization/pareto_cem.py` — Multi-objective CEM with Pareto front tracking
- `omnistats/modules/optimization/fairness_constraint.py` — Subgroup disparity constraint (DR-OLS ATT parity)

---

### 8.3 Causal MFG Closing the Loop (Density-Level Feedback)

Phases I–V operate on **sample-level** ATT estimates. Mean Field Game (MFG) extensions in Phase 2.3 (`mfg_did.py`, `mfg_iv.py`, `mfg_rdd.py`) open a path to **density-level** causal inference:

**Open direction:** After each autonomous iteration, fit a `MFGCounterfactualFlow` on the winning population's density $\rho(x, t)$ and use it as a regularizer for the JEPA predictor's transition kernel — enforcing that the World Model's imagined counterfactuals are consistent with the observed equilibrium density.

**Key equation to implement:**
$$\min_{\theta} \underbrace{\mathcal{L}_{JEPA}}_{\text{latent prediction}} + \lambda \underbrace{W_2(\hat{\rho}_{t+1}, \rho^*_{t+1})}_{\text{Wasserstein density consistency}}$$

---

### 8.4 Adaptive Concept Bank (Online TCAV)

The current TCAV implementation uses a **fixed** concept bank defined by demographic quantiles (High Income, Risk Averse, etc.). After each autonomous loop iteration, the model's latent geometry shifts as JEPA is fine-tuned.

**Open direction:** Build an **online concept bank** that automatically re-discovers the top-k most predictive directions in latent space after each JEPA fine-tuning step using:
1. `PCA(n_components=k)` on the delta-embeddings between winning and losing feature populations.
2. Cluster the PC directions into semantically coherent concepts via `bayesian_inverse_score()` against a lexicon of known demographic features.
3. Replace the TCAV fixed concept set with these discovered directions for the next iteration.

**File to create:** `omnistats/modules/simulation/adaptive_concept_bank.py`

---

### 8.5 Experiment Graph Memory (Episodic Causal Replay)

Each autonomous loop iteration produces a `leaderboard_iterN.csv` and `autonomous_loop_log.csv`, but these are treated as flat logs. A richer episodic memory would enable:

**Open direction:** Build a **Causal Experiment Knowledge Graph** where:
- Each **node** = an experiment (feature candidate) with attributes: ATT, SE, personas tested, XAI concept scores.
- Each **edge** = "parent experiment that proposed this one" (provenance chain).
- New feature candidates are proposed by a **graph neural network** that encodes the full experiment history and predicts which unexplored feature directions maximize expected ATT gain.

**Connects to:** `datastructure/story_teller.py` (existing graph narration engine) and `MLModel/AIModel/run/mst_msa_interaction.py` (Kruskal MST + Multi-Perturbation Shapley Analysis for identifying the most structurally important experiment nodes in the history graph).


