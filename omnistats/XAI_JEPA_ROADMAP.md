# Quantum-Inspired XAI for JEPA: Concepts, Implementation & Next Steps

> **Location:** `omnistats/XAI_JEPA_ROADMAP.md`
> **Covers:** All concepts from the handwritten notes (Aug-2020, Feb-2022, Nov-2022),
> what was implemented, and the full next-step frontier.

---

## 1. Theoretical Foundation (Your Notes → Code)

### 1.1 Probability → Decision → Information Theory (Aug 2020)

```
Probability ─┬─ Decision Boundary
             │    (y - t)^p
             └─ Bayesian
```

| Concept from Notes | Implemented In | Formula |
|---|---|---|
| **Partial Permutation** $P_n^k = \frac{n!}{(n-k)!}$ | `story_teller.py` combinatorics | Combinatorial state counting |
| **Regression vs Classification** | `MLModel/` | Real-valued vs discrete output |
| **Sample Efficiency** via representable priors | `omnistats/modules/bayesian/` | β-VAE + Maxwell prior |
| **Bayes inverse problem** $n \to y$ | `information_theory.py::bayesian_inverse_score()` | ℓ₂ posterior over concept bank |
| **Beta-Binomial** | `omnistats/modules/bayesian/beta_binomial.py` | Conjugate prior for proportions |
| **JKL Divergence** | `information_theory.py::jeffreys_kl()` | $J(p,q) = KL(p\|q) + KL(q\|p)$ |
| **Shannon: Uncertainty + KL = Entropy** | `information_theory.py::shannon_entropy()` | $H(p) = -\sum p \log p$ |
| **Forward: let $q(x)$ known → $p(x) \log \frac{p(x)}{q(x)}$** | `information_theory.py::kl_divergence()` | KL divergence |

### 1.2 Probability Distributions — Basics (Feb 2022)

```
Basics:
  Geometric dist:   X ~ Geom(p)  →  P(X) = 1/p
  Uniform:          [0,1],  x = 0.1, 0.25  →  P(x) ~ 0.003
  Exp:              e^{t/τ}, t ≥ 0, τ = 2/π  →  P(t) = e^{2/π}
  Gaussian:         e^{-x²/2σ²} / (√2πσ²)
  Maxwell:          p_max(v) = √(2/π)(v²/σ³)exp(-v²/2σ²)
  Kinetic energy:   E_k(v) = ½mv²  →  E(z) = ‖z‖²/β
```

| Distribution | Role in XAI Pipeline |
|---|---|
| **Geometric** | Prior for discrete latent modes (number of concept clusters) |
| **Gaussian** | Free-particle density matrix $p^{free}(z,z',\beta)$ |
| **Maxwell-Boltzmann** | **Next step:** prior on `energy_hat` in `APADecoder` |
| **Exponential** | Waiting time between high-energy events in latent trajectory |
| **Kinetic Energy** $E_k = \|z\|^2/\beta$ | `information_theory.py::boltzmann_energy()` |

### 1.3 Classical → Quantum Density Matrix (Nov 2022)

$$\pi(x,n) \propto e^{-\beta E_n} \psi_n(x) \psi_n^*(x)$$

The density matrix formalism:

$$p(x, x', \beta) = \sum_n e^{-\beta E_n} \psi_n(x) \psi_n^*(x')$$

| Theoretical Property | Implemented In | Notes |
|---|---|---|
| **Convolution (Path Integral):** $\int dx' p(x,x',\beta_1)p(x',x'',\beta_2) = p(x,x'',\beta_1+\beta_2)$ | `dequantized_jepa_predictor.py::path_integral_rollout()` | Multi-step = additive β |
| **Free density matrix** $p^{free}(z,z',\beta) \propto e^{-\|z-z'\|^2/4\sigma^2\beta}$ | `dequantized_jepa_predictor.py::free_density_matrix()` | Gaussian kernel in latent space |
| **High-Temperature limit** $p \approx e^{-\beta/2 V(x)} p^{free} e^{-\beta/2 V(x')}$ | `DequantizedLatentTransition.forward()` | `potential_weight × z_weighted` |
| **Partition Function** $Z(\beta) = \text{Tr}(\rho) = \sum_n e^{-\beta E_n}$ | `information_theory.py::partition_function()` | Log-space stable |
| **Boltzmann weights** $w_n = e^{-\beta E_n}/Z(\beta)$ | `DequantizedLatentTransition.boltzmann_weights()` | Softmax of $-\beta E_n$ |

### 1.4 Tang Dequantization (2023 Thesis)

> **Key insight:** Preparing a quantum state $|y\rangle$ using QSVT requires the same
> ℓ₂ sampling access as a classical sketch. Classical machines can simulate QSVT
> in `O(k · poly(log d))` time when the matrix has effective rank `k ≪ d`.

| Tang Concept | Implemented In |
|---|---|
| ℓ₂ importance sampling (SQ oracle) | `L2ImportanceSampler.sketch_matrix_vector()` |
| Low-rank matrix sketch | `DequantizedLatentTransition` (eigenstate bank, rank `k`) |
| Clenshaw recursion for matrix polynomials | **Next step** (Frontier 2) |
| Dequantized recommendation systems | **Analogy:** `bayesian_inverse_score()` = dequantized quantum retrieval |

---

## 2. What Was Implemented (Current State)

### 2.1 New Files

| File | Purpose |
|---|---|
| [`omnistats/modules/information_theory.py`](information_theory.py) | Shannon Entropy, JKL, Bayesian Inverse Score, Boltzmann Energy, Partition Function, `compute_xai_metrics()` |
| [`datastructure/Lesson/dequantized_jepa_predictor.py`](../datastructure/Lesson/dequantized_jepa_predictor.py) | `L2ImportanceSampler`, `DequantizedLatentTransition`, `free_density_matrix()`, `path_integral_rollout()`, `compute_partition_function()` |
| [`omnistats/modules/xai_visualisation.py`](xai_visualisation.py) | `plot_energy_landscape()`, `plot_information_decision_boundary()`, `plot_partition_function_evolution()` |
| [`omnistats/tests/test_information_theory.py`](../tests/test_information_theory.py) | 21 unit tests — all passing ✅ |

### 2.2 Modified Files

| File | Change |
|---|---|
| [`omnistats/modules/jepa_bridge.py`](jepa_bridge.py) | `APADecoder` extended: `energy_hat`, `beta_hat`, `entropy` output heads; graceful import of `compute_xai_metrics()` |

### 2.3 Architecture: APADecoder (Extended)

```
z [B, D_latent]
      │
   SharedNet (MLP + LayerNorm + GELU)
      │
   ┌──┴───────────────────────────────┐
   ▼           ▼           ▼          ▼
att_hat    risk_hat   energy_hat   beta_hat
[B]         [B]         [B]          [B]
(ATT)    (Softplus)  (Softplus)  (Softplus)
                     + entropy [B]
                     + jkl_from_prior [B]  ← if references given
                     + posterior [B, N]    ← Bayesian Inverse Score
```

### 2.4 Architecture: DequantizedLatentTransition

```
z_input [B, D]
      │
  PotentialNet → V(z) [B]       "Energy cost of state"
      │
  e^{-β/2 V(z)} weighting       "High-Temperature approx"
      │
  EigenstateBank [k, D]          "Low-rank spectral basis"
  (Boltzmann weighted: w_n = softmax(-β E_n))
      │
  L2ImportanceSampler            "Tang ℓ₂ dequantization"
      │
  TransitionHead [D+k → 2D]     "Mean + LogVar of next state"
      │
  z_next_mean [B,D] + z_next_logvar [B,D]
```

---

## 3. How to Use the XAI Pipeline

```python
import torch
from omnistats.modules.information_theory import compute_xai_metrics
from omnistats.modules.xai_visualisation import (
    plot_energy_landscape,
    plot_information_decision_boundary,
    plot_partition_function_evolution,
)
from datastructure.Lesson.dequantized_jepa_predictor import (
    DequantizedLatentTransition,
    path_integral_rollout,
    compute_partition_function,
)

# ── Step 1: Get latent embeddings from JEPA encoder ──────────────────────────
z = jepa_encoder(context_frames)              # [B, D, T, H, W]

# ── Step 2: Compute full XAI metrics ─────────────────────────────────────────
concept_bank = torch.load("concept_anchors.pt")   # [N, D] labelled concepts
metrics = compute_xai_metrics(z, reference_embeddings=concept_bank, beta=1.0)

print(metrics["energy"])         # [B] — how "activated" is each sample?
print(metrics["entropy"])        # [B] — how uncertain is each prediction?
print(metrics["jkl_from_prior"]) # [B] — how confident is the model?
print(metrics["posterior"])      # [B, N] — which concept does z correspond to?

# ── Step 3: Dequantized rollout ───────────────────────────────────────────────
predictor = DequantizedLatentTransition(d_latent=D, rank_k=32)
rollout = path_integral_rollout(predictor, z_flat, T=5)

# ── Step 4: Visualize ─────────────────────────────────────────────────────────
z_np      = z_flat.detach().numpy()
energy_np = metrics["energy"].detach().numpy()
jkl_np    = metrics["jkl_from_prior"].detach().numpy()
entropy_np = metrics["entropy"].detach().numpy()

plot_energy_landscape(z_np, energy_np)
plot_information_decision_boundary(z_np, jkl_np, entropy_np)

energy_traj = rollout["energies"].detach().numpy()  # [B, T]
plot_partition_function_evolution(energy_traj.T)    # [T, B]
```

---

## 4. Next Steps (Ordered by Priority)

### 🔴 P0 — Chebyshev QSVT for the Dequantized Predictor

**File:** `datastructure/Lesson/dequantized_jepa_predictor.py`

Replace the `TransitionHead` MLP with a **Chebyshev polynomial of the eigenstate matrix**,
implemented via Clenshaw recursion (Tang Ch. 6).

```python
# Instead of:  out = self.transition_head(cat([z, scores]))
# Use:         out = chebyshev_sketch(A=self.eigenstates, v=z_weighted, degree=k)
```

**Why:** Matrix polynomials via Chebyshev allow arbitrary spectral filtering —
low-degree = smooth (explore), high-degree = sharp (exploit high-energy modes).

---

### 🔴 P0 — Quantum-Augmented MPPI Planner in EB-JEPA

**File:** `eb_jepa/eb_jepa/planning.py`

- Replace Gaussian noise `ε ~ N(0, σ²)` in MPPI with **Boltzmann-weighted noise**:
  `ε ~ softmax(-β E_n) · ψ_n` from the eigenstate bank
- Add JKL regularization to the MPPI cost:
  `cost = prediction_error + λ · JKL(q(z_next) || Boltzmann_prior)`
- β anneals over the planning horizon: `β(t) = β₀ × (T-t)/T`

---

### 🟠 P1 — Maxwell-Boltzmann Prior for `energy_hat`

**File:** `omnistats/modules/bayesian/maxwell_prior.py` *(new)*

```python
def maxwell_log_prob(energy: torch.Tensor, sigma: float) -> torch.Tensor:
    """log p(E) where E ~ Maxwell-Boltzmann with scale sigma."""
    # p(v) ∝ v² exp(-v²/2σ²) → for energy E = v², p(E) ∝ √E exp(-E/2σ²)
    ...
```

Set as the default prior for `energy_hat` in `APADecoder` to enforce physical realism.

---

### 🟠 P1 — WAIC / LOO for Bayesian Model Selection

**File:** `omnistats/modules/bayesian/waic_loo.py` *(new)*

```python
def waic(log_likelihoods: np.ndarray) -> dict:
    """Watanabe-Akaike IC: WAIC = -2(lppd - p_waic)"""
    ...

def loo_cv(log_likelihoods: np.ndarray) -> dict:
    """Leave-One-Out CV via Pareto-smoothed importance sampling (PSIS-LOO)."""
    ...
```

Feed into `omnistats/modules/apa_report.py` to auto-select the best causal estimator.

---

### 🟠 P1 — Quantum Kalman Filter in `causal_impact.py`

**File:** `omnistats/modules/timeseries/causal_impact.py`

Replace standard Kalman prediction step with the density matrix propagator:

```
Standard Kalman:    x_{t+1} = F x_t + noise
Quantum Kalman:     p(x_{t+1} | x_t) via p(x, x', β/N) convolution
```

High-temperature limit makes it robust to non-Gaussian shocks.

---

### 🟡 P2 — Quantum β-VAE (`VAE/Beta-VAE/`)

Replace the isotropic Gaussian prior `p(z) = N(0, I)` with the Boltzmann prior
`p(z) ∝ exp(-β V(z))` from `DequantizedLatentTransition.potential_net`.

**Physical interpretation of β:**
- `β → 0` (hot): fully disentangled, high-entropy representation (explores latent space)
- `β → ∞` (cold): collapsed, low-entropy attractor states (pure exploitation)

---

### 🟡 P2 — Marchenko-Pastur Spectral Regularization (`MLModel/`)

Add a loss term to any neural network that penalizes weight matrices whose
eigenvalue spectrum departs from the Marchenko-Pastur bulk:

```
L_spectral = KL(ρ_empirical || ρ_Marchenko-Pastur)
```

Eigenvalues outside the bulk = learned signal. Those inside = noise.
Connect to `concept_integrator.py::analyze_spectral_properties()`.

---

### 🟢 P3 — story_teller.py XAI Node Embedding

Embed each story graph node as a JEPA latent vector.
Use `bayesian_inverse_score()` to attribute each narrative event to its
closest concept anchor. Store `energy_hat` and `entropy` as node attributes.
Visualize with `plot_information_decision_boundary()`.

---

## 5. Open Questions

> **Q1 — Rank `k`:** What is the effective intrinsic dimension of the JEPA
> latent space on your datasets? This sets the `rank_k` budget for ℓ₂ sampling.
> Measure: PCA explained variance ratio, or the Marchenko-Pastur fit.

> **Q2 — β schedule:** For MPPI, should β anneal cold→hot (exploit→explore)
> or hot→cold (explore→exploit)? Classical annealing does hot→cold,
> but quantum speedup arguments favor the reverse.

> **Q3 — iVAE identifiability:** The Quantum β-VAE changes the prior.
> Does this preserve the identifiability guarantee of Khemakhem et al. (2020)?
> Likely yes if the Boltzmann prior factorizes across coordinates, but
> needs formal verification before implementation.

---

## 6. Reference Index

| Source | Key Contribution |
|---|---|
| Handwritten notes (Aug 2020) | JKL, Shannon Entropy, Bayesian inverse problems, Beta-Binomial |
| Handwritten notes (Feb 2022) | Maxwell, Gaussian, Exponential, Geometric distributions; kinetic energy analogy |
| Handwritten notes (Nov 2022) | Density matrix $p(x,x',\beta)$, path integral convolution, high-temperature limit, partition function $Z(\beta)$ |
| Tang, E. (2023). *Quantum ML Without Any Quantum*. UW PhD Thesis | ℓ₂ importance sampling, QSVT dequantization, Chebyshev recursion, rank-k approximation |
| Khemakhem et al. (2020). *Variational Autoencoders and Nonlinear ICA* | iVAE identifiability theory |
| Feynman & Hibbs (1965). *Quantum Mechanics and Path Integrals* | Path integral formalism, partition function |
| Marchenko & Pastur (1967). *Distribution of eigenvalues...* | Random Matrix Theory spectral law |
