Read [index.md](index.md) for the full library tour. This README tracks major milestones.

---

## 🚀 Latest: Quantum-Inspired XAI for JEPA (Phase 1 & 2 — Complete)

The repo now includes a full **Quantum-Inspired Explainable AI (XAI)** pipeline for the JEPA world model, grounded in Boltzmann thermodynamics, path integral theory (Nov-2022 notes), and Tang (2023) ℓ₂ dequantization.

See **[omnistats/XAI_JEPA_ROADMAP.md](omnistats/XAI_JEPA_ROADMAP.md)** for the complete concept map, architecture diagrams, usage examples, and P2–P3 next-step frontier.

### Phase 1 — Core XAI (✅ Complete — 21/21 unit tests pass)

| File | What It Does |
|---|---|
| [`omnistats/modules/information_theory.py`](omnistats/modules/information_theory.py) | Shannon Entropy, JKL Divergence, Bayesian Inverse Score via ℓ₂ sampling, Boltzmann Energy, Partition Function Z(β) |
| [`datastructure/Lesson/dequantized_jepa_predictor.py`](datastructure/Lesson/dequantized_jepa_predictor.py) | `DequantizedLatentTransition` — ℓ₂ importance sampling replaces deterministic MLP; `path_integral_rollout()` |
| [`omnistats/modules/xai_visualisation.py`](omnistats/modules/xai_visualisation.py) | Dark-mode XAI plots: Energy Landscape, JKL Decision Boundary, Partition Function evolution |
| [`omnistats/modules/jepa_bridge.py`](omnistats/modules/jepa_bridge.py) | `APADecoder` extended with `energy_hat`, `beta_hat`, `entropy` output heads |

### Phase 2 — Frontier P0 / P1 (✅ Complete — 5/5 verified)

| File | Priority | What It Does |
|---|---|---|
| [`datastructure/Lesson/chebyshev_qsvt.py`](datastructure/Lesson/chebyshev_qsvt.py) | **P0** | Chebyshev-QSVT (Tang Ch.6 Clenshaw recursion) — SOTA spectral heat-kernel / low-pass filter of JEPA eigenstate bank; `ChebyshevDequantizedTransition` |
| [`eb_jepa/eb_jepa/quantum_mppi.py`](eb_jepa/eb_jepa/quantum_mppi.py) | **P0** | `QuantumMPPIPlanner` — Boltzmann-weighted action noise + JKL cost regularization + β annealing schedule |
| [`omnistats/modules/bayesian/maxwell_prior.py`](omnistats/modules/bayesian/maxwell_prior.py) | **P1** | Maxwell-Boltzmann prior for `energy_hat`; `maxwell_prior_loss()` for APADecoder training |
| [`omnistats/modules/bayesian/waic_loo.py`](omnistats/modules/bayesian/waic_loo.py) | **P1** | WAIC + PSIS-LOO + `compare_models()` — SOTA Bayesian model selection replacing AIC/BIC |
| [`omnistats/modules/timeseries/quantum_kalman.py`](omnistats/modules/timeseries/quantum_kalman.py) | **P1** | `QuantumKalmanFilter` + RTS smoother — density-matrix prediction step; robust to non-Gaussian shocks |

---

### Class Imbalance Mitigation Notes
* **Imbalance Sweep**: SMOTEENN on Random Forest increased F1 from **0.2326 → 0.7119** (+206%). See [MLModel/run/README.md](MLModel/run/README.md).

---

### OmniStats: Causal & Experimentation Pipeline

[omnistats/](omnistats/) is a production-grade 5-stage pipeline:

1. **Stage 1 — LPA**: GMM user segmentation → `profile_prob_max` for variance reduction.
2. **Stage 2 — Sequential Bayesian A/B**: Beta-Binomial + StudentT on PyMC NUTS. Stops via Expected Loss.
3. **Stage 3 — CUPED**: Monotonic regression on LPA posteriors as pre-experiment covariates.
4. **Stage 4 — Causal Suite**: Staggered DiD · IV · RDD · SCM · Matrix Completion · **Quantum Kalman BSTS** ✅
5. **Stage 5 — APA Report**: APA 7th edition Word doc with **WAIC/LOO model selection** ✅.

**XAI:** `APADecoder` outputs `energy_hat`, `beta_hat`, Shannon Entropy, JKL — every decision grounded in measurable physical quantities.

**Next:** 
- **Hardware/GPU Scaling**: Migrate CPU fallbacks to GPU using `torch.compile` (Triton) for JEPA transitions, `vmap` for MPPI rollouts, and **JAX/NumPyro** for XLA-compiled Bayesian sampling.
- **Algorithms**: Quantum β-VAE · Marchenko-Pastur spectral regularizer · story_teller XAI embeddings.
→ See [omnistats/XAI_JEPA_ROADMAP.md §5](omnistats/XAI_JEPA_ROADMAP.md) for the full P2–P3 plan.

---

### MLModel: SOTA Convex Optimization

1. **FISTA** — L1 Sparse Logistic Regression, O(1/k²) convergence. See [fista_logistic.py](MLModel/model/fista_logistic.py).
2. **ADMM** — Hinge Loss SVM. See [admm_svm.py](MLModel/model/admm_svm.py).
3. **Marchenko-Pastur Spectral Regularization** *(P2 upcoming)* — penalize weight matrices deviating from the MP bulk.

---

### Legacy Teaching Notes

1) bai tap. mo folder bang commandline

2) excel : trình bày -> tìm cách : python : thực tiễn: implement

+ Đọc story_teller: biết thuật toán - data_structure

+ OOP : Object Oriented Programming - code SOLID 
  
+ Phân biệt: imperative # functional programming

+ tạo def temp_close_0 trong story_teller

+ summary: https://goodresearch.dev/decoupled.html

3) subscribe: https://www.dailycodingproblem.com/
+ sort and timing

4) Thuat toan:
+   https://github.com/TheAlgorithms/Python
+    https://github.com/keon/algorithms.git
   
5) Solid coding/ Clean code:
+ https://github.com/PacktPublishing/Clean-Code-in-Python-Second-Edition.git
+ https://github.com/mynameisfiber/high_performance_python_2e.git

6) ML/AI: 
+ https://probml.github.io/pml-book/
+ https://github.com/Atcold/pytorch-Deep-Learning.git
+ https://github.com/eriklindernoren/ML-From-Scratch.git
+ https://github.com/ageron/handson-ml2.git

Note: Do not duplicate story or lesson. Created.
