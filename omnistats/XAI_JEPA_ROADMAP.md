# XAI-JEPA + XAI-DoE + Harness — OmniStats Unified Roadmap

> **Status key:** ✅ Done · 🔨 In Progress · 🟡 Next · 🔵 Future

---

## Phase 1 — Core XAI Foundation ✅

**Goal:** Make JEPA embeddings explainable via information theory.

| Module | Status | Key Deliverable |
|---|---|---|
| `modules/information_theory.py` | ✅ | Shannon entropy, JKL, Bayesian inverse score, Boltzmann energy |
| `modules/xai_visualisation.py` | ✅ | Energy landscape, JKL decision boundary, partition function plots |
| `modules/jepa_bridge.py` (APADecoder) | ✅ | `energy_hat`, `beta_hat`, `entropy` heads on top of ATT decoder |
| `datastructure/Lesson/dequantized_jepa_predictor.py` | ✅ | `L2ImportanceSampler`, `free_density_matrix()`, `path_integral_rollout()` |
| `tests/test_information_theory.py` | ✅ | 21/21 unit tests passing |

**Math:** Concept bank maps $z \in \mathbb{R}^D$ → human-readable latent profiles via $\ell_2$ Bayesian inverse score. Partition function $Z(\beta) = \sum_n e^{-\beta E_n}$ tracks global latent energy.

---

## Phase 2 — SOTA Quantum Predictors ✅

**Goal:** Replace standard JEPA transition with physics-inspired SOTA predictors.

| Module | Status | Key Deliverable |
|---|---|---|
| `datastructure/Lesson/chebyshev_qsvt.py` | ✅ | Clenshaw recursion + `ChebyshevDequantizedTransition` (heat / low-pass filter) |
| `eb_jepa/eb_jepa/quantum_mppi.py` | ✅ | `QuantumMPPIPlanner`: Boltzmann noise + JKL cost + β annealing |
| `modules/bayesian/maxwell_prior.py` | ✅ | Maxwell-Boltzmann prior for `energy_hat`, `maxwell_prior_loss()` |
| `modules/bayesian/waic_loo.py` | ✅ | WAIC + PSIS-LOO + `compare_models()` |
| `modules/timeseries/quantum_kalman.py` | ✅ | `QuantumKalmanFilter` + RTS smoother + ATT estimation |

**Math core:**
$$Q_\text{quantum} = 2\sigma^2\beta \cdot e^{-\frac{\beta}{2}V(x_t)} \cdot I \qquad \text{(high-energy state = more uncertainty)}$$

---

## Phase 3 — Design of Experiments (DoE) 🔨

**Goal:** Replace ad-hoc A/B with rigorous multi-factor experimental designs from the Gendex/VIASM workshop.

**Milestone:** `modules/doe/` package scaffolded and integrated into `experiment_design.py`.

| Module | Status | Key Deliverable |
|---|---|---|
| `modules/doe/definitive_screening.py` | 🟡 | DSD: screen $m$ factors in $2m+1$ runs, zero main-effect / 2FI aliasing |
| `modules/doe/alpha_design.py` | 🟡 | Resolvable incomplete block schedules; minimises inter-block variance |
| `modules/doe/row_column_design.py` | 🟡 | 2D nuisance blocking (cohort × time) — `RCDBuilder`, `ResolvableRCDBuilder` |
| `modules/doe/response_surface.py` | 🟡 | CCD / BBD / SCD / APD builders + stationary point $x^* = -\tfrac{1}{2}B^{-1}b$ |
| `modules/doe/sod_optimizer.py` | 🟡 | Swap-permutation search minimising orthogonality loss $f$ and $g$ |
| `modules/doe/xai_doe_bridge.py` | 🔵 | DSD / RSM orthogonal grid in JEPA latent space (replaces random noise probing) |

**Sequential pipeline:**
```
DSD screen (2m+1 runs)
    → Alpha / RCD block deployment
        → RSM surface fit & x* optimum
```

---

## Phase 4 — Mean Field Games & Causal MFGs 🟡

**Goal:** Replace heuristic causal estimators (DiD, IV, RDD) with density-level optimal transport.

| Module | Status | Key Deliverable |
|---|---|---|
| `modules/game_theory/mean_field_jepa.py` | 🟡 | Solve coupled HJB + Fokker-Planck for latent flow regularisation |
| `modules/game_theory/continuous_shapley.py` | 🟡 | Integrated gradients / continuous Shapley over latent manifold |
| `modules/causal/mfg_did.py` | 🟡 | Wasserstein counterfactual trajectory flow (replaces linear DiD) |
| `modules/causal/mfg_iv.py` | 🟡 | HJB-FP velocity steering of treatment density (replaces 2-stage IV) |
| `modules/causal/mfg_rdd.py` | 🟡 | HJB jump interface at running-variable cutoff (replaces dual-Transformer RDD) |

**Math core:**
$$\min_\theta \underbrace{\mathcal{L}_\text{JEPA}}_\text{latent prediction} + \lambda\underbrace{W_2(\hat\rho_{t+1},\rho^*_{t+1})}_\text{Wasserstein density consistency}$$

---

## Phase 5 — PINN Safety & Computer-Assisted Verification 🟡

**Goal:** Prove mathematically that autonomous rollouts cannot explode or enter unsafe latent regions.

| Module | Status | Key Deliverable |
|---|---|---|
| `modules/simulation/pinn_safety_oracle.py` | 🟡 | Interval enclosures + Neumann contraction bound $r < 1$ per trajectory |
| `latent_ode_jepa.py` (gradient-norm re-weighting) | 🟡 | $w_k = \frac{1}{\|\nabla_z \mathcal{R}_k\| + \epsilon}$ loss weights for sharp-gradient ODE regions |

**Gate rule (required before real traffic):** `ATT_lower_CI > 0` **AND** Neumann radius $r < 1$.

---

## Phase 6 — Autonomous Harness (RSI Loop) 🔵

**Goal:** Treat the pipeline code itself as an optimizable artifact (Weng 2026 harness engineering).

### 6.1 Harness Modules

| Module | Key Role |
|---|---|
| `modules/harness/oahl_controller.py` | Goal-directed loop: PLAN → EXECUTE → OBSERVE → IMPROVE |
| `modules/harness/subagent_manager.py` | Launch / poll / merge parallel sub-agents via file-system status |
| `modules/harness/self_harness_evolver.py` | Weakness mining → bounded edit → regression validation |
| `modules/harness/workflow_evolver.py` | AlphaEvolve `# EVOLVE-BLOCK` search over DoE-JEPA workflow configs |
| `modules/harness/service_agent_bridge.py` | ADK/MCP RAG enrichment of DSD factors from `service-agent/` |
| `modules/harness/growmetrics_bridge.py` | A2A XAI briefing video delegation to `growmetrics/` |

### 6.2 Persistent Harness Memory (file-system, never in context)

```
omnistats/harness/memory/
├── context_playbook.json     # ACE/MCE itemised bullet context — curator never rewrites full blob
├── experiment_graph.json     # Provenance DAG: node = experiment, edge = parent proposal
├── att_timeline.parquet      # Full ATT history for Quantum Kalman pre-period calibration
├── doe_schedules/            # Per-iteration DSD / Alpha / RCD layouts
├── safety_certificates/      # PINN interval enclosures per rollout
└── failure_traces/           # Structured JSON: {stage, root_cause, causal_status}
```

### 6.3 Sub-Agent Registry

| Sub-Agent | Project | Trigger | Output |
|---|---|---|---|
| **service-agent** | `service-agent/` | Pre-iteration | Enriched DSD factor boundaries from ChromaDB RAG |
| **growmetrics-agent** | `growmetrics/` | Post-iteration | `.mp4` XAI briefing (energy landscape + RSM contour + ATT chart) |
| **jepa-trainer** | `eb_jepa/` | Post-ATT approval | Updated JEPA weights + refreshed Concept Bank |
| **doe-optimizer** | `modules/doe/` | Post-RSM fit | Next-iteration DSD / Alpha / RCD schedule from $x^*$ |

### 6.4 Self-Harness Safety Boundaries (read-only surfaces — cannot be edited by evolver)
- PINN safety oracle
- Human-in-the-loop ATT approval gate
- Base JEPA model weights
- `run_autonomous_loop.py` main entry point

### 6.5 Failure Modes & Safeguards

| Failure | Safeguard |
|---|---|
| Over-optimism / p-hacking | PINN oracle: `ATT_lower_CI > 0` mandatory |
| Reward hacking | Safety oracle + verifier are read-only harness surfaces |
| Memory degradation | File-system harness memory; no state in LLM context |
| Diversity collapse | Cosine-similarity rejection sampling (ShinkaEvolve pattern) |
| RSM saddle point | Auto-trigger `ridge_analysis()` on mixed-sign Hessian |
| RAG returns empty | `service-agent` re-syncs Drive folder, re-indexes APA reports |

---

## Development Checklist

```
Phase 1  ✅  information_theory · xai_visualisation · jepa_bridge · dequantized_jepa
Phase 2  ✅  chebyshev_qsvt · quantum_mppi · maxwell_prior · waic_loo · quantum_kalman
Phase 3  🔨  doe/definitive_screening · alpha_design · row_column_design
         🟡  doe/response_surface · sod_optimizer · xai_doe_bridge
Phase 4  🟡  mean_field_jepa · continuous_shapley · mfg_did · mfg_iv · mfg_rdd
Phase 5  🟡  pinn_safety_oracle · latent_ode gradient-norm reweighting
Phase 6  🔵  harness/ · subagent_manager · service_agent_bridge · growmetrics_bridge
              self_harness_evolver · workflow_evolver
```

---

## Key References

| Source | Contribution |
|---|---|
| Tang (2023) | ℓ₂ importance sampling, Chebyshev/Clenshaw QSVT dequantization |
| Nguyen (2018) VIASM | DSD, Alpha, IBD, RCD, RSM — Gendex DoE suite |
| Jones & Nachtsheim (2011) | Definitive Screening Designs |
| Box & Wilson (1951) | Central Composite Designs & RSM |
| Kalman (1960) | Classical filter — Quantum Kalman baseline |
| Buckmaster et al. (2026) | PINNs + gradient-norm re-weighting + computer-assisted proofs |
| Weng (2026) | Harness engineering for recursive self-improvement |
| Zhang et al. (2026) | Self-Harness: weakness mining → bounded edit → validation |
| Lin et al. (2026) | AHE: component / experience / decision observability |
| Novikov et al. (2025) | AlphaEvolve: `# EVOLVE-BLOCK` evolutionary harness search |
| Zelikman et al. (2023) | STOP: recursive scaffolding self-improvement |
