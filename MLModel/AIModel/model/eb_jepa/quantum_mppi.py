"""
MLModel/AIModel/model/eb_jepa/quantum_mppi.py
----------------------------------------------
Quantum-Augmented MPPI Planner.

Subclasses MPPIPlanner from planning.py, replacing:
  - Gaussian noise perturbations  → Boltzmann-weighted noise from eigenstate bank
  - Standard MPPI cost            → cost + JKL regularization
  - Fixed temperature             → annealing beta schedule over planning horizon

Theoretical grounding
---------------------
Standard MPPI samples action perturbations from N(0, sigma^2).
Quantum MPPI samples from the Boltzmann distribution of the eigenstate bank:
    epsilon_t ~ softmax(-beta_t * E_n) @ psi_n   (superposition of eigenstates)

This implements the path integral control formulation where the optimal
controller minimizes free energy F = E - (1/beta) * H (energy minus entropy),
exactly matching the Feynman-Hibbs partition function:
    Z(beta) = integral exp(-beta * S[trajectory]) D[trajectory]

The beta annealing schedule: beta(t) = beta_0 * (T - t) / T
- At t=0 (start of horizon): beta = beta_0 (cold, exploiting known low-cost regions)
- At t=T (end of horizon):   beta = 0      (hot, fully exploratory)
This is the quantum annealing / simulated annealing schedule in reverse
(we want precise short-term plans, exploratory long-term imagination).

References
----------
- Williams et al. (2017). Information Theoretic MPC for Model-Based RL.
- Tang, E. (2023). Quantum ML Without Any Quantum. UW PhD Thesis.
- Your notes (Nov 2022): Z(beta) partition function, high-temperature limit.
"""
from __future__ import annotations

import sys
import os
import math
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

# Locate planning.py and import MPPIPlanner
sys.path.insert(0, os.path.dirname(__file__))
from planning import MPPIPlanner, PlanningResult
from logging import get_logger

# Locate dequantized predictor for eigenstate bank
_LESSON_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "datastructure", "Lesson"
    )
)
if _LESSON_DIR not in sys.path:
    sys.path.insert(0, _LESSON_DIR)
from dequantized_jepa_predictor import DequantizedLatentTransition

logger = get_logger(__name__)


# =============================================================================
# THIN BRIDGE: JKL from information_theory.py (handles import path differences)
# =============================================================================

def _jkl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Jeffreys-KL divergence J(p,q) = KL(p||q) + KL(q||p).
    Operates on last dim; returns [B].
    """
    eps = 1e-10
    p = p.clamp(min=eps); q = q.clamp(min=eps)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    return kl_pq + kl_qp


# =============================================================================
# 1.  QUANTUM MPPI PLANNER
# =============================================================================

class QuantumMPPIPlanner(MPPIPlanner):
    """
    Quantum-Augmented MPPI Planner.

    Extends MPPIPlanner with three quantum-inspired upgrades:

    (A) Boltzmann-weighted noise sampling
        Instead of: epsilon ~ N(0, sigma^2)
        We use:     epsilon = sum_n w_n * psi_n * xi_n,  xi_n ~ N(0,1)
        where w_n = softmax(-beta_t * E_n) are the Boltzmann weights
        and psi_n are the eigenstates of the latent transition.
        This biases exploration toward low-energy latent directions.

    (B) JKL regularization in the MPPI cost
        cost_quantum = cost_original + lambda_jkl * JKL(q_rollout || Boltzmann_prior)
        Forces plans to stay in the Boltzmann manifold — physically realistic plans.

    (C) Beta annealing over the planning horizon
        beta(t) = beta_0 * (T - t) / T
        Short-term: precise (cold), Long-term: exploratory (hot).

    Parameters
    ----------
    All MPPIPlanner params, plus:
    quantum_predictor : DequantizedLatentTransition  — provides eigenstate bank
    lambda_jkl        : float   — weight for JKL regularization (default 0.1)
    beta_0            : float   — initial inverse temperature (default 2.0)
    """

    def __init__(
        self,
        unroll: Callable,
        quantum_predictor: DequantizedLatentTransition,
        lambda_jkl: float = 0.1,
        beta_0: float = 2.0,
        **mppi_kwargs,
    ):
        super().__init__(unroll=unroll, **mppi_kwargs)
        self.quantum_predictor = quantum_predictor
        self.lambda_jkl        = lambda_jkl
        self.beta_0            = beta_0

    # ── Boltzmann noise sampling ─────────────────────────────────────────────

    def _boltzmann_noise(
        self,
        plan_length: int,
        t_step: int,
    ) -> torch.Tensor:
        """
        Sample action noise from the Boltzmann distribution of the eigenstate bank.

        epsilon_t = sum_n w_n(beta_t) * psi_n[: action_dim] * xi_n,  xi_n ~ N(0,1)

        Beta annealing: beta_t = beta_0 * (T - t) / T
          t=0        → beta = beta_0 (precise, cold)
          t=T-1      → beta ≈ 0      (exploratory, hot)

        Returns
        -------
        noise : [plan_length, num_samples, action_dim]
        """
        predictor = self.quantum_predictor

        # Per-step annealed beta
        betas = torch.tensor(
            [self.beta_0 * max(0.0, (plan_length - t)) / plan_length
             for t in range(plan_length)],
            device=self.device, dtype=torch.float32,
        )  # [T]

        E_n = predictor.energy_levels()   # [k]
        k   = E_n.shape[0]
        ad  = self.action_dim

        # Boltzmann weights per time step: w_n(t) = softmax(-beta_t * E_n)
        # [T, k]
        log_w = -betas.unsqueeze(-1) * E_n.unsqueeze(0)  # [T, k]
        w = torch.softmax(log_w, dim=-1)                  # [T, k]

        # Eigenstates projected to action dimension
        psi = predictor.eigenstates[:, :ad] if predictor.eigenstates.shape[-1] >= ad \
              else F.pad(predictor.eigenstates, (0, ad - predictor.eigenstates.shape[-1]))
        # psi: [k, ad]

        # Standard Gaussian noise
        xi = torch.randn(plan_length, self.num_samples, k, device=self.device)  # [T, B, k]

        # Weighted sum: epsilon_t = sum_n w_n(t) * xi_n * psi_n
        # [T, B, k] * [T, 1, k] -> [T, B, k]; then @ [k, ad] -> [T, B, ad]
        noise = (xi * w.unsqueeze(1)) @ psi   # [T, B, ad]

        # Scale to match expected std
        noise = noise * self.max_std / (noise.std() + 1e-8)
        return noise

    # ── JKL regularization ──────────────────────────────────────────────────

    def _jkl_regularization(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute JKL regularization: how far are the proposed actions from
        the Boltzmann prior (uninformative = uniform over eigenstates)?

        actions : [T, B, A]
        Returns : [B] JKL penalty per sample
        """
        predictor = self.quantum_predictor
        k  = predictor.eigenstates.shape[0]

        # Project actions onto eigenstate basis
        ad = self.action_dim
        psi = predictor.eigenstates[:, :ad].detach()   # [k, ad]

        # Posterior: softmax of action-eigenstate similarity scores
        # actions mean over T: [B, A] @ [A, k] -> [B, k]
        a_mean = actions.mean(dim=0)   # [B, A]
        scores = a_mean @ psi.T        # [B, k]
        q_posterior = torch.softmax(scores, dim=-1)   # [B, k]

        # Prior: Boltzmann distribution at current predictor temperature
        p_prior = predictor.boltzmann_weights().unsqueeze(0).expand_as(q_posterior)  # [B, k]

        return _jkl(q_posterior, p_prior)   # [B]

    # ── Overridden plan() ────────────────────────────────────────────────────

    @torch.no_grad()
    def plan(
        self, obs_init, t0=False, eval_mode=False, steps_left=None, plan_vis_path=None
    ) -> PlanningResult:
        """
        Quantum-Augmented MPPI plan step.

        Modifications vs standard MPPIPlanner:
          1. Noise sampling: Boltzmann-weighted instead of isotropic Gaussian
          2. Cost: original cost + lambda_jkl * JKL(q_actions || Boltzmann_prior)
          3. Beta annealing: shorter-horizon steps get colder (more precise) noise
        """
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std  = self.max_std * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(
            plan_length, self.num_samples, self.action_dim, device=self.device
        )

        losses, elite_means, elite_stds = [], [], []

        for iter_idx in range(self.n_iters):
            # ── (A) Boltzmann-weighted noise ─────────────────────────────────
            q_noise = self._boltzmann_noise(plan_length, t_step=iter_idx)  # [T, B, A]
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * q_noise

            # ── Standard MPPI cost ────────────────────────────────────────────
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)   # [B, 1]

            # ── (B) JKL regularization ────────────────────────────────────────
            if self.lambda_jkl > 0:
                jkl_penalty = self._jkl_regularization(actions)   # [B]
                cost = cost + self.lambda_jkl * jkl_penalty.unsqueeze(1)

            losses.append(cost.min().item())

            # Elite selection & parameter update (identical to standard MPPI)
            elite_idxs   = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss   = cost[elite_idxs]
            elite_actions = actions[:, elite_idxs]

            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            min_cost = cost.min(0)[0]
            score    = torch.exp(self.temperature * (min_cost - elite_loss[:, 0]))
            score   /= score.sum(0)

            mean = torch.sum(
                score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1
            ) / (score.sum(0) + 1e-9)
            std  = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2)
                    * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,
                ) / (score.sum(0) + 1e-9)
            )

        score   = score.cpu().numpy()
        actions = elite_actions[
            :, np.random.choice(np.arange(score.shape[0]), p=score)
        ]

        self._prev_mean = mean
        if not eval_mode:
            # Final noise with annealed beta at t=plan_length-1 (hottest step)
            last_noise = self._boltzmann_noise(plan_length, t_step=self.n_iters - 1)
            actions += std * last_noise[:, 0, :]   # [T, A]

        return PlanningResult(
            actions=actions,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
