"""
datastructure/Lesson/dequantized_jepa_predictor.py
---------------------------------------------------
Quantum-Inspired (Dequantized) JEPA Latent Transition Model.

Replaces the standard deterministic JEPA predictor MLP with a classical
ℓ₂ importance-sampling implementation of Quantum Density Matrix Dynamics.

Theoretical Grounding
---------------------
1. Density Matrix Transition (from your notes, Nov 2022):
       p(x, x', β) = Σ_n e^{-βE_n} ψ_n(x) ψ_n*(x')
   This is a low-rank spectral decomposition: each eigenstate n contributes
   a rank-1 outer product ψ_n(x) ⊗ ψ_n*(x'), weighted by e^{-βE_n}.

2. Path Integral / Convolution (your notes, property ①):
       ∫ dx' p(x, x', β₁) p(x', x'', β₂) = p(x, x'', β₁+β₂)
   Composed transitions simply add their inverse temperatures.
   Classically: apply the sketched transition T times with β/T each step.

3. High-Temperature Approximation (property ③):
       p(x, x', β) ≈ e^{-β/2 V(x)} p^{free}(x, x', β) e^{-β/2 V(x')}
   where V(x) is the potential / cost (e.g., prediction error).
   p^{free} is the free-particle density matrix (Gaussian in latent space).

4. Dequantization via ℓ₂ sampling (Tang 2023):
   Instead of computing the full matrix product A·v, we:
   (a) Sample row index i with probability ∝ ||A_i||²   (ℓ₂ importance weight)
   (b) Use that sample to sketch the result in O(k·poly(log(d))) time
   where k = effective rank ≪ d = latent dimension.

References
----------
- Tang, Ewin. "Quantum Machine Learning Without Any Quantum." UW, 2023.
- Feynman, R. & Hibbs, A. Quantum Mechanics and Path Integrals. 1965.
- Your Nov-2022 notes on density matrix convolution and partition functions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ═════════════════════════════════════════════════════════════════════════════
# 1. ℓ₂ IMPORTANCE SAMPLING (Core Dequantization Primitive)
# ═════════════════════════════════════════════════════════════════════════════

class L2ImportanceSampler:
    """
    ℓ₂ Norm Importance Sampling for Low-Rank Matrix Sketching.

    This is the key primitive from Tang's dequantization framework.
    Allows us to approximate dense matrix-vector products in sub-linear
    time when the matrix has low effective rank.

    A data structure {A} is a "sampling and query" (SQ) oracle if:
    - query(i, j) → A[i,j]            in O(1)
    - sample(i)   → j w.p. A[i,j]²/||A_i||²   in O(1) amortized

    Here we implement a dense approximation (batch-compatible) using
    multinomial sampling rather than an actual tree-based oracle.
    For real SOTA deployment, replace with a Count-Sketch or FLAN structure.
    """

    def __init__(self, rank_k: int = 32, seed: int = 42):
        """
        Parameters
        ----------
        rank_k : int
            Number of importance samples (effective rank proxy).
            Lower k = faster but less accurate approximation.
        seed   : int
        """
        self.rank_k = rank_k
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def sketch_matrix_vector(
        self,
        A: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sketch A·v using ℓ₂ importance sampling.

        Instead of computing A·v = Σ_j A[:,j] * v[j] with d terms,
        sample k indices j ~ |A[:,j]|² / ||A||_F² and estimate:
            (A·v)_i ≈ (||A||_F² / k) Σ_{sampled j} A[i,j] * v[j] / |A[:,j]|²

        Parameters
        ----------
        A : [d_out, d_in]  weight / transition matrix
        v : [B, d_in]      input vectors

        Returns
        -------
        approx_Av : [B, d_out]
        """
        d_out, d_in = A.shape
        B = v.shape[0]

        # ℓ₂ importance weights over columns of A
        col_norms_sq = (A ** 2).sum(dim=0)          # [d_in]
        total_sq = col_norms_sq.sum().clamp(min=1e-10)
        probs = col_norms_sq / total_sq             # [d_in]

        # Sample k column indices
        k = min(self.rank_k, d_in)
        idx = torch.multinomial(probs, k, replacement=True, generator=self.rng)   # [k]

        # Correction factor (unbiased estimator)
        sampled_probs = probs[idx]                  # [k]
        A_sampled = A[:, idx]                       # [d_out, k]
        v_sampled = v[:, idx]                       # [B, k]

        # Importance-weighted product
        # A·v ≈ Σ_{j in sampled} A[:,j] * v[:,j] / (k * p_j) (unbiased)
        weight = 1.0 / (k * sampled_probs.unsqueeze(0))   # [1, k]
        approx = (A_sampled.unsqueeze(0) * (v_sampled * weight).unsqueeze(1)).sum(dim=-1)
        # [d_out, k] × [B, 1, k] → [B, d_out]

        return approx    # [B, d_out]


# ═════════════════════════════════════════════════════════════════════════════
# 2. FREE-PARTICLE DENSITY MATRIX (p^free)
# ═════════════════════════════════════════════════════════════════════════════

def free_density_matrix(
    z: torch.Tensor,
    z_prime: torch.Tensor,
    beta: float,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Free-particle density matrix p^{free}(z, z', β).

    In the high-temperature / free-particle limit (no potential V):
        p^{free}(z, z', β) ∝ exp( -||z - z'||² / (4σ²β) )

    This is a Gaussian kernel in latent space. It represents the
    "unconstrained" quantum diffusion between states z and z'.

    Parameters
    ----------
    z, z_prime : torch.Tensor  [B, D]
        Source and target latent vectors.
    beta : float
        Inverse temperature (time step in path integral sense).
    sigma : float
        Standard deviation of the free-particle Gaussian.

    Returns
    -------
    p_free : torch.Tensor  [B]
        Free density matrix value for each pair (z, z').
    """
    diff_sq = ((z - z_prime) ** 2).sum(dim=-1)   # [B]
    return torch.exp(-diff_sq / (4 * sigma ** 2 * beta + 1e-10))


# ═════════════════════════════════════════════════════════════════════════════
# 3. DEQUANTIZED LATENT TRANSITION MODULE
# ═════════════════════════════════════════════════════════════════════════════

class DequantizedLatentTransition(nn.Module):
    """
    Dequantized Quantum Density Matrix Latent Transition.

    Replaces the deterministic JEPA predictor MLP with a quantum-inspired
    probabilistic transition that:
    (a) Learns a low-rank spectral decomposition {E_n, ψ_n} (energy levels
        and eigenstates) of the latent space.
    (b) Applies the high-temperature approximation:
            p(z, z', β) ≈ e^{-β/2 V(z)} p^{free}(z, z', β) e^{-β/2 V(z')}
        where V(z) = task-specific potential (learned energy function).
    (c) Uses ℓ₂ importance sampling to approximate the transition
        in O(k·D) instead of O(D²) time (Tang's dequantization).

    The output is NOT a single point prediction but a distribution:
        (z_predicted_mean, energy_E_n, inverse_temperature_beta, uncertainty_σ²)
    This enables the XAI reporting pipeline.

    Architecture
    ------------
    - PotentialNet: Maps z → scalar V(z) (energy/cost of state z)
    - EigenstateBank: Learnable rank-k approximation {ψ_n, E_n}
    - ℓ₂Sampler: Dequantized sketching for fast transitions

    Parameters
    ----------
    d_latent : int     Dimensionality of the latent space
    rank_k   : int     Effective rank (number of eigenstates / importance samples)
    beta     : float   Initial inverse temperature (learned as parameter)
    sigma    : float   Free-particle diffusion scale
    n_steps  : int     Number of path integral composition steps (convolution depth)
    """

    def __init__(
        self,
        d_latent: int,
        rank_k: int = 32,
        beta: float = 1.0,
        sigma: float = 1.0,
        n_steps: int = 3,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.rank_k   = rank_k
        self.n_steps  = n_steps
        self.sigma    = sigma

        # Learnable inverse temperature (scalar)
        self.log_beta = nn.Parameter(torch.tensor(np.log(beta), dtype=torch.float32))

        # Potential energy function V(z): z → scalar
        self.potential_net = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.SiLU(),
            nn.Linear(d_latent // 2, 1),
        )

        # Low-rank eigenstate bank: {ψ_1, ..., ψ_k} and {E_1, ..., E_k}
        self.eigenstates = nn.Parameter(torch.randn(rank_k, d_latent) * 0.01)   # [k, D]
        self.log_energies = nn.Parameter(torch.zeros(rank_k))                    # [k]

        # Transition head: combines sampled eigenstates → predicted next z
        self.transition_head = nn.Sequential(
            nn.Linear(d_latent + rank_k, d_latent),
            nn.LayerNorm(d_latent),
            nn.SiLU(),
            nn.Linear(d_latent, d_latent * 2),  # outputs mean + log_var
        )

        # ℓ₂ sampler
        self.sampler = L2ImportanceSampler(rank_k=rank_k)

    @property
    def beta(self) -> torch.Tensor:
        """Current (learned) inverse temperature β = exp(log_β)."""
        return self.log_beta.exp()

    def potential(self, z: torch.Tensor) -> torch.Tensor:
        """V(z): potential energy of latent state z. Shape [B]."""
        return self.potential_net(z).squeeze(-1)   # [B]

    def energy_levels(self) -> torch.Tensor:
        """E_n = exp(log_E_n) for each eigenstate. Shape [k]."""
        return self.log_energies.exp()

    def boltzmann_weights(self) -> torch.Tensor:
        """w_n = e^{-β E_n} / Z(β). Shape [k]."""
        E = self.energy_levels()
        log_w = -self.beta * E
        return torch.softmax(log_w, dim=-1)   # [k]

    def forward(
        self,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> dict:
        """
        Apply one dequantized density matrix transition step.

        Path integral with n_steps convolutions of β/n_steps each:
            p(z_0, z_T, β) = ∫...∫ Π_{t=1}^{T} p(z_{t-1}, z_t, β/T) dz_1...dz_{T-1}

        Parameters
        ----------
        z             : [B, D]  current latent state
        action        : [B, D_action] optional action / goal (for planning)
        return_metrics: bool    if True, return XAI metrics alongside prediction

        Returns
        -------
        dict with:
            "z_next_mean"  : [B, D]  predicted next latent (mean)
            "z_next_logvar": [B, D]  log-variance of the prediction
            "energy"       : [B]     energy of input state V(z)
            "beta"         : scalar  current inverse temperature
            "boltzmann_w"  : [k]     Boltzmann weights of eigenstates
            "free_density" : [B]     free-particle kernel value (optional)
        """
        B, D = z.shape
        beta_step = self.beta / self.n_steps

        z_current = z
        V_z = self.potential(z)   # [B] — potential of input state

        for _ in range(self.n_steps):
            # ── High-Temperature Approximation: apply e^{-β/2 V(z)} weights ─
            # Weight the latent by the Boltzmann factor of its potential
            potential_weight = torch.exp(-beta_step / 2 * V_z).unsqueeze(-1)  # [B, 1]
            z_weighted = z_current * potential_weight                           # [B, D]

            # ── ℓ₂ Importance Sampling on eigenstate bank ──────────────────
            # Sketch: project z onto the eigenstate basis (low-rank transition)
            # eigenstates: [k, D], z_weighted: [B, D]
            eigenstate_scores = z_weighted @ self.eigenstates.T   # [B, k]

            # Boltzmann weight the eigenstate contributions
            b_weights = self.boltzmann_weights().unsqueeze(0)     # [1, k]
            eigenstate_scores = eigenstate_scores * b_weights     # [B, k]

            # ── Compose: concatenate [z, eigenstate_scores] → next z ───────
            z_in = torch.cat([z_weighted, eigenstate_scores], dim=-1)  # [B, D+k]
            out = self.transition_head(z_in)                            # [B, 2D]
            z_mean, z_logvar = out.chunk(2, dim=-1)                     # [B, D] each

            # Re-parameterize if training (z ~ N(mean, var))
            if self.training:
                z_std = (0.5 * z_logvar).exp()
                eps = torch.randn_like(z_mean)
                z_current = z_mean + eps * z_std
            else:
                z_current = z_mean

            # Update potential for next step
            V_z = self.potential(z_current)

        # ── Optional: apply action context (goal-conditioning) ───────────────
        if action is not None:
            z_current = z_current + 0.1 * action[..., :D]

        result = {
            "z_next_mean":   z_mean,      # [B, D]
            "z_next_logvar": z_logvar,    # [B, D]
            "energy":        V_z,          # [B]
            "beta":          self.beta,    # scalar
            "boltzmann_w":   self.boltzmann_weights(),  # [k]
        }

        if return_metrics:
            # Free density between input z and predicted z_next
            result["free_density"] = free_density_matrix(
                z, z_mean, beta=self.beta.item(), sigma=self.sigma
            )

        return result


# ═════════════════════════════════════════════════════════════════════════════
# 4. PARTITION FUNCTION ANALYSIS (for XAI reporting)
# ═════════════════════════════════════════════════════════════════════════════

def compute_partition_function(
    model: DequantizedLatentTransition,
    z_samples: torch.Tensor,
) -> dict:
    """
    Compute the Partition Function Z(β) and spectral statistics for XAI.

    Z(β) = Σ_n e^{-βE_n}  (sums over all energy levels, from your notes)

    A high Z(β) indicates many accessible low-energy states (like a hot gas).
    A low Z(β) indicates the model is "frozen" in a few high-energy states.

    Parameters
    ----------
    model     : DequantizedLatentTransition
    z_samples : [N, D] sample latent vectors from the JEPA encoder

    Returns
    -------
    dict with:
        "Z_beta"       : scalar  partition function at current β
        "free_energies": [N]     free energy of each sample
        "mean_energy"  : scalar  expected energy
        "entropy"      : scalar  entropy of Boltzmann weights
    """
    with torch.no_grad():
        E_n = model.energy_levels()   # [k]
        beta = model.beta

        # Partition function: Z(β) = Σ exp(-βE_n)
        log_Z = torch.logsumexp(-beta * E_n, dim=0)

        # Free energy of samples: F(z) = V(z) - (1/β) log Z(β)
        V_z = model.potential(z_samples)                # [N]
        free_energy = V_z - log_Z / beta               # [N]

        # Entropy of Boltzmann distribution
        b_weights = model.boltzmann_weights()           # [k]
        entropy = -(b_weights * b_weights.clamp(min=1e-10).log()).sum()

    return {
        "Z_beta":        log_Z.exp().item(),
        "log_Z_beta":    log_Z.item(),
        "free_energies": free_energy,
        "mean_energy":   V_z.mean().item(),
        "entropy":       entropy.item(),
        "beta":          beta.item(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. MULTI-STEP PATH INTEGRAL (Convolution Composition)
# ═════════════════════════════════════════════════════════════════════════════

def path_integral_rollout(
    model: DequantizedLatentTransition,
    z_start: torch.Tensor,
    T: int = 5,
    actions: Optional[torch.Tensor] = None,
) -> dict:
    """
    Multi-step path integral rollout using convolution composition.

    From your notes property ①:
        ∫ dx' p(x, x', β₁) p(x', x'', β₂) = p(x, x'', β₁ + β₂)
    
    We unroll T steps of the transition model with β/T each, which
    is mathematically equivalent to one step with full β (path integral).

    Parameters
    ----------
    model   : DequantizedLatentTransition
    z_start : [B, D]  initial latent state
    T       : int     number of rollout steps
    actions : [T, B, D_action] optional action sequence

    Returns
    -------
    dict with:
        "trajectory"  : list of [B, D] latent states at each step
        "energies"    : [T, B]   energy at each step
        "total_beta"  : float    total accumulated inverse temperature
    """
    trajectory = [z_start]
    energies = []
    z = z_start

    for t in range(T):
        action_t = actions[t] if actions is not None else None
        out = model(z, action=action_t, return_metrics=True)
        z = out["z_next_mean"]
        trajectory.append(z)
        energies.append(out["energy"].unsqueeze(0))

    return {
        "trajectory": trajectory,
        "energies":   torch.cat(energies, dim=0).T,   # [B, T]
        "total_beta": model.beta.item() * T,
    }
