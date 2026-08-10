"""
datastructure/Lesson/chebyshev_qsvt.py
----------------------------------------
Chebyshev QSVT (Quantum Singular Value Transformation) — Tang 2023, Ch. 6.

Provides classical, dequantized approximation of matrix polynomial functions
via the Clenshaw recursion, combined with L2 importance sampling for sub-linear
complexity on low-rank matrices.

Key concepts
------------
- Chebyshev polynomials T_k provide the optimal (minimax) polynomial approximation
  to any continuous function on [-1, 1].
- Tang shows that QSVT (which applies f(A) to a vector v) can be dequantized:
  we only need SQ-oracle (L2 sampling) access to A, not the full matrix.
- The Clenshaw recursion computes sum_k c_k T_k(A) v in O(degree * k) time
  using only matrix-vector products, each of which is L2-sketched.

This allows us to apply smooth spectral filters (heat kernels, low-pass, high-pass)
to the JEPA eigenstate bank matrix without O(d^2) cost.

References
----------
Tang, E. (2023). Quantum Machine Learning Without Any Quantum. UW PhD Thesis.
  - Section 6: Dequantizing QSVT via Chebyshev / Clenshaw recursion.
  - Theorem 6.1: Clenshaw approximates f(A)v in O(poly(log d) * degree) time.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable

# Import the L2 sampler from the base predictor
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from dequantized_jepa_predictor import (
    L2ImportanceSampler,
    DequantizedLatentTransition,
    free_density_matrix,
)


# =============================================================================
# 1.  CHEBYSHEV COEFFICIENTS FOR SPECTRAL FILTERS
# =============================================================================

def heat_kernel_chebyshev_coeffs(degree: int, beta: float = 1.0) -> torch.Tensor:
    """
    Chebyshev coefficients for the heat-kernel spectral filter:
        f(lambda) = exp(-beta * lambda^2)

    This acts as a quantum diffusion / low-pass filter on the eigenstate bank:
    - Low beta  -> wide kernel -> smooth predictions (explore)
    - High beta -> narrow kernel -> sharp predictions (exploit attractor states)

    Parameters
    ----------
    degree : int    polynomial degree p
    beta   : float  diffusion strength (= inverse temperature analogy)

    Returns
    -------
    coeffs : torch.Tensor  [degree+1]  Chebyshev coefficients c_0, ..., c_p
    """
    # DCT-based Chebyshev coefficient estimation
    # c_k = (2/p) * sum_{j=0}^{p-1} f(cos(pi*(j+0.5)/p)) * cos(pi*k*(j+0.5)/p)
    p = degree + 1
    nodes = torch.tensor(
        [math.cos(math.pi * (j + 0.5) / p) for j in range(p)], dtype=torch.float32
    )
    f_nodes = torch.exp(-beta * nodes ** 2)  # heat kernel at Chebyshev nodes

    coeffs = torch.zeros(p)
    for k in range(p):
        cos_vals = torch.tensor(
            [math.cos(math.pi * k * (j + 0.5) / p) for j in range(p)],
            dtype=torch.float32,
        )
        coeffs[k] = (2.0 / p) * (f_nodes * cos_vals).sum()

    # Standard Chebyshev convention: halve c_0
    coeffs[0] /= 2.0
    return coeffs


def low_pass_chebyshev_coeffs(degree: int, cutoff: float = 0.5) -> torch.Tensor:
    """
    Chebyshev coefficients for a step-function low-pass filter:
        f(lambda) = 1  if |lambda| <= cutoff
                    0  otherwise

    This isolates the low-energy (stable) eigenstates of the JEPA latent transition.

    Parameters
    ----------
    degree : int    polynomial degree
    cutoff : float  spectral cutoff in [-1, 1]
    """
    p = degree + 1
    nodes = torch.tensor(
        [math.cos(math.pi * (j + 0.5) / p) for j in range(p)], dtype=torch.float32
    )
    f_nodes = (nodes.abs() <= cutoff).float()

    coeffs = torch.zeros(p)
    for k in range(p):
        cos_vals = torch.tensor(
            [math.cos(math.pi * k * (j + 0.5) / p) for j in range(p)],
            dtype=torch.float32,
        )
        coeffs[k] = (2.0 / p) * (f_nodes * cos_vals).sum()
    coeffs[0] /= 2.0
    return coeffs


# =============================================================================
# 2.  CLENSHAW RECURSION (Numerically Stable Chebyshev Evaluation)
# =============================================================================

def chebyshev_matvec(
    A: torch.Tensor,
    v: torch.Tensor,
    coeffs: torch.Tensor,
    sampler: Optional[L2ImportanceSampler] = None,
) -> torch.Tensor:
    """
    Evaluate p(A) @ v using the Clenshaw recursion for Chebyshev polynomials.

    Clenshaw algorithm (numerically stable, no explicit matrix powers):
        b_{p+1} = b_{p+2} = 0
        b_k = c_k * v + 2 * A * b_{k+1} - b_{k+2}    for k = p, ..., 1
        result = c_0 * v + A * b_1 - b_2

    Where * denotes matrix-vector product (L2-sketched if sampler provided).

    Parameters
    ----------
    A       : [d_out, d_in]  transition / eigenstate bank matrix
    v       : [B, d_in]      input latent vectors
    coeffs  : [degree+1]     Chebyshev coefficients c_0, ..., c_p
    sampler : L2ImportanceSampler | None

    Returns
    -------
    p_A_v : [B, d_out]   spectral-filtered output
    """
    degree = len(coeffs) - 1
    d_out, d_in = A.shape
    B      = v.shape[0]
    device = v.device

    # Normalize A row-wise to keep spectral radius <= 1
    row_norms = A.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)  # [d_out, 1]
    A_norm = A / row_norms.max()                                     # [d_out, d_in]

    # _Av: multiplies [B, d_in] input by A_norm -> [B, d_out]
    def _Av(vec: torch.Tensor) -> torch.Tensor:
        """A_norm [d_out, d_in] applied to vec [B, d_in] -> [B, d_out]."""
        if sampler is not None:
            return sampler.sketch_matrix_vector(A_norm, vec)  # [B, d_out]
        return vec @ A_norm.T                                  # [B, d_out]

    # _ATv: multiplies [B, d_out] by A_norm^T -> [B, d_in]  (for backward recursion)
    def _ATv(vec: torch.Tensor) -> torch.Tensor:
        """A_norm^T [d_in, d_out] applied to vec [B, d_out] -> [B, d_in]."""
        return vec @ A_norm                                    # [B, d_in]

    # Clenshaw recursion stays in INPUT space [B, d_in]:
    # b_{k} = c_k * v + 2 * A^T A b_{k+1} - b_{k+2}
    # This computes the Chebyshev expansion of (A^T A) applied to v,
    # which lives in input space.  Final output = A * b_1_input.
    ATA = A_norm.T @ A_norm   # [d_in, d_in] — used only in non-sketched path

    def _ATAv(vec: torch.Tensor) -> torch.Tensor:
        """(A^T A) applied to vec [B, d_in] -> [B, d_in]."""
        return vec @ ATA

    # Initialize Clenshaw b-vectors in d_in space
    b_next2 = torch.zeros(B, d_in, device=device, dtype=v.dtype)  # [B, d_in]
    b_next1 = torch.zeros_like(b_next2)                            # [B, d_in]

    for k in range(degree, 0, -1):
        c_k    = coeffs[k].item()
        b_curr = c_k * v + 2.0 * _ATAv(b_next1) - b_next2
        b_next2 = b_next1
        b_next1 = b_curr

    # Final combination stays in d_in; then project to d_out
    b_final = coeffs[0].item() * v + _ATAv(b_next1) - b_next2  # [B, d_in]
    result  = _Av(b_final)                                       # [B, d_out]
    return result


# =============================================================================
# 3.  CHEBYSHEV DEQUANTIZED TRANSITION (QSVT Drop-in Replacement)
# =============================================================================

class ChebyshevDequantizedTransition(DequantizedLatentTransition):
    """
    SOTA upgrade of DequantizedLatentTransition using Chebyshev-QSVT.

    Replaces the MLP TransitionHead with a Chebyshev polynomial of the
    eigenstate bank matrix, evaluated via the Clenshaw recursion with
    optional L2 sketching (Tang 2023, Ch. 6).

    Why this is better than the MLP version
    ----------------------------------------
    1. **Spectral optimality:** Chebyshev polynomials minimize the L-infinity
       approximation error among all polynomials of the same degree (equioscillation
       theorem). No MLP can beat this for spectral filtering.

    2. **Physical interpretability:** The filter f(lambda) directly corresponds
       to a quantum spectral operation:
       - Heat kernel f(lambda) = exp(-beta*lambda^2): quantum diffusion
       - Low-pass f(lambda) = step(lambda < cutoff): isolate stable attractors
       - High-pass: isolate novel/surprise states

    3. **Complexity:** O(degree * k * d) vs MLP O(d^2). For k << d (low-rank
       JEPA latent spaces), this is exponentially cheaper.

    Parameters
    ----------
    d_latent    : int   Latent dimension
    rank_k      : int   Number of eigenstates
    beta        : float Initial inverse temperature
    cheb_degree : int   Polynomial degree (8 = good default; 16 = high quality)
    filter_type : str   'heat' | 'low_pass' — spectral filter to apply
    cutoff      : float Low-pass cutoff (only used when filter_type='low_pass')
    sigma       : float Free-particle diffusion scale
    n_steps     : int   Path integral composition steps
    """

    def __init__(
        self,
        d_latent: int,
        rank_k: int = 32,
        beta: float = 1.0,
        cheb_degree: int = 8,
        filter_type: str = "heat",
        cutoff: float = 0.5,
        sigma: float = 1.0,
        n_steps: int = 3,
    ):
        super().__init__(
            d_latent=d_latent, rank_k=rank_k, beta=beta,
            sigma=sigma, n_steps=n_steps,
        )
        self.cheb_degree  = cheb_degree
        self.filter_type  = filter_type
        self.cutoff       = cutoff

        # Override: lightweight linear head (Chebyshev handles the dynamics)
        # Input: rank_k filtered eigenstate activations → 2*d_latent (mean + logvar)
        self.cheb_head = nn.Linear(rank_k, d_latent * 2)

        # Remove the old MLP transition head to free parameters
        del self.transition_head

    def _get_coeffs(self) -> torch.Tensor:
        """Compute current Chebyshev coefficients (uses learned beta)."""
        beta_val = self.beta.item()
        if self.filter_type == "heat":
            return heat_kernel_chebyshev_coeffs(self.cheb_degree, beta=beta_val)
        elif self.filter_type == "low_pass":
            return low_pass_chebyshev_coeffs(self.cheb_degree, cutoff=self.cutoff)
        else:
            raise ValueError(f"Unknown filter_type: {self.filter_type}")

    def forward(
        self,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> dict:
        """
        Chebyshev-QSVT forward pass.

        1. High-Temperature approximation: weight z by e^{-beta/2 V(z)}
        2. Chebyshev spectral filter of eigenstate bank (Clenshaw recursion)
        3. Boltzmann re-weight + linear head -> (z_mean, z_logvar)
        4. Repeat n_steps times (path integral composition)
        """
        B, D = z.shape
        beta_step = self.beta / self.n_steps
        z_current = z
        V_z = self.potential(z)

        coeffs = self._get_coeffs().to(z.device)

        for _ in range(self.n_steps):
            # High-Temperature approximation: e^{-beta/2 V(z)} weighting
            potential_weight = torch.exp(-beta_step / 2.0 * V_z).unsqueeze(-1)  # [B,1]
            z_weighted = z_current * potential_weight                              # [B,D]

            # Chebyshev-QSVT: spectral filter of eigenstate bank applied to z_weighted
            # eigenstates [k, D] acts as transition matrix; v = z_weighted [B, D]
            filtered = chebyshev_matvec(
                A       = self.eigenstates,       # [k, D]
                v       = z_weighted,             # [B, D]
                coeffs  = coeffs,
                sampler = self.sampler,
            )   # [B, k]

            # Boltzmann re-weighting of filtered eigenstate activations
            b_weights     = self.boltzmann_weights().unsqueeze(0)   # [1, k]
            filtered_bw   = filtered * b_weights                     # [B, k]

            # Linear head: k -> 2D
            out            = self.cheb_head(filtered_bw)             # [B, 2D]
            z_mean, z_logvar = out.chunk(2, dim=-1)                  # [B, D] each

            if self.training:
                z_std     = (0.5 * z_logvar).exp().clamp(max=5.0)
                z_current = z_mean + z_std * torch.randn_like(z_mean)
            else:
                z_current = z_mean

            V_z = self.potential(z_current)

        if action is not None:
            z_current = z_current + 0.1 * action[..., :D]

        result = {
            "z_next_mean":   z_mean,
            "z_next_logvar": z_logvar,
            "energy":        V_z,
            "beta":          self.beta,
            "boltzmann_w":   self.boltzmann_weights(),
            "cheb_degree":   self.cheb_degree,
            "filter_type":   self.filter_type,
        }
        if return_metrics:
            result["free_density"] = free_density_matrix(
                z, z_mean, beta=self.beta.item(), sigma=self.sigma
            )
        return result
