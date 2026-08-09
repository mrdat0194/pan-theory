"""
omnistats/modules/information_theory.py
-----------------------------------------
Quantum-Inspired Information Theory Module for Explainable AI (XAI).

Provides classical, fully-differentiable implementations of:
  - Shannon Entropy     : Measures uncertainty of a distribution
  - JKL Divergence      : Jeffreys-KL (symmetric) divergence between prior and posterior
  - KL Divergence (one-way): For Bayesian update scoring
  - Mutual Information  : Joint information between two distributions
  - Bayesian Inverse Score: How well a latent vector z reconstructs the true abstract n

Grounded in the theoretical framework from:
  - Ewin Tang's dequantization via ℓ₂ importance sampling
  - Classical → Quantum analogy: π(x,n) ∝ e^{-βE_n} ψ_n(x)ψ_n*(x)
  - Bayesian Inverse Problem: n → y (2D → 3D image, health → number-game)
  - Shannon Entropy / KL / JKL as decision boundaries

References
----------
- Tang, Ewin. "Quantum Machine Learning Without Any Quantum." UW PhD Thesis, 2023.
- Shannon, C.E. (1948). A Mathematical Theory of Communication.
- Jeffreys, H. (1946). An Invariant Form for the Prior Probability in Estimation Problems.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple

_EPS = 1e-10   # numerical floor to prevent log(0)


# ═════════════════════════════════════════════════════════════════════════════
# 1. SHANNON ENTROPY
# ═════════════════════════════════════════════════════════════════════════════

def shannon_entropy(
    probs: Union[np.ndarray, torch.Tensor],
    base: float = 2.0,
    dim: int = -1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Shannon Entropy: H(p) = -Σ p(x) log_b p(x)

    Measures the uncertainty / surprise in a probability distribution.
    In the JEPA XAI context, this scores how "uncertain" the model is
    about its latent prediction—high entropy = highly uncertain latent state.

    Parameters
    ----------
    probs : array or tensor  [... , K]
        Probability distribution(s) over K categories. Must be non-negative
        and sum to 1 along `dim`.
    base  : float  (default 2 → bits; use e → nats)
    dim   : int    dimension along which to compute entropy

    Returns
    -------
    H : same type as `probs`, shape [...]
        Shannon entropy value(s).

    Notes
    -----
    The maximum entropy for K categories is log_b(K) (uniform distribution).
    Connecting to your notes: H is directly the "Uncertainty" in JKL and the
    forward model: let q(x) known → KL = p(x) log p(x)/q(x).
    """
    if isinstance(probs, np.ndarray):
        p = np.clip(probs, _EPS, None)
        H = -np.sum(p * np.log(p) / np.log(base), axis=dim)
        return H

    # torch path
    p = probs.clamp(min=_EPS)
    H = -(p * torch.log(p) / np.log(base)).sum(dim=dim)
    return H


def differential_entropy_gaussian(
    sigma: Union[np.ndarray, torch.Tensor],
    dim: int = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Differential entropy of a Gaussian: h = 0.5 * log(2πe σ²)

    Relevant for continuous latent Gaussian embeddings in JEPA.
    Connects to your notes on the Gaussian distribution:
        ∫ (1/√2π) exp(-x²/2) dx = 1 − erf(1/√2) ~ 0.6827

    Parameters
    ----------
    sigma : standard deviation(s) of the Gaussian latent embedding

    Returns
    -------
    h : differential entropy (nats)
    """
    if isinstance(sigma, np.ndarray):
        return 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)
    return 0.5 * torch.log(2 * torch.pi * torch.e * sigma ** 2)


# ═════════════════════════════════════════════════════════════════════════════
# 2. KL DIVERGENCE (ONE-WAY)
# ═════════════════════════════════════════════════════════════════════════════

def kl_divergence(
    p: Union[np.ndarray, torch.Tensor],
    q: Union[np.ndarray, torch.Tensor],
    base: float = 2.0,
    dim: int = -1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    KL Divergence: KL(p || q) = Σ p(x) log_b [p(x) / q(x)]

    Measures the information lost when q is used to approximate p.
    In your notes: "Forward—let q(x) known → p(x) log p(x)/q(x)".

    In the JEPA context:
      - p = posterior latent distribution (after encoding context)
      - q = prior latent distribution (uninformative / Maxwell)
    A high KL means the model has learned something significant about
    the future state—low KL means the context adds little information.

    Parameters
    ----------
    p, q : array or tensor  [..., K]
        Probability distributions over K categories.
    base : float
    dim  : int  dimension over which to compute

    Returns
    -------
    KL : shape [...], KL divergence value(s).
    """
    if isinstance(p, np.ndarray):
        p = np.clip(p, _EPS, None)
        q = np.clip(q, _EPS, None)
        return np.sum(p * np.log(p / q) / np.log(base), axis=dim)

    p = p.clamp(min=_EPS)
    q = q.clamp(min=_EPS)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=dim) / np.log(base)


# ═════════════════════════════════════════════════════════════════════════════
# 3. JEFFREYS-KL (SYMMETRIC KL)
# ═════════════════════════════════════════════════════════════════════════════

def jeffreys_kl(
    p: Union[np.ndarray, torch.Tensor],
    q: Union[np.ndarray, torch.Tensor],
    base: float = 2.0,
    dim: int = -1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Jeffreys KL (JKL): J(p, q) = KL(p || q) + KL(q || p)

    The symmetric version of KL divergence, recommended for decision boundaries.
    From your notes: "JKL: Prior → Uninformative, Likelihood ≈ Posterior".

    In JEPA XAI:
      - At a decision boundary, JKL is minimal: the prior and posterior are
        maximally similar → model is "undecided" about the future state.
      - High JKL = model is confidently transitioning from one latent mode
        to another (high-information event).

    Parameters
    ----------
    p, q : array or tensor  [..., K]
    base : float
    dim  : int

    Returns
    -------
    J(p, q) = KL(p||q) + KL(q||p), shape [...]
    """
    return kl_divergence(p, q, base=base, dim=dim) + \
           kl_divergence(q, p, base=base, dim=dim)


# ═════════════════════════════════════════════════════════════════════════════
# 4. MUTUAL INFORMATION
# ═════════════════════════════════════════════════════════════════════════════

def mutual_information(
    joint: Union[np.ndarray, torch.Tensor],
    dim_x: int = -2,
    dim_y: int = -1,
    base: float = 2.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)

    In JEPA XAI, measures how much information the context encoding
    shares with the target encoding. High MI = latent space is well-aligned.

    Parameters
    ----------
    joint : [..., K_x, K_y]
        Joint probability distribution over X and Y.

    Returns
    -------
    I(X;Y) scalar
    """
    if isinstance(joint, np.ndarray):
        p_x = joint.sum(axis=dim_y)
        p_y = joint.sum(axis=dim_x)
        H_X  = shannon_entropy(p_x, base=base, dim=-1)
        H_Y  = shannon_entropy(p_y, base=base, dim=-1)
        H_XY = shannon_entropy(joint.reshape(*joint.shape[:-2], -1), base=base, dim=-1)
        return H_X + H_Y - H_XY

    p_x = joint.sum(dim=dim_y)
    p_y = joint.sum(dim=dim_x)
    H_X  = shannon_entropy(p_x, base=base, dim=-1)
    H_Y  = shannon_entropy(p_y, base=base, dim=-1)
    H_XY = shannon_entropy(joint.reshape(*joint.shape[:-2], -1), base=base, dim=-1)
    return H_X + H_Y - H_XY


# ═════════════════════════════════════════════════════════════════════════════
# 5. BAYESIAN INVERSE SCORE (ℓ₂ Dequantized)
# ═════════════════════════════════════════════════════════════════════════════

def bayesian_inverse_score(
    z: torch.Tensor,
    reference_embeddings: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bayesian Inverse Score via ℓ₂ Importance Sampling.

    This is the core "Explainability" function: it scores how well each JEPA
    latent vector z corresponds to a set of reference (known) embeddings
    using Bayesian inverse scoring, grounded in Tang's ℓ₂ dequantization.

    Theoretical basis
    -----------------
    From Bayes: n → y (inverse problem), where:
      - n = abstract concept (the "true" latent structure, e.g. 3D scene)
      - y = observed embedding (JEPA latent z, e.g. encoded 2D context)

    The key insight from Tang's thesis: to prepare a quantum state |y>,
    we only need ℓ₂ sampling access to the row of the data matrix.
    Classically, this means: sample an index j with probability proportional
    to |z_j|², then use that to approximate the Bayesian posterior.

    Implementation
    --------------
    1. Compute ℓ₂ norm importance weights: w_j = |z_j|² / ||z||²
    2. Compute similarity scores: s_i = z · ref_i / (||z|| ||ref_i||)
    3. Apply softmax to get Bayesian posterior: P(ref_i | z)
    4. Return: (posterior over references, JKL vs. uniform prior)

    Parameters
    ----------
    z : torch.Tensor  [B, D]
        JEPA latent vectors to score.
    reference_embeddings : torch.Tensor  [N, D]
        N known reference latent vectors (e.g. from labelled concepts).
    temperature : float
        Softmax temperature (lower = sharper posterior).

    Returns
    -------
    posterior : torch.Tensor  [B, N]
        Posterior probability of each reference given z.
    jkl_from_prior : torch.Tensor  [B]
        JKL divergence from the uniform (uninformative) prior.
        High JKL = model has high confidence about which reference matches.
    """
    # Normalize for cosine similarity
    z_norm = F.normalize(z, dim=-1)                          # [B, D]
    ref_norm = F.normalize(reference_embeddings, dim=-1)     # [N, D]

    # Similarity matrix (ℓ₂ inner product in normalized space)
    sim = z_norm @ ref_norm.T                                # [B, N]

    # Bayesian posterior via softmax with temperature
    posterior = torch.softmax(sim / temperature, dim=-1)    # [B, N]

    # Uninformative (uniform) prior
    N = reference_embeddings.shape[0]
    prior = torch.full_like(posterior, 1.0 / N)             # [B, N]

    # JKL divergence: how far posterior is from uniform prior
    jkl = jeffreys_kl(posterior, prior, dim=-1)             # [B]

    return posterior, jkl


# ═════════════════════════════════════════════════════════════════════════════
# 6. ENERGY LEVEL ESTIMATION (Boltzmann / Quantum Analogy)
# ═════════════════════════════════════════════════════════════════════════════

def boltzmann_energy(
    z: torch.Tensor,
    beta: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """
    Compute the effective Boltzmann Energy of a latent embedding.

    From the Classical → Quantum analogy in your notes:
        π(x, n) ∝ e^{-βE_n} ψ_n(x) ψ_n*(x)
    where E_n is the energy level of eigenstate n.

    Here we use the ℓ₂ norm squared of the latent vector as a proxy
    for the kinetic energy (analogous to the Maxwell distribution):
        E(z) = (1/β) * ||z||² = kinetic energy (high norm = high energy state)

    Low Energy  → z is near the origin (degenerate / uninformative)
    High Energy → z is far from origin (activated, high information state)

    Parameters
    ----------
    z    : torch.Tensor  [B, D]
    beta : float or Tensor — inverse temperature (1/kT analogy)

    Returns
    -------
    E : torch.Tensor  [B]
        Effective energy level of each latent state.
    """
    norm_sq = (z ** 2).sum(dim=-1)   # [B]
    return norm_sq / beta


def partition_function(
    energies: torch.Tensor,
    beta: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """
    Classical (log-space stable) Partition Function: Z(β) = Σ e^{-βE_n}

    From your notes: Z(β) = Tr(ρ) = Σ_n (partition function).
    High Z = many accessible low-energy states (high entropy).
    Low Z = model is concentrated in a few high-energy states (collapsed).

    Parameters
    ----------
    energies : torch.Tensor  [N]
        Energy levels E_n.
    beta     : float or Tensor

    Returns
    -------
    Z : scalar Tensor
    """
    return torch.logsumexp(-beta * energies, dim=-1).exp()


# ═════════════════════════════════════════════════════════════════════════════
# 7. SUMMARISE XAI METRICS (convenient aggregator for reporting)
# ═════════════════════════════════════════════════════════════════════════════

def compute_xai_metrics(
    z: torch.Tensor,
    reference_embeddings: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    temperature: float = 1.0,
) -> dict:
    """
    Compute a full suite of XAI metrics for a batch of JEPA latent vectors.

    Parameters
    ----------
    z                    : [B, D] JEPA latent vectors
    reference_embeddings : [N, D] known reference concepts (optional)
    beta                 : inverse temperature for energy estimation
    temperature          : softmax temperature for Bayesian scoring

    Returns
    -------
    dict with keys:
        "energy"         : [B] Boltzmann energy of each z
        "entropy"        : [B] Shannon entropy of softmax(z) (latent activation)
        "jkl_from_prior" : [B] JKL from uniform prior (if references provided)
        "posterior"      : [B, N] Bayesian posterior (if references provided)
        "partition_fn"   : scalar Z(β) across the batch
    """
    z_flat = z.reshape(z.shape[0], -1)   # flatten [B, D*T*H*W] → [B, D]

    metrics: dict = {}

    # Energy levels
    metrics["energy"] = boltzmann_energy(z_flat, beta=beta)  # [B]

    # Shannon Entropy of the softmax activation (latent state certainty)
    latent_probs = torch.softmax(z_flat / temperature, dim=-1)   # [B, D]
    metrics["entropy"] = shannon_entropy(latent_probs, base=2.0, dim=-1)  # [B]

    # Partition function across batch (Z(β))
    metrics["partition_fn"] = partition_function(metrics["energy"], beta=beta)

    # Bayesian Inverse Score (if references provided)
    if reference_embeddings is not None:
        posterior, jkl = bayesian_inverse_score(
            z_flat, reference_embeddings, temperature=temperature
        )
        metrics["posterior"]      = posterior   # [B, N]
        metrics["jkl_from_prior"] = jkl         # [B]
    else:
        metrics["jkl_from_prior"] = torch.zeros(z_flat.shape[0], device=z.device)

    return metrics
