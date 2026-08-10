"""
omnistats/modules/bayesian/maxwell_prior.py
--------------------------------------------
Maxwell-Boltzmann Prior for Latent Energy Variables.

Provides the Maxwell-Boltzmann distribution as a physically-motivated prior
for the `energy_hat` output of APADecoder (jepa_bridge.py).

Theoretical basis (Feb-2022 notes)
------------------------------------
The Maxwell-Boltzmann distribution models the speed distribution of particles
in an ideal gas at thermodynamic equilibrium:
    p(v) = sqrt(2/pi) * (v^2 / sigma^3) * exp(-v^2 / 2*sigma^2)

For kinetic energy E = (1/2)*m*v^2 = ||z||^2 / beta (our convention):
    p(E) ∝ sqrt(E) * exp(-E / 2*sigma^2)

This is the chi-squared distribution with 3 degrees of freedom (chi2(3)).

In the JEPA XAI context
------------------------
- Latent speed  v  = ||z||  (L2 norm of embedding = "speed" in latent space)
- Latent energy E  = v^2 / beta = ||z||^2 / beta
- sigma controls the "temperature" of the latent gas:
  - High sigma: wide distribution, many energy levels accessible (hot / uncertain)
  - Low sigma : concentrated near E=0, cold / confident states dominate

This gives a principled, physically interpretable prior for energy_hat
instead of an improper flat or Gaussian prior.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Chi2


# =============================================================================
# 1.  MAXWELL-BOLTZMANN DISTRIBUTION
# =============================================================================

class MaxwellBoltzmannPrior:
    """
    Maxwell-Boltzmann prior for latent kinetic energy E = ||z||^2 / beta.

    Parameterized by scale sigma (= sqrt(k_B T / m) in physics).

    Log-probability:
        log p(E) = log(sqrt(2/pi)) + 2*log(v) - v^2/(2*sigma^2) - 3*log(sigma)
    where v = sqrt(beta * E)  (latent speed reconstructed from energy).

    Parameters
    ----------
    sigma : float   Scale parameter (controls temperature of latent gas).
                    Default 1.0 — isotropic unit-temperature prior.
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def log_prob(
        self,
        energy: torch.Tensor,
        beta: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """
        Log-probability of energy under the Maxwell-Boltzmann prior.

        Parameters
        ----------
        energy : torch.Tensor  [B]  latent energies (E = ||z||^2 / beta)
        beta   : float or [B]   inverse temperature

        Returns
        -------
        log_p : torch.Tensor  [B]
        """
        E = energy.clamp(min=1e-10)
        if isinstance(beta, float):
            beta_t = torch.tensor(beta, dtype=E.dtype, device=E.device)
        else:
            beta_t = beta.to(E.device)

        # Reconstruct speed v from energy: E = v^2/beta → v = sqrt(beta*E)
        v = torch.sqrt(beta_t * E).clamp(min=1e-10)

        sigma = self.sigma
        log_p = (
            math.log(math.sqrt(2.0 / math.pi))
            + 2.0 * v.log()
            - v**2 / (2.0 * sigma**2)
            - 3.0 * math.log(sigma)
        )
        return log_p

    def sample(
        self,
        n: int,
        beta: float = 1.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Sample energies from the Maxwell-Boltzmann prior.

        Uses the fact that if v ~ Maxwell(sigma), then v^2 ~ chi2(3) * sigma^2,
        so E = v^2 / beta ~ chi2(3) * sigma^2 / beta.

        Parameters
        ----------
        n      : int   number of samples
        beta   : float inverse temperature
        device : str

        Returns
        -------
        energies : torch.Tensor  [n]
        """
        chi2_dist = Chi2(df=torch.tensor(3.0))
        v2 = chi2_dist.sample((n,)) * self.sigma**2   # v^2 ~ chi2(3)*sigma^2
        return (v2 / beta).to(device)

    def kl_from_gaussian(self, mu: torch.Tensor, sigma_q: torch.Tensor) -> torch.Tensor:
        """
        Approximate KL divergence from a Gaussian posterior q ~ N(mu, sigma_q^2)
        to this Maxwell-Boltzmann prior (via Monte Carlo estimate).

        Parameters
        ----------
        mu      : [B]  posterior means
        sigma_q : [B]  posterior standard deviations

        Returns
        -------
        kl : [B]
        """
        # MC estimate: sample from q, evaluate log q(E) - log p(E)
        n_mc = 64
        eps = torch.randn(n_mc, *mu.shape, device=mu.device)   # [n_mc, B]
        E_samples = (mu.unsqueeze(0) + sigma_q.unsqueeze(0) * eps).clamp(min=1e-10)  # [n_mc, B]

        log_q = -0.5 * ((E_samples - mu.unsqueeze(0)) / sigma_q.unsqueeze(0).clamp(min=1e-8))**2 \
                - sigma_q.unsqueeze(0).clamp(min=1e-8).log() - 0.5 * math.log(2 * math.pi)
        log_p = torch.stack([self.log_prob(E_samples[i]) for i in range(n_mc)])  # [n_mc, B]

        kl = (log_q - log_p).mean(dim=0)   # [B]
        return kl.clamp(min=0.0)


# =============================================================================
# 2.  MAXWELL PRIOR LOSS TERM (for use in APADecoder training)
# =============================================================================

def maxwell_prior_loss(
    energy_hat: torch.Tensor,
    beta_hat: torch.Tensor,
    sigma: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Negative log-probability under the Maxwell-Boltzmann prior.

    Use as an auxiliary loss term during APADecoder training to encourage
    the energy_hat predictions to be physically realistic:

        L_maxwell = -E_{z}[log p_MB(energy_hat(z))]

    Parameters
    ----------
    energy_hat : [B]   predicted Boltzmann energies from APADecoder
    beta_hat   : [B]   predicted inverse temperatures from APADecoder
    sigma      : float prior scale
    reduction  : str   'mean' | 'sum' | 'none'

    Returns
    -------
    loss : scalar or [B]
    """
    prior = MaxwellBoltzmannPrior(sigma=sigma)
    log_p = prior.log_prob(energy_hat, beta=beta_hat)   # [B]
    nll   = -log_p

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    return nll
