"""
Bayesian/waiting_time_unification.py
-------------------------------------
Unified framework for discrete, continuous, and deterministic waiting-time distributions.
Includes:
  1. Geometric (Discrete memoryless)
  2. Exponential (Continuous memoryless)
  3. Erlang (Sum of continuous memoryless arrivals)
  4. Dirac Delta (Deterministic limit of Erlang with zero variance)
  5. Phase-Type (CTMC absorption time generalizing Exponential/Erlang)
"""

import os
import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class WaitingTimeDistribution(ABC):
    """Abstract base class for waiting-time distributions."""

    @abstractmethod
    def pdf(self, t: np.ndarray) -> np.ndarray:
        """Probability Density Function (PDF) or Probability Mass Function (PMF)."""
        pass

    @abstractmethod
    def cdf(self, t: np.ndarray) -> np.ndarray:
        """Cumulative Distribution Function (CDF)."""
        pass

    @abstractmethod
    def sample(self, size: int) -> np.ndarray:
        """Generate random samples from the distribution."""
        pass

    @abstractmethod
    def theoretical_mean(self) -> float:
        """Return the exact mathematical mean of the distribution."""
        pass

    @abstractmethod
    def theoretical_variance(self) -> float:
        """Return the exact mathematical variance of the distribution."""
        pass


class GeometricWaiting(WaitingTimeDistribution):
    """
    Geometric distribution representing discrete waiting time until the first success.
    PMF: P(X = k) = (1 - p)^{k - 1} * p, for k = 1, 2, ...
    """
    def __init__(self, p: float):
        if not (0.0 < p <= 1.0):
            raise ValueError("Probability p must be in (0, 1]")
        self.p = p

    def pdf(self, t: np.ndarray) -> np.ndarray:
        # PMF is only defined on positive integers
        t_arr = np.atleast_1d(t)
        pmf_vals = np.zeros_like(t_arr, dtype=float)
        mask = (t_arr >= 1) & (np.floor(t_arr) == t_arr)
        pmf_vals[mask] = ((1.0 - self.p) ** (t_arr[mask] - 1.0)) * self.p
        return pmf_vals if isinstance(t, np.ndarray) else pmf_vals[0]

    def cdf(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.atleast_1d(t)
        cdf_vals = np.zeros_like(t_arr, dtype=float)
        mask = t_arr >= 1
        # Discrete CDF: step function at each integer
        k = np.floor(t_arr[mask])
        cdf_vals[mask] = 1.0 - (1.0 - self.p) ** k
        return cdf_vals if isinstance(t, np.ndarray) else cdf_vals[0]

    def sample(self, size: int) -> np.ndarray:
        return np.random.geometric(self.p, size=size)

    def theoretical_mean(self) -> float:
        return 1.0 / self.p

    def theoretical_variance(self) -> float:
        return (1.0 - self.p) / (self.p ** 2)


class ExponentialWaiting(WaitingTimeDistribution):
    """
    Exponential distribution representing continuous memoryless waiting time.
    PDF: f(t) = rate * exp(-rate * t), for t >= 0.
    """
    def __init__(self, rate: float):
        if rate <= 0.0:
            raise ValueError("Rate must be strictly positive")
        self.rate = rate

    def pdf(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.atleast_1d(t)
        pdf_vals = np.zeros_like(t_arr, dtype=float)
        mask = t_arr >= 0.0
        pdf_vals[mask] = self.rate * np.exp(-self.rate * t_arr[mask])
        return pdf_vals if isinstance(t, np.ndarray) else pdf_vals[0]

    def cdf(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.atleast_1d(t)
        cdf_vals = np.zeros_like(t_arr, dtype=float)
        mask = t_arr >= 0.0
        cdf_vals[mask] = 1.0 - np.exp(-self.rate * t_arr[mask])
        return cdf_vals if isinstance(t, np.ndarray) else cdf_vals[0]

    def sample(self, size: int) -> np.ndarray:
        return np.random.exponential(1.0 / self.rate, size=size)

    def theoretical_mean(self) -> float:
        return 1.0 / self.rate

    def theoretical_variance(self) -> float:
        return 1.0 / (self.rate ** 2)


class ErlangWaiting(WaitingTimeDistribution):
    """
    Erlang distribution representing the waiting time for k events in a Poisson process.
    Equivalent to the sum of k i.i.d. Exponential random variables.
    PDF: f(t) = (rate^k * t^{k-1} * exp(-rate * t)) / (k - 1)!, for t >= 0.
    """
    def __init__(self, k: int, rate: float):
        if k < 1:
            raise ValueError("Shape parameter k must be an integer >= 1")
        if rate <= 0.0:
            raise ValueError("Rate parameter must be strictly positive")
        self.k = k
        self.rate = rate

    def pdf(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.atleast_1d(t)
        pdf_vals = np.zeros_like(t_arr, dtype=float)
        mask = t_arr >= 0.0
        # Use stats.gamma for robust evaluation avoiding factorial overflow
        pdf_vals[mask] = stats.gamma.pdf(t_arr[mask], a=self.k, scale=1.0/self.rate)
        return pdf_vals if isinstance(t, np.ndarray) else pdf_vals[0]

    def cdf(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.atleast_1d(t)
        cdf_vals = np.zeros_like(t_arr, dtype=float)
        mask = t_arr >= 0.0
        cdf_vals[mask] = stats.gamma.cdf(t_arr[mask], a=self.k, scale=1.0/self.rate)
        return cdf_vals if isinstance(t, np.ndarray) else cdf_vals[0]

    def sample(self, size: int) -> np.ndarray:
        # Sum of k exponentials
        return np.random.gamma(self.k, 1.0 / self.rate, size=size)

    def theoretical_mean(self) -> float:
        return self.k / self.rate

    def theoretical_variance(self) -> float:
        return self.k / (self.rate ** 2)


class DiracDeltaWaiting(WaitingTimeDistribution):
    """
    Dirac Delta distribution representing deterministic waiting time at exactly t = t0.
    Modeled numerically as a smoothed Dirac delta (narrow Gaussian kernel),
    matching the evaluation pattern in Bayesian/importance_sampling_bayesian.py.
    """
    def __init__(self, t0: float, sigma_dirac: float = 0.01):
        if t0 < 0.0:
            raise ValueError("Deterministic waiting time t0 must be non-negative")
        if sigma_dirac <= 0.0:
            raise ValueError("Smoothed Dirac width sigma_dirac must be positive")
        self.t0 = t0
        self.sigma = sigma_dirac

    def pdf(self, t: np.ndarray) -> np.ndarray:
        # Smoothed Dirac delta representation
        t_arr = np.atleast_1d(t)
        res = stats.norm.pdf(t_arr, loc=self.t0, scale=self.sigma)
        return res if isinstance(t, np.ndarray) else res[0]

    def cdf(self, t: np.ndarray) -> np.ndarray:
        # Approximated by standard normal CDF
        t_arr = np.atleast_1d(t)
        res = stats.norm.cdf(t_arr, loc=self.t0, scale=self.sigma)
        return res if isinstance(t, np.ndarray) else res[0]

    def sample(self, size: int) -> np.ndarray:
        # Generates deterministic values with tiny numeric noise
        return np.random.normal(loc=self.t0, scale=self.sigma, size=size)

    def theoretical_mean(self) -> float:
        return self.t0

    def theoretical_variance(self) -> float:
        # Strictly zero theoretically, but returns self.sigma^2 for the numerical approximation
        return self.sigma ** 2


class PhaseTypeWaiting(WaitingTimeDistribution):
    """
    Continuous Phase-Type (PH) distribution representing the time until absorption
    in a Markov Chain with absorbing state.
    Generalized representation that can represent Exponential, Erlang, and mixtures.
    Parameters:
      - alpha: Initial state probability vector (shape: m)
      - S: Subintensity generator matrix (shape: m x m)
    """
    def __init__(self, alpha: np.ndarray, S: np.ndarray):
        self.alpha = np.atleast_1d(alpha).astype(float)
        self.S = np.atleast_2d(S).astype(float)
        self.m = len(self.alpha)
        
        if self.S.shape != (self.m, self.m):
            raise ValueError("Initial vector and transition matrix dimensions must match")
        if not np.isclose(np.sum(self.alpha), 1.0):
            raise ValueError("Initial state vector alpha must sum to 1")
            
        # Exit rates to the absorbing state
        self.S0 = -np.sum(self.S, axis=1)
        if np.any(self.S0 < -1e-10):
            raise ValueError("Subintensity matrix S must have non-positive row sums")

    def pdf(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.atleast_1d(t)
        pdf_vals = np.zeros_like(t_arr, dtype=float)
        for i, val in enumerate(t_arr):
            if val >= 0.0:
                # f(t) = alpha * expm(S * t) * S0
                pdf_vals[i] = self.alpha @ linalg.expm(self.S * val) @ self.S0
        return pdf_vals if isinstance(t, np.ndarray) else pdf_vals[0]

    def cdf(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.atleast_1d(t)
        cdf_vals = np.zeros_like(t_arr, dtype=float)
        ones = np.ones(self.m)
        for i, val in enumerate(t_arr):
            if val >= 0.0:
                # F(t) = 1 - alpha * expm(S * t) * 1
                cdf_vals[i] = 1.0 - (self.alpha @ linalg.expm(self.S * val) @ ones)
        return cdf_vals if isinstance(t, np.ndarray) else cdf_vals[0]

    def sample(self, size: int) -> np.ndarray:
        # Simulate Markov Chain directly until absorption
        samples = np.zeros(size)
        for s in range(size):
            # Start state according to alpha
            state = np.random.choice(self.m, p=self.alpha)
            time_elapsed = 0.0
            
            while True:
                # Holding rate is -S[state, state]
                holding_rate = -self.S[state, state]
                time_elapsed += np.random.exponential(1.0 / holding_rate)
                
                # Determine next state
                # Transition probabilities to transient states
                rates = np.copy(self.S[state, :])
                rates[state] = 0.0 # remove self transition
                
                exit_rate = self.S0[state]
                total_rate = np.sum(rates) + exit_rate
                
                probs = np.append(rates, exit_rate) / total_rate
                next_choice = np.random.choice(self.m + 1, p=probs)
                
                if next_choice == self.m: # Absorbing state
                    break
                state = next_choice
                
            samples[s] = time_elapsed
        return samples

    def theoretical_mean(self) -> float:
        # Mean = -alpha * S^-1 * 1
        S_inv = linalg.inv(self.S)
        return -float(self.alpha @ S_inv @ np.ones(self.m))

    def theoretical_variance(self) -> float:
        # Var = 2 * alpha * S^-2 * 1 - (Mean)^2
        S_inv = linalg.inv(self.S)
        mean = self.theoretical_mean()
        term2 = 2.0 * float(self.alpha @ np.linalg.matrix_power(S_inv, 2) @ np.ones(self.m))
        return term2 - (mean ** 2)


# ── Demonstration ─────────────────────────────────────────────────────────────
def run_waiting_time_demo():
    print("--- 1. Discrete vs Continuous Limit (Geometric to Exponential) ---")
    dt = 0.1
    rate = 0.5
    p = rate * dt
    
    geom = GeometricWaiting(p)
    expo = ExponentialWaiting(rate)
    
    print(f"Geometric Mean (trials): {geom.theoretical_mean():.4f}")
    print(f"Geometric Mean (scaled time t = trials * dt): {geom.theoretical_mean() * dt:.4f}")
    print(f"Exponential Mean: {expo.theoretical_mean():.4f}")
    
    t_grid = np.linspace(0.0, 10.0, 100)
    
    geom_pdf = [geom.pdf(t / dt) / dt for t in t_grid]
    expo_pdf = expo.pdf(t_grid)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_grid, geom_pdf, 'o-', ms=4, label=f"Scaled Geometric (p={p})", alpha=0.7)
    plt.plot(t_grid, expo_pdf, 'r--', label=f"Exponential (rate={rate})", linewidth=2)
    plt.title("Convergence of Scaled Geometric to Exponential Distribution")
    plt.xlabel("Time (t)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, linestyle=":")
    
    # Save in current directory or outputs
    plt.savefig("waiting_time_geometric_exponential.png", dpi=150)
    plt.close()
    print("Saved plot -> waiting_time_geometric_exponential.png")

    print("\n--- 2. Erlang to Dirac Delta (Deterministic) Limit ---")
    t0 = 4.0
    plt.figure(figsize=(10, 4))
    
    for k in [2, 10, 50, 200]:
        rate_k = k / t0
        erlang = ErlangWaiting(k, rate_k)
        plt.plot(t_grid, erlang.pdf(t_grid), label=f"Erlang (k={k}, Mean={t0})", alpha=0.8)
        
    dirac = DiracDeltaWaiting(t0, sigma_dirac=0.08)
    plt.plot(t_grid, dirac.pdf(t_grid), 'k:', label=f"Smoothed Dirac Delta (t0={t0}, σ={dirac.sigma})", linewidth=2.5)
    
    plt.title("Convergence of Erlang Distribution to Dirac Delta Spike")
    plt.xlabel("Time (t)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.savefig("waiting_time_erlang_dirac.png", dpi=150)
    plt.close()
    print("Saved plot -> waiting_time_erlang_dirac.png")

    print("\n--- 3. Phase-Type Matrix Unification Verification ---")
    k_val = 3
    rate_val = 1.5
    
    alpha = np.zeros(k_val)
    alpha[0] = 1.0
    
    S = np.zeros((k_val, k_val))
    for i in range(k_val):
        S[i, i] = -rate_val
        if i < k_val - 1:
            S[i, i + 1] = rate_val
            
    ph_dist = PhaseTypeWaiting(alpha, S)
    erlang_dist = ErlangWaiting(k_val, rate_val)
    
    print(f"Erlang Mean:       {erlang_dist.theoretical_mean():.4f} | PH Mean:       {ph_dist.theoretical_mean():.4f}")
    print(f"Erlang Variance:   {erlang_dist.theoretical_variance():.4f} | PH Variance:   {ph_dist.theoretical_variance():.4f}")
    
    ph_pdf = ph_dist.pdf(t_grid)
    erl_pdf = erlang_dist.pdf(t_grid)
    max_diff = np.max(np.abs(ph_pdf - erl_pdf))
    print(f"Maximum absolute difference between Erlang and PH PDF: {max_diff:.2e}")


if __name__ == "__main__":
    run_waiting_time_demo()
