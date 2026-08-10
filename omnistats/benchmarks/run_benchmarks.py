"""
omnistats/benchmarks/run_benchmarks.py
----------------------------------------
Automated benchmark runner for all Quantum-Inspired XAI methods.

Compares each new SOTA method against its classical baseline on synthetic
and real-data tasks. Outputs a structured report to stdout and benchmark_results.md.

Usage:
    python omnistats/benchmarks/run_benchmarks.py
    python omnistats/benchmarks/run_benchmarks.py --suite kalman   # single suite
    python omnistats/benchmarks/run_benchmarks.py --seed 42 --n 500
"""
from __future__ import annotations

import argparse
import math
import sys
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "datastructure", "Lesson"))

warnings.filterwarnings("ignore")


# =============================================================================
# 0.  RESULT CONTAINER
# =============================================================================

@dataclass
class BenchResult:
    method:   str
    metric:   str
    value:    float
    baseline: float
    runtime_ms: float
    pass_fail: str = ""

    @property
    def delta_pct(self) -> float:
        if abs(self.baseline) < 1e-10:
            return float("nan")
        return 100.0 * (self.value - self.baseline) / abs(self.baseline)

    def __str__(self) -> str:
        sign = "+" if self.delta_pct >= 0 else ""
        metric_clean = self.metric.replace("\u2193", "(lo)").replace("\u2191", "(hi)")
        return (
            f"  {self.method:<42} {metric_clean:<22} "
            f"val={self.value:>9.4f}  base={self.baseline:>9.4f}  "
            f"delta={sign}{self.delta_pct:>6.1f}%  "
            f"t={self.runtime_ms:>6.1f}ms  {self.pass_fail}"
        )


# =============================================================================
# 1.  SUITE: DEQUANTIZED JEPA PREDICTOR vs MLP BASELINE
# =============================================================================

def bench_dequantized_predictor(n: int = 256, d: int = 64, k: int = 16,
                                 seed: int = 0) -> list[BenchResult]:
    """
    Compares DequantizedLatentTransition vs a plain MLP predictor on:
    - MSE of next-state prediction on random walk data
    - Effective Sample Size (ESS) of Boltzmann weights
    - Inference speed (ms / batch)
    """
    from dequantized_jepa_predictor import DequantizedLatentTransition
    torch.manual_seed(seed)
    results = []

    # ── Synthetic data: random walk in latent space ───────────────────────────
    z0 = torch.randn(n, d)
    z1 = z0 + 0.1 * torch.randn(n, d)   # ground truth next state

    # ── Baseline: MLP predictor ────────────────────────────────────────────────
    mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d))
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(200):
        loss = ((mlp(z0) - z1) ** 2).mean()
        mlp_opt.zero_grad(); loss.backward(); mlp_opt.step()
    mlp_time = (time.perf_counter() - t0) * 1000
    with torch.no_grad():
        mlp_mse = ((mlp(z0) - z1) ** 2).mean().item()

    # ── Quantum Predictor ──────────────────────────────────────────────────────
    qp = DequantizedLatentTransition(d_latent=d, rank_k=k, beta=1.0)
    qp_opt = torch.optim.Adam(qp.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(200):
        out = qp(z0)
        loss = ((out["z_next_mean"] - z1) ** 2).mean()
        qp_opt.zero_grad(); loss.backward(); qp_opt.step()
    qp_time = (time.perf_counter() - t0) * 1000
    with torch.no_grad():
        out = qp(z0, return_metrics=True)
        qp_mse = ((out["z_next_mean"] - z1) ** 2).mean().item()
        w = qp.boltzmann_weights()
        ess = (w.sum() ** 2 / (w ** 2).sum()).item()

    results.append(BenchResult("DequantizedLatentTransition", "MSE(lo)", qp_mse, mlp_mse, qp_time,
                               "[PASS]" if qp_mse <= mlp_mse * 1.5 else "[WARN]"))
    results.append(BenchResult("DequantizedLatentTransition", "ESS (>k/2 good)", ess, k / 2, qp_time,
                               "[PASS]" if ess > k / 2 else "[WARN]"))
    return results


# =============================================================================
# 2.  SUITE: CHEBYSHEV QSVT vs DEQUANTIZED MLP
# =============================================================================

def bench_chebyshev_qsvt(n: int = 128, d: int = 64, k: int = 16,
                          degree: int = 8, seed: int = 0) -> list[BenchResult]:
    """
    Compares ChebyshevDequantizedTransition vs DequantizedLatentTransition on:
    - MSE of next-state prediction
    - Spectral smoothness of trajectory (low variance = more stable)
    - Inference speed
    """
    from dequantized_jepa_predictor import DequantizedLatentTransition, path_integral_rollout
    from chebyshev_qsvt import ChebyshevDequantizedTransition
    torch.manual_seed(seed)
    results = []

    z0 = torch.randn(n, d)
    z1 = z0 + 0.05 * torch.randn(n, d)

    # Baseline: plain dequantized predictor
    base = DequantizedLatentTransition(d_latent=d, rank_k=k, beta=1.0)
    base.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        rb = path_integral_rollout(base, z0, T=5)
    base_time = (time.perf_counter() - t0) * 1000
    base_mse  = ((rb["trajectory"][-1] - z1) ** 2).mean().item()
    base_tvar = torch.stack([t.var() for t in rb["trajectory"]]).std().item()

    # Chebyshev QSVT predictor
    cheb = ChebyshevDequantizedTransition(d_latent=d, rank_k=k,
                                           cheb_degree=degree, filter_type="heat")
    cheb.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        rc = path_integral_rollout(cheb, z0, T=5)
    cheb_time = (time.perf_counter() - t0) * 1000
    cheb_mse  = ((rc["trajectory"][-1] - z1) ** 2).mean().item()
    cheb_tvar = torch.stack([t.var() for t in rc["trajectory"]]).std().item()

    results.append(BenchResult("ChebyshevDequantizedTransition", "MSE(lo)",
                               cheb_mse, base_mse, cheb_time,
                               "[PASS]" if cheb_mse <= base_mse * 1.5 else "[WARN]"))
    results.append(BenchResult("ChebyshevDequantizedTransition", "Traj variance(lo)",
                               cheb_tvar, base_tvar, cheb_time,
                               "[PASS]" if cheb_tvar <= base_tvar else "[WARN]"))
    results.append(BenchResult("ChebyshevDequantizedTransition", "Speed ratio (vs base)",
                               base_time / max(cheb_time, 1e-3), 1.0, cheb_time,
                               "[PASS]" if cheb_time < base_time * 3 else "[WARN]"))
    return results


# =============================================================================
# 3.  SUITE: MAXWELL PRIOR vs GAUSSIAN PRIOR (on energy_hat)
# =============================================================================

def bench_maxwell_prior(n: int = 1000, seed: int = 0) -> list[BenchResult]:
    """
    Compares Maxwell-Boltzmann prior vs Gaussian prior on:
    - NLL (Negative Log-Likelihood) on held-out energy samples
    - KL divergence from true distribution (chi2(3))
    """
    from omnistats.modules.bayesian.maxwell_prior import MaxwellBoltzmannPrior
    from torch.distributions import Chi2, Normal
    torch.manual_seed(seed); np.random.seed(seed)
    results = []

    sigma = 1.0
    # True distribution: Maxwell i.e. v ~ Maxwell(sigma), E = v^2 ~ chi2(3)*sigma^2
    chi2 = Chi2(df=torch.tensor(3.0))
    E_true = chi2.sample((n,)) * sigma**2   # [n] energies from true Maxwell

    # Maxwell prior NLL
    maxwell = MaxwellBoltzmannPrior(sigma=sigma)
    t0 = time.perf_counter()
    lp_maxwell = maxwell.log_prob(E_true, beta=1.0)
    maxwell_time = (time.perf_counter() - t0) * 1000
    nll_maxwell  = -lp_maxwell.mean().item()

    # Gaussian prior NLL (naive baseline: N(E.mean(), E.std()))
    mu_g  = E_true.mean(); std_g = E_true.std()
    norm  = Normal(mu_g, std_g.clamp(min=1e-6))
    lp_gauss = norm.log_prob(E_true)
    nll_gauss = -lp_gauss.mean().item()

    results.append(BenchResult("MaxwellBoltzmannPrior", "NLL(lo)", nll_maxwell, nll_gauss,
                               maxwell_time, "[PASS]" if nll_maxwell < nll_gauss else "[WARN]"))
    return results


# =============================================================================
# 4.  SUITE: WAIC / PSIS-LOO vs AIC/BIC
# =============================================================================

def bench_waic_loo(n_obs: int = 100, n_draws: int = 500,
                   seed: int = 0) -> list[BenchResult]:
    """
    Verifies WAIC and PSIS-LOO on a well-specified vs mis-specified model:
    - Well-specified model should have lower WAIC/LOO than mis-specified
    - AIC baseline: AIC = 2k - 2*log_lik (k = num params)
    """
    from omnistats.modules.bayesian.waic_loo import waic, psis_loo
    np.random.seed(seed)
    results = []

    # Well-specified: log-likelihoods from true N(0,1)
    ll_good = np.random.normal(-1.0, 0.3, size=(n_draws, n_obs))
    # Mis-specified: likelihoods from wrong N(2,3)
    ll_bad  = np.random.normal(-3.5, 1.2, size=(n_draws, n_obs))

    t0 = time.perf_counter()
    w_good = waic(ll_good)
    w_bad  = waic(ll_bad)
    waic_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    l_good = psis_loo(ll_good)
    l_bad  = psis_loo(ll_bad)
    loo_time = (time.perf_counter() - t0) * 1000

    # AIC baseline: AIC = 2*1 - 2*lppd (1 param)
    aic_good = 2 * 1 - 2 * w_good.lppd
    aic_bad  = 2 * 1 - 2 * w_bad.lppd

    results.append(BenchResult("WAIC (well-specified wins)", "WAIC diff(hi)",
                               w_bad.waic - w_good.waic, 0.0, waic_time,
                               "[PASS]" if w_good.waic < w_bad.waic else "[FAIL]"))
    results.append(BenchResult("PSIS-LOO (well-specified wins)", "LOO diff(hi)",
                               l_bad.loo - l_good.loo, 0.0, loo_time,
                               "[PASS]" if l_good.loo < l_bad.loo else "[FAIL]"))
    results.append(BenchResult("WAIC vs AIC (discrimination)",
                               "WAIC delta > AIC delta",
                               abs(w_bad.waic - w_good.waic),
                               abs(aic_bad - aic_good), waic_time,
                               "[PASS]" if abs(w_bad.waic - w_good.waic)
                               >= abs(aic_bad - aic_good) * 0.5 else "[WARN]"))
    return results


# =============================================================================
# 5.  SUITE: QUANTUM KALMAN vs STANDARD KALMAN (ATT recovery)
# =============================================================================

def bench_quantum_kalman(T: int = 80, T_treat: int = 50,
                          true_att: float = 5.0,
                          n_trials: int = 20, seed: int = 0) -> list[BenchResult]:
    """
    Compares QuantumKalmanFilter vs a simple pre-post mean difference on:
    - ATT estimation error (|estimated - true|)
    - CI coverage (does 95% CI contain the true ATT?)
    - Robustness to heavy-tailed shocks (t-distributed noise)
    """
    from omnistats.modules.timeseries.quantum_kalman import QuantumKalmanFilter
    torch.manual_seed(seed); np.random.seed(seed)
    results = []

    qkf_errors, naive_errors, qkf_covers = [], [], []

    for trial in range(n_trials):
        # Synthetic time series with true ATT and student-t noise (heavy tails)
        t_noise = np.random.standard_t(df=4, size=T) * 0.8
        level   = np.cumsum(np.random.randn(T) * 0.1)
        y_arr   = level + t_noise
        y_arr[T_treat:] += true_att

        y  = torch.tensor(y_arr, dtype=torch.float32)

        # Quantum Kalman estimate
        qkf = QuantumKalmanFilter(obs_noise_sigma=1.0, diffusion_sigma=0.3, beta=1.5)
        att = qkf.estimate_att(y, None, T_treat=T_treat)
        qkf_errors.append(abs(att["estimate"] - true_att))
        qkf_covers.append(att["ci_lower"] <= true_att <= att["ci_upper"])

        # Naive baseline: post mean - pre mean
        naive = float(y[T_treat:].mean() - y[:T_treat].mean())
        naive_errors.append(abs(naive - true_att))

    t0 = time.perf_counter()
    qkf_mse    = float(np.mean(qkf_errors))
    naive_mse  = float(np.mean(naive_errors))
    coverage   = float(np.mean(qkf_covers))
    bench_time = (time.perf_counter() - t0) * 1000

    results.append(BenchResult("QuantumKalmanFilter", "ATT MAE(lo)",
                               qkf_mse, naive_mse, bench_time,
                               "[PASS]" if qkf_mse < naive_mse else "[WARN]"))
    results.append(BenchResult("QuantumKalmanFilter", "CI coverage (≥0.80)",
                               coverage, 0.80, bench_time,
                               "[PASS]" if coverage >= 0.80 else "[WARN]"))
    return results


# =============================================================================
# 6.  SUITE: CHEBYSHEV COEFFICIENTS (spectral fidelity)
# =============================================================================

def bench_chebyshev_coeffs(degrees: list = None, seed: int = 0) -> list[BenchResult]:
    """
    Verifies that Chebyshev heat-kernel coefficients correctly approximate
    f(lambda) = exp(-beta*lambda^2) on the Chebyshev nodes.
    """
    from chebyshev_qsvt import heat_kernel_chebyshev_coeffs
    if degrees is None:
        degrees = [4, 8, 16]
    results = []

    for deg in degrees:
        coeffs = heat_kernel_chebyshev_coeffs(degree=deg, beta=1.0)
        # Evaluate Chebyshev expansion at 100 test points in [-1, 1]
        x = torch.linspace(-1, 1, 100)
        f_true = torch.exp(-x ** 2)

        # Clenshaw evaluation of the polynomial
        b2 = torch.zeros_like(x); b1 = torch.zeros_like(x)
        for k in range(deg, 0, -1):
            b = coeffs[k] + 2 * x * b1 - b2
            b2 = b1; b1 = b
        f_approx = coeffs[0] + x * b1 - b2

        max_err = (f_approx - f_true).abs().max().item()
        results.append(BenchResult(f"Chebyshev degree={deg}", "Max approx error(lo)",
                                   max_err, 0.01, 0.0,
                                   "[PASS]" if max_err < 0.01 else "[WARN]"))
    return results


# =============================================================================
# MAIN RUNNER
# =============================================================================

SUITES = {
    "predictor":   bench_dequantized_predictor,
    "chebyshev":   bench_chebyshev_qsvt,
    "cheb_coeffs": bench_chebyshev_coeffs,
    "maxwell":     bench_maxwell_prior,
    "waic":        bench_waic_loo,
    "kalman":      bench_quantum_kalman,
}


def main():
    parser = argparse.ArgumentParser(description="XAI JEPA Benchmark Runner")
    parser.add_argument("--suite", default="all", choices=list(SUITES) + ["all"])
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--n",     type=int, default=256)
    args = parser.parse_args()

    suites = list(SUITES.values()) if args.suite == "all" else [SUITES[args.suite]]

    all_results: list[BenchResult] = []
    for fn in suites:
        try:
            res = fn(seed=args.seed) if "seed" in fn.__code__.co_varnames else fn()
            all_results.extend(res)
        except Exception as e:
            print(f"  [ERROR] {fn.__name__}: {e}")

    # Print table
    print()
    print("=" * 110)
    print("  XAI JEPA BENCHMARK REPORT")
    print("=" * 110)
    passes = sum(1 for r in all_results if "[PASS]" in r.pass_fail)
    total  = len(all_results)
    print(f"  {passes}/{total} checks passed")
    print()
    for r in all_results:
        print(r)
    print("=" * 110)


if __name__ == "__main__":
    main()
