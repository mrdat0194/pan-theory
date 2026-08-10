"""
omnistats/modules/bayesian/waic_loo.py
---------------------------------------
SOTA Bayesian Model Selection: WAIC and PSIS-LOO.

Implements the two gold-standard Bayesian model comparison criteria that
replace classical AIC/BIC for comparing causal estimators in OmniStats.

WAIC (Watanabe-Akaike Information Criterion)
--------------------------------------------
    WAIC = -2 * (lppd - p_waic)
where:
    lppd   = sum_i log E_theta[p(y_i | theta)]     (log pointwise predictive density)
    p_waic = sum_i Var_theta[log p(y_i | theta)]   (effective number of parameters)

PSIS-LOO (Pareto-Smoothed Importance Sampling Leave-One-Out CV)
---------------------------------------------------------------
Vehtari et al. (2017) show that LOO-CV can be estimated from the full posterior
without refitting using Pareto-smoothed importance sampling:
    LOO = sum_i log p(y_i | y_{-i})  ≈ sum_i IS-weighted log p(y_i | theta^s)

The Pareto tail index k (k-hat) diagnoses reliability:
    k < 0.5  → very reliable
    k < 0.7  → OK
    k > 0.7  → LOO unreliable, refit needed

These metrics are then fed into OmniStats APA reports to automatically
select the best causal estimator (DiD, RDD, IV, SCM, BMA) per experiment.

References
----------
Watanabe, S. (2010). Asymptotic Equivalence of Bayes Cross Validation and
    Widely Applicable Information Criterion in Singular Learning Theory.
Vehtari, A., Gelman, A., Gabry, J. (2017). Practical Bayesian model evaluation
    using leave-one-out cross-validation and WAIC. Statistics and Computing.
"""
from __future__ import annotations

import math
import numpy as np
import warnings
from dataclasses import dataclass


# =============================================================================
# 1.  DATA CLASS FOR RESULTS
# =============================================================================

@dataclass
class WAICResult:
    """
    WAIC computation result.

    Attributes
    ----------
    waic       : float   WAIC score (lower = better model)
    lppd       : float   Log pointwise predictive density
    p_waic     : float   Effective number of parameters
    waic_i     : np.ndarray [N]  Pointwise WAIC contributions
    se         : float   Standard error of WAIC
    """
    waic:   float
    lppd:   float
    p_waic: float
    waic_i: np.ndarray
    se:     float


@dataclass
class LOOResult:
    """
    PSIS-LOO computation result.

    Attributes
    ----------
    loo        : float    LOO-CV score (lower = better model)
    lppd       : float    LOO log pointwise predictive density
    p_loo      : float    Effective number of parameters (LOO)
    loo_i      : np.ndarray [N]  Pointwise LOO contributions
    k_hat      : np.ndarray [N]  Pareto k-hat diagnostics per observation
    se         : float    Standard error of LOO
    n_bad_k    : int      Number of observations with k_hat > 0.7 (unreliable)
    """
    loo:    float
    lppd:   float
    p_loo:  float
    loo_i:  np.ndarray
    k_hat:  np.ndarray
    se:     float
    n_bad_k: int


# =============================================================================
# 2.  WAIC
# =============================================================================

def waic(log_likelihoods: np.ndarray) -> WAICResult:
    """
    Compute WAIC from a matrix of pointwise log-likelihoods.

    Parameters
    ----------
    log_likelihoods : np.ndarray  [S, N]
        S = posterior draws, N = observations.
        log_likelihoods[s, i] = log p(y_i | theta^s)

    Returns
    -------
    WAICResult
    """
    S, N = log_likelihoods.shape

    # lppd: log E_theta[p(y_i | theta)] = log mean_s exp(ll_si)
    # numerically stable via log-sum-exp
    log_mean_exp = (
        np.log(S)
        - np.log(S)
        + np.array([
            _logsumexp(log_likelihoods[:, i]) - math.log(S)
            for i in range(N)
        ])
    )
    lppd = log_mean_exp.sum()

    # p_waic: Var_theta[log p(y_i | theta)] = variance across posterior draws
    p_waic_i = log_likelihoods.var(axis=0)   # [N]
    p_waic   = p_waic_i.sum()

    # WAIC = -2 * (lppd - p_waic)
    waic_i = -2.0 * (log_mean_exp - p_waic_i)
    waic_val = waic_i.sum()
    se = math.sqrt(N * waic_i.var())

    return WAICResult(
        waic   = float(waic_val),
        lppd   = float(lppd),
        p_waic = float(p_waic),
        waic_i = waic_i,
        se     = float(se),
    )


# =============================================================================
# 3.  PSIS-LOO
# =============================================================================

def psis_loo(log_likelihoods: np.ndarray) -> LOOResult:
    """
    Pareto-Smoothed Importance Sampling LOO-CV.

    Parameters
    ----------
    log_likelihoods : np.ndarray  [S, N]
        S = posterior draws, N = observations.

    Returns
    -------
    LOOResult
    """
    S, N = log_likelihoods.shape

    loo_i  = np.empty(N)
    k_hats = np.empty(N)

    for i in range(N):
        ll_i = log_likelihoods[:, i]   # [S]
        # Importance weights: r_s = 1/p(y_i|theta^s) = exp(-ll_i)
        log_r = -ll_i

        # Pareto-smooth the log importance weights
        log_r_smoothed, k_hat = _psis_smooth(log_r)

        # LOO for observation i (importance-weighted log-likelihood)
        log_w = log_r_smoothed - _logsumexp(log_r_smoothed)   # normalized
        loo_i[i] = _logsumexp(ll_i + log_w)
        k_hats[i] = k_hat

    lppd = (
        np.array([_logsumexp(log_likelihoods[:, i]) - math.log(S) for i in range(N)])
        .sum()
    )
    p_loo     = lppd - loo_i.sum()
    loo_score = -2.0 * loo_i.sum()
    se        = math.sqrt(N * ((-2.0 * loo_i) - (loo_score / N)).var())
    n_bad_k   = int((k_hats > 0.7).sum())

    if n_bad_k > 0:
        warnings.warn(
            f"[PSIS-LOO] {n_bad_k}/{N} observations have k_hat > 0.7. "
            "LOO estimate may be unreliable for those points. "
            "Consider refitting or using moment matching.",
            UserWarning,
            stacklevel=2,
        )

    return LOOResult(
        loo     = float(loo_score),
        lppd    = float(lppd),
        p_loo   = float(p_loo),
        loo_i   = loo_i,
        k_hat   = k_hats,
        se      = float(se),
        n_bad_k = n_bad_k,
    )


# =============================================================================
# 4.  MODEL COMPARISON
# =============================================================================

def compare_models(
    model_log_likelihoods: dict[str, np.ndarray],
    criterion: str = "loo",
) -> dict:
    """
    Compare multiple models and rank them by WAIC or LOO-CV.

    Parameters
    ----------
    model_log_likelihoods : dict[str, np.ndarray]
        Keys = model names (e.g. 'DiD', 'RDD', 'IV', 'SCM', 'BMA').
        Values = [S, N] log-likelihood matrices.
    criterion : 'waic' | 'loo'

    Returns
    -------
    dict with keys:
        'ranking'  : list of model names from best to worst
        'scores'   : dict[name -> score]
        'se'       : dict[name -> standard error]
        'winner'   : str  best model name
        'results'  : dict[name -> WAICResult or LOOResult]
    """
    results = {}
    scores  = {}
    ses     = {}

    for name, ll in model_log_likelihoods.items():
        if criterion == "waic":
            r = waic(ll)
        else:
            r = psis_loo(ll)
        results[name] = r
        scores[name]  = r.waic if criterion == "waic" else r.loo
        ses[name]     = r.se

    ranking = sorted(scores, key=lambda n: scores[n])   # lower = better
    winner  = ranking[0]

    return {
        "ranking": ranking,
        "scores":  scores,
        "se":      ses,
        "winner":  winner,
        "results": results,
        "criterion": criterion,
    }


# =============================================================================
# 5.  UTILITIES
# =============================================================================

def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = x.max()
    return float(c + math.log(np.exp(x - c).sum()))


def _psis_smooth(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Pareto-smooth a vector of log importance weights.

    Fits a Generalized Pareto Distribution (GPD) to the upper tail
    and replaces tail values with GPD quantiles for variance reduction.

    Returns
    -------
    log_w_smoothed : np.ndarray  [S]
    k_hat          : float       Pareto tail index (< 0.5 = reliable)
    """
    S = len(log_w)
    # Sort in ascending order
    w = np.exp(log_w - log_w.max())   # unnormalized weights for sorting
    order = np.argsort(w)
    w_sorted = w[order]

    # Tail threshold: top M = min(S/5, 3*sqrt(S)) samples
    M = min(int(S / 5), int(3 * math.sqrt(S)))
    M = max(M, 5)

    tail = w_sorted[-M:]

    # Fit GPD to tail via moment matching (Zhang & Stephens 2009)
    k_hat, sigma = _gpd_fit(tail)

    if k_hat < 0.5 and sigma > 0:
        # Replace tail with GPD quantiles
        p = (np.arange(1, M + 1) - 0.5) / M
        gpd_q = tail[0] + sigma * (np.power(1 - p, -k_hat) - 1) / k_hat
        w_smoothed = w_sorted.copy()
        w_smoothed[-M:] = np.minimum(gpd_q, w_sorted.max())
        w_out = np.empty_like(log_w)
        w_out[order] = w_smoothed
        log_w_smoothed = np.log(w_out.clip(min=1e-300)) + log_w.max()
    else:
        log_w_smoothed = log_w

    return log_w_smoothed, float(k_hat)


def _gpd_fit(tail: np.ndarray) -> tuple[float, float]:
    """
    Fit Generalized Pareto Distribution to tail using the
    Zhang & Stephens (2009) method.

    Returns (k_hat, sigma_hat).
    """
    n = len(tail)
    if n < 5:
        return 0.0, float(tail.std())

    x = tail - tail.min()
    if x.max() < 1e-10:
        return 0.0, 1.0

    x = x / x.max()

    # Method of moments
    x_mean = x.mean()
    x_var  = x.var()

    if x_var < 1e-10:
        return 0.0, float(tail.std())

    k_hat     = 0.5 * (1 - x_mean**2 / x_var)
    sigma_hat = x_mean * (1 - k_hat) * tail.max()

    return float(k_hat), float(sigma_hat)
