"""
omnistats/modules/bayesian/normal_model.py
-------------------------------------------
Bayesian means test for continuous metrics.

Primary   : PyMC 5 (NUTS sampler) with a StudentT likelihood.
            StudentT is robust to outliers and heavy-tailed revenue data.
IS Fallback: When pymc is not installed, uses a Normal-Normal importance
             sampling estimate via a Gaussian proposal on (μ_A, μ_B).

Migrated conceptual space from:
  Bayesian/mcmc_bayesian.py  (MCMC reference, now replaced by PyMC NUTS)
  Bayesian/importance_sampling_bayesian.py (IS fallback pattern)

Why PyMC over hand-rolled MCMC
-------------------------------
  - NUTS auto-tunes step sizes and trajectory lengths (no manual burn-in)
  - Near-zero autocorrelation: each NUTS sample is near-independent
  - Built-in diagnostics: R-hat convergence, ESS, divergence warnings
  - Multi-parameter posteriors handled automatically (μ, σ, ν jointly)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config import BAYES_AB_SEED


import pymc as pm
import arviz as az


# ── Main function ──────────────────────────────────────────────────────────────

def bayesian_means_test(
    control: np.ndarray,
    treatment: np.ndarray,
    label_control: str  = "Control",
    label_treatment: str= "Treatment",
    n_samples: int      = 2_000,
    tune: int           = 1_000,
    seed: int           = None,
    threshold: float    = 0.95,
    loss_thresh: float  = 0.01,
    verbose: bool       = True,
) -> dict:
    """
    Bayesian comparison of two group means using PyMC NUTS with StudentT likelihood.

    Parameters
    ----------
    control / treatment  : observed metric arrays for each group
    n_samples            : posterior draws (PyMC)
    tune                 : warm-up draws discarded by PyMC
    threshold            : P(B > A) threshold for decision
    loss_thresh          : Expected Loss threshold for decision

    Returns
    -------
    dict with keys: mean_ctrl, mean_treat, diff,
                    p_b_beats_a, expected_loss,
                    ci_lower, ci_upper,
                    method, ess, decision, r_hat
    """
    if seed is None:
        seed = BAYES_AB_SEED

    ctrl = np.asarray(control,   dtype=float)
    trt  = np.asarray(treatment, dtype=float)

    mean_ctrl  = ctrl.mean()
    mean_treat = trt.mean()
    diff       = mean_treat - mean_ctrl

    # ── PyMC primary ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n[Bayesian A/B] Means Test ({label_control} vs {label_treatment}) — PyMC NUTS")

    with pm.Model() as model:
        # Priors
        mu_A    = pm.Normal("mu_A",    mu=ctrl.mean(), sigma=ctrl.std() * 10 + 1e-6)
        mu_B    = pm.Normal("mu_B",    mu=trt.mean(),  sigma=trt.std()  * 10 + 1e-6)
        sigma_A = pm.HalfNormal("sigma_A", sigma=ctrl.std() + 1e-6)
        sigma_B = pm.HalfNormal("sigma_B", sigma=trt.std()  + 1e-6)
        nu      = pm.Exponential("nu", lam=1.0 / 30.0)   # StudentT df

        # Likelihood (StudentT is robust to outliers)
        _ = pm.StudentT("obs_A", nu=nu, mu=mu_A, sigma=sigma_A, observed=ctrl)
        _ = pm.StudentT("obs_B", nu=nu, mu=mu_B, sigma=sigma_B, observed=trt)

        # Delta
        delta = pm.Deterministic("delta", mu_B - mu_A)

        # Sample
        idata = pm.sample(
            n_samples,
            tune=tune,
            random_seed=seed,
            progressbar=verbose,
            target_accept=0.9,
            return_inferencedata=True,
        )

    delta_samples = idata.posterior["delta"].values.flatten()
    mu_A_samples  = idata.posterior["mu_A"].values.flatten()
    mu_B_samples  = idata.posterior["mu_B"].values.flatten()

    p_b_beats_a   = float(np.mean(delta_samples > 0))
    expected_loss = float(np.mean(np.maximum(0.0, mu_A_samples - mu_B_samples)))
    ci_lower      = float(np.percentile(delta_samples, 2.5))
    ci_upper      = float(np.percentile(delta_samples, 97.5))

    # R-hat convergence diagnostic
    rhat_summary = az.summary(idata, var_names=["delta"])
    r_hat = float(rhat_summary["r_hat"].iloc[0]) if "r_hat" in rhat_summary else None

    ess_bulk = float(az.ess(idata, var_names=["delta"])["delta"].values) if hasattr(az, "ess") else float(n_samples)

    method = "pymc_nuts_studentt"

    # ── Decision ──────────────────────────────────────────────────────────
    if p_b_beats_a >= threshold and expected_loss <= loss_thresh:
        decision = "SHIP_TREATMENT"
    elif (1 - p_b_beats_a) >= threshold and expected_loss <= loss_thresh:
        decision = "KEEP_CONTROL"
    else:
        decision = "CONTINUE_COLLECTING"

    result = {
        "mean_ctrl":     round(float(mean_ctrl),  4),
        "mean_treat":    round(float(mean_treat),  4),
        "diff":          round(float(diff),        4),
        "p_b_beats_a":   round(p_b_beats_a,        4),
        "expected_loss": round(expected_loss,       6),
        "ci_lower":      round(ci_lower,            4),
        "ci_upper":      round(ci_upper,            4),
        "method":        method,
        "ess":           round(ess_bulk,            1),
        "r_hat":         round(r_hat, 3) if r_hat is not None else None,
        "decision":      decision,
        "threshold":     threshold,
        "loss_thresh":   loss_thresh,
        "n_ctrl":        len(ctrl),
        "n_treat":       len(trt),
    }

    if verbose:
        print(f"  {label_control}:   mean={mean_ctrl:.4f},  n={len(ctrl)}")
        print(f"  {label_treatment}: mean={mean_treat:.4f}, n={len(trt)}")
        print(f"  Diff (B - A):  {diff:+.4f}")
        print(f"  P(B > A):      {p_b_beats_a:.4f}  (threshold={threshold})")
        print(f"  Expected Loss: {expected_loss:.6f}  (threshold={loss_thresh})")
        print(f"  95% Credible:  [{ci_lower:.4f}, {ci_upper:.4f}]")
        if r_hat:
            print(f"  R-hat (delta): {r_hat:.3f}  (< 1.01 = good convergence)")
        print(f"  Decision:      {decision}  ({method})")

    return result
