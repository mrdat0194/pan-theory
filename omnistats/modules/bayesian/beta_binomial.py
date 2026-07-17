"""
omnistats/modules/bayesian/beta_binomial.py
--------------------------------------------
Bayesian proportion test using the Beta-Binomial conjugate model.

Primary path  : Conjugate closed-form Beta update (exact, no sampling needed).

Migrated from:
  Bayesian/abtesting/abtesting_suite.py

Mathematical basis
------------------
  Prior:   θ_A ~ Beta(α₀, β₀),  θ_B ~ Beta(α₀, β₀)
  Update:  θ_A|D ~ Beta(α₀ + k_A, β₀ + n_A - k_A)
           θ_B|D ~ Beta(α₀ + k_B, β₀ + n_B - k_B)

  Posterior probability via Monte Carlo:
    P(θ_B > θ_A | D) = mean(θ_B_samples > θ_A_samples)

  Expected Loss (EVSI):
    EL = E[max(0, θ_A - θ_B) | D]
"""

import numpy as np
from scipy.special import betaln


# ── Main function ──────────────────────────────────────────────────────────────

def bayesian_proportion_test(
    n_control: int,
    conv_control: int,
    n_treatment: int,
    conv_treatment: int,
    prior_alpha: float = 1.0,
    prior_beta: float  = 1.0,
    threshold: float   = 0.95,
    loss_thresh: float = 0.01,
    n_samples: int     = 50_000,
    seed: int          = 42,
    verbose: bool      = True,
) -> dict:
    """
    Bayesian proportion test: Beta-Binomial conjugate update.

    Parameters
    ----------
    n_control / n_treatment         : group sample sizes
    conv_control / conv_treatment   : conversion counts
    prior_alpha / prior_beta        : Beta prior parameters (default = Uniform)
    threshold                       : P(B > A) to declare treatment winner
    loss_thresh                     : Expected Loss threshold to stop test
    n_samples                       : posterior MC samples for probability calc

    Returns
    -------
    dict with keys:
      p_ctrl, p_treat, lift_pct,
      p_b_beats_a, expected_loss,
      ci_lower, ci_upper,
      method, ess,
      decision, threshold, loss_thresh
    """
    rng = np.random.default_rng(seed)

    # ── Conjugate posterior parameters ────────────────────────────────────
    alpha_A = prior_alpha + conv_control
    beta_A  = prior_beta  + (n_control - conv_control)
    alpha_B = prior_alpha + conv_treatment
    beta_B  = prior_beta  + (n_treatment - conv_treatment)

    # ── Sample posteriors ─────────────────────────────────────────────────
    theta_A = rng.beta(alpha_A, beta_A, size=n_samples)
    theta_B = rng.beta(alpha_B, beta_B, size=n_samples)

    method = "conjugate_beta_binomial"
    ess    = float(n_samples)     # independent samples → ESS = N

    # ── Posterior summary ─────────────────────────────────────────────────
    p_b_beats_a  = float(np.mean(theta_B > theta_A))
    expected_loss = float(np.mean(np.maximum(0.0, theta_A - theta_B)))

    diff = theta_B - theta_A
    ci_lower = float(np.percentile(diff, 2.5))
    ci_upper = float(np.percentile(diff, 97.5))

    p_ctrl  = conv_control  / n_control  if n_control  > 0 else float("nan")
    p_treat = conv_treatment / n_treatment if n_treatment > 0 else float("nan")
    lift_pct = (p_treat - p_ctrl) / p_ctrl * 100 if p_ctrl > 0 else float("nan")

    # ── Decision ──────────────────────────────────────────────────────────
    if p_b_beats_a >= threshold and expected_loss <= loss_thresh:
        decision = "SHIP_TREATMENT"
    elif (1 - p_b_beats_a) >= threshold and expected_loss <= loss_thresh:
        decision = "KEEP_CONTROL"
    else:
        decision = "CONTINUE_COLLECTING"

    result = {
        "p_ctrl":         round(p_ctrl,  4),
        "p_treat":        round(p_treat, 4),
        "lift_pct":       round(lift_pct, 2),
        "p_b_beats_a":    round(p_b_beats_a, 4),
        "expected_loss":  round(expected_loss, 6),
        "ci_lower":       round(ci_lower, 4),
        "ci_upper":       round(ci_upper, 4),
        "method":         method,
        "ess":            round(ess, 1),
        "decision":       decision,
        "threshold":      threshold,
        "loss_thresh":    loss_thresh,
    }

    if verbose:
        print("\n[Bayesian A/B] Proportion Test (Beta-Binomial)")
        print(f"  Control:       {p_ctrl:.2%}  (n={n_control}, conversions={conv_control})")
        print(f"  Treatment:     {p_treat:.2%} (n={n_treatment}, conversions={conv_treatment})")
        print(f"  Lift:          {lift_pct:+.2f}%")
        print(f"  P(B > A):      {p_b_beats_a:.4f}  (threshold={threshold})")
        print(f"  Expected Loss: {expected_loss:.6f}  (threshold={loss_thresh})")
        print(f"  95% Credible Interval on diff: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Decision:      {decision}  ({method})")

    return result
