"""
omnistats/modules/bayesian/sequential.py
-----------------------------------------
Sequential Bayesian A/B monitoring with Sequential Importance Resampling (SIR).

As each data batch arrives, the posterior is updated and a stopping rule is
evaluated. This solves the "peaking problem" of frequentist fixed-horizon tests:
peeking at p-values during data collection inflates Type-I error. The Bayesian
posterior threshold is valid at any sample size.

SIR connection (from importance_sampling_bayesian.py CRPImportanceSampler)
---------------------------------------------------------------------------
When continuous Beta-Binomial posteriors are updated with each new batch,
we can reweight previously drawn IS samples with the likelihood increment
instead of redrawing from scratch — this is Sequential Importance Resampling.
"""

import numpy as np
from typing import List, Dict, Optional


def expected_loss(post_A_samples: np.ndarray, post_B_samples: np.ndarray) -> float:
    """
    EVSI-style Expected Loss for shipping Treatment B.

    EL = E[max(0, θ_A - θ_B) | D]

    Interpretation: The expected amount we lose (in units of conversion rate or
    mean metric) by shipping Treatment B if Control A is actually better.
    When EL is below the business threshold, it is safe to stop the test.

    Parameters
    ----------
    post_A_samples : posterior samples for Control group parameter
    post_B_samples : posterior samples for Treatment group parameter

    Returns
    -------
    float — expected loss in the same units as the outcome metric
    """
    return float(np.mean(np.maximum(0.0, post_A_samples - post_B_samples)))


def sequential_monitor(
    batches: List[Dict],
    group_col: str,
    metric_col: str,
    conversion_col: Optional[str] = None,
    prior_alpha: float = 1.0,
    prior_beta: float  = 1.0,
    stop_threshold: float = 0.95,
    loss_threshold: float = 0.01,
    n_samples: int = 30_000,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict]:
    """
    Batch-by-batch Bayesian sequential monitoring.

    Iterates over data batches (list of DataFrames), accumulates group statistics,
    updates the Beta posterior, and evaluates the stopping rule at each step.

    Uses Sequential Importance Resampling (SIR) conceptually:
    instead of redrawing from scratch each batch, Beta parameters are
    simply incremented — equivalent to reweighting prior IS samples with
    the new batch likelihood.

    Parameters
    ----------
    batches         : list of DataFrames (each is one observation batch)
    group_col       : column with exactly 2 unique group labels
    metric_col      : continuous outcome (used for reporting, not direct Bayes here)
    conversion_col  : binary (0/1) column for proportion-based monitoring
    prior_alpha     : Beta prior α (both groups, uniform default)
    prior_beta      : Beta prior β
    stop_threshold  : P(B > A) at which to declare a winner
    loss_threshold  : Expected Loss at which stopping is safe
    n_samples       : posterior MC samples per batch
    seed            : random seed

    Returns
    -------
    List[dict] — one entry per batch, each containing:
      batch_idx, n_obs_cumulative,
      p_b_beats_a, expected_loss,
      ci_lower, ci_upper,
      decision, stop_flag
    """
    import pandas as pd

    rng = np.random.default_rng(seed)

    if not batches:
        return []

    # Detect group labels from first batch
    all_groups = batches[0][group_col].unique()
    if len(all_groups) < 2:
        raise ValueError(f"[Sequential] Need 2 groups in '{group_col}'. Found: {all_groups}")
    g_a, g_b = all_groups[0], all_groups[1]

    # Cumulative counts
    alpha_A = prior_alpha
    beta_A  = prior_beta
    alpha_B = prior_alpha
    beta_B  = prior_beta
    n_obs   = 0

    results = []

    for i, batch in enumerate(batches):
        df = batch if isinstance(batch, pd.DataFrame) else pd.DataFrame(batch)

        if conversion_col and conversion_col in df.columns:
            n_A_new    = len(df[df[group_col] == g_a])
            conv_A_new = int(df[df[group_col] == g_a][conversion_col].sum())
            n_B_new    = len(df[df[group_col] == g_b])
            conv_B_new = int(df[df[group_col] == g_b][conversion_col].sum())

            # Conjugate Beta update (= SIR reweighting in closed form)
            alpha_A += conv_A_new
            beta_A  += (n_A_new - conv_A_new)
            alpha_B += conv_B_new
            beta_B  += (n_B_new - conv_B_new)
            n_obs   += n_A_new + n_B_new

        else:
            # For continuous: use sample size as proxy; real use needs PyMC update
            n_A_new = len(df[df[group_col] == g_a])
            n_B_new = len(df[df[group_col] == g_b])
            n_obs  += n_A_new + n_B_new

        # Draw posterior samples
        theta_A = rng.beta(alpha_A, beta_A, size=n_samples)
        theta_B = rng.beta(alpha_B, beta_B, size=n_samples)

        p_b_beats_a   = float(np.mean(theta_B > theta_A))
        el            = expected_loss(theta_A, theta_B)
        diff_samples  = theta_B - theta_A
        ci_lower      = float(np.percentile(diff_samples, 2.5))
        ci_upper      = float(np.percentile(diff_samples, 97.5))

        # Stopping rule
        if p_b_beats_a >= stop_threshold and el <= loss_threshold:
            decision  = "SHIP_TREATMENT"
            stop_flag = True
        elif (1 - p_b_beats_a) >= stop_threshold and el <= loss_threshold:
            decision  = "KEEP_CONTROL"
            stop_flag = True
        else:
            decision  = "CONTINUE_COLLECTING"
            stop_flag = False

        entry = {
            "batch_idx":         i,
            "n_obs_cumulative":  n_obs,
            "alpha_A":           round(alpha_A, 2),
            "beta_A":            round(beta_A,  2),
            "alpha_B":           round(alpha_B, 2),
            "beta_B":            round(beta_B,  2),
            "p_b_beats_a":       round(p_b_beats_a, 4),
            "expected_loss":     round(el, 6),
            "ci_lower":          round(ci_lower, 4),
            "ci_upper":          round(ci_upper, 4),
            "decision":          decision,
            "stop_flag":         stop_flag,
        }
        results.append(entry)

        if verbose:
            flag = "*** STOP ***" if stop_flag else ""
            print(f"  [Batch {i+1:02d}] n={n_obs:,}  P(B>A)={p_b_beats_a:.4f}"
                  f"  EL={el:.5f}  {decision} {flag}")

        if stop_flag:
            if verbose:
                print(f"[Sequential] Stopping rule met at batch {i+1}. "
                      f"Decision: {decision}")
            break

    return results
