"""
omnistats/modules/causal/scm.py
--------------------------------
Synthetic Control Method (SCM) — Abadie, Diamond & Hainmueller (2010).

Constructs a "synthetic" control unit as a convex combination of donor
(untreated) units that minimises pre-treatment outcome divergence.

Why SCM instead of basic DiD
------------------------------
  Standard DiD assumes "parallel trends" — that treated and control units
  would have moved in parallel without treatment. SCM drops this assumption.
  Instead, it finds the optimal weighted blend of control units that
  PERFECTLY matches the treated unit's pre-intervention trajectory.
  By matching the full pre-period curve, SCM implicitly controls for
  time-varying unobserved confounders that DiD cannot handle.

APA Explainability
------------------
  The weight vector W* is directly reportable in an APA table:
  "The synthetic control comprised 40% Unit A, 35% Unit C, 25% Unit D."
  This total transparency is unique to SCM among causal methods.

Estimand: Average Treatment Effect on the Treated unit post-intervention
  ATT = Y_treated(post) - Y_synthetic(post)

Library strategy
----------------
  Primary:  cvxpy — convex optimisation for constrained weight solve
  Fallback: scipy.optimize.minimize with SLSQP and non-negativity constraints
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config import OUTPUT_DIR, CAUSAL_USE_SYNTHETIC

# ── cvxpy primary ──────────────────────────────────────────────────────────────
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


# ── Synthetic data generator ───────────────────────────────────────────────────
def _make_synthetic_scm() -> dict:
    """
    Minimal panel for SCM demonstration.
    1 treated unit, 5 donor control units, 20 time periods.
    Treatment starts at t=11. True ATT = 3.0.
    """
    rng = np.random.RandomState(42)
    T = 20
    T_treat = 10   # last pre-treatment period (1-indexed)
    n_donors = 5
    true_att = 3.0

    # Donor units: random walks
    donors = np.cumsum(rng.randn(T, n_donors), axis=0)

    # Treated unit = 0.4*D1 + 0.6*D3 + noise pre-treatment
    treated = 0.4 * donors[:, 0] + 0.6 * donors[:, 2] + rng.randn(T) * 0.3
    # Post-treatment: add true ATT
    treated[T_treat:] += true_att

    donor_labels = [f"Control_{i+1}" for i in range(n_donors)]

    return {
        "Y_treated": treated,
        "Y_donors":  donors,
        "donor_labels": donor_labels,
        "T_treat":   T_treat,
        "n_periods": T,
        "true_att":  true_att,
    }


# ── Convex weight solver ───────────────────────────────────────────────────────
def _solve_weights_cvxpy(Y_pre_treated: np.ndarray,
                          Y_pre_donors: np.ndarray) -> np.ndarray:
    """Solve W* = argmin ||Y_pre_treated - Y_pre_donors @ W||²  s.t. W≥0, sum=1"""
    n_donors = Y_pre_donors.shape[1]
    W = cp.Variable(n_donors, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(Y_pre_donors @ W - Y_pre_treated))
    constraints = [cp.sum(W) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, warm_start=True)
    return W.value if W.value is not None else np.ones(n_donors) / n_donors


def _solve_weights_scipy(Y_pre_treated: np.ndarray,
                          Y_pre_donors: np.ndarray) -> np.ndarray:
    """Fallback: scipy SLSQP"""
    from scipy.optimize import minimize
    n_donors = Y_pre_donors.shape[1]

    def loss(W):
        return np.sum((Y_pre_donors @ W - Y_pre_treated) ** 2)

    constraints = [{"type": "eq", "fun": lambda W: W.sum() - 1}]
    bounds = [(0, 1)] * n_donors
    x0 = np.ones(n_donors) / n_donors
    res = minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x if res.success else x0


# ── Main estimator ─────────────────────────────────────────────────────────────
def synthetic_control(
    Y_treated: np.ndarray | None = None,
    Y_donors:  np.ndarray | None = None,
    donor_labels: list | None    = None,
    T_treat: int | None          = None,
    verbose: bool = True,
) -> dict:
    """
    Synthetic Control Method estimator.

    Parameters
    ----------
    Y_treated    : 1D array, length T — outcome for the treated unit
    Y_donors     : 2D array, shape (T, n_donors) — outcomes for donor units
    donor_labels : list of donor unit names (for APA weight table)
    T_treat      : index of last pre-treatment period (0-indexed, exclusive)
    verbose      : print results

    Returns
    -------
    Standardised result dict (same schema as did.py / iv.py / rdd.py):
      method, estimand, estimate, se, ci_lower, ci_upper, ci_type,
      p_value, n_obs, diagnostics, warnings
    """
    warns = []

    # ── Load synthetic demo if needed ──────────────────────────────────────
    if CAUSAL_USE_SYNTHETIC or Y_treated is None:
        data = _make_synthetic_scm()
        Y_treated    = data["Y_treated"]
        Y_donors     = data["Y_donors"]
        donor_labels = data["donor_labels"]
        T_treat      = data["T_treat"]
        warns.append("Using synthetic SCM demo data (CAUSAL_USE_SYNTHETIC=True)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    T = len(Y_treated)
    n_donors = Y_donors.shape[1]

    if donor_labels is None:
        donor_labels = [f"Donor_{i+1}" for i in range(n_donors)]

    Y_pre_treated = Y_treated[:T_treat]
    Y_pre_donors  = Y_donors[:T_treat, :]
    Y_post_treated = Y_treated[T_treat:]
    Y_post_donors  = Y_donors[T_treat:, :]

    # ── Solve weights ──────────────────────────────────────────────────────
    if _HAS_CVXPY:
        W_star = _solve_weights_cvxpy(Y_pre_treated, Y_pre_donors)
        solver_used = "cvxpy_ECOS"
    else:
        W_star = _solve_weights_scipy(Y_pre_treated, Y_pre_donors)
        solver_used = "scipy_SLSQP"
        warns.append("[WARNING] cvxpy not installed — using scipy SLSQP fallback")

    # ── Synthetic control series ───────────────────────────────────────────
    Y_synthetic_pre  = Y_pre_donors  @ W_star
    Y_synthetic_post = Y_post_donors @ W_star

    # ── ATT estimate (mean post-period gap) ───────────────────────────────
    gaps = Y_post_treated - Y_synthetic_post
    att  = float(np.mean(gaps))

    # ── SE via in-space placebo (each donor gets synthetic control weights) ─
    placebo_atts = []
    for d in range(n_donors):
        Y_d_pre  = Y_donors[:T_treat, d]
        Y_d_post = Y_donors[T_treat:, d]
        Y_donors_minus_d_pre  = np.delete(Y_donors[:T_treat, :],  d, axis=1)
        Y_donors_minus_d_post = np.delete(Y_donors[T_treat:, :], d, axis=1)

        if Y_donors_minus_d_pre.shape[1] == 0:
            continue
        try:
            W_p = (_solve_weights_cvxpy(Y_d_pre, Y_donors_minus_d_pre)
                   if _HAS_CVXPY
                   else _solve_weights_scipy(Y_d_pre, Y_donors_minus_d_pre))
            placebo_atts.append(float(np.mean(Y_d_post - Y_donors_minus_d_post @ W_p)))
        except Exception:
            pass

    if placebo_atts:
        se       = float(np.std(placebo_atts, ddof=1))
        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se
        ci_type  = "placebo_in_space"
        p_value  = float(np.mean(np.abs(placebo_atts) >= abs(att)))
    else:
        se = ci_lower = ci_upper = float("nan")
        ci_type = "not_available"
        p_value = float("nan")
        warns.append("[WARNING] Insufficient donors for placebo inference")

    # ── Pre-treatment fit quality ──────────────────────────────────────────
    pre_rmse = float(np.sqrt(np.mean((Y_pre_treated - Y_synthetic_pre) ** 2)))

    # ── Weight table ───────────────────────────────────────────────────────
    weight_df = pd.DataFrame({
        "donor":  donor_labels,
        "weight": [round(float(w), 4) for w in W_star],
    }).sort_values("weight", ascending=False).reset_index(drop=True)
    weight_df.to_csv(os.path.join(OUTPUT_DIR, "scm_weights.csv"), index=False)

    # ── Gap series CSV ─────────────────────────────────────────────────────
    gap_df = pd.DataFrame({
        "period":          range(T),
        "Y_treated":       Y_treated,
        "Y_synthetic":     np.concatenate([Y_synthetic_pre, Y_synthetic_post]),
        "gap":             np.concatenate([Y_pre_treated - Y_synthetic_pre, gaps]),
        "post_treatment":  [0] * T_treat + [1] * (T - T_treat),
    })
    gap_df.to_csv(os.path.join(OUTPUT_DIR, "scm_gaps.csv"), index=False)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    periods = np.arange(T)
    ax.plot(periods, Y_treated, "k-o", ms=4, label="Treated Unit", linewidth=2)
    ax.plot(periods, np.concatenate([Y_synthetic_pre, Y_synthetic_post]),
            "r--o", ms=4, label="Synthetic Control", linewidth=2)
    ax.axvline(T_treat - 0.5, color="grey", linestyle=":", linewidth=1.5,
               label="Treatment Start")
    ax.set_title("SCM: Treated vs. Synthetic Control", fontweight="bold")
    ax.set_xlabel("Period"); ax.set_ylabel("Outcome")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    all_gaps = np.concatenate([Y_pre_treated - Y_synthetic_pre, gaps])
    ax.bar(periods[:T_treat], Y_pre_treated - Y_synthetic_pre, color="#2E86AB", alpha=0.6,
           label="Pre-treatment gap")
    ax.bar(periods[T_treat:], gaps, color="#E84855", alpha=0.8, label="Post-treatment gap (ATT)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(T_treat - 0.5, color="grey", linestyle=":", linewidth=1.5)
    ax.set_title(f"SCM Gap Series  (ATT = {att:.3f})", fontweight="bold")
    ax.set_xlabel("Period"); ax.set_ylabel("Gap (Treated − Synthetic)")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "scm_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "method":   "Synthetic Control Method (SCM)",
        "estimand": "ATT_treated_unit",
        "estimate": round(att, 4),
        "se":       round(se, 4) if not np.isnan(se) else float("nan"),
        "ci_lower": round(ci_lower, 4) if not np.isnan(ci_lower) else float("nan"),
        "ci_upper": round(ci_upper, 4) if not np.isnan(ci_upper) else float("nan"),
        "ci_type":  ci_type,
        "p_value":  round(p_value, 4) if not np.isnan(p_value) else float("nan"),
        "n_obs":    int(T),
        "diagnostics": {
            "pre_rmse":     round(pre_rmse, 4),
            "n_donors":     n_donors,
            "T_treat":      T_treat,
            "solver":       solver_used,
            "top_donor":    weight_df.iloc[0]["donor"],
            "top_weight":   weight_df.iloc[0]["weight"],
        },
        "warnings": warns,
    }

    if verbose:
        print("\n[Causal] Synthetic Control Method (SCM)")
        print(f"  ATT estimate:      {att:.4f}")
        print(f"  SE (placebo):      {se:.4f}" if not np.isnan(se) else "  SE: N/A")
        print(f"  95% CI:            [{ci_lower:.4f}, {ci_upper:.4f}]" if not np.isnan(ci_lower) else "  95% CI: N/A")
        print(f"  Placebo p-value:   {p_value:.4f}" if not np.isnan(p_value) else "  p-value: N/A")
        print(f"  Pre-treatment RMSE: {pre_rmse:.4f}")
        print(f"  Donor weights:")
        for _, row in weight_df.iterrows():
            if row["weight"] > 0.001:
                print(f"    {row['donor']}: {row['weight']:.3f}")
        print(f"  Saved -> scm_weights.csv, scm_gaps.csv, scm_plot.png")
        if warns:
            for w in warns:
                print(f"  {w}")

    return result
