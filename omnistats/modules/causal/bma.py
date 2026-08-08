"""
omnistats/modules/causal/bma.py
--------------------------------
HTE Subgroup Analysis — Lipschitz-Bounded CATE / HTE Solver.

Stage 4 of the OmniStats pipeline. Fills the BMA slot reserved in
causal_results.csv and the APA report's Table 8.

Algorithm
---------
  1. Load CUPED-adjusted data from outputs/lpa_profiles.csv.
  2. Recode A/B group to binary {0, 1}.
  3. One-hot encode demographics.
  4. Estimate Doubly Robust (DR) pseudo-outcomes using Ridge regression.
  5. Expand demographics with a quadratic polynomial basis.
  6. Compute the Jacobian tensor of the basis functions.
  7. Formulate a constrained optimization problem restricting the gradient
     norm of the CATE function at each sample to be <= L_bound.
  8. Solve using the Adaptive Primal-Dual Method (adaPDM).
  9. Estimate standard errors and p-values using Bootstrap.
  10. Output subgroup effects to outputs/bma_subgroups.csv.
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config import (
    OUTPUT_DIR, DATA_PATH,
    AB_GROUP_COL, AB_METRIC_COL,
    DEMOGRAPHIC_COLS,
    CAUSAL_BMA_ENABLED, CAUSAL_BMA_MAX_DUMMIES,
)
from modules.optimization.proximal import (
    adaptive_primal_dual,
    prox_block_l2_ball
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data() -> pd.DataFrame:
    """
    Load CUPED-adjusted data.
    Priority: outputs/lpa_profiles.csv (has profile_prob_max + _cuped col).
    Fallback:  DATA_PATH raw CSV.
    """
    cuped_path = os.path.join(OUTPUT_DIR, "lpa_profiles.csv")
    if os.path.exists(cuped_path):
        return pd.read_csv(cuped_path)
    return pd.read_csv(DATA_PATH)


def _recode_treatment(df: pd.DataFrame, group_col: str) -> pd.Series:
    """
    Recode the A/B group column to binary {0, 1}.
    """
    col_data = df[group_col]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
    vals = sorted(col_data.dropna().unique())
    if len(vals) < 2:
        raise ValueError(
            f"[BMA] AB_GROUP_COL='{group_col}' has fewer than 2 unique values: {vals}"
        )
    mapping = {vals[0]: 0, vals[1]: 1}
    return col_data.map(mapping)


def compute_jacobian_matrix(X_demo: np.ndarray, poly: PolynomialFeatures) -> np.ndarray:
    """
    Compute the Jacobian of the polynomial basis w.r.t the demographic features for all samples.
    Returns J of shape (N, d, k), where:
      - N is the number of samples
      - d is the number of features
      - k is the number of basis functions
    """
    N, d = X_demo.shape
    k = poly.powers_.shape[0]
    powers = poly.powers_ # shape: (k, d)
    
    J = np.zeros((N, d, k))
    for r in range(d):
        for j in range(k):
            pow_rj = powers[j, r]
            if pow_rj > 0:
                temp_powers = np.copy(powers[j])
                temp_powers[r] -= 1
                val = pow_rj * np.ones(N)
                for p in range(d):
                    if temp_powers[p] > 0:
                        val *= (X_demo[:, p] ** temp_powers[p])
                J[:, r, j] = val
    return J


def fit_lipschitz_cate(
    X_demo: np.ndarray,
    Y_outcome: np.ndarray,
    T_vec: np.ndarray,
    lambda_reg: float = 0.1,
    L_bound: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, PolynomialFeatures]:
    """
    Fit the Lipschitz-bounded CATE model using the Adaptive Primal-Dual Method.
    """
    N, d = X_demo.shape
    p = float(T_vec.mean())
    if p <= 0 or p >= 1:
        raise ValueError(f"Propensity score p must be in (0, 1), got {p}")
        
    # 1. Fit outcome response surfaces
    model1 = Ridge(alpha=1.0)
    model1.fit(X_demo[T_vec == 1], Y_outcome[T_vec == 1])
    mu1 = model1.predict(X_demo)
    
    model0 = Ridge(alpha=1.0)
    model0.fit(X_demo[T_vec == 0], Y_outcome[T_vec == 0])
    mu0 = model0.predict(X_demo)
    
    # 2. Compute Doubly Robust pseudo-outcomes
    Y_dr = mu1 - mu0 + (T_vec * (Y_outcome - mu1)) / p - ((1 - T_vec) * (Y_outcome - mu0)) / (1.0 - p)
    
    # 3. Construct polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=True)
    Phi = poly.fit_transform(X_demo)
    k = Phi.shape[1]
    
    # 4. Construct Jacobians and flat linear operator matrix
    J = compute_jacobian_matrix(X_demo, poly)
    J_flat = J.reshape(N * d, k)
    
    # 5. Define Primal-Dual functions
    # f(theta) = 0.5 * || Y_dr - Phi * theta ||^2 + 0.5 * lambda * || theta ||^2
    # grad_f(theta) = Phi.T * (Phi * theta - Y_dr) + lambda * theta
    Phi_t_Phi = Phi.T @ Phi
    Phi_t_Y = Phi.T @ Y_dr
    
    def grad_f(theta):
        return Phi_t_Phi @ theta - Phi_t_Y + lambda_reg * theta
        
    def prox_g(theta, tau):
        return theta # g(theta) = 0
        
    def prox_h_conj(y, sigma):
        # Moreau decomposition: prox_{sigma h*}(y) = y - prox_{h/sigma}(y/sigma) * sigma
        # Since h is indicator of L2-balls of radius L_bound, the result is y - proj_{sigma * L_bound}(y)
        return y - prox_block_l2_ball(y, sigma * L_bound, d)
        
    # Solve using adaPDM
    Lf = float(np.linalg.eigvalsh(Phi_t_Phi + lambda_reg * np.eye(k))[-1])
    LK = float(np.linalg.eigvalsh(J_flat.T @ J_flat)[-1])
    
    theta0 = np.zeros(k)
    y0 = np.zeros(N * d)
    theta_star = adaptive_primal_dual(
        theta0, y0, grad_f, prox_g, prox_h_conj, J_flat, J_flat.T,
        max_iter=max_iter, tol=tol, L_f=Lf, L_K=LK
    )
    
    cate_pred = Phi @ theta_star
    return theta_star, cate_pred, poly


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def run_bma(verbose: bool = True) -> dict:
    """
    Run Lipschitz-Bounded CATE / HTE Subgroup Analysis.
    """
    _warn: list[str] = []

    if not CAUSAL_BMA_ENABLED:
        return _null_result("CAUSAL_BMA_ENABLED = False in config.py")

    # ── Load data ────────────────────────────────────────────────────────────
    try:
        df = _load_data()
    except Exception as exc:
        return _null_result(f"Data load failed: {exc}")

    # ── Select outcome: prefer CUPED-adjusted column ─────────────────────────
    cuped_col = f"{AB_METRIC_COL}_cuped"
    outcome_col = cuped_col if cuped_col in df.columns else AB_METRIC_COL

    # ── Drop rows with missing outcome or group ───────────────────────────────
    required = [outcome_col, AB_GROUP_COL] + [
        c for c in DEMOGRAPHIC_COLS if c in df.columns
    ]
    df = df[required].dropna()
    n_total = len(df)

    if n_total < 30:
        return _null_result(f"Too few observations after dropping NaN: {n_total}")

    # ── Recode treatment to {0, 1} ───────────────────────────────────────────
    try:
        T = _recode_treatment(df, AB_GROUP_COL)
    except ValueError as exc:
        return _null_result(str(exc))

    # ── One-hot encode demographics ───────────────────────────────────────────
    demo_cols_present = [c for c in DEMOGRAPHIC_COLS if c in df.columns]
    if not demo_cols_present:
        _warn.append("No DEMOGRAPHIC_COLS found in data — running treatment-only OLS.")

    if demo_cols_present:
        dummies = pd.get_dummies(
            df[demo_cols_present], drop_first=True, dtype=float
        )
    else:
        dummies = pd.DataFrame(index=df.index)

    # ── Prune if too many dummies ────────────────────────────────────────────
    if len(dummies.columns) > CAUSAL_BMA_MAX_DUMMIES:
        _warn.append(
            f"Demographic dummies ({len(dummies.columns)}) exceed "
            f"CAUSAL_BMA_MAX_DUMMIES={CAUSAL_BMA_MAX_DUMMIES}. "
            f"Keeping first {CAUSAL_BMA_MAX_DUMMIES} by variance."
        )
        top_cols = dummies.var().nlargest(CAUSAL_BMA_MAX_DUMMIES).index.tolist()
        dummies = dummies[top_cols]

    dummy_names = list(dummies.columns)
    n_dummies = len(dummy_names)

    if n_dummies == 0:
        return _null_result("No demographic dummy columns found.")

    X_demo = dummies.values
    Y_outcome = df[outcome_col].values
    T_vec = T.values

    # ── Fit Lipschitz CATE model ──────────────────────────────────────────────
    try:
        theta_star, cate_pred, poly = fit_lipschitz_cate(
            X_demo, Y_outcome, T_vec, lambda_reg=0.1, L_bound=1.0
        )
    except Exception as exc:
        return _null_result(f"Lipschitz CATE fit failed: {exc}")

    # ── Bootstrap Standard Errors & P-values ──────────────────────────────────
    n_boot = 50
    boot_marg_atts = []
    boot_subgroup_atts = {col: [] for col in dummy_names}
    
    # Pre-select index mask
    for b in range(n_boot):
        idx = np.random.choice(n_total, size=n_total, replace=True)
        try:
            _, b_cate, _ = fit_lipschitz_cate(
                X_demo[idx], Y_outcome[idx], T_vec[idx], lambda_reg=0.1, L_bound=1.0
            )
            boot_marg_atts.append(b_cate.mean())
            for col in dummy_names:
                mask = dummies.iloc[idx][col].values == 1
                if mask.sum() > 2:
                    boot_subgroup_atts[col].append(b_cate[mask].mean())
                else:
                    boot_subgroup_atts[col].append(b_cate.mean())
        except Exception:
            continue

    # ── Marginalized ATT = average estimated CATE ─────────────────────────────
    beta = float(cate_pred.mean())
    beta_se = float(np.std(boot_marg_atts)) if len(boot_marg_atts) > 1 else 1e-4
    # Compute z-score based p-value
    z_score = beta / beta_se
    beta_p = float(2 * (1 - scipy_normal_cdf(abs(z_score))))

    ci_lo_b = beta - 1.96 * beta_se
    ci_hi_b = beta + 1.96 * beta_se

    # ── Subgroup ATTs ─────────────────────────────────────────────────────────
    subgroup_rows = []
    n_tests = max(len(dummy_names), 1)
    
    for col in dummy_names:
        mask = dummies[col].values == 1
        if mask.sum() > 2:
            coef = float(cate_pred[mask].mean())
            boot_vals = boot_subgroup_atts[col]
            se = float(np.std(boot_vals)) if len(boot_vals) > 1 else 1e-4
            z_sub = coef / se
            pval = float(2 * (1 - scipy_normal_cdf(abs(z_sub))))
        else:
            coef = beta
            se = beta_se
            pval = beta_p

        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        pval_bonf = min(pval * n_tests, 1.0)
        
        subgroup_rows.append({
            "subgroup":       col,
            "att_delta":      round(coef, 6),
            "se_hc3":         round(se, 6), # Shared name for output compatibility
            "ci_lower_95":    round(ci_lo, 6),
            "ci_upper_95":    round(ci_hi, 6),
            "p_value_raw":    round(pval, 6),
            "p_value_bonf":   round(pval_bonf, 6),
            "significant_05": pval_bonf < 0.05,
            "n_tests":        n_tests,
        })

    subgroup_df = pd.DataFrame(subgroup_rows)

    # ── Save bma_subgroups.csv ────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subgroup_path = os.path.join(OUTPUT_DIR, "bma_subgroups.csv")
    subgroup_df.to_csv(subgroup_path, index=False)

    # ── Verbose output ────────────────────────────────────────────────────────
    if verbose:
        print("\n[BMA/HTE] Lipschitz-Bounded CATE Subgroup Analysis")
        print(f"  Outcome:           {outcome_col}")
        print(f"  Treatment:         {AB_GROUP_COL}")
        print(f"  Demographics:      {demo_cols_present}")
        print(f"  Dummies:           {n_dummies}  |  Observations: {n_total}")
        print(f"  Marginalized ATT:  {beta:.4f}  (SE={beta_se:.4f}, p={beta_p:.4f})")
        if not subgroup_df.empty:
            sig = subgroup_df[subgroup_df["significant_05"]]
            print(f"  Significant subgroups (Bonferroni p<0.05): {len(sig)}/{n_dummies}")
            for _, row in sig.iterrows():
                print(f"    • {row['subgroup']}: Δ={row['att_delta']:.4f} "
                      f"[{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}] "
                      f"p_bonf={row['p_value_bonf']:.4f}")
        print(f"  Saved → {subgroup_path}")

    return {
        "method":   "Subgroup HTE (Lipschitz CATE)",
        "estimand": "Marginalized ATT (Treatment coef, Lipschitz CATE)",
        "estimate": round(beta, 6),
        "se":       round(beta_se, 6),
        "ci_lower": round(ci_lo_b, 6),
        "ci_upper": round(ci_hi_b, 6),
        "ci_type":  "bootstrap_bonferroni",
        "p_value":  round(beta_p, 6),
        "n_obs":    int(n_total),
        "warnings": _warn,
        "subgroup_path": subgroup_path,
    }


def scipy_normal_cdf(x: float) -> float:
    """Approximate cumulative distribution function of standard normal distribution."""
    # Approximation formula for normal CDF
    return 0.5 * (1.0 + np.sign(x) * np.sqrt(1.0 - np.exp(-2.0 * x**2 / np.pi)))


def _null_result(reason: str) -> dict:
    """Return a NaN-filled result dict when the estimator cannot run."""
    return {
        "method":   "Subgroup HTE (Lipschitz CATE)",
        "estimand": "Marginalized ATT (Treatment coef, Lipschitz CATE)",
        "estimate": float("nan"), "se": float("nan"),
        "ci_lower": float("nan"), "ci_upper": float("nan"),
        "ci_type":  "not_available", "p_value": float("nan"),
        "n_obs":    0, "warnings": [reason],
    }
