"""
omnistats/modules/causal/iv.py
-------------------------------
Robust Instrumental Variables / Two-Stage Least Squares estimator.

Primary  : linearmodels >= 6.0  (IV2SLS with robust SEs, KP rk-F stat)
Fallback : Anderson-Rubin CI via ivmodels >= 0.4 when KP rk-F < 10
           (identification-robust CI remains valid even under weak instruments)

Estimand : Local Average Treatment Effect (LATE / Wald-IV)

Return schema (standardised)
  method   : "IV 2SLS (linearmodels)"
  estimand : "LATE"
  estimate : 2SLS coefficient on treatment
  se       : HC3 heteroscedasticity-robust SE
  ci_lower / ci_upper : 95 % CI (AR if weak, Wald otherwise)
  ci_type  : "wald_hc3" | "anderson_rubin"
  p_value  : two-sided Wald p (AR p if weak)
  n_obs    : number of observations
  diagnostics : dict
      kp_rk_f        -- Kleibergen-Paap rk F-statistic
      first_stage_f  -- first-stage F on excluded instruments
      weak_instrument -- bool
  warnings : list of strings
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config import OUTPUT_DIR, CAUSAL_USE_SYNTHETIC


_RNG = np.random.RandomState(42)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _make_synthetic_iv() -> pd.DataFrame:
    """
    n=600: binary instrument Z, unobserved confounder U, endogenous D, outcome Y.
    True LATE = 3.0.
    """
    n  = 600
    z  = _RNG.binomial(1, 0.5, n)               # random instrument
    u  = _RNG.normal(0, 1, n)                    # unobserved confounder
    d  = ((0.6 * z + 0.4 * u + _RNG.normal(0, 0.3, n)) > 0).astype(float)
    y  = 5 + 3.0 * d + 2 * u + _RNG.normal(0, 1, n)
    return pd.DataFrame({"outcome": y, "treatment": d, "instrument": z})


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def iv_2sls(
    df: pd.DataFrame    = None,
    outcome_col:   str  = "outcome",
    treatment_col: str  = "treatment",
    instrument_cols     = ("instrument",),
    covariate_cols      = (),
    robust:        str  = "HC3",
    verbose:       bool = True,
) -> dict:
    """
    Two-Stage Least Squares with identification-robust diagnostics.

    Parameters
    ----------
    df              : DataFrame. None -> synthetic data.
    outcome_col     : Dependent variable.
    treatment_col   : Endogenous regressor (instrumented).
    instrument_cols : Sequence of excluded instrument column names.
    covariate_cols  : Exogenous controls (included in both stages).
    robust          : Covariance type for linearmodels ('HC3', 'kernel', ...).
    verbose         : Print summary.
    """
    warn_list: list[str] = []

    if df is None or CAUSAL_USE_SYNTHETIC:
        df = _make_synthetic_iv()
        outcome_col, treatment_col = "outcome", "treatment"
        instrument_cols = ("instrument",)
        if verbose:
            print("[IV] Using synthetic data (true LATE = 3.0, n=600)")

    n_obs = len(df)
    instrument_cols = list(instrument_cols)
    covariate_cols  = list(covariate_cols)

    # ---- linearmodels IV2SLS -------------------------------------------------
    try:
        from linearmodels.iv import IV2SLS   # type: ignore

        endog  = df[[outcome_col]]
        exog   = df[covariate_cols] if covariate_cols else None
        instr  = df[instrument_cols]
        treat  = df[[treatment_col]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = IV2SLS(endog, exog, treat, instr).fit(
                cov_type=robust, debiased=True)

        est  = float(res.params[treatment_col])
        se   = float(res.std_errors[treatment_col])
        cil  = float(res.conf_int().loc[treatment_col, "lower"])
        ciu  = float(res.conf_int().loc[treatment_col, "upper"])
        p    = float(res.pvalues[treatment_col])
        ci_type = "wald_hc3"

        # KP rk-F statistic
        try:
            kp_f = float(res.first_stage.diagnostics["KP F-stat"])
        except Exception:
            kp_f = np.nan
        fs_f = np.nan
        try:
            fs_f = float(res.first_stage.individual[treatment_col]["f.stat"])
        except Exception:
            pass

        weak = (not np.isnan(kp_f)) and (kp_f < 10)

        # Anderson-Rubin CI when weak instrument
        if weak:
            warn_list.append(
                f"Weak instrument: KP rk-F = {kp_f:.2f} < 10. "
                "Reporting Anderson-Rubin identification-robust CI instead."
            )
            try:
                from ivmodels import KClass   # type: ignore
                y_arr  = df[outcome_col].values
                d_arr  = df[treatment_col].values
                z_arr  = df[instrument_cols].values
                x_arr  = df[covariate_cols].values if covariate_cols else np.ones((n_obs, 1))
                ar_res = KClass(k=0).fit(y_arr, d_arr, z_arr, x_arr)
                cil, ciu = float(ar_res.conf_int()[0][0]), float(ar_res.conf_int()[0][1])
                p        = float(ar_res.pvalue()[0])
                ci_type  = "anderson_rubin"
            except ImportError:
                warn_list.append("`ivmodels` not installed; AR CI unavailable (pip install ivmodels)")

        diag = {
            "kp_rk_f":       round(kp_f, 3) if not np.isnan(kp_f) else None,
            "first_stage_f": round(fs_f, 3) if not np.isnan(fs_f) else None,
            "weak_instrument": weak,
        }

    # ---- Fallback: numpy Wald IV (single instrument only) --------------------
    except ImportError:
        warn_list.append(
            "`linearmodels` not installed; using numpy Wald-IV fallback "
            "(no robust SEs). Install: pip install linearmodels"
        )
        est, se, cil, ciu, p, ci_type, diag = _wald_fallback(
            df, outcome_col, treatment_col, instrument_cols[0])

    # ---- CSV -----------------------------------------------------------------
    if verbose:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pd.DataFrame([{
            "estimate": est, "se": se, "ci_lower": cil, "ci_upper": ciu, "p": p,
            **diag,
        }]).to_csv(os.path.join(OUTPUT_DIR, "iv_estimates.csv"), index=False)
        print(f"\n[IV] LATE = {est:.4f}  SE = {se:.4f}  "
              f"95% {ci_type} CI [{cil:.4f}, {ciu:.4f}]  p = {p:.4f}")
        print(f"  KP rk-F = {diag.get('kp_rk_f')}  |  first-stage F = {diag.get('first_stage_f')}")
        for w in warn_list:
            print(f"  [WARNING] {w}")

    return {
        "method":      "IV 2SLS (linearmodels)",
        "estimand":    "LATE",
        "estimate":    round(est, 6),
        "se":          round(se, 6),
        "ci_lower":    round(cil, 6),
        "ci_upper":    round(ciu, 6),
        "ci_type":     ci_type,
        "p_value":     round(p, 6),
        "n_obs":       n_obs,
        "diagnostics": diag,
        "warnings":    warn_list,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wald_fallback(df, outcome_col, treatment_col, instrument_col):
    """Numpy Wald-IV (single binary instrument, no robust SEs)."""
    y = df[outcome_col].values
    d = df[treatment_col].values
    z = df[instrument_col].values
    n = len(y)

    cov_dz = np.cov(d, z)[0, 1]
    var_z  = np.var(z)
    first  = cov_dz / var_z
    d_hat  = first * z

    dm = d_hat - d_hat.mean()
    ym = y - y.mean()
    late = np.sum(dm * ym) / np.sum(dm ** 2)

    resid = y - late * d
    s2    = np.var(resid)
    var_b = s2 / (np.sum(dm ** 2))
    se    = float(np.sqrt(var_b))

    from scipy.stats import t as t_dist
    p = float(2 * t_dist.sf(abs(late / se), df=n - 2))
    cil = late - 1.96 * se
    ciu = late + 1.96 * se
    fs_f = float((first ** 2 * var_z * n) / np.var(d - first * z))
    diag = {"kp_rk_f": None, "first_stage_f": round(fs_f, 3),
            "weak_instrument": fs_f < 10}
    return float(late), se, float(cil), float(ciu), p, "wald_numpy", diag
