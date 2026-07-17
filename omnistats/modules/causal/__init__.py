"""
omnistats/modules/causal/__init__.py
--------------------------------------
Robust causal inference subpackage — Stage 4 of the OmniStats pipeline.

All seven estimators share a standardised result schema and are written
to causal_results.csv, which feeds Table 8 of the APA report.

Exports:
  staggered_did()      — Callaway & Sant-Anna (2021) staggered DiD, ATT(g,t)
  iv_2sls()            — linearmodels IV2SLS with Anderson-Rubin fallback
  rdd_robust()         — rdrobust CCT optimal-bandwidth RDD + rddensity test
  synthetic_control()  — Abadie et al. SCM with placebo-in-space inference
  matrix_completion()  — Athey et al. nuclear norm imputation for panel data
  run_causalimpact()   — Google CausalImpact BSTS (time-series causal)
  run_causal_suite()   — runs all six estimators, saves causal_results.csv
"""

from .did               import staggered_did
from .iv                import iv_2sls
from .rdd               import rdd_robust
from .scm               import synthetic_control
from .matrix_completion import matrix_completion

# CausalImpact lives in modules/timeseries/ but is exposed here
# as the time-series causal estimator within Stage 4's unified suite.
try:
    from ..timeseries.causal_impact import run_causalimpact
    _HAS_CI = True
except ImportError:
    _HAS_CI = False
    def run_causalimpact(verbose=True):
        return {
            "method":   "CausalImpact (BSTS)",
            "estimand": "Average ATT post-intervention",
            "estimate": float("nan"), "se": float("nan"),
            "ci_lower": float("nan"), "ci_upper": float("nan"),
            "ci_type":  "not_available", "p_value": float("nan"),
            "n_obs":    0, "warnings": ["timeseries module not available"],
        }


def run_causal_suite(verbose: bool = True) -> dict:
    """
    Run all Stage 4 causal estimators and save causal_results.csv.

    Estimators:
      1. Staggered DiD (Callaway & Sant-Anna)
      2. IV/2SLS (linearmodels)
      3. RDD (rdrobust CCT)
      4. Synthetic Control Method (cvxpy)
      5. Matrix Completion (SoftImpute)
      6. CausalImpact BSTS (time-series causal — same Stage 4)

    All results written to causal_results.csv for APA Table 8.

    Returns
    -------
    dict with keys "did", "iv", "rdd", "scm", "mc", "causal_impact";
    each value is the standardised result dict.
    """
    import os
    import pandas as pd
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from config import OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "did": staggered_did(verbose=verbose),
        "iv":  iv_2sls(verbose=verbose),
        "rdd": rdd_robust(verbose=verbose),
        "scm": synthetic_control(verbose=verbose),
        "mc":  matrix_completion(verbose=verbose),
        "causal_impact": run_causalimpact(verbose=verbose),
    }

    # All six share the same standardised schema → single causal_results.csv
    COLS = ["method", "estimand", "estimate", "se", "ci_lower", "ci_upper",
            "ci_type", "p_value", "n_obs"]
    rows = [{c: res.get(c) for c in COLS} for res in results.values()]
    pd.DataFrame(rows).reindex(columns=COLS).to_csv(
        os.path.join(OUTPUT_DIR, "causal_results.csv"), index=False
    )
    if verbose:
        print(f"\n[Causal] All Stage 4 results saved -> {OUTPUT_DIR}/causal_results.csv")
        print(f"         (DiD, IV, RDD, SCM, Matrix Completion, CausalImpact)")

    return results


__all__ = [
    "staggered_did",
    "iv_2sls",
    "rdd_robust",
    "synthetic_control",
    "matrix_completion",
    "run_causalimpact",
    "run_causal_suite",
]
