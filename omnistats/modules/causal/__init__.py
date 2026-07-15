"""
omnistats/modules/causal/__init__.py
--------------------------------------
Robust causal inference subpackage -- Stage 3 of the OmniStats pipeline.

Exports:
  staggered_did()    -- Callaway & Sant-Anna (2021) staggered DiD, ATT(g,t)
  iv_2sls()          -- linearmodels IV2SLS with Anderson-Rubin fallback
  rdd_robust()       -- rdrobust CCT optimal-bandwidth RDD + rddensity test
  run_causal_suite() -- runs all three, saves causal_results.csv
"""

from .did import staggered_did
from .iv  import iv_2sls
from .rdd import rdd_robust


def run_causal_suite(verbose: bool = True) -> dict:
    """
    Run all three robust causal estimators and save a combined summary CSV.

    Returns
    -------
    dict with keys "did", "iv", "rdd"; each value is the standardised
    result dict from the respective estimator.
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
    }

    # Standardised CSV with shared schema only
    COLS = ["method", "estimand", "estimate", "se", "ci_lower", "ci_upper",
            "ci_type", "p_value", "n_obs"]
    rows = [{c: res.get(c) for c in COLS} for res in results.values()]
    pd.DataFrame(rows).reindex(columns=COLS).to_csv(
        os.path.join(OUTPUT_DIR, "causal_results.csv"), index=False
    )
    if verbose:
        print(f"\n[Causal] Summary saved -> {OUTPUT_DIR}/causal_results.csv")

    return results


__all__ = ["staggered_did", "iv_2sls", "rdd_robust", "run_causal_suite"]
