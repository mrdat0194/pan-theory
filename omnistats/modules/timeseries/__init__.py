"""
omnistats/modules/timeseries/__init__.py
-----------------------------------------
Bayesian Counterfactual Time-Series subpackage — Stage 4 of OmniStats.

Migrated from:  Bayesian/prophet.ipynb  (conceptual space)
Upgraded to:    Google CausalImpact (Bayesian Structural Time Series)

Exports:
  run_causalimpact()        — BSTS counterfactual with control selection
  run_timeseries_suite()    — orchestrator, saves timeseries_causal_results.csv

Why CausalImpact over Prophet
------------------------------
  Prophet extrapolates a single time-series trend. CausalImpact uses
  Bayesian Structural Time Series (BSTS) to also incorporate other "control"
  time series (untreated regions/metrics) via spike-and-slab variable selection.
  This means CausalImpact can separate treatment effects from macroeconomic
  shocks that affect all units simultaneously.

  Spike-and-Slab APA Explainability: outputs posterior inclusion probabilities
  for each control series — "Control B was selected with 91% probability."

Relation to Stage 3 DiD
------------------------
  CausalImpact is DiD generalised to continuous time. Instead of a single
  binary pre/post comparison with parallel trends, BSTS models the full
  counterfactual trajectory with credible uncertainty bands.

Pyro SVI connection
-------------------
  BSTS uses Stochastic Variational Inference (SVI) to approximate the posterior
  distribution of the latent components and control coefficients.
  The credible intervals in the output are SVI-derived.
"""

from .causal_impact import run_causalimpact


def run_timeseries_suite(verbose: bool = True) -> dict:
    """
    Run the CausalImpact time-series suite.
    Saves timeseries_causal_results.csv and plots to OUTPUT_DIR.

    Returns
    -------
    dict with key "causal_impact": standardised result dict
    """
    import os
    import pandas as pd
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from config import OUTPUT_DIR, TS_CAUSALIMPACT_ENABLED

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not TS_CAUSALIMPACT_ENABLED:
        if verbose:
            print("\n[TimeSeries] TS_CAUSALIMPACT_ENABLED=False. "
                  "Set True in config.py and supply TS_DATE_COL, "
                  "TS_METRIC_COL, TS_INTERVENTION_DATE to use real data.")

    result_ci = run_causalimpact(verbose=verbose)
    results = {"causal_impact": result_ci}

    COLS = ["method", "estimand", "estimate", "se", "ci_lower", "ci_upper",
            "ci_type", "p_value", "n_obs"]
    rows = [{c: results["causal_impact"].get(c) for c in COLS}]
    pd.DataFrame(rows).reindex(columns=COLS).to_csv(
        os.path.join(OUTPUT_DIR, "timeseries_causal_results.csv"), index=False
    )
    if verbose:
        print(f"[TimeSeries] Results saved -> {OUTPUT_DIR}/timeseries_causal_results.csv")

    return results


__all__ = ["run_causalimpact", "run_timeseries_suite"]
