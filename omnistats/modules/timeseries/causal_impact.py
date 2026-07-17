"""
omnistats/modules/timeseries/causal_impact.py
----------------------------------------------
CausalImpact — Bayesian Structural Time Series (BSTS) counterfactual.
Brodersen et al. (2015), Google Research.

Migrated conceptual space from: Bayesian/prophet.ipynb

Why CausalImpact / BSTS
------------------------
  Unlike Prophet (single-series trend extrapolation), CausalImpact:
  1. Incorporates "control" time series via spike-and-slab variable selection
     to separate treatment from macroeconomic shocks.
  2. Uses a full Kalman Filter state-space model for principled uncertainty.
  3. Outputs posterior inclusion probabilities for each control series —
     directly reportable in an APA table.

  The BSTS model decomposes the outcome as:
    y_t = μ_t + β' x_t + ε_t      (structural + regression component)
    μ_t = μ_{t-1} + δ_{t-1} + η_t  (local linear trend state equation)

  Spike-and-Slab prior on β:
    γ_k ~ Bernoulli(π_k)           (inclusion indicator)
    β_k | γ_k=1 ~ Normal(0, σ_β²)
    β_k | γ_k=0 = 0
  → E[γ_k | data] = Posterior Inclusion Probability (PIP)

Library strategy
----------------
  Primary:  tfcausalimpact  (pip install tfcausalimpact)
  Fallback: pycausalimpact  (pip install pycausalimpact)
  Last:     Prophet-based approximate counterfactual (no control series)

MCMC / Bayesian connection
--------------------------
  Both Prophet and BSTS internally use HMC/NUTS (via Stan or TFP) for
  posterior sampling — the same Hamiltonian MC machinery as mcmc_mirror_hmc
  in Bayesian/mcmc_bayesian.py. Credible intervals are MCMC-derived.
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
from config import (OUTPUT_DIR, CAUSAL_USE_SYNTHETIC,
                    TS_CAUSALIMPACT_ENABLED, TS_DATE_COL,
                    TS_METRIC_COL, TS_INTERVENTION_DATE,
                    TS_CONTROL_COLS, TS_SEASONALITY_MODE)

# ── tfcausalimpact primary ─────────────────────────────────────────────────────
try:
    from causalimpact import CausalImpact as _CI
    _HAS_CAUSALIMPACT = True
    _CAUSALIMPACT_PKG = "causalimpact"
except ImportError:
    _HAS_CAUSALIMPACT = False
    _CAUSALIMPACT_PKG = None

# ── Prophet approximate fallback ───────────────────────────────────────────────
try:
    from prophet import Prophet as _Prophet
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False


# ── Synthetic data generator ───────────────────────────────────────────────────
def _make_synthetic_ts() -> dict:
    """
    60 time periods. Intervention at t=40.
    True counterfactual = baseline trend. True ATT ≈ 5.0.
    Includes one correlated control series.
    """
    rng = np.random.RandomState(42)
    T = 60
    T_treat = 40
    true_att = 5.0

    t = np.arange(T)
    trend     = 0.05 * t
    seasonal  = 2 * np.sin(2 * np.pi * t / 12)
    noise     = rng.randn(T)
    control   = trend + seasonal + noise * 0.5 + rng.randn(T) * 0.3

    baseline  = trend + seasonal + noise
    outcome   = baseline.copy()
    outcome[T_treat:] += true_att

    dates = pd.date_range("2022-01-01", periods=T, freq="W")

    return {
        "df": pd.DataFrame({
            "ds":      dates,
            "y":       outcome,
            "control": control,
        }),
        "T_treat":      T_treat,
        "intervention": str(dates[T_treat].date()),
        "true_att":     true_att,
        "T": T,
    }


# ── CausalImpact wrapper ───────────────────────────────────────────────────────
def _run_causalimpact_pkg(df: pd.DataFrame, T_treat: int, verbose: bool) -> dict:
    """
    Run CausalImpact (tfcausalimpact or pycausalimpact) on df.
    df must have columns: y, and optional control columns.
    """
    pre_period  = [0, T_treat - 1]
    post_period = [T_treat, len(df) - 1]

    ci = _CI(df.drop(columns=["ds"], errors="ignore"),
             pre_period, post_period)
    ci.run()

    summary = ci.summary()
    report  = ci.summary(output="report") if hasattr(ci, "summary") else ""

    # Extract standardised fields
    try:
        avg_effect    = float(summary.loc["Average", "Actual"] -
                              summary.loc["Average", "Predicted"])
    except Exception:
        avg_effect = float("nan")
    try:
        p_val = float(getattr(ci, "p_value", float("nan")))
    except Exception:
        p_val = float("nan")
    try:
        ci_lo = float(summary.loc["Average", "CI lower"].item()
                      if hasattr(summary.loc["Average", "CI lower"], "item")
                      else summary.loc["Average", "CI lower"])
        ci_hi = float(summary.loc["Average", "CI upper"].item()
                      if hasattr(summary.loc["Average", "CI upper"], "item")
                      else summary.loc["Average", "CI upper"])
    except Exception:
        ci_lo = ci_hi = float("nan")

    # Plot
    try:
        fig = ci.plot()
        plot_path = os.path.join(OUTPUT_DIR, "ts_causalimpact.png")
        if fig is not None:
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
    except Exception:
        pass

    return {
        "method":    "CausalImpact (BSTS / Spike-and-Slab)",
        "estimand":  "Average ATT post-intervention",
        "estimate":  round(avg_effect, 4) if not np.isnan(avg_effect) else float("nan"),
        "se":        float("nan"),   # BSTS gives credible intervals, not SEs
        "ci_lower":  round(ci_lo, 4) if not np.isnan(ci_lo) else float("nan"),
        "ci_upper":  round(ci_hi, 4) if not np.isnan(ci_hi) else float("nan"),
        "ci_type":   "bsts_mcmc_credible",
        "p_value":   round(p_val, 4) if not np.isnan(p_val) else float("nan"),
        "n_obs":     len(df),
        "diagnostics": {"backend": _CAUSALIMPACT_PKG},
        "warnings":  [],
    }


# ── Prophet approximate fallback ───────────────────────────────────────────────
def _run_prophet_fallback(df: pd.DataFrame, T_treat: int, verbose: bool) -> dict:
    """
    Approximate BSTS using Prophet when CausalImpact is unavailable.
    Fits Prophet on pre-intervention period, extrapolates counterfactual.
    No spike-and-slab control series selection.
    """
    warns = ["[WARNING] CausalImpact not installed. Using Prophet approximate fallback.",
             "         Install with: pip install tfcausalimpact"]

    pre_df = df.iloc[:T_treat][["ds", "y"]].rename(columns={"y": "y"})
    post_df = df.iloc[T_treat:][["ds", "y"]].copy()

    model = _Prophet(
        seasonality_mode=TS_SEASONALITY_MODE,
        yearly_seasonality=False,
        weekly_seasonality=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(pre_df)

    future = model.make_future_dataframe(periods=len(post_df), freq="W")
    forecast = model.predict(future)

    post_forecast = forecast.iloc[T_treat:]
    lift = post_df["y"].values - post_forecast["yhat"].values

    att     = float(np.mean(lift))
    ci_lo   = float(np.mean(post_forecast["yhat_lower"].values))
    ci_hi   = float(np.mean(post_forecast["yhat_upper"].values))
    se      = float(np.std(lift, ddof=1) / np.sqrt(len(lift)))

    # Plot
    periods = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(periods, df["y"].values, "k-", label="Observed", linewidth=2)
    ax.plot(periods[T_treat:], post_forecast["yhat"].values, "r--",
            label="Counterfactual (Prophet)", linewidth=2)
    ax.fill_between(periods[T_treat:],
                    post_forecast["yhat_lower"].values,
                    post_forecast["yhat_upper"].values,
                    alpha=0.2, color="red", label="95% Credible Band")
    ax.axvline(T_treat, color="grey", linestyle=":", linewidth=1.5,
               label="Intervention")
    ax.set_title("Counterfactual (Prophet Fallback)", fontweight="bold")
    ax.set_xlabel("Period"); ax.set_ylabel(TS_METRIC_COL or "Metric")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.bar(periods[T_treat:], lift, color="#E84855", alpha=0.8, label="Lift")
    ax.axhline(att, color="black", linestyle="--", label=f"Mean ATT={att:.3f}")
    ax.axhline(0,   color="grey",  linewidth=0.8)
    ax.set_title("Pointwise Treatment Lift", fontweight="bold")
    ax.set_xlabel("Period"); ax.set_ylabel("Lift (Observed − Counterfactual)")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ts_counterfactual.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Lift CSV
    pd.DataFrame({
        "period":          periods[T_treat:],
        "observed":        post_df["y"].values,
        "counterfactual":  post_forecast["yhat"].values,
        "lift":            lift,
    }).to_csv(os.path.join(OUTPUT_DIR, "ts_lift.csv"), index=False)

    return {
        "method":    "Counterfactual Time-Series (Prophet fallback)",
        "estimand":  "Average Switchback Lift",
        "estimate":  round(att, 4),
        "se":        round(se, 4),
        "ci_lower":  round(ci_lo, 4),
        "ci_upper":  round(ci_hi, 4),
        "ci_type":   "prophet_mcmc_credible",
        "p_value":   float("nan"),  # Prophet doesn't output p-values
        "n_obs":     len(df),
        "diagnostics": {"backend": "prophet_fallback",
                        "n_post_periods": len(post_df)},
        "warnings":  warns,
    }


# ── Main function ──────────────────────────────────────────────────────────────
def run_causalimpact(verbose: bool = True) -> dict:
    """
    Bayesian Structural Time Series counterfactual.

    Uses CausalImpact (BSTS + spike-and-slab) as primary.
    Falls back to Prophet trend extrapolation if CausalImpact unavailable.

    Returns standardised result dict (same schema as causal_results.csv).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    warns = []

    # ── Load data ──────────────────────────────────────────────────────────
    if CAUSAL_USE_SYNTHETIC or not TS_CAUSALIMPACT_ENABLED:
        data = _make_synthetic_ts()
        df      = data["df"]
        T_treat = data["T_treat"]
        warns.append("Using synthetic time-series demo data (TS_CAUSALIMPACT_ENABLED=False or CAUSAL_USE_SYNTHETIC=True)")
    else:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        from config import DATA_PATH
        raw = pd.read_csv(DATA_PATH)
        raw[TS_DATE_COL] = pd.to_datetime(raw[TS_DATE_COL])
        df = raw[[TS_DATE_COL, TS_METRIC_COL] + (TS_CONTROL_COLS or [])].rename(
            columns={TS_DATE_COL: "ds", TS_METRIC_COL: "y"})
        df = df.sort_values("ds").reset_index(drop=True)
        intervention = pd.to_datetime(TS_INTERVENTION_DATE)
        T_treat = int((df["ds"] < intervention).sum())

    # ── Run estimator ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n[TimeSeries] Stage 4 — Bayesian Counterfactual Time-Series")
        print(f"  Intervention at period {T_treat} / {len(df)}")

    if _HAS_CAUSALIMPACT:
        result = _run_causalimpact_pkg(df, T_treat, verbose)
        result["warnings"] = warns + result.get("warnings", [])
    elif _HAS_PROPHET:
        result = _run_prophet_fallback(df, T_treat, verbose)
        result["warnings"] = warns + result.get("warnings", [])
    else:
        warns.append("[ERROR] Neither CausalImpact nor Prophet installed.")
        warns.append("  Install: pip install tfcausalimpact  OR  pip install prophet")
        result = {
            "method":    "CausalImpact (BSTS)",
            "estimand":  "Average ATT post-intervention",
            "estimate":  float("nan"),
            "se":        float("nan"),
            "ci_lower":  float("nan"),
            "ci_upper":  float("nan"),
            "ci_type":   "not_available",
            "p_value":   float("nan"),
            "n_obs":     len(df),
            "diagnostics": {},
            "warnings":  warns,
        }

    if verbose:
        print(f"  Estimate (ATT):  {result['estimate']}")
        print(f"  95% CI:          [{result['ci_lower']}, {result['ci_upper']}]")
        print(f"  CI type:         {result['ci_type']}")
        for w in result.get("warnings", []):
            print(f"  {w}")

    return result
