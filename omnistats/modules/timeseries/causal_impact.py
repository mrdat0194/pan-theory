"""
omnistats/modules/timeseries/causal_impact.py
----------------------------------------------
CausalImpact — Bayesian Structural Time Series (BSTS) counterfactual.
Google CausalImpact algorithm ported to PyTorch & Pyro.

Runs a Bayesian Structural Time Series model on CPU:
  Level component: level_t = level_{t-1} + noise_t
  Regression component: y_t = level_t + beta * X_t + obs_noise_t
Inference is performed via Stochastic Variational Inference (SVI).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config import (OUTPUT_DIR, CAUSAL_USE_SYNTHETIC,
                    TS_CAUSALIMPACT_ENABLED, TS_DATE_COL,
                    TS_METRIC_COL, TS_INTERVENTION_DATE,
                    TS_CONTROL_COLS)




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


# ── Pyro BSTS Model ────────────────────────────────────────────────────────────

def _bsts_model(y, X):
    T = y.shape[0]
    J = X.shape[1] if X is not None else 0
    
    # Observation and level standard deviation priors
    sigma_obs = pyro.sample("sigma_obs", dist.HalfNormal(1.0))
    sigma_level = pyro.sample("sigma_level", dist.HalfNormal(0.5))
    
    # Regression coefficients prior
    if J > 0:
        beta = pyro.sample("beta", dist.Normal(0.0, 1.0).expand([J]).to_event(1))
    else:
        beta = None
        
    # Latent local level trend modeled vectorially
    level_0 = pyro.sample("level_0", dist.Normal(y[0], 1.0))
    level_steps = pyro.sample("level_steps", dist.Normal(0.0, sigma_level).expand([T - 1]).to_event(1))
    
    level = torch.cat([level_0.unsqueeze(0), level_0 + torch.cumsum(level_steps, dim=0)])
    
    pred_mean = level
    if J > 0 and X is not None:
        pred_mean = pred_mean + torch.matmul(X, beta)
        
    with pyro.plate("data", T):
        pyro.sample("obs", dist.Normal(pred_mean, sigma_obs), obs=y)


# ── Custom Pyro CausalImpact Engine ────────────────────────────────────────────

def _run_pyro_causalimpact(df: pd.DataFrame, T_treat: int, verbose: bool) -> dict:
    """
    Run CausalImpact BSTS using Pyro SVI on CPU.
    """
    T = len(df)
    y = df["y"].values
    
    # Extract controls if present
    control_cols = [c for c in df.columns if c not in ["ds", "y"]]
    X = df[control_cols].values if len(control_cols) > 0 else None
    
    y_pre = y[:T_treat]
    X_pre = X[:T_treat, :] if X is not None else None
    X_post = X[T_treat:, :] if X is not None else None
    
    # Convert pre-treatment data to PyTorch tensors
    y_pre_tensor = torch.tensor(y_pre, dtype=torch.float32)
    X_pre_tensor = torch.tensor(X_pre, dtype=torch.float32) if X_pre is not None else None
    
    # ── SVI optimization on pre-treatment period ──────────────────────────
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(_bsts_model)
    svi = SVI(_bsts_model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())
    
    num_steps = 1500
    if verbose:
        print(f"  Fitting BSTS model via Pyro SVI (steps={num_steps})...")
        
    for step in range(num_steps):
        loss = svi.step(y_pre_tensor, X_pre_tensor)
        
    # ── Draw posterior samples ─────────────────────────────────────────────
    num_samples = 1000
    predictive = Predictive(guide, num_samples=num_samples)
    posterior_samples = predictive(y_pre_tensor, X_pre_tensor)
    
    # Reconstruct parameters with explicit reshaping to avoid squeezing J=1 or T_pre=2 dimensions
    sigma_level_s = posterior_samples["sigma_level"].reshape(num_samples)
    sigma_obs_s = posterior_samples["sigma_obs"].reshape(num_samples)
    level_0_s = posterior_samples["level_0"].reshape(num_samples)
    level_steps_s = posterior_samples["level_steps"].reshape(num_samples, T_treat - 1)
    
    if X is not None and "beta" in posterior_samples:
        beta_s = posterior_samples["beta"].reshape(num_samples, len(control_cols))
    else:
        beta_s = None
        
    # ── Pre-treatment counterfactual fit ──────────────────────────────────
    pre_levels = []
    for s in range(num_samples):
        steps = level_steps_s[s]
        l_0 = level_0_s[s]
        l_s = torch.cat([l_0.unsqueeze(0), l_0 + torch.cumsum(steps, dim=0)])
        pre_levels.append(l_s)
    pre_levels = torch.stack(pre_levels) # (num_samples, T_pre)
    
    pre_mean = pre_levels
    if X is not None and beta_s is not None:
        X_pre_tensor = torch.tensor(X_pre, dtype=torch.float32)
        pre_mean = pre_mean + torch.einsum("sj,tj->st", beta_s, X_pre_tensor)
    pre_mean_np = pre_mean.numpy()
    
    # ── Post-treatment counterfactual forecast ────────────────────────────
    level_last_s = level_0_s + level_steps_s.sum(dim=1)
    
    future_levels = []
    current_levels = level_last_s.clone()
    
    # Use deterministic seed for reproducibility
    torch.manual_seed(42)
    T_post = T - T_treat
    
    for t in range(T_post):
        step_noise = torch.randn(num_samples) * sigma_level_s
        current_levels = current_levels + step_noise
        future_levels.append(current_levels.clone())
    future_levels = torch.stack(future_levels, dim=1) # (num_samples, T_post)
    
    post_mean = future_levels
    if X is not None and beta_s is not None:
        X_post_tensor = torch.tensor(X_post, dtype=torch.float32)
        post_mean = post_mean + torch.einsum("sj,tj->st", beta_s, X_post_tensor)
    post_mean_np = post_mean.numpy()
    
    # ── Combine pre and post counterfactuals ──────────────────────────────
    cf_samples = np.concatenate([pre_mean_np, post_mean_np], axis=1) # (num_samples, T)
    cf_mean = cf_samples.mean(axis=0)
    cf_lower = np.percentile(cf_samples, 2.5, axis=0)
    cf_upper = np.percentile(cf_samples, 97.5, axis=0)
    
    # Pointwise lift statistics
    pointwise_effect = y - cf_mean
    pointwise_lower = y - cf_upper
    pointwise_upper = y - cf_lower
    
    # Cumulative post-treatment lift
    post_obs = y[T_treat:]
    post_cf = cf_samples[:, T_treat:]
    cum_samples = np.cumsum(post_obs[None, :] - post_cf, axis=1) # (num_samples, T_post)
    cum_mean = cum_samples.mean(axis=0)
    cum_lower = np.percentile(cum_samples, 2.5, axis=0)
    cum_upper = np.percentile(cum_samples, 97.5, axis=0)
    
    # Summaries
    avg_effect = float(pointwise_effect[T_treat:].mean())
    ci_lo = float(pointwise_lower[T_treat:].mean())
    ci_hi = float(pointwise_upper[T_treat:].mean())
    
    # Bayesian p-value equivalent: P(Effect <= 0 | data)
    post_effects = (post_obs[None, :] - post_cf).mean(axis=1) # (num_samples,)
    p_val = float(np.mean(post_effects <= 0))
    
    # ── Generate Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    periods = np.arange(T)
    
    # Top Panel: Observed vs Counterfactual
    ax = axes[0]
    ax.plot(periods, y, 'k-', label='Observed', linewidth=1.5)
    ax.plot(periods, cf_mean, 'r--', label='Counterfactual (BSTS)', linewidth=1.5)
    ax.fill_between(periods, cf_lower, cf_upper, color='red', alpha=0.15, label='95% Credible Interval')
    ax.axvline(T_treat - 0.5, color='gray', linestyle=':', linewidth=1.5, label='Intervention')
    ax.set_title("CausalImpact: Observed vs. Counterfactual", fontweight="bold")
    ax.legend(loc='upper left')
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    
    # Middle Panel: Pointwise Effect
    ax = axes[1]
    ax.plot(periods, pointwise_effect, 'r-', linewidth=1.2, label='Pointwise Effect')
    ax.fill_between(periods, pointwise_lower, pointwise_upper, color='red', alpha=0.15)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(T_treat - 0.5, color='gray', linestyle=':', linewidth=1.5)
    ax.set_title("Pointwise Treatment Effect", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    
    # Bottom Panel: Cumulative Effect
    ax = axes[2]
    cum_periods = periods[T_treat:]
    ax.plot(cum_periods, cum_mean, 'r-', linewidth=1.5, label='Cumulative Effect')
    ax.fill_between(cum_periods, cum_lower, cum_upper, color='red', alpha=0.15)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(T_treat - 0.5, color='gray', linestyle=':', linewidth=1.5)
    ax.set_title("Cumulative Treatment Effect (ATT)", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "ts_causalimpact.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Save lift CSV
    pd.DataFrame({
        "period":          cum_periods,
        "observed":        post_obs,
        "counterfactual":  cf_mean[T_treat:],
        "lift":            pointwise_effect[T_treat:],
        "cumulative_lift": cum_mean,
    }).to_csv(os.path.join(OUTPUT_DIR, "ts_lift.csv"), index=False)
    
    return {
        "method":    "CausalImpact (Pyro BSTS SVI)",
        "estimand":  "Average ATT post-intervention",
        "estimate":  round(avg_effect, 4) if not np.isnan(avg_effect) else float("nan"),
        "se":        float("nan"),
        "ci_lower":  round(ci_lo, 4) if not np.isnan(ci_lo) else float("nan"),
        "ci_upper":  round(ci_hi, 4) if not np.isnan(ci_hi) else float("nan"),
        "ci_type":   "pyro_svi_credible",
        "p_value":   round(p_val, 4) if not np.isnan(p_val) else float("nan"),
        "n_obs":     len(df),
        "diagnostics": {"backend": "pyro_svi_cpu"},
        "warnings":  [],
    }


# ── Main function ──────────────────────────────────────────────────────────────
def run_causalimpact(verbose: bool = True) -> dict:
    """
    Bayesian Structural Time Series counterfactual using Pyro BSTS model on CPU.

    Returns standardised result dict.
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

    try:
        result = _run_pyro_causalimpact(df, T_treat, verbose)
        result["warnings"] = warns + result.get("warnings", [])
    except Exception as e:
        warns.append(f"[ERROR] Pyro BSTS failed: {str(e)}")
        result = {
            "method":    "CausalImpact (Pyro BSTS)",
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
