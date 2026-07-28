"""
omnistats/experiment_design.py
───────────────────────────────
Phase I — Pre-Experiment Design (SOTA CAR)

Run this script **before** launching your A/B test.  It:

  1. Estimates the required sample size per arm (TTestIndPower).
  2. Loads baseline subject data, applies SOTA Covariate-Adaptive
     Randomization (CAR) using Mahalanobis-distance minimization to
     produce a perfectly balanced treatment schedule.
  3. Validates balance via a 1 000-replication bootstrap simulation and
     reports standardised mean differences (SMD) per covariate.

Outputs
-------
  outputs/randomization_schedule.csv   — subject → arm assignment
  outputs/car_balance_report.csv       — per-covariate SMD summary
  outputs/power_analysis.csv           — power curve (n vs. power) table

Usage
-----
    python -X utf8 experiment_design.py
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

# ─── path bootstrap ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import (
    DATA_PATH, OUTPUT_DIR,
    INDICATOR_COLS, DEMOGRAPHIC_COLS,
    AB_METRIC_COL,
    DESIGN_MDE_RELATIVE,
    DESIGN_POWER,
    DESIGN_ALPHA,
    DESIGN_STRATIFY_COLS,
    DESIGN_N_SIMULATIONS,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    w = 68
    print("\n" + "=" * w)
    print(f"  {msg}")
    print("=" * w)


def _substage(tag: str, desc: str) -> None:
    print(f"\n[{tag}] {desc}")


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled SD) — standardised mean difference."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_sd = np.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
        / (n_a + n_b - 2)
    )
    if pooled_sd < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


# =============================================================================
# STEP 1 — Power Analysis
# =============================================================================

def run_power_analysis(
    baseline_mean: float | None = None,
    baseline_std: float | None = None,
    mde_relative: float = DESIGN_MDE_RELATIVE,
    power: float = DESIGN_POWER,
    alpha: float = DESIGN_ALPHA,
    verbose: bool = True,
) -> dict:
    """
    Estimate required sample size per arm using TTestIndPower (statsmodels).

    Parameters
    ----------
    baseline_mean, baseline_std:
        Computed from DATA_PATH if not supplied.
    mde_relative:
        Minimum Detectable Effect as a fraction of the baseline mean.
    power, alpha:
        Statistical power and significance level (two-sided).

    Returns
    -------
    dict with keys:
        n_per_arm, effect_size_cohen_d, alpha, power, mde_absolute,
        baseline_mean, baseline_std, power_curve (DataFrame)
    """
    from statsmodels.stats.power import TTestIndPower

    if verbose:
        _substage("1.1", "Loading baseline data for power analysis")

    # ── load baseline ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(DATA_PATH)
        if AB_METRIC_COL in df.columns:
            col = df[AB_METRIC_COL].dropna()
        else:
            col = df.select_dtypes(include="number").iloc[:, 0].dropna()
            warnings.warn(
                f"[PowerAnalysis] AB_METRIC_COL='{AB_METRIC_COL}' not found; "
                f"using '{col.name}' instead.", stacklevel=2
            )
    except FileNotFoundError:
        warnings.warn(
            f"[PowerAnalysis] DATA_PATH '{DATA_PATH}' not found; "
            "using synthetic mu=100, sigma=20.", stacklevel=2
        )
        col = pd.Series(np.random.default_rng(0).normal(100, 20, 500))

    if baseline_mean is None:
        baseline_mean = float(col.mean())
    if baseline_std is None:
        baseline_std = float(col.std(ddof=1))

    # ── effect size ───────────────────────────────────────────────────────────
    mde_absolute = baseline_mean * mde_relative
    if baseline_std < 1e-12:
        warnings.warn("[PowerAnalysis] baseline_std ~0; defaulting to 1.", stacklevel=2)
        baseline_std = 1.0
    effect_size_d = mde_absolute / baseline_std   # Cohen's d

    # ── required n ────────────────────────────────────────────────────────────
    analysis = TTestIndPower()
    n_per_arm = analysis.solve_power(
        effect_size=effect_size_d,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    n_per_arm = int(np.ceil(n_per_arm))

    # ── power curve ───────────────────────────────────────────────────────────
    n_lo = max(10, n_per_arm // 4)
    n_hi = n_per_arm * 3
    step = max(5, n_per_arm // 20)
    n_grid = np.arange(n_lo, n_hi, step)
    pwr_grid = [
        analysis.solve_power(effect_size=effect_size_d, alpha=alpha, nobs1=n,
                              alternative="two-sided")
        for n in n_grid
    ]
    power_curve = pd.DataFrame({"n_per_arm": n_grid, "power": pwr_grid})

    # ── save & display ────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    power_curve.to_csv(os.path.join(OUTPUT_DIR, "power_analysis.csv"), index=False)

    result = dict(
        n_per_arm=n_per_arm,
        n_total=n_per_arm * 2,
        effect_size_cohen_d=round(effect_size_d, 4),
        alpha=alpha,
        power=power,
        mde_relative=mde_relative,
        mde_absolute=round(mde_absolute, 4),
        baseline_mean=round(baseline_mean, 4),
        baseline_std=round(baseline_std, 4),
        power_curve=power_curve,
    )

    if verbose:
        _substage("1.2", "Power analysis complete")
        print(f"  Baseline mean  : {baseline_mean:.4f}")
        print(f"  Baseline sigma : {baseline_std:.4f}")
        print(f"  MDE (relative) : {mde_relative:.1%}  ->  absolute = {mde_absolute:.4f}")
        print(f"  Cohen's d      : {effect_size_d:.4f}")
        print(f"  alpha = {alpha},  target power = {power:.0%}")
        print(f"  -------------------------------------------------")
        print(f"  Required n per arm : {n_per_arm}")
        print(f"  Total subjects     : {n_per_arm * 2}")
        print(f"  Power curve -> outputs/power_analysis.csv")

    return result


# =============================================================================
# STEP 2 — SOTA Covariate-Adaptive Randomization (CAR)
# =============================================================================

def run_car_randomization(
    df: pd.DataFrame | None = None,
    stratify_cols: list[str] | None = None,
    n_per_arm: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    SOTA Covariate-Adaptive Randomization (CAR).

    Algorithm — Mahalanobis-distance minimization (restricted randomization):
      1. Encode the stratification covariates (one-hot for categorical,
         z-score for continuous).
      2. Greedily assign subjects to control / treatment so that the
         Mahalanobis distance between arm covariate means is minimised.
         At each step, the next shuffled subject is tentatively assigned to
         each arm; the arm that would keep its mean closest to the global
         target mean wins the assignment.
      3. If the dataset has more subjects than 2 x n_per_arm, cap before
         assignment.

    Parameters
    ----------
    df            : DataFrame of potential subjects.  Loaded from DATA_PATH
                    if None.
    stratify_cols : Columns to balance.  Defaults to DESIGN_STRATIFY_COLS.
    n_per_arm     : Cap per arm.  If None, all subjects are assigned (half each).
    verbose       : Print progress.

    Returns
    -------
    DataFrame with original columns plus:
        arm        : "control" | "treatment"
        subject_id : sequential integer
    Also writes outputs/randomization_schedule.csv.
    """
    from scipy.linalg import pinvh

    if stratify_cols is None:
        stratify_cols = DESIGN_STRATIFY_COLS

    if verbose:
        _substage("2.1", f"Loading data for CAR (stratify on {stratify_cols})")

    # ── load & validate ───────────────────────────────────────────────────────
    if df is None:
        try:
            df = pd.read_csv(DATA_PATH)
        except FileNotFoundError:
            warnings.warn(
                f"[CAR] DATA_PATH '{DATA_PATH}' not found. "
                "Generating 400 synthetic subjects.", stacklevel=2
            )
            rng = np.random.default_rng(42)
            df = pd.DataFrame({
                "Age":      rng.integers(18, 70, 400).tolist(),
                "Fare":     rng.exponential(30, 400).tolist(),
                "Sex":      rng.choice(["male", "female"], 400).tolist(),
                "Pclass":   rng.choice([1, 2, 3], 400).tolist(),
                "Embarked": rng.choice(["S", "C", "Q"], 400).tolist(),
            })

    df = df.copy().reset_index(drop=True)

    # ── filter to valid stratify columns ─────────────────────────────────────
    valid_stratify = [c for c in stratify_cols if c in df.columns]
    if not valid_stratify:
        warnings.warn(
            f"[CAR] None of stratify_cols {stratify_cols} found in data. "
            "Falling back to simple 50/50 random assignment.", stacklevel=2
        )
        rng = np.random.default_rng(99)
        arms = ["control"] * (len(df) // 2) + ["treatment"] * (len(df) - len(df) // 2)
        df["arm"] = rng.permutation(arms)
        df["subject_id"] = range(len(df))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(os.path.join(OUTPUT_DIR, "randomization_schedule.csv"), index=False)
        return df

    if verbose:
        _substage("2.2", f"Encoding {len(valid_stratify)} stratification features")

    # ── feature matrix ────────────────────────────────────────────────────────
    cat_cols = [
        c for c in valid_stratify
        if df[c].dtype == object or str(df[c].dtype).startswith("category")
    ]
    num_cols = [c for c in valid_stratify if c not in cat_cols]

    frames = []
    if num_cols:
        num_df = df[num_cols].fillna(df[num_cols].median())
        mu    = num_df.mean()
        sigma = num_df.std(ddof=1).replace(0, 1)
        frames.append((num_df - mu) / sigma)
    if cat_cols:
        cat_df = pd.get_dummies(df[cat_cols].fillna("_NA_"), drop_first=True, dtype=float)
        frames.append(cat_df)

    X = pd.concat(frames, axis=1).values.astype(float)

    # ── optionally cap subjects ───────────────────────────────────────────────
    total_needed = (n_per_arm * 2) if n_per_arm else len(df)
    if len(df) > total_needed:
        idx = np.random.default_rng(42).choice(len(df), total_needed, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        X  = X[idx]

    n    = len(df)
    half = n // 2

    if verbose:
        _substage("2.3",
                  f"Mahalanobis-distance CAR on {n} subjects, {X.shape[1]} features")

    # ── precompute regularised inverse covariance ──────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if X.shape[1] > 1:
            cov = np.cov(X, rowvar=False)
        else:
            cov = np.array([[float(np.var(X, ddof=1))]])
    VI = pinvh(cov + np.eye(cov.shape[0]) * 1e-8)

    target_mean = X.mean(axis=0)   # both arms aim for this global mean

    # ── greedy restricted randomisation ───────────────────────────────────────
    assignment   = np.full(n, -1, dtype=int)   # -1=unassigned, 0=control, 1=treatment
    control_idx: list[int] = []
    treat_idx:   list[int] = []
    shuffled = np.random.default_rng(42).permutation(n)

    for s in shuffled:
        n_ctrl = len(control_idx)
        n_trt  = len(treat_idx)

        if n_ctrl >= half:
            arm = 1
        elif n_trt >= half:
            arm = 0
        elif n_ctrl == 0 and n_trt == 0:
            arm = int(np.random.default_rng(int(s)).integers(0, 2))
        else:
            # tentatively assign to each arm and pick the one with smaller
            # squared Mahalanobis distance of new mean from target
            def _sq_mah(existing_idx: list[int]) -> float:
                combined = existing_idx + [int(s)]
                diff = X[combined].mean(axis=0) - target_mean
                return float(diff @ VI @ diff)

            d_ctrl  = _sq_mah(control_idx)
            d_treat = _sq_mah(treat_idx)
            arm = 0 if d_ctrl <= d_treat else 1

        assignment[s] = arm
        if arm == 0:
            control_idx.append(int(s))
        else:
            treat_idx.append(int(s))

    df = df.copy()
    df["arm"]        = np.where(assignment == 0, "control", "treatment")
    df["subject_id"] = range(n)

    # ── save schedule ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    schedule_path = os.path.join(OUTPUT_DIR, "randomization_schedule.csv")
    cols_front = ["subject_id", "arm"] + valid_stratify
    extra_cols  = [c for c in df.columns if c not in cols_front]
    df[cols_front + extra_cols].to_csv(schedule_path, index=False)

    if verbose:
        ctrl_n = (df["arm"] == "control").sum()
        trt_n  = (df["arm"] == "treatment").sum()
        print(f"  Assigned: control={ctrl_n}, treatment={trt_n}  (total={n})")
        print(f"  Randomization schedule -> {schedule_path}")

    return df


# =============================================================================
# STEP 3 — Bootstrap Imbalance Simulation Check
# =============================================================================

def run_simulation_check(
    df_assigned: pd.DataFrame,
    stratify_cols: list[str] | None = None,
    n_simulations: int = DESIGN_N_SIMULATIONS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    1 000-bootstrap imbalance validation.

    For each continuous (or one-hot encoded categorical) stratification
    covariate, compare:
      - SMD achieved by CAR assignment (from df_assigned)
      - Distribution of SMDs under naive simple random assignment (SRS)

    A good CAR SMD should lie well below the median of the SRS distribution
    (ideally bottom 5th percentile).

    Parameters
    ----------
    df_assigned   : Output of run_car_randomization().
    stratify_cols : Same list used in CAR.
    n_simulations : Bootstrap replications (default 1 000).

    Returns
    -------
    DataFrame with columns:
        covariate, car_smd, srs_median_smd, srs_p05_smd,
        srs_p95_smd, car_percentile, balance_ok
    Also writes outputs/car_balance_report.csv.
    """
    if stratify_cols is None:
        stratify_cols = DESIGN_STRATIFY_COLS

    if verbose:
        _substage("3.1", f"Bootstrap imbalance check ({n_simulations} reps)")

    valid_cols = [c for c in stratify_cols if c in df_assigned.columns]

    # separate numeric from categorical
    num_cols = [
        c for c in valid_cols
        if df_assigned[c].dtype not in (object,)
        and not str(df_assigned[c].dtype).startswith("category")
    ]
    cat_cols = [c for c in valid_cols if c not in num_cols]

    # encode categoricals as dummies
    if cat_cols:
        dummy_df = pd.get_dummies(
            df_assigned[cat_cols].fillna("_NA_"), drop_first=False, dtype=float
        )
        df_check = pd.concat(
            [df_assigned[["arm"]], df_assigned[num_cols], dummy_df], axis=1
        )
        check_cols = num_cols + dummy_df.columns.tolist()
    else:
        df_check  = df_assigned
        check_cols = num_cols

    if not check_cols:
        warnings.warn("[SimCheck] No numeric/encodable stratify columns — skipping.",
                      stacklevel=2)
        return pd.DataFrame()

    ctrl_mask = df_check["arm"].values == "control"
    n_ctrl    = ctrl_mask.sum()
    n_total   = len(df_check)

    rng  = np.random.default_rng(0)
    rows = []

    for col in check_cols:
        vals = df_check[col].fillna(df_check[col].median()).values.astype(float)

        # CAR SMD (absolute)
        car_smd = abs(_smd(vals[ctrl_mask], vals[~ctrl_mask]))

        # SRS distribution via bootstrap
        srs_smds: list[float] = []
        p_ctrl = n_ctrl / n_total
        for _ in range(n_simulations):
            rand_ctrl = rng.random(n_total) < p_ctrl
            if rand_ctrl.sum() < 2 or (~rand_ctrl).sum() < 2:
                continue
            srs_smds.append(abs(_smd(vals[rand_ctrl], vals[~rand_ctrl])))

        srs_arr  = np.array(srs_smds)
        pct      = float(np.mean(srs_arr <= car_smd)) * 100   # CAR's percentile in SRS dist.

        rows.append(dict(
            covariate      = col,
            car_smd        = round(car_smd, 4),
            srs_median_smd = round(float(np.median(srs_arr)), 4),
            srs_p05_smd    = round(float(np.percentile(srs_arr, 5)), 4),
            srs_p95_smd    = round(float(np.percentile(srs_arr, 95)), 4),
            car_percentile = round(pct, 1),
            balance_ok     = bool(car_smd < 0.10),   # CONSORT: |SMD| < 0.10
        ))

    report = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report.to_csv(os.path.join(OUTPUT_DIR, "car_balance_report.csv"), index=False)

    if verbose:
        _substage("3.2", "Balance report")
        hdr = f"  {'Covariate':<25} {'CAR SMD':>8} {'SRS med':>8} {'CAR %ile':>9} {'OK?':>5}"
        sep = f"  {'-'*25} {'-'*8} {'-'*8} {'-'*9} {'-'*5}"
        print(f"\n{hdr}\n{sep}")
        for _, row in report.iterrows():
            ok = "OK" if row["balance_ok"] else "FAIL"
            print(
                f"  {row['covariate']:<25} {row['car_smd']:>8.4f} "
                f"{row['srs_median_smd']:>8.4f} {row['car_percentile']:>8.1f}% "
                f"  {ok}"
            )
        n_ok = int(report["balance_ok"].sum())
        print(f"\n  {n_ok}/{len(report)} covariates balanced (|SMD| < 0.10)")
        print(f"  Balance report -> {OUTPUT_DIR}/car_balance_report.csv")

    return report


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main() -> None:
    import time
    t0 = time.perf_counter()

    _banner("PHASE I -- Pre-Experiment Design  (SOTA CAR)")
    print(f"\n  Config loaded from : {os.path.join(BASE_DIR, 'config.py')}")
    print(f"  Output directory   : {OUTPUT_DIR}")
    print(f"  MDE (relative)     : {DESIGN_MDE_RELATIVE:.1%}")
    print(f"  Target power       : {DESIGN_POWER:.0%}")
    print(f"  Alpha (two-sided)  : {DESIGN_ALPHA}")
    print(f"  Stratify columns   : {DESIGN_STRATIFY_COLS}")
    print(f"  Simulations        : {DESIGN_N_SIMULATIONS}")

    # ── Step 1: Power Analysis ─────────────────────────────────────────────────
    _banner("Step 1 -- Power Analysis (TTestIndPower)")
    power_result = run_power_analysis(verbose=True)

    # ── Step 2: SOTA CAR Randomization ────────────────────────────────────────
    _banner("Step 2 -- SOTA Covariate-Adaptive Randomization (Mahalanobis)")
    df_assigned = run_car_randomization(
        stratify_cols=DESIGN_STRATIFY_COLS,
        n_per_arm=power_result["n_per_arm"],
        verbose=True,
    )

    # ── Step 3: Bootstrap Imbalance Simulation ─────────────────────────────────
    _banner("Step 3 -- Bootstrap Imbalance Simulation Check")
    balance_report = run_simulation_check(
        df_assigned,
        stratify_cols=DESIGN_STRATIFY_COLS,
        n_simulations=DESIGN_N_SIMULATIONS,
        verbose=True,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    _banner(f"PHASE I COMPLETE  ({elapsed:.1f}s)")

    print(f"\n  Required n per arm : {power_result['n_per_arm']}")
    print(f"  Total subjects     : {power_result['n_total']}")

    n_balanced  = int(balance_report["balance_ok"].sum()) if len(balance_report) else 0
    n_total_cov = len(balance_report)
    print(f"  Balance check      : {n_balanced}/{n_total_cov} covariates |SMD| < 0.10")

    print(f"\n  -- Outputs written to: {OUTPUT_DIR} --")
    print("  power_analysis.csv           -- power curve (n vs. power)")
    print("  randomization_schedule.csv   -- subject -> arm assignment (share with Eng.)")
    print("  car_balance_report.csv       -- per-covariate SMD: CAR vs. SRS baseline")
    print()
    print("  Next step: Share randomization_schedule.csv with Engineering.")
    print("  After the experiment, run:  python -X utf8 main.py")


if __name__ == "__main__":
    main()
