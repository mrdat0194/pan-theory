"""
omnistats/main.py
-----------------
OmniStats Unified 4-Stage Statistical & AI Pipeline Orchestrator

Usage:
    python main.py [--mode design|eval|plan|all]

Modes:
    design : Stage 1 -- Pre-Experiment Design (Power analysis & CAR schedule)
    eval   : Stage 3 -- Post-Experiment Evaluation (Diagnostics, LPA, A/B, CUPED, Causal Suite, APA Report) [DEFAULT]
    plan   : Stage 4 -- World Model Planning (JEPA Bridge & CEM/MPPI optimization)
    all    : Run all stages sequentially (Stage 1 -> Stage 3 -> Stage 4)
"""

import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import (
    N_PROFILES, OUTPUT_DIR,
    AB_GROUP_COL, AB_METRIC_COL, AB_CONVERSION_COL,
    BAYES_AB_PRIOR_ALPHA, BAYES_AB_PRIOR_BETA,
    BAYES_AB_THRESHOLD, BAYES_AB_LOSS_THRESH,
    BAYES_AB_N_SAMPLES, BAYES_AB_TUNE, BAYES_AB_SEED,
    CUPED_ENABLED, CUPED_COVARIATE_COL,
    CUPED_MONOTONE_DIR, CUPED_USE_CATBOOST,
)


def banner(msg: str):
    w = 68
    print("\n" + "=" * w)
    print(f"  {msg}")
    print("=" * w)


def substage(stage_id: str, desc: str):
    print(f"\n[Stage {stage_id}] {desc}")


def run_stage_eval():
    """Stage 3: Post-Experiment Evaluation (Diagnostics, LPA, A/B, CUPED, Causal Suite, APA Report)."""
    t0 = time.perf_counter()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from data_manager import load_and_prepare
    df = load_and_prepare()

    # Diagnostics
    banner("STAGE 3.0 -- Pre-Flight Diagnostics (VALIDATE)")
    substage("3.0", "Run Pre-Flight Linear Algebra, MMD & Statistical Diagnostics")
    from modules.diagnostics import run_stage0_diagnostics
    from config import INDICATOR_COLS
    diag_res = run_stage0_diagnostics(
        df,
        indicator_cols=INDICATOR_COLS,
        ab_group_col=AB_GROUP_COL,
        ab_metric_col=AB_METRIC_COL
    )

    # Latent Profile Analysis
    banner("STAGE 3.1 -- Latent Profile Analysis  [DESCRIBE]")
    substage("3.1.1", f"Run LPA (K_MIN..K_MAX), assign K={N_PROFILES} profiles")
    from modules.lpa import run_lpa
    df_profiles, fit_df = run_lpa(df)

    substage("3.1.2", "Welch ANOVA + Games-Howell post-hoc")
    from modules.anova import run_anova
    run_anova(df_profiles)

    substage("3.1.3", "Chi-square tests + Cramer's V")
    from modules.chi_square import run_chi_square
    run_chi_square(df_profiles)

    substage("3.1.4", "Visualise profiles and demographics")
    from modules.visualisation import (
        plot_lpa_profiles, plot_demographics,
        plot_posthoc_heatmap, plot_chi_square_mosaic,
    )
    plot_lpa_profiles(df_profiles)
    plot_demographics(df_profiles)

    # A/B Testing
    banner("STAGE 3.2 -- A/B Testing (Frequentist + Bayesian Sequential)  [COMPARE]")
    substage("3.2.1", "Frequentist A/B: proportion z-test, Welch t-test, MMD")
    from modules.ab_testing import run_ab_tests
    run_ab_tests(
        df_profiles,
        group_col=AB_GROUP_COL,
        metric_col=AB_METRIC_COL,
        conversion_col=AB_CONVERSION_COL if AB_CONVERSION_COL in df_profiles.columns else None,
        mmd_val=diag_res.get("mmd_val"),
    )

    substage("3.2.2", "Bayesian A/B: StudentT means (PyMC NUTS)")
    from modules.bayesian import run_bayesian_ab_tests
    run_bayesian_ab_tests(
        df_profiles,
        group_col=AB_GROUP_COL,
        metric_col=AB_METRIC_COL,
        conversion_col=AB_CONVERSION_COL if AB_CONVERSION_COL in df_profiles.columns else None,
        prior_alpha=BAYES_AB_PRIOR_ALPHA,
        prior_beta=BAYES_AB_PRIOR_BETA,
        threshold=BAYES_AB_THRESHOLD,
        loss_thresh=BAYES_AB_LOSS_THRESH,
        n_samples=BAYES_AB_N_SAMPLES,
        tune=BAYES_AB_TUNE,
        seed=BAYES_AB_SEED,
    )

    # CUPED Variance Reduction
    banner("STAGE 3.3 -- CUPED Variance Reduction  [SHARPEN]")
    substage("3.3.1", f"CUPED: monotonic covariate adjustment ('{CUPED_COVARIATE_COL}')")
    from modules.cuped import run_cuped
    if CUPED_ENABLED and CUPED_COVARIATE_COL in df_profiles.columns:
        df_cuped = run_cuped(
            df_profiles,
            outcome_col=AB_METRIC_COL,
            covariate_col=CUPED_COVARIATE_COL,
            group_col=AB_GROUP_COL,
            monotone_dir=CUPED_MONOTONE_DIR,
            use_catboost=CUPED_USE_CATBOOST,
        )
    else:
        df_cuped = df_profiles.copy()

    # Causal Inference Suite
    banner("STAGE 3.4 -- Causal Inference Suite  [ATTRIBUTE]")
    substage("3.4.1", "Run full causal suite: DiD, IV, RDD, SCM, MC, BSTS, HTE")
    from modules.causal import run_causal_suite
    run_causal_suite()

    # APA Report Generation
    banner("STAGE 3.5 -- APA Report Generation  [CONSOLIDATE]")
    substage("3.5.1", "Generate APA 7th edition report (Tables 1-8)")
    from modules.apa_report import build_report
    build_report()

    elapsed = time.perf_counter() - t0
    banner(f"STAGE 3 EVALUATION COMPLETE  ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="OmniStats Unified 4-Stage Statistical & AI Pipeline Orchestrator"
    )
    parser.add_argument(
        "--mode",
        choices=["design", "eval", "plan", "all"],
        default="eval",
        help="Pipeline execution mode: design (Stage 1), eval (Stage 3) [DEFAULT], plan (Stage 4), all (1->3->4)",
    )
    # Pass-through args for Stage 4 planning
    parser.add_argument("--planner", choices=["cem", "mppi"], default="cem")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--plan-length", type=int, default=5)

    args, unknown = parser.parse_known_args()

    if args.mode in ("design", "all"):
        banner("STAGE 1 -- Pre-Experiment Design (experiment_design.py)")
        import experiment_design
        experiment_design.main()

    if args.mode in ("execution", "all"):
        banner("STAGE 2 -- Execution (Outside OmniStats / Field Trial)")
        print("\n  [Stage 2] Live A/B test run executed by Engineering on traffic.")
        print("  Raw experimental data collected and saved to DATA_PATH.")

    if args.mode in ("eval", "all"):
        run_stage_eval()

    if args.mode in ("plan", "all"):
        banner("STAGE 4 -- World Model Planning (plan_experiment.py)")
        import plan_experiment
        plan_args = [
            "--planner", args.planner,
            "--epochs", str(args.epochs),
            "--n-iters", str(args.n_iters),
            "--n-samples", str(args.n_samples),
            "--plan-length", str(args.plan_length),
        ]
        plan_experiment.main(plan_args)


if __name__ == "__main__":
    main()
