"""
omnistats/main.py
-----------------
OmniStats Pipeline Orchestrator -- v3

STAGE PURPOSE MAP
-----------------
  Stage 1 -- Latent Profile Analysis (LPA)
    PURPOSE: DESCRIBE -- Segment the population into meaningfully distinct
    behavioural profiles using Gaussian Mixture Models.
    Q: "How many different user types exist, and how do they differ?"

  Stage 2 -- A/B Testing (Frequentist + Bayesian Sequential)
    PURPOSE: COMPARE -- Measure the size of a treatment effect with valid
    uncertainty quantification. Solves the peaking problem via Bayesian
    posterior thresholds rather than fixed-horizon p-values.
    Q: "Did Treatment B improve the metric? By how much, with what certainty?"

  Stage 3 -- CUPED Variance Reduction
    PURPOSE: SHARPEN -- Reduce outcome variance using the LPA profile score
    (Stage 1 output) as a pre-experiment covariate.
    Output df_cuped is the DIRECT INPUT to Stage 4 Causal Inference.
    Q: "Can we remove predictable variance to tighten all downstream CIs?"

  Stage 4 -- Causal Inference (All Estimators + CausalImpact BSTS)
    PURPOSE: ATTRIBUTE -- Explain *why* an effect occurred.
    Estimators: DiD, IV, RDD, SCM, Matrix Completion, CausalImpact (BSTS).
    CausalImpact is DiD generalised to continuous time -- same Stage 4.
    All results written to causal_results.csv for APA Table 8.
    Q: "Is the difference caused by the treatment, not an unobserved confound?"

  Stage 5 -- APA Report (CONSOLIDATE)
    PURPOSE: Read output CSVs from ALL stages 1-4 and render APA 7th edition.
    No new estimates are produced in Stage 5.
    Table order: 1-4 LPA | 5 Freq A/B | 6 Bayesian A/B | 7 CUPED | 8 Causal Suite

Usage
-----
    python -X utf8 main.py

Tip: After Stage 1 Step 2, inspect outputs/lpa_fit_stats.csv.
     Set N_PROFILES = K in config.py and re-run.
"""

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


def step(n, desc: str):
    print(f"\n[Step {n}] {desc}")


def main():
    t0 = time.perf_counter()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # STAGE 1 -- Latent Profile Analysis
    # PURPOSE: DESCRIBE -- segment users into behavioural profiles
    # =========================================================================
    banner("STAGE 1 -- Latent Profile Analysis  [DESCRIBE]")

    step(1, "Load & prepare data")
    from data_manager import load_and_prepare
    df = load_and_prepare()

    step(2, f"Run LPA (K_MIN..K_MAX), assign K={N_PROFILES} profiles")
    from modules.lpa import run_lpa
    df_profiles, fit_df = run_lpa(df)

    step(3, "Welch ANOVA + Games-Howell post-hoc")
    from modules.anova import run_anova
    anova_df, posthoc_df = run_anova(df_profiles)

    step(4, "Chi-square tests + Cramer's V")
    from modules.chi_square import run_chi_square
    chi2_df, crosstab_df = run_chi_square(df_profiles)

    step(5, "Visualise profiles and demographics")
    from modules.visualisation import (
        plot_lpa_profiles, plot_demographics,
        plot_posthoc_heatmap, plot_chi_square_mosaic,
    )
    plot_lpa_profiles(df_profiles)
    plot_demographics(df_profiles)
    plot_posthoc_heatmap(posthoc_df)
    plot_chi_square_mosaic(df_profiles)

    # =========================================================================
    # STAGE 2 -- A/B Testing (Frequentist + Bayesian Sequential)
    # PURPOSE: COMPARE -- measure effect size with valid uncertainty
    # =========================================================================
    banner("STAGE 2 -- A/B Testing (Frequentist + Bayesian Sequential)  [COMPARE]")

    step("6a", f"Frequentist A/B: proportion z-test, Welch t-test, dist. fit")
    from modules.ab_testing import run_ab_tests
    ab_results = run_ab_tests(
        df_profiles,
        group_col=AB_GROUP_COL,
        metric_col=AB_METRIC_COL,
        conversion_col=AB_CONVERSION_COL if AB_CONVERSION_COL in df_profiles.columns else None,
    )

    step("6b", "Bayesian A/B: Beta-Binomial (PyMC / IS fallback), StudentT means (PyMC NUTS)")
    from modules.bayesian import run_bayesian_ab_tests
    bayesian_ab_results = run_bayesian_ab_tests(
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

    # =========================================================================
    # STAGE 3 -- CUPED Variance Reduction
    # PURPOSE: SHARPEN -- adjust outcome so Stage 4 causal CIs are tighter
    # df_cuped IS the direct input to run_causal_suite() in Stage 4
    # =========================================================================
    banner("STAGE 3 -- CUPED Variance Reduction  [SHARPEN]")

    step("6c", f"CUPED: monotonic covariate adjustment (covariate='{CUPED_COVARIATE_COL}')")
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
        if CUPED_ENABLED:
            print(f"  [CUPED] Covariate '{CUPED_COVARIATE_COL}' not found -- "
                  f"skipping. Run Stage 1 LPA first.")
        df_cuped = df_profiles.copy()

    # =========================================================================
    # STAGE 4 -- Causal Inference: ALL estimators including CausalImpact BSTS
    # PURPOSE: ATTRIBUTE -- DiD, IV, RDD, SCM, MC, CausalImpact (same stage)
    # CausalImpact is DiD generalised to continuous time (BSTS + spike-and-slab)
    # All six results written to causal_results.csv -> APA Table 8
    # =========================================================================
    banner("STAGE 4 -- Causal Inference (DiD / IV / RDD / SCM / MC / CausalImpact)  [ATTRIBUTE]")

    step(7, "Run full causal suite: DiD, IV, RDD, SCM, Matrix Completion, "
            "CausalImpact BSTS -- all operating on df_cuped")
    from modules.causal import run_causal_suite
    causal_results = run_causal_suite()

    # =========================================================================
    # STAGE 5 -- APA Report (CONSOLIDATE)
    # PURPOSE: Read output CSVs from ALL stages 1-4; render Word document.
    # No new estimates are computed here -- pure read-and-render pass.
    # Table 1-4: LPA  |  Table 5: Freq A/B  |  Table 6: Bayesian A/B
    # Table 7: CUPED  |  Table 8: Full Causal Suite (incl. CausalImpact)
    # =========================================================================
    banner("STAGE 5 -- APA Report  [CONSOLIDATE]")

    step(8, "Generate APA 7th edition report (Tables 1-8) from all stage outputs")
    from modules.apa_report import build_report
    build_report()

    # =========================================================================
    # DONE
    # =========================================================================
    elapsed = time.perf_counter() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")

    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print()
    print("  -- Stage 1 (LPA) ---------------------------------------------------")
    print("  lpa_fit_stats.csv          -- Review to choose K, then re-run")
    print("  lpa_profiles.csv           -- Full dataset with profile assignments")
    print("  anova_results.csv          -- Welch ANOVA summary")
    print("  anova_posthoc.csv          -- Games-Howell pairwise comparisons")
    print("  chi_square_results.csv     -- Chi-square + Cramer's V")
    print("  profiles_lineplot.png      -- LPA profile means plot")
    print("  demographics_plot.png      -- Demographic stacked bar charts")
    print()
    print("  -- Stage 2 (A/B Testing) -------------------------------------------")
    print("  ab_test_results.csv        -- Frequentist A/B summary  -> APA Table 5")
    print("  bayesian_ab_results.csv    -- P(B>A), Expected Loss, ESS -> APA Table 6")
    print("  dist_fit_*.png             -- Distribution fit plots")
    print()
    print("  -- Stage 3 (CUPED) -------------------------------------------------")
    print("  cuped_variance_reduction.csv -- theta, variance reduction % -> APA Table 7")
    print()
    print("  -- Stage 4 (Causal Inference + CausalImpact) -----------------------")
    print("  causal_results.csv         -- DiD/IV/RDD/SCM/MC/CausalImpact -> APA Table 8")
    print("  did_attgt.csv              -- Full ATT(g,t) table")
    print("  iv_estimates.csv           -- IV 2SLS + diagnostics")
    print("  rdd_results.csv            -- RDD estimate + CCT bandwidth")
    print("  scm_weights.csv            -- SCM donor unit weights")
    print("  scm_gaps.csv / scm_plot.png")
    print("  mc_gaps.csv / mc_plot.png  -- Matrix Completion counterfactual")
    print("  ts_causalimpact.png        -- CausalImpact BSTS plot")
    print("  ts_counterfactual.png      -- Prophet fallback plot (if needed)")
    print()
    print("  -- Stage 5 (APA Report) --------------------------------------------")
    print("  apa_report.docx            -- APA 7th edition (Tables 1-8)")
    print()
    print("Next step: Open outputs/lpa_fit_stats.csv, agree on K,")
    print("  set N_PROFILES = K in config.py, then re-run: python -X utf8 main.py")


if __name__ == "__main__":
    main()
