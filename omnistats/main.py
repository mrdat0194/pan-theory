"""
omnistats/main.py
─────────────────
OmniStats Pipeline Orchestrator
════════════════════════════════
Runs the full four-stage statistical analysis pipeline:

  Stage 1 — Latent Profile Analysis (LPA)
    Step 1 : Load & prepare data
    Step 2 : Fit GMM models (K_MIN .. K_MAX), assign profiles (N_PROFILES)
    Step 3 : Welch ANOVA + Games-Howell post-hoc per indicator
    Step 4 : Chi-square tests + Cramér's V per demographic variable
    Step 5 : Visualise profiles and demographics
    Step 6 : Build APA 7th edition Word report (Tables 1–4)

  Stage 2 — A/B Testing
    Step 7 : Proportion test, Welch t-test, distribution fit test

  Stage 3 — Causal Inference
    Step 8 : Difference-in-Differences, IV/2SLS, Regression Discontinuity

  Stage 4 — Consolidated APA Report
    Step 9 : Append A/B and Causal results to the Word document (Tables 5–6)

Usage
─────
    python main.py

Tip — Choose your number of profiles:
  After Stage 1 Step 2, inspect outputs/lpa_fit_stats.csv.
  Set N_PROFILES = K in config.py and re-run.
"""
import os
import sys
import time

# Make sure modules can find config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import (
    N_PROFILES, OUTPUT_DIR,
    AB_GROUP_COL, AB_METRIC_COL, AB_CONVERSION_COL,
)


def banner(msg: str):
    w = 64
    print("\n" + "=" * w)
    print(f"  {msg}")
    print("=" * w)


def step(n: int, desc: str):
    print(f"\n[Step {n}] {desc}")


def main():
    t0 = time.perf_counter()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── STAGE 1: LPA ─────────────────────────────────────────────────────────
    banner("STAGE 1 — Latent Profile Analysis")

    step(1, "Load & prepare data")
    from data_manager import load_and_prepare
    df = load_and_prepare()

    step(2, f"Run LPA (K_MIN..K_MAX), assign K={N_PROFILES} profiles")
    from modules.lpa import run_lpa
    df_profiles, fit_df = run_lpa(df)

    step(3, "Welch ANOVA + Games-Howell post-hoc")
    from modules.anova import run_anova
    anova_df, posthoc_df = run_anova(df_profiles)

    step(4, "Chi-square tests + Cramér's V")
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

    # ─── STAGE 2: A/B TESTING ─────────────────────────────────────────────────
    banner("STAGE 2 — A/B Testing")

    step(6, f"A/B tests on '{AB_GROUP_COL}' groups using '{AB_METRIC_COL}'")
    from modules.ab_testing import run_ab_tests
    ab_results = run_ab_tests(
        df_profiles,
        group_col=AB_GROUP_COL,
        metric_col=AB_METRIC_COL,
        conversion_col=AB_CONVERSION_COL if AB_CONVERSION_COL in df_profiles.columns else None,
    )

    # ─── STAGE 3: CAUSAL INFERENCE ────────────────────────────────────────────
    banner("STAGE 3 — Causal Inference (Synthetic Demo)")

    step(7, "Run DiD, IV/2SLS, and RDD causal estimators")
    from modules.causal_inference import run_causal_suite
    causal_results = run_causal_suite()

    # ─── STAGE 4: APA REPORT ──────────────────────────────────────────────────
    banner("STAGE 4 — Build APA 7th Edition Report")

    step(8, "Generate consolidated Word document")
    from modules.apa_report import build_report
    build_report()

    # ─── DONE ─────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")

    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print("  lpa_fit_stats.csv         — LPA model fit (review to choose K)")
    print("  lpa_profiles.csv          — Full dataset with profile assignments")
    print("  anova_results.csv         — Welch ANOVA summary")
    print("  anova_posthoc.csv         — Games-Howell pairwise comparisons")
    print("  chi_square_results.csv    — Chi-square + Cramér's V")
    print("  chi_square_tables.csv     — Observed frequency crosstabs")
    print("  ab_test_results.csv       — A/B test summary")
    print("  causal_results.csv        — DiD / IV / RDD causal estimates")
    print("  profiles_lineplot.png     — LPA profile means plot")
    print("  demographics_plot.png     — Demographic stacked bar charts")
    print("  posthoc_heatmap.png       — Games-Howell p-value heatmap")
    print("  chi_square_mosaic.png     — Chi-square tile chart")
    print("  rdd_plot.png              — RDD scatter + fit lines")
    print("  apa_report.docx           — Full APA 7th edition tables (Tables 1-6)")
    print(f"\nNext step: Open outputs/lpa_fit_stats.csv, agree on K,")
    print(f"  set N_PROFILES = K in config.py, then re-run: python -X utf8 main.py")


if __name__ == "__main__":
    main()
