"""
main.py — LPA Pipeline Orchestrator
─────────────────────────────────────
Runs all 6 steps sequentially.

Usage:
    python main.py

After Step 2, open outputs/lpa_fit_stats.csv, decide on K with your co-author,
then update N_PROFILES in config.py and re-run from Step 2 onward.
"""
import os
import sys
import time

# Ensure the lpa_analysis dir is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, N_PROFILES


def banner(msg: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Prepare data ──────────────────────────────────────────────────
    banner("Step 1 / 6 — Preparing data")
    t0 = time.time()
    from step1_prepare_data import prepare_data
    df = prepare_data()
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 2: Run LPA ───────────────────────────────────────────────────────
    banner(f"Step 2 / 6 — Running LPA (K = 1–6), assigning profiles with K = {N_PROFILES}")
    t0 = time.time()
    from step2_run_lpa import run_lpa
    df_profiles, fit_df = run_lpa(df)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(
        f"\n  *** Review outputs/lpa_fit_stats.csv with your co-author. ***\n"
        f"  *** Update N_PROFILES in config.py, then re-run if needed.  ***"
    )

    # ── Step 3: Welch ANOVA ───────────────────────────────────────────────────
    banner("Step 3 / 6 — Welch ANOVA + Games-Howell post-hoc")
    t0 = time.time()
    from step3_test_anova import run_anova
    anova_df, posthoc_df = run_anova(df_profiles)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 4: Chi-square ────────────────────────────────────────────────────
    banner("Step 4 / 6 — Chi-square + Cramér's V")
    t0 = time.time()
    from step4_test_chi_square import run_chi_square
    chi2_df, _ = run_chi_square(df_profiles)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 5: Line plot & Demographics Plot ─────────────────────────────────
    banner("Step 5 / 6 — Profile & Demographics plots")
    t0 = time.time()
    from step5_plot_profiles import plot_profiles
    plot_profiles(df_profiles)
    from step5b_plot_demographics import plot_demographics
    plot_demographics(df_profiles)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 6: APA tables ────────────────────────────────────────────────────
    banner("Step 6 / 6 — APA 7th edition Word tables")
    t0 = time.time()
    from step6_apa_tables import build_apa_tables
    build_apa_tables()
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Pipeline Complete")
    print(f"  Outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"    lpa_fit_stats.csv    — model fit table (choose K here)")
    print(f"    lpa_profiles.csv     — data with profile assignments")
    print(f"    anova_results.csv    — Welch ANOVA per indicator")
    print(f"    anova_posthoc.csv    — Games-Howell pairwise comparisons")
    print(f"    chi_square_results.csv — chi-square per demographic")
    print(f"    profiles_lineplot.png  — publication-quality line plot")
    print(f"    apa_tables.docx        — APA 7th edition Word report")


if __name__ == "__main__":
    main()
