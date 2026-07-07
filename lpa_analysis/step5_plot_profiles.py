"""
step5_plot_profiles.py
──────────────────────
Create a publication-quality line plot of latent profile indicator means
(standardised z-scores) with 95% CI error bars.

  X-axis  : Indicator variable labels
  Y-axis  : Standardised mean score (z)
  Lines   : One per profile (distinct colour + marker)
  Error   : ±1.96 × SEM (95% CI)

Saves:
  - outputs/profiles_lineplot.png   (300 dpi, for publication)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from config import OUTPUT_DIR, INDICATOR_COLS, N_PROFILES

# ── Publication-style palette (colourblind-friendly) ──────────────────────────
PALETTE  = ["#2E86AB", "#E84855", "#3BB273", "#F18F01", "#7B2D8B", "#FF6B6B"]
MARKERS  = ["o", "s", "^", "D", "v", "P"]
LABELS   = [f"Profile {k+1}" for k in range(N_PROFILES)]


def plot_profiles(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    z_cols = [f"{c}_z" for c in INDICATOR_COLS]
    x_labels = INDICATOR_COLS

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    for k in range(1, N_PROFILES + 1):
        group  = df[df["Profile"] == k][z_cols]
        means  = group.mean().values
        sems   = (group.sem() * 1.96).values    # 95% CI half-width

        color  = PALETTE[(k - 1) % len(PALETTE)]
        marker = MARKERS[(k - 1) % len(MARKERS)]

        ax.plot(
            x_labels, means,
            marker=marker, color=color, linewidth=2.2,
            markersize=7, label=f"Profile {k} (n={len(group)})",
        )
        ax.fill_between(
            x_labels,
            means - sems,
            means + sems,
            alpha=0.12, color=color,
        )
        ax.errorbar(
            x_labels, means, yerr=sems,
            fmt="none", color=color, capsize=4, linewidth=1.2,
        )

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xlabel("Indicator Variable", fontsize=12, labelpad=8)
    ax.set_ylabel("Standardised Mean Score (z)", fontsize=12, labelpad=8)
    ax.set_title(
        f"Latent Profile Indicator Means (K = {N_PROFILES})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(
        title="Profile", fontsize=10, title_fontsize=10,
        framealpha=0.9, edgecolor="#cccccc",
    )
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "profiles_lineplot.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Step 5] Line plot saved → {out_path}")


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "lpa_profiles.csv"))
    plot_profiles(df)
