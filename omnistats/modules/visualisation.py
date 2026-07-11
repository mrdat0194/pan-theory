"""
omnistats/modules/visualisation.py
────────────────────────────────────
Unified visualisation module.
Combines plots from lpa_analysis/step5*.py and adds new comparison charts.

Functions
─────────
plot_lpa_profiles()       — Line plot of profile indicator z-means with 95% CI
plot_demographics()       — Stacked bar charts of demographic composition
plot_posthoc_heatmap()    — Heatmap of Games-Howell p-values across indicators
plot_chi_square_mosaic()  — Mosaic / tile plot of chi-square crosstabs
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, INDICATOR_COLS, DEMOGRAPHIC_COLS, N_PROFILES

# ── Shared palette ─────────────────────────────────────────────────────────────
PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F18F01", "#7B2D8B", "#FF6B6B"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


# ─── 1. LPA Profile Line Plot ─────────────────────────────────────────────────

def plot_lpa_profiles(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Publication-quality line plot of latent profile indicator z-means (±95% CI).
    Saved to outputs/profiles_lineplot.png
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    z_cols   = [f"{c}_z" for c in INDICATOR_COLS if f"{c}_z" in df.columns]
    x_labels = [c.replace("_z", "") for c in z_cols]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    for k in range(1, N_PROFILES + 1):
        grp   = df[df["Profile"] == k][z_cols]
        means = grp.mean().values
        sems  = (grp.sem() * 1.96).values

        c = PALETTE[(k - 1) % len(PALETTE)]
        m = MARKERS[(k - 1) % len(MARKERS)]

        ax.plot(x_labels, means, marker=m, color=c, linewidth=2.2,
                markersize=7, label=f"Profile {k} (n={len(grp)})")
        ax.fill_between(x_labels, means - sems, means + sems, alpha=0.12, color=c)
        ax.errorbar(x_labels, means, yerr=sems, fmt="none", color=c, capsize=4, linewidth=1.2)

    ax.set_xlabel("Indicator Variable", fontsize=12, labelpad=8)
    ax.set_ylabel("Standardised Mean (z)", fontsize=12, labelpad=8)
    ax.set_title(f"Latent Profile Indicator Means (K = {N_PROFILES})", fontsize=13, fontweight="bold", pad=12)
    ax.legend(title="Profile", fontsize=10, title_fontsize=10, framealpha=0.9, edgecolor="#cccccc")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "profiles_lineplot.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"[Viz] Profile line plot saved -> {path}")


# ─── 2. Demographic Stacked Bar Chart ────────────────────────────────────────

def plot_demographics(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Stacked bar charts showing demographic composition (%) within each profile.
    Saved to outputs/demographics_plot.png
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    demo_cols = [c for c in DEMOGRAPHIC_COLS if c in df.columns]
    if not demo_cols:
        return

    df_plot = df.copy()
    label_maps = {
        "Pclass":    {1: "1st", 2: "2nd", 3: "3rd"},
        "Embarked":  {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"},
    }
    for col, lmap in label_maps.items():
        if col in df_plot.columns:
            df_plot[col] = df_plot[col].map(lmap).fillna(df_plot[col].astype(str))
    if "Sex" in df_plot.columns:
        df_plot["Sex"] = df_plot["Sex"].str.title()

    fig, axes = plt.subplots(1, len(demo_cols), figsize=(5.5 * len(demo_cols), 5.5), sharey=False)
    if len(demo_cols) == 1:
        axes = [axes]

    cmap = plt.get_cmap("Set2")

    for i, col in enumerate(demo_cols):
        ax = axes[i]
        ct = pd.crosstab(df_plot["Profile"], df_plot[col], normalize="index") * 100
        colors = [cmap(j / max(len(ct.columns) - 1, 1)) for j in range(len(ct.columns))]
        ct.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)

        ax.set_title(f"Distribution of {col}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Latent Profile", fontsize=10)
        ax.set_ylabel("Percentage (%)" if i == 0 else "", fontsize=10)
        ax.set_xticklabels([f"Profile {x}" for x in ct.index], rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(title=col, framealpha=0.9, edgecolor="#cccccc", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        for container in ax.containers:
            labels = [f"{v:.1f}%" if v > 5 else "" for v in container.datavalues]
            ax.bar_label(container, labels=labels, label_type="center",
                         fontsize=9, fontweight="bold", color="white")

    plt.suptitle("Demographic Composition of Latent Profiles", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "demographics_plot.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"[Viz] Demographic plot saved -> {path}")


# ─── 3. Post-Hoc p-value Heatmap ─────────────────────────────────────────────

def plot_posthoc_heatmap(posthoc_df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Heatmap of Games-Howell p-values for each indicator × profile pair.
    Cells are colour-coded by significance (green < .05, yellow .05–.10, red > .10).
    Saved to outputs/posthoc_heatmap.png
    """
    if posthoc_df.empty:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    indicators = posthoc_df["Indicator"].unique() if "Indicator" in posthoc_df.columns else []
    if len(indicators) == 0:
        return

    pairs = [(row["Group_A"], row["Group_B"]) for _, row in posthoc_df.drop_duplicates(["Group_A", "Group_B"]).iterrows()]
    pair_labels = [f"P{a} vs P{b}" for a, b in pairs]

    matrix = np.full((len(indicators), len(pairs)), np.nan)
    for i, ind in enumerate(indicators):
        sub = posthoc_df[posthoc_df["Indicator"] == ind]
        for j, (a, b) in enumerate(pairs):
            row = sub[(sub["Group_A"] == a) & (sub["Group_B"] == b)]
            if not row.empty:
                matrix[i, j] = row.iloc[0]["p"]

    fig, ax = plt.subplots(figsize=(max(5, len(pairs) * 1.2), max(3, len(indicators) * 0.8)))
    cmap = plt.get_cmap("RdYlGn_r")
    im   = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=0.15, aspect="auto")

    ax.set_xticks(range(len(pair_labels))); ax.set_xticklabels(pair_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(indicators)));  ax.set_yticklabels(indicators, fontsize=10)

    for i in range(len(indicators)):
        for j in range(len(pairs)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if (val < 0.02 or val > 0.12) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=text_color)

    plt.colorbar(im, ax=ax, label="p-value")
    ax.set_title("Games-Howell Post-Hoc p-values", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "posthoc_heatmap.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"[Viz] Post-hoc heatmap saved -> {path}")


# ─── 4. Chi-Square Mosaic / Tile Plot ────────────────────────────────────────

def plot_chi_square_mosaic(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Tile (heatmap-style) chart showing % of each demographic category per profile.
    Saved to outputs/chi_square_mosaic.png
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    demo_cols = [c for c in DEMOGRAPHIC_COLS if c in df.columns]
    if not demo_cols:
        return

    n_cols = len(demo_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, demo_cols):
        ct = pd.crosstab(df[col], df["Profile"], normalize="columns") * 100
        im = ax.imshow(ct.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels([f"P{c}" for c in ct.columns], fontsize=9)
        ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index.astype(str), fontsize=9)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("Profile"); ax.set_ylabel("Category")
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                val = ct.values[i, j]
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if val > 55 else "black")
        plt.colorbar(im, ax=ax, label="%")

    plt.suptitle("Demographic Category Distribution by Profile", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chi_square_mosaic.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"[Viz] Chi-square mosaic saved -> {path}")
