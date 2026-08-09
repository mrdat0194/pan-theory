"""
omnistats/modules/xai_visualisation.py
-----------------------------------------
Quantum-Inspired XAI Visualizations for JEPA World Model.

Three new publication-quality dark-mode plot functions that expose the
internal quantum thermodynamics of the JEPA latent space:

  plot_energy_landscape()          - Boltzmann Energy heatmap + histogram
  plot_information_decision_boundary() - JKL / Shannon Entropy PCA scatter
  plot_partition_function_evolution()  - Z(beta) rollout trace

These connect directly to the theoretical framework from the handwritten notes:
  - Aug 2020: JKL, Shannon Entropy, Decision Boundaries
  - Nov 2022: Partition Function Z(beta), Path Integral convolution
  - Tang 2023: Dequantized l2 importance sampling of low-rank latent spaces
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

# Shared dark palette matching main visualisation.py
_DARK_BG    = "#0d1117"
_PANEL_BG   = "#161b22"
_BORDER     = "#30363d"
_BLUE       = "#58a6ff"
_GREEN      = "#3bb273"
_RED        = "#f78166"
_WHITE      = "white"


# =============================================================================
# 5. XAI — ENERGY LANDSCAPE (Boltzmann / Quantum-Inspired)
# =============================================================================

def plot_energy_landscape(
    z_embeddings: np.ndarray,
    energies: np.ndarray,
    labels: list[str] | None = None,
    title: str = "JEPA Latent Energy Landscape",
    verbose: bool = True,
) -> None:
    """
    Visualize the Boltzmann Energy landscape of JEPA latent embeddings.

    Projects the latent space to 2D via PCA and overlays energy E(z) as colour.
    Low-energy regions are stable attractor states in the quantum world model.
    High-energy regions correspond to high-surprise / information-rich states.

    Grounded in pi(x,n) ∝ e^{-beta E_n} psi_n(x) psi_n*(x) (Nov-2022 notes).

    Parameters
    ----------
    z_embeddings : np.ndarray  [N, D]  JEPA latent vectors
    energies     : np.ndarray  [N]     Boltzmann energy of each embedding
    labels       : list[str]   optional annotation labels for each point
    title        : str
    verbose      : bool
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("[Viz/XAI] scikit-learn not installed — skipping energy landscape.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pca  = PCA(n_components=2)
    z_2d = pca.fit_transform(z_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(_DARK_BG)

    # ── Left: 2-D scatter coloured by energy ───────────────────────────────
    ax = axes[0]
    ax.set_facecolor(_PANEL_BG)
    sc = ax.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=energies, cmap="plasma", s=60, alpha=0.85,
        edgecolors=_WHITE, linewidths=0.3,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Boltzmann Energy E(z)", color=_WHITE)
    cbar.ax.yaxis.set_tick_params(color=_WHITE)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_WHITE)
    ax.set_xlabel("PC1", color=_WHITE)
    ax.set_ylabel("PC2", color=_WHITE)
    ax.set_title("Latent Energy Distribution", color=_WHITE)
    ax.tick_params(colors=_WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    if labels is not None:
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, z_2d[i], fontsize=7, color="lightgray", alpha=0.7)

    # ── Right: Energy histogram (partition function slice) ─────────────────
    ax2 = axes[1]
    ax2.set_facecolor(_PANEL_BG)
    ax2.hist(energies, bins=30, color=_BLUE, edgecolor=_DARK_BG, alpha=0.85)
    ax2.axvline(
        energies.mean(), color=_RED, linestyle="--", linewidth=2,
        label="Mean E = {:.2f}".format(energies.mean()),
    )
    ax2.set_xlabel("Energy E(z)", color=_WHITE)
    ax2.set_ylabel("Count", color=_WHITE)
    ax2.set_title("Energy Level Histogram", color=_WHITE)
    ax2.tick_params(colors=_WHITE)
    ax2.legend(facecolor=_BORDER, labelcolor=_WHITE)
    for spine in ax2.spines.values():
        spine.set_edgecolor(_BORDER)

    plt.suptitle(title, fontsize=14, fontweight="bold", color=_WHITE, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "xai_energy_landscape.png")
    plt.savefig(path, dpi=250, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    if verbose:
        print("[Viz/XAI] Energy landscape saved -> {}".format(path))


# =============================================================================
# 6. XAI — INFORMATION DECISION BOUNDARY (JKL + Shannon Entropy)
# =============================================================================

def plot_information_decision_boundary(
    z_embeddings: np.ndarray,
    jkl_values: np.ndarray,
    entropies: np.ndarray,
    verbose: bool = True,
) -> None:
    """
    Plot the Information Theory Decision Boundaries in the JEPA latent space.

    Left panel  : JKL Divergence from uniform prior (confidence of prediction)
    Right panel : Shannon Entropy of latent state (uncertainty)

    Decision boundary = locus where JKL is minimal (model is undecided).
    High JKL = high-information transition (confident state change).

    Directly implements the Aug-2020 note:
        'Shannon: Uncertainty + KL: Entropy. Forward let q(x) known => p(x) log p(x)/q(x)'

    Parameters
    ----------
    z_embeddings : np.ndarray  [N, D]   JEPA latent vectors
    jkl_values   : np.ndarray  [N]      JKL divergence from prior for each z
    entropies    : np.ndarray  [N]      Shannon entropy of each latent state
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("[Viz/XAI] scikit-learn not installed — skipping decision boundary.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pca  = PCA(n_components=2)
    z_2d = pca.fit_transform(z_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(_DARK_BG)

    panel_data = [
        (jkl_values, "JKL Divergence (from Uniform Prior)", "RdYlGn"),
        (entropies,  "Shannon Entropy H(z)",                "YlOrBr_r"),
    ]
    for ax, (vals, label, cmap) in zip(axes, panel_data):
        ax.set_facecolor(_PANEL_BG)
        sc = ax.scatter(
            z_2d[:, 0], z_2d[:, 1],
            c=vals, cmap=cmap, s=60, alpha=0.85,
            edgecolors=_WHITE, linewidths=0.3,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(label, color=_WHITE)
        cbar.ax.yaxis.set_tick_params(color=_WHITE)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_WHITE)
        ax.set_xlabel("PC1", color=_WHITE)
        ax.set_ylabel("PC2", color=_WHITE)
        ax.set_title(label, color=_WHITE)
        ax.tick_params(colors=_WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)

    plt.suptitle(
        "Information Theory Decision Boundaries (JEPA Latent Space)",
        fontsize=13, fontweight="bold", color=_WHITE, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "xai_information_boundary.png")
    plt.savefig(path, dpi=250, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    if verbose:
        print("[Viz/XAI] Information boundary saved -> {}".format(path))


# =============================================================================
# 7. XAI — PARTITION FUNCTION EVOLUTION (Path Integral Trace)
# =============================================================================

def plot_partition_function_evolution(
    energy_trajectory: np.ndarray,
    beta_values: np.ndarray | None = None,
    step_labels: list[str] | None = None,
    verbose: bool = True,
) -> None:
    """
    Plot how Partition Function Z(beta) evolves along a JEPA path integral rollout.

    From Nov-2022 notes (property of density matrix convolution):
        Z(beta) = ∫ dx0 dx1 ... dx_{N-1} p(x0, x1, beta/N) ...
    A rising Z = more accessible low-energy states (hot / uncertain).
    A falling Z = model freezes into stable attractor (cold / confident).

    Parameters
    ----------
    energy_trajectory : np.ndarray  [T, B] or [T]
        Energy values at each rollout step.
    beta_values       : np.ndarray  [T]  optional per-step inverse temperature
    step_labels       : list[str]   optional x-axis tick labels
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if energy_trajectory.ndim == 2:
        mean_energy = energy_trajectory.mean(axis=1)
        std_energy  = energy_trajectory.std(axis=1)
    else:
        mean_energy = energy_trajectory
        std_energy  = np.zeros_like(mean_energy)

    T        = len(mean_energy)
    steps    = np.arange(T)
    beta_arr = beta_values if beta_values is not None else np.ones(T)
    Z_approx = np.exp(-beta_arr * mean_energy)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(_DARK_BG)

    # ── Left: Energy trajectory ─────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(_PANEL_BG)
    ax.plot(steps, mean_energy, color=_BLUE, linewidth=2.5, label="Mean E(z)")
    ax.fill_between(
        steps,
        mean_energy - std_energy,
        mean_energy + std_energy,
        color=_BLUE, alpha=0.2, label="+/-1 sigma",
    )
    ax.set_xlabel("Rollout Step (t)", color=_WHITE)
    ax.set_ylabel("Energy E(z)", color=_WHITE)
    ax.set_title("Energy Over Path Integral Rollout", color=_WHITE)
    ax.tick_params(colors=_WHITE)
    ax.legend(facecolor=_BORDER, labelcolor=_WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    if step_labels:
        ax.set_xticks(steps)
        ax.set_xticklabels(step_labels, rotation=45, color=_WHITE)

    # ── Right: Partition function Z(beta) ───────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(_PANEL_BG)
    ax2.plot(steps, Z_approx, color=_GREEN, linewidth=2.5, label="Z(beta) approx")
    ax2.set_xlabel("Rollout Step (t)", color=_WHITE)
    ax2.set_ylabel("Z(beta)", color=_WHITE)
    ax2.set_title("Partition Function Z(beta) Evolution", color=_WHITE)
    ax2.tick_params(colors=_WHITE)
    ax2.legend(facecolor=_BORDER, labelcolor=_WHITE)
    for spine in ax2.spines.values():
        spine.set_edgecolor(_BORDER)

    plt.suptitle(
        "Path Integral Rollout: Energy and Partition Function",
        fontsize=13, fontweight="bold", color=_WHITE, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "xai_partition_function.png")
    plt.savefig(path, dpi=250, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    if verbose:
        print("[Viz/XAI] Partition function evolution saved -> {}".format(path))
