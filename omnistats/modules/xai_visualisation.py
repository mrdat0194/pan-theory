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
        print("[Viz/XAI] Partition function evolution saved -> {}\n".format(path))


# =============================================================================
# 8. XAI — TCAV: Concept Probing for MatrAIx Persona Concepts
# =============================================================================

def plot_tcav_concept_probing(
    model_encode_fn,
    concept_examples: dict,
    non_concept_examples,
    query_examples,
    concept_names: list | None = None,
    title: str = "TCAV: Persona Concept Probing (MUMO-JEPA)",
    verbose: bool = True,
) -> dict:
    """
    Testing with Concept Activation Vectors (TCAV) for MUMO-JEPA.

    Probes the multimodal latent space to find which human-interpretable
    persona concepts (e.g. 'High Income', 'Risk Averse', 'Tech-Savvy')
    are directionally embedded in the model's CLS representation.

    Algorithm
    ---------
    1. For each concept C:
       a. Encode concept_examples[C] and non_concept_examples via model_encode_fn.
       b. Fit a linear binary classifier (CAV) to separate concept from non-concept
          embeddings. The CAV direction is the normal vector of the decision boundary.
       c. Compute TCAV score = fraction of query examples for which the directional
          derivative of the model output w.r.t. the CAV direction is positive.

    Parameters
    ----------
    model_encode_fn    : Callable  Takes (mod_a, mod_b) tensors, returns [B, D] embeddings.
    concept_examples   : dict      Maps concept_name -> (mod_a, mod_b) tensors.
    non_concept_examples: tuple    (mod_a, mod_b) tensors for non-concept "random" examples.
    query_examples     : tuple     (mod_a, mod_b) tensors for the target population.
    concept_names      : list[str] Labels for x-axis (defaults to concept_examples keys).
    title              : str
    verbose            : bool

    Returns
    -------
    dict mapping concept_name -> TCAV score (0.0–1.0).
    """
    try:
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[Viz/XAI/TCAV] scikit-learn or torch not installed — skipping TCAV.")
        return {}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Encode non-concept baseline
    with torch.no_grad():
        non_emb = model_encode_fn(*non_concept_examples).cpu().numpy()   # [N_neg, D]
        query_emb = model_encode_fn(*query_examples).cpu().numpy()        # [N_q, D]

    tcav_scores = {}
    names = concept_names or list(concept_examples.keys())

    for name, (cmod_a, cmod_b) in concept_examples.items():
        with torch.no_grad():
            concept_emb = model_encode_fn(cmod_a, cmod_b).cpu().numpy()  # [N_pos, D]

        # Combine positives and negatives for CAV training
        X = np.concatenate([concept_emb, non_emb], axis=0)
        y = np.concatenate([
            np.ones(len(concept_emb)),
            np.zeros(len(non_emb)),
        ])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        query_scaled = scaler.transform(query_emb)

        # Fit linear classifier — CAV direction is its weight vector
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_scaled, y)
        cav = clf.coef_[0]  # [D]

        # TCAV score: fraction of query examples with positive directional projection
        projections = query_scaled @ cav   # [N_q]
        tcav_score = float((projections > 0).mean())
        tcav_scores[name] = tcav_score

        if verbose:
            print(f"[Viz/XAI/TCAV] '{name}': TCAV score = {tcav_score:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(tcav_scores)), 5))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    sorted_items = sorted(tcav_scores.items(), key=lambda x: x[1], reverse=True)
    bars_x = [k for k, _ in sorted_items]
    bars_h = [v for _, v in sorted_items]
    colors = [_GREEN if h > 0.5 else _RED for h in bars_h]

    ax.bar(bars_x, bars_h, color=colors, edgecolor=_DARK_BG, alpha=0.9)
    ax.axhline(0.5, color=_WHITE, linestyle="--", linewidth=1.5, label="Random (0.5)")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("TCAV Score", color=_WHITE)
    ax.set_xlabel("MatrAIx Persona Concept", color=_WHITE)
    ax.set_title(title, color=_WHITE)
    ax.tick_params(colors=_WHITE, axis="both")
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color=_WHITE)
    ax.legend(facecolor=_BORDER, labelcolor=_WHITE)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "xai_tcav_concept_probing.png")
    plt.savefig(path, dpi=250, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    if verbose:
        print("[Viz/XAI] TCAV concept probing saved -> {}".format(path))

    return tcav_scores


# =============================================================================
# 9. XAI — Vector-Target Attribution for MUMO Modalities
# =============================================================================

def plot_vector_target_attribution(
    model_encode_fn,
    mod_a_examples,
    mod_b_examples,
    target_col: np.ndarray,
    modality_names: list | None = None,
    title: str = "Vector-Target Attribution: MUMO Modality Contributions",
    verbose: bool = True,
) -> dict:
    """
    Vector-Target Attribution for MUMO-JEPA Multimodal Embeddings.

    Measures how much each MUMO modality's sub-embedding contributes
    to predicting the target outcome (e.g., simulated A/B metric lift).

    Method
    ------
    For each modality M:
    1. Encode the full population using model_encode_fn (both modalities → CLS).
    2. Zero-out / ablate modality M by replacing its input with zero tensors.
    3. Re-encode. The drop in alignment with the target direction is the
       attribution score of modality M.

    This is a model-agnostic ablation-based attribution technique suited for
    the MUMO fusion architecture, where modalities interact through the
    CrossModalFusionLayer before being pruned.

    Parameters
    ----------
    model_encode_fn : Callable (mod_a, mod_b) -> [B, D] CLS embedding.
    mod_a_examples  : torch.Tensor  [B, 1, H, W]  (Modality A: Psychographic).
    mod_b_examples  : torch.Tensor  [B, C, T]     (Modality B: Demographic/AB).
    target_col      : np.ndarray    [B]  Target outcome (e.g., AB_METRIC_COL values).
    modality_names  : list[str]     Labels for each modality.
    title           : str
    verbose         : bool

    Returns
    -------
    dict mapping modality_name -> attribution_score (0.0–1.0, higher = more important).
    """
    try:
        import torch
    except ImportError:
        print("[Viz/XAI/Attribution] torch not installed — skipping.")
        return {}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = modality_names or ["Psychographic (Mod A)", "Demographic+Metric (Mod B)"]

    target = torch.tensor(target_col, dtype=torch.float32)

    # Full encoding
    with torch.no_grad():
        full_emb = model_encode_fn(mod_a_examples, mod_b_examples)   # [B, D]

    # Compute correlation of full CLS with target direction
    def _corr_with_target(emb: "torch.Tensor") -> float:
        # Project embedding onto the direction most correlated with target
        emb_np = emb.cpu().numpy()
        tgt_np = target.cpu().numpy()
        # Simple cosine alignment of first PC with target
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X = sc.fit_transform(emb_np)
        reg = Ridge(alpha=1.0).fit(X, tgt_np)
        preds = reg.predict(X)
        corr = float(np.corrcoef(preds, tgt_np)[0, 1])
        return max(corr, 0.0)

    full_score = _corr_with_target(full_emb)

    attributions = {}
    zero_a = torch.zeros_like(mod_a_examples)
    zero_b = torch.zeros_like(mod_b_examples)

    # Ablate Modality A (zero-out psychographic)
    with torch.no_grad():
        ablated_a_emb = model_encode_fn(zero_a, mod_b_examples)
    score_no_a = _corr_with_target(ablated_a_emb)
    attr_a = max(full_score - score_no_a, 0.0)

    # Ablate Modality B (zero-out demographics)
    with torch.no_grad():
        ablated_b_emb = model_encode_fn(mod_a_examples, zero_b)
    score_no_b = _corr_with_target(ablated_b_emb)
    attr_b = max(full_score - score_no_b, 0.0)

    total = attr_a + attr_b + 1e-9
    attributions[names[0]] = attr_a / total
    attributions[names[1]] = attr_b / total

    if verbose:
        for n_, v_ in attributions.items():
            print(f"[Viz/XAI/Attribution] '{n_}': {v_:.4f}")

    # ── Pie Chart ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(_DARK_BG)

    # Left: Pie chart
    ax = axes[0]
    ax.set_facecolor(_PANEL_BG)
    pie_vals = list(attributions.values())
    pie_labs = [f"{n_}\n{v_:.1%}" for n_, v_ in attributions.items()]
    colors_pie = [_BLUE, _GREEN, _RED, "#c792ea", "#f78c6c"][:len(pie_vals)]
    wedges, texts = ax.pie(
        pie_vals, labels=pie_labs, colors=colors_pie[:len(pie_vals)],
        startangle=140, textprops={"color": _WHITE, "fontsize": 10},
        wedgeprops={"edgecolor": _DARK_BG, "linewidth": 2},
    )
    ax.set_title("Modality Attribution (Ablation)", color=_WHITE, fontsize=12)

    # Right: Bar chart
    ax2 = axes[1]
    ax2.set_facecolor(_PANEL_BG)
    x_pos = np.arange(len(attributions))
    bar_vals = list(attributions.values())
    ax2.bar(x_pos, bar_vals, color=colors_pie[:len(bar_vals)],
            edgecolor=_DARK_BG, alpha=0.9, width=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(list(attributions.keys()), color=_WHITE, rotation=15, ha="right")
    ax2.set_ylabel("Normalised Attribution Score", color=_WHITE)
    ax2.set_ylim(0.0, 1.0)
    ax2.tick_params(colors=_WHITE)
    for spine in ax2.spines.values():
        spine.set_edgecolor(_BORDER)

    plt.suptitle(title, fontsize=13, fontweight="bold", color=_WHITE, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "xai_vector_target_attribution.png")
    plt.savefig(path, dpi=250, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close()
    if verbose:
        print("[Viz/XAI] Vector-target attribution saved -> {}".format(path))

    return attributions
