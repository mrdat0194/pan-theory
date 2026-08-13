"""
omnistats/modules/simulation/mumo_jepa_trainer.py
──────────────────────────────────────────────────
Phase V: MUMO-JEPA Fine-Tuner on Best Simulated Data.

Responsibilities
----------------
1. Import MuMoJEPAWrapper from MLModel/AIModel/model/mumo_wrapper.py.
2. Convert the winning MatrAIx simulated DataFrame into dual-modality tensors:
   - Modality A (Psychographic):  [B, 1, H, W]  (treated as a spectrogram-like 2D map)
   - Modality B (Demographic/AB): [B, C, T]      (treated as a 1D sequence)
3. Run a self-supervised fine-tuning pass using SIGReg loss.
4. Save the fine-tuned checkpoint to outputs/mumo_jepa_checkpoint.pt.
5. Return the trained model for downstream XAI (TCAV / Vector-Target Attribution).

Import Path
-----------
MUMO model is imported from:
    C:/Users/mrdat/PycharmProjects/pan-theory/MLModel/AIModel/model/mumo_wrapper.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── resolve omnistats root ────────────────────────────────────────────────────
_OMNI_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_OMNI_ROOT))

# ── resolve MLModel/AIModel for MUMO import ──────────────────────────────────
_AIMODEL_ROOT = Path("C:/Users/mrdat/PycharmProjects/pan-theory/MLModel/AIModel")
sys.path.insert(0, str(_AIMODEL_ROOT))

from config import OUTPUT_DIR

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "mumo_jepa_checkpoint.pt")

# ── Persona column groups for modality construction ───────────────────────────
_PSYCHO_COLS = [
    "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism", "risk_tolerance",
    "tech_savviness", "impulsivity",
]

_DEMO_COLS = ["age", "income", "education_level", "location_tier"]


# =============================================================================
# Public API
# =============================================================================

def build_mumo_model(
    embed_dim: int = 64,
    n_fusion_tokens: int = 8,
    n_layers: int = 2,
    n_heads: int = 4,
    verbose: bool = True,
) -> "MuMoJEPAWrapper":
    """
    Build a MuMoJEPAWrapper configured for tabular persona data.

    Modality A: Psychographic traits (8 features) → reshaped as [B, 1, 4, 2] patch-like input.
    Modality B: Demographic + AB metric (5 features) → treated as [B, 5, 1] 1D sequence.

    Returns
    -------
    MuMoJEPAWrapper instance (untrained or from checkpoint if available).
    """
    try:
        import torch
        from model.mumo_wrapper import MuMoJEPAWrapper, build_audio_mumo_jepa
        from model.ajepa_backbone import AudioPatchEmbed
        from model.jepa_backbone import SequenceStem

        if verbose:
            print("[MuMoTrainer] Importing MuMoJEPAWrapper from MLModel/AIModel ✓")

        # Modality A stem: wraps psychographic features as a patch-like embedding
        # Uses AudioPatchEmbed with patch_size=1 so each feature becomes a token
        stem_a = AudioPatchEmbed(
            in_chans=1,
            patch_size=(2, 2),
            embed_dim=embed_dim,
            n_freq_bins=4,
            n_time_frames=2,
        )

        # Modality B stem: wraps demographic features as 1D sequence
        stem_b = SequenceStem(
            in_channels=len(_DEMO_COLS) + 1,  # demographics + metric
            embed_dim=embed_dim,
        )

        model = MuMoJEPAWrapper(
            stem_a=stem_a,
            stem_b=stem_b,
            embed_dim=embed_dim,
            n_fusion_tokens=n_fusion_tokens,
            n_layers=n_layers,
            n_heads=n_heads,
            sigreg_lambda=1.0,
        )

        # Load checkpoint if it exists from a previous iteration
        if os.path.exists(CHECKPOINT_PATH):
            state = torch.load(CHECKPOINT_PATH, map_location="cpu")
            model.load_state_dict(state)
            if verbose:
                print(f"[MuMoTrainer] Loaded checkpoint from {CHECKPOINT_PATH}")

        return model

    except ImportError as e:
        raise ImportError(
            f"[MuMoTrainer] Could not import MuMoJEPAWrapper. "
            f"Ensure MLModel/AIModel is accessible. Error: {e}"
        )


def prepare_tensors(
    df: pd.DataFrame,
    device: str = "cpu",
) -> tuple:
    """
    Convert the winning simulated DataFrame into dual-modality tensors.

    Parameters
    ----------
    df     : Winning feature DataFrame from mumo_ab_tester.find_best_feature().
    device : Torch device string.

    Returns
    -------
    (mod_a, mod_b) tensors ready for MuMoJEPAWrapper.forward().
    """
    import torch

    # Fill missing columns with median
    for col in _PSYCHO_COLS + _DEMO_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(df[col].median())

    # Normalise income (log scale)
    if "income" in df.columns:
        df["income"] = np.log1p(df["income"]) / 12.0

    n = len(df)

    # Modality A: psychographic traits → [B, 1, 4, 2] (patch-like)
    psycho = df[_PSYCHO_COLS].to_numpy(dtype=np.float32)  # [N, 8]
    mod_a = torch.tensor(psycho).view(n, 1, 4, 2).to(device)

    # Modality B: demographics + metric → [B, C, 1]
    from config import AB_METRIC_COL
    metric_col = AB_METRIC_COL or "metric"
    demo_cols_present = [c for c in _DEMO_COLS if c in df.columns]
    if metric_col in df.columns:
        all_demo = df[demo_cols_present + [metric_col]].to_numpy(dtype=np.float32)
    else:
        all_demo = df[demo_cols_present].to_numpy(dtype=np.float32)

    mod_b = torch.tensor(all_demo).unsqueeze(-1).to(device)  # [B, C, 1]

    return mod_a, mod_b


def fine_tune(
    model: "MuMoJEPAWrapper",
    best_df: pd.DataFrame,
    n_epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> "MuMoJEPAWrapper":
    """
    Fine-tune MuMoJEPAWrapper on the best simulated experiment data.

    Training objective: SIGReg self-supervised regularization on the
    joint multimodal CLS embedding. This pushes the model to learn
    the intrinsic distribution of winning persona behaviors.

    Parameters
    ----------
    model      : MuMoJEPAWrapper (from build_mumo_model).
    best_df    : Winning simulated DataFrame.
    n_epochs   : int   Number of training epochs.
    batch_size : int   Mini-batch size.
    lr         : float Learning rate.
    device     : str   'cpu' or 'cuda'.
    verbose    : bool

    Returns
    -------
    Fine-tuned MuMoJEPAWrapper model.
    """
    import torch
    import torch.optim as optim

    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    mod_a_full, mod_b_full = prepare_tensors(best_df, device=device)
    n = mod_a_full.shape[0]

    if verbose:
        print(f"\n[MuMoTrainer] Fine-tuning on {n} winning personas | "
              f"epochs={n_epochs}, batch={batch_size}, lr={lr}")

    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_a = mod_a_full[idx]
            batch_b = mod_b_full[idx]

            optimizer.zero_grad()
            loss, (pred_loss, sigreg_loss) = model(batch_a, batch_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if verbose:
            print(f"  Epoch {epoch + 1}/{n_epochs} | Loss={avg_loss:.5f}")

    # Save checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import torch
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    if verbose:
        print(f"[MuMoTrainer] Checkpoint saved -> {CHECKPOINT_PATH}")

    return model
