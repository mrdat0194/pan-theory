"""
omnistats/modules/jepa_bridge.py
---------------------------------
Bridge between OmniStats APA CSV outputs and the EB-JEPA world model.

Responsibilities
----------------
1. load_state_context() — reads lpa_profiles.csv + cuped_variance_reduction.csv
   and returns a float32 torch Tensor representing the current user-segment state.

2. APADecoder — a lightweight MLP that maps a JEPA latent vector z →
   (ATT_hat, Bayesian_Risk_hat).  Trained jointly with the JEPA predictor
   via train_apa_decoder() so that the latent space geometrically encodes
   causal structure.

Tensor conventions
------------------
All tensors have shape compatible with eb_jepa's planner interface:
  state_context: [1, D_state, 1, 1, 1]  (batch=1, D_state features, T=H=W=1)
  latent:        [B, D_latent, T, 1, 1]  (from JEPA encoder/predictor)

Why 5-D?
  eb_jepa's encoder and predictor operate on [B, C, T, H, W] tensors
  (batch, channels, time, spatial height, spatial width).
  For tabular / non-image data we set H=W=1 (degenerate spatial axes)
  so that the existing planner unroll() calls work without modification.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── resolve omnistats base dir so we can import config ───────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]   # omnistats/
sys.path.insert(0, str(BASE_DIR))
from config import OUTPUT_DIR, AB_METRIC_COL, AB_GROUP_COL, DEMOGRAPHIC_COLS

# ── XAI: Information Theory module (Quantum-Inspired) ─────────────────────
try:
    from omnistats.modules.information_theory import compute_xai_metrics
    _XAI_AVAILABLE = True
except ImportError:
    _XAI_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# 1.  STATE CONTEXT LOADER
# ═════════════════════════════════════════════════════════════════════════════

_STATE_FEATURE_NAMES: list[str] = []   # populated on first call


def load_state_context(
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> Tuple[torch.Tensor, list[str]]:
    """
    Build the initial world-model state vector from OmniStats CSV outputs.

    Sources (in priority order):
      1. outputs/lpa_profiles.csv  — profile posterior probabilities +
                                     CUPED-adjusted metric.
      2. outputs/cuped_variance_reduction.csv — theta, pct reduction.
         (Used as a scalar feature indicating how much noise was cleaned.)

    Returns
    -------
    state : torch.Tensor  shape [1, D, 1, 1, 1]  (float32)
        D = len(feature_names)
    feature_names : list[str]
        Names of the D features in the state vector.
    """
    global _STATE_FEATURE_NAMES

    features: dict[str, float] = {}

    # ── 1a. LPA profiles ────────────────────────────────────────────────────
    lpa_path = os.path.join(OUTPUT_DIR, "lpa_profiles.csv")
    if os.path.exists(lpa_path):
        lpa_df = pd.read_csv(lpa_path)

        # Profile posterior probabilities (profile_prob_1, profile_prob_2, ...)
        prob_cols = [c for c in lpa_df.columns if c.startswith("profile_prob_")]
        for col in prob_cols:
            features[col] = float(lpa_df[col].mean())

        # Profile assignment distribution (fraction of users in each profile)
        if "profile" in lpa_df.columns:
            n_profiles = lpa_df["profile"].nunique()
            for k in range(1, n_profiles + 1):
                features[f"profile_frac_{k}"] = float(
                    (lpa_df["profile"] == k).mean()
                )

        # Mean CUPED-adjusted metric
        cuped_col = f"{AB_METRIC_COL}_cuped"
        if cuped_col in lpa_df.columns:
            features["cuped_mean"] = float(lpa_df[cuped_col].mean())
            features["cuped_std"]  = float(lpa_df[cuped_col].std(ddof=1))
        elif AB_METRIC_COL in lpa_df.columns:
            features["metric_mean"] = float(lpa_df[AB_METRIC_COL].mean())
            features["metric_std"]  = float(lpa_df[AB_METRIC_COL].std(ddof=1))

        # Demographic mix (mean of one-hot encoded demographics)
        demo_cols = [c for c in DEMOGRAPHIC_COLS if c in lpa_df.columns]
        if demo_cols:
            dummy_df = pd.get_dummies(
                lpa_df[demo_cols].fillna("_NA_"), drop_first=True, dtype=float
            )
            for col in dummy_df.columns:
                features[f"demo_{col}_rate"] = float(dummy_df[col].mean())

        if verbose:
            print(f"[JEPABridge] Loaded LPA state from {lpa_path} "
                  f"({len(lpa_df)} rows, {len(features)} features so far)")
    else:
        warnings.warn(
            f"[JEPABridge] {lpa_path} not found. "
            "Run main.py Stage 1 first.  Using zero state.", stacklevel=2
        )
        features["cuped_mean"] = 0.0
        features["cuped_std"]  = 1.0

    # ── 1b. CUPED variance reduction scalar ──────────────────────────────────
    cuped_csv = os.path.join(OUTPUT_DIR, "cuped_variance_reduction.csv")
    if os.path.exists(cuped_csv):
        cuped_meta = pd.read_csv(cuped_csv)
        if "theta" in cuped_meta.columns:
            features["cuped_theta"] = float(cuped_meta["theta"].iloc[-1])
        if "variance_reduction_pct" in cuped_meta.columns:
            features["cuped_var_red_pct"] = float(
                cuped_meta["variance_reduction_pct"].iloc[-1]
            )

    # ── 1c. Bayesian A/B prior (from bayesian_ab_results.csv) ────────────────
    bayes_csv = os.path.join(OUTPUT_DIR, "bayesian_ab_results.csv")
    if os.path.exists(bayes_csv):
        bayes_df = pd.read_csv(bayes_csv)
        if "prob_b_gt_a" in bayes_df.columns:
            features["bayesian_prob_b_gt_a"] = float(
                bayes_df["prob_b_gt_a"].iloc[-1]
            )
        if "expected_loss" in bayes_df.columns:
            features["bayesian_expected_loss"] = float(
                bayes_df["expected_loss"].iloc[-1]
            )

    # ── 1d. Historical ATT from causal_results.csv ───────────────────────────
    causal_csv = os.path.join(OUTPUT_DIR, "causal_results.csv")
    if os.path.exists(causal_csv):
        cr = pd.read_csv(causal_csv)
        if "estimate" in cr.columns and len(cr) > 0:
            features["historical_att_mean"] = float(cr["estimate"].mean())
            features["historical_att_std"]  = float(cr["estimate"].std(ddof=1))
        if "p_value" in cr.columns and len(cr) > 0:
            features["historical_p_value_min"] = float(cr["p_value"].min())

    # ── Assemble into tensor ─────────────────────────────────────────────────
    feature_names = list(features.keys())
    _STATE_FEATURE_NAMES = feature_names
    values = np.array([features[k] for k in feature_names], dtype=np.float32)

    # Normalise to zero-mean unit-variance using the values themselves
    # (single context → no separate mean/std available).
    # This is a no-op when values are already normalised (profile probs, rates).
    std_safe = np.where(np.abs(values) > 1e-8, np.abs(values), 1.0)
    values_norm = values / std_safe   # rough z-score without a global μ

    # [1, D, 1, 1, 1] — compatible with eb_jepa's [B, C, T, H, W] convention
    state = torch.tensor(values_norm, dtype=torch.float32, device=device)
    state = state.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

    if verbose:
        print(f"[JEPABridge] State tensor shape: {state.shape}  "
              f"(D={len(feature_names)} features)")

    return state, feature_names


# ═════════════════════════════════════════════════════════════════════════════
# 2.  APA DECODER  (latent z → ATT_hat, Risk_hat)
# ═════════════════════════════════════════════════════════════════════════════

class APADecoder(nn.Module):
    """
    Quantum-Inspired APA Decoder: JEPA latent vector → APA scalars + XAI metrics.

    Extends the standard (ATT, Risk) decoder with two new output heads:
      - energy_hat  : Boltzmann Energy E_n of the latent state
      - beta_hat    : Inverse Temperature β (model confidence / precision)

    These outputs directly correspond to the thermodynamic quantities
    from your notes (π(x,n) ∝ e^{-βE_n} ψ_n(x)) and enable full XAI
    reporting via the information_theory module.

    Inputs
    ------
    z : torch.Tensor  [B, D_latent]   (flattened latent state)

    Outputs
    -------
    dict with:
        att_hat    : [B]   predicted Average Treatment Effect
        risk_hat   : [B]   predicted Bayesian Expected Loss (≥ 0)
        energy_hat : [B]   predicted Boltzmann Energy of latent state
        beta_hat   : [B]   predicted Inverse Temperature (≥ 0)
        entropy    : [B]   Shannon Entropy of latent activations
    """

    def __init__(
        self,
        d_latent: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        beta_init: float = 1.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Original APA heads
        self.att_head  = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),          # risk ≥ 0
        )
        # XAI: Quantum-Inspired heads
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),          # energy ≥ 0
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),          # β ≥ 0
        )

    def forward(
        self,
        z: torch.Tensor,
        reference_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            z: [B, D_latent]  — flattened latent (see _flatten_latent helper)
            reference_embeddings: [N, D_latent] optional concept anchors for
                Bayesian Inverse Scoring (JKL from prior)
        Returns:
            dict with att_hat, risk_hat, energy_hat, beta_hat, entropy, [jkl]
        """
        h = self.net(z)
        energy_hat = self.energy_head(h).squeeze(-1)   # [B]
        beta_hat   = self.beta_head(h).squeeze(-1)     # [B]

        # Shannon Entropy of latent activations (XAI: uncertainty of state)
        latent_probs = torch.softmax(z, dim=-1)                     # [B, D]
        p_clamped = latent_probs.clamp(min=1e-10)
        entropy = -(p_clamped * p_clamped.log() / torch.log(torch.tensor(2.0))).sum(-1)  # [B]

        out = {
            "att_hat":    self.att_head(h).squeeze(-1),   # [B]
            "risk_hat":   self.risk_head(h).squeeze(-1),  # [B]
            "energy_hat": energy_hat,                      # [B]
            "beta_hat":   beta_hat,                        # [B]
            "entropy":    entropy,                         # [B]
        }

        # Bayesian Inverse Score (JKL from uniform prior)
        if reference_embeddings is not None and _XAI_AVAILABLE:
            xai = compute_xai_metrics(
                z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),   # fake [B,D,1,1,1]
                reference_embeddings=reference_embeddings,
            )
            out["jkl_from_prior"] = xai["jkl_from_prior"]  # [B]
            out["posterior"]      = xai["posterior"]         # [B, N]

        return out


def _flatten_latent(z: torch.Tensor) -> torch.Tensor:
    """
    Flatten a [B, D, T, H, W] latent tensor to [B, D*T*H*W].
    For our tabular case T=H=W=1 so this is just [B, D].
    """
    B = z.shape[0]
    return z.reshape(B, -1)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  JOINT TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train_apa_decoder(
    jepa_model: nn.Module,
    decoder: APADecoder,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> APADecoder:
    """
    Joint training: fine-tune JEPA predictor AND APADecoder simultaneously
    using historical OmniStats causal_results.csv as ground-truth labels.

    The loss is:
        L_total = L_ATT + w_risk * L_Risk

    where L_ATT  = MSE(att_hat, true_ATT)
          L_Risk = MSE(risk_hat, true_Expected_Loss)

    This forces the JEPA latent space to geometrically align with APA
    causal treatment effects.

    Parameters
    ----------
    jepa_model : nn.Module
        The JEPA model (JEPAbase or JEPA).  Must have an .encoder attribute.
    decoder : APADecoder
        The APA decoder to train (and update in-place).
    n_epochs : int
        Training epochs (iterations over the historical dataset).
    lr : float
        Learning rate for the joint AdamW optimizer.
    device : torch.device | str
    verbose : bool

    Returns
    -------
    decoder : APADecoder  (updated in-place, also returned for chaining)
    """
    # ── Load historical APA labels ────────────────────────────────────────────
    causal_csv = os.path.join(OUTPUT_DIR, "causal_results.csv")
    bayes_csv  = os.path.join(OUTPUT_DIR, "bayesian_ab_results.csv")

    if not os.path.exists(causal_csv):
        warnings.warn(
            f"[JEPABridge] {causal_csv} not found — skipping decoder training. "
            "Run main.py first.", stacklevel=2
        )
        return decoder

    cr = pd.read_csv(causal_csv).dropna(subset=["estimate"])
    if len(cr) == 0:
        warnings.warn("[JEPABridge] causal_results.csv is empty.", stacklevel=2)
        return decoder

    # Ground-truth ATT values (one per causal estimator row)
    att_labels   = torch.tensor(cr["estimate"].values, dtype=torch.float32, device=device)

    # Ground-truth Expected Loss (from Bayesian A/B; broadcast to match ATT rows)
    risk_labels = torch.zeros_like(att_labels)
    if os.path.exists(bayes_csv):
        bayes_df = pd.read_csv(bayes_csv)
        if "expected_loss" in bayes_df.columns and len(bayes_df) > 0:
            risk_val = float(bayes_df["expected_loss"].iloc[-1])
            risk_labels = torch.full_like(att_labels, risk_val)

    # ── Build synthetic latent states ─────────────────────────────────────────
    # For each historical ATT row we create a small perturbation of the current
    # state context so the decoder sees variation.  In production, each row
    # would correspond to a distinct historical experiment's encoded state.
    state_ctx, feat_names = load_state_context(device=device, verbose=False)
    D_state = state_ctx.shape[1]   # number of state features

    n_samples = len(att_labels)
    noise = 0.05 * torch.randn(n_samples, D_state, 1, 1, 1, device=device)
    states = state_ctx.expand(n_samples, -1, -1, -1, -1) + noise  # [N, D, 1, 1, 1]

    # ── Encode states ─────────────────────────────────────────────────────────
    # We need the JEPA encoder to handle our tabular 5-D tensors.
    # If the encoder is a standard image CNN it will fail here; in that case
    # the JEPATabularEncoder (see jepa_bridge_encoder.py) must be used instead.
    try:
        with torch.no_grad():
            latents = jepa_model.encoder(states)        # [N, D_latent, 1, 1, 1]
    except Exception as exc:
        warnings.warn(
            f"[JEPABridge] JEPA encoder raised {exc}. "
            "Proceeding with raw state features as latent proxy.", stacklevel=2
        )
        latents = states

    # ── Joint optimizer ───────────────────────────────────────────────────────
    params = list(decoder.parameters()) + list(jepa_model.encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    decoder.train()
    jepa_model.train()

    w_risk = 0.3   # weight for risk loss term

    if verbose:
        print(f"\n[JEPABridge] Joint training: {n_epochs} epochs, "
              f"{n_samples} historical samples")

    for epoch in range(n_epochs):
        # Re-encode at each epoch so encoder gradients are live
        latents_live = jepa_model.encoder(states)   # [N, D_latent, 1, 1, 1]
        z_flat = _flatten_latent(latents_live)       # [N, D_latent]

        preds = decoder(z_flat)
        loss_att  = F.mse_loss(preds["att_hat"],  att_labels)
        loss_risk = F.mse_loss(preds["risk_hat"], risk_labels)
        loss_total = loss_att + w_risk * loss_risk

        optimizer.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(f"  epoch {epoch+1:4d}/{n_epochs}  "
                  f"L_total={loss_total.item():.4f}  "
                  f"L_ATT={loss_att.item():.4f}  "
                  f"L_Risk={loss_risk.item():.4f}")

    decoder.eval()
    jepa_model.eval()

    if verbose:
        print("[JEPABridge] Decoder training complete.")

    return decoder
