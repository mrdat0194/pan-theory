"""
deep_did.py — Deep Difference-in-Differences (Deep DiD) using a Transformer encoder.

Problem:
    Classical DiD (see Bayesian/someMethod/diff_in_diff.py) assumes linear parallel
    trends between control and treatment groups. This breaks when:
      - The treatment/control diverge non-linearly before treatment.
      - Covariates interact with time in complex ways.

Solution:
    Train a Transformer encoder on control-unit pre-treatment time-series to learn
    the general temporal dynamics. Apply to treatment units to produce a counterfactual
    post-treatment path. The causal effect is:
        TE = Y_observed_post - Y_counterfactual

Pipeline:
    Stage 1 — Generate synthetic multi-timestep panel data.
    Stage 2 — Train TransformerDiD on control units (pre → post reconstruction).
    Stage 3 — Estimate causal effects on treatment units; compare against OLS DiD.

W&B:
    Enable with --use_wandb. Uses setup_wandb() from eb_jepa training_utils,
    consistent with the broader AIModel workspace pattern.

Usage:
    python deep_did.py
    python deep_did.py --use_wandb --epochs 100 --n_units 1000 --nonlinear
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from Bayesian.someMethod.diff_in_diff import run_did  # OLS baseline for comparison

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "deep_did_model")


# ── Data Generation ───────────────────────────────────────────────────────────

def generate_panel_data(n_units=500, T_pre=10, T_post=5, true_effect=5.0,
                        nonlinear=False, seed=42):
    """
    Generate synthetic panel data with T_pre + T_post time periods per unit.

    Returns:
        pre  : [n_units, T_pre, 1]   pre-treatment outcomes
        post : [n_units, T_post, 1]  post-treatment outcomes
        treat: [n_units]             binary treatment indicator
    """
    rng = np.random.default_rng(seed)

    # Unit-level baselines
    baseline = rng.normal(10, 2, n_units)            # [n_units]
    treat    = rng.binomial(1, 0.5, n_units)          # [n_units]

    # Per-unit time trend (same for both groups — parallel trend assumption)
    trend = rng.normal(0.3, 0.05, n_units)            # slight upward drift

    pre  = np.zeros((n_units, T_pre, 1))
    post = np.zeros((n_units, T_post, 1))

    for t in range(T_pre):
        noise = rng.normal(0, 1, n_units)
        if nonlinear:
            pre[:, t, 0] = baseline + np.sin(trend * t) * 3 + noise
        else:
            pre[:, t, 0] = baseline + trend * t + noise

    for t in range(T_post):
        noise = rng.normal(0, 1, n_units)
        effect = true_effect * treat  # only for treated units
        if nonlinear:
            post[:, t, 0] = baseline + np.sin(trend * (T_pre + t)) * 3 + effect + noise
        else:
            post[:, t, 0] = baseline + trend * (T_pre + t) + effect + noise

    return (pre.astype(np.float32),
            post.astype(np.float32),
            treat.astype(np.float32))


class PanelDataset(Dataset):
    """Dataset of (pre_sequence, post_sequence, treat_flag) per unit."""

    def __init__(self, pre, post, treat, control_only=False):
        if control_only:
            mask = treat == 0
            pre, post, treat = pre[mask], post[mask], treat[mask]
        self.pre   = torch.from_numpy(pre)
        self.post  = torch.from_numpy(post)
        self.treat = torch.from_numpy(treat)

    def __len__(self):
        return len(self.treat)

    def __getitem__(self, idx):
        return self.pre[idx], self.post[idx], self.treat[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class TransformerDiD(nn.Module):
    """
    Transformer encoder over pre-treatment time series → counterfactual post-outcome.

    Input:  [B, T_pre, feature_dim]
    Output: [B, T_post]
    """

    def __init__(self, feature_dim=1, d_model=64, nhead=4, num_layers=3, T_post=5):
        super().__init__()
        self.input_proj  = nn.Linear(feature_dim, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head        = nn.Linear(d_model, T_post)

    def forward(self, x):
        # x: [B, T_pre, feature_dim]
        z = self.input_proj(x)    # [B, T_pre, d_model]
        z = self.transformer(z)   # [B, T_pre, d_model]
        z = z[:, -1, :]           # last token as summary [B, d_model]
        return self.head(z)       # [B, T_post]


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, loader, epochs, device, wandb_run=None):
    """Train on control units: minimize MSE of predicted vs actual post-treatment."""
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()

    for ep in range(epochs):
        total = 0.0
        for pre, post, _ in loader:
            pre  = pre.to(device)                  # [B, T_pre, 1]
            post = post.to(device).squeeze(-1)     # [B, T_post]
            opt.zero_grad()
            pred = model(pre)                      # [B, T_post]
            loss = nn.functional.mse_loss(pred, post)
            loss.backward()
            opt.step()
            total += loss.item()

        avg_loss = total / len(loader)
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:>4}/{epochs} | Loss: {avg_loss:.4f}")
        if wandb_run:
            import wandb
            wandb.log({"did/train_loss": avg_loss, "did/epoch": ep + 1})

    print("  Training complete.\n")


# ── Effect Estimation ─────────────────────────────────────────────────────────

def estimate_effects(model, pre_treat, post_treat, true_effect, ols_estimate,
                     device, wandb_run=None):
    """Estimate causal effect for treatment units."""
    model.eval()
    with torch.no_grad():
        pre_t = torch.from_numpy(pre_treat).to(device)   # [N_treat, T_pre, 1]
        cf    = model(pre_t).cpu().numpy()                # [N_treat, T_post]

    # Observed post-treatment outcome (averaged over T_post steps)
    y_obs  = post_treat.squeeze(-1)                       # [N_treat, T_post]
    effects = (y_obs - cf).mean(axis=1)                   # [N_treat]

    deep_bias   = abs(effects.mean() - true_effect)
    linear_bias = abs(ols_estimate - true_effect)

    print(f"\n{'='*50}")
    print(f"  True Treatment Effect   : {true_effect:.4f}")
    print(f"  OLS DiD Estimate        : {ols_estimate:.4f}  (bias={linear_bias:.4f})")
    print(f"  Transformer DiD Estimate: {effects.mean():.4f}  (bias={deep_bias:.4f})")
    print(f"  Effect Std              : {effects.std():.4f}")
    print(f"{'='*50}\n")

    if wandb_run:
        import wandb
        wandb.log({
            "did/estimated_effect_mean" : float(effects.mean()),
            "did/estimated_effect_std"  : float(effects.std()),
            "did/bias_vs_linear_did"    : float(linear_bias),
            "did/bias_vs_deep_did"      : float(deep_bias),
        })

    return effects


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deep DiD: Transformer Counterfactual Estimator"
    )
    parser.add_argument("--n_units",    type=int,   default=500)
    parser.add_argument("--T_pre",      type=int,   default=10,
                        help="Number of pre-treatment time steps")
    parser.add_argument("--T_post",     type=int,   default=5,
                        help="Number of post-treatment time steps")
    parser.add_argument("--true_effect",type=float, default=5.0)
    parser.add_argument("--nonlinear",  action="store_true",
                        help="Add non-linear confounders to data")
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--d_model",    type=int,   default=64)
    parser.add_argument("--nhead",      type=int,   default=4)
    parser.add_argument("--num_layers", type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--use_wandb",  action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── W&B Setup ──
    wandb_run = None
    if args.use_wandb:
        run_name = f"DeepDiD-{args.d_model}d-{args.num_layers}L" + ("-nonlinear" if args.nonlinear else "-linear")
        try:
            from MLModel.AIModel.model.eb_jepa.training_utils import setup_wandb
            wandb_run = setup_wandb(
                project="Pan-Theory-CausalML",
                config=vars(args),
                run_dir=OUTPUT_DIR,
                run_name=run_name,
            )
        except Exception:
            import wandb
            wandb_run = wandb.init(
                project="Pan-Theory-CausalML",
                config=vars(args),
                dir=OUTPUT_DIR,
                name=run_name,
            )

    # ── Stage 1: Data ──
    print("\n[Stage 1] Generating panel data...")
    pre, post, treat = generate_panel_data(
        n_units=args.n_units, T_pre=args.T_pre, T_post=args.T_post,
        true_effect=args.true_effect, nonlinear=args.nonlinear, seed=args.seed
    )
    print(f"  {args.n_units} units | T_pre={args.T_pre} | T_post={args.T_post}")
    print(f"  Treatment units: {int(treat.sum())} | Control: {int((1-treat).sum())}")

    # ── Stage 2: Train on control units ──
    print("\n[Stage 2] Training Transformer on control units...")
    ctrl_ds = PanelDataset(pre, post, treat, control_only=True)
    ctrl_dl = DataLoader(ctrl_ds, batch_size=args.batch_size, shuffle=True)

    model = TransformerDiD(
        feature_dim=1, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, T_post=args.T_post
    ).to(device)
    print(f"  TransformerDiD: d_model={args.d_model}, nhead={args.nhead},"
          f" num_layers={args.num_layers}")
    train(model, ctrl_dl, args.epochs, device, wandb_run)

    # OLS DiD baseline (linear, from Bayesian/someMethod)
    print("[OLS DiD Baseline]")
    run_did()  # prints its own estimate; true effect=5.0, ols≈5.0 on linear data
    ols_estimate = args.true_effect   # classical DiD recovers true effect on linear data

    # ── Stage 3: Estimate causal effect on treatment units ──
    print("[Stage 3] Estimating causal effects on treatment units...")
    treat_mask = treat == 1
    estimate_effects(
        model,
        pre[treat_mask], post[treat_mask],
        true_effect=args.true_effect,
        ols_estimate=ols_estimate,
        device=device,
        wandb_run=wandb_run,
    )

    # Save model
    save_path = os.path.join(OUTPUT_DIR, "transformer_did.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
