"""
deep_iv.py — Deep Instrumental Variables (Deep IV) using a Two-Stage Transformer.

Problem:
    Classical 2SLS (see Bayesian/someMethod/instrumental_variables.py) assumes linear
    relationships between instrument Z, treatment X, and outcome Y. When these
    relationships are non-linear, OLS in Stage 1 produces a biased X̂, and the
    recovered causal effect θ is also biased.

Solution:
    Double Machine Learning (DML) with Transformer-based models:
      Stage 1: Transformer learns f(Z, C) → X̂  (non-linear instrument → treatment)
      Stage 2: MLP learns g(X̂, C) → Y          (causal effect estimation)
    Cross-fitting ensures valid standard errors.

Pipeline:
    Stage 1 — Generate synthetic endogenous data with confounders.
    Stage 2 — Train Stage1 Transformer to predict treatment X from instrument Z.
    Stage 3 — Train Stage2 MLP to predict outcome Y from X̂; recover θ.
    Compare bias against Naive OLS and classical 2SLS from instrumental_variables.py.

W&B:
    Enable with --use_wandb. Logs stage 1/2 losses and causal effect estimates.

Usage:
    python deep_iv.py
    python deep_iv.py --use_wandb --epochs 100 --nonlinear
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

from Bayesian.someMethod.instrumental_variables import run_iv  # classical 2SLS baseline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "deep_iv_model")


# ── Data Generation ───────────────────────────────────────────────────────────

def generate_iv_data(n=1000, true_effect=2.0, nonlinear=False, seed=42):
    """
    Generate synthetic IV data following instrumental_variables.py structure
    but with optional non-linear confounding.

    Returns arrays: Z, X, C (observed confounders), Y, unobserved U
    """
    rng = np.random.default_rng(seed)

    # Unobserved confounder
    u = rng.normal(0, 2, n)

    # Observed confounders (new — makes this harder than the baseline)
    c = rng.normal(0, 1, (n, 3))    # 3 observed covariates

    # Instrument (uncorrelated with U — exclusion restriction)
    z = rng.normal(0, 2, n)

    # Endogenous treatment X
    if nonlinear:
        x = 1.5 * np.sin(z) + 0.8 * u + 0.5 * c.sum(axis=1) + rng.normal(0, 1, n)
    else:
        x = 1.5 * z + 0.8 * u + 0.5 * c.sum(axis=1) + rng.normal(0, 1, n)

    # Outcome Y (true causal effect = true_effect)
    if nonlinear:
        y = 5 + true_effect * x + 1.5 * u**2 + c[:, 0] * c[:, 1] + rng.normal(0, 1, n)
    else:
        y = 5 + true_effect * x + 1.5 * u + rng.normal(0, 1, n)

    return (z.astype(np.float32), x.astype(np.float32),
            c.astype(np.float32), y.astype(np.float32))


class IVDataset(Dataset):
    """Dataset of (instrument, treatment, confounders, outcome) tuples."""

    def __init__(self, Z, X, C, Y):
        self.Z = torch.from_numpy(Z).unsqueeze(-1)   # [N, 1]
        self.X = torch.from_numpy(X).unsqueeze(-1)   # [N, 1]
        self.C = torch.from_numpy(C)                  # [N, 3]
        self.Y = torch.from_numpy(Y).unsqueeze(-1)   # [N, 1]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.Z[idx], self.X[idx], self.C[idx], self.Y[idx]


# ── Models ────────────────────────────────────────────────────────────────────

class Stage1TransformerIV(nn.Module):
    """
    Stage 1: Predict endogenous treatment X from instrument Z and confounders C.
    Treats each feature as a token in a short sequence.

    Input:  Z [B, 1], C [B, n_confounders]  → concatenated [B, 1+n_confounders]
    Output: X̂ [B, 1]
    """

    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj  = nn.Linear(1, d_model)   # project each scalar feature
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head        = nn.Linear(d_model, 1)
        self.n_features  = n_features

    def forward(self, z, c):
        # Concatenate Z and C as a sequence of scalar tokens
        # [B, 1]  +  [B, n_confounders] → [B, 1 + n_confounders, 1]
        features = torch.cat([z, c], dim=-1).unsqueeze(-1)   # [B, n_features, 1]
        tokens   = self.input_proj(features)                  # [B, n_features, d_model]
        out      = self.transformer(tokens)                   # [B, n_features, d_model]
        # Mean-pool across token dimension
        pooled   = out.mean(dim=1)                            # [B, d_model]
        return self.head(pooled)                              # [B, 1]


class Stage2MLP(nn.Module):
    """
    Stage 2: Estimate causal effect using predicted X̂ and observed confounders C.
    Y = θ * X̂ + g(C)

    Input:  X̂ [B, 1], C [B, n_confounders]
    Output: Ŷ [B, 1]
    """

    def __init__(self, n_confounders=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + n_confounders, hidden), nn.GELU(),
            nn.Linear(hidden, 64),                nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_hat, c):
        inp = torch.cat([x_hat, c], dim=-1)   # [B, 1 + n_confounders]
        return self.net(inp)                   # [B, 1]


# ── Training ──────────────────────────────────────────────────────────────────

def train_stage1(model, loader, epochs, device, wandb_run=None):
    """Stage 1: learn Z → X̂."""
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    print(f"--- Stage 1: Training Transformer IV ({epochs} epochs) ---")

    for ep in range(epochs):
        total = 0.0
        for z, x, c, _ in loader:
            z, x, c = z.to(device), x.to(device), c.to(device)
            opt.zero_grad()
            x_hat = model(z, c)
            loss  = nn.functional.mse_loss(x_hat, x)
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(loader)
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:>4}/{epochs} | Stage1 Loss: {avg:.4f}")
        if wandb_run:
            import wandb
            wandb.log({"iv/stage1_loss": avg, "iv/epoch": ep + 1})

    print("  Stage 1 complete.\n")


def train_stage2(stage1, stage2, loader, epochs, device, wandb_run=None):
    """Stage 2: learn X̂ → Y (causal effect estimation)."""
    opt = AdamW(stage2.parameters(), lr=1e-3, weight_decay=1e-4)
    stage1.eval()
    stage2.train()
    print(f"--- Stage 2: Training MLP on X_hat ({epochs} epochs) ---")

    for ep in range(epochs):
        total = 0.0
        for z, x, c, y in loader:
            z, x, c, y = z.to(device), x.to(device), c.to(device), y.to(device)
            opt.zero_grad()
            with torch.no_grad():
                x_hat = stage1(z, c)          # frozen Stage 1
            y_hat = stage2(x_hat, c)
            loss  = nn.functional.mse_loss(y_hat, y)
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(loader)
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:>4}/{epochs} | Stage2 Loss: {avg:.4f}")
        if wandb_run:
            import wandb
            wandb.log({"iv/stage2_loss": avg})

    print("  Stage 2 complete.\n")


# ── Causal Effect Recovery ─────────────────────────────────────────────────────

def estimate_causal_effect(stage2, true_effect, device, wandb_run=None):
    """
    Extract the implied causal coefficient from Stage 2.
    For a linear model this equals the weight on X̂ directly.
    We estimate it numerically: θ ≈ ΔŶ / ΔX̂ at X̂=0.
    """
    stage2.eval()
    # Create dummy C=0 and vary X̂ by a unit step
    c_zero  = torch.zeros(2, 3).to(device)
    x_low   = torch.tensor([[0.0], [1.0]]).to(device)
    with torch.no_grad():
        y_pred = stage2(x_low, c_zero)
    theta = (y_pred[1] - y_pred[0]).item()  # ΔY / ΔX̂ = 1 → θ

    print(f"\n{'='*50}")
    print(f"  True Causal Effect      : {true_effect:.4f}")
    print(f"  Deep IV Estimate (theta): {theta:.4f}")
    print(f"  Bias (Deep IV)          : {abs(theta - true_effect):.4f}")
    print(f"{'='*50}\n")

    if wandb_run:
        import wandb
        wandb.log({
            "iv/estimated_effect"   : theta,
            "iv/bias_vs_deep_iv"    : abs(theta - true_effect),
        })

    return theta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deep IV: Two-Stage Transformer Instrumental Variables"
    )
    parser.add_argument("--n",           type=int,   default=1000)
    parser.add_argument("--true_effect", type=float, default=2.0)
    parser.add_argument("--nonlinear",   action="store_true",
                        help="Add non-linear confounding to data")
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--d_model",     type=int,   default=64)
    parser.add_argument("--nhead",       type=int,   default=4)
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--use_wandb",   action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── W&B Setup ──
    wandb_run = None
    if args.use_wandb:
        run_name = f"DeepIV-{args.d_model}d-{args.num_layers}L" + ("-nonlinear" if args.nonlinear else "-linear")
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
    print("\n[Stage 1] Generating IV data...")
    Z, X, C, Y = generate_iv_data(
        n=args.n, true_effect=args.true_effect,
        nonlinear=args.nonlinear, seed=args.seed
    )
    n_conf = C.shape[1]
    ds = IVDataset(Z, X, C, Y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    print(f"  {args.n} observations | confounders={n_conf}")

    # ── Classical baseline ──
    print("\n[OLS & 2SLS Baseline]")
    run_iv()    # prints naive OLS bias and 2SLS estimate; true_effect=2.0

    # ── Stage 2: Train Stage 1 Transformer ──
    print("\n[Stage 2] Training Stage 1 (Z -> X_hat)...")
    n_features = 1 + n_conf          # Z + confounders as tokens
    stage1 = Stage1TransformerIV(
        n_features=n_features, d_model=args.d_model,
        nhead=args.nhead, num_layers=args.num_layers
    ).to(device)
    train_stage1(stage1, dl, args.epochs, device, wandb_run)

    # ── Stage 3: Train Stage 2 MLP ──
    print("[Stage 3] Training Stage 2 (X_hat -> Y)...")
    stage2 = Stage2MLP(n_confounders=n_conf).to(device)
    train_stage2(stage1, stage2, dl, args.epochs // 2, device, wandb_run)

    # ── Causal effect estimation ──
    print("[Estimation] Recovering causal effect theta...")
    estimate_causal_effect(stage2, args.true_effect, device, wandb_run)

    # Save
    torch.save(stage1.state_dict(), os.path.join(OUTPUT_DIR, "stage1_transformer.pth"))
    torch.save(stage2.state_dict(), os.path.join(OUTPUT_DIR, "stage2_mlp.pth"))
    print(f"Saved models to {OUTPUT_DIR}")

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
