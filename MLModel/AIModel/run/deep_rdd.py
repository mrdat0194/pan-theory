"""
deep_rdd.py — Neural Regression Discontinuity Design (Neural RDD) using dual Transformers.

Problem:
    Classical RDD (see Bayesian/someMethod/rdd.py) fits a linear regression on each
    side of the cutoff and reads off the jump. This is unreliable when:
      - The outcome function is non-linear near the cutoff.
      - The bandwidth is wide (far-from-cutoff units bias the linear fit).
      - The running variable has a complex density.

Solution:
    Train two independent Transformer models — one for each side of the cutoff —
    to approximate the non-linear outcome function. The causal effect is estimated as:
        TE = f_right(cutoff) - f_left(cutoff)
    evaluated by querying each model at exactly the cutoff point.

Pipeline:
    Stage 1 — Generate synthetic RDD data with optional non-linear outcome function.
    Stage 2 — Train left-side Transformer on units with running variable < cutoff.
    Stage 3 — Train right-side Transformer on units with running variable >= cutoff.
    Stage 4 — Compute the discontinuity jump; compare against OLS RDD baseline.

W&B:
    Enable with --use_wandb. Logs left/right losses and the estimated jump.

Usage:
    python deep_rdd.py
    python deep_rdd.py --use_wandb --epochs 150 --nonlinear
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

from Bayesian.someMethod.rdd import run_rdd  # classical RDD baseline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "deep_rdd_model")


# ── Data Generation ───────────────────────────────────────────────────────────

def generate_rdd_data(n=2000, cutoff=0.0, true_effect=15.0,
                      nonlinear=False, seed=42):
    """
    Generate synthetic RDD data mirroring rdd.py but with optional non-linear
    outcome function on each side.

    Returns:
        x     : [n]  running variable (centered at cutoff)
        y     : [n]  observed outcome
        treat : [n]  binary indicator (x >= cutoff)
    """
    rng = np.random.default_rng(seed)

    x     = rng.uniform(-3, 3, n)
    treat = (x >= cutoff).astype(np.float32)

    if nonlinear:
        # Non-linear smooth outcome + discontinuous jump at cutoff
        y = (2 * np.sin(x) + 0.5 * x**2
             + true_effect * treat
             + rng.normal(0, 2, n))
    else:
        # Linear outcome (matches rdd.py exactly)
        y = (5 + 2 * x
             + true_effect * treat
             + rng.normal(0, 2, n))

    return (x.astype(np.float32),
            y.astype(np.float32),
            treat.astype(np.float32))


class RDDDataset(Dataset):
    """Dataset of (running_variable, outcome) pairs for one side of the cutoff."""

    def __init__(self, x, y, cutoff=0.0, side="left"):
        if side == "left":
            mask = x < cutoff
        else:
            mask = x >= cutoff
        # Center the running variable at the cutoff
        self.x = torch.from_numpy((x[mask] - cutoff)).unsqueeze(-1)   # [N, 1]
        self.y = torch.from_numpy(y[mask]).unsqueeze(-1)               # [N, 1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class TransformerRDDSide(nn.Module):
    """
    Fits a smooth outcome function on one side of the RDD cutoff.
    Each unit's running variable (a single scalar) is projected into d_model
    and processed by a Transformer to produce a flexible non-linear regression.

    Input:  x [B, 1]   — centered running variable (x - cutoff)
    Output: y [B, 1]   — predicted outcome
    """

    def __init__(self, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        # Fourier-style feature expansion before Transformer input
        self.n_fourier   = 16
        self.input_proj  = nn.Linear(1 + 2 * self.n_fourier, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.05,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head        = nn.Linear(d_model, 1)

    def _fourier_features(self, x):
        """
        Expand scalar x into Fourier features to help Transformer learn
        periodic / smooth functions near the cutoff.
        Output: [B, 1 + 2 * n_fourier]
        """
        freqs  = torch.arange(1, self.n_fourier + 1, device=x.device, dtype=x.dtype)
        angles = x * freqs * torch.pi               # [B, n_fourier]
        return torch.cat([x, torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, 1+2k]

    def forward(self, x):
        # x: [B, 1]
        feat   = self._fourier_features(x)           # [B, 1 + 2*n_fourier]
        tokens = self.input_proj(feat).unsqueeze(1)  # [B, 1, d_model]
        out    = self.transformer(tokens)             # [B, 1, d_model]
        return self.head(out[:, 0, :])               # [B, 1]


class NeuralRDD(nn.Module):
    """
    Combined Neural RDD model with independent left and right side Transformers.
    Causal effect = f_right(cutoff) - f_left(cutoff).
    """

    def __init__(self, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.left  = TransformerRDDSide(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.right = TransformerRDDSide(d_model=d_model, nhead=nhead, num_layers=num_layers)

    def causal_effect(self, device):
        """Evaluate both sides at the cutoff (x=0) and return the jump."""
        zero = torch.zeros(1, 1, device=device)
        self.left.eval()
        self.right.eval()
        with torch.no_grad():
            y_right = self.right(zero).item()
            y_left  = self.left(zero).item()
        return y_right - y_left


# ── Training ──────────────────────────────────────────────────────────────────

def train_side(model_side, loader, epochs, device, label, wandb_run=None):
    """Train one side of the RDD model."""
    opt = AdamW(model_side.parameters(), lr=1e-3, weight_decay=1e-4)
    model_side.train()
    print(f"--- RDD {label}-side Training ({epochs} epochs) ---")

    for ep in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            y_hat = model_side(x)
            loss  = nn.functional.mse_loss(y_hat, y)
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(loader)
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:>4}/{epochs} | {label}-side Loss: {avg:.4f}")
        if wandb_run:
            import wandb
            wandb.log({
                f"rdd/{label.lower()}_loss": avg,
                "rdd/epoch": ep + 1,
            })

    print(f"  {label}-side training complete.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Neural RDD: Dual Transformer Regression Discontinuity Design"
    )
    parser.add_argument("--n",           type=int,   default=2000)
    parser.add_argument("--cutoff",      type=float, default=0.0)
    parser.add_argument("--true_effect", type=float, default=15.0)
    parser.add_argument("--nonlinear",   action="store_true",
                        help="Use non-linear outcome function")
    parser.add_argument("--epochs",      type=int,   default=120)
    parser.add_argument("--d_model",     type=int,   default=64)
    parser.add_argument("--nhead",       type=int,   default=4)
    parser.add_argument("--num_layers",  type=int,   default=3)
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
        run_name = f"NeuralRDD-{args.d_model}d-{args.num_layers}L" + ("-nonlinear" if args.nonlinear else "-linear")
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
    print("\n[Stage 1] Generating RDD data...")
    x, y, treat = generate_rdd_data(
        n=args.n, cutoff=args.cutoff, true_effect=args.true_effect,
        nonlinear=args.nonlinear, seed=args.seed
    )
    n_left  = int((x < args.cutoff).sum())
    n_right = int((x >= args.cutoff).sum())
    print(f"  {args.n} observations | cutoff={args.cutoff}")
    print(f"  Left side: {n_left} | Right side: {n_right}")

    # ── Classical baseline ──
    print("\n[OLS RDD Baseline]")
    run_rdd()    # prints OLS regression discontinuity estimate; true_effect=15.0

    # ── Build model ──
    model = NeuralRDD(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers
    ).to(device)

    # ── Stage 2: Train left side ──
    print("\n[Stage 2] Training left-side Transformer (x < cutoff)...")
    left_ds = RDDDataset(x, y, cutoff=args.cutoff, side="left")
    left_dl = DataLoader(left_ds, batch_size=args.batch_size, shuffle=True)
    train_side(model.left, left_dl, args.epochs, device, "Left", wandb_run)

    # ── Stage 3: Train right side ──
    print("[Stage 3] Training right-side Transformer (x >= cutoff)...")
    right_ds = RDDDataset(x, y, cutoff=args.cutoff, side="right")
    right_dl = DataLoader(right_ds, batch_size=args.batch_size, shuffle=True)
    train_side(model.right, right_dl, args.epochs, device, "Right", wandb_run)

    # ── Stage 4: Estimate causal effect ──
    print("[Stage 4] Estimating discontinuity jump at cutoff...")
    effect      = model.causal_effect(device)
    deep_bias   = abs(effect - args.true_effect)

    print(f"\n{'='*50}")
    print(f"  True Treatment Effect    : {args.true_effect:.4f}")
    print(f"  Neural RDD Estimate      : {effect:.4f}  (bias={deep_bias:.4f})")
    print(f"{'='*50}\n")

    if wandb_run:
        import wandb
        wandb.log({
            "rdd/estimated_effect"    : effect,
            "rdd/bias_vs_neural_rdd"  : deep_bias,
        })

    # Save
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "neural_rdd.pth"))
    print(f"Saved model to {OUTPUT_DIR}")

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
