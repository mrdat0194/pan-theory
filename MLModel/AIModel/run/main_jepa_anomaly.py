"""
MTS-JEPA Anomaly Detection Inference Script (Upgraded: Latent ODE JEPA)
========================================================================
Loads a pre-trained **Latent ODE JEPA** backbone and evaluates the ODE
prediction error (MSE) on the Synth_data_20042020_speed40 dataset to
detect time-series anomalies.

Upgrade from v1 (MTS-JEPA):
    BEFORE: Discrete Transformer (ARPredictor) predicts z[t+1] from z[t].
    AFTER:  Neural ODE integrates dz/dt = f_theta(z, t) continuously
            via RK4, enabling robust detection on irregular time series.

Usage examples:
    # Evaluate a balanced sample of 20 files per defect type
    python -m MLModel.AIModel.run.main_jepa_anomaly

    # Evaluate ALL files
    python -m MLModel.AIModel.run.main_jepa_anomaly --per_type 0

    # Evaluate only crack files, flag Anomaly Score > 0.5
    python -m MLModel.AIModel.run.main_jepa_anomaly --defect crack --threshold 0.5

    # Reproducible run with a fixed seed
    python -m MLModel.AIModel.run.main_jepa_anomaly --seed 42
"""

import argparse
import glob
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import normalize

from MLModel.AIModel.model.latent_ode_jepa import build_jepa, compute_anomaly_score

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration & Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Data has been moved to MLData/Lira...
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'MLData', 'Lira', 'Synth_data_20042020_speed40'))
# Mock Model Directory for demonstration (in practice, load from checkpoint)
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'model_nn_save'))

SEQ_SIZE = 150
INPUT_SIZE = 2 # acceleration + severity

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_mat_file(filepath):
    """
    Load .mat file → tensor (1, input_size=2, seq_size).
    Note: JEPA 1D Conv expects [Batch, Channels, Time] which is [1, 2, 150].
    """
    try:
        f = loadmat(filepath)
        acc = f["acceleration"].reshape(1, -1)   # (1, T)
        severity = f["severity"].reshape(1, -1)  # (1, T)

        col1 = normalize(acc)[0]       # (T,)
        col2 = normalize(severity)[0]  # (T,)

        # Slice to the same fixed window as VAE_LSTM
        col1 = col1[250:400]
        col2 = col2[250:400]

        if len(col1) < SEQ_SIZE:
            return None

        # Data shape: [Channels, Time] -> [2, 150]
        data = np.stack([col1, col2], axis=0)
        # Add batch dim: [1, 2, 150]
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return tensor
    except Exception as e:
        # Some files might be malformed
        return None

def collect_files(data_dir: str, defect_filter=None, per_type: int = 20, seed: int = 0):
    """Gather .mat files grouped by defect type."""
    all_files = glob.glob(os.path.join(data_dir, "*.mat"))
    by_type = {}
    for fp in all_files:
        parts = os.path.basename(fp).split("_")
        dtype = parts[-3] if len(parts) >= 4 else "unknown"
        by_type.setdefault(dtype, []).append(fp)

    rng = random.Random(seed)
    selected = []
    for dtype, files in sorted(by_type.items()):
        if defect_filter and dtype != defect_filter:
            continue
        files = sorted(files)
        if per_type and per_type < len(files):
            files = rng.sample(files, per_type)
        selected.extend(files)

    return sorted(selected)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="MTS-JEPA Anomaly Detection")
    parser.add_argument("--per_type", type=int, default=20, help="Files evaluated per defect type (0 = all)")
    parser.add_argument("--defect", type=str, default=None, help="Evaluate only this defect type")
    parser.add_argument("--threshold", type=float, default=None, help="Prediction error threshold for anomalies")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # Training arguments
    parser.add_argument("--train", action="store_true", help="Run training mode before inference")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    parser.add_argument("--use_rankfeat", action="store_true", help="Apply RankFeat at inference time to remove rank-1 feature")
    parser.add_argument("--use_rankweight", action="store_true", help="Apply RankWeight to strip rank-1 component from weights")
    
    return parser.parse_args()

def train_model(model, args, device):
    print(f"\n--- Starting JEPA Training ---")
    # Collect all available files for self-supervised training
    # (JEPA learns normal dynamics from the available data)
    files = collect_files(DATA_DIR, defect_filter=None, per_type=0, seed=args.seed)
    if not files:
        print("No data found for training.")
        return

    print(f"Loading {len(files)} files for training...")
    all_tensors = []
    for fp in files:
        tensor = load_mat_file(fp)
        if tensor is not None:
            # Remove the batch dimension added by load_mat_file for stacking
            all_tensors.append(tensor.squeeze(0))
    
    if not all_tensors:
        print("No valid tensors extracted for training.")
        return

    # Stack into [N, Channels, Time]
    X_train = torch.stack(all_tensors)
    dataset = torch.utils.data.TensorDataset(X_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Unroll JEPA to predict future states
            _, losses = model.unroll(
                data,
                actions=None,
                nsteps=2,
                unroll_mode="parallel",
                compute_loss=True,
                return_all_steps=False
            )
            loss, regl, rloss_unweight, regldict, pl = losses
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Loss: {total_loss / len(train_loader):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "jepa_backbone.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to:\n  {save_path}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nInitializing MTS-JEPA Backbone...")
    # Instantiate JEPA Backbone (in_channels=2 for acc+sev)
    model = build_jepa(in_channels=INPUT_SIZE, hidden_dim=64, latent_dim=128).to(device)
    
    if args.train:
        train_model(model, args, device)

    weights_path = os.path.join(MODEL_DIR, "jepa_backbone.pth")
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("WARNING: No trained weights found. Using random initialized weights.")
        print("Run with '--train' to train the model first.")
        
    if args.use_rankweight:
        from MLModel.AIModel.model.latent_ode_jepa import apply_rankweight
        print("Applying RankWeight surgery to model weights...")
        apply_rankweight(model)

    model.eval()
    print("Model ready.\n")

    # ── Collect files ─────────────────────────────────────────────────────────
    files = collect_files(DATA_DIR, defect_filter=args.defect, per_type=args.per_type, seed=args.seed)
    if not files:
        print(f"No .mat files found in:\n  {DATA_DIR}")
        return

    n_all = len(glob.glob(os.path.join(DATA_DIR, "*.mat")))
    scope = f"defect={args.defect}" if args.defect else "all defect types"
    print(f"Dataset  : {n_all} files total — evaluating {len(files)} ({scope})")
    print()
    print(f"{'File':<62}  {'Defect':<8}  {'Score':>10}")
    print("-" * 86)

    results = []
    skipped = 0

    for fp in files:
        basename = os.path.basename(fp)
        parts = basename.split("_")
        dtype = parts[-3] if len(parts) >= 4 else "unknown"

        tensor = load_mat_file(fp)
        if tensor is None:
            skipped += 1
            continue
            
        tensor = tensor.to(device)
        
        # JEPA Anomaly Score based on multi-step predictive error in latent space
        # Now passing the use_rankfeat flag down to the score function!
        score = compute_anomaly_score(model, tensor, steps=5, use_rankfeat=args.use_rankfeat)
        if isinstance(score, torch.Tensor):
            score = score.mean().item()
        results.append({"file": basename, "defect": dtype, "score": score})

        flag = ""
        if args.threshold is not None and score > args.threshold:
            flag = "  *** ANOMALY"
        print(f"{basename:<62}  {dtype:<8}  {score:>10.6f}{flag}")

    if not results:
        print("No valid files were processed.")
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    scores = [r["score"] for r in results]
    print("\n" + "=" * 86)
    print(f"  Files evaluated : {len(results)}" + (f"  |  skipped: {skipped}" if skipped else ""))
    print(f"  Overall Score   : mean={np.mean(scores):.6f}  std={np.std(scores):.6f}")

    defect_types = sorted({r["defect"] for r in results})
    if len(defect_types) > 1:
        print("\n  Per-defect Score breakdown:")
        print(f"    {'Defect':<10}  {'n':>5}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
        print("    " + "-" * 60)
        for dt in defect_types:
            vals = [r["score"] for r in results if r["defect"] == dt]
            print(f"    {dt:<10}  {len(vals):>5}  {np.mean(vals):>10.6f}  {np.std(vals):>10.6f}  {min(vals):>10.6f}  {max(vals):>10.6f}")

    if args.threshold is None:
        suggested = np.mean(scores) + 2 * np.std(scores)
        print(f"\n  Suggested threshold : mean + 2 * std = {suggested:.6f}")
        print(f"  Re-run with:  --threshold {suggested:.4f}")
    else:
        anomalies = [r for r in results if r["score"] > args.threshold]
        print(f"\n  Threshold = {args.threshold}")
        print(f"  Anomalies detected : {len(anomalies)}  ({100*len(anomalies)/len(results):.1f}%)")

if __name__ == "__main__":
    main()
