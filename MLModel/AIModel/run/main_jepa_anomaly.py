"""
MTS-JEPA Anomaly Detection Inference Script
===========================================
Loads a pre-trained JEPA backbone and evaluates latent prediction error (MSE)
on the Synth_data_20042020_speed40 dataset to detect time-series anomalies.

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

from MLModel.AIModel.model.jepa_backbone import build_jepa, compute_anomaly_score

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
    return parser.parse_args()

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
    
    # In a real scenario, you would load the trained weights here:
    # model.load_state_dict(torch.load('path_to_weights.pth'))
    # For now, it will compute scores using initialized weights (random representations)
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
        score = compute_anomaly_score(model, tensor, steps=5).mean().item()
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
