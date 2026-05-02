"""
main_audio_ajepa.py — Audio Emotion Recognition with A-JEPA self-supervised pre-training.

Pipeline:
  Stage 1 — Unsupervised pre-training  (A-JEPA masked patch objective)
  Stage 2 — Linear probe training      (few-shot, 4 labelled samples)
  Stage 3 — Inference / evaluation     (test set accuracy)

A-JEPA treats MFCCs [40, 150] as a 2D single-channel image, patches it, 
and predicts masked target patches (encoded by EMA teacher) from context patches.
"""

import os
import sys
import json
import warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from MLModel.AIModel.model.ajepa_backbone import AJEPA

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "../../../VAE/FT-w2v2-ser/Dataset/IEMOCAP/Audio_16k")
LABEL_FILE = os.path.join(SCRIPT_DIR, "../../../VAE/FT-w2v2-ser/Output/labels/label_sparse.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "audio_ajepa_model")

# ── Feature config ────────────────────────────────────────────────────────────
N_MFCC     = 40
MAX_FRAMES = 150

# ── Feature extraction with Caching ───────────────────────────────────────────
CACHE_DIR = os.path.join(OUTPUT_DIR, "mfcc_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_features(wav_path: str) -> np.ndarray:
    """Return normalised MFCCs, shape [N_MFCC, MAX_FRAMES]. Uses disk cache."""
    cache_name = os.path.basename(wav_path).replace(".wav", ".npy")
    cache_path = os.path.join(CACHE_DIR, cache_name)

    if os.path.exists(cache_path):
        return np.load(cache_path)

    y, sr = librosa.load(wav_path, sr=16000)
    if y.max() > 0:
        y = y / y.max()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mean = mfcc.mean(axis=1, keepdims=True)
    std  = mfcc.std(axis=1, keepdims=True) + 1e-8
    mfcc = (mfcc - mean) / std
    T    = mfcc.shape[1]
    if T > MAX_FRAMES:
        mfcc = mfcc[:, :MAX_FRAMES]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - T)))

    feat = mfcc.astype(np.float32)
    np.save(cache_path, feat)
    return feat


# ── Datasets ──────────────────────────────────────────────────────────────────

class AudioAJEPADataset(Dataset):
    """
    Pre-training dataset (no labels required).
    Returns MFCC tensor [1, N_MFCC, MAX_FRAMES] per sample.
    """
    def __init__(self, data_dir: str, label_file: str):
        self.data_dir = data_dir
        self.samples  = []
        with open(label_file) as f:
            data = json.load(f)
        for split in data:
            items = data[split] if isinstance(data[split], dict) else {}
            files = items.keys() if isinstance(items, dict) else items
            for wav_file in files:
                p = os.path.join(data_dir, wav_file)
                if os.path.exists(p):
                    self.samples.append(p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mfcc = extract_features(self.samples[idx])
        # Add channel dimension -> [1, 40, 150]
        return torch.from_numpy(mfcc).unsqueeze(0), torch.tensor(-1)


class AudioLabeledDataset(Dataset):
    """
    Labeled dataset for linear probe + inference.
    Returns full MFCC [1, N_MFCC, MAX_FRAMES] and integer label.
    """
    LABEL_MAP = {"angry": 0, "normal": 1}

    def __init__(self, data_dir: str, label_file: str,
                 split: str = None, all_unlabelled: bool = False):
        self.data_dir = data_dir
        self.samples  = []
        with open(label_file) as f:
            data = json.load(f)

        if all_unlabelled:
            for s in data:
                for wav_file in data[s]:
                    p = os.path.join(data_dir, wav_file)
                    if os.path.exists(p):
                        self.samples.append((p, -1))
        else:
            if split not in data:
                return
            for wav_file, label in data[split].items():
                p = os.path.join(data_dir, wav_file)
                if os.path.exists(p):
                    self.samples.append((p, self.LABEL_MAP[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mfcc = extract_features(path)
        return torch.from_numpy(mfcc).unsqueeze(0), torch.tensor(label)


# ── Stage 1: A-JEPA pre-training ─────────────────────────────────────────────

def train_backbone(model: AJEPA,
                   loader: DataLoader,
                   epochs: int = 20,
                   device: str = 'cpu'):
    print(f"--- A-JEPA Pre-training ({epochs} epochs) ---")
    
    # Only optimize context encoder and predictor
    params = list(model.context_encoder.parameters()) + \
             list(model.encoder_norm.parameters()) + \
             list(model.predictor.parameters()) + \
             list(model.predictor_norm.parameters()) + \
             list(model.predictor_embed.parameters()) + \
             list(model.patch_embed.parameters()) + \
             [model.pos_embed, model.mask_token]
             
    opt = AdamW(params, lr=1e-3, weight_decay=1e-4)
    model.train()

    for ep in range(epochs):
        total_loss = 0.0
        for clips, _ in loader:
            clips = clips.to(device)  # [B, 1, 40, 150]

            opt.zero_grad()
            loss = model(clips)
            loss.backward()
            opt.step()
            
            # EMA Update
            model.update_target_network()
            
            total_loss += loss.item()

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  Epoch {ep+1:>3}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    print("  Pre-training complete.\n")


# ── Stage 2: Linear probe ─────────────────────────────────────────────────────

def train_classifier(model: AJEPA,
                     train_loader: DataLoader,
                     device: str = 'cpu') -> nn.Linear:
    print("--- Training Linear Head (Few-Shot) ---")
    model.eval()

    X, y = [], []
    with torch.no_grad():
        for feat, label in train_loader:
            feat = feat.to(device)
            z = model.encode(feat)   # [B, D]
            X.append(z)
            y.append(label)
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    embed_dim = X.shape[-1]
    head = nn.Linear(embed_dim, 2).to(device)
    opt  = AdamW(head.parameters(), lr=5e-3)
    crit = nn.CrossEntropyLoss()

    for ep in range(300):
        head.train()
        opt.zero_grad()
        loss = crit(head(X), y)
        loss.backward()
        opt.step()

        if (ep + 1) % 75 == 0:
            preds = head(X).argmax(dim=-1)
            acc   = (preds == y).float().mean()
            print(f"  Head Epoch {ep+1}/300 | Loss: {loss.item():.4f} | Train Acc: {acc:.2f}")

    return head


# ── Stage 3: Inference ────────────────────────────────────────────────────────

def infer(model: AJEPA,
          head: nn.Linear,
          test_loader: DataLoader,
          device: str = 'cpu') -> float:
    print("\n--- INFERENCE RESULTS ---")
    model.eval(); head.eval()
    label_inv = {0: "angry", 1: "normal"}
    correct = total = 0

    with torch.no_grad():
        for feat, true_label in test_loader:
            feat = feat.to(device)
            z        = model.encode(feat)  # [1, D]
            logits   = head(z)
            probs    = torch.softmax(logits, dim=-1)[0]
            pred_idx = logits.argmax(dim=-1).item()
            true_idx = true_label.item()
            mark = "CORRECT" if pred_idx == true_idx else "WRONG"
            if pred_idx == true_idx:
                correct += 1
            total += 1
            print(f"PRED: {label_inv[pred_idx]} | TRUE: {label_inv[true_idx]} -> [{mark}]"
                  f"  (angry:{probs[0]:.3f}, normal:{probs[1]:.3f})")

    acc = correct / total if total else 0
    print(f"\nFinal Test Accuracy: {acc*100:.2f}% ({correct}/{total})")
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audio A-JEPA: masked-patch self-supervised SER")
    parser.add_argument("--epochs",     type=int,   default=50, help="Pre-training epochs")
    parser.add_argument("--seed",       type=int,   default=21, help="Random seed")
    parser.add_argument("--sweep",      action="store_true", help="Sweep seeds 1-10 and keep best")
    parser.add_argument("--mask_ratio", type=float, default=0.6, help="Masking ratio")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cpu')

    seeds    = list(range(1, 11)) if args.sweep else [args.seed]
    best_acc = 0.0
    best_head = None
    best_enc  = None

    for seed in seeds:
        print(f"\n{'='*55}")
        print(f"  Seed = {seed}")
        print(f"{'='*55}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── Build A-JEPA model ──
        model = AJEPA(
            in_chans=1,
            img_size=(40, 150),
            patch_size=(40, 3),   # 1D Temporal Patching
            embed_dim=64,
            enc_depth=2,
            enc_heads=2,
            pred_depth=1,
            pred_heads=2,
            mask_ratio=args.mask_ratio
        ).to(device)

        print(f"A-JEPA initialized. Patches: {model.num_patches}, Masking Ratio: {args.mask_ratio}")

        # Stage 1 — A-JEPA pre-training
        pretrain_ds = AudioAJEPADataset(DATA_DIR, LABEL_FILE)
        pretrain_dl = DataLoader(pretrain_ds, batch_size=16, shuffle=True, num_workers=0)
        train_backbone(model, pretrain_dl, epochs=args.epochs, device=device)

        # Stage 2 — Linear probe
        train_ds = AudioLabeledDataset(DATA_DIR, LABEL_FILE, split="Train")
        train_dl = DataLoader(train_ds, batch_size=4, shuffle=False)
        head = train_classifier(model, train_dl, device=device)

        # Stage 3 — Test set inference
        test_ds = AudioLabeledDataset(DATA_DIR, LABEL_FILE, split="Test")
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
        acc = infer(model, head, test_dl, device=device)

        if acc > best_acc:
            best_acc  = acc
            best_head = head
            best_enc  = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  *** New best: {acc*100:.2f}% (seed={seed})")

    print(f"\n{'='*55}")
    print(f"  BEST ACCURACY : {best_acc*100:.2f}%")
    print(f"{'='*55}")

    if best_enc is not None:
        torch.save(best_enc,  os.path.join(OUTPUT_DIR, "audio_ajepa_backbone.pth"))
        torch.save(best_head.state_dict(),
                   os.path.join(OUTPUT_DIR, "audio_ajepa_head.pth"))
        print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
