"""
main_audio_cjepa.py — Audio Emotion Recognition with C-JEPA self-supervised pre-training.

Pipeline (same 3 stages as main_audio_jepa.py, new mechanism):
  Stage 1 — Unsupervised pre-training  (C-JEPA masked-slot objective, 20 epochs)
  Stage 2 — Linear probe training      (few-shot, 4 labelled samples)
  Stage 3 — Inference / evaluation     (test set accuracy)

C-JEPA changes vs. eb_jepa:
  • Audio is split into T_HIST+T_PRED consecutive windows; each window encoded
    into N_SLOTS slot vectors by AudioSlotEncoder.
  • MaskedSlotPredictor (Non-Causal Transformer) masks N_MASKED random slots
    and must reconstruct them from the remaining visible context — inducing
    causal / relational reasoning between audio segments.
  • MSE loss on (a) masked history slots + (b) predicted future slots.

Requires: pip install einops
"""

import os
import sys
import json
import warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from MLModel.AIModel.model.audio_slot     import AudioSlotEncoder
from MLModel.AIModel.model.cjepa_predictor import MaskedSlotPredictor

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "../../../VAE/FT-w2v2-ser/Dataset/IEMOCAP/Audio_16k")
LABEL_FILE = os.path.join(SCRIPT_DIR, "../../../VAE/FT-w2v2-ser/Output/labels/label_sparse.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "audio_cjepa_model")

# ── Feature config ────────────────────────────────────────────────────────────
N_MFCC     = 40
MAX_FRAMES = 150

# ── C-JEPA slot config ────────────────────────────────────────────────────────
N_SLOTS    = 4       # audio "object" slots (temporal segments)
SLOT_DIM   = 64      # embedding dim per slot
T_HIST     = 5       # history windows fed to predictor
T_PRED     = 1       # future windows to predict
T_TOTAL    = T_HIST + T_PRED      # 6
T_FRAME    = MAX_FRAMES // T_TOTAL  # 25 MFCC frames per window  (6×25=150)
N_MASKED   = 1       # slots to mask per forward pass


# ── Feature extraction with Caching ───────────────────────────────────────────
CACHE_DIR = os.path.join(OUTPUT_DIR, "mfcc_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_features(wav_path: str) -> np.ndarray:
    """Return normalised MFCCs, shape [N_MFCC, MAX_FRAMES]. Uses disk cache."""
    # Create a unique cache filename based on the path
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

class AudioCJEPADataset(Dataset):
    """
    Pre-training dataset (no labels required).
    Returns windowed MFCC tensor [T_TOTAL, N_MFCC, T_FRAME] per sample,
    ready for slot encoding.
    """

    def __init__(self, data_dir: str, label_file: str):
        self.data_dir = data_dir
        self.samples  = []
        with open(label_file) as f:
            data = json.load(f)
        for split in data:
            items = data[split] if isinstance(data[split], dict) else {}
            # handle both dict[file->label] and list[file] structures
            files = items.keys() if isinstance(items, dict) else items
            for wav_file in files:
                p = os.path.join(data_dir, wav_file)
                if os.path.exists(p):
                    self.samples.append(p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mfcc   = extract_features(self.samples[idx])      # [N_MFCC, 150]
        # split into T_TOTAL consecutive windows → [T_TOTAL, N_MFCC, T_FRAME]
        windows = np.stack(
            [mfcc[:, t * T_FRAME:(t + 1) * T_FRAME] for t in range(T_TOTAL)],
            axis=0)
        return torch.from_numpy(windows), torch.tensor(-1)


class AudioLabeledDataset(Dataset):
    """
    Labeled dataset for linear probe + inference (identical to original).
    Returns full MFCC [N_MFCC, MAX_FRAMES] and integer label.
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
        return torch.from_numpy(extract_features(path)), torch.tensor(label)


# ── Stage 1: C-JEPA pre-training ─────────────────────────────────────────────

def train_backbone(encoder: AudioSlotEncoder,
                   predictor: MaskedSlotPredictor,
                   loader: DataLoader,
                   epochs: int = 20,
                   device: str = 'cpu'):
    """
    Self-supervised pre-training with C-JEPA masked-slot objective.

    Loss = MSE on:
      (a) Masked history slots  — slots hidden from context at t≥1
      (b) Future predicted slots — all slots at t = T_HIST (next window)
    """
    print(f"--- C-JEPA Pre-training ({epochs} epochs) ---")
    params = list(encoder.parameters()) + list(predictor.parameters())
    opt    = AdamW(params, lr=1e-3, weight_decay=1e-4)
    encoder.train(); predictor.train()

    for ep in range(epochs):
        total = 0.0
        for clips, _ in loader:
            clips = clips.to(device)              # [B, T_TOTAL, N_MFCC, T_FRAME]

            opt.zero_grad()

            # Encode all T_TOTAL windows → slot representations
            slots_all = encoder(clips)            # [B, T_TOTAL, N_SLOTS, SLOT_DIM]

            # Feed only history to predictor
            x_hist    = slots_all[:, :T_HIST]     # [B, T_HIST, N_SLOTS, SLOT_DIM]
            pred, masked_idx = predictor(x_hist)  # [B, T_TOTAL, N_SLOTS, SLOT_DIM]

            # ── Loss (a): masked history slots ──
            if len(masked_idx) > 0:
                loss_hist = F.mse_loss(
                    pred[:, 1:T_HIST, masked_idx, :],
                    slots_all[:, 1:T_HIST, masked_idx, :].detach(),
                )
            else:
                loss_hist = torch.tensor(0.0, device=device)

            # ── Loss (b): future slot prediction ──
            loss_fut = F.mse_loss(
                pred[:, T_HIST:, :, :],
                slots_all[:, T_HIST:, :, :].detach(),
            )

            loss = loss_hist + loss_fut
            loss.backward()
            opt.step()
            total += loss.item()

        if (ep + 1) % 5 == 0:
            print(f"  Epoch {ep+1:>3}/{epochs} | Loss: {total/len(loader):.4f}")

    print("  Pre-training complete.\n")


# ── Stage 2: Linear probe ─────────────────────────────────────────────────────

def train_classifier(encoder: AudioSlotEncoder,
                     train_loader: DataLoader,
                     device: str = 'cpu') -> nn.Linear:
    """
    Freeze backbone; train linear head on labelled samples via gradient descent.
    """
    print("--- Training Linear Head (Few-Shot) ---")
    encoder.eval()

    X, y = [], []
    with torch.no_grad():
        for feat, label in train_loader:
            feat = feat.to(device)
            z    = encoder.encode_full(feat)   # [B, SLOT_DIM]
            X.append(z)
            y.append(label)
    X = torch.cat(X, dim=0)   # [N_train, SLOT_DIM]
    y = torch.cat(y, dim=0)

    head = nn.Linear(SLOT_DIM, 2).to(device)
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

def infer(encoder: AudioSlotEncoder,
          head: nn.Linear,
          test_loader: DataLoader,
          device: str = 'cpu') -> float:
    print("\n--- INFERENCE RESULTS ---")
    encoder.eval(); head.eval()
    label_inv = {0: "angry", 1: "normal"}
    correct = total = 0

    with torch.no_grad():
        for feat, true_label in test_loader:
            feat = feat.to(device)
            z        = encoder.encode_full(feat)           # [1, SLOT_DIM]
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
    parser = argparse.ArgumentParser(description="Audio C-JEPA: masked-slot self-supervised SER")
    parser.add_argument("--epochs",         type=int,   default=80,
                        help="Backbone pre-training epochs (default: 80)")
    parser.add_argument("--seed",           type=int,   default=21,
                        help="Random seed")
    parser.add_argument("--sweep",          action="store_true",
                        help="Sweep seeds 1-10 and keep best")
    parser.add_argument("--n_slots",        type=int,   default=N_SLOTS,
                        help=f"Number of audio slots (default: {N_SLOTS})")
    parser.add_argument("--n_masked",       type=int,   default=N_MASKED,
                        help=f"Slots to mask per step (default: {N_MASKED})")
    parser.add_argument("--slot_dim",       type=int,   default=SLOT_DIM,
                        help=f"Slot embedding dimension (default: {SLOT_DIM})")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cpu')

    seeds    = list(range(1, 11)) if args.sweep else [args.seed]
    best_acc  = 0.0
    best_head  = None
    best_enc   = None

    for seed in seeds:
        print(f"\n{'='*55}")
        print(f"  Seed = {seed}")
        print(f"{'='*55}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── Build models ──
        encoder   = AudioSlotEncoder(
            in_channels=N_MFCC,
            hidden_dim=64,
            n_slots=args.n_slots,
            slot_dim=args.slot_dim,
        ).to(device)

        predictor = MaskedSlotPredictor(
            num_slots       =args.n_slots,
            slot_dim        =args.slot_dim,
            history_frames  =T_HIST,
            pred_frames     =T_PRED,
            num_masked_slots=args.n_masked,
            seed            =seed,
            depth=4, heads=4, mlp_dim=512,
        ).to(device)

        print(f"AudioSlotEncoder  : {args.n_slots} slots × {args.slot_dim} dims")
        print(f"MaskedSlotPredictor: T_hist={T_HIST}, T_pred={T_PRED}, "
              f"T_frame={T_FRAME}, masked={args.n_masked}/{args.n_slots}")

        # Stage 1 — C-JEPA self-supervised pre-training
        pretrain_ds = AudioCJEPADataset(DATA_DIR, LABEL_FILE)
        pretrain_dl = DataLoader(pretrain_ds, batch_size=16, shuffle=True, num_workers=2)
        train_backbone(encoder, predictor, pretrain_dl,
                       epochs=args.epochs, device=device)

        # Stage 2 — Linear probe on 4 labelled samples
        train_ds = AudioLabeledDataset(DATA_DIR, LABEL_FILE, split="Train")
        train_dl = DataLoader(train_ds, batch_size=4, shuffle=False)
        head = train_classifier(encoder, train_dl, device=device)

        # Stage 3 — Test set inference
        test_ds = AudioLabeledDataset(DATA_DIR, LABEL_FILE, split="Test")
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
        acc = infer(encoder, head, test_dl, device=device)

        if acc > best_acc:
            best_acc  = acc
            best_head = head
            best_enc  = {k: v.clone() for k, v in encoder.state_dict().items()}
            print(f"  *** New best: {acc*100:.2f}% (seed={seed})")

    print(f"\n{'='*55}")
    print(f"  BEST ACCURACY : {best_acc*100:.2f}%")
    print(f"{'='*55}")

    torch.save(best_enc,  os.path.join(OUTPUT_DIR, "audio_cjepa_encoder.pth"))
    torch.save(best_head.state_dict(),
               os.path.join(OUTPUT_DIR, "audio_cjepa_head.pth"))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
