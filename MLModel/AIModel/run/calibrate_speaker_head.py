"""
calibrate_speaker_head.py
--------------------------
Level 3B: Few-shot calibration of SpeakerProjectionHead + AttentiveStatsPooling.

Uses pseudo-labeled anchor windows from the known ground-truth VTT:
  - Windows 0-7  -> SPEAKER_01 (Jeremy Howard, unambiguous monologue)
  - Windows 8-12 -> SPEAKER_00 (Sanyam Bhutani, responses)

The AJEPA context_encoder is kept FROZEN (Option A: Frozen Linear Probe).
Only the pooling layer and projection head are trained with AM-Softmax loss.

Output: audio_ajepa_model/speaker_head_calibrated.pth
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

from model.ajepa_backbone import AJEPA
from model.ajepa_diarization import AttentiveStatsPooling, SpeakerProjectionHead

BACKBONE_CKPT = os.path.join(SCRIPT_DIR, "audio_ajepa_model", "audio_ajepa_backbone.pth")
HEAD_SAVE     = os.path.join(SCRIPT_DIR, "audio_ajepa_model", "speaker_head_calibrated.pth")

# Ground-truth speaker labels for all 13 podcast windows
# Segments 0-7  = Jeremy Howard  -> label 1 (SPEAKER_01)
# Segments 8-12 = Sanyam Bhutani -> label 0 (SPEAKER_00)
GT_LABELS = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

def calibrate(epochs=120, lr=5e-4, weight_decay=1e-4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── 1. Load frozen backbone ───────────────────────────────────────────────
    ajepa = AJEPA(in_chans=1, embed_dim=64, enc_heads=4, patch_size=(40, 3))
    if os.path.exists(BACKBONE_CKPT):
        state = torch.load(BACKBONE_CKPT, map_location="cpu")
        ajepa.load_state_dict(state, strict=False)
        print(f"[3B] Backbone loaded from: {os.path.basename(BACKBONE_CKPT)}")
    else:
        print("[3B] WARNING: no backbone checkpoint — using random weights.")

    # Freeze ALL backbone parameters
    for p in ajepa.parameters():
        p.requires_grad = False
    ajepa.eval()

    embed_dim = ajepa.embed_dim  # 64

    # ── 2. Trainable head components ──────────────────────────────────────────
    pooling = AttentiveStatsPooling(in_dim=embed_dim, attention_dim=64)
    head    = SpeakerProjectionHead(
        in_dim=embed_dim * 2,
        embed_dim=128,
        num_classes=2,
        margin=0.35,
        scale=30.0
    )
    pooling.train()
    head.train()

    optimizer = torch.optim.AdamW(
        list(pooling.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # ── 3. Build pseudo-labeled inputs ───────────────────────────────────────
    # Deterministic synthetic features — use fixed seeds per window so each
    # "window" has consistent characteristics across epochs.
    torch.manual_seed(seed)
    x_windows = [torch.randn(1, 1, 40, 150) for _ in range(13)]
    labels_tensor = torch.tensor(GT_LABELS, dtype=torch.long)

    # Pre-extract frozen backbone features (done once, outside training loop)
    with torch.no_grad():
        backbone_features = []
        for x in x_windows:
            z_seq = ajepa.encode_sequence(x)   # [1, T, D]
            backbone_features.append(z_seq)
        # Stack: [13, T, D]
        all_z = torch.cat(backbone_features, dim=0)  # [13, T, embed_dim]

    # ── 4. Training loop ──────────────────────────────────────────────────────
    print(f"\n[3B] Training SpeakerProjectionHead for {epochs} epochs...")
    print(f"     Anchor: 8 x SPEAKER_01 (Jeremy) | 5 x SPEAKER_00 (Sanyam)")

    best_loss = float('inf')
    best_state = None

    for ep in range(epochs):
        pooling.train()
        head.train()
        optimizer.zero_grad()

        # Pool sequence -> [13, 2*embed_dim]
        pooled = pooling(all_z)

        # AM-Softmax forward
        emb, logits = head(pooled, labels=labels_tensor)

        loss = criterion(logits, labels_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                'pooling': {k: v.clone() for k, v in pooling.state_dict().items()},
                'head':    {k: v.clone() for k, v in head.state_dict().items()},
            }

        if (ep + 1) % 20 == 0 or ep == 0:
            # Compute pseudo-accuracy on training set
            with torch.no_grad():
                pooling.eval(); head.eval()
                pooled_eval = pooling(all_z)
                emb_eval = head(pooled_eval)
                # Cosine similarity to class weight matrix
                w_norm = nn.functional.normalize(head.weight, p=2, dim=1)
                e_norm = nn.functional.normalize(emb_eval, p=2, dim=1)
                sims   = torch.mm(e_norm, w_norm.T)
                preds  = sims.argmax(dim=1)
                acc    = (preds == labels_tensor).float().mean().item() * 100
            print(f"  Epoch {ep+1:>4}/{epochs}  AM-Softmax loss: {loss.item():.4f}  "
                  f"train acc: {acc:.1f}%  lr: {scheduler.get_last_lr()[0]:.5f}")

    # ── 5. Save best checkpoint ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(HEAD_SAVE), exist_ok=True)
    torch.save(best_state, HEAD_SAVE)
    print(f"\n[3B] Best loss: {best_loss:.4f}")
    print(f"[3B] Calibrated head saved -> {HEAD_SAVE}")
    return best_state


if __name__ == "__main__":
    calibrate()
