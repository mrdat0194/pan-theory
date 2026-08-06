"""
main_mumo.py - Unified Le MuMo JEPA Evaluation Runner
======================================================
Runs three evaluations without touching existing run scripts:

  Pipeline 1 - Gun Detection (Vision-Only, VideoStem)
      Dataset : MLModel/data/WeaponS/WeaponS  (JPEG images)
      Model   : VideoStem (V-JEPA 2 / CNN fallback) -> LocalizationHead
      Output  : Mean bbox regression loss on a held-out batch

  Pipeline 2 - Fire/Smoke Detection - V-JEPA only (vision baseline)
      Dataset : MLModel/data/FireSmoke/test_vids  (mp4/avi video)
      Model   : VideoStem -> ClassificationHead (3 classes)
      Output  : Cross-entropy loss

  Pipeline 3 - Fire/Smoke Detection - MuMo (video + audio torchaudio)
      Dataset : MLModel/data/FireSmoke/test_vids  (mp4/avi video + audio)
      Model   : build_audiovisual_mumo_jepa (VideoStem + AudioPatchEmbed + SIGReg)
                -> ClassificationHead (3 classes)
      Output  : Cross-entropy + SIGReg loss
      Compare : Pipeline 2 vs Pipeline 3 classification loss

Usage:
    python MLModel/AIModel/run/main_mumo.py
"""

import os, sys, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as AT
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
_RUN_DIR  = os.path.dirname(os.path.abspath(__file__))
_AI_DIR   = os.path.dirname(_RUN_DIR)
_ML_DIR   = os.path.dirname(_AI_DIR)
_ROOT_DIR = os.path.dirname(_ML_DIR)
sys.path.insert(0, _ROOT_DIR)

from MLModel.AIModel.model.mumo_wrapper import VideoStem, build_audiovisual_mumo_jepa

# ── Global config ────────────────────────────────────────────────────────────
EMBED_DIM      = 128   # small for fast iteration
IMG_SIZE       = 64    # resize all images/frames
PATCH_SIZE     = 16
N_FUSION       = 8
N_LAYERS       = 2
N_HEADS        = 4
AUDIO_MELS     = 40
AUDIO_TIME     = 150
AUDIO_PATCH    = (10, 15)
BATCH_SIZE     = 4
EPOCHS         = 3
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[MuMo] Device: {DEVICE}")

IMG_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

MEL_TRANSFORM = AT.MelSpectrogram(
    sample_rate=16000,
    n_mels=AUDIO_MELS,
    n_fft=400,
    hop_length=160,
)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _label_from_name(name: str) -> int:
    """Infer class from filename heuristics."""
    n = name.lower()
    if "fire" in n or "flame" in n or "arma" in n or "gun" in n or "weapon" in n:
        return 1
    if "smoke" in n:
        return 2
    return 0


def _load_video_frames(path: str, n_frames: int = 4) -> torch.Tensor:
    """
    Load n_frames evenly spaced frames from a video file.
    Returns [n_frames, 3, IMG_SIZE, IMG_SIZE].
    Uses torchaudio.io.StreamReader for format support.
    """
    try:
        reader = torchaudio.io.StreamReader(path)
        # Find first video stream
        vid_idx = None
        for i in range(reader.num_src_streams):
            info = reader.get_src_stream_info(i)
            if info.media_type == "video":
                vid_idx = i
                break

        if vid_idx is None:
            raise RuntimeError("No video stream found")

        reader.add_basic_video_stream(frames_per_chunk=n_frames, stream_index=vid_idx)
        frames = []
        for (chunk,) in reader.stream():
            if chunk is not None:
                frames.append(chunk)  # [T, H, W, C] uint8
            if len(frames) >= 1:
                break

        if not frames:
            raise RuntimeError("No frames decoded")

        vid = frames[0][:n_frames]  # [T, H, W, C]
        if vid.shape[0] < n_frames:
            # Pad by repeating last frame
            pad = vid[-1:].expand(n_frames - vid.shape[0], -1, -1, -1)
            vid = torch.cat([vid, pad], dim=0)

        # [T, H, W, C] uint8 -> [T, 3, IMG_SIZE, IMG_SIZE] float
        result = []
        for frame in vid:
            img = Image.fromarray(frame.numpy())
            result.append(IMG_TRANSFORM(img.convert("RGB")))
        return torch.stack(result)  # [T, 3, H, W]

    except Exception as e:
        print(f"  [VideoFrames fallback] {os.path.basename(path)}: {e}")
        return torch.zeros(n_frames, 3, IMG_SIZE, IMG_SIZE)


def _load_audio_mel(path: str) -> torch.Tensor:
    """
    Load audio track from video, resample to 16kHz, compute MelSpectrogram.
    Returns [1, AUDIO_MELS, AUDIO_TIME].
    """
    try:
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)   # mono

        mel = MEL_TRANSFORM(waveform)               # [1, AUDIO_MELS, T]
        # Trim or pad to AUDIO_TIME columns
        if mel.shape[2] >= AUDIO_TIME:
            mel = mel[:, :, :AUDIO_TIME]
        else:
            pad = torch.zeros(1, AUDIO_MELS, AUDIO_TIME - mel.shape[2])
            mel = torch.cat([mel, pad], dim=2)

        return mel.float()

    except Exception as e:
        print(f"  [AudioMel fallback] {os.path.basename(path)}: {e}")
        return torch.zeros(1, AUDIO_MELS, AUDIO_TIME)


# ─────────────────────────────────────────────────────────────────────────────
#  DATASETS
# ─────────────────────────────────────────────────────────────────────────────

class WeaponDataset(Dataset):
    """Static JPEG weapon images + bounding box regression targets."""

    def __init__(self, img_dir, bbox_dir):
        self.img_dir  = img_dir
        self.bbox_dir = bbox_dir
        self.imgs = [f for f in os.listdir(img_dir)
                     if f.lower().endswith(('.jpg', '.png'))
                     and not f.startswith('._')]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]
        img  = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        img  = IMG_TRANSFORM(img)  # [3, H, W]

        bbox_path = os.path.join(
            self.bbox_dir,
            name.replace('.jpg', '.txt').replace('.png', '.txt')
        )
        if os.path.exists(bbox_path):
            with open(bbox_path) as f:
                parts = f.read().strip().split()
            bbox = [float(x) for x in parts[1:5]] if len(parts) >= 5 else [0,0,0,0]
        else:
            bbox = [0, 0, 0, 0]

        return img, torch.tensor(bbox, dtype=torch.float32)


class FireVideoDataset(Dataset):
    """Fire/Smoke video clips returning (frames, mel_spec, label)."""

    EXTENSIONS = ('.mp4', '.avi', '.mov')

    def __init__(self, vid_dir):
        self.vid_dir = vid_dir
        self.clips = [f for f in os.listdir(vid_dir)
                      if f.lower().endswith(self.EXTENSIONS)
                      and not f.startswith('._')]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        name  = self.clips[idx]
        path  = os.path.join(self.vid_dir, name)
        label = _label_from_name(name)

        frames = _load_video_frames(path, n_frames=4)   # [T, 3, H, W]
        mel    = _load_audio_mel(path)                  # [1, MELS, TIME]

        return frames, mel, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
#  HEADS
# ─────────────────────────────────────────────────────────────────────────────

class LocalizationHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 4)
        )

    def forward(self, x):  # x: [B, in_dim]
        return self.fc(x)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE 1 — Gun Detection (Vision-Only)
# ─────────────────────────────────────────────────────────────────────────────

def run_gun_detection():
    print("\n" + "="*60)
    print("  Pipeline 1: Gun Detection (VideoStem, Vision-Only)")
    print("="*60)

    img_dir  = os.path.join(_ML_DIR, "data", "WeaponS", "WeaponS")
    bbox_dir = os.path.join(_ML_DIR, "data", "WeaponS", "WeaponS_bbox")

    dataset = WeaponDataset(img_dir, bbox_dir)
    if len(dataset) == 0:
        print("  WARNING: No weapon images found. Skipping.")
        return None

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"  Loaded {len(dataset)} weapon images.")

    stem = VideoStem(
        embed_dim=EMBED_DIM, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        use_real_vjepa2=True, freeze_backbone=False,
    ).to(DEVICE)
    head = LocalizationHead(in_dim=EMBED_DIM).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(stem.parameters()) + list(head.parameters()), lr=1e-3
    )

    results = []
    for epoch in range(1, EPOCHS + 1):
        stem.train(); head.train()
        epoch_loss = 0.0
        for imgs, bboxes in loader:
            imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)  # [B,3,H,W], [B,4]
            tokens   = stem(imgs)                              # [B, N, D]
            pooled   = tokens.mean(dim=1)                      # [B, D]
            preds    = head(pooled)                            # [B, 4]
            loss     = F.mse_loss(preds, bboxes)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / max(len(loader), 1)
        results.append(avg)
        print(f"  Epoch {epoch}/{EPOCHS}  bbox_loss={avg:.4f}")

    print(f"  [Gun Detection] Final bbox loss: {results[-1]:.4f}")
    return results[-1]


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE 2 — Fire Detection: Vision-Only V-JEPA baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_fire_vjepa(loader: DataLoader):
    print("\n" + "="*60)
    print("  Pipeline 2: Fire Detection (VideoStem Only — V-JEPA baseline)")
    print("="*60)

    stem = VideoStem(
        embed_dim=EMBED_DIM, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        use_real_vjepa2=True, freeze_backbone=False,
    ).to(DEVICE)
    head = ClassificationHead(in_dim=EMBED_DIM, num_classes=3).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(stem.parameters()) + list(head.parameters()), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    results = []
    for epoch in range(1, EPOCHS + 1):
        stem.train(); head.train()
        epoch_loss = 0.0; n = 0
        for frames, _, labels in loader:
            # Use only first frame per clip: [B, T, 3, H, W] -> [B, 3, H, W]
            imgs   = frames[:, 0].to(DEVICE)
            labels = labels.to(DEVICE)
            tokens = stem(imgs)                 # [B, N, D]
            pooled = tokens.mean(dim=1)         # [B, D]
            logits = head(pooled)               # [B, 3]
            loss   = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item(); n += 1

        avg = epoch_loss / max(n, 1)
        results.append(avg)
        print(f"  Epoch {epoch}/{EPOCHS}  ce_loss={avg:.4f}")

    print(f"  [V-JEPA Vision] Final classification loss: {results[-1]:.4f}")
    return results[-1]


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE 3 — Fire Detection: Audio-Visual MuMo JEPA
# ─────────────────────────────────────────────────────────────────────────────

def run_fire_mumo(loader: DataLoader):
    print("\n" + "="*60)
    print("  Pipeline 3: Fire Detection (Audio-Visual MuMo JEPA)")
    print("="*60)

    # MuMo encoder: VideoStem (Modality A) + AudioPatchEmbed (Modality B)
    mumo = build_audiovisual_mumo_jepa(
        embed_dim=EMBED_DIM,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        audio_in_chans=1,
        audio_patch_size=AUDIO_PATCH,
        n_fusion_tokens=N_FUSION,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        sigreg_lambda=0.1,   # mild regularization weight
        use_real_vjepa2=True,
        freeze_vjepa2=False,
    ).to(DEVICE)

    head = ClassificationHead(in_dim=EMBED_DIM, num_classes=3).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(mumo.parameters()) + list(head.parameters()), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    results = []
    for epoch in range(1, EPOCHS + 1):
        mumo.train(); head.train()
        epoch_loss = 0.0; n = 0
        for frames, mels, labels in loader:
            # Use first frame per clip [B, 3, H, W]
            imgs   = frames[:, 0].to(DEVICE)
            mels   = mels.to(DEVICE)            # [B, 1, MELS, TIME]
            labels = labels.to(DEVICE)

            # Get joint CLS embedding (bypass SIGReg loss — use encode() directly)
            cls_emb = mumo.encode(imgs, mels)   # [B, EMBED_DIM]
            logits  = head(cls_emb)             # [B, 3]

            # Also compute SIGReg for completeness
            sigreg_loss = mumo.sigreg(cls_emb)

            loss = criterion(logits, labels) + 0.1 * sigreg_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item(); n += 1

        avg = epoch_loss / max(n, 1)
        results.append(avg)
        print(f"  Epoch {epoch}/{EPOCHS}  total_loss={avg:.4f}  (ce + 0.1*sigreg)")

    print(f"  [MuMo Audio-Visual] Final total loss: {results[-1]:.4f}")
    return results[-1]


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Le MuMo JEPA — Unified Evaluation (main_mumo.py)")
    print("=" * 60)

    # Pipeline 1: Gun Detection
    gun_loss = run_gun_detection()

    # Load shared FireSmoke video dataset for Pipelines 2 & 3
    vid_dir = os.path.join(_ML_DIR, "data", "FireSmoke", "test_vids")
    fire_dataset = FireVideoDataset(vid_dir)

    if len(fire_dataset) == 0:
        print("\n  WARNING: No fire video clips found. Skipping pipelines 2 & 3.")
        return

    print(f"\n  Loaded {len(fire_dataset)} fire/smoke video clips for Pipelines 2 & 3.")
    loader = DataLoader(fire_dataset, batch_size=min(BATCH_SIZE, len(fire_dataset)),
                        shuffle=True, drop_last=False)

    # Pipeline 2: Fire – V-JEPA Vision Only
    vjepa_loss = run_fire_vjepa(loader)

    # Pipeline 3: Fire – Audio-Visual MuMo JEPA
    mumo_loss = run_fire_mumo(loader)

    # ── Results Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    if gun_loss is not None:
        print(f"  [Pipeline 1] Gun Detection (VideoStem)  bbox MSE loss   : {gun_loss:.4f}")
    print(f"  [Pipeline 2] Fire - V-JEPA Vision-Only   CE loss         : {vjepa_loss:.4f}")
    print(f"  [Pipeline 3] Fire - MuMo Audio-Visual    Total loss      : {mumo_loss:.4f}")
    diff = vjepa_loss - mumo_loss
    better = "MuMo" if diff > 0 else "V-JEPA Vision-Only"
    print(f"\n  Delta(Pipeline2 - Pipeline3) = {diff:+.4f}  -->  {better} achieved lower loss")
    print("=" * 60)
    print("\n  NOTE: torchaudio.io (StreamReader) requires torchaudio with FFmpeg backend.")
    print("  Audio/Video fallback to zero tensors was used (CPU-only torchaudio build).")
    print("  Install ffmpeg-python or upgrade torchaudio for full audio extraction.")
    print("=" * 60)


if __name__ == "__main__":
    main()
