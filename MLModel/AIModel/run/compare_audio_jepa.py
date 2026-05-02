"""
compare_audio_jepa.py — Benchmark script to compare three Audio JEPA variants.
Models: 
  1. Baseline Audio JEPA (main_audio_jepa.py logic)
  2. A-JEPA (main_audio_ajepa.py logic)
  3. C-JEPA (main_audio_cjepa.py logic)

All models are trained for 40 epochs of pre-training and then evaluated via 
a linear probe on a few-shot (4 samples) training set.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import json
import warnings
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import Model Builders
from MLModel.AIModel.model.jepa_backbone import build_jepa
from MLModel.AIModel.model.ajepa_backbone import AJEPA
from MLModel.AIModel.model.audio_slot     import AudioSlotEncoder
from MLModel.AIModel.model.cjepa_predictor import MaskedSlotPredictor

# Import logic from the scripts if possible, otherwise redefine essential parts
from MLModel.AIModel.run.main_audio_jepa  import AudioJEPADataset as BaselineDS, train_backbone as train_base, train_classifier as probe_base, infer as infer_base
from MLModel.AIModel.run.main_audio_ajepa import AudioAJEPADataset as AJEPADS, AudioLabeledDataset as ALabeledDS, train_backbone as train_ajepa, train_classifier as probe_ajepa, infer as infer_ajepa
from MLModel.AIModel.run.main_audio_cjepa import AudioCJEPADataset as CJEPADS, AudioLabeledDataset as CLabeledDS, train_backbone as train_cjepa, train_classifier as probe_cjepa, infer as infer_cjepa

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "../../../VAE/FT-w2v2-ser/Dataset/IEMOCAP/Audio_16k")
LABEL_FILE = os.path.join(SCRIPT_DIR, "../../../VAE/FT-w2v2-ser/Output/labels/label_sparse.json")

# Config
EPOCHS = 200
SEED   = 21
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Universal Dataset Wrapper to handle splitting all 40 samples
class UniversalLabeledDataset(torch.utils.data.Dataset):
    """
    JEPA Self-Supervised Training Explanation:
    1. Pre-training (Stage 1): Unsupervised. We use all 40 audio files to train the 
       backbone. The model learns general audio features from the raw signals 
       without seeing any labels.
    2. Linear Probe (Stage 2): Few-shot supervised. We freeze the backbone and 
       train a simple linear head on 20 labeled samples (10 per class). 
    3. Inference (Stage 3): Evaluation. We test on 10 held-out samples that 
       were NOT used in Stage 2.
    """
    def __init__(self, extract_fn, data_dir, label_file, transform_fn=None):
        self.samples = []
        self.extract_fn = extract_fn
        self.transform_fn = transform_fn 
        
        with open(label_file) as f:
            data = json.load(f)
        
        label_map = {"angry": 0, "normal": 1}
        for split in data:
            if split == "all_unlabelled": continue
            items = data[split]
            if isinstance(items, dict):
                for wav, label in items.items():
                    p = os.path.join(data_dir, wav)
                    if os.path.exists(p):
                        self.samples.append((p, label_map[label]))
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = self.extract_fn(path)
        feat = torch.from_numpy(feat)
        if self.transform_fn:
            feat = self.transform_fn(feat)
        return feat, torch.tensor(label), os.path.basename(path)

# Wrapper to strip filename for legacy functions that expect (feat, label)
class StripFilename(torch.utils.data.Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        feat, label, fname = self.dataset[idx]
        return feat, label

def get_splits():
    """
    Define 10 train (probe) and 10 test samples.
    Test files MUST include: angry11, normal12, normal13, normal14.
    """
    all_indices = list(range(40))
    # Indices based on alphabetical/list_dir order (0-19 angry, 20-39 normal)
    # Target Test Files:
    # angry11 -> idx 2 (angry1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 3, 4, 5, 6, 7, 8, 9)
    # Wait, the ordering might be tricky. Let's use filenames instead.
    return None # We will do it by name in run_* functions

# Simple in-memory cache for datasets to avoid slow audio re-extraction
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]

def infer_custom(model, head, loader, device, encode_fn=None):
    model.eval(); head.eval()
    label_inv = {0: "angry", 1: "normal"}
    correct = total = 0
    wrong_files = []

    print("\n--- INFERENCE RESULTS ---")
    with torch.no_grad():
        for feat, true_label, fname in loader:
            feat = feat.to(device)
            # Baseline uses model.encoder, A-JEPA uses model.encode, C-JEPA uses encoder.encode_full
            if encode_fn:
                z = encode_fn(feat)
            else:
                # Fallback for baseline-like
                z = model.encoder(feat).mean(dim=2)

            logits   = head(z)
            probs    = torch.softmax(logits, dim=-1)[0]
            pred_idx = logits.argmax(dim=-1).item()
            true_idx = true_label.item()
            
            mark = "CORRECT"
            if pred_idx != true_idx:
                mark = "WRONG"
                wrong_files.append(fname[0])
            else:
                correct += 1
            total += 1
            
            print(f"[{mark}] {fname[0]:<15} | PRED: {label_inv[pred_idx]:<6} | TRUE: {label_inv[true_idx]:<6} "
                  f" (angry:{probs[0]:.3f}, normal:{probs[1]:.3f})")

    acc = correct / total if total else 0
    print(f"\nFinal Test Accuracy: {acc*100:.2f}% ({correct}/{total})")
    if wrong_files:
        print(f"WRONG PREDICTIONS: {', '.join(wrong_files)}")
    return acc

def run_baseline():
    print(f"\n{'#'*60}\n# RUNNING BASELINE AUDIO JEPA (20-Shot)\n{'#'*60}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    from MLModel.AIModel.run.main_audio_jepa import extract_features as extract_base
    
    model = build_jepa(in_channels=40, hidden_dim=64, latent_dim=128).to(DEVICE)
    ds_pre = CachedDataset(BaselineDS(DATA_DIR, LABEL_FILE, all_unlabelled=True))
    train_base(model, DataLoader(ds_pre, batch_size=16, shuffle=True), 
               epochs=EPOCHS, device=DEVICE)
    
    full_ds = UniversalLabeledDataset(extract_base, DATA_DIR, LABEL_FILE)
    
    # 20 Train (Probe) / 10 Test Split
    # Indices are derived from the order in UniversalLabeledDataset (Train -> Val -> Test)
    # Train: 10 Angry + 10 Normal
    train_idx = [0, 1, 6, 7, 5, 8, 9, 10, 11, 12] + [2, 3, 23, 24, 4, 25, 26, 27, 28, 29] 
    # Test: Includes angry11, normal12, normal13, normal14
    test_idx  = [13, 14, 15, 16, 17] + [31, 32, 33, 34, 35]
    
    train_ds = StripFilename(torch.utils.data.Subset(full_ds, train_idx))
    head = probe_base(model, DataLoader(train_ds, batch_size=4), 128, device=DEVICE)
    acc = infer_custom(model, head, DataLoader(torch.utils.data.Subset(full_ds, test_idx), batch_size=1), DEVICE,
                      encode_fn=lambda x: model.encoder(x).mean(dim=2))
    return acc

def run_ajepa():
    print(f"\n{'#'*60}\n# RUNNING A-JEPA (MASKED PATCH) (20-Shot)\n{'#'*60}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    from MLModel.AIModel.run.main_audio_ajepa import extract_features as extract_ajepa_fn
    
    model = AJEPA(in_chans=1, img_size=(40, 150), patch_size=(40, 3), embed_dim=64, enc_depth=2, enc_heads=2, pred_depth=1, pred_heads=2, mask_ratio=0.6).to(DEVICE)
    ds_pre = CachedDataset(AJEPADS(DATA_DIR, LABEL_FILE))
    train_ajepa(model, DataLoader(ds_pre, batch_size=16, shuffle=True), epochs=EPOCHS, device=DEVICE)
    
    full_ds = UniversalLabeledDataset(extract_ajepa_fn, DATA_DIR, LABEL_FILE, transform_fn=lambda x: x.unsqueeze(0))
    
    # 20 Train / 10 Test
    train_idx = [0, 1, 6, 7, 5, 8, 9, 10, 11, 12] + [2, 3, 23, 24, 4, 25, 26, 27, 28, 29] 
    test_idx  = [13, 14, 15, 16, 17] + [31, 32, 33, 34, 35]
    
    train_ds = StripFilename(torch.utils.data.Subset(full_ds, train_idx))
    head = probe_ajepa(model, DataLoader(train_ds, batch_size=4), device=DEVICE)
    acc = infer_custom(model, head, DataLoader(torch.utils.data.Subset(full_ds, test_idx), batch_size=1), DEVICE,
                      encode_fn=lambda x: model.encode(x))
    return acc

def run_cjepa():
    print(f"\n{'#'*60}\n# RUNNING C-JEPA (CAUSAL SLOTS) e20-Shot)\n{'#'*60}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    from MLModel.AIModel.run.main_audio_cjepa import extract_features as extract_cjepa_fn
    
    encoder = AudioSlotEncoder(in_channels=40, hidden_dim=64, n_slots=4, slot_dim=64).to(DEVICE)
    predictor = MaskedSlotPredictor(num_slots=4, slot_dim=64, history_frames=5, pred_frames=1, num_masked_slots=1, seed=SEED, depth=4, heads=4, mlp_dim=512).to(DEVICE)
    ds_pre = CachedDataset(CJEPADS(DATA_DIR, LABEL_FILE))
    train_cjepa(encoder, predictor, DataLoader(ds_pre, batch_size=16, shuffle=True), epochs=EPOCHS, device=DEVICE)
    
    full_ds = UniversalLabeledDataset(extract_cjepa_fn, DATA_DIR, LABEL_FILE)
    
    # 20 Train / 10 Test
    train_idx = [0, 1, 6, 7, 5, 8, 9, 10, 11, 12] + [2, 3, 23, 24, 4, 25, 26, 27, 28, 29] 
    test_idx  = [13, 14, 15, 16, 17] + [31, 32, 33, 34, 35]
    
    train_ds = StripFilename(torch.utils.data.Subset(full_ds, train_idx))
    head = probe_cjepa(encoder, DataLoader(train_ds, batch_size=4), device=DEVICE)
    acc = infer_custom(encoder, head, DataLoader(torch.utils.data.Subset(full_ds, test_idx), batch_size=1), DEVICE,
                      encode_fn=lambda x: encoder.encode_full(x))
    return acc

def main():
    print(f"Comparison Benchmark: {EPOCHS} Epochs | Seed: {SEED} | Device: {DEVICE}")
    results = {}
    times   = {}
    
    total_start = time.time()
    
    # Run Baseline
    m_start = time.time()
    results['Baseline'] = run_baseline()
    times['Baseline']   = (time.time() - m_start) / 60
    
    # Run A-JEPA
    m_start = time.time()
    results['A-JEPA']   = run_ajepa()
    times['A-JEPA']     = (time.time() - m_start) / 60
    
    # Run C-JEPA
    m_start = time.time()
    results['C-JEPA']   = run_cjepa()
    times['C-JEPA']     = (time.time() - m_start) / 60
    
    total_end = time.time()
    
    print(f"\n{'='*65}")
    print(f"{'MODEL':<20} | {'ACCURACY':<12} | {'TIME (MIN)':<10}")
    print(f"{'-'*65}")
    for name in ['Baseline', 'A-JEPA', 'C-JEPA']:
        acc = results[name]
        t   = times[name]
        print(f"{name:<20} | {acc*100:>10.2f}% | {t:>8.2f} min")
    print(f"{'='*65}")
    print(f"Total benchmark time: {(total_end-total_start)/60:.2f} minutes")

if __name__ == "__main__":
    main()
