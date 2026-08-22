import torch
import sys
import os
import json
import numpy as np

# Ensure the parent directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.ajepa_backbone import AJEPA
from model.ajepa_diarization import AttentiveStatsPooling, SpeakerProjectionHead, AJEPAClusterDiarizer

def run_ajepa_diarization(audio_features_path=None):
    """
    Runs the custom Audio-JEPA Diarization Pipeline.
    """
    # 1. Initialize models
    ajepa = AJEPA(in_chans=1, patch_size=(40, 15))
    ajepa.eval()
    
    embed_dim = ajepa.embed_dim
    
    pooling = AttentiveStatsPooling(in_dim=embed_dim)
    pooling.eval()
    
    projector = SpeakerProjectionHead(in_dim=embed_dim * 2)
    projector.eval()
    
    diarizer = AJEPAClusterDiarizer(max_speakers=2)
    
    # 2. Mock input: [Batch, Channel, Freq, Time] -> e.g., 13 segments of audio windowed
    # 13 windows
    x_windows = [torch.randn(1, 1, 40, 150) for _ in range(13)]
    
    window_embeddings = []
    with torch.no_grad():
        for x in x_windows:
            # Get frame sequence [B, T, D]
            z_seq = ajepa.encode_sequence(x)
            
            # Pool to [B, 2D]
            pooled = pooling(z_seq)
            
            # Project to speaker space
            spk_emb = projector(pooled)
            
            window_embeddings.append(spk_emb.numpy()[0])
            
    embeddings = np.array(window_embeddings)
    
    # 3. Cluster
    labels, dist_matrix = diarizer.diarize(embeddings)
    
    # 4. Level 2 – Speaker Anchoring
    # Window 0 is Jeremy's long unambiguous opening monologue -> ground-truth: SPEAKER_01
    labels = anchor_cluster_labels(
        embeddings=embeddings,
        labels=labels,
        anchor_window_idx=0,
        anchor_target_label=1   # 1 -> SPEAKER_01 (Jeremy)
    )
    
    # Return mock aligned text to match Hamel format
    mock_texts = [
        "Hi, this is Jeremy Howard, and you're listening to Coffee Time Data Science, a podcast for data science enthusiasts, where I interview practitioners, researchers, and Kagglers about their journey, experience, and talk all things data science.",
        "And before we begin, I apologize for the change to our schedule.",
        "Of course, usually you would be seeing Chai Time Data Science on this channel with Sanyam Bhutani.",
        "Unfortunately, he's not available today.",
        "He had a prior appointment on another podcast, and he was not able to join Chai Time Data Science.",
        "We hope you enjoy this special episode of Coffee Time Data Science.",
        "And without further ado, I would like to invite our very special VIP guest, newly anointed Kaggle Grand Master, Sanyam Bhutani.",
        "Sanyam, welcome to Coffee Time Data Science.",
        "Thank you, Jeremy.",
        "Usually, I'm very anti coffee, but I'll have to allow that.",
        "I still can't believe you weren't kidding.",
        "And I mentioned in our message also, like I, I think I don't deserve this.",
        "But thank you."
    ]
    
    results = []
    starts = [0.248, 13.531, 17.151, 22.593, 24.373, 29.974, 34.338, 45.148, 48.372, 49.073, 53.537, 55.678, 59.421]
    ends = [13.531, 17.151, 22.593, 24.373, 29.514, 34.338, 45.148, 47.190, 49.073, 53.537, 55.678, 59.421, 60.042]
    
    for i, label in enumerate(labels):
        results.append({
            "start": starts[i],
            "end": ends[i],
            "speaker": f"SPEAKER_{label:02d}",
            "text": mock_texts[i]
        })
        
    return results


def anchor_cluster_labels(embeddings, labels, anchor_window_idx=0, anchor_target_label=1):
    """
    Level 2 – Speaker Anchoring.

    Orients the unsupervised cluster IDs so that the cluster whose centroid is
    closest (cosine similarity) to the *anchor* window embedding is assigned
    `anchor_target_label` (e.g. 1 -> SPEAKER_01).

    Args:
        embeddings:          np.ndarray [N, D] - one row per diarization window.
        labels:              np.ndarray [N]    - raw cluster IDs from AHC.
        anchor_window_idx:   int              - index of the window with known identity.
        anchor_target_label: int              - the cluster ID the anchor should map TO.

    Returns:
        remapped_labels: np.ndarray [N] with corrected speaker IDs.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        # Only one speaker detected – nothing to remap.
        return labels

    # L2-normalise all embeddings
    eps = 1e-8
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + eps)

    # Compute per-cluster centroids (in normalised space)
    centroids = {
        lbl: emb_norm[labels == lbl].mean(axis=0)
        for lbl in unique_labels
    }

    # Anchor reference vector (the known reference window)
    anchor_vec = emb_norm[anchor_window_idx]

    # Cosine similarity between anchor and each centroid
    sims = {lbl: float(np.dot(anchor_vec, c)) for lbl, c in centroids.items()}

    # The cluster most similar to anchor should become `anchor_target_label`
    anchor_cluster = max(sims, key=sims.get)
    other_cluster  = [l for l in unique_labels if l != anchor_cluster][0]
    other_target   = 1 - anchor_target_label  # the remaining target ID

    mapping = {anchor_cluster: anchor_target_label, other_cluster: other_target}

    remapped = np.array([mapping[l] for l in labels])
    print(f"[AnchorLabels] anchor window {anchor_window_idx}: "
          f"cluster {anchor_cluster} -> SPEAKER_{anchor_target_label:02d}, "
          f"cluster {other_cluster} -> SPEAKER_{other_target:02d} "
          f"| cosine sims: {sims}")
    return remapped

if __name__ == "__main__":
    res = run_ajepa_diarization()
    print("AJEPA Diarization Result:", json.dumps(res, indent=2))
