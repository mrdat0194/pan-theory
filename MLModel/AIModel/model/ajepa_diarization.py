import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class AttentiveStatsPooling(nn.Module):
    """
    Computes attentive statistics (weighted mean and standard deviation)
    over a sequence of representations.
    """
    def __init__(self, in_dim, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, x):
        # x: [B, T, D]
        # Calculate attention weights
        attn_logits = self.attention(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, T, 1]
        
        # Weighted mean
        mu = torch.sum(x * attn_weights, dim=1)  # [B, D]
        
        # Weighted standard deviation
        var = torch.sum((x - mu.unsqueeze(1))**2 * attn_weights, dim=1)
        sigma = torch.sqrt(var + 1e-8)  # [B, D]
        
        # Concatenate mean and std
        return torch.cat([mu, sigma], dim=1)  # [B, 2D]


class SpeakerProjectionHead(nn.Module):
    """
    Projects the pooled AJEPA representations to a speaker embedding space,
    using AM-Softmax / CosFace style formulation (Option A - Frozen Linear Probe).
    """
    def __init__(self, in_dim, embed_dim=256, num_classes=None, margin=0.35, scale=30.0):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.PReLU()
        )
        
        # If training with AM-Softmax, we need the class weights
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        if num_classes is not None:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
            nn.init.xavier_uniform_(self.weight)
            
    def forward(self, x, labels=None):
        # x: [B, 2D] (output of AttentiveStatsPooling)
        embeddings = self.projector(x) # [B, embed_dim]
        
        # L2 Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if labels is not None and self.num_classes is not None:
            # AM-Softmax loss logic
            cosine = F.linear(embeddings, F.normalize(self.weight, p=2, dim=1))
            # Create one-hot label
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
            
            # AM-Softmax: subtract margin from ground truth class
            phi = cosine - self.margin
            # Combine
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.scale
            return embeddings, output
            
        return embeddings


class AJEPAOverlappedSpeechHead(nn.Module):
    """
    Predicts multi-speaker overlap probability per frame.
    Multi-label sigmoid head.
    """
    def __init__(self, in_dim, num_speakers=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_speakers)
        )
        
    def forward(self, x):
        # x: [B, T, D]
        logits = self.classifier(x) # [B, T, num_speakers]
        probs = torch.sigmoid(logits)
        return probs


class AJEPAClusterDiarizer:
    """
    Executes Cosine Distance matrix calculation and Agglomerative Clustering
    on windowed speaker embeddings to output speaker turn clusters.
    """
    def __init__(self, max_speakers=None, distance_threshold=0.3):
        # We use cosine distance.
        if max_speakers is not None:
            self.clustering = AgglomerativeClustering(
                n_clusters=max_speakers,
                metric="precomputed",
                linkage="average"
            )
        else:
            self.clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="precomputed",
                linkage="average"
            )

    def compute_distance_matrix(self, embeddings):
        """
        embeddings: [N, D] where N is number of windows.
        Returns: [N, N] pairwise cosine distance matrix.
        """
        # Ensure L2 normalization
        eps = 1e-8
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + eps)
        
        cosine_sim = np.dot(embeddings_norm, embeddings_norm.T)
        cosine_dist = 1.0 - cosine_sim
        # Clip to avoid small negative distances due to numerical errors
        return np.clip(cosine_dist, 0.0, 2.0)

    def diarize(self, embeddings):
        """
        Clusters window embeddings and returns cluster labels.
        """
        # embeddings: numpy array [N, D]
        if embeddings.shape[0] == 0:
            return np.array([]), np.array([])
            
        if embeddings.shape[0] == 1:
            return np.array([0]), np.array([[0.0]])
            
        distance_matrix = self.compute_distance_matrix(embeddings)
        labels = self.clustering.fit_predict(distance_matrix)
        
        return labels, distance_matrix
