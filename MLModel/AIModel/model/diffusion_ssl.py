"""
diffusion_ssl.py
================
Diffusion-Based Self-Supervised Representation Learning (Option B).

Mathematical Context
--------------------
Standard diffusion models (DDPM, Score SDE) learn to reverse a noise process.
The FORWARD process gradually adds Gaussian noise to data x_0:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    
    where epsilon ~ N(0, I)  and  alpha_bar_t = prod_{s=1}^{t} (1 - beta_s)

The BACKWARD process (reverse ODE / Probability Flow ODE):
    dx = [-1/2 * beta(t) * x - beta(t) * score(x, t)] dt

where score(x, t) = nabla_x log p_t(x) is the SCORE FUNCTION.

The SSL Insight:
    Once the SCORE NETWORK s_theta(x_t, t) ≈ score(x, t) is trained, we
    DISCARD the generative head and instead use the INTERMEDIATE FEATURES
    of the score network as robust, semantically rich representations.
    
    These features serve as the encoder backbone for downstream tasks 
    (classification, anomaly detection, segmentation) — the same way a 
    pre-trained JEPA or ImageNet model is fine-tuned.

Architecture
------------
    x_0 (clean data) [B, C, T]
         │
    Forward ODE: add noise at level t  →  x_t [B, C, T]
         │
    ScoreNet(x_t, t)              ← Transformer + time embedding
         ├── score output: ε_pred  ← used for SSL pre-training loss
         └── forward_features()   ← used for downstream SSL tasks
         │
    SSL Loss: MSE(ε_pred, ε_true) ← noise-prediction = score matching

Relationship to Latent ODE JEPA (Option A):
    Both use ODEs as the modelling backbone.
    Option A: ODE operates in LATENT SPACE (low-dim z).
    Option B: ODE operates in DATA SPACE (full-dim x), adding/removing noise.

Usage
-----
    # Pre-train the score network (SSL objective)
    from MLModel.AIModel.model.diffusion_ssl import build_diffusion_ssl, DiffusionSSL
    model = build_diffusion_ssl(in_channels=2, seq_len=150)
    loss = model.compute_loss(x_batch)

    # Extract features for a downstream task
    feats = model.forward_features(x_batch, t_level=0.3)  # shape [B, D]
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. Noise Schedule — defines the forward ODE (how fast noise is added)
# ─────────────────────────────────────────────────────────────────────────────

class NoiseSchedule:
    """
    Linear beta schedule for the forward diffusion process.

    Controls the Signal-to-Noise ratio at each diffusion step t ∈ [0, T].
        beta(t): variance added per step
        alpha_bar(t): cumulative product of (1 - beta_s), giving the
                      fraction of the original signal remaining at step t.

    At t=0: x_t = x_0 (no noise)
    At t=T: x_t ≈ N(0, I) (pure noise)
    """
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)          # [T]
        alphas = 1.0 - betas                                      # [T]
        self.alpha_bar = torch.cumprod(alphas, dim=0)             # [T]: cumulative product

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Sample x_t from the forward process q(x_t | x_0) at diffusion step t.
        
        Closed-form: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Args:
            x0: clean data [B, C, T_seq]
            t:  integer diffusion step indices [B], each in [0, T-1]
        Returns:
            (x_t, epsilon): noisy sample and the exact noise added
        """
        alpha_bar_t = self.alpha_bar[t].to(x0.device)            # [B]
        # Reshape for broadcasting over [B, C, T_seq]
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1)

        epsilon = torch.randn_like(x0)                            # N(0, I)
        x_t = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * epsilon

        return x_t, epsilon


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sinusoidal Time Embedding — encodes the diffusion step t as a feature vector
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Encode the scalar diffusion step t ∈ [0, T] into a feature vector using
    sinusoidal positional encodings (identical to Transformer positional encodings).

    This allows the ScoreNet to condition its predictions on the noise level t.
    The embedding is then projected to the model dimension via a small MLP.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Project the sinusoidal embedding to the full embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: diffusion steps [B], values in [0, T-1]
        Returns:
            embedding: [B, embed_dim]
        """
        device = t.device
        half_dim = self.embed_dim // 2
        # Frequencies: 1 / 10000^(2i/embed_dim)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )
        # Outer product: [B, half_dim]
        angles = t[:, None].float() * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)   # [B, embed_dim]
        return self.proj(emb)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Score Network — learns s_theta(x_t, t) ≈ −epsilon / sqrt(1 − alpha_bar_t)
# ─────────────────────────────────────────────────────────────────────────────

class ScoreNet(nn.Module):
    """
    The core neural network of the Diffusion-SSL model.

    This network takes a noisy time-series x_t and the diffusion step t,
    and predicts the noise epsilon that was added (equivalent to predicting
    the score function nabla log p_t(x)).

    Architecture:
        - 1D Conv stem to map raw channels to model dimension
        - Sinusoidal time embedding injected at each Transformer layer
        - A shallow Transformer encoder for temporal context
        - Output head mapping back to signal space (noise prediction)

    The key method for SSL is `forward_features()`:
        After pre-training on score-matching, we freeze the model and
        extract the penultimate Transformer hidden states as feature vectors.
        These are semantically rich representations learned without labels.
    """
    def __init__(self, in_channels: int, seq_len: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 4):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Stem: project raw signal channels to model dimension
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        self.input_norm = nn.LayerNorm(d_model)

        # Time embedding: encode diffusion step t → feature vector
        self.time_embed = SinusoidalTimeEmbedding(d_model)

        # Transformer encoder layers (each layer receives time embedding as bias)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.0
            )
            for _ in range(num_layers)
        ])

        # Time-conditioning projection per layer (adds t-embedding to residual stream)
        self.time_proj = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_layers)
        ])

        # Output head: map hidden states back to signal space for noise prediction
        self.output_head = nn.Conv1d(d_model, in_channels, kernel_size=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise epsilon that was added to x_0 to produce x_t.
        
        Args:
            x_t: noisy input [B, C, T_seq]
            t:   diffusion step indices [B]
        Returns:
            epsilon_pred: predicted noise [B, C, T_seq]
        """
        # Project input channels to model dimension: [B, d_model, T_seq]
        h = self.input_proj(x_t)
        # Transpose for Transformer: [B, T_seq, d_model]
        h = h.transpose(1, 2)
        h = self.input_norm(h)

        # Time embedding: [B, d_model]
        t_emb = self.time_embed(t)

        # Pass through Transformer layers, injecting t at each layer
        for layer, t_proj in zip(self.transformer_layers, self.time_proj):
            # Broadcast time embedding over sequence: [B, 1, d_model] → [B, T, d_model]
            h = h + t_proj(t_emb).unsqueeze(1)
            h = layer(h)

        # Store penultimate features before the output head (for SSL feature extraction)
        self._last_hidden = h  # [B, T_seq, d_model]

        # Output head: predict noise epsilon in signal space
        # Transpose back: [B, d_model, T_seq]
        epsilon_pred = self.output_head(h.transpose(1, 2))  # [B, C, T_seq]
        return epsilon_pred

    def forward_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        SSL Feature Extraction — the core of Diffusion-SSL.

        Run the forward pass (score prediction), then return the mean-pooled
        penultimate Transformer hidden states instead of the noise prediction.

        These features are the "JEPA equivalent" of Diffusion-SSL:
            - They encode what makes the signal unique at noise level t.
            - They are robust to noise (by design — the model was trained to
              de-noise signals).
            - No labels required for pre-training.

        Args:
            x: input signal [B, C, T_seq]
            t: diffusion time level (integer tensor [B] or float noise fraction)
        Returns:
            features: mean-pooled hidden states [B, d_model]
        """
        with torch.no_grad():
            self.forward(x, t)         # populates self._last_hidden
        # Mean pool over the time dimension: [B, T_seq, d_model] → [B, d_model]
        return self._last_hidden.mean(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DiffusionSSL — wraps the ScoreNet with the noise schedule and SSL loss
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionSSL(nn.Module):
    """
    Full Diffusion-Based Self-Supervised Learning model.

    Pre-training objective:
        For each batch x_0:
          1. Sample random diffusion steps t ~ Uniform[1, T]
          2. Add noise: x_t = forward_ODE(x_0, t)
          3. Predict the noise: epsilon_pred = score_net(x_t, t)
          4. Minimize MSE(epsilon_pred, epsilon_true)

    Once trained, call `forward_features(x, t_noise_level)` to extract
    representations for downstream SSL tasks, discarding the output head.
    """
    def __init__(self, score_net: ScoreNet, schedule: NoiseSchedule):
        super().__init__()
        self.score_net = score_net
        self.schedule = schedule

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute the Diffusion-SSL (score-matching) loss for a batch.

        Process:
            1. Sample random t for each sample in the batch
            2. Add noise using the forward ODE (closed-form)
            3. Predict the noise with the score network
            4. Return MSE(predicted_noise, true_noise)

        Args:
            x0: clean batch [B, C, T_seq]
        Returns:
            loss: scalar MSE loss
        """
        B = x0.shape[0]

        # Step 1: Sample random diffusion steps t ∈ [1, T] for each sample
        t = torch.randint(1, self.schedule.T, (B,), device=x0.device)

        # Step 2: Apply forward ODE — add noise to x0 at level t
        x_t, epsilon_true = self.schedule.q_sample(x0, t)

        # Step 3: Predict the noise that was added
        epsilon_pred = self.score_net(x_t, t)

        # Step 4: Score-matching loss (equivalent to noise-prediction MSE)
        loss = torch.nn.functional.mse_loss(epsilon_pred, epsilon_true)
        return loss

    def forward_features(self, x: torch.Tensor, t_fraction: float = 0.3) -> torch.Tensor:
        """
        Extract SSL features from the penultimate Transformer hidden states.

        Args:
            x:            input signal [B, C, T_seq]
            t_fraction:   noise level as a fraction of T (e.g. 0.3 = 30% noise).
                          Lower t = cleaner signal, features focus on fine details.
                          Higher t = more noise, features focus on coarse structure.
        Returns:
            features: [B, d_model] representation vectors
        """
        B = x.shape[0]
        t_int = int(t_fraction * self.schedule.T)
        t = torch.full((B,), t_int, dtype=torch.long, device=x.device)

        # Add noise at the requested level first
        x_t, _ = self.schedule.q_sample(x, t)

        # Extract features from the score network's hidden states
        return self.score_net.forward_features(x_t, t)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_diffusion_ssl(in_channels: int = 2, seq_len: int = 150,
                         d_model: int = 128, nhead: int = 4,
                         num_layers: int = 4, T: int = 1000) -> DiffusionSSL:
    """
    Build the Diffusion-SSL model for time-series representation learning.

    Args:
        in_channels: number of signal channels (e.g. 2 for acc + severity)
        seq_len:     length of the time series window
        d_model:     Transformer model dimension
        nhead:       number of attention heads
        num_layers:  number of Transformer encoder layers
        T:           total diffusion steps in the noise schedule
    """
    schedule  = NoiseSchedule(T=T)
    score_net = ScoreNet(in_channels, seq_len, d_model, nhead, num_layers)
    return DiffusionSSL(score_net, schedule)


if __name__ == "__main__":
    print("=" * 60)
    print(" Diffusion-SSL: Self-Supervised Smoke Test")
    print("=" * 60)

    model = build_diffusion_ssl(in_channels=2, seq_len=150, d_model=64, num_layers=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel built. Params: {total_params:,}")

    x = torch.randn(4, 2, 150)   # batch of 4 signals, 2 channels, 150 time steps

    # Pre-training forward pass
    loss = model.compute_loss(x)
    print(f"SSL Loss (score-matching MSE): {loss.item():.4f}")

    # Feature extraction (the SSL downstream representation)
    feats = model.forward_features(x, t_fraction=0.3)
    print(f"Extracted feature shape: {feats.shape}")   # [4, 64]
    print("\nSUCCESS: Diffusion-SSL model runs correctly.")
