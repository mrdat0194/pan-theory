"""
Audio Slot Encoder for C-JEPA-style self-supervised training.

Maps windowed MFCC features into N_slots object-centric slot representations
per time step, analogous to visual object slots in C-JEPA.
"""
import torch
import torch.nn as nn


class Encoder1D(nn.Module):
    """Shared 1D CNN backbone. [B, C_in, T] -> [B, C_out, T]"""

    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioSlotEncoder(nn.Module):
    """
    Encodes windowed MFCC audio into N_slots slot representations per time step.

    The CNN outputs N_slots * slot_dim channels, which are mean-pooled over T_frame
    and reshaped into [B, N_slots, slot_dim].

    Usage:
      - Pre-training :  forward(x)        x: [B, T_hist, C, T_frame] -> [B, T_hist, N_slots, slot_dim]
      - Linear probe :  encode_full(x)    x: [B, C, T]               -> [B, slot_dim]  (mean over slots)
    """

    def __init__(self, in_channels: int, hidden_dim: int, n_slots: int, slot_dim: int):
        super().__init__()
        self.n_slots   = n_slots
        self.slot_dim  = slot_dim
        # wider encoder so that we can split output into N independent slots
        self.encoder   = Encoder1D(in_channels, hidden_dim, n_slots * slot_dim)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _encode_window(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T_frame] -> [B, N_slots, slot_dim]"""
        B = x.shape[0]
        z = self.encoder(x)          # [B, N_slots*slot_dim, T_frame]
        z = z.mean(dim=-1)           # [B, N_slots*slot_dim]
        return z.view(B, self.n_slots, self.slot_dim)

    # ── public API ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T_hist, C, T_frame] -> [B, T_hist, N_slots, slot_dim]"""
        B, T, C, T_frame = x.shape
        z = self._encode_window(x.view(B * T, C, T_frame))   # [B*T, N, D]
        return z.view(B, T, self.n_slots, self.slot_dim)

    def encode_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a full (non-windowed) audio sample for linear probing / inference.
        [B, C, T] -> [B, slot_dim]   (mean-pool over T then over slots)
        """
        B = x.shape[0]
        z = self.encoder(x)            # [B, N_slots*slot_dim, T]
        z = z.mean(dim=-1)             # [B, N_slots*slot_dim]
        z = z.view(B, self.n_slots, self.slot_dim)
        return z.mean(dim=1)           # [B, slot_dim]
