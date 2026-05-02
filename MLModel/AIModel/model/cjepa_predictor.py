"""
C-JEPA Predictor — adapted for 1-D audio slot representations.

Ported from galilai-group/cjepa (src/cjepa_predictor.py, Apache-2.0 License).
Authors: Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, Randall Balestriero.
Paper : https://arxiv.org/abs/2602.11389

Adaptations for audio:
  - Lighter default hyper-params (depth=4, heads=4, mlp_dim=512)
  - Action-conditioning variant removed (not needed for audio SER)
  - Input/output shapes use slot_dim=64 by default
"""
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


# ── Building block ────────────────────────────────────────────────────────────

class NonCausalTransformer(nn.Module):
    """Standard Transformer Encoder with full (non-causal) self-attention."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout),
                ),
            ]))

    def forward(self, x: torch.Tensor,
                return_attention: bool = False):
        attn_list = [] if return_attention else None
        for attn, ff in self.layers:
            if return_attention:
                out, w = attn(x, x, x, need_weights=True, average_attn_weights=True)
                attn_list.append(w)
            else:
                out, _ = attn(x, x, x)
            x = x + out
            x = x + ff(x)
        x = self.norm(x)
        return (x, attn_list) if return_attention else x


# ── C-JEPA core ───────────────────────────────────────────────────────────────

class MaskedSlotPredictor(nn.Module):
    """
    Object-level masked slot predictor (C-JEPA).

    For each mini-batch:
      1. Randomly choose num_masked_slots "objects" to hide.
      2. Replace those slots with mask-token + time-PE + id-anchor queries.
      3. Run Non-Causal Transformer over all (visible + masked + future) tokens.
      4. Return predictions for every position; caller computes MSE loss on
         masked positions.

    Args:
        num_slots       : total slots per time step (N)
        slot_dim        : feature dimension per slot  (D)
        history_frames  : number of visible (history) frames   (T_hist)
        pred_frames     : number of future frames to predict    (T_pred)
        num_masked_slots: how many history slots to mask        (K)
        seed            : RNG seed for reproducible masking
    """

    def __init__(
        self,
        num_slots       : int,
        slot_dim        : int  = 64,
        history_frames  : int  = 5,
        pred_frames     : int  = 1,
        num_masked_slots: int  = 1,
        seed            : int  = 42,
        depth           : int  = 4,
        heads           : int  = 4,
        dim_head        : int  = 64,
        mlp_dim         : int  = 512,
        dropout         : float = 0.1,
    ):
        super().__init__()
        self.num_slots        = num_slots
        self.slot_dim         = slot_dim
        self.history_frames   = history_frames
        self.pred_frames      = pred_frames
        self.total_frames     = history_frames + pred_frames
        self.num_masked_slots = num_masked_slots
        self.seed             = seed

        # learnable mask token — stands in for any hidden slot
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # time positional embedding: one per (total) frame, shared across slots
        self.time_pos_embed = nn.Parameter(
            torch.randn(1, self.total_frames, 1, slot_dim))

        # identity projector: anchors the masked slot to its t=0 state
        self.id_projector  = nn.Linear(slot_dim, slot_dim)

        # transformer backbone
        self.transformer   = NonCausalTransformer(
            dim=slot_dim, depth=depth, heads=heads,
            dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)

        # output projection
        self.to_out        = nn.Linear(slot_dim, slot_dim)

    # ── masking helpers ───────────────────────────────────────────────────────

    def _get_mask_indices(self, device: torch.device):
        # Use stochastic selection to improve representation learning
        idx = np.random.choice(self.num_slots, self.num_masked_slots, replace=False)
        is_masked = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
        is_masked[idx] = True
        return is_masked, torch.from_numpy(idx).to(device)

    def _prepare_input(self, x: torch.Tensor):
        """
        Build the full token grid that is fed to the Transformer.

        x: [B, T_hist, S, D]
        returns: full_input [B, T_total, S, D], masked_indices [K]
        """
        B, T_hist, S, D = x.shape
        T_total  = self.total_frames
        device   = x.device

        # ── masking ──
        if self.num_masked_slots > 0:
            is_masked, masked_idx = self._get_mask_indices(device)
        else:
            is_masked  = torch.zeros(S, dtype=torch.bool, device=device)
            masked_idx = torch.tensor([], dtype=torch.long, device=device)

        # ── anchor queries (t=0 identity for every slot) ──
        anchors       = x[:, 0]                          # [B, S, D]
        anchor_q      = self.id_projector(anchors)        # [B, S, D]

        # ── base query grid = mask_token + timePE + anchorQuery ──
        tok_grid    = self.mask_token.expand(B, T_total, S, D)
        pos_grid    = self.time_pos_embed.expand(B, T_total, S, D)
        anc_grid    = anchor_q.unsqueeze(1).expand(B, T_total, S, D)
        query_input = tok_grid + pos_grid + anc_grid      # [B, T_total, S, D]

        final_input = query_input.clone()

        # t=0: always real data for all slots (identity anchor)
        final_input[:, 0] = x[:, 0] + self.time_pos_embed[:, 0]

        # t=1..T_hist-1: real data only for *unmasked* slots
        unmasked_idx = torch.where(~is_masked)[0]
        if len(unmasked_idx) > 0 and T_hist > 1:
            real_hist  = x[:, 1:, unmasked_idx]           # [B, T_hist-1, K', D]
            hist_pe    = self.time_pos_embed[:, 1:T_hist, :, :].expand(B, T_hist-1, S, D)
            hist_pe_um = hist_pe[:, :, unmasked_idx]
            final_input[:, 1:T_hist, unmasked_idx] = real_hist + hist_pe_um

        return final_input, masked_idx

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        x: [B, T_hist, S, D]
        returns:
          pred          : [B, T_total, S, D]
          masked_indices: [K]  (which slot indices were hidden)
        """
        B, T_hist, S, D = x.shape
        x_input, masked_idx = self._prepare_input(x)       # [B, T_total, S, D]
        x_flat  = rearrange(x_input, 'b t s d -> b (t s) d')
        out_flat = self.transformer(x_flat)
        out      = rearrange(out_flat, 'b (t s) d -> b t s d',
                             t=self.total_frames, s=S)
        return self.to_out(out), masked_idx

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rollout: given full history, predict T_pred future frames (no masking).
        x: [B, T_hist, S, D] -> [B, T_pred, S, D]
        """
        B, T_hist, S, D = x.shape
        T_pred  = self.pred_frames
        T_total = T_hist + T_pred
        pe      = self.time_pos_embed[:, -T_total:]         # reuse last T_total PEs

        anchors  = x[:, 0]
        anchor_q = self.id_projector(anchors)

        hist_in  = x + pe[:, :T_hist]                      # real history
        tok_f    = self.mask_token.expand(B, T_pred, S, D)
        pos_f    = pe[:, T_hist:].expand(B, T_pred, S, D)
        anc_f    = anchor_q.unsqueeze(1).expand(B, T_pred, S, D)
        fut_in   = tok_f + pos_f + anc_f

        full_in  = torch.cat([hist_in, fut_in], dim=1)     # [B, T_total, S, D]
        x_flat   = rearrange(full_in, 'b t s d -> b (t s) d')
        out_flat = self.transformer(x_flat)
        out      = rearrange(out_flat, 'b (t s) d -> b t s d', t=T_total, s=S)
        return self.to_out(out)[:, T_hist:]
