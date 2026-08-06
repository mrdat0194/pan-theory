"""
Le MuMo JEPA - Multimodal Wrapper for Pan-Theory Audio/1D modalities.

This module bridges existing unimodal JEPAs to the Le MuMo JEPA
fusion-token architecture (arXiv:2603.24327).

Architecture overview (per forward pass):
    Modality A (e.g. Audio spectrogram) -> AudioPatchEmbed -> [B, N, D] tokens
    Modality B (e.g. 1D sequence/actions) -> SequenceStem  -> [B, T, D] tokens
    Both token sequences -> Learnable Fusion Tokens (cross-attention bottleneck)
    Fusion tokens (after pruning modality tokens) -> Shared Transformer -> CLS
    CLS embedding -> SIGReg (via lejepa) for self-supervised regularization

Unchanged by this module:
    - jepa_backbone.py  / main_audio_jepa.py  (90.00 pct IEMOCAP baseline)
    - ajepa_backbone.py / main_audio_ajepa.py (masked patch A-JEPA)
    - cjepa_predictor.py + audio_slot.py / main_audio_cjepa.py (C-JEPA)
    - eb_jepa/ / main_jepa_anomaly.py (MTS-JEPA anomaly detection)
    - All other run scripts

Usage::

    from model.mumo_wrapper import build_audio_mumo_jepa, MuMoJEPAWrapper

    model = build_audio_mumo_jepa(
        audio_in_chans=1,
        seq_in_channels=6,
        embed_dim=256,
        n_fusion_tokens=16,
        n_layers=4,
        n_heads=8,
        sigreg_lambda=1.0,
    )
    loss, (pred_loss, sigreg_loss) = model(audio_spec, seq_data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lejepa

from .ajepa_backbone import AudioPatchEmbed
from .jepa_backbone import SequenceStem


class FusionTransformerBlock(nn.Module):
    """Standard Pre-Norm Transformer block for the shared trunk."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CrossModalFusionLayer(nn.Module):
    """
    Layer 0 of Le MuMo JEPA (pruned fusion strategy, arXiv:2603.24327).

    Fusion tokens query both modality A and B tokens, absorbing cross-modal
    information into a compact bottleneck. After this layer, modality-specific
    tokens are discarded (pruned), reducing attention cost in subsequent layers.
    """

    def __init__(self, dim: int, n_heads: int, drop: float = 0.0):
        super().__init__()
        self.norm_fusion = nn.LayerNorm(dim)
        self.norm_mod_a  = nn.LayerNorm(dim)
        self.norm_mod_b  = nn.LayerNorm(dim)
        self.cross_attn_a = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.cross_attn_b = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        # Learnable blend gate between modality contributions
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        fusion: torch.Tensor,  # [B, N_f, D]
        mod_a: torch.Tensor,   # [B, N_a, D]  (audio patch tokens)
        mod_b: torch.Tensor,   # [B, N_b, D]  (1D sequence tokens)
    ) -> torch.Tensor:
        """Returns updated fusion tokens [B, N_f, D]."""
        q_f = self.norm_fusion(fusion)
        ka  = self.norm_mod_a(mod_a)
        kb  = self.norm_mod_b(mod_b)
        out_a, _ = self.cross_attn_a(q_f, ka, ka, need_weights=False)
        out_b, _ = self.cross_attn_b(q_f, kb, kb, need_weights=False)
        gate = self.gate.sigmoid()
        return self.norm_out(fusion + gate * out_a + (1.0 - gate) * out_b)


class MuMoJEPAWrapper(nn.Module):
    """
    Pan-Theory Le MuMo JEPA: Dual-modality JEPA with Learnable Fusion Tokens + SIGReg.

    Pruned-fusion strategy (arXiv:2603.24327):
    1. Modality-specific stems tokenize inputs.
    2. CrossModalFusionLayer (Layer 0) merges both modalities into fusion tokens.
    3. Modality tokens pruned; only [CLS + fusion_tokens] continue in shared transformer.
    4. Joint CLS embedding regularized by SIGReg (lejepa).

    Args:
        stem_a (nn.Module):    Modality A stem (AudioPatchEmbed).
                               Output: [B, N_a, embed_dim].
        stem_b (nn.Module):    Modality B stem (SequenceStem / Encoder1D).
                               Output: [B, embed_dim, T] (transposed internally).
        embed_dim (int):       Shared embedding dimension.
        n_fusion_tokens (int): Number of learnable fusion tokens.
        n_layers (int):        Shared transformer depth after fusion.
        n_heads (int):         Attention heads.
        sigreg_lambda (float): SIGReg loss weight.
    """

    def __init__(
        self,
        stem_a: nn.Module,
        stem_b: nn.Module,
        embed_dim: int = 256,
        n_fusion_tokens: int = 16,
        n_layers: int = 4,
        n_heads: int = 8,
        sigreg_lambda: float = 1.0,
    ):
        super().__init__()
        self.stem_a = stem_a
        self.stem_b = stem_b
        self.embed_dim = embed_dim
        self.sigreg_lambda = sigreg_lambda

        # Learnable fusion tokens (cross-modal latent bottleneck)
        self.fusion_tokens    = nn.Parameter(torch.zeros(1, n_fusion_tokens, embed_dim))
        self.fusion_pos_embed = nn.Parameter(torch.zeros(1, n_fusion_tokens, embed_dim))
        self.cls_token        = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.fusion_tokens,    std=0.02)
        nn.init.trunc_normal_(self.fusion_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token,         std=0.02)

        # Layer 0: cross-modal attention into fusion tokens
        self.fusion_layer = CrossModalFusionLayer(embed_dim, n_heads)

        # Layers 1+: shared transformer on [CLS + fusion_tokens] only
        self.transformer = nn.ModuleList([
            FusionTransformerBlock(embed_dim, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # LeJEPA SIGReg on joint multimodal CLS embedding
        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test, num_slices=1024
        )

    def encode(self, mod_a_input: torch.Tensor, mod_b_input: torch.Tensor) -> torch.Tensor:
        """
        Encode two modalities and return the joint CLS embedding [B, embed_dim].

        Args:
            mod_a_input: Modality A input (e.g. spectrogram [B, 1, H, W]).
            mod_b_input: Modality B input (e.g. 1D sequence [B, C, T]).

        Returns:
            cls_emb: [B, embed_dim] joint multimodal representation.
        """
        B = mod_a_input.shape[0]

        # 1. Tokenize each modality
        tokens_a = self.stem_a(mod_a_input)   # [B, N_a, D]
        tokens_b = self.stem_b(mod_b_input)   # [B, D, T] from Conv1D
        if tokens_b.dim() == 3 and tokens_b.shape[1] == self.embed_dim:
            tokens_b = tokens_b.transpose(1, 2)  # [B, T, D]

        # 2. Expand learnable fusion tokens with positional embeddings
        fusion = self.fusion_tokens.expand(B, -1, -1) + self.fusion_pos_embed

        # 3. Layer 0: fuse both modalities into fusion tokens, then prune modality tokens
        fusion = self.fusion_layer(fusion, tokens_a, tokens_b)

        # 4. Shared transformer: [CLS + fusion_tokens] only
        x = torch.cat([self.cls_token.expand(B, -1, -1), fusion], dim=1)
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)

        return x[:, 0]  # CLS token [B, D]

    def forward(self, mod_a_input: torch.Tensor, mod_b_input: torch.Tensor):
        """
        Self-supervised forward pass with SIGReg loss.

        Returns:
            total_loss (Tensor):          Combined SIGReg + self-consistency loss.
            (pred_loss, sigreg_loss):     Individual loss components for logging.
        """
        cls_emb = self.encode(mod_a_input, mod_b_input)

        # SIGReg: enforces isotropic Gaussian on the joint CLS embedding
        sigreg_loss = self.sigreg(cls_emb)

        # Self-consistency prediction loss over two stochastic forward passes
        cls_emb2 = self.encode(mod_a_input, mod_b_input)
        pred_loss = F.mse_loss(cls_emb, cls_emb2.detach())

        total_loss = pred_loss + self.sigreg_lambda * sigreg_loss
        return total_loss, (pred_loss, sigreg_loss)


def build_audio_mumo_jepa(
    audio_in_chans: int = 1,
    audio_patch_size: tuple = (10, 15),
    seq_in_channels: int = 6,
    embed_dim: int = 256,
    n_fusion_tokens: int = 16,
    n_layers: int = 4,
    n_heads: int = 8,
    sigreg_lambda: float = 1.0,
) -> MuMoJEPAWrapper:
    """
    Build Le MuMo JEPA combining:
      - Modality A: Audio spectrogram via AudioPatchEmbed (from ajepa_backbone)
      - Modality B: 1D prosodic sequence / actions via SequenceStem (from jepa_backbone)

    This is the recommended entry point for the Music-JEPA Action-Conditioned
    speech paradigm (arXiv:2607.22000), treating F0 / RMS energy prosodic
    parameters as 1D actions alongside the audio spectrogram.

    Args:
        audio_in_chans:   Channels in audio spectrogram (usually 1).
        audio_patch_size: (freq_patch, time_patch) for AudioPatchEmbed.
        seq_in_channels:  1D sequence channels (e.g. 2 for F0+RMS, 6 for multi-axis).
        embed_dim:        Shared embedding dimension.
        n_fusion_tokens:  Number of learnable fusion bottleneck tokens.
        n_layers:         Shared transformer depth post-fusion.
        n_heads:          Attention heads.
        sigreg_lambda:    SIGReg loss weight.

    Returns:
        MuMoJEPAWrapper ready for training.
    """
    stem_a = AudioPatchEmbed(
        in_chans=audio_in_chans,
        embed_dim=embed_dim,
        patch_size=audio_patch_size,
    )
    stem_b = SequenceStem(
        in_channels=seq_in_channels,
        hidden_dim=embed_dim // 2,
        out_dim=embed_dim,
    )
    return MuMoJEPAWrapper(
        stem_a=stem_a,
        stem_b=stem_b,
        embed_dim=embed_dim,
        n_fusion_tokens=n_fusion_tokens,
        n_layers=n_layers,
        n_heads=n_heads,
        sigreg_lambda=sigreg_lambda,
    )


# ─────────────────────────────────────────────
#  V-JEPA 2 Video Stem Adapter
# ─────────────────────────────────────────────

class VideoStem(nn.Module):
    """
    Video / Image modality stem for MuMoJEPAWrapper.

    Wraps facebookresearch/vjepa2 (loaded via torch.hub) when available,
    and falls back to a lightweight Conv2D tokenizer otherwise. This matches
    the same "pseudo V-JEPA 2 Backbone" pattern already used in
    main_vjepa2_gun.py and main_vjepa2_fire.py.

    Output: [B, N_patches, embed_dim] token sequence compatible with
            MuMoJEPAWrapper's stem_a / stem_b interface.

    Args:
        embed_dim (int):      Token embedding dimension.
        img_size (int):       Input image spatial resolution (assumes square).
        patch_size (int):     Patch size for tokenisation.
        model_name (str):     torch.hub model name for vjepa2
                              (e.g. 'vjepa2_vit_small'). Used only when
                              use_real_vjepa2=True.
        use_real_vjepa2 (bool): If True, attempts torch.hub.load from
                              'facebookresearch/vjepa2'. Falls back to CNN
                              stub on failure (matching existing run scripts).
        freeze_backbone (bool): If True, freezes V-JEPA 2 weights and only
                              trains the projection head (recommended when
                              using pretrained V-JEPA 2 weights).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        img_size: int = 224,
        patch_size: int = 16,
        model_name: str = 'vjepa2_vit_small',
        use_real_vjepa2: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        self._using_real_vjepa2 = False

        if use_real_vjepa2:
            try:
                backbone = torch.hub.load(
                    'facebookresearch/vjepa2',
                    model_name,
                    pretrained=True,
                    trust_repo=True,
                )
                # Extract ViT dim from the backbone's head or embed_dim attribute
                vit_dim = getattr(backbone, 'embed_dim', 384)
                self.backbone = backbone
                if freeze_backbone:
                    for p in self.backbone.parameters():
                        p.requires_grad = False
                self.proj = nn.Linear(vit_dim, embed_dim)
                self._using_real_vjepa2 = True
                print(f"[VideoStem] Loaded real V-JEPA 2 backbone '{model_name}' (frozen={freeze_backbone})")
            except Exception as e:
                print(f"[VideoStem] Could not load V-JEPA 2 via torch.hub ({e}). "
                      "Falling back to CNN stub (matches main_vjepa2_gun.py behaviour).")

        if not self._using_real_vjepa2:
            # CNN stub — same approach as main_vjepa2_gun.py / main_vjepa2_fire.py
            self.backbone = nn.Sequential(
                nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
            )
            self.proj = nn.Identity()
            self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image/video frame tensor [B, 3, H, W] or [B, T, 3, H, W].
               If 5D (video), frames are flattened into batch dimension and
               the token sequence is averaged across time.

        Returns:
            tokens: [B, N_patches, embed_dim]
        """
        # Handle video input [B, T, 3, H, W] by averaging across time
        is_video = x.dim() == 5
        if is_video:
            B, T, C, H, W = x.shape
            x = x.flatten(0, 1)              # [B*T, 3, H, W]

        if self._using_real_vjepa2:
            # V-JEPA 2 ViT returns patch tokens [B, N, vit_dim]
            with torch.no_grad() if not self.training or not any(
                p.requires_grad for p in self.backbone.parameters()
            ) else torch.enable_grad():
                tokens = self.backbone.forward_features(x)  # [B, N, vit_dim]
                if tokens.dim() == 2:
                    # Some backbones return CLS only; add a dummy spatial dim
                    tokens = tokens.unsqueeze(1)
            tokens = self.proj(tokens)   # [B, N, embed_dim]
        else:
            # CNN stub: [B, embed_dim, H/p, W/p] → [B, N, embed_dim]
            feats = self.backbone(x)     # [B, embed_dim, H_p, W_p]
            B_eff = feats.shape[0]
            tokens = feats.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        if is_video:
            # Average across time frames to get one token set per original sample
            tokens = tokens.reshape(B, T, tokens.shape[1], self.embed_dim)
            tokens = tokens.mean(dim=1)   # [B, N, embed_dim]

        return tokens


# ─────────────────────────────────────────────
#  Audio-Visual MuMo JEPA Factory
# ─────────────────────────────────────────────

def build_audiovisual_mumo_jepa(
    embed_dim: int = 256,
    img_size: int = 224,
    patch_size: int = 16,
    audio_in_chans: int = 1,
    audio_patch_size: tuple = (10, 15),
    n_fusion_tokens: int = 16,
    n_layers: int = 4,
    n_heads: int = 8,
    sigreg_lambda: float = 1.0,
    use_real_vjepa2: bool = True,
    freeze_vjepa2: bool = True,
    vjepa2_model_name: str = 'vjepa2_vit_small',
) -> MuMoJEPAWrapper:
    """
    Build an Audio-Visual Le MuMo JEPA model combining:
      - Modality A: Video / Image via VideoStem (wraps facebookresearch/vjepa2
        when available, falls back to CNN stub matching existing run scripts).
      - Modality B: Audio Spectrogram via AudioPatchEmbed (from ajepa_backbone).

    This extends the Video-JEPA 2 pattern already present in
    main_vjepa2_gun.py and main_vjepa2_fire.py to a full multimodal
    self-supervised setting with learnable fusion tokens and SIGReg.

    Args:
        embed_dim:           Shared token embedding dimension.
        img_size:            Input image resolution (square assumed).
        patch_size:          Spatial patch size for video tokenisation.
        audio_in_chans:      Channels in audio spectrogram (usually 1).
        audio_patch_size:    (freq_patch, time_patch) for AudioPatchEmbed.
        n_fusion_tokens:     Number of learnable fusion bottleneck tokens.
        n_layers:            Shared transformer depth post-fusion.
        n_heads:             Attention heads.
        sigreg_lambda:       SIGReg loss weight.
        use_real_vjepa2:     Try loading facebookresearch/vjepa2 via torch.hub.
        freeze_vjepa2:       Freeze V-JEPA 2 weights (train only fusion + proj).
        vjepa2_model_name:   torch.hub model name for V-JEPA 2.

    Returns:
        MuMoJEPAWrapper ready for audio-visual self-supervised training.
    """
    stem_a = VideoStem(
        embed_dim=embed_dim,
        img_size=img_size,
        patch_size=patch_size,
        model_name=vjepa2_model_name,
        use_real_vjepa2=use_real_vjepa2,
        freeze_backbone=freeze_vjepa2,
    )
    stem_b = AudioPatchEmbed(
        in_chans=audio_in_chans,
        embed_dim=embed_dim,
        patch_size=audio_patch_size,
    )
    return MuMoJEPAWrapper(
        stem_a=stem_a,
        stem_b=stem_b,
        embed_dim=embed_dim,
        n_fusion_tokens=n_fusion_tokens,
        n_layers=n_layers,
        n_heads=n_heads,
        sigreg_lambda=sigreg_lambda,
    )
