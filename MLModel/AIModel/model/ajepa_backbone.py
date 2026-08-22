"""
A-JEPA Backbone: Audio Joint-Embedding Predictive Architecture.

Used by: MLModel/AIModel/run/main_audio_ajepa.py
Purpose: Masked patch self-supervised pre-training on audio spectrograms
         (MFCCs treated as 2D single-channel images).

Note: ``AudioPatchEmbed`` (alias of ``PatchEmbed``) is exposed at module level
      so it can be imported as a modality stem into Le MuMo JEPA::

          from model.ajepa_backbone import AudioPatchEmbed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbed(nn.Module):
    """Conv-Stem Patch Embedding for Audio Spectrograms"""
    def __init__(self, in_chans=1, embed_dim=256, patch_size=(40, 3)):
        super().__init__()
        self.patch_size = patch_size
        t_stride = patch_size[1]
        
        # Treat 40 frequency bins as channels
        self.stem = nn.Sequential(
            nn.Conv1d(40, embed_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=t_stride, stride=t_stride, padding=0),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, 1, 40, 150]
        x = x.squeeze(1)  # [B, 40, 150]
        x = self.stem(x)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        return x


# Expose as named stem for Le MuMo JEPA compatibility
AudioPatchEmbed = PatchEmbed


class AJEPA(nn.Module):
    def __init__(self, 
                 in_chans=1,
                 img_size=(40, 150), 
                 patch_size=(10, 15),
                 embed_dim=256,
                 enc_depth=4,
                 enc_heads=8,
                 pred_depth=2,
                 pred_heads=8,
                 mask_ratio=0.6,
                 ema_momentum=0.996):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.ema_momentum = ema_momentum
        
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.num_patches = num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # Context Encoder (Student)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=enc_heads, dim_feedforward=embed_dim*4, activation="gelu", batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Target Encoder (Teacher - EMA of Context Encoder)
        self.target_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_depth)
        self.target_norm = nn.LayerNorm(embed_dim)
        
        # Initialize target encoder with context encoder weights
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.encoder_norm.parameters(), self.target_norm.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Predictor
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)
        
        self.predictor_embed = nn.Linear(embed_dim, embed_dim)
        pred_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=pred_heads, dim_feedforward=embed_dim*4, activation="gelu", batch_first=True, norm_first=True)
        self.predictor = nn.TransformerEncoder(pred_layer, num_layers=pred_depth)
        self.predictor_norm = nn.LayerNorm(embed_dim)

    @torch.no_grad()
    def update_target_network(self, momentum=None):
        m = momentum if momentum is not None else self.ema_momentum
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        for param_q, param_k in zip(self.encoder_norm.parameters(), self.target_norm.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        
        # Gather kept tokens
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        return x_kept, ids_keep, ids_mask, ids_restore

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns loss and (if needed) predictions.
        """
        B = x.shape[0]
        # 1. Patch embedding
        x_embed = self.patch_embed(x)
        
        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_embed = torch.cat((cls_tokens, x_embed), dim=1)
        
        # Add positional embeddings
        x_embed = x_embed + self.pos_embed
        
        # We only mask the patches, not the CLS token
        x_patches = x_embed[:, 1:, :]
        cls_token_embed = x_embed[:, 0:1, :]
        
        # 2. Masking
        x_kept, ids_keep, ids_mask, ids_restore = self.random_masking(x_patches, self.mask_ratio)
        
        # Prepend CLS token back for encoder
        x_kept = torch.cat((cls_token_embed, x_kept), dim=1)
        
        # 3. Context Encoder
        context_out = self.context_encoder(x_kept)
        context_out = self.encoder_norm(context_out)
        
        # 4. Target Encoder (EMA)
        with torch.no_grad():
            target_out = self.target_encoder(x_embed)
            target_out = self.target_norm(target_out)
            # Extract target representations for masked patches
            # ids_mask is 0-indexed for patches, but target_out has CLS token at 0.
            # So we need to add 1 to ids_mask
            target_masked = torch.gather(target_out, dim=1, index=(ids_mask + 1).unsqueeze(-1).expand(-1, -1, self.embed_dim))
            
        # 5. Predictor
        context_out = self.predictor_embed(context_out)
        
        # context_out has CLS at 0, followed by kept patches
        cls_out = context_out[:, 0:1, :]
        kept_out = context_out[:, 1:, :]
        
        # Reconstruct full sequence with mask tokens
        mask_tokens = self.mask_token.expand(B, self.num_patches - kept_out.shape[1], -1)
        x_full = torch.cat([kept_out, mask_tokens], dim=1)
        
        # Unshuffle to original positions
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        
        # Prepend CLS token
        x_full = torch.cat([cls_out, x_full], dim=1)
        
        # Add positional embeddings again for predictor
        x_full = x_full + self.pos_embed
        
        # Predictor forward
        pred_out = self.predictor(x_full)
        pred_out = self.predictor_norm(pred_out)
        
        # Extract predictions for masked tokens (again, +1 for CLS)
        pred_masked = torch.gather(pred_out, dim=1, index=(ids_mask + 1).unsqueeze(-1).expand(-1, -1, self.embed_dim))
        
        # 6. Loss
        # MSE Loss is standard for JEPA
        loss = F.mse_loss(pred_masked, target_masked)
        
        return loss

    def encode(self, x):
        """
        For downstream tasks: encode full input without masking.
        Returns the CLS token representation.
        """
        B = x.shape[0]
        x_embed = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_embed = torch.cat((cls_tokens, x_embed), dim=1)
        x_embed = x_embed + self.pos_embed
        
        out = self.context_encoder(x_embed)
        out = self.encoder_norm(out)
        # Return CLS token
        return out[:, 0, :]

    def encode_sequence(self, x):
        """
        For downstream tasks (like diarization): encode full input without masking.
        Returns the full sequence of representations: [B, T, D]
        (skipping the CLS token at index 0).
        """
        B = x.shape[0]
        x_embed = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_embed = torch.cat((cls_tokens, x_embed), dim=1)
        x_embed = x_embed + self.pos_embed
        
        out = self.context_encoder(x_embed)
        out = self.encoder_norm(out)
        # Return all patch tokens (skip CLS token at index 0)
        return out[:, 1:, :]
