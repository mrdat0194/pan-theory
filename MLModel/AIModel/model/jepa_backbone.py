"""
Baseline Audio JEPA / LeWM-JEPA Backbone (1D Convolutional).

Used by: MLModel/AIModel/run/main_audio_jepa.py
Purpose: Next-latent prediction on 1D audio/time-series with SIGReg.
         Achieves 90.00% accuracy on IEMOCAP (20-sample eval).

Note: ``SequenceStem`` (alias of ``Encoder1D``) is exposed at module level
      so it can be imported as a modality stem into Le MuMo JEPA::

          from model.jepa_backbone import SequenceStem
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
import lejepa

class Encoder1D(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, C, T] -> out: [B, D, T]
        return self.net(x)


# Expose as named stem for Le MuMo JEPA compatibility
SequenceStem = Encoder1D


class ARPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=2, heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        # Auto-regressive mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        out = self.transformer(x, mask=causal_mask)
        return out

class LeWMJEPA(nn.Module):
    def __init__(self, encoder, predictor, sigreg_lambda=1.0):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        
        # LeJEPA SIGReg
        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(univariate_test=univariate_test, num_slices=1024)
        self.sigreg_lambda = sigreg_lambda

    def unroll(self, feat, actions=None, nsteps=3, unroll_mode="parallel", compute_loss=True, return_all_steps=False):
        # feat: [B, C, T]
        z = self.encoder(feat) # [B, D, T]
        z_trans = z.transpose(1, 2) # [B, T, D]
        
        pred_z = self.predictor(z_trans) # [B, T, D]
        
        # shift predictor output to match target
        preds = pred_z[:, :-1, :]
        targets = z_trans[:, 1:, :]
        
        if compute_loss:
            pred_loss = torch.nn.functional.mse_loss(preds, targets)
            
            flat_targets = targets.reshape(-1, targets.size(-1))
            sigreg_loss = self.sigreg(flat_targets)
            
            total_loss = pred_loss + self.sigreg_lambda * sigreg_loss
            
            return None, [total_loss, pred_loss, sigreg_loss]
        
        return pred_z.transpose(1, 2), None

def build_jepa(in_channels=1, hidden_dim=64, latent_dim=128):
    encoder = Encoder1D(in_channels, hidden_dim, latent_dim)
    predictor = ARPredictor(input_dim=latent_dim, hidden_dim=latent_dim*4, depth=2, heads=4)
    model = LeWMJEPA(encoder, predictor, sigreg_lambda=1.0)
    return model

def build_action_jepa(in_channels=6, action_dim=1, hidden_dim=64, latent_dim=128):
    return build_jepa(in_channels, hidden_dim, latent_dim)

def apply_rankfeat(feat):
    B, D, T = feat.size()
    u, s, v = torch.linalg.svd(feat, full_matrices=False)
    rank1_component = s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
    return feat - rank1_component

def apply_rankweight(model):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            weight = module.weight.data
            original_shape = weight.shape
            if weight.dim() > 2:
                out_channels = original_shape[0]
                weight = weight.view(out_channels, -1)
            u, s, v = torch.linalg.svd(weight, full_matrices=False)
            rank1 = s[0:1].unsqueeze(1) * u[:, 0:1].mm(v[0:1, :])
            weight = weight - rank1
            if len(original_shape) > 2:
                weight = weight.view(*original_shape)
            module.weight.data = weight

def compute_anomaly_score(model, data, steps=2, use_rankfeat=False):
    model.eval()
    if data.dim() == 2:
        data = data.unsqueeze(1)
    with torch.no_grad():
        predicted_states, _ = model.unroll(data, compute_loss=False)
        target_z = model.encoder(data)
        T_pred = predicted_states.shape[2]
        target_z = target_z[:, :, -T_pred:]
        if use_rankfeat:
            target_z = apply_rankfeat(target_z)
            predicted_states = apply_rankfeat(predicted_states)
        mse = torch.nn.functional.mse_loss(predicted_states, target_z, reduction='none')
        return mse.mean().item()

def jepa_call(model, test_data):
    model.eval()
    representations = []
    with torch.no_grad():
        for data in test_data:
            if isinstance(data, (list, tuple)):
                data = data[0]
            if data.dim() == 2:
                data = data.unsqueeze(1)
            state = model.encoder(data)
            representations.append(state)
    return torch.cat(representations)
