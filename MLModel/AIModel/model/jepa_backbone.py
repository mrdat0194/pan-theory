import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from MLModel.AIModel.model.eb_jepa.jepa import JEPA
from MLModel.AIModel.model.eb_jepa.architectures import Projector
from MLModel.AIModel.model.eb_jepa.losses import SquareLossSeq, VCLoss

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

class Predictor1D(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class StateOnlyPredictor1D(nn.Module):
    def __init__(self, predictor, context_length=2):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = False
        self.context_length = context_length

    def forward(self, x, a=None):
        # x: [B, D, T]
        prev_state = x[:, :, :-1]
        next_state = x[:, :, 1:]
        combined_xa = torch.cat((prev_state, next_state), dim=1)
        return self.predictor(combined_xa)

def build_jepa(in_channels=1, hidden_dim=64, latent_dim=128):
    encoder = Encoder1D(in_channels, hidden_dim, latent_dim)
    # Predictor takes 2 concatenated states
    predictor_model = Predictor1D(latent_dim * 2, hidden_dim, latent_dim)
    predictor = StateOnlyPredictor1D(predictor_model, context_length=2)
    
    # Using VC Loss for regularizing the representation (Variance-Covariance)
    projector = Projector(f"{latent_dim}-{latent_dim*2}-{latent_dim*2}")
    regularizer = VCLoss(std_coeff=25.0, cov_coeff=25.0, proj=projector)
    
    # Prediction loss (Mean Squared Error on the latent space)
    ploss = SquareLossSeq(projector)
    
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss)
    return jepa

class ActionConditionedPredictor1D(nn.Module):
    def __init__(self, predictor, context_length=2):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = False
        self.context_length = context_length

    def forward(self, x, a=None):
        # x: [B, D, T]
        # a: [B, A, T-1] (actions applied to reach the next state)
        prev_state = x[:, :, :-1]
        next_state = x[:, :, 1:]
        
        if a is not None:
            # We concatenate current latent state, next latent state, and action
            # a: [B, A, T-1] 
            combined_xa = torch.cat((prev_state, next_state, a), dim=1)
        else:
            # Fallback if no action is provided (shouldn't happen in action-conditioned setup)
            combined_xa = torch.cat((prev_state, next_state), dim=1)
            
        return self.predictor(combined_xa)

def build_action_jepa(in_channels=6, action_dim=1, hidden_dim=64, latent_dim=128):
    encoder = Encoder1D(in_channels, hidden_dim, latent_dim)
    # Predictor takes 2 concatenated states + action
    predictor_model = Predictor1D(latent_dim * 2 + action_dim, hidden_dim, latent_dim)
    predictor = ActionConditionedPredictor1D(predictor_model, context_length=2)
    
    # Using VC Loss for regularizing the representation (Variance-Covariance)
    projector = Projector(f"{latent_dim}-{latent_dim*2}-{latent_dim*2}")
    regularizer = VCLoss(std_coeff=25.0, cov_coeff=25.0, proj=projector)
    
    # Prediction loss (Mean Squared Error on the latent space)
    ploss = SquareLossSeq(projector)
    
    aencoder = nn.Identity()
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss)
    return jepa

def model_jepa(train_loader, epochs=10, lr=1e-3, in_channels=1, steps=2):
    """
    Trains the JEPA backbone on time-series data.
    train_loader should yield tensors of shape [B, C, T]
    """
    model = build_jepa(in_channels=in_channels)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, (list, tuple)):
                data = data[0]
            
            # Ensure shape is [B, C, T]
            if data.dim() == 2:
                # If [B, T], add channel dim
                data = data.unsqueeze(1)
                
            optimizer.zero_grad()
            
            # Unroll JEPA to predict future states and compute loss
            _, losses = model.unroll(
                data,
                actions=None,
                nsteps=steps,
                unroll_mode="parallel",
                compute_loss=True,
                return_all_steps=False
            )
            
            loss, regl, rloss_unweight, regldict, pl = losses
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}')
            
    return model

def apply_rankfeat(feat):
    """
    Applies RankFeat: removes the rank-1 component from the features.
    feat shape: [Batch, Dim, Time]
    """
    B, D, T = feat.size()
    u, s, v = torch.linalg.svd(feat, full_matrices=False)
    
    # rank-1 component
    rank1_component = s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
    return feat - rank1_component

def apply_rankweight(model):
    """
    Applies RankWeight by removing the rank-1 component from the weights of all Conv1d and Linear layers.
    This modifies the model weights in-place.
    """
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            weight = module.weight.data
            original_shape = weight.shape
            
            # Flatten spatial dimensions for SVD if it's a Conv layer
            if weight.dim() > 2:
                out_channels = original_shape[0]
                weight = weight.view(out_channels, -1)
                
            u, s, v = torch.linalg.svd(weight, full_matrices=False)
            # Remove rank-1 component
            rank1 = s[0:1].unsqueeze(1) * u[:, 0:1].mm(v[0:1, :])
            weight = weight - rank1
            
            # Restore original shape
            if len(original_shape) > 2:
                weight = weight.view(*original_shape)
                
            module.weight.data = weight

def compute_anomaly_score(model, data, steps=2, use_rankfeat=False):
    """
    Given a sequence `data` [1, C, T], predict future states and compute MSE against encoded targets.
    """
    model.eval()
    if data.dim() == 2:
        data = data.unsqueeze(1)
        
    with torch.no_grad():
        # model.unroll predicts the state sequence
        predicted_states, _ = model.unroll(
            data,
            actions=None,
            nsteps=steps,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=False
        )
        # target representations
        target_z = model.encoder(data)
        
        # Align lengths (JEPA predicts the *next* states, so we shift target)
        # e.g., if predicted_states is T-1 length, we compare to target_z from index 1 to end.
        T_pred = predicted_states.shape[2]
        target_z = target_z[:, :, -T_pred:]
        
        if use_rankfeat:
            # Strip the dominant rank-1 feature from both sequences to highlight anomalies
            target_z = apply_rankfeat(target_z)
            predicted_states = apply_rankfeat(predicted_states)

        mse = torch.nn.functional.mse_loss(predicted_states, target_z, reduction='none')
        # mean across feature dims and sequence length
        score = mse.mean().item()
        return score

def jepa_call(model, test_data):
    """
    Inference: Extract latent representations (backbone features) for downstream tasks.
    test_data: Iterable of tensors [B, C, T]
    Returns: Concatenated representations [N, D, T]
    """
    model.eval()
    representations = []
    with torch.no_grad():
        for data in test_data:
            if isinstance(data, (list, tuple)):
                data = data[0]
            
            if data.dim() == 2:
                data = data.unsqueeze(1)
                
            state = model.encode(data)
            representations.append(state)
    return torch.cat(representations)


