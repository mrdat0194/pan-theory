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

def compute_anomaly_score(model, data, steps=1):
    """
    Computes an anomaly score based on JEPA's predictive error in the latent space.
    Higher score indicates higher likelihood of anomaly (shock/transition).
    """
    model.eval()
    if data.dim() == 2:
        data = data.unsqueeze(1)
        
    with torch.no_grad():
        state = model.encode(data)
        
        predicted_states = state
        for _ in range(steps):
            predicted_states = model.predictor(predicted_states, None)[:, :, :-1]
            predicted_states = torch.cat((state[:, :, :model.predictor.context_length], predicted_states), dim=2)
            
        # Compare prediction with actual encoded state
        # The prediction applies to state[:, :, steps:]
        target = state[:, :, steps:]
        pred = predicted_states[:, :, steps:]
        
        # Mean squared error in latent space across time
        error = torch.mean((pred - target) ** 2, dim=1) # [B, T']
        return error
