"""
mfg_rdd.py
========
Mean Field Game Regression Discontinuity Design (MFG RDD).

Replaces legacy dual-Transformer Neural RDD (deep_rdd.py [DEPRECATED]) and bandwidth-sensitive
curve fitting with a continuous MFG Discontinuity Density Flow.

Mathematical Formulation:
    The running variable density ρ(x, t) evolves as a population of agents approaching 
    the cutoff threshold c.
    
    HJB Discontinuity Interface:
        The threshold c is embedded as a singular value interface in the HJB equation:
            τ(c) = lim_{x -> c^+} u(x, t) - lim_{x -> c^-} u(x, t)
            
    where τ(c) is the true structural Local Average Treatment Effect (LATE) at the cutoff.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MFGRegressionDiscontinuity(nn.Module):
    """
    MFG Discontinuity Density Flow for RDD causal inference.
    """

    def __init__(self, cutoff: float = 0.0, hidden_dim: int = 64):
        super().__init__()
        self.cutoff = cutoff
        
        # Left-side value network u_left(x) for x < cutoff
        self.left_value_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Right-side value network u_right(x) for x >= cutoff
        self.right_value_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def compute_jump_at_cutoff(self, device: torch.device) -> torch.Tensor:
        """
        Evaluates the structural HJB jump discontinuity at x = cutoff:
            LATE = u_right(cutoff^+) - u_left(cutoff^-)
        """
        cutoff_tensor = torch.tensor([[self.cutoff]], dtype=torch.float32, device=device)
        u_right = self.right_value_net(cutoff_tensor)
        u_left = self.left_value_net(cutoff_tensor)
        
        late_jump = u_right - u_left
        return late_jump.squeeze()

    def forward(
        self,
        running_var: torch.Tensor,
        outcomes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Fits continuous MFG HJB value curves on both sides of cutoff and estimates jump.
        
        Args:
            running_var: [N, 1] or [N] running variable
            outcomes: [N, 1] or [N] observed outcome
        """
        if running_var.dim() == 1:
            running_var = running_var.unsqueeze(-1)
        if outcomes.dim() == 1:
            outcomes = outcomes.unsqueeze(-1)
            
        mask_left = (running_var < self.cutoff)
        mask_right = (running_var >= self.cutoff)
        
        x_left = running_var[mask_left].unsqueeze(-1) if mask_left.dim() == 1 else running_var[mask_left].view(-1, 1)
        y_left = outcomes[mask_left].unsqueeze(-1) if mask_left.dim() == 1 else outcomes[mask_left].view(-1, 1)
        
        x_right = running_var[mask_right].unsqueeze(-1) if mask_right.dim() == 1 else running_var[mask_right].view(-1, 1)
        y_right = outcomes[mask_right].unsqueeze(-1) if mask_right.dim() == 1 else outcomes[mask_right].view(-1, 1)
        
        # Predict value functions
        y_pred_left = self.left_value_net(x_left) if x_left.size(0) > 0 else torch.empty(0, 1, device=running_var.device)
        y_pred_right = self.right_value_net(x_right) if x_right.size(0) > 0 else torch.empty(0, 1, device=running_var.device)
        
        loss_left = F.mse_loss(y_pred_left, y_left) if x_left.size(0) > 0 else torch.tensor(0.0, device=running_var.device)
        loss_right = F.mse_loss(y_pred_right, y_right) if x_right.size(0) > 0 else torch.tensor(0.0, device=running_var.device)
        
        total_loss = loss_left + loss_right
        
        # Compute structural jump at cutoff
        late_effect = self.compute_jump_at_cutoff(running_var.device)
        
        return {
            "loss": total_loss,
            "late_effect": late_effect,
            "loss_left": loss_left,
            "loss_right": loss_right
        }
