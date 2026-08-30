"""
continuous_shapley.py
=====================
Continuous Shapley Value & Integrated Gradients Module.

Replaces discrete permutation MSA (kuffmode/msa) with continuous path integrals
over continuous latent manifolds z ∈ R^D.

Mathematical Formulation:
    Continuous Shapley Value (Aumann-Shapley / Integrated Gradients):
    
        Shapley_i(z) = (z_i - z_i') * ∫_0^1 (∂f(z' + α(z - z')) / ∂z_i) dα
        
    where z' is a baseline reference latent state (e.g. isotropic Gaussian mean),
    and α ∈ [0, 1] parameterizes the straight-line path between baseline and z.

Properties (Axiomatic Guarantees):
    1. Completeness: ∑_i Shapley_i(z) = f(z) - f(z')
    2. Implementation Invariance: Identical functions yield identical attributions
    3. Linearity: Shapley_i(a f + b g) = a Shapley_i(f) + b Shapley_i(g)
    4. Symmetry & Dummy Player: Zero gradient along dimension i ⇒ Shapley_i = 0
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Dict, Tuple


def integrated_gradients_shapley(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    latent_z: torch.Tensor,
    baseline_z: Optional[torch.Tensor] = None,
    steps: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Continuous Shapley values via Integrated Gradients.
    
    Args:
        model_fn: Function mapping latent z [B, D] to a scalar output [B, 1] or [B].
        latent_z: Current latent vector [B, D].
        baseline_z: Baseline reference latent vector [B, D] (default: zeros).
        steps: Number of quadrature steps along the path integral α ∈ [0, 1].
        
    Returns:
        shapley_values: Attribution per latent dimension [B, D].
        delta_f: Difference f(z) - f(baseline) [B, 1] verifying completeness.
    """
    if baseline_z is None:
        baseline_z = torch.zeros_like(latent_z)

    batch_size, d_latent = latent_z.shape
    diff = latent_z - baseline_z  # [B, D]

    # Sample steps along straight-line trajectory
    alphas = torch.linspace(0.0, 1.0, steps, device=latent_z.device)
    
    grads_list = []
    
    for alpha in alphas:
        # Interpolated point along trajectory
        z_alpha = (baseline_z + alpha * diff).detach().requires_grad_(True)
        out = model_fn(z_alpha)
        
        if out.dim() > 1:
            out = out.sum(dim=-1)
            
        grad = torch.autograd.grad(out.sum(), z_alpha, create_graph=False)[0]  # [B, D]
        grads_list.append(grad)
        
    # Average gradients along path
    avg_grads = torch.stack(grads_list, dim=0).mean(dim=0)  # [B, D]
    
    # Continuous Shapley values
    shapley_values = diff * avg_grads  # [B, D]
    
    # Verify completeness: sum(Shapley_i) ≈ f(z) - f(baseline)
    with torch.no_grad():
        f_z = model_fn(latent_z)
        f_base = model_fn(baseline_z)
        if f_z.dim() == 1:
            f_z = f_z.unsqueeze(-1)
            f_base = f_base.unsqueeze(-1)
        delta_f = f_z - f_base
        
    return shapley_values, delta_f


class ContinuousShapleyAttributor(nn.Module):
    """
    PyTorch Module wrapper for Continuous Shapley Attribution over JEPA latent flows.
    """

    def __init__(self, steps: int = 50):
        super().__init__()
        self.steps = steps

    def forward(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        latent_z: torch.Tensor,
        baseline_z: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates Shapley values and completeness gap.
        """
        shapley, delta_f = integrated_gradients_shapley(
            model_fn=model_fn,
            latent_z=latent_z,
            baseline_z=baseline_z,
            steps=self.steps
        )
        
        completeness_error = (shapley.sum(dim=-1, keepdim=True) - delta_f).abs().mean()
        
        return {
            "shapley_values": shapley,
            "delta_f": delta_f,
            "completeness_error": completeness_error
        }
