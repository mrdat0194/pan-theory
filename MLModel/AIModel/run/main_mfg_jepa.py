"""
main_mfg_jepa.py
================
Execution Script for Mean Field Game (MFG) JEPA & Causal Optimal Transport.

Demonstrates:
1. Continuous Latent Flow regularized via Hamilton-Jacobi-Bellman (HJB) equations.
2. Continuous Shapley Value Attribution over latent dimensions.
3. MFG Counterfactual Difference-in-Differences (mfg_did).
4. MFG Confounder-Density Matching (mfg_iv).

Usage:
    python MLModel/AIModel/run/main_mfg_jepa.py
"""

import sys
import os
import torch

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from omnistats.modules.game_theory.mean_field_jepa import MeanFieldJEPASolver
from omnistats.modules.game_theory.continuous_shapley import ContinuousShapleyAttributor
from omnistats.modules.causal.mfg_did import MFGDifferenceInDifferences
from omnistats.modules.causal.mfg_iv import MFGInstrumentalVariables
from omnistats.modules.causal.mfg_rdd import MFGRegressionDiscontinuity


def run_mfg_jepa_demo():
    print("=" * 65)
    print("SOTA MEAN FIELD GAME (MFG) JEPA & CAUSAL SUITE DEMO")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 1. Continuous Latent Flow (MFG Solver: HJB + Fokker-Planck)
    # ---------------------------------------------------------
    print("\n1. Running Mean Field Game Latent Transport Solver...")
    d_latent = 16
    batch_size = 8
    
    solver = MeanFieldJEPASolver(d_latent=d_latent, sigma=0.05).to(device)
    z_init = torch.randn(batch_size, d_latent, device=device)
    
    mfg_out = solver(z_init, n_steps=5, dt=0.1)
    print(f"   [+] Trajectory Shape: {mfg_out['trajectory'].shape}  [B, T, D]")
    print(f"   [+] HJB Residual Loss: {mfg_out['hjb_loss'].item():.6f}")

    # ---------------------------------------------------------
    # 2. Continuous Shapley Attribution (Integrated Gradients)
    # ---------------------------------------------------------
    print("\n2. Running Continuous Shapley Value Attribution...")
    attributor = ContinuousShapleyAttributor(steps=30).to(device)
    
    # Dummy downstream predictor f(z) -> scalar target
    dummy_head = torch.nn.Linear(d_latent, 1).to(device)
    model_fn = lambda z: dummy_head(z)
    
    shapley_res = attributor(model_fn, z_init)
    print(f"   [+] Shapley Values Shape: {shapley_res['shapley_values'].shape}  [B, D]")
    print(f"   [+] Completeness Gap Error: {shapley_res['completeness_error'].item():.6e}")

    # ---------------------------------------------------------
    # 3. MFG Counterfactual Difference-in-Differences (mfg_did)
    # ---------------------------------------------------------
    print("\n3. Running MFG Counterfactual Trajectory Flow (DiD)...")
    mfg_did = MFGDifferenceInDifferences(d_features=4).to(device)
    
    ctrl_pre = torch.randn(10, 5, 4, device=device)
    ctrl_post = torch.randn(10, 5, 4, device=device)
    trt_pre = torch.randn(10, 5, 4, device=device)
    trt_post = torch.randn(10, 5, 4, device=device) + 0.75  # true treatment effect +0.75
    
    did_out = mfg_did(ctrl_pre, ctrl_post, trt_pre, trt_post)
    print(f"   [+] Estimated Counterfactual ATT: {did_out['att_overall'].item():.4f}")
    print(f"   [+] Density Fit Loss: {did_out['fit_loss'].item():.6f}")

    # ---------------------------------------------------------
    # 4. MFG Confounder-Density Matching (mfg_iv)
    # ---------------------------------------------------------
    print("\n4. Running MFG Confounder-Density Matching (IV)...")
    mfg_iv = MFGInstrumentalVariables(d_instrument=2, d_treatment=1, d_outcome=1).to(device)
    
    Z = torch.randn(32, 2, device=device)
    X = 0.5 * Z.sum(dim=-1, keepdim=True) + torch.randn(32, 1, device=device)
    Y = 2.0 * X + torch.randn(32, 1, device=device)
    
    iv_out = mfg_iv(Z, X, Y)
    print(f"   [+] Causal Outcome Loss: {iv_out['outcome_loss'].item():.6f}")
    print(f"   [+] Transport Cost Penalty: {iv_out['transport_cost'].item():.6f}")

    # ---------------------------------------------------------
    # 5. MFG Discontinuity Density Flow (mfg_rdd)
    # ---------------------------------------------------------
    print("\n5. Running MFG Discontinuity Density Flow (RDD)...")
    mfg_rdd = MFGRegressionDiscontinuity(cutoff=0.0).to(device)
    
    r_var = torch.linspace(-2.0, 2.0, 100, device=device).unsqueeze(-1)
    # True jump at cutoff = 15.0
    y_rdd = 3.0 * r_var + torch.where(r_var >= 0.0, torch.tensor(15.0, device=device), torch.tensor(0.0, device=device)) + torch.randn_like(r_var) * 0.1
    
    rdd_out = mfg_rdd(r_var, y_rdd)
    print(f"   [+] Estimated Cutoff LATE Jump: {rdd_out['late_effect'].item():.4f}")
    print(f"   [+] Density Fit Loss (Left+Right): {rdd_out['loss'].item():.6f}")

    print("\n" + "=" * 65)
    print("ALL SOTA MEAN FIELD GAME (MFG) MODULES EXECUTED CLEANLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_mfg_jepa_demo()
