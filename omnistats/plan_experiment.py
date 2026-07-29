"""
omnistats/plan_experiment.py
─────────────────────────────
Phase IV World Model Orchestrator — Energy-Based Experiment Planning.

This script uses the EB-JEPA backbone + OmniStats APA knowledge to find
the optimal experiment design via continuous-relaxation CEM or MPPI.

Usage
-----
    python plan_experiment.py [--planner cem|mppi] [--epochs N]
                              [--n-iters N] [--n-samples N] [--plan-length N]

Outputs
-------
    outputs/jepa_experiment_plan.csv   — optimal continuous action plan
    outputs/jepa_planning_losses.csv   — cost convergence curve

Workflow
--------
  1. Load OmniStats APA state context (LPA profiles + CUPED + causal history).
  2. Build a lightweight JEPA world model (TabularPredictor) that unrolls
     experiment actions in the APA feature latent space.
  3. Train APADecoder jointly with the predictor using historical causal_results.csv.
  4. Run CEM or MPPI planner to minimise APACausalMPCObjective:
         Cost = -w_att * ATT_hat + w_risk * Risk_hat + w_disp * Disparity_hat
  5. Save and display the optimal experiment plan.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── resolve omnistats/ on path ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import OUTPUT_DIR

# ── EB-JEPA planning components ───────────────────────────────────────────────
# Try fallback to user-owned copy in MLModel first.
_EBJEPA_REPO = BASE_DIR.parent / "MLModel" / "AIModel" / "model"
if str(_EBJEPA_REPO) not in sys.path and _EBJEPA_REPO.exists():
    sys.path.insert(0, str(_EBJEPA_REPO))

try:
    from eb_jepa.planning import (
        CEMPlanner,
        MPPIPlanner,
        APACausalMPCObjective,
        PlanningResult,
    )
    _HAS_EBJEPA = True
except ImportError:
    _HAS_EBJEPA = False
    APACausalMPCObjective = None  # will be imported after TabularJEPA block

# ── OmniStats bridge (always available) ──────────────────────────────────────
from modules.jepa_bridge import (
    load_state_context,
    APADecoder,
    train_apa_decoder,
    _flatten_latent,
)

# If eb_jepa.planning could not be imported via package install, load from
# the sibling repo file directly using importlib.
if not _HAS_EBJEPA:
    import importlib.util as _ilu
    _plan_file = _EBJEPA_REPO / "eb_jepa" / "planning.py"
    if _plan_file.exists():
        _spec = _ilu.spec_from_file_location("eb_jepa_planning", _plan_file)
        _mod  = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        CEMPlanner            = _mod.CEMPlanner
        MPPIPlanner           = _mod.MPPIPlanner
        APACausalMPCObjective = _mod.APACausalMPCObjective
        PlanningResult        = _mod.PlanningResult
        _HAS_EBJEPA = True
    else:
        raise ImportError(
            "eb_jepa not found as a package or sibling repo directory. "
            f"Expected at: {_EBJEPA_REPO}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Minimal Tabular JEPA World Model
# ─────────────────────────────────────────────────────────────────────────────
# When eb_jepa is not installed or its CNN encoder cannot process tabular data,
# we substitute a lightweight MLP encoder + predictor that is fully compatible
# with jepa_bridge and the APA planning objective.
# ═════════════════════════════════════════════════════════════════════════════

class TabularEncoder(nn.Module):
    """
    MLP encoder: [B, D_in, 1, 1, 1] → [B, D_lat, 1, 1, 1]

    Replaces the CNN encoder in JEPAbase for tabular state inputs.
    """
    def __init__(self, d_in: int, d_lat: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_lat),
            nn.LayerNorm(d_lat),
        )
        self.d_lat = d_lat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D_in, 1, 1, 1]  →  flatten spatial → [B, D_in]
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        out = self.net(x_flat)           # [B, D_lat]
        return out.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, D_lat, 1, 1, 1]


class TabularActionEncoder(nn.Module):
    """
    MLP action encoder: [B, A, T] → [B, D_lat, T, 1, 1]
    Maps continuous experiment-design action vectors into latent space.
    """
    def __init__(self, a_dim: int, d_lat: int = 32):
        super().__init__()
        self.net = nn.Linear(a_dim, d_lat)
        self.d_lat = d_lat

    def forward(self, a: torch.Tensor | None) -> torch.Tensor | None:
        if a is None:
            return None
        # a: [B, A, T]  →  permute to [B, T, A]  →  linear  →  [B, T, D_lat]
        B, A, T = a.shape
        a_perm = a.permute(0, 2, 1)           # [B, T, A]
        out = self.net(a_perm)                 # [B, T, D_lat]
        return out.permute(0, 2, 1).unsqueeze(3).unsqueeze(4)  # [B, D_lat, T, 1, 1]


class TabularPredictor(nn.Module):
    """
    MLP predictor: state [B, D_lat, T, 1, 1] + action [B, D_lat, T, 1, 1]
    → next state [B, D_lat, T, 1, 1]

    Unrolls autoregressive experiment trajectories in latent space.
    """
    is_rnn = False
    context_length = 1

    def __init__(self, d_lat: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_lat * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_lat),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # state:  [B, D_lat, T, 1, 1]
        # action: [B, D_lat, T, 1, 1] | None
        B, D, T, H, W = state.shape
        s = state.reshape(B * T, D)

        if action is not None:
            a = action.reshape(B * T, -1)
            # pad or truncate action to match D_lat
            if a.shape[-1] < D:
                a = torch.cat([a, torch.zeros(B * T, D - a.shape[-1],
                                               device=a.device)], dim=-1)
            elif a.shape[-1] > D:
                a = a[:, :D]
            inp = torch.cat([s, a], dim=-1)
        else:
            inp = torch.cat([s, torch.zeros_like(s)], dim=-1)

        out = self.net(inp)              # [B*T, D_lat]
        return out.reshape(B, D, T, H, W)


class TabularJEPA(nn.Module):
    """
    Minimal trainable JEPA for tabular state + experiment action spaces.
    Compatible with jepa_bridge's load_state_context() and train_apa_decoder().
    """
    def __init__(self, encoder: TabularEncoder,
                 action_encoder: TabularActionEncoder,
                 predictor: TabularPredictor):
        super().__init__()
        self.encoder       = encoder
        self.action_encoder = action_encoder
        self.predictor     = predictor

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def unroll(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor | None,
        nsteps: int = 1,
        unroll_mode: str = "autoregressive",
        ctxt_window_time: int = 1,
        compute_loss: bool = False,
        return_all_steps: bool = False,
    ):
        state = self.encoder(observations)   # [B, D, 1, 1, 1]
        if actions is not None:
            actions_enc = self.action_encoder(actions)  # [B, D, T, 1, 1]
        else:
            actions_enc = None

        predicted = state
        all_steps = []

        for i in range(nsteps):
            a_i = actions_enc[:, :, i:i+1] if actions_enc is not None else None
            predicted = self.predictor(predicted, a_i)
            if return_all_steps:
                all_steps.append(predicted.clone())

        if return_all_steps:
            return all_steps, None
        return predicted, None


# ═════════════════════════════════════════════════════════════════════════════
# The Experiment-Design Unroll Wrapper (mirrors GCAgent.unroll() interface)
# ═════════════════════════════════════════════════════════════════════════════

def make_experiment_unroll(jepa_model: TabularJEPA, plan_length: int):
    """
    Returns an unroll() callable compatible with CEMPlanner / MPPIPlanner.

    The planner calls:  unroll(obs_init, actions)  → predicted_states [B, D, T, H, W]

    Actions from the planner are [B, A, T] where:
        A = action_dim = number of continuous experiment parameters.
        T = plan_length = number of experiment steps in the horizon.
    """
    def _unroll(obs_init: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size = actions.shape[0]
        obs = obs_init.expand(batch_size, -1, -1, -1, -1)
        predicted, _ = jepa_model.unroll(
            obs, actions,
            nsteps=plan_length,
            unroll_mode="autoregressive",
            compute_loss=False,
            return_all_steps=False,
        )
        return predicted  # [B, D, 1, 1, 1]  (T=1 for last step)
    return _unroll


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def _banner(msg: str) -> None:
    w = 68
    print("\n" + "=" * w)
    print(f"  {msg}")
    print("=" * w)


def main(args_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase IV — Energy-Based Experiment Planning (JEPA × OmniStats)"
    )
    parser.add_argument("--planner",     choices=["cem", "mppi"], default="cem",
                        help="MPC optimiser to use (default: cem)")
    parser.add_argument("--epochs",      type=int, default=50,
                        help="Joint decoder training epochs (default: 50)")
    parser.add_argument("--n-iters",     type=int, default=20,
                        help="Planner optimisation iterations (default: 20)")
    parser.add_argument("--n-samples",   type=int, default=200,
                        help="Action samples per iteration (default: 200)")
    parser.add_argument("--plan-length", type=int, default=5,
                        help="Planning horizon — experiment steps (default: 5)")
    parser.add_argument("--action-dim",  type=int, default=4,
                        help="Continuous action dimensions (default: 4)")
    parser.add_argument("--d-latent",    type=int, default=32,
                        help="JEPA latent dimension (default: 32)")
    parser.add_argument("--w-att",       type=float, default=1.0)
    parser.add_argument("--w-risk",      type=float, default=0.5)
    parser.add_argument("--w-disp",      type=float, default=0.3)
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Joint training learning rate (default: 1e-3)")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args(args_list)

    device = torch.device(args.device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load APA state context ────────────────────────────────────────
    _banner("Step 1 — Load APA State Context")
    state_ctx, feature_names = load_state_context(device=device, verbose=True)
    d_state = state_ctx.shape[1]
    print(f"  State features ({d_state}): {feature_names}")

    # ── Step 2: Build Tabular JEPA world model ────────────────────────────────
    _banner("Step 2 — Build Tabular JEPA World Model")
    encoder       = TabularEncoder(d_in=d_state, d_lat=args.d_latent)
    action_encoder = TabularActionEncoder(a_dim=args.action_dim, d_lat=args.d_latent)
    predictor     = TabularPredictor(d_lat=args.d_latent)
    jepa          = TabularJEPA(encoder, action_encoder, predictor).to(device)
    print(f"  TabularJEPA: D_state={d_state}, D_latent={args.d_latent}, "
          f"A_dim={args.action_dim}")

    # ── Step 3: Build & jointly train APA decoder ─────────────────────────────
    _banner("Step 3 — Joint Training: APADecoder × JEPA Encoder")
    decoder = APADecoder(d_latent=args.d_latent).to(device)
    decoder = train_apa_decoder(
        jepa_model=jepa,
        decoder=decoder,
        n_epochs=args.epochs,
        lr=args.lr,
        device=device,
        verbose=True,
    )

    # ── Step 4: Build APA planning objective ──────────────────────────────────
    _banner("Step 4 — Build APACausalMPCObjective")
    # Estimate ATT baseline from historical data for scale normalisation
    causal_csv = os.path.join(OUTPUT_DIR, "causal_results.csv")
    att_baseline = 1.0
    if os.path.exists(causal_csv):
        cr = pd.read_csv(causal_csv).dropna(subset=["estimate"])
        if len(cr) > 0:
            att_baseline = float(cr["estimate"].abs().mean()) or 1.0

    objective = APACausalMPCObjective(
        decoder=decoder,
        w_att=args.w_att,
        w_risk=args.w_risk,
        w_disp=args.w_disp,
        att_baseline=att_baseline,
    )
    print(f"  Objective: -w_att*ATT + w_risk*Risk + w_disp*Disparity")
    print(f"  Weights: w_att={args.w_att}, w_risk={args.w_risk}, "
          f"w_disp={args.w_disp}")
    print(f"  ATT baseline (normalisation): {att_baseline:.4f}")

    # ── Step 5: Build planner ─────────────────────────────────────────────────
    _banner(f"Step 5 — Build {args.planner.upper()} Planner")
    unroll_fn = make_experiment_unroll(jepa, plan_length=args.plan_length)

    planner_kwargs = dict(
        unroll=unroll_fn,
        action_dim=args.action_dim,
        n_iters=args.n_iters,
        num_samples=args.n_samples,
        plan_length=args.plan_length,
        decode_each_iteration=False,
        decode_loc_to_pixel=None,
    )

    if args.planner == "cem":
        planner = CEMPlanner(**planner_kwargs, num_elites=max(5, args.n_samples // 10))
    else:
        planner = MPPIPlanner(**planner_kwargs, temperature=0.01,
                              num_elites=max(10, args.n_samples // 5))

    planner.set_objective(objective)
    print(f"  Planner: {type(planner).__name__}")
    print(f"  Horizon={args.plan_length} steps, Samples={args.n_samples}, "
          f"Iters={args.n_iters}")

    # ── Step 6: Plan ──────────────────────────────────────────────────────────
    _banner("Step 6 — Planning (Gradient-Free Continuous Optimisation)")
    with torch.no_grad():
        result: PlanningResult = planner.plan(
            obs_init=state_ctx,
            steps_left=args.plan_length,
            eval_mode=True,
            t0=True,
        )

    best_actions = result.actions.cpu().numpy()   # [T, A]
    losses       = result.losses.cpu().numpy()    # [n_iters, 1]

    print(f"\n  Planning converged. Final cost: {losses[-1, 0]:.4f}")
    print(f"  Best action plan shape: {best_actions.shape}  (T={args.plan_length}, A={args.action_dim})")

    # ── Step 7: Interpret & save plan ─────────────────────────────────────────
    _banner("Step 7 — Experiment Plan Output")

    _base_labels = [
        "treatment_fraction",          # [0,1] — proportion allocated to treatment arm
        "segment_focus_index",         # [0,1] — 0=all users, 1=highest-HTE segment
        "sample_size_relative",        # [0,1] — fraction of power-analysis n_per_arm
        "observation_horizon_factor",  # [0,1] — 0=minimum, 1=full experiment duration
    ]
    extra = [f"action_{i}" for i in range(args.action_dim - len(_base_labels))]
    action_labels = (_base_labels + extra)[:args.action_dim]

    plan_rows = []
    for t in range(args.plan_length):
        row = {"step": t + 1}
        for j, label in enumerate(action_labels):
            # Sigmoid to interpret raw continuous action as [0,1] allocation
            val = float(torch.sigmoid(torch.tensor(best_actions[t, j])).item())
            row[label] = round(val, 4)
        plan_rows.append(row)

    plan_df = pd.DataFrame(plan_rows)
    plan_path = os.path.join(OUTPUT_DIR, "jepa_experiment_plan.csv")
    plan_df.to_csv(plan_path, index=False)

    losses_df = pd.DataFrame({"iteration": range(1, len(losses) + 1),
                               "apa_cost": losses[:, 0]})
    losses_path = os.path.join(OUTPUT_DIR, "jepa_planning_losses.csv")
    losses_df.to_csv(losses_path, index=False)

    print(f"\n  Optimal Experiment Plan:")
    print(plan_df.to_string(index=False))
    print(f"\n  Plan saved   -> {plan_path}")
    print(f"  Losses saved -> {losses_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("PHASE IV COMPLETE")
    print(f"\n  Model: TabularJEPA + APADecoder (joint training, {args.epochs} epochs)")
    print(f"  Planner: {args.planner.upper()}  ({args.n_iters} iters × {args.n_samples} samples)")
    print(f"  Horizon: {args.plan_length} experiment steps")
    print(f"  Final APA cost: {losses[-1, 0]:.4f}")
    print()
    print("  Interpretation of action dimensions:")
    for j, label in enumerate(action_labels):
        val = float(torch.sigmoid(torch.tensor(best_actions[0, j])).item())
        print(f"    [{j}] {label:<30} = {val:.3f}")
    print()
    print("  Next step: Feed plan into experiment_design.py to generate")
    print("  the final randomization_schedule.csv for Engineering.")


if __name__ == "__main__":
    main()
