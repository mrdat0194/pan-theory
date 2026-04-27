import sys
import os
import torch
import numpy as np
from torch.optim import SGD

# Adjust PYTHONPATH so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'eb_jepa')))

from MLModel.AIModel.model.optimal_control import (
    f, 
    cost_logsumexp
)
from eb_jepa.planning import CEMPlanner

def main():
    print("=== Comparing SGD (Old) vs JEPA CEM Planner (Advanced) ===")
    
    # Define problem matching the old run
    x_x, x_y = 5.0, 1.0
    s_init = 1.0
    T = 15
    dt = 1.0
    epochs = 100 # Iterations for both methods
    stepsize = 0.005
    
    # ---------------------------------------------------------
    # 1. SGD Optimal Control (Old Way)
    # ---------------------------------------------------------
    print("\n--- Running SGD Optimal Control ---")
    u_sgd = torch.nn.Parameter(torch.zeros(T, 2))
    optimizer = SGD((u_sgd,), lr=stepsize)
    sgd_loss_history = []
    
    for epoch in range(epochs):
        x = [torch.tensor((0., 0., 0., float(s_init)), dtype=torch.float32)]
        for t in range(1, T+1):
            x.append(x[-1] + f(x[-1], u_sgd[t-1]) * dt)
        x_t = torch.stack(x)
        cost = cost_logsumexp(x_t, (x_x, x_y))
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        sgd_loss_history.append(cost.item())
    
    print(f"SGD Final Cost: {sgd_loss_history[-1]:.4f}")
    
    # ---------------------------------------------------------
    # 2. JEPA CEM Planner (Advanced Implementation)
    # ---------------------------------------------------------
    print("\n--- Running JEPA CEM Planner ---")
    
    def analytical_unroll(obs_init, actions, nsteps=None, compute_loss=False, **kwargs):
        """
        obs_init: (B, D)
        actions: (B, A, T)
        """
        B = actions.shape[0]
        T_steps = actions.shape[2]
        
        current_state = obs_init.expand(B, -1).clone()
        states = []
        for t in range(T_steps):
            u_t = actions[:, :, t]
            
            dx = torch.zeros_like(current_state)
            s = current_state[:, 3]
            θ = current_state[:, 2]
            ϕ = u_t[:, 0]
            a = u_t[:, 1]
            
            dx[:, 0] = s * torch.cos(θ)
            dx[:, 1] = s * torch.sin(θ)
            dx[:, 2] = s / 1.0 * torch.tan(ϕ) # L=1
            dx[:, 3] = a
            
            current_state = current_state + dx * dt
            states.append(current_state.clone())
            
        predicted_states = torch.stack(states, dim=2)
        return predicted_states

    def cem_objective(predicted_states):
        """
        predicted_states: (B, D, T)
        """
        dists = (predicted_states[:, 0, :] - x_x).pow(2) + (predicted_states[:, 1, :] - x_y).pow(2)
        cost = -1 * torch.logsumexp(-1 * dists, dim=1)
        return cost

    planner = CEMPlanner(
        unroll=analytical_unroll,
        n_iters=epochs,
        num_samples=200,
        plan_length=T,
        action_dim=2,
        var_scale=0.5,
        num_elites=20,
        decode_each_iteration=False
    )
    planner.set_objective(cem_objective)
    
    obs_init = torch.tensor([[0.0, 0.0, 0.0, s_init]], dtype=torch.float32)
    result = planner.plan(obs_init)
    
    raw_cem_loss = result.prev_elite_losses_mean.tolist()
    # Flatten it just in case it's a list of lists or 2D array
    if len(raw_cem_loss) > 0 and isinstance(raw_cem_loss[0], list):
        cem_loss_history = [item[0] if len(item) > 0 else 0.0 for item in raw_cem_loss]
    else:
        cem_loss_history = raw_cem_loss
        
    print(f"CEM Planner Final Cost: {cem_loss_history[-1]:.4f}")
    
    # ---------------------------------------------------------
    # 3. Comparison
    # ---------------------------------------------------------
    print("\n--- Comparison: Loss Drop ---")
    print(f"{'Iteration':<10} | {'SGD Loss':<15} | {'CEM Loss':<15}")
    print("-" * 45)
    for i in range(epochs):
        sgd_val = sgd_loss_history[i]
        cem_val = cem_loss_history[i] if i < len(cem_loss_history) else cem_loss_history[-1]
        print(f"{i+1:<10} | {sgd_val:<15.4f} | {cem_val:<15.4f}")

if __name__ == "__main__":
    main()
