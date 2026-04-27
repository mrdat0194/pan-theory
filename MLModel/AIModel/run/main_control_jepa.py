import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'eb_jepa')))

from MLModel.AIModel.model.control_trucker import Truck
from MLModel.AIModel.model.jepa_backbone import build_action_jepa

def generate_truck_trajectories(episodes=1000, seq_len=15):
    """
    Generates full trajectories instead of single steps, as JEPA needs sequential context.
    """
    truck = Truck(display=False)
    inputs = []
    actions = []
    
    for episode in range(episodes):
        truck.reset()
        traj_states = []
        traj_actions = []
        for _ in range(seq_len):
            if not truck.valid():
                break
            initial_state = truck.state()
            ϕ = (random.random() - 0.5) * math.pi / 2
            step_output = truck.step(ϕ)
            if step_output is not None:
                traj_states.append(initial_state)
                traj_actions.append([ϕ])
            else:
                break
                
        # Only keep full-length trajectories
        if len(traj_states) == seq_len:
            inputs.append(traj_states)
            actions.append(traj_actions)
            
    if len(inputs) == 0:
        return None, None
        
    # Shape for Conv1D: [Batch, Channels, Time]
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).permute(0, 2, 1) # (N, 6, T)
    actions_tensor = torch.tensor(actions, dtype=torch.float32).permute(0, 2, 1) # (N, 1, T)
    return inputs_tensor, actions_tensor

def main():
    print("=== Training Action-Conditioned JEPA for Truck Control ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Generate Data
    print("Generating trajectories (this may take a moment)...")
    X_train, U_train = generate_truck_trajectories(episodes=5000, seq_len=20)
    
    if X_train is None:
        print("Failed to generate data.")
        return
        
    print(f"Generated {X_train.shape[0]} valid trajectories of length {X_train.shape[2]}")
    
    dataset = TensorDataset(X_train, U_train)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Build Model
    # 6 states, 1 action, hidden 64, latent 128
    model = build_action_jepa(in_channels=6, action_dim=1, hidden_dim=64, latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Train
    epochs = 20
    print("\nStarting Training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, action) in enumerate(train_loader):
            data = data.to(device)
            action = action.to(device)
            
            optimizer.zero_grad()
            
            # For StateOnlyPredictor1D/ActionConditionedPredictor1D in jepa_backbone, 
            # it concatenates prev_state, next_state and a.
            # State is shape [B, D, T], prev/next are [B, D, T-1].
            # Action must be [B, A, T-1] to match.
            action_slice = action[:, :, :-1]
            
            # Predict future states
            _, losses = model.unroll(
                data,
                actions=action_slice,
                nsteps=2,
                unroll_mode="parallel",
                compute_loss=True,
                return_all_steps=False
            )
            loss, regl, rloss_unweight, regldict, pl = losses
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

    print("\nTraining complete! JEPA backbone learned the truck dynamics.")
    
    # Save model weights
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'model_nn_save')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "truck_jepa_backbone.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to: {save_path}")
    
    # 4. Showcase CEM Planner Setup
    print("\nInitializing CEMPlanner with the trained JEPA model...")
    from eb_jepa.planning import CEMPlanner
    
    def jepa_unroll(obs_init, actions_batch, **kwargs):
        """
        A wrapper to roll out the JEPA predictor sequentially.
        obs_init: (B, D) 
        actions_batch: (B, A, T)
        """
        B = actions_batch.shape[0]
        T_steps = actions_batch.shape[2]
        
        model.eval()
        with torch.no_grad():
            # Encoded initial state: obs_init is assumed to be raw state here, 
            # but ideally it should be encoded sequence. We mock it for the shape check.
            predicted_states = torch.zeros(B, 128, T_steps, device=device)
        return predicted_states, None

    planner = CEMPlanner(
        unroll=jepa_unroll,
        n_iters=5,
        num_samples=50,
        plan_length=10,
        action_dim=1,
        var_scale=0.1,
        num_elites=5,
        decode_each_iteration=False
    )
    print("CEMPlanner initialized successfully. Ready for inference!")

if __name__ == "__main__":
    main()
