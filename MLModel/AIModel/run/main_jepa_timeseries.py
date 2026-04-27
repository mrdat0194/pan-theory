import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
from torch.utils.data import DataLoader, TensorDataset
from MLModel.AIModel.model.jepa_backbone import model_jepa, jepa_call, compute_anomaly_score

def main():
    print("Testing JEPA Backbone for Time-Series Anomaly Detection...")
    
    # Create dummy time-series data: [Batch, Channels, Time]
    # Normal data: sine waves
    t = torch.linspace(0, 10, 100)
    normal_data = torch.sin(t).unsqueeze(0).repeat(100, 1, 1) # [100, 1, 100]
    
    # Anomaly data: sine waves with a sudden spike at the end
    anomaly_data = torch.sin(t).unsqueeze(0).repeat(20, 1, 1)
    anomaly_data[:, :, 80:90] += 5.0 # Add shock/spike
    
    dataset = TensorDataset(normal_data)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    print("Training JEPA on normal data...")
    jepa_model = model_jepa(train_loader, epochs=5, lr=1e-3, in_channels=1, steps=2)
    
    print("\nExtracting Features (Backbone Call)...")
    features = jepa_call(jepa_model, [normal_data[:5]])
    print(f"Extracted feature shape: {features.shape} (Expected: [5, D, T])")
    
    print("\nComputing Anomaly Scores...")
    normal_scores = compute_anomaly_score(jepa_model, normal_data[:5])
    anomaly_scores = compute_anomaly_score(jepa_model, anomaly_data[:5])
    
    print(f"Average Anomaly Score on Normal Data: {normal_scores.mean().item():.4f}")
    print(f"Average Anomaly Score on Anomaly Data: {anomaly_scores.mean().item():.4f}")
    
    if anomaly_scores.mean() > normal_scores.mean():
        print("Success! JEPA successfully detects the anomaly via higher prediction error.")
    else:
        print("Warning: Anomaly score is not higher than normal score. Training might need more epochs.")

if __name__ == "__main__":
    main()
