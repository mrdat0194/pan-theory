# AI Models Execution Scripts (`/run`)

This directory contains all the main executable scripts for training, testing, and comparing the various Deep Learning and Control algorithms in the `AIModel` module.

## JEPA (Joint-Embedding Predictive Architecture)

- **`main_jepa_anomaly.py`**
  The primary inference script for performing Time-Series Anomaly Detection on structural vibration data (e.g., detecting cracks vs. patches vs. potholes). 
  - **Features**: Uses a pre-trained JEPA backbone to predict future states in the latent space. Calculates the MSE prediction error. 
  - **OOD Integrations**: Supports advanced rank-1 feature removal techniques via CLI flags to dramatically boost detection margins:
    - `--use_rankfeat`: Runs SVD on the latent activations to strip dominant noise.
    - `--use_rankweight`: Modifies the JEPA weights in-memory via SVD to permanently strip the rank-1 component before inference.
  - **Benchmark Results (Separation Margin)**:
    - *Baseline*: `Crack: 0.24` | `Pothole: 0.17` | `Patch: 0.11`
    - *RankFeat*: `Crack: 0.23` | `Pothole: 0.15` | `Patch: 0.05` *(Massive drop in noise)*
    - *RankWeight*: `Crack: 0.77` | `Pothole: 0.62` | `Patch: 0.46` *(Massive increase in anomaly scale)*
    - *Both Flags*: `Crack: 0.30` | `Pothole: 0.22` | `Patch: 0.09` *(Separation gap triples from baseline)*

- **`main_control_jepa.py`**
  The training pipeline for the Action-Conditioned JEPA model. 
  - **Features**: It generates a large synthetic dataset of Truck-and-Trailer kinematic trajectories by interacting with `control_trucker.py`. It then self-supervisedly trains the JEPA encoder/predictor to understand the latent dynamics conditioned on steering actions.
  - **Output**: Saves the weights to `model_nn_save/truck_jepa_backbone.pth`.

- **`main_optimal_jepa.py`**
  A direct mathematical benchmarking script for Planning algorithms.
  - **Features**: Compares the old analytical Backpropagation (SGD) control pathing against the advanced Cross-Entropy Method (`CEMPlanner`) from the `eb_jepa` package. Both methods optimize the exact same analytical equation (`cost_logsumexp`) to prove which method drops the loss faster and avoids vanishing gradients.

- **`main_jepa_timeseries.py`**
  A basic setup script for applying JEPA to standard univariate time-series data.

## Generative & Probabilistic Models

- **`main_vae.py`**
  Execution script for the Variational Autoencoder (VAE). Tests generative capacity and latent space distribution mappings.

- **`main_bnn.py`**
  Execution script for the Bayesian Neural Network (BNN). Designed to model uncertainty and provide probabilistic confidence intervals around predictions.
