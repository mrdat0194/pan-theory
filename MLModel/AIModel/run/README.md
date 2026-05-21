# AI Models Execution Scripts (`/run`)

This directory contains all the main executable scripts for training, testing, and comparing the various Deep Learning and Control algorithms in the `AIModel` module.

## 🎵 Audio JEPA Suite

We have implemented three distinct variants of the Joint-Embedding Predictive Architecture (JEPA) for Audio Emotion Recognition (SER), benchmarking them against the IEMOCAP dataset.

### 1. **Baseline Audio JEPA** (`main_audio_jepa.py`)
The foundational implementation refactored to follow the **LeWorldModel (Le-WM)** architecture.
- **Architecture**: 1D Convolutional Encoder combined with a Transformer-based Autoregressive Predictor (`ARPredictor`).
- **Objective**: Next-latent prediction regularized by **LeJEPA's Sketched Isotropic Gaussian Regularization (SIGReg)**, which removes the need for heuristics like EMA or Variance-Covariance loss.
- **Status**: Currently yields **90.00% accuracy** on the 20/20 train-test split, establishing a highly stable and superior baseline.

### 2. **A-JEPA: Audio-JEPA** (`main_audio_ajepa.py`)
An advanced variant adopting the Masked Patch strategy from Vision-JEPA (V-JEPA).
- **Architecture**: Transformer-based encoder/predictor treating MFCCs as a 2D single-channel image.
- **Mechanism**: **Masked Patch Reconstruction** (typically 60% masking) with an **EMA (Exponential Moving Average) Teacher** network for target stability.
- **Key Feature**: Captures both local spectral details and global temporal correlations.

### 3. **C-JEPA: Causal-JEPA** (`main_audio_cjepa.py`)
A world-model inspired approach focused on object-centric reasoning.
- **Architecture**: `AudioSlotEncoder` + `MaskedSlotPredictor` (Non-Causal Transformer).
- **Mechanism**: **Temporal Slot Masking**. Splits audio into windows (slots), masks random slots, and requires the model to reconstruct them.
- **Key Feature**: Induces relational reasoning between audio segments, modeling the "causal" structure of the sound.

---

## 📊 Benchmark & Comparison

### **Comparison Script** (`compare_audio_jepa.py`)
This script provides a unified environment to benchmark all three models under identical conditions.
- **Command**: `python MLModel/AIModel/run/compare_audio_jepa.py`
- **Standard Config**: 300 Epochs of pre-training, Seed 21, 20 Train / 20 Test sample split.

### **Current Performance Analysis: LeWM-Baseline Stability**
Following the transition to the **Le-WM** architecture, the Baseline model employs `SIGReg` to enforce an isotropic Gaussian distribution in the latent space. This rigorous approach prevents representation collapse without needing EMA networks or complex multi-term VC losses.
- **Efficiency and Generalization**: The combination of a CNN-based encoder (capturing local audio features) with an `ARPredictor` and `SIGReg` generalizes remarkably well, reaching **90.00% accuracy** on the 20-sample evaluation. It balances inductive biases with temporal reasoning.
- **Transformer Scaling**: When training samples are restricted (only 20 training samples for the probe), the data-hungry A-JEPA collapses to **45.00%** accuracy, while C-JEPA stays at **80.00%**. The LeWM-Baseline outperforms both by a large margin while providing a mathematically principled guarantee against representation collapse thanks to `LeJEPA`.

---

## 🛠️ Other JEPA Implementations

- **`main_jepa_anomaly.py`**: MTS-JEPA implementation for anomaly detection (replaces legacy VAE-LSTM). Built for structural vibration data, inspired by the multi-resolution architecture described in [MTS-JEPA (arxiv:2602.04643)](https://arxiv.org/html/2602.04643v1). Utilizes `eb_jepa` backbone with RankFeat/RankWeight noise suppression.
- **`main_control_jepa.py`**: Training action-conditioned JEPA for Truck-and-Trailer dynamics.
- **`main_optimal_jepa.py`**: Benchmarking `CEMPlanner` (Cross-Entropy Method) against standard SGD for optimal control.

## 🚀 Future Improvements

- **Larger Pre-training Datasets**: Scale pre-training to larger unlabeled corpora (e.g., AudioSet or LibriSpeech) to fully unlock the potential of the Transformer-based A-JEPA and C-JEPA.
- **Task-Specific Slots**: Refine C-JEPA slots to represent specific acoustic events (pitch, rhythm, timbre) rather than arbitrary temporal windows.
- **Hybrid Backbones**: Integrate the robustness of the CNN baseline (for local features) with the global reasoning of C-JEPA.
- **Data Augmentation**: Implement advanced SpecAugment and temporal shifting during the JEPA pre-training phase.

---

## Generative & Probabilistic Models

- **`main_vae.py`**: Variational Autoencoder (VAE) for generative latent mapping.
- **`main_bnn.py`**: Bayesian Neural Network (BNN) for uncertainty modeling and confidence intervals.
