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

## 👁️ Vision-JEPA 2 Suite

We have integrated the official **V-JEPA 2** architecture (`facebookresearch/vjepa2`) to handle spatiotemporal video/image patches for downstream Object Detection tasks. The datasets for these tasks have been centralized into `MLModel/data` using our unified `data_pipeline`.

### 1. **Gun Detection** (`main_vjepa2_gun.py`)
- **Dataset**: `MLModel/data/WeaponS` (contains bounding box annotations).
- **Architecture**: V-JEPA 2 Backbone + Localization Head.
- **Objective**: Replaces legacy SSD/YOLO methodologies for detecting weapons in frames.

### 2. **Fire/Smoke Detection** (`main_vjepa2_fire.py`)
- **Dataset**: `MLModel/data/FireSmoke` (FireSense, FurgFire, etc.).
- **Architecture**: V-JEPA 2 Backbone + Multi-class Classification Head.
- **Objective**: Categorizes surveillance frames into Flame vs Smoke vs Safe.

### 🔮 Future Action: Face-Recognition Migration
In the future, the `face-recognition` repository will be migrated to the V-JEPA 2 backbone:
1. **SSL Pre-training**: Unsupervised patch masking on unlabeled face datasets (e.g., WIDER FACE) to learn geometric face embeddings.
2. **Supervised Fine-Tuning**: Attaching the ArcFace (Additive Angular Margin Loss) head to the frozen V-JEPA 2 encoder for 1:1 Cosine Similarity matching.

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

---

## ⚖️ Causal Machine Learning Suite

We have implemented three deep causal ML scripts that upgrade standard linear inference with Transformer encoders, allowing estimation under non-linear confounding and complex dynamics. Each matches a baseline from `Bayesian/someMethod/`.

### 1. **Deep Difference-in-Differences (Deep DiD)** (`deep_did.py`)
- **Classical Problem**: Assumes linear parallel trends between control and treatment groups.
- **Deep/ML Solution**: A Transformer encoder models control-unit pre-treatment time-series to reconstruct non-linear counterfactual paths.
- **Use Case**: Multi-timestep panel data with non-linear baseline trends.

### 2. **Deep Instrumental Variables (Deep IV)** (`deep_iv.py`)
- **Classical Problem**: 2SLS is biased under non-linear instrument-to-treatment relationships or interactions.
- **Deep/ML Solution**: Double Machine Learning (DML) using a Stage-1 Transformer (`Z -> X_hat`) and a Stage-2 MLP (`X_hat -> Y`).
- **Use Case**: Confounded systems with non-linear instrumental interactions.

### 3. **Neural Regression Discontinuity Design (Neural RDD)** (`deep_rdd.py`)
- **Classical Problem**: Linear RDD fits near cutoffs are highly sensitive to bandwidth size and outcome non-linearities.
- **Deep/ML Solution**: Dual independent Transformers with Fourier feature expansion to flexibly fit curves on both sides of the boundary.
- **Use Case**: Discontinuous treatment assignment with non-linear outcome functions.

### 📊 Dataset Structures & Stage Meanings

These three models solve different causal problems and use **completely different, independent datasets**:

* **Deep DiD Dataset**: **Panel Data** (Time-Series) tracked for the same units over pre-treatment and post-treatment intervals.
* **Deep RDD Dataset**: **Cross-Sectional Data** where treatment assignment is strictly and deterministically based on whether a **Running Variable ($x$)** crosses a cutoff (e.g. $x \ge 0$). No instrument $Z$ is present.
* **Deep IV Dataset**: **Cross-Sectional Data** utilizing a helper **Instrument ($Z$)** to clean an endogenous treatment ($X$) that is otherwise corrupted by unobserved confounders. Because of this complexity, it uses a two-stage approach:
  * **Stage 1 (`stage1_transformer.pth`)**: A Transformer learns the relationship between the instrument and treatment (`Z -> X_hat`). This outputs a "clean" treatment prediction ($\hat{X}$) free from hidden confounder bias.
  * **Stage 2 (`stage2_mlp.pth`)**: An MLP uses the clean predicted treatment ($\hat{X}$) to map to the outcome ($Y$), yielding the true unbiased causal effect.

---

## 📈 Controlling W&B Logging

All executable scripts in this folder integrate Weights & Biases (W&B) for experiment tracking. You can control logging behavior as follows:

1. **Enable Logging via CLI**:
   Pass the `--use_wandb` flag to any causal run script:
   ```powershell
   python MLModel/AIModel/run/deep_did.py --epochs 10 --use_wandb
   ```
2. **Disable Logging via Environment**:
   To completely block network requests or disable W&B locally (e.g., in offline or CI environments), set:
   ```powershell
   $env:WANDB_DISABLED="true"  # PowerShell
   # or
   export WANDB_DISABLED=true  # Linux/macOS
   ```
3. **W&B Offline Mode**:
   To save logs locally without syncing to the cloud, run:
   ```powershell
   wandb offline
   ```
   To sync offline logs later:
   ```powershell
   wandb sync
   ```

