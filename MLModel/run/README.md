# Machine Learning Models (`MLModel/run`)

This directory contains the training and execution scripts for direct predictive modeling, specifically focusing on supervised learning architectures for structured data.

## Current Models & Comparison

We recently executed a comprehensive sweep of all models in this directory. After fixing data leakage issues and implementing automated threshold tuning (F1-Optimization), the current realistic benchmarks are:

| Model | Accuracy | Best F1 (Label 1) | Threshold | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Neural Network** | **0.91** | **0.3146** | 0.65 | **DINN Architecture**. Best performer after F1 tuning. |
| **Random Forest** | 0.70 | 0.2326 | 0.78 | Balanced detection with high recall. |
| **SVM** | 0.71 | 0.1600 | - | Solid accuracy after proper scaling. |
| **Bayesian (NB)** | 0.66 | 0.1121 | 0.41 | Baseline Gaussian Naive Bayes. |
| **Logistic Reg.** | 0.58 | 0.1166 | 0.73 | Interpretable but weaker on this complex set. |
| **AdaBoost** | 0.58 | 0.1100 | - | Performance mirrors Logistic Regression. |

### Core Architectures

#### Logistic Regression Baseline (`main_logistic.py`)
- **Test Accuracy:** 58.00%
- **Best F1-Score:** 0.1166 (Threshold: 0.73)
- **Pros:** Fast, highly interpretable.
- **Cons:** Struggles with the highly imbalanced, non-linear feature set.

#### Dynamic Interaction Neural Network (`train_nn.py`)
- **Test Accuracy:** 91.14%
- **Best F1-Score:** 0.3146 (Threshold: 0.65)
- **Pros:** Automatically discovers and weighs complex interactions. Significantly higher F1-score than linear baselines after threshold tuning.
- **Cons:** Requires precise threshold tuning for imbalanced sets.

---

## Comparison with `AIModel/run`

While `MLModel/run` focuses on **Supervised, Direct Predictive Learning** (mapping $X \rightarrow Y$ directly), the `AIModel/run` directory focuses heavily on **Self-Supervised and Generative Architectures** (like VAEs, BNNs, and various JEPA implementations). 

- **`MLModel`** is optimized for high accuracy on labeled, tabular datasets.
- **`AIModel`** is optimized for representation learning, handling raw/unstructured data (like audio or time-series), and building latent "world models."

## The Future: Adapting JEPA for Structured Data

Joint-Embedding Predictive Architectures (JEPA) have shown incredible promise in `AIModel` for learning robust representations by predicting missing parts of the input in a latent space, rather than trying to reconstruct the raw pixels/data (like an Autoencoder).

**Why adapt JEPA here in the future?**
1. **Semi-Supervised Learning on Tabular Data:** If we have massive amounts of unlabelled tabular data (e.g., user events, logs) and only a few labeled target rows, we can train a Tabular-JEPA to learn the underlying "physics" or patterns of the data in an unsupervised manner.
2. **Combining JEPA with DINN:** We could use a JEPA encoder to map noisy raw features into a clean, abstract latent space. We could then attach our **DINN** as the predictive head on top of the JEPA representations. This would give us the best of both worlds: JEPA's robust, noise-resistant feature extraction combined with DINN's explicit interaction modeling for the final prediction!
