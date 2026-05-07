# Machine Learning Models (`MLModel/run`)

This directory contains the training and execution scripts for direct predictive modeling, specifically focusing on supervised learning architectures for structured data.

## Current Models & Comparison

We recently upgraded the core neural network architecture to the **Dynamic Interaction Neural Network (DINN)** (adapted from the DWLR methodology). This network dynamically generates weights for each feature and explicitly calculates pairwise feature interactions.

### Logistic Regression Baseline (`main_logistic.py`)
- **Test Accuracy:** ~77.00%
- **Pros:** Fast, highly interpretable, robust to unscaled data.
- **Cons:** Fails to capture complex, non-linear relationships and interactions between features unless manually engineered.

### Dynamic Interaction Neural Network (`train_nn.py`)
- **Test Accuracy:** ~87.88%
- **Test AUC:** ~0.9244
- **Pros:** Massive jump in accuracy. Automatically discovers and weighs complex interactions between features dynamically.
- **Cons:** Highly sensitive to the scale of the input data. (We solved this by implementing `StandardScaler` in the pipeline to prevent exploding gradient issues during interaction multiplication).

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
