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

### Class Imbalance Mitigation Sweep

Because the dataset is highly imbalanced (~0.89% positive class), we conducted a systematic sweep using `run_imbalance_sweep.py` across different mitigation techniques: **SMOTE** (Synthetic Minority Over-sampling Technique), **SMOTEENN** (SMOTE + Edited Nearest Neighbors), and **Cost-Sensitive** class weights.

The benchmarks demonstrate massive performance improvements, particularly for non-linear models like Random Forest:

| Model | Strategy | Optimal Threshold | Accuracy | Precision | Recall | Best F1 (Label 1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **SMOTEENN** | **0.54** | **0.9753** | **0.7590** | **0.6702** | **0.7119** *(+206% vs Baseline)* |
| **Random Forest** | SMOTE | 0.63 | 0.9680 | 0.6409 | 0.6773 | 0.6586 *(+183% vs Baseline)* |
| **Random Forest** | Baseline / Cost-Sensitive | 0.55 | 0.7040 | 0.1319 | 0.9858 | 0.2326 |
| **Logistic Reg.** | SMOTE | 0.58 | 0.8061 | 0.0864 | 0.3404 | 0.1378 *(+18% vs Baseline)* |
| **Logistic Reg.** | Baseline / Cost-Sensitive | 0.73 / 0.72 | 0.6623 | 0.0662 | 0.4894 | 0.1166 |
| **SVM** | SMOTE | 0.75 | 0.6358 | 0.0738 | 0.6064 | 0.1316 *(+13% vs Baseline)* |
| **SVM** | Baseline | 0.50 | 0.6592 | 0.0656 | 0.4894 | 0.1156 |
| **AdaBoost** | SMOTE | 0.76 | 0.8878 | 0.0990 | 0.1809 | 0.1280 *(+10% vs Baseline)* |
| **AdaBoost** | Baseline | 0.66 | 0.6602 | 0.0657 | 0.4894 | 0.1159 |

#### Key Insights from the Sweep:
1. **SMOTEENN combined with Random Forest** is the top performer, achieving a **0.7119 F1-score** and **97.5% accuracy**, which drastically outperforms the Neural Network's best F1 of `0.3146`.
2. **Resampling Techniques (SMOTE/SMOTEENN)** provide significant value by synthetically balancing the representation of the minority class in the training split. This enables Random Forest to learn robust decision boundaries instead of defaulting to high recall with extremely low precision.
3. **Linear and Boosted-Linear Models (Logistic Regression, SVM, AdaBoost with LR base estimator)** show much smaller gains from SMOTE/SMOTEENN. This is because they lack the capacity to model the underlying non-linear interactions in this dataset, even when class distributions are balanced.

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
