# Data Pipeline Directory

This directory contains key dataset files, pipeline scripts, helper modules, and analysis notebooks used to preprocess, clean, augment, and split data for machine learning models.

---

## Directory Contents

| Component | Description |
| :--- | :--- |
| [data_helper.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/MLModel/data_pipeline/data_helper.py) | Utility library containing functions for data loading, deduplication, feature scaling, train-test splits, class-imbalance resolution, and prediction threshold tuning. |
| [faceforensics_download_v4.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/MLModel/data_pipeline/faceforensics_download_v4.py) | Downloader script for the FaceForensics dataset. |
| [imbalance.ipynb](file:///C:/Users/mrdat/PycharmProjects/pan-theory/MLModel/data_pipeline/imbalance.ipynb) | Jupyter Notebook analyzing class-imbalance solutions on the KDD2004 dataset using `imbalanced-learn` (SMOTE, SMOTEENN, Balanced Random Forest, Cost-Sensitive classification) with native threshold-sweep visualizations. |
| [kdd2004.csv](file:///C:/Users/mrdat/PycharmProjects/pan-theory/MLModel/data_pipeline/kdd2004.csv) | The protein homology dataset (extremely imbalanced, ~0.89% positive class) analyzed in `imbalance.ipynb`. |

---

## Detailed Component Highlights

### 1. data_helper.py
Includes:
* **Data Cleansing**: Duplicates removal using row hashes.
* **Feature Selection**: RFE (Recursive Feature Elimination) wrapping linear models.
* **Custom Imbalance Solvers**: Domain-specific heuristics (`imbalance_solve`) to artificially augment minority class points (e.g., perturbing attributes like age).
* **Evaluation Utilities**: `find_best_threshold` finds the decision threshold that maximizes the F1-Score on out-of-fold probabilities.

### 2. imbalance.ipynb (Class Imbalance Solutions)
Explores class imbalance mitigation for the KDD2004 dataset:
* **Resampling**: Uses Synthetic Minority Over-sampling (`SMOTE`) and combined Edited Nearest Neighbors clean-up (`SMOTEENN`).
* **Ensemble Learning**: Compares performance of the `BalancedRandomForestClassifier`.
* **Discrimination Sweep**: Maps F1, Precision, and Recall curves across decision thresholds from `0.0` to `1.0` to establish optimal decision boundaries.
