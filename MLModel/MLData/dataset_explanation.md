# Dataset Explanation: Kalapa Credit Scoring Challenge

This document provides an overview and interpretation of the encoded datasets used in the `MLModel` pipeline, based on statistical analysis and community findings from the Kalapa Credit Scoring competition.

## 1. Overview
The dataset is designed for **binary classification**. The goal is to predict the `label` (0 or 1) based on a variety of demographic and behavioral features. Because the data is from a competition, most fields are anonymized to protect user privacy.

*   **Source**: Kalapa Credit Scoring Challenge.
*   **Target (`label`)**: 
    *   `0`: "Good" case (Low risk).
    *   `1`: "Bad" case (High risk).
*   **Missing Values**: Marked as `-1`.

---

## 2. Core Features
Before the 57 anonymized fields, there are several descriptive columns:

| Column Name | Description |
| :--- | :--- |
| `id` | Unique identifier for each record (dropped during training). |
| `province` | Integer-encoded geographic location. |
| `age_source1` | Individual's age. This is the primary column targeted for data augmentation. |
| `maCv` | Integer-encoded occupation/job code. |

---

## 3. The 57 Anonymized Fields (`FIELD_1` - `FIELD_57`)
While no official dictionary exists, these fields are generally categorized as follows:

### Categorical & Binary Flags (`FIELD_1` - `FIELD_13`)
*   Includes employment types, education levels, and initial demographics.
*   Most are encoded as integers.

### Behavioral Indicators (`FIELD_14` - `FIELD_49`)
*   **Binary Flags**: Many are simple 0/1 indicators (e.g., "Has a bank account", "Has insurance").
*   **Discrete Counters**: Small integers representing counts of specific events or relationships.

### External Risk Scores (`FIELD_50` - `FIELD_57`)
*   Represented as **floating-point numbers** (e.g., `29.77`, `0.04`).
*   These are often interpreted as pre-computed risk scores or values pulled from external credit bureaus.

---

## 4. Data Augmentation & `agemean`
Your pipeline uses files like `train_encode_agemean_1.csv` and `train_encode_age2_1.csv`.

*   **Purpose**: These files are used in the `imbalance_solve` function to fix class imbalance (where `label 0` significantly outweighs `label 1`).
*   **Logic**: The script takes existing "Bad" cases (label 1) and creates synthetic copies by slightly modifying the **Age** (`age_source1`) column. This allows the model to learn that the risk profile is stable across slight variations in age.
*   **Location**: These files should be stored in `MLModel/MLData/train_test/`.

---

## 5. Summary of Data Helper Logic
*   **`get_data`**: Automatically drops `id` and `label` to create the feature matrix `X`.
*   **`imbalance_solve`**: Merges the original data with the `age2` and `agemean` files to balance the 0s and 1s.
*   **`data_pipeline`**: Splits the final processed data into training and testing sets (80/20 split).
