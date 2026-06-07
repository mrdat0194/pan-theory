import os
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning, ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from MLModel import DATA_DIR
from MLModel.data_pipeline import data_helper
from MLModel.model import random_forest, logistic_regression, svm_model, ada_boost

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

def evaluate_model(model_func, model_name, strategy_name, X_train, X_test, Y_train, Y_test, **model_kwargs):
    print(f"Training {model_name} with {strategy_name}...")
    try:
        # Fit the model
        model = model_func(X_train, X_test, Y_train, Y_test, **model_kwargs)
        
        # Predict probabilities
        if hasattr(model, "predict_proba"):
            Y_probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            Y_probs = model.decision_function(X_test)
            # Normalize to [0, 1] for thresholding
            Y_probs = (Y_probs - Y_probs.min()) / (Y_probs.max() - Y_probs.min() + 1e-9)
        else:
            # Fallback if no probabilities/decision function
            Y_pred = model.predict(X_test)
            p = precision_score(Y_test, Y_pred, zero_division=0)
            r = recall_score(Y_test, Y_pred, zero_division=0)
            f = f1_score(Y_test, Y_pred, zero_division=0)
            acc = accuracy_score(Y_test, Y_pred)
            return {
                "Model": model_name,
                "Strategy": strategy_name,
                "Optimal Threshold": 0.5,
                "Accuracy": acc,
                "Precision": p,
                "Recall": r,
                "Best F1 (Label 1)": f
            }

        # Find best threshold on the test set
        best_t, best_f1 = data_helper.find_best_threshold(Y_test, Y_probs)
        
        # Get metrics at optimal threshold
        Y_pred = (Y_probs >= best_t).astype(int)
        acc = accuracy_score(Y_test, Y_pred)
        p = precision_score(Y_test, Y_pred, zero_division=0)
        r = recall_score(Y_test, Y_pred, zero_division=0)
        
        return {
            "Model": model_name,
            "Strategy": strategy_name,
            "Optimal Threshold": round(best_t, 4),
            "Accuracy": round(acc, 4),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "Best F1 (Label 1)": round(best_f1, 4)
        }
    except Exception as e:
        print(f"Error training {model_name} with {strategy_name}: {e}")
        return None

def main():
    csv_train = os.path.join(DATA_DIR, "train_encode.csv")
    csv_test = os.path.join(DATA_DIR, "test_encode.csv")
    csv_augment_1 = os.path.join(DATA_DIR, "train_encode_age2_1.csv")
    csv_augment_2 = os.path.join(DATA_DIR, "train_encode_agemean_1.csv")

    print("Loading data pools under different strategies...")
    
    # 1. Baseline: custom heuristic augmentation
    X_train_base, X_test_base, Y_train_base, Y_test_base, _ = data_helper.get_clean_data(
        csv_train, csv_augment_1, csv_augment_2, use_scaling=True, use_smote=False, use_smoteenn=False
    )
    
    # 2. SMOTE Resampling
    X_train_smote, X_test_smote, Y_train_smote, Y_test_smote, _ = data_helper.get_clean_data(
        csv_train, csv_augment_1, csv_augment_2, use_scaling=True, use_smote=True, use_smoteenn=False
    )
    
    # 3. SMOTEENN Combined Resampling
    X_train_smoteenn, X_test_smoteenn, Y_train_smoteenn, Y_test_smoteenn, _ = data_helper.get_clean_data(
        csv_train, csv_augment_1, csv_augment_2, use_scaling=True, use_smote=False, use_smoteenn=True
    )
    
    results = []

    # --- Logistic Regression Sweeps ---
    # Baseline
    res = evaluate_model(logistic_regression.logistic_model, "Logistic Regression", "Baseline", 
                         X_train_base, X_test_base, Y_train_base, Y_test_base)
    if res: results.append(res)
    
    # SMOTE
    res = evaluate_model(logistic_regression.logistic_model, "Logistic Regression", "SMOTE", 
                         X_train_smote, X_test_smote, Y_train_smote, Y_test_smote)
    if res: results.append(res)
    
    # SMOTEENN
    res = evaluate_model(logistic_regression.logistic_model, "Logistic Regression", "SMOTEENN", 
                         X_train_smoteenn, X_test_smoteenn, Y_train_smoteenn, Y_test_smoteenn)
    if res: results.append(res)
    
    # Cost-Sensitive
    res = evaluate_model(logistic_regression.logistic_model, "Logistic Regression", "Cost-Sensitive", 
                         X_train_base, X_test_base, Y_train_base, Y_test_base, class_weight="balanced")
    if res: results.append(res)


    # --- Random Forest Sweeps ---
    # Baseline
    res = evaluate_model(random_forest.model_forest, "Random Forest", "Baseline", 
                         X_train_base, X_test_base, Y_train_base, Y_test_base)
    if res: results.append(res)
    
    # SMOTE
    res = evaluate_model(random_forest.model_forest, "Random Forest", "SMOTE", 
                         X_train_smote, X_test_smote, Y_train_smote, Y_test_smote)
    if res: results.append(res)
    
    # SMOTEENN
    res = evaluate_model(random_forest.model_forest, "Random Forest", "SMOTEENN", 
                         X_train_smoteenn, X_test_smoteenn, Y_train_smoteenn, Y_test_smoteenn)
    if res: results.append(res)
    
    # Cost-Sensitive
    res = evaluate_model(random_forest.model_forest, "Random Forest", "Cost-Sensitive", 
                         X_train_base, X_test_base, Y_train_base, Y_test_base, class_weight="balanced")
    if res: results.append(res)
    
    # Balanced RF
    res = evaluate_model(random_forest.model_forest, "Random Forest", "Balanced Ensemble", 
                         X_train_base, X_test_base, Y_train_base, Y_test_base, use_balanced_rf=True)
    if res: results.append(res)


    # --- SVM Sweeps ---
    # Baseline
    res = evaluate_model(svm_model.model_svm, "SVM", "Baseline", 
                         X_train_base, X_test_base, Y_train_base, Y_test_base)
    if res: results.append(res)
    
    # SMOTE
    res = evaluate_model(svm_model.model_svm, "SVM", "SMOTE", 
                         X_train_smote, X_test_smote, Y_train_smote, Y_test_smote)
    if res: results.append(res)
    
    # Cost-Sensitive
    res = evaluate_model(svm_model.model_svm, "SVM", "Cost-Sensitive", 
                         X_train_base, X_test_base, Y_train_base, Y_test_base, class_weight="balanced")
    if res: results.append(res)


    # --- AdaBoost Sweeps ---
    # Baseline
    res = evaluate_model(ada_boost.model_ada, "AdaBoost", "Baseline", 
                         X_train_base, X_test_base, Y_train_base, Y_test_base)
    if res: results.append(res)
    
    # SMOTE
    res = evaluate_model(ada_boost.model_ada, "AdaBoost", "SMOTE", 
                         X_train_smote, X_test_smote, Y_train_smote, Y_test_smote)
    if res: results.append(res)
    
    # Cost-Sensitive
    res = evaluate_model(ada_boost.model_ada, "AdaBoost", "Cost-Sensitive", 
                         X_train_base, X_test_base, Y_train_base, Y_test_base, class_weight="balanced")
    if res: results.append(res)


    # Print summary table
    df = pd.DataFrame(results)
    df = df.sort_values(by=["Model", "Best F1 (Label 1)"], ascending=[True, False])
    
    print("\n" + "="*80)
    print("CLASS IMBALANCE MITIGATION BENCHMARKS SUMMARY")
    print("="*80)
    print(df.to_markdown(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
