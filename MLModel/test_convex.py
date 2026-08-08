import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from model.logistic_regression import logistic_model
from model.svm_model import model_svm

def main():
    print("Generating synthetic classification dataset...")
    X, y = make_classification(n_samples=500, n_features=20, n_informative=5, 
                               n_redundant=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n" + "="*50)
    print("Testing Custom FISTA Logistic Regression (L1-regularized)")
    print("="*50)
    # The logistic_model wrapper now calls FISTALogisticRegression internally
    log_reg_model = logistic_model(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*50)
    print("Testing Custom ADMM SVM (L2-regularized Hinge Loss)")
    print("="*50)
    # The model_svm wrapper now calls ADMMSVM internally
    svm_clf_model = model_svm(X_train, X_test, y_train, y_test)
    
    print("\nAll custom convex optimization models ran successfully.")

if __name__ == "__main__":
    main()
