import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Import custom SOTA convex optimization solver
from .admm_svm import ADMMSVM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def model_svm(X_train, X_test, Y_train, Y_test, class_weight=None):
    # Note: Custom ADMM doesn't natively support class_weight right now
    # We can use C to control regularization strength.
    svm = ADMMSVM(C=1.0, rho=1.0, max_iter=1000, tol=1e-4)
    svm.fit(X_train, Y_train)
    
    Y_pred = svm.predict(X_test)
    
    # Map back -1 to 0 for sklearn metrics if original labels were {0, 1}
    # This prevents classification_report from crashing if it expects 0 and gets -1.
    if set(np.unique(Y_test)).issubset({0, 1}) and set(np.unique(Y_pred)).issubset({-1, 1}):
        Y_pred = np.where(Y_pred == -1, 0, 1)

    print('')
    print('Accuracy of ADMM SVM classifier on test set: {:.2f}'.format(svm.score(X_test, Y_test)))
    confus_matrix = confusion_matrix(Y_test, Y_pred)
    print('')
    print('Confusion matrix: ')
    print(20 * ' ', confus_matrix[0])
    print(20 * ' ', confus_matrix[1])
    print('')
    print(classification_report(Y_test, Y_pred))

    return svm

def svm_call(X_test, svm):
    Y_predicted = svm.predict(X_test)
    return Y_predicted