import os
import sys

# Import custom SOTA convex optimization solver
from .fista_logistic import FISTALogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def logistic_model(X_train, X_test, Y_train, Y_test, class_weight=None):
    # Note: Custom FISTA doesn't natively support class_weight right now, 
    # but we can configure lambda_reg to control sparsity.
    logReg = FISTALogisticRegression(lambda_reg=0.01, max_iter=2000, tol=1e-5)
    
    logReg.fit(X_train, Y_train)
    Y_pred = logReg.predict(X_test)

    print('')
    print('Accuracy of FISTA logistic regression on test set: {:.2f}'.format(logReg.score(X_test, Y_test)))
    confus_matrix = confusion_matrix(Y_test, Y_pred)
    print('')
    print('Confusion matrix: ')
    print(20 * ' ', confus_matrix[0])
    print(20 * ' ', confus_matrix[1])
    print('')
    print(classification_report(Y_test, Y_pred))

    return logReg

def logistic_call(X_test, logReg):
    Y_predicted = logReg.predict(X_test)
    return Y_predicted
