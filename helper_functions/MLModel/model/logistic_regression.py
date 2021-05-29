from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow.python.util import deprecation
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
from sklearn.ensemble import BaggingRegressor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


def logistic_model(X_train, X_test, Y_train, Y_test):

    logReg = LogisticRegression(max_iter=10000, tol=1e-4, C=1e5,penalty='l2', solver='liblinear')
    # logReg = LogisticRegression(max_iter=10000, tol=1e-4, C=1e5, penalty='l2')
    # logReg = BaggingRegressor(logReg, n_estimators=10, bootstrap=True)
    logReg.fit(X_train, Y_train)
    Y_pred = logReg.predict(X_test)

    print('')
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logReg.score(X_test, Y_test)))
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

