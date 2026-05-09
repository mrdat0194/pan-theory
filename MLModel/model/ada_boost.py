from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier

# from tensorflow.python.util import deprecation
from sklearn.linear_model import LogisticRegression

# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# deprecation._PRINT_DEPRECATION_WARNINGS = False

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def model_ada(X_train, X_test, Y_train, Y_test):
    logReg = LogisticRegression(max_iter=100, tol=1e-4, C=1e5, penalty='l2', solver='liblinear')
    ada = AdaBoostClassifier(estimator=logReg, n_estimators=100, learning_rate=1)
    vote = VotingClassifier(estimators=[('lr', logReg), ('rf', ada)], voting='soft')
    ndc = vote
    ndc.fit(X_train, Y_train)
    Y_pred = ndc.predict(X_test)

    print('')
    print('Accuracy of boost classifier on test set: {:.2f}'.format(ndc.score(X_test, Y_test)))
    confus_matrix = confusion_matrix(Y_test, Y_pred.round())
    print('')
    print('Confusion matrix: ')
    print(20 * ' ', confus_matrix[0])
    print(20 * ' ', confus_matrix[1])
    print('')
    print(classification_report(Y_test, Y_pred.round()))

    return ndc


def ada_call(X_test, ndc):

    Y_predicted = ndc.predict(X_test)

    return Y_predicted.round()
