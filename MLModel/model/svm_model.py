from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def model_svm(X_train, X_test, Y_train, Y_test):

    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, Y_train)
    Y_pred = svm.predict(X_test)

    print('')
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, Y_test)))
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