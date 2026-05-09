from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def model_forest(X_train, X_test, Y_train, Y_test):

    forest = RandomForestClassifier()
    forest.fit(X_train, Y_train)
    Y_pred = forest.predict(X_test)

    print('')
    print('Accuracy of random_forest classifier on test set: {:.2f}'.format(forest.score(X_test, Y_test)))
    confus_matrix = confusion_matrix(Y_test, Y_pred)
    print('')
    print('Confusion matrix: ')
    print(20 * ' ', confus_matrix[0])
    print(20 * ' ', confus_matrix[1])
    print('')
    print(classification_report(Y_test, Y_pred))

    return forest


def forest_call(X_test, forest):

    Y_predicted = forest.predict(X_test)

    return Y_predicted