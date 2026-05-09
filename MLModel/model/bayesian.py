from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import BayesianRidge


def model_bayes(X_train, X_test, Y_train, Y_test):
    bayes = GaussianNB()
    bayes.fit(X_train, Y_train)

    Y_pred = bayes.predict(X_test)

    print('')
    print('Accuracy of bayesian classifier on test set: {:.2f}'.format(bayes.score(X_test, Y_test)))
    confus_matrix = confusion_matrix(Y_test, Y_pred)
    print('')
    print('Confusion matrix: ')
    print(20 * ' ', confus_matrix[0])
    print(20 * ' ', confus_matrix[1])
    print('')
    print(classification_report(Y_test, Y_pred))

    return bayes


def bayes_call(X_test, bayes):

    Y_predicted = bayes.predict(X_test)

    return Y_predicted