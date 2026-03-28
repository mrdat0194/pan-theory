from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, RFE, chi2
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression

def get_unique(X_matrix, y_vector):
    '''
    x = [[1, 2, 3],
     [4, 5, 6],
     [1, 2, 3]]

    y = [1,
         2,
         1]

    initial_number_of_data_points = len(x)
    x, y = get_unique(x, y)
    data_points_removed = initial_number_of_data_points - len(x)
    print("Number of duplicates removed:", data_points_removed )
    :param X_matrix:
    :param y_vector:
    :return:
    '''
    Xy = list(set(list(zip([tuple(x) for x in X_matrix], y_vector))))
    X_matrix = [list(l[0]) for l in Xy]
    y_vector = [l[1] for l in Xy]
    return X_matrix, y_vector

#
# x, y = get_unique(x, y)
# data_points_removed = initial_number_of_data_points - len(x)
# print("Number of duplicates removed:", data_points_removed )
def get_unique(X_matrix, y_vector):
    Xy = list(set(list(zip([tuple(x) for x in X_matrix], y_vector))))
    X_matrix = [list(l[0]) for l in Xy]
    y_vector = [l[1] for l in Xy]
    return X_matrix, y_vector


def feature_selection(X, Y, n_feature):
    model = LinearRegression()
    rfe = RFE(model, n_feature)
    fit = rfe.fit(X, Y)
    # test = SelectKBest(score_func=chi2, k=n_feature)
    # fit = test.fit(X, Y)
    return fit


def get_data(link):
    data = pd.read_csv(link)
    # data = pd.read_excel(link)
    data = data.drop_duplicates(subset=data.columns.difference(['label']))
    Y = data['label'].values
    data.drop(['id', 'label'], axis=1, inplace=True)
    train_data = data.values
    X = []
    for i in range(len(train_data)):
        X.append(train_data[i])
    X = np.asarray(X)
    Y = np.asarray(Y)
    Y.reshape(-1, 1)

    return X, Y


def get_data_test(link):

    data = pd.read_csv(link)
    ID = data['id'].values
    data.drop(['id'], axis=1, inplace=True)

    test_data = data.values
    X = []
    for i in range(len(test_data)):
        X.append(test_data[i])
    X = np.array(X)
    ID = np.array(ID)
    return X, ID


def imbalance_solve_v2(X, Y, X_augment_1, Y_augment_1, X_augment_2, Y_augment_2):

    X_extend = np.concatenate((X, X_augment_1, X_augment_2))
    Y_extend = np.concatenate((Y, Y_augment_1, Y_augment_2))

    X_label1 = []
    len_label0 = 0

    X_final = list(X_extend.copy())
    Y_final = list(Y_extend.copy())

    for i in range(len(Y_extend)):
        if Y_extend[i] == 1:
            X_label1.append(X_extend[i])
        else:
            len_label0 += 1

    X_augment = []
    Y_augment = []

    for age in range(1, int(len_label0 / len(X_label1))):
        for i in range(len(X_label1)):
            X_row = X_label1[i].copy()
            X_row[1] = X_row[1] + age
            X_augment.append(X_row)
            Y_augment.append(1)

    X_augment = np.array(X_augment)
    Y_augment = np.array(Y_augment)
    Y_augment.reshape(-1, 1)
    X_final.extend(X_augment)
    Y_final.extend(Y_augment)
    X_final = np.array(X_final)
    Y_final = np.array(Y_final)

    return X_final, Y_final


def imbalance_solve(X, Y, X_augment_1, Y_augment_1, X_augment_2, Y_augment_2, rm_values, rm_thres=0.5):

    X_extend = np.concatenate((X, X_augment_1, X_augment_2))
    Y_extend = np.concatenate((Y, Y_augment_1, Y_augment_2))

    X_final = []
    Y_final = []
    X_label1 = []
    len_label0 = 0

    for i in range(len(Y_extend)):
        if Y_extend[i] == 1:
            X_final.append(X_extend[i])
            Y_final.append(Y_extend[i])
            X_label1.append(X_extend[i])
        else:
            X_row = X_extend[i].copy()
            X_row = list(X_row)
            if X_row.count(rm_values) < rm_thres * len(X_row):
                X_final.append(X_extend[i])
                Y_final.append(Y_extend[i])
                len_label0 += 1
    # for i in range(len(Y_extend)):
    #     X_row = X_extend[i].copy()
    #     X_row = list(X_row)
    #     if X_row.count(rm_values) < rm_thres * len(X_row):
    #         X_final.append(X_extend[i])
    #         Y_final.append(Y_extend[i])
    #         if Y_extend[i] == 1:
    #             X_label1.append(X_extend[i])
    #         else:
    #             len_label0 += 1

    X_augment = []
    Y_augment = []

    for age in range(1, int(len_label0 / len(X_label1))):
        for i in range(len(X_label1)):
            X_row = X_label1[i].copy()
            X_row[1] = X_row[1] + age
            X_augment.append(X_row)
            Y_augment.append(1)

    for i in range(len(X_label1)):
        X_row = X_label1[i].copy()
        if X_row[9] == 1:
            X_row[9] = 0
            X_augment.append(X_row)
            Y_augment.append(1)
        elif X_row[9] == 0:
            X_row[9] = 1
            X_augment.append(X_row)
            Y_augment.append(1)

    for i in range(len(X_label1)):
        X_row = X_label1[i].copy()
        if X_row[0] == 0:
            X_row[0] = 1
            X_augment.append(X_row)
            Y_augment.append(1)
        elif X_row[0] == 1:
            X_row[0] = 0
            X_augment.append(X_row)
            Y_augment.append(1)
        elif X_row[0] == -1:
            X_row[0] = 0
            X_augment.append(X_row)
            Y_augment.append(1)

    X_augment = np.array(X_augment)
    Y_augment = np.array(Y_augment)

    Y_augment.reshape(-1, 1)
    X_final.extend(X_augment)
    Y_final.extend(Y_augment)
    X_final = np.array(X_final)
    Y_final = np.array(Y_final)

    print(len(Y_final[Y_final == 1]), len(Y_final[Y_final == 0]))

    return X_final, Y_final


def remove_duplicate(X_final, Y_final):
    data = pd.DataFrame(X_final)
    data.insert(0, 'label', Y_final)
    data = data.drop_duplicates(subset=data.columns.difference(['label']), keep='last')

    Y = data['label'].values
    data.drop(['label'], axis=1, inplace=True)
    train_data = data.values
    X_0 = []
    X_1 = []
    Y_1 = []
    Y_0 = []
    for i in range(len(train_data)):
        if Y[i] == 1:
            X_1.append(train_data[i])
            Y_1.append(1)
        elif Y[i] == 0:
            X_0.append(train_data[i])

    X_0_resample = resample(np.array(X_0), n_samples=len(X_1))
    for i in range(len(X_0_resample)):
        Y_0.append(0)
    # print(np.array(Y_0))

    X = np.concatenate((np.array(X_0_resample), X_1))
    print(len(X))
    Y = np.concatenate((Y_0, Y_1))
    print(len(Y_0), len(Y_1))
    # X = np.asarray(X)
    # Y = np.asarray(Y)
    #
    # Y.reshape(-1, 1)

    return X, Y


def data_pipeline(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    return X_train, X_test, Y_train, Y_test


def data_pipeline_nn(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)


    Y_train = to_categorical(Y_train, num_classes=2)
    Y_test = to_categorical(Y_test, num_classes=2)
    Y_val = to_categorical(Y_val, num_classes=2)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val

