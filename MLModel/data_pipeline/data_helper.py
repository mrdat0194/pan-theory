from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, RFE, chi2
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.preprocessing import StandardScaler

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
    seen = set()
    new_X, new_y = [], []
    for x, y in zip(X_matrix, y_vector):
        t = (tuple(x), y)
        if t not in seen:
            seen.add(t)
            new_X.append(list(x))
            new_y.append(y)
    return new_X, new_y


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
    X = data.values
    return X, Y


def get_data_test(link):

    data = pd.read_csv(link)
    ID = data['id'].values
    data.drop(['id'], axis=1, inplace=True)
    X = data.values
    return X, ID


def imbalance_solve_v2(X, Y, X_augment_1, Y_augment_1, X_augment_2, Y_augment_2):

    X_extend = np.concatenate((X, X_augment_1, X_augment_2))
    Y_extend = np.concatenate((Y, Y_augment_1, Y_augment_2))

    mask_1 = (Y_extend == 1)
    X_label1 = X_extend[mask_1]
    len_label0 = np.sum(~mask_1)
    len_label1 = len(X_label1)

    X_augment = []
    Y_augment = []

    if len_label1 > 0:
        multiplier = int(len_label0 / len_label1)
        for age in range(1, multiplier):
            X_aug_batch = X_label1.copy()
            X_aug_batch[:, 1] += age
            X_augment.append(X_aug_batch)
            Y_augment.append(np.ones(len_label1, dtype=Y_extend.dtype))

    if X_augment:
        X_augment = np.vstack(X_augment)
        Y_augment = np.concatenate(Y_augment)
        X_final = np.concatenate((X_extend, X_augment))
        Y_final = np.concatenate((Y_extend, Y_augment))
    else:
        X_final = X_extend
        Y_final = Y_extend

    return X_final, Y_final


def imbalance_solve(X, Y, X_augment_1, Y_augment_1, X_augment_2, Y_augment_2, rm_values, rm_thres=0.5):

    X_extend = np.concatenate((X, X_augment_1, X_augment_2))
    Y_extend = np.concatenate((Y, Y_augment_1, Y_augment_2))

    mask_1 = (Y_extend == 1)
    
    rm_counts = np.sum(X_extend == rm_values, axis=1)
    keep_mask_0 = (~mask_1) & (rm_counts < (rm_thres * X_extend.shape[1]))
    
    keep_mask = mask_1 | keep_mask_0
    
    X_final = X_extend[keep_mask]
    Y_final = Y_extend[keep_mask]
    
    X_label1 = X_extend[mask_1]
    len_label0 = np.sum(keep_mask_0)
    len_label1 = len(X_label1)

    X_augment = []
    Y_augment = []

    if len_label1 > 0:
        multiplier = int(len_label0 / len_label1)
        for age in range(1, multiplier):
            X_aug_batch = X_label1.copy()
            X_aug_batch[:, 1] += age
            X_augment.append(X_aug_batch)
            Y_augment.append(np.ones(len_label1, dtype=Y_extend.dtype))

        mask_9_1 = (X_label1[:, 9] == 1)
        mask_9_0 = (X_label1[:, 9] == 0)
        
        if np.any(mask_9_1):
            aug_batch = X_label1[mask_9_1].copy()
            aug_batch[:, 9] = 0
            X_augment.append(aug_batch)
            Y_augment.append(np.ones(len(aug_batch), dtype=Y_extend.dtype))
        if np.any(mask_9_0):
            aug_batch = X_label1[mask_9_0].copy()
            aug_batch[:, 9] = 1
            X_augment.append(aug_batch)
            Y_augment.append(np.ones(len(aug_batch), dtype=Y_extend.dtype))

        mask_0_0 = (X_label1[:, 0] == 0)
        mask_0_1 = (X_label1[:, 0] == 1)
        mask_0_m1 = (X_label1[:, 0] == -1)
        
        if np.any(mask_0_0):
            aug_batch = X_label1[mask_0_0].copy()
            aug_batch[:, 0] = 1
            X_augment.append(aug_batch)
            Y_augment.append(np.ones(len(aug_batch), dtype=Y_extend.dtype))
        if np.any(mask_0_1):
            aug_batch = X_label1[mask_0_1].copy()
            aug_batch[:, 0] = 0
            X_augment.append(aug_batch)
            Y_augment.append(np.ones(len(aug_batch), dtype=Y_extend.dtype))
        if np.any(mask_0_m1):
            aug_batch = X_label1[mask_0_m1].copy()
            aug_batch[:, 0] = 0
            X_augment.append(aug_batch)
            Y_augment.append(np.ones(len(aug_batch), dtype=Y_extend.dtype))

    if X_augment:
        X_augment = np.vstack(X_augment)
        Y_augment = np.concatenate(Y_augment)
        X_final = np.concatenate((X_final, X_augment))
        Y_final = np.concatenate((Y_final, Y_augment))

    print(np.sum(Y_final == 1), np.sum(Y_final == 0))

    return X_final, Y_final


def remove_duplicate(X_final, Y_final):
    data = pd.DataFrame(X_final)
    data.insert(0, 'label', Y_final)
    data = data.drop_duplicates(subset=data.columns.difference(['label']), keep='last')

    Y = data['label'].values
    X_matrix = data.drop(['label'], axis=1).values
    
    mask_1 = (Y == 1)
    mask_0 = (Y == 0)
    
    X_1 = X_matrix[mask_1]
    X_0 = X_matrix[mask_0]
    
    len_X_1 = len(X_1)
    if len_X_1 > 0 and len(X_0) > 0:
        X_0_resample = resample(X_0, n_samples=len_X_1)
        X = np.concatenate((X_0_resample, X_1))
        Y = np.concatenate((np.zeros(len_X_1, dtype=int), np.ones(len_X_1, dtype=int)))
    else:
        # Fallback if classes are heavily skewed or missing
        X = X_matrix
        Y = Y

    print(len(X))
    print(np.sum(Y == 0), np.sum(Y == 1))

    return X, Y


def data_pipeline(X, Y, random_state=42):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    return X_train, X_test, Y_train, Y_test


def data_pipeline_nn(X, Y, random_state=42):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_state)


    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    Y_val = np.array(Y_val)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val

def get_clean_data(train_link, aug_link_1, aug_link_2, use_scaling=True, use_downsampling=False, use_smote=False, use_smoteenn=False):
    # 1. Load data pools
    X_p, Y_p = get_data(train_link)
    X_a1, Y_a1 = get_data(aug_link_1)
    X_a2, Y_a2 = get_data(aug_link_2)
    X_pool = np.concatenate((X_p, X_a1, X_a2))
    Y_pool = np.concatenate((Y_p, Y_a1, Y_a2))

    # 2. Split FIRST to prevent data leakage from augmentation
    X_train_raw, X_test, Y_train_raw, Y_test = train_test_split(X_pool, Y_pool, test_size=0.2, random_state=42)

    # 3. Augment ONLY the training portion
    if use_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train, Y_train = smote.fit_resample(X_train_raw, Y_train_raw)
    elif use_smoteenn:
        from imblearn.combine import SMOTEENN
        smoteenn = SMOTEENN(random_state=42)
        X_train, Y_train = smoteenn.fit_resample(X_train_raw, Y_train_raw)
    else:
        # We pass empty arrays for the external augment files because they are already in the pool
        X_train, Y_train = imbalance_solve(X_train_raw, Y_train_raw, 
                                          np.empty((0, X_train_raw.shape[1])), np.empty(0), 
                                          np.empty((0, X_train_raw.shape[1])), np.empty(0), 
                                          -1, 0.5)

    # 4. Apply downsampling if requested (usually for Random Forest)
    if use_downsampling:
        X_train, Y_train = remove_duplicate(X_train, Y_train)

    # 5. Standard Scaling
    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test, scaler

def find_best_threshold(y_true, y_probs):
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0
    best_threshold = 0.5
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold, best_f1
