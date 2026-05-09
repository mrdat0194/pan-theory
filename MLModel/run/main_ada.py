from MLModel.data_pipeline import data_helper
from MLModel.model import ada_boost
import pandas as pd
import warnings
import os
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import UndefinedMetricWarning
from MLModel import DATA_DIR


warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def run(train_link, test_link, result_link, aug_link_1, aug_link_2, save_result=0):

    X_train, X_test, Y_train, Y_test, scaler = data_helper.get_clean_data(train_link, aug_link_1, aug_link_2, use_scaling=True)
    ada_model = ada_boost.model_ada(X_train, X_test, Y_train, Y_test)

    # Threshold Tuning
    Y_probs = ada_model.predict_proba(X_test)[:, 1]
    best_t, best_f1 = data_helper.find_best_threshold(Y_test, Y_probs)
    print(f'Optimal Threshold: {best_t:.4f}, Best F1-Score: {best_f1:.4f}')

    X_final_test, ID = data_helper.get_data_test(test_link)
    X_final_test = scaler.transform(X_final_test)
    Y_test_probs = ada_model.predict_proba(X_final_test)[:, 1]
    Y_predicted = (Y_test_probs >= best_t).astype(int)

    print(Y_predicted)

    if save_result == 1:

        if os.path.exists(result_link):
            print('Result file existed :))')
        else:
            result_matrix = {'id': ID, 'label': Y_predicted}

            df = pd.DataFrame(result_matrix)
            df.to_csv(result_link, index=False)

if __name__ == "__main__":
    csv_train = os.path.join(DATA_DIR, "train_encode.csv")
    csv_test = os.path.join(DATA_DIR, "test_encode.csv")
    csv_augment_1 = os.path.join(DATA_DIR, "train_encode_age2_1.csv")
    csv_augment_2 = os.path.join(DATA_DIR, "train_encode_agemean_1.csv")

    result = os.path.join(DATA_DIR, "MLResult","ada","result_adaboost_2.csv")
    run(csv_train, csv_test, result, csv_augment_1, csv_augment_2, save_result=0)
