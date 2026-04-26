from MLModel.data_pipeline import data_helper
from MLModel.AIModel.model import bnn
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os
from sklearn.exceptions import DataConversionWarning
from MLModel import DATA_DIR

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def run(train_link, test_link, result_link, aug_link_1, aug_link_2, save_result=0):
    X, Y = data_helper.get_data(train_link)
    X_aug_1, Y_aug_1 = data_helper.get_data(aug_link_1)
    X_aug_2, Y_aug_2 = data_helper.get_data(aug_link_2)

    X_final, Y_final = data_helper.imbalance_solve(X, Y, X_aug_1, Y_aug_1, X_aug_2, Y_aug_2, -1, 0.5)
    X_final, Y_final = data_helper.remove_duplicate(X_final, Y_final)

    X_train, X_test, Y_train, Y_test = data_helper.data_pipeline(X_final, Y_final)

    input_dim = X_train.shape[1]
    # In MLModel, Y_train is often class indices. If it's a binary classification problem, output_dim is 2
    output_dim = len(set(Y_train)) if len(set(Y_train)) > 0 else 2 

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=32, shuffle=True)

    # Train BNN Model
    print("Training BNN Model...")
    bnn_model = bnn.model_bnn(train_loader, epochs=15, input_dim=input_dim, output_dim=output_dim)

    # Inference on Test data
    X_final_test, ID = data_helper.get_data_test(test_link)
    X_final_test_tensor = torch.tensor(X_final_test, dtype=torch.float32)
    
    print("Running Inference...")
    mean_preds, variance_preds = bnn.bnn_call(bnn_model, X_final_test_tensor, num_samples=10)
    
    # Convert logits/probabilities to class predictions using argmax on the mean predictions
    Y_predicted = torch.argmax(mean_preds, dim=1).numpy()
    print("Predictions: ", Y_predicted)

    if save_result == 1:
        if os.path.exists(result_link):
            print('Result file existed :))')
        else:
            result_matrix = {'id': ID, 'label': Y_predicted}
            df = pd.DataFrame(result_matrix)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(result_link), exist_ok=True)
            df.to_csv(result_link, index=False)
            print(f"Result saved to {result_link}")

if __name__ == "__main__":
    csv_train = os.path.join(DATA_DIR, "train_encode.csv")
    csv_test = os.path.join(DATA_DIR, "test_encode.csv")
    csv_augment_1 = os.path.join(DATA_DIR, "train_encode_age2_1.csv")
    csv_augment_2 = os.path.join(DATA_DIR, "train_encode_agemean_1.csv")

    result = os.path.join(DATA_DIR, "MLResult", "bnn", "result_1_0.5_v1.csv")
    run(csv_train, csv_test, result, csv_augment_1, csv_augment_2, save_result=0)
