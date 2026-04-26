from MLModel.data_pipeline import data_helper
from MLModel.AIModel.model import vae
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

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)

    # Train VAE Model
    print("Training VAE Model...")
    vae_model = vae.model_vae(train_loader, epochs=15, input_dim=input_dim)

    # Inference on Test data
    X_final_test, ID = data_helper.get_data_test(test_link)
    X_final_test_tensor = torch.tensor(X_final_test, dtype=torch.float32)
    
    print("Running Inference...")
    reconstructed_data = vae.vae_call(vae_model, X_final_test_tensor)
    
    # Normally VAE is unsupervised, but if we need to output predictions we could use reconstruction error as anomaly score
    # For now, we will just print the reconstructed shape to verify it ran
    print(f"Reconstructed Data Shape: {reconstructed_data.shape}")

    if save_result == 1:
        if os.path.exists(result_link):
            print('Result file existed :))')
        else:
            # Assuming you want to save the first dimension of reconstructed data as an example
            # Adjust according to your exact anomaly detection or feature extraction logic
            result_matrix = {'id': ID, 'feature_0': reconstructed_data[:, 0].numpy()}
            df = pd.DataFrame(result_matrix)
            os.makedirs(os.path.dirname(result_link), exist_ok=True)
            df.to_csv(result_link, index=False)
            print(f"Result saved to {result_link}")

if __name__ == "__main__":
    csv_train = os.path.join(DATA_DIR, "train_encode.csv")
    csv_test = os.path.join(DATA_DIR, "test_encode.csv")
    csv_augment_1 = os.path.join(DATA_DIR, "train_encode_age2_1.csv")
    csv_augment_2 = os.path.join(DATA_DIR, "train_encode_agemean_1.csv")

    # Make sure MLResult/vae directory exists or it will error if save_result=1
    result = os.path.join(DATA_DIR, "MLResult", "vae", "result_1_0.5_v1.csv")
    run(csv_train, csv_test, result, csv_augment_1, csv_augment_2, save_result=0)
