import os
import warnings
import pandas as pd
import torch
import joblib

from MLModel.data_pipeline import data_helper
from MLModel.model import neural_network
from MLModel import DATA_DIR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def test(test_link, result_link):
    # Get data for predict
    X_final_test, ID = data_helper.get_data_test(test_link)

    # Model and Scaler path
    save_model_dir = os.path.join(os.path.dirname(__file__), '..', 'model_nn_save')
    scaler_path = os.path.join(save_model_dir, 'scaler.pkl')
    model_path = os.path.join(save_model_dir, 'model_nn.pth')

    # Load and apply scaler
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler not found at {scaler_path}. Please train the model first.")
        return
    
    scaler = joblib.load(scaler_path)
    X_final_test = scaler.transform(X_final_test)

    # Convert to PyTorch Tensor
    X_tensor = torch.tensor(X_final_test, dtype=torch.float32)

    # Initialize model
    input_dim = X_tensor.shape[1]
    model = neural_network.model_nn((input_dim, ))

    # Load PyTorch weights
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(X_tensor)
        # Apply threshold of 0.5 to convert probabilities to classes (0 or 1)
        results = (outputs.cpu().numpy() > 0.5).astype(int).flatten()

    if os.path.exists(result_link):
        print('Result file existed :))')
    else:
        result_matrix = {'id': ID, 'label': results}

        df = pd.DataFrame(result_matrix)
        # Make sure directory exists
        os.makedirs(os.path.dirname(result_link), exist_ok=True)
        df.to_csv(result_link, index=False)
        print(f'Predictions saved to {result_link}')


if __name__ == "__main__":
    csv_test = os.path.join(DATA_DIR, "test_encode.csv")
    result = os.path.join(DATA_DIR, "MLResult", "nn", "result_23_0.5_v1.csv")

    test(csv_test, result)
else:
    print("Classification is being imported into another module.")
