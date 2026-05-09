import os
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

from MLModel.data_pipeline import data_helper
from MLModel.model import neural_network
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import UndefinedMetricWarning
from MLModel import DATA_DIR

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def run(train_link, aug_link_1, aug_link_2):
    print('')

    import numpy as np
    X_p, Y_p = data_helper.get_data(train_link)
    X_a1, Y_a1 = data_helper.get_data(aug_link_1)
    X_a2, Y_a2 = data_helper.get_data(aug_link_2)
    X_pool = np.concatenate((X_p, X_a1, X_a2))
    Y_pool = np.concatenate((Y_p, Y_a1, Y_a2))

    X_train_raw, X_test_raw, X_val_raw, Y_train_raw, Y_test, Y_val = data_helper.data_pipeline_nn(X_pool, Y_pool, random_state=42)

    # Augment and downsample only the training set
    X_train, Y_train = data_helper.imbalance_solve(X_train_raw, Y_train_raw, 
                                                  np.empty((0, X_train_raw.shape[1])), np.empty(0), 
                                                  np.empty((0, X_train_raw.shape[1])), np.empty(0), 
                                                  -1, 0.5)
    X_train, Y_train = data_helper.remove_duplicate(X_train, Y_train)
    X_val = X_val_raw
    X_test = X_test_raw

    # Implement StandardScaler to fix the neural network gradient explosion
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save the scaler so test_nn.py can use exactly the same scaling rules
    save_model_dir = os.path.join(os.path.dirname(__file__), '..', 'model_nn_save')
    os.makedirs(save_model_dir, exist_ok=True)
    scaler_path = os.path.join(save_model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    # Setup PyTorch Tensors and DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32).view(-1, 1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32).view(-1, 1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32).view(-1, 1))

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    input_dim = X_train.shape[1]
    model = neural_network.model_nn((input_dim, ))
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 64
    print(f'Starting training for {epochs} epochs...')

    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predictions = (outputs > 0.5).float()
            correct_train += (predictions == targets).sum().item()
            total_train += targets.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                predictions = (outputs > 0.5).float()
                correct_val += (predictions == targets).sum().item()
                total_val += targets.size(0)
        
        val_loss /= len(val_dataset)
        val_acc = correct_val / total_val
        
        history['loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(epoch_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Acc: {epoch_acc:.4f} - Val Acc: {val_acc:.4f}")

    # Evaluate on test set
    model.eval()
    y_true_test = []
    y_pred_probs_test = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            y_pred_probs_test.extend(outputs.cpu().numpy())
            y_true_test.extend(targets.cpu().numpy())

    import numpy as np
    y_true_test = np.array(y_true_test)
    y_pred_probs_test = np.array(y_pred_probs_test)

    # Threshold Tuning
    best_t, best_f1 = data_helper.find_best_threshold(y_true_test, y_pred_probs_test)
    print(f'Optimal Threshold: {best_t:.4f}, Best F1-Score: {best_f1:.4f}')

    y_pred_test = (y_pred_probs_test >= best_t).astype(int)

    test_loss = log_loss(y_true_test, y_pred_probs_test)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_auc = roc_auc_score(y_true_test, y_pred_probs_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print('')
    print('Classification Report (at Optimal Threshold):')
    print(classification_report(y_true_test, y_pred_test))
    print('')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true_test, y_pred_test))

    print('')
    print(f'Result [loss, accuracy, auc]: [{test_loss:.4f}, {test_acc:.4f}, {test_auc:.4f}]')

    # Save PyTorch model state dict
    weight_model_path = os.path.join(save_model_dir, 'model_nn.pth')
    torch.save(model.state_dict(), weight_model_path)

    print('')
    print(f'Saved PyTorch model to {weight_model_path}')

    # Plot training & validation accuracy values
    plt.figure(figsize=(19, 9))
    plt.subplot(1, 2, 1)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(save_model_dir, 'training_result_lr0.0001.png'))
    plt.close()


if __name__ == "__main__":
    csv_train = os.path.join(DATA_DIR, "train_encode.csv")
    csv_test = os.path.join(DATA_DIR, "test_encode.csv")
    csv_augment_1 = os.path.join(DATA_DIR, "train_encode_age2_1.csv")
    csv_augment_2 = os.path.join(DATA_DIR, "train_encode_agemean_1.csv")

    run(csv_train, csv_augment_1, csv_augment_2)
