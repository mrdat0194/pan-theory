import torch
import torch.nn as nn
import torch.optim as optim

# Note: For a true Bayesian Neural Network, you typically use libraries like Pyro, 
# torchbnn, or implement variational layers manually. Here is a structural representation
# mimicking a Bayesian approach (e.g. Monte Carlo Dropout or specific VI layers).

class BNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5) # Example of MC Dropout for Bayesian approximation
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

def model_bnn(train_loader, epochs=10, lr=1e-3, input_dim=10, output_dim=2):
    model = BNN(input_dim=input_dim, hidden_dim=50, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            # In a true VI BNN, you would add the KL divergence to the loss here
            loss.backward()
            optimizer.step()
            
    return model

def bnn_call(model, test_data, num_samples=10):
    # To get uncertainty estimates, we keep dropout on (MC Dropout) during inference
    # Or in a proper variational network, we sample weights multiple times
    model.train() 
    
    all_predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            sample_preds = []
            for data in test_data:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                output = model(data)
                sample_preds.append(output)
            all_predictions.append(torch.cat(sample_preds))
            
    # Stack predictions from all passes to compute mean and variance (uncertainty)
    stacked_preds = torch.stack(all_predictions)
    mean_preds = stacked_preds.mean(dim=0)
    variance_preds = stacked_preds.var(dim=0)
    
    return mean_preds, variance_preds
