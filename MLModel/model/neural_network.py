import torch
import torch.nn as nn

class DINN(nn.Module):
    """
    Dynamic Interaction Neural Network (DINN).
    Based on Dynamic Weight Logistic Regression (DWLR).
    """
    def __init__(self, input_dim, activation=torch.sigmoid):
        super(DINN, self).__init__()
        self.input_dim = input_dim
        self.activation = activation

        # Dynamic weights: Equivalent to D linear layers mapping (D -> 1), 
        # but efficiently vectorized as a single linear layer (D -> D)
        self.dynamic_weights = nn.Linear(input_dim, input_dim)

        # Pairwise interaction terms
        num_interactions = int(input_dim * (input_dim - 1) / 2)
        if num_interactions > 0:
            self.interaction_weights = nn.Parameter(torch.randn(num_interactions))
            # Pre-compute indices for upper triangle (i < j) to extract pairwise combinations
            self.register_buffer('triu_indices', torch.triu_indices(input_dim, input_dim, offset=1))
        else:
            self.interaction_weights = None

    def forward(self, x):
        # 1. Compute dynamic weights: [Batch, D]
        dyn_weights = self.activation(self.dynamic_weights(x))

        # 2. Compute weighted features and sum them: [Batch, 1]
        weighted_features_sum = torch.sum(dyn_weights * x, dim=1, keepdim=True)

        # 3. Compute interaction terms
        if self.interaction_weights is not None:
            # Extract pairwise products: x_i * x_j for all i < j
            interactions = x[:, self.triu_indices[0]] * x[:, self.triu_indices[1]]
            
            # Weighted sum of interactions
            interaction_sum = torch.matmul(interactions, self.interaction_weights).unsqueeze(1)
        else:
            interaction_sum = 0

        # 4. Combine into log-odds and apply final activation
        log_odds = weighted_features_sum + interaction_sum
        probs = torch.sigmoid(log_odds)
        
        return probs

def model_nn(input_shape, n_classes=None):
    """
    Compatibility wrapper to replace the old Keras model_nn function.
    input_shape should be a tuple like (55, )
    """
    input_dim = input_shape[0]
    model = DINN(input_dim=input_dim, activation=torch.sigmoid)
    return model
