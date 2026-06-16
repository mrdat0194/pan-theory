from MLModel.model import bayesian
import pandas as pd
import warnings
import os
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import UndefinedMetricWarning
from MLModel import DATA_DIR
import numpy as np
from sklearn.model_selection import train_test_split

import random
import torch
from torch import nn, optim
import math
from IPython import display

from MLModel.model.res.plot_lib import plot_data, plot_model, set_default
from matplotlib import pyplot as plt

set_default()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


seed = 1
random.seed(seed)
torch.manual_seed(seed)
N = 1000  # num_samples_per_class
D = 1  # dimensions
C = 1  # num_classes
H = 100  #

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

data = pd.read_csv('/Users/petern/ML-From-Scratch/mlfromscratch/data/TempLinkoping2016.txt', sep="\t")

time = np.atleast_2d(data["time"].values).T
temp = np.atleast_2d(data["temp"].values).T

X = time # fraction of the year [0, 1]
y = temp

X = torch.from_numpy(X)
y = torch.from_numpy(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4)

# # First trial

learning_rate = 1e-3
lambda_l2 = 1e-5

# Number of networks
n_networks = 10
models = list()
y_pretrain = list()

# nn package also has different loss functions.
# we use MSE for a regression task
criterion = torch.nn.MSELoss()

for mod in range(n_networks):
    # nn package to create our linear model
    # each Linear module has a weight and bias
    model = nn.Sequential(
        nn.Linear(D, H),
        nn.ReLU() if mod < n_networks // 2 else nn.Tanh(),
        nn.Linear(H, H),
        # torch.nn.Flatten(0, 1)
        nn.ReLU() if mod < n_networks // 2 else nn.Tanh(),
        nn.Linear(H, H),
        nn.ReLU() if mod < n_networks // 2 else nn.Tanh(),
        nn.Linear(H, H),
        nn.ReLU() if mod < n_networks // 2 else nn.Tanh(),
        nn.Linear(H, H),
        nn.ReLU() if mod < n_networks // 2 else nn.Tanh(),
        nn.Linear(H, H),
        nn.ReLU() if mod < n_networks // 2 else nn.Tanh(),
        nn.Linear(H, C)
    )
    model.to(device)

    # Append models
    models.append(model)

    # we use the optim package to apply
    # ADAM for our parameter updates
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2) # built-in L2
    # optimizer = torch.optim.LBFGS(model.parameters(), history_size=10, max_iter=4)
    # e = 1.  # plotting purpose

    # Training
    for t in range(1000):

        # Feed forward to get the logits
        y_pred = model(X_train.float())

        # Append pre-train output
        if t == 0:
            y_pretrain.append(y_pred.detach())

        # Compute the loss and accuracy
        loss = criterion(y_pred, Y_train.float())
        print(f"[MODEL]: {mod + 1}, [EPOCH]: {t}, [LOSS]: {loss.item():.6f}")
        display.clear_output(wait=True)

        # zero the gradients before running
        # the backward pass.
        optimizer.zero_grad()

        # Backward pass to compute the gradient
        # of loss w.r.t our learnable params.
        loss.backward()

        # Update params
        optimizer.step()

print(models[0], models[-1])


y_pred = list()
relu_models = models[:n_networks // 2]
tanh_models = models[n_networks // 2:]
# plt.figure(figsize=(20, 10))

def dense_prediction(models, non_linearity, zoom):
    plt.subplot(1, 2, 1 if non_linearity == 'ReLU' else 2)
    for model in models:

        # Getting predictions from input
        with torch.no_grad():
            y_pred.append(model(X_test.float()))

    plt.plot(336* X_test.cpu().numpy(), y_pred[-1].cpu().numpy(), 'r-', lw=1)


    plt.scatter(366 *X.cpu().numpy(), y.cpu().numpy(), label='data')
    # plt.axis('square')
    # plt.axis(torch.tensor((-1.1, 1.1, -1.1, 1.1)) * zoom);
    y_combo = torch.stack(y_pred)
    plt.plot(366 *X_test.cpu().numpy(),  y_combo.var(dim=0).cpu().sqrt().numpy(), 'y', label='10 × std')
    plt.plot(366 *X_test.cpu().numpy(),  y_combo.var(dim=0).cpu().numpy(), 'g', label='30 × variance')
    # plt.legend()
    plt.title(non_linearity + ' models')

z = 1  # try 1 or 4
dense_prediction(relu_models, 'ReLU', zoom=z)
dense_prediction(tanh_models, 'Tanh', zoom=z)

# Color map
cmap = plt.get_cmap('viridis')

# m1 = plt.scatter(366 * X_train, Y_train, color=cmap(0.9), s=10)
# m2 = plt.scatter(366 * X_test, Y_test, color=cmap(0.5), s=10)
plt.show()



# Second trial
# import math
#
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.optim import Adam, LBFGS, SGD
# from torch.utils.data import Dataset, DataLoader
#
# class PolynomialModel(nn.Module):
#     def __init__(self, degree):
#         super().__init__()
#         self._degree = degree
#         self.linear = nn.Linear(self._degree, 1)
#
#     def forward(self, x):
#         return self.linear(self._polynomial_features(x))
#
#     def _polynomial_features(self, x):
#         x = x.unsqueeze(1)
#         return torch.cat([x ** i for i in range(1, self._degree + 1)], 1)
#
#
# def train_step(model, data, optimizer, criterion):
#     running_loss = 0.0
#
#     for i, (x, y) in enumerate(data):
#
#         x_ = Variable(x, requires_grad=True)
#         y_ = Variable(y.unsqueeze(1)) # unsqueeze to match dimensions with y_pred later
#
#         def closure():
#             # Zero gradients
#             optimizer.zero_grad()
#
#             # Forward pass
#             y_pred = model(x_)
#
#             # Compute loss
#             loss = criterion(y_pred, y_)
#
#             # Backward pass
#             loss.backward()
#
#             return loss
#
#         # Update weights
#         optimizer.step(closure)
#
#         # Update the running loss
#         loss = closure()
#         running_loss += loss.item()
#     return running_loss
#
# criterion = nn.MSELoss()
#
# pm_cubic_adam_20 = PolynomialModel(degree=4)
# optimizer = Adam(pm_cubic_adam_20.parameters(), weight_decay=0.0001)
#
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dtype = torch.float
# # x = torch.linspace(-math.pi, math.pi, steps=20, dtype=dtype, device=device)
# # cubic_y = x ** 3 + 2 * x ** 2 - 3 * x + 5
# # sine_y = torch.sin(x)
#
# class DummyData(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#     def __len__(self):
#         return self.x.shape[0]
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]
#
#
# cubic_data = DummyData(X.float().reshape(-1, 1), y.float().reshape(-1, 1))
#
# for epoch in range(20):
#     running_loss = train_step(model=pm_cubic_adam_20,
#                               data=cubic_data,
#                               optimizer=optimizer,
#                               criterion=criterion)
#     print(f"Epoch: {epoch + 1:02}/20 Loss: {running_loss:.5e}")
#
#
#
# X_test = X_test[0:140].reshape(35, 4)
#
# Y_test = Y_test[0:140]
#
# cubic_y_pred_adam_20 = pm_cubic_adam_20(Variable(X_test.float())).detach().cpu().numpy()
#
# num_pred_points = X_test.shape[0]
#
# cubic_y_pred_adam_20 = cubic_y_pred_adam_20.reshape(num_pred_points,4)
#
#
# fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12, 10), squeeze=True)
#
# ax1.plot(366 *X_test, cubic_y_pred_adam_20, "r", label="Predictions with Adam (20 training epochs)", alpha=0.4, lw=2)
#
# # Color map
# cmap = plt.get_cmap('viridis')
#
# m1 = ax1.scatter(366 * X_train, Y_train, color=cmap(0.9), s=10)
# m2 = ax1.scatter(366 * X_test, Y_test, color=cmap(0.5), s=10)
#
# plt.show()



