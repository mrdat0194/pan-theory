import torch
from torch import nn
from torch.optim import SGD
import numpy as np


def f(x, u, t=None):
    """
    Kinematic model for tricycle
    x: states (x, y, θ, s)
    u: control (ϕ, a)
    t: time
    """
    L = 1  # m
    x_pos, y, θ, s = x
    
    ϕ, a = u
    dx = torch.zeros(4)
    dx[0] = s * torch.cos(θ)
    dx[1] = s * torch.sin(θ)
    dx[2] = s / L * torch.tan(ϕ)
    dx[3] = a
    return dx


# Costs definition
def vanilla_cost(state, target):
    x_x, x_y = target
    return (state[-1][0] - x_x).pow(2) + (state[-1][1] - x_y).pow(2)


def cost_with_target_s(state, target):
    x_x, x_y = target
    return (state[-1][0] - x_x).pow(2) + (state[-1][1] - x_y).pow(2) + (state[-1][-1]).pow(2)


def cost_sum_distances(state, target):
    x_x, x_y = target
    dists = ((state[:, 0] - x_x).pow(2) + (state[:, 1] - x_y).pow(2)).pow(0.5)
    return dists.mean()


def cost_sum_square_distances(state, target):
    x_x, x_y = target
    dists = ((state[:, 0] - x_x).pow(2) + (state[:, 1] - x_y).pow(2))
    return dists.mean()


def cost_logsumexp(state, target):
    x_x, x_y = target
    dists = ((state[:, 0] - x_x).pow(2) + (state[:, 1] - x_y).pow(2))
    return -1 * torch.logsumexp(-1 * dists, dim=0)


def model_optimal_control(x_x, x_y, s, T, epochs, stepsize, cost_f_name='vanilla_cost'):
    """
    Path planning for tricycle (Learning via backpropagation on control sequence)
    x_x: target x component of position vector
    x_y: target y component of position vector
    s: initial speed
    T: time steps
    epochs: number of epochs for back propagation
    stepsize: stepsize for back propagation
    cost_f_name: name of the cost function to use
    """
    u = nn.Parameter(torch.zeros(T, 2))
    optimizer = SGD((u,), lr=stepsize)
    dt = 1  # s
    
    cost_functions = {
        'vanilla_cost': vanilla_cost,
        'cost_with_target_s': cost_with_target_s,
        'cost_sum_distances': cost_sum_distances,
        'cost_sum_square_distances': cost_sum_square_distances,
        'cost_logsumexp': cost_logsumexp
    }
    
    cost_f = cost_functions.get(cost_f_name, vanilla_cost)
    
    for epoch in range(epochs):
        x = [torch.tensor((0., 0., 0., float(s)), dtype=torch.float32)]
        for t in range(1, T+1):
            x.append(x[-1] + f(x[-1], u[t-1]) * dt)
        x_t = torch.stack(x)
        cost = cost_f(x_t, (x_x, x_y))
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
    return u.detach()


def optimal_control_call(initial_state, u, dt=1):
    """
    Generates the trajectory given initial state and optimal control sequence u
    initial_state: tuple/list (x, y, θ, s)
    u: optimal control tensor of shape (T, 2)
    """
    x = torch.tensor(initial_state, dtype=torch.float32)
    trajectory = [x.clone()]
    
    T = u.shape[0]
    for t in range(T):
        x += f(x, u[t]) * dt
        trajectory.append(x.clone())
        
    return torch.stack(trajectory)
