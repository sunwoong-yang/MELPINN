import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
from network import PINN
import matplotlib.pyplot as plt
import os

def residual(net, t):
    L = 0.1
    K = 1
    g = 9.81
    rho = 1000  # Density of the water
    A_t = 50  # Area of the tank 50m^2
    A_p = 0.0314  # Area of the pipe with diameter of 0.2m

    def void_fraction(h):
        # Calculate the void fraction based on the height
        a = torch.ones_like(h, device=device)
        a[h >= 0.2] = 0
        mask = (h > 0) & (h < 0.2)
        a[mask] = 1 - h[mask] / 0.2
        return a

    u = net(t)
    vel = u[:, :N_tank - 1]
    height = u[:, N_tank - 1:]

    # Calculate gradients dynamically
    du_dt = [torch.autograd.grad(u[:, i], t, grad_outputs=torch.ones_like(u[:, i], device=device), create_graph=True)[0]
             for i in range(u.shape[1])]

    a_void = [void_fraction(height[:, [i]]) for i in range(N_tank - 1)]

    # Dynamically calculate momentum equations
    momentum = [L * du_dt[i] - g * height[:, [i]] + K * torch.abs(vel[:, [i]]) * vel[:, [i]] / 2 for i in range(N_tank - 1)]

    # Dynamically calculate continuity equations
    continuity = []
    continuity.append(du_dt[N_tank - 1] * rho * A_t + rho * vel[:, [0]] * A_p * (1 - a_void[0]))
    for i in range(1, N_tank - 1):
        continuity.append(du_dt[N_tank - 1 + i] * rho * A_t + rho * vel[:, [i]] * A_p * (1 - a_void[i]) - rho * vel[:, [
            i - 1]] * A_p * (1 - a_void[i - 1]))
    continuity.append(du_dt[2 * N_tank - 2] * rho * A_t - rho * vel[:, [-1]] * A_p * (1 - a_void[-1]))

    return momentum + continuity

# Training the PINN
def train_pinn(weights=None, use_hard_BC=True, epochs=30000, N_collo=2000, end_time=2758, plot_period=1000,
               hidden_layers=[20, 20, 20], activation_function='Tanh'):
    if weights is None:
        weights = [1] * (2 * N_tank - 1)  # Default weights are all 1

    assert len(weights) == (2 * N_tank - 1), "Weights list must have (2 * N_tank - 1) elements."

    hard_B
