import torch
import torch.nn as nn
from utils import get_activation_function

class PINN(nn.Module):
    def __init__(self, hard_BC, hidden_layers, activation_function):
        super(PINN, self).__init__()
        layers = []
        input_dim = 1  # TIME variable
        activation = get_activation_function(activation_function)  # Use the selected activation function

        # Define the hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim

        # Define the output layer
        # The number of final output nodes is : 2 * N_tank - 1 (momentum eqn: N_tank - 1, continuity eqn: N_tank)
        layers.append(nn.Linear(input_dim, 2 * N_tank - 1))

        self.network = nn.Sequential(*layers)
        self.hard_BC = hard_BC
        if self.hard_BC is not None:
            print("Applying hard BC/IC.")

    def forward(self, t):
        u = self.network(t)
        if self.hard_BC is not None:
            u = self.hard_BC(t, u)
        return torch.abs(u)  # Ensuring non-negative values for physical variables
