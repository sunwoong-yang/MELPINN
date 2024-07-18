import subprocess
import sys


def install(package):
	subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# List of required packages
required_packages = [
	"torch",
	"numpy",
	"matplotlib",
	"tqdm"
]

# Install required packages
for package in required_packages:
	try:
		__import__(package)
	except ImportError:
		install(package)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
	gpu_name = torch.cuda.get_device_name(0)
	print('GPU name:', gpu_name)

N_tank = 6
epochs = 50000
N_collo = 5000
hidden_layers = [64] * 7
end_time = 2758
use_hard_BC = True


# Function to clear the /figures directory
def make_or_clear_figures_directory():
	os.makedirs('figures', exist_ok=True)

	for file in os.listdir('figures'):
		file_path = os.path.join('figures', file)
		if os.path.isfile(file_path):
			os.unlink(file_path)


# Clear the /figures directory
make_or_clear_figures_directory()


# Neural Network Definition
class PINN(nn.Module):
	def __init__(self, hard_BC, hidden_layers):
		super(PINN, self).__init__()
		layers = []
		input_dim = 1

		# Define the hidden layers
		for hidden_dim in hidden_layers:
			layers.append(nn.Linear(input_dim, hidden_dim))
			layers.append(nn.Tanh())
			input_dim = hidden_dim

		# Define the output layer
		layers.append(nn.Linear(input_dim, 2 * N_tank - 1))

		self.network = nn.Sequential(*layers)
		self.hard_BC = hard_BC
		if self.hard_BC is not None:
			print("Applying hard BC/IC.")

	def forward(self, t):
		u = self.network(t)
		if self.hard_BC is not None:
			u = hard_BC(t, u)
		return u


# Define the PDE residual
def residual(net, t):
	L = 0.1
	K = 1
	g = 9.81
	rho = 1000  # density of the water
	A_t = 50  # area of the tank 50m^2
	A_p = 0.0314  # area of the pipe with diameter of 0.2m

	def void_fraction(h):
		a = torch.ones_like(h)
		a[h >= 0.2] = 0
		mask = (h > 0) & (h < 0.2)
		a[mask] = 1 - h[mask] / 0.2
		return a

	u = net(t)
	vel = u[:, :N_tank - 1]
	height = u[:, N_tank - 1:]

	# Calculate gradients dynamically
	du_dt = [torch.autograd.grad(u[:, i], t, grad_outputs=torch.ones_like(u[:, i]), create_graph=True)[0] for i in
	         range(u.shape[1])]
	a_void = [void_fraction(height[:, [i]]) for i in range(N_tank - 1)]

	momentum1 = L * du_dt[0] - g * height[:, [0]] + K * torch.abs(vel[:, [0]]) * vel[:, [0]] / 2
	momentum2 = L * du_dt[1] - g * height[:, [1]] + K * torch.abs(vel[:, [1]]) * vel[:, [1]] / 2
	momentum3 = L * du_dt[2] - g * height[:, [2]] + K * torch.abs(vel[:, [2]]) * vel[:, [2]] / 2
	momentum4 = L * du_dt[3] - g * height[:, [3]] + K * torch.abs(vel[:, [3]]) * vel[:, [3]] / 2
	momentum5 = L * du_dt[4] - g * height[:, [4]] + K * torch.abs(vel[:, [4]]) * vel[:, [4]] / 2

	continuity1 = du_dt[5] * rho * A_t + rho * vel[:, [0]] * A_p * (1 - a_void[0])
	continuity2 = du_dt[6] * rho * A_t + rho * vel[:, [1]] * A_p * (1 - a_void[1]) - rho * vel[:, [0]] * A_p * (
			1 - a_void[0])
	continuity3 = du_dt[7] * rho * A_t + rho * vel[:, [2]] * A_p * (1 - a_void[2]) - rho * vel[:, [1]] * A_p * (
			1 - a_void[1])
	continuity4 = du_dt[8] * rho * A_t + rho * vel[:, [3]] * A_p * (1 - a_void[3]) - rho * vel[:, [2]] * A_p * (
			1 - a_void[2])
	continuity5 = du_dt[9] * rho * A_t + rho * vel[:, [4]] * A_p * (1 - a_void[4]) - rho * vel[:, [3]] * A_p * (
			1 - a_void[3])
	continuity6 = du_dt[10] * rho * A_t - rho * vel[:, [4]] * A_p * (1 - a_void[4])

	return momentum1, momentum2, momentum3, momentum4, momentum5, \
		continuity1, continuity2, continuity3, continuity4, continuity5, continuity6


# Impose hard BC/IC
def hard_BC(t, u):
	transformed_u = []
	for u_idx in range(u.shape[1]):
		if u_idx == N_tank - 1:  # for the first tank, height IC is imposed as 2.
			transformed_u.append(torch.abs(2. - t * u[:, [u_idx]]))  # 2 - t * height of 1st tank
		else:  # for other tanks, velocity/height IC is imposed as 0.
			transformed_u.append(torch.abs(t * u[:, [u_idx]]))  # t * (velocity or height)
	# Reason for torch.abs -> to prevent the velocity/height from being negative values
	return torch.hstack(transformed_u)


# Training the PINN
def train_pinn(weights=None, use_hard_BC=True, epochs=30000, N_collo=2000, end_time=2758, plot_period=1000,
               hidden_layers=[20, 20, 20]):
	if weights is None:
		weights = [1] * 11  # Default weights are all 1

	assert len(weights) == 11, "Weights list must have 11 elements."

	hard_BC_func = hard_BC if use_hard_BC else None
	net = PINN(hard_BC=hard_BC_func, hidden_layers=hidden_layers)
	optimizer = optim.Adam(net.parameters(), lr=1e-3)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

	# Clear the content of the file at the start
	with open('figures/tank_height.txt', 'w') as file:
		file.write("")

	# Collocation points
	t_lower_bound, t_upper_bound = 0, end_time
	t_collocation = (t_upper_bound - t_lower_bound) * torch.rand((N_collo, 1)) + t_lower_bound
	t_collocation.requires_grad = True

	start_time = time.time()  # Start time

	progress_bar = tqdm(range(epochs), desc="Training Epochs")
	for epoch in progress_bar:
		optimizer.zero_grad()

		# Physics loss
		momentum1, momentum2, momentum3, momentum4, momentum5, \
			continuity1, continuity2, continuity3, continuity4, continuity5, continuity6 = residual(net, t_collocation)

		# Calculate individual losses
		loss_momentum1 = nn.MSELoss()(momentum1, torch.zeros_like(momentum1))
		loss_momentum2 = nn.MSELoss()(momentum2, torch.zeros_like(momentum2))
		loss_momentum3 = nn.MSELoss()(momentum3, torch.zeros_like(momentum3))
		loss_momentum4 = nn.MSELoss()(momentum4, torch.zeros_like(momentum4))
		loss_momentum5 = nn.MSELoss()(momentum5, torch.zeros_like(momentum5))

		loss_continuity1 = nn.MSELoss()(continuity1, torch.zeros_like(continuity1))
		loss_continuity2 = nn.MSELoss()(continuity2, torch.zeros_like(continuity2))
		loss_continuity3 = nn.MSELoss()(continuity3, torch.zeros_like(continuity3))
		loss_continuity4 = nn.MSELoss()(continuity4, torch.zeros_like(continuity4))
		loss_continuity5 = nn.MSELoss()(continuity5, torch.zeros_like(continuity5))
		loss_continuity6 = nn.MSELoss()(continuity6, torch.zeros_like(continuity6))

		# Apply weights to individual losses
		loss = (weights[0] * loss_momentum1 + weights[1] * loss_momentum2 + weights[2] * loss_momentum3 +
		        weights[3] * loss_momentum4 + weights[4] * loss_momentum5 + weights[5] * loss_continuity1 +
		        weights[6] * loss_continuity2 + weights[7] * loss_continuity3 + weights[8] * loss_continuity4 +
		        weights[9] * loss_continuity5 + weights[10] * loss_continuity6)

		loss.backward()  # No need to retain the graph now
		optimizer.step()
		scheduler.step()

		losses = {
			'Total Loss': loss.item(),
			'Momentum Losses': f'{loss_momentum1.item():.4f}, {loss_momentum2.item():.4f}, {loss_momentum3.item():.4f}, {loss_momentum4.item():.4f}, {loss_momentum5.item():.4f}',
			'Continuity Losses': f'{loss_continuity1.item():.4f}, {loss_continuity2.item():.4f}, {loss_continuity3.item():.4f}, {loss_continuity4.item():.4f}, {loss_continuity5.item():.4f}, {loss_continuity6.item():.4f}'
		}

		progress_bar.set_postfix(losses)

		if epoch % plot_period == 0:
			save_results(net, input=end_time, epoch=epoch, losses=losses)

	end_time = time.time()  # End time
	print(f'Total Training Time: {end_time - start_time:.2f} seconds')

	return net


def save_results(net, input, epoch=None, losses=None):
    input = torch.tensor([[input]], dtype=torch.float32)
    final_results = net(input).squeeze().detach().cpu().numpy()

    with open('figures/tank_height.txt', 'a') as file:
        if epoch is not None:
            file.write(f"epoch={epoch}\n")
        for tank in range(N_tank):
            file.write(f"Tank {tank + 1} height: {final_results[N_tank - 1 + tank]:.6f}\n")
        file.write("\n")
        if losses is not None:
            file.write(f"Total Loss: {losses['Total Loss']:.4f}\n")
            file.write(f"Momentum Losses: {losses['Momentum Losses']}\n")
            file.write(f"Continuity Losses: {losses['Continuity Losses']}\n")
            file.write("\n")

    if epoch is not None:
        plot_heights(net, t_end=end_time, epoch=epoch)

    return final_results


def plot_heights(net, t_end=3000, epoch=None):
	t_start = 0
	t_values = np.linspace(t_start, t_end, t_end)
	heights = np.zeros((t_end, N_tank))

	for i, t in enumerate(t_values):
		input = torch.tensor([[t]], dtype=torch.float32)
		u_pred = net(input).squeeze().detach().numpy()
		heights[i, :] = u_pred[N_tank - 1:]

	plt.figure(figsize=(10, 6))
	for tank in range(N_tank):
		plt.plot(t_values, heights[:, tank], label=f'Tank {tank + 1} Height')

	plt.xlabel('Time (seconds)', fontsize=20)
	plt.ylabel('Height', fontsize=20)
	plt.title('Tank Heights Over Time', fontsize=24)
	plt.legend(fontsize=16)

	if epoch is not None:
		plt.title(f'Tank Heights Over Time (End of Epoch {epoch})', fontsize=24)
		plt.savefig(f'figures/heights_epoch_{epoch}.png', dpi=300)
	else:
		plt.savefig('figures/Final_heights.png', dpi=300)

	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.show()
	plt.close()


# Train and validate the model
net = train_pinn(weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], use_hard_BC=use_hard_BC, epochs=epochs, N_collo=N_collo,
                 end_time=end_time, plot_period=1000, hidden_layers=hidden_layers)
final_results = save_results(net, input=end_time)
plot_heights(net, t_end=end_time)
