import subprocess
import sys
import argparse

# Function to install required packages if not already installed
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

torch.set_num_threads(12)  # Set the number of threads for PyTorch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
	gpu_name = torch.cuda.get_device_name(0)
	print('GPU name:', gpu_name)

# Argument parsing
parser = argparse.ArgumentParser(description='Train a Physics-Informed Neural Network (PINN).')
parser.add_argument('-tag', type=str, default='', help='Directory tag.')
parser.add_argument('-e', '--epochs', type=int, default=30000, help='Number of epochs for training.')
parser.add_argument('-T', '--N_tank', type=int, default=6, help='Number of tanks.')
parser.add_argument('-N', '--N_collo', type=int, default=5000, help='Number of collocation points.')
parser.add_argument('-hl', '--hidden_layers', type=int, nargs='+', default=[64] * 7,
                    help='List of hidden layers sizes.')
parser.add_argument('-t', '--end_time', type=int, default=2758, help='End time for the simulation.')
parser.add_argument('-BC', '--use_hard_BC', type=str, default='True', help='Use hard boundary conditions (True/False).')
parser.add_argument('-sc', '--scheduler', nargs=2, type=int, default=[1000, 0.98],
                    help='List with two elements: [step, gamma].')
parser.add_argument('-w', '--loss_weight', nargs=2, type=int, default=[10, 1],
                    help='List with two elements: [mom_weight, conti_weight].')
parser.add_argument('-af', '--activation_function', type=str, default='Tanh',
                    choices=['Tanh', 'ReLU', 'Sigmoid', 'LeakyReLU'],
                    help='Activation function to use in the neural network.')

args = parser.parse_args()

# Parsing command-line arguments
directory_tag = args.tag
N_tank = args.N_tank
epochs = args.epochs
N_collo = args.N_collo
hidden_layers = args.hidden_layers
end_time = args.end_time
use_hard_BC = args.use_hard_BC.lower() in ['t', 'true']
scheduler_step, scheduler_gamma = args.scheduler[0], args.scheduler[1]
momentum_weight, continuity_weight = args.loss_weight[0], args.loss_weight[1]
activation_function = args.activation_function

# Construct the folder name based on the argument values
hidden_layers_str = 'x'.join(map(str, hidden_layers))
folder_name = (
	f"figures/[{directory_tag}]Ntanks{N_tank}-e{epochs}-N{N_collo}"
	f"-hl{hidden_layers_str}-t{end_time}-BC{use_hard_BC}"
	f"-lr[{scheduler_step}, {scheduler_gamma}]"
	f"-lossW[{momentum_weight}, {continuity_weight}]"
	f"-af{activation_function}"
)

# Function to clear the specified directory or create it if it does not exist
def make_or_clear_figures_directory(folder_name):
	os.makedirs(folder_name, exist_ok=True)
	for file in os.listdir(folder_name):
		file_path = os.path.join(folder_name, file)
		if os.path.isfile(file_path):
			os.unlink(file_path)

# Clear or create the specified figures directory
make_or_clear_figures_directory(folder_name)

# Function to get the activation function based on user input
def get_activation_function(name):
	if name == 'Tanh':
		return nn.Tanh()
	elif name == 'ReLU':
		return nn.ReLU()
	elif name == 'Sigmoid':
		return nn.Sigmoid()
	elif name == 'LeakyReLU':
		return nn.LeakyReLU(0.01)
	else:
		raise ValueError(f"Unsupported activation function: {name}")

# Neural Network Definition
class PINN(nn.Module):
	def __init__(self, hard_BC, hidden_layers, activation_function):
		super(PINN, self).__init__()
		layers = []
		input_dim = 1
		activation = get_activation_function(activation_function)  # Use the selected activation function

		# Define the hidden layers
		for hidden_dim in hidden_layers:
			layers.append(nn.Linear(input_dim, hidden_dim))
			layers.append(activation)
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
		return torch.abs(u)  # Ensuring non-negative values for physical variables

# Define the PDE residual
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

# Impose hard BC/IC
def hard_BC(t, u):
	transformed_u = []
	for u_idx in range(u.shape[1]):
		if u_idx == N_tank - 1:  # For the first tank, height IC is imposed as 2.
			transformed_u.append(2. - t * u[:, [u_idx]])  # 2 - t * height of 1st tank
		else:  # For other tanks, velocity/height IC is imposed as 0.
			transformed_u.append(t * u[:, [u_idx]])  # t * (velocity or height)
	return torch.hstack(transformed_u)

# Training the PINN
def train_pinn(weights=None, use_hard_BC=True, epochs=30000, N_collo=2000, end_time=2758, plot_period=1000,
               hidden_layers=[20, 20, 20], activation_function='Tanh'):
	if weights is None:
		weights = [1] * (2 * N_tank - 1)  # Default weights are all 1

	assert len(weights) == (2 * N_tank - 1), "Weights list must have (2 * N_tank - 1) elements."

	hard_BC_func = hard_BC if use_hard_BC else None
	net = PINN(hard_BC=hard_BC_func, hidden_layers=hidden_layers, activation_function=activation_function).to(device)
	optimizer = optim.Adam(net.parameters(), lr=1e-3)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

	# Clear the content of the file at the start
	with open('figures/tank_height.txt', 'w') as file:
		file.write("")

	# Collocation points
	t_lower_bound, t_upper_bound = 0, end_time
	t_collocation = (t_upper_bound - t_lower_bound) * torch.rand((N_collo, 1), device=device) + t_lower_bound
	t_collocation.requires_grad = True

	# Initial condition points
	t_ic = torch.zeros((N_collo, 1), requires_grad=True, device=device)

	start_time = time.time()  # Start time

	progress_bar = tqdm(range(epochs), desc="Training Epochs")
	for epoch in progress_bar:
		optimizer.zero_grad()

		# Physics loss
		residuals = residual(net, t_collocation)

		# Calculate individual losses for momentum
		momentum_losses = [nn.MSELoss()(residuals[i], torch.zeros_like(residuals[i])) for i in range(N_tank - 1)]

		# Calculate individual losses for continuity
		continuity_losses = [nn.MSELoss()(residuals[N_tank - 1 + i], torch.zeros_like(residuals[N_tank - 1 + i])) for i
		                     in range(N_tank)]

		# Combine all losses
		loss_momentum = sum(weights[i] * momentum_losses[i] for i in range(N_tank - 1))
		loss_continuity = sum(weights[N_tank - 1 + i] * continuity_losses[i] for i in range(N_tank))

		# Initial condition loss
		if not use_hard_BC:
			u_ic = net(t_ic)
			loss_ic = nn.MSELoss()(u_ic[:, :N_tank - 1], torch.zeros_like(u_ic[:, :N_tank - 1])) + \
			          nn.MSELoss()(u_ic[:, N_tank - 1], torch.ones_like(u_ic[:, N_tank - 1]) * 2) + \
			          nn.MSELoss()(u_ic[:, N_tank:], torch.zeros_like(u_ic[:, N_tank:]))
		else:
			loss_ic = 0

		loss = loss_momentum + loss_continuity + loss_ic

		loss.backward()  # No need to retain the graph now
		optimizer.step()
		scheduler.step()

		losses = {
			'Total Loss': loss.item(),
			'Momentum Losses': ', '.join([f'{momentum_losses[i].item():.4f}' for i in range(N_tank - 1)]),
			'Continuity Losses': ', '.join([f'{continuity_losses[i].item():.4f}' for i in range(N_tank)]),
			'IC Losses': f'{loss_ic.item():.4f}' if not use_hard_BC else 'N/A'
		}

		progress_bar.set_postfix(losses)

		if epoch % plot_period == 0:
			save_results(net, input=end_time, epoch=epoch, losses=losses)

	end_time = time.time()  # End time
	print(f'Total Training Time: {end_time - start_time:.2f} seconds')

	return net

# Save results to file and plot
def save_results(net, input, epoch=None, losses=None):
	input = torch.tensor([[input]], dtype=torch.float32, device=device)
	final_results = net(input).squeeze().detach().cpu().numpy()

	with open(f'{folder_name}/tank_height.txt', 'a') as file:
		if epoch is not None:
			file.write(f"epoch={epoch}\n")
		for tank in range(N_tank):
			file.write(f"Tank {tank + 1} height: {final_results[N_tank - 1 + tank]:.6f}\n")
		file.write("\n")
		if losses is not None:
			file.write(f"Total Loss: {losses['Total Loss']:.4f}\n")
			file.write(f"Momentum Losses: {losses['Momentum Losses']}\n")
			file.write(f"Continuity Losses: {losses['Continuity Losses']}\n")
			file.write(f"IC Losses: {losses['IC Losses']}\n")
			file.write("\n")

	if epoch is not None:
		plot_heights(net, t_end=end_time, epoch=epoch)
		plot_velocities(net, t_end=end_time, epoch=epoch)

	return final_results

# Plot tank heights over time
def plot_heights(net, t_end=3000, epoch=None):
	t_start = 0
	t_values = np.linspace(t_start, t_end, t_end)
	heights = np.zeros((t_end, N_tank))

	for i, t in enumerate(t_values):
		input = torch.tensor([[t]], dtype=torch.float32, device=device)
		u_pred = net(input).squeeze().detach().cpu().numpy()
		heights[i, :] = u_pred[N_tank - 1:]

	plt.figure(figsize=(10, 6))
	for tank in range(N_tank):
		plt.plot(t_values, heights[:, tank], label=f'Tank {tank + 1} Height', alpha=0.6, linewidth=2.5)

	plt.xlabel('Time (seconds)', fontsize=20)
	plt.ylabel('Height', fontsize=20)
	plt.title('Tank Heights Over Time', fontsize=24, fontweight='bold')
	plt.legend(fontsize=16, shadow=True)

	if epoch is not None:
		plt.title(f'Tank Heights Over Time (End of Epoch {epoch})', fontsize=24, fontweight='bold')
		plt.savefig(f'{folder_name}/heights_epoch_{epoch}.png', dpi=300)
	else:
		plt.savefig(f'{folder_name}/Final_heights.png', dpi=300)

	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.show()
	plt.close()

# Plot tank velocities over time
def plot_velocities(net, t_end=3000, epoch=None):
	t_start = 0
	t_values = np.linspace(t_start, t_end, t_end)
	velocities = np.zeros((t_end, N_tank - 1))

	for i, t in enumerate(t_values):
		input = torch.tensor([[t]], dtype=torch.float32, device=device)
		u_pred = net(input).squeeze().detach().cpu().numpy()
		velocities[i, :] = u_pred[:N_tank - 1]

	plt.figure(figsize=(10, 6))
	for tank in range(N_tank - 1):
		plt.plot(t_values, velocities[:, tank], label=f'Tank {tank + 1} Velocity', alpha=0.6, linewidth=2.5)

	plt.xlabel('Time (seconds)', fontsize=20)
	plt.ylabel('Velocity', fontsize=20)
	plt.title('Tank Velocities Over Time', fontsize=24, fontweight='bold')
	plt.legend(fontsize=16, shadow=True)

	if epoch is not None:
		plt.title(f'Tank Velocities Over Time (End of Epoch {epoch})', fontsize=24, fontweight='bold')
		plt.savefig(f'{folder_name}/velocities_epoch_{epoch}.png', dpi=300)
	else:
		plt.savefig(f'{folder_name}/Final_velocities.png', dpi=300)

	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.show()
	plt.close()

# Train and validate the model
net = train_pinn(weights=[momentum_weight] * (N_tank - 1) + [continuity_weight] * N_tank, use_hard_BC=use_hard_BC,
                 epochs=epochs, N_collo=N_collo,
                 end_time=end_time, plot_period=1000, hidden_layers=hidden_layers,
                 activation_function=activation_function)
final_results = save_results(net, input=end_time)
plot_heights(net, t_end=end_time)
plot_velocities(net, t_end=end_time)
