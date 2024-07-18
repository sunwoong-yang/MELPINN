import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

nu = 0.1 / np.pi

# Neural Network Definition
class PINN(nn.Module):
	def __init__(self):
		super(PINN, self).__init__()
		self.fc1 = nn.Linear(2, 20)
		self.fc2 = nn.Linear(20, 20)
		self.fc3 = nn.Linear(20, 20)
		self.fc4 = nn.Linear(20, 1)

	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.fc2(x))
		x = torch.tanh(self.fc3(x))
		x = self.fc4(x)
		return x


# Define the PDE residual
def residual(net, x, t, nu):
	u = net(torch.cat([x, t], dim=1))
	u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
	u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
	u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
	res = u_t + u * u_x - nu * u_xx
	return res


# Training the PINN
def train_pinn():
	net = PINN()
	optimizer = optim.Adam(net.parameters(), lr=1e-3)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
	epochs = 30000

	# Collocation points
	x_collocation = 2 * torch.rand((1000, 1)) - 1
	x_collocation.requires_grad = True
	t_collocation = torch.rand((1000, 1), requires_grad=True)

	for epoch in range(epochs):
		optimizer.zero_grad()

		# Boundary and initial conditions
		x_bc = torch.cat([torch.ones(100, 1) * -1, torch.ones(100, 1)], dim=0)
		t_bc = torch.cat([torch.linspace(0, 1, 100).view(-1, 1), torch.linspace(0, 1, 100).view(-1, 1)], dim=0)
		u_bc = torch.zeros(200, 1)

		x_ic = torch.linspace(-1, 1, 100).view(-1, 1)
		t_ic = torch.zeros(100, 1)
		u_ic = -torch.sin(np.pi * x_ic)

		u_bc_pred = net(torch.cat([x_bc, t_bc], dim=1))
		bc_loss = nn.MSELoss()(u_bc_pred, u_bc)

		u_ic_pred = net(torch.cat([x_ic, t_ic], dim=1))
		ic_loss = nn.MSELoss()(u_ic_pred, u_ic)

		# Physics loss
		res = residual(net, x_collocation, t_collocation, nu)
		physics_loss = nn.MSELoss()(res, torch.zeros_like(res))

		# Total loss
		loss = bc_loss + ic_loss + physics_loss
		loss.backward()  # No need to retain the graph now
		optimizer.step()
		scheduler.step()

		if epoch % 500 == 0:
			print(f'Epoch {epoch}: Loss = {loss.item()}')

	return net


# Define the analytical solution (assuming an initial condition for simplicity)
def analytical_solution(x, t, nu):
	return -np.sin(np.pi * x) * np.exp(-nu * np.pi ** 2 * t)
	# return -2 * nu * (np.pi ** -0.5) * np.exp(-((x - 4 * nu * t) ** 2) / (4 * nu * (t + 1))) / (
	# 		1 + np.exp(-((x - 4 * nu * t) ** 2) / (4 * nu * (t + 1))))


# Extend the validation function to include the analytical solution comparison and plotting
def validate_and_plot_pinn(net, nu):
	x = torch.linspace(-1, 1, 100).view(-1)
	t = torch.linspace(0, 1, 100).view(-1)
	x_grid, t_grid = torch.meshgrid(x, t, indexing='ij')

	x_flat = x_grid.reshape(-1, 1)
	t_flat = t_grid.reshape(-1, 1)

	u_pred = net(torch.cat([x_flat, t_flat], dim=1)).detach().numpy().reshape(100, 100)
	u_analytical = analytical_solution(x_grid.numpy(), t_grid.numpy(), nu)

	# Plotting the results
	plt.figure(figsize=(12, 5))

	# PINN Solution
	plt.subplot(1, 2, 1)
	plt.contourf(t_grid.numpy(), x_grid.numpy(), u_pred, 100, cmap='jet')
	plt.colorbar()
	plt.title('PINN Solution')
	plt.xlabel('t')
	plt.ylabel('x')

	# Analytical Solution
	plt.subplot(1, 2, 2)
	plt.contourf(t_grid.numpy(), x_grid.numpy(), u_analytical, 100, cmap='jet')
	plt.colorbar()
	plt.title('Analytical Solution')
	plt.xlabel('t')
	plt.ylabel('x')

	plt.tight_layout()
	plt.show()


# Train and validate the model
net = train_pinn()
validate_and_plot_pinn(net, nu)
