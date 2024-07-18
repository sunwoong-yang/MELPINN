import deepxde as dde
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
dde.config.set_random_seed(42)

if device.type == 'cuda':
	gpu_name = torch.cuda.get_device_name(0)
	print('GPU name:', gpu_name)


def save_results(net, input):
	if input is None:
		input = torch.tensor([[0.5, 2758]], dtype=torch.float32)
	final_results = net(input).squeeze()

	text_to_save = (
		f"Tank 1 height: {final_results[5]}\n"
		f"Tank 2 height: {final_results[6]}\n"
		f"Tank 3 height: {final_results[7]}\n"
		f"Tank 4 height: {final_results[8]}\n"
		f"Tank 5 height: {final_results[9]}\n"
		f"Tank 6 height: {final_results[10]}\n"
	)

	# Save text to a file
	with open('tank_height.txt', 'w') as file:
		file.write(text_to_save)

	print(f"Tank 1 height: {final_results[5]}")
	print(f"Tank 2 height: {final_results[6]}")
	print(f"Tank 3 height: {final_results[7]}")
	print(f"Tank 4 height: {final_results[8]}")
	print(f"Tank 5 height: {final_results[9]}")
	print(f"Tank 6 height: {final_results[10]}")

	return final_results

def MELCOR_pde(x, u):
	L = 0.1
	K = 1
	g = 9.81
	rho = 1000  # density of the water
	# t = 1
	A_t = 50  # area of the tank 50m^2
	A_p = 0.0314  # area of the pipe with diameter of 0.2m
	# elev = 1.8  # elevation difference between each tanks before the pipes are connected
	# a_o1, a_o2, a_o3, a_o4, a_o5, a_o6 = 0, 1, 1, 1, 1, 0

	def void_fraction(h):
		# Create an array to store the results
		a = torch.ones_like(h)

		# Apply the conditions
		a[h >= 0.2] = 0
		mask = (h > 0) & (h < 0.2)
		a[mask] = 1 - h[mask] / 0.2

		return a

	# u : 5(velocity btw two tanks) + 6(mass of each tank) dimension
	# x : 1(time-domain)
	vel = u[:, :5]
	height = u[:, 5:]

	# Note that below jacobians has 'j=1' due to the fake domain variable
	# First component of x means fake domain variable
	# Second component of x means temporal domain variable
	du1_dt = dde.grad.jacobian(u, x, i=0, j=1)
	du2_dt = dde.grad.jacobian(u, x, i=1, j=1)
	du3_dt = dde.grad.jacobian(u, x, i=2, j=1)
	du4_dt = dde.grad.jacobian(u, x, i=3, j=1)
	du5_dt = dde.grad.jacobian(u, x, i=4, j=1)

	dm1_dt = dde.grad.jacobian(u, x, i=5, j=1)
	dm2_dt = dde.grad.jacobian(u, x, i=6, j=1)
	dm3_dt = dde.grad.jacobian(u, x, i=7, j=1)
	dm4_dt = dde.grad.jacobian(u, x, i=8, j=1)
	dm5_dt = dde.grad.jacobian(u, x, i=9, j=1)
	dm6_dt = dde.grad.jacobian(u, x, i=10, j=1)
	# dm_receiver_dt = dde.grad.jacobian(u, t, i=1, j=0)

	# 주의: h랑 a_o는 고정된 상수가 아닌듯... 매번 update되는듯해서 이렇게 고려하는게 맞는지 고민 필요
	h1 = height[:, [0]]
	h2 = height[:, [1]]
	h3 = height[:, [2]]
	h4 = height[:, [3]]
	h5 = height[:, [4]]

	a_o1 = void_fraction(h1)
	a_o2 = void_fraction(h2)
	a_o3 = void_fraction(h3)
	a_o4 = void_fraction(h4)
	a_o5 = void_fraction(h5)

	# a_o1 = 0.  # void_fraction(h1) if a_o1 is None else a_o1
	# a_o2 = 1.  # void_fraction(h2) if a_o2 is None else a_o2
	# a_o3 = 1.  # void_fraction(h3) if a_o3 is None else a_o3
	# a_o4 = 1.  # void_fraction(h4) if a_o4 is None else a_o4
	# a_o5 = 1.  # void_fraction(h5) if a_o5 is None else a_o5

	momentum1 = L * du1_dt - g * h1 + K * torch.abs(vel[:, [0]]) * vel[:, [0]] / 2
	# print('111', torch.max(L * du1_dt), torch.min(L * du1_dt))
	# print('222', torch.max(- g * h1), torch.min(- g * h1))
	# print('333', torch.max(K * torch.abs(vel[:, [0]]) * vel[:, [0]] / 2), torch.min(K * torch.abs(vel[:, [0]]) * vel[:, [0]] / 2))
	# print('333', torch.sum(K * torch.abs(vel[:, [0]]) * vel[:, [0]] / 2))
	# print('111', L * du1_dt)
	momentum2 = L * du2_dt - g * h2 + K * torch.abs(vel[:, [1]]) * vel[:, [1]] / 2
	momentum3 = L * du3_dt - g * h3 + K * torch.abs(vel[:, [2]]) * vel[:, [2]] / 2
	momentum4 = L * du4_dt - g * h4 + K * torch.abs(vel[:, [3]]) * vel[:, [3]] / 2
	momentum5 = L * du5_dt - g * h5 + K * torch.abs(vel[:, [4]]) * vel[:, [4]] / 2

	# term1 = L * du1_dt
	# term2 = -g * h1
	# term3 = K * torch.abs(vel[0]) * vel[0] / 2

	# Print each term
	# print("Term 1 (L * du1_dt):", term1)
	# print("Term 2 (-g * h1):", term2)
	# print("Term 3 (K * torch.abs(vel[0]) * vel[0] / 2):", term3)
	# # print("ALL:", torch.max(term1 + term2 + term3))
	# print("ALL:", (term1 + term2 + term3))
	# print('vel0', vel[[0]], mass[0])

	# def update_void_fraction(m_d):
	# 	h_d = m_d / (rho * A_t)
	# 	a_o = void_fraction(h_d)
	# 	return a_o

	# a_o1 = update_void_fraction(m_d)

	continuity1 = dm1_dt/rho/A_t + vel[:, [0]] * A_p * (1 - a_o1) / A_t
	continuity2 = dm2_dt/rho/A_t + vel[:, [1]] * A_p * (1 - a_o2) / A_t - vel[:, [0]] * A_p * (1 - a_o1) / A_t
	continuity3 = dm3_dt/rho/A_t + vel[:, [2]] * A_p * (1 - a_o3) / A_t - vel[:, [1]] * A_p * (1 - a_o2) / A_t
	continuity4 = dm4_dt/rho/A_t + vel[:, [3]] * A_p * (1 - a_o4) / A_t - vel[:, [2]] * A_p * (1 - a_o3) / A_t
	continuity5 = dm5_dt/rho/A_t + vel[:, [4]] * A_p * (1 - a_o5) / A_t - vel[:, [3]] * A_p * (1 - a_o4) / A_t
	continuity6 = dm6_dt/rho/A_t - vel[:, [4]] * A_p * (1 - a_o5) / A_t

	# continuity_r = d(receiver mass)/dt = rho * v_n_n * A_p * (1 - a_o)
	# continuity_d = d(donor mass)/dt = -rho * v_n_n * A_p * (1 - a_o)

	return [momentum1, momentum2, momentum3, momentum4, momentum5, continuity1, continuity2, continuity3, continuity4,
	        continuity5, continuity6]


geom = dde.geometry.Interval(0., 1.)  # Fake - is not required for MELCOR_pde
# timedomain = dde.geometry.TimeDomain(0, 3000)
timedomain = dde.geometry.TimeDomain(0, 3)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# ic_mass_1 = dde.icbc.IC(geomtime, lambda x: 100000., lambda _, on_initial: on_initial, component=5)
# ic_mass_2 = dde.icbc.IC(geomtime, lambda x: 0., lambda _, on_initial: on_initial, component=6)
# ic_mass_3 = dde.icbc.IC(geomtime, lambda x: 0., lambda _, on_initial: on_initial, component=7)
# ic_mass_4 = dde.icbc.IC(geomtime, lambda x: 0., lambda _, on_initial: on_initial, component=8)
# ic_mass_5 = dde.icbc.IC(geomtime, lambda x: 0., lambda _, on_initial: on_initial, component=9)
# ic_mass_6 = dde.icbc.IC(geomtime, lambda x: 0., lambda _, on_initial: on_initial, component=10)
#
# ic_vel_1 = dde.icbc.IC(geomtime, lambda x: 0., lambda _, on_initial: on_initial, component=0)
# ic_vel_2 = dde.icbc.IC(geomtime, lambda x: 0., lambda _, on_initial: on_initial, component=1)
# ic_vel_3 = dde.icbc.IC(geomtime, lambda x: 0., lambda _, on_initial: on_initial, component=2)
# ic_vel_4 = dde.icbc.IC(geomtime, lambda x: 0., lambda _, on_initial: on_initial, component=3)
# ic_vel_5 = dde.icbc.IC(geomtime, lambda x: 0., lambda _, on_initial: on_initial, component=4)


data = dde.data.TimePDE(
	geomtime,
	MELCOR_pde,
	[
		# ic_mass_1,
		# ic_mass_2,
		# ic_mass_3,
		# ic_mass_4,
		# ic_mass_5,
		# ic_mass_6,
		# ic_vel_1,
		# ic_vel_2,
		# ic_vel_3,
		# ic_vel_4,
		# ic_vel_5,
	],
	num_domain=100000,
	num_initial=1000,
	num_test=1000,
)  # fake geom domain때문에 2차원이됨,, (x & t)

# print(data.__dict__)

data.train_x_all[:,[0]] = 0.5
data.train_x_bc[:,[0]] = 0.5
data.train_x[:,[0]] = 0.5
data.test_x[:,[0]] = 0.5

net = dde.nn.FNN([2] + 3 * [32] + [11], "tanh", "Glorot normal")


def hard_IC(x, y):
	others = []
	for y_idx in range(y.shape[1]):
		if y_idx == 5:
			others.append(2. - x[:, [1]] * y[:, [y_idx]])  # 2 - t * height of tank5
		else:
			others.append(x[:, [1]] * y[:, [y_idx]])  # t * height of tank except 5
	# return torch.hstack((others[0], others[1], others[2], others[3], others[4], mass_tank1, others[5], others[6], others[7], others[8], others[9]))
	return torch.hstack(others)


net.apply_output_transform(hard_IC)

model = dde.Model(data, net)

# model.compile("adam", lr=1e-3, loss_weights=[10 ** -19] * 5 + [10 ** -12] * 2 + [1] * 4)
model.compile("adam", lr=1e-3, loss_weights=[1] * 11, decay=("step", 1000, 0.97))

model.train(iterations=10000, display_every=500)
input = torch.tensor([[0.5, 2]], dtype=torch.float32)
final_results = save_results(net, input)

model.train(iterations=10000, display_every=500)
final_results = save_results(net, input)

model.train(iterations=10000, display_every=500)
final_results = save_results(net, input)

model.compile("L-BFGS", loss_weights=[1] * 11)
# # model.compile("L-BFGS", loss_weights=[1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
losshistory, train_state = model.train()
final_results = save_results(net, input)

#
# asdf = torch.tensor([[0.5, 2757]], dtype=torch.float32)
# final_results = net(asdf).squeeze()
#
# print(f"Tank 1 height: {final_results[5]}")
# print(f"Tank 2 height: {final_results[6]}")
# print(f"Tank 3 height: {final_results[7]}")
# print(f"Tank 4 height: {final_results[8]}")
# print(f"Tank 5 height: {final_results[9]}")
# print(f"Tank 6 height: {final_results[10]}")