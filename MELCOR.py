"""
the water flow is going to be calculated
% v_n_n = (v_o + t / L * (g * z) + K * t / (2 * L) * v_n_o ^ 2) / (1 + K * t / L * (v_n_o))
made by CW Kim on 2024/03/30
"""
import pandas as pd
import numpy as np
import math
from math import sqrt
# import sympy as sp
import time
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
start = time.time()
# Define the parameters:

# vairables
n_tanks = 6

# Initial values for the height of water in the tank:
H1, H2, H3, H4, H5, H6 = [2], [0], [0], [0], [0], [0]

# Initial values for the velocity of water in the tank:
V_1, V_2, V_3, V_4, V_5 = [0], [0], [0], [0], [0]

# initial values for the velocity of air in the tank:
V_a_1, V_a_2, V_a_3, V_a_4, V_a_5 = [0], [0], [0], [0], [0]

# Void fraction analysis
A_1, A_2, A_3, A_4, A_5 = [0], [1], [1], [1], [1]

# initial mass of the CV
m1, m2, m3, m4, m5, m6 = 100000, 0, 0, 0, 0, 0

M_1 = [m1]
M_2 = [m2]
M_3 = [m3]
M_4 = [m4]
M_5 = [m5]
M_6 = [m6]

# Initial values for the height and water to be calculated for water in the tank:
h = 0
v_o = 0
v_o1, v_o2, v_o3, v_o4, v_o5 = 0, 0, 0, 0, 0

v_o_a1, v_o_a2, v_o_a3, v_o_a4, v_o_a5 = 0, 0, 0, 0, 0

# Initial values for the new velocity of water in the tank:
v_n_o = 0
v_n_p = v_o
v_n_n1, v_n_n2, v_n_n3, v_n_n4, v_n_n5 = 0, 0, 0, 0, 0

v_n_a = 0

# initial void fraction of the flowpath
a = 0
a_1, a_2, a_3, a_4, a_5 = 0, 1, 1, 1, 1
a_o1, a_o2, a_o3, a_o4, a_o5 = 0, 0, 0, 0, 0
# a_o1, a_o2, a_o3, a_o4, a_o5 = 1111110, 1111110, 1111110, 1111110, 1111110

# Define the parameters for the calculation:
L = 0.1
L_2 = 0.1
K = 1
f = 1
g = 9.81
rho = 1000  # density of the water
rho_a = 1.225  # density of the air
t = 1
A_t = 50  # area of the tank 50m^2
A_p = 0.0314  # area of the pipe with diameter of 0.2m
elev = 1.8  # elevation difference between each tanks before the pipes are connected
iter = 0
iteration = [0]


# Calculate the speed of the fluid:
def velocity_calculator_water(v_o, h, t, L, K, g, a_o) -> float:
	v_n_o = v_o
	v_n_p = v_n_o
	a_n = void_fraction(h)
	v_o_2 = v_o + (a_o - a_n) * (- v_o)

	while True:
		# Original Version
		v_n_n = (
				(
						v_o_2
						+ t / (L * rho) * (rho * g * h + v_o_2 * rho * v_o_2 * A_p / A_t)
						+ K * t * (v_n_o * v_n_p) / (2 * L)
				) / (
						1
						+ K * t * (v_n_o + v_n_p) / (2 * L)
						+ a_n * f * t * L_2 / (L * rho)
						+ t / L * v_o_2 * A_p / A_t
				)
		)

		# New Version
		# v_n_n = v_o_2 + t * (g * h - 1/2 * K * [np.abs(v_n_o + v_n_o) * v_n_n - np.abs(v_n_o) * v_n_o]) / L
		# v_n_n = ((v_o_2 + t / (L * rho) * (rho * g * h) + K * t * (v_n_o * v_n_p) / (2 * L)) / (
		#             1 + K * t * (v_n_o + v_n_p) / (2 * L)))

		if abs(v_n_n - v_n_o) < 0.09 * (v_n_o):
			break
		v_n_o = v_n_n
		v_n_p = v_n_o
	return v_n_n  # v_n_n is the new velocity of water in the tank, v_a is the velocity of air in the tank


def outer_iteration(v_o, t, L, K, g, a_n, m_d, m_r) -> tuple:
	# Calculating the new height of water in the source and destination tanks based on the mass transfer.
	h_d = m_d / (rho * A_t) + elev
	h_d1 = h_d - elev
	h_r = m_r / (rho * A_t)
	if h_r < elev:
		h = h_d - elev
	else:
		h = h_d - h_r
	a_o = void_fraction(h_d1)

	# Performing the momentum conservation calculations to determine the new velocity of water.
	v_n_n = velocity_calculator_water(v_o, h, t, L, K, g, a_o)

	# Continuity eqn
	m_d_n = m_d - rho * v_n_n * t * A_p * (1 - a_o)
	m_r_n = m_r + rho * v_n_n * t * A_p * (1 - a_o)
	return m_d_n, m_r_n, a_o, v_n_n, v_n_a  # return되는 a_n이 input을 그대로 뱉는데?


# Update the new values to the list. Once this function is used, H[-1] value is no longer the same
def add_list(M, m) -> list:
	M.append(m)
	return M


def update_list(H, h) -> list:
	H[-1] = h
	return H


def void_fraction(h) -> float:
	if h >= 0.2:
		a = 0
	elif 0 < h < 0.2:
		a = 1 - h / 0.2
	else:
		a = 1
	return a


def update_flow(tank_from, tank_to, v_o, a_o, void_fraction, outer_iteration, rho, A_t, t, L, K, g):
	m_d = tank_from['mass'][-1]
	m_r = tank_to['mass'][-1]

	h = m_d / (rho * A_t)

	a_o = void_fraction(h) if a_o is None else a_o

	if h <= 0.0001:
		v_n_n = 0.0
		v_n_a = 0.0
		m_d = 0.0
		m_r = 0.0
		a = 1.0
	else:
		m_d, m_r, a, v_n_n, v_n_a = outer_iteration(v_o, t, L, K, g, a_o, m_d, m_r)

	tank_from['mass'].append(m_d)
	tank_to['mass'].append(m_r)
	tank_from['height'].append(m_d / (rho * A_t))
	tank_from['velocity'].append(v_n_n)
	tank_from['velocity_a'].append(v_n_a)
	tank_from['void_fraction'].append(a)

	return m_d, m_r, a, v_n_n, v_n_a


def add_list(existing_list, value):
	return existing_list + [value]


def update_list(existing_list, value):
	existing_list[-1] = value
	return existing_list


H_init = [H1, H2, H3, H4, H5, H6]
M_init = [M_1, M_2, M_3, M_4, M_5, M_6]
vel_init = [V_1, V_2, V_3, V_4, V_5, [0]]
vel_a_init = [V_a_1, V_a_2, V_a_3, V_a_4, V_a_5, [0]]
void_init = [A_1, A_2, A_3, A_4, A_5, [0]]

tanks = {i: {'mass': [M_init[i - 1][-1]], 'height': [H_init[i - 1][-1]], 'velocity': [vel_init[i - 1][-1]],
             'velocity_a': [vel_a_init[i - 1][-1]], 'void_fraction': [void_init[i - 1][-1]]}
         for i in range(1, n_tanks + 1)}

iter = 0
while tanks[list(tanks)[-1]]['height'][-1] <= tanks[list(tanks)[-2]]['height'][-1] + elev:
	for i in range(1, n_tanks):
		# print('vv1', tanks[i]['void_fraction'])
		v_o = tanks[i]['velocity'][-1] if tanks[i]['velocity'] else v_o1
		# if not tanks[i]['void_fraction']:
		# 	print('yyyyyyyyyyyyy')
		a_o = tanks[i]['void_fraction'][-1] if tanks[i]['void_fraction'] else a_o1
		m_d, m_r, a, v_n_n, v_n_a = update_flow(
			tanks[i], tanks[i + 1], v_o, a_o, void_fraction, outer_iteration, rho, A_t, t, L, K, g
		)

		if i == 1:
			print("1st tank velocity: ", v_n_n)

		tanks[i]['velocity'][-1] = v_n_n
		tanks[i]['velocity_a'][-1] = v_n_a

	print(f"Tank 1 height: {tanks[1]['height'][-1]}")
	print(f"Tank 2 height: {tanks[2]['height'][-1]}")
	print(f"Tank 3 height: {tanks[3]['height'][-1]}")
	print(f"Tank 4 height: {tanks[4]['height'][-1]}")
	print(f"Tank 5 height: {tanks[5]['height'][-1]}")
	print(f"Tank 6 height: {tanks[6]['height'][-1]}\n")

	print(f"Tank 1 mass: {tanks[1]['mass'][-1]}")
	print(f"Tank 2 mass: {tanks[2]['mass'][-1]}")
	print(f"Tank 3 mass: {tanks[3]['mass'][-1]}")
	print(f"Tank 4 mass: {tanks[4]['mass'][-1]}")
	print(f"Tank 5 mass: {tanks[5]['mass'][-1]}")
	print(f"Tank 6 mass: {tanks[6]['mass'][-1]}")

	# the water level in tank 6
	h6 = tanks[list(tanks)[-1]]['mass'][-1] / (rho * A_t)
	tanks[list(tanks)[-1]]['height'].append(h6)

	iter += 1
	iteration.append(iter)
	print(f"THIS IS ITERATION: {iter}\n")
# if iter >= 2000:
#     break

end = time.time()

print(f"{end - start:.5f} sec")

# *****************END OF CALCULATION ***********************

# make the tanks to have 2727 rows and 6 columns for height 2727 rows and 5 columns for velocity


# *****************PLOTTING THE GRAPH*************************
# # plot the graph for x = iteration y = height H1~H6
for i in range(1, n_tanks + 1):
	plt.plot(iteration, tanks[i]['height'], label='H' + str(i))
plt.plot(iteration, np.ones(len(iteration)) * elev, label='1.8')
plt.legend(('H1', 'H2', 'H3', 'H4', 'H5', 'H6'), loc='right')
plt.title('Water Level in the Tank (YSW)')
plt.xlabel('Iteration')
plt.ylabel('Height')
plt.savefig("figures/MELCOR_Height.png")
plt.show()

# plot the graph for x = iteration y = velocity V1~V6
for i in range(1, n_tanks):
	plt.plot(iteration, tanks[i]['velocity'], label='V' + str(i))
plt.legend(('V_1', 'V_2', 'V_3', 'V_4', 'V_5'), loc='upper right')
plt.title('Water Flow in the Tank (YSW)')
plt.xlabel('Iteration')
plt.ylabel('Velocity')
plt.savefig("figures/MELCOR_Velocity.png")
plt.show()

H1 = [x + elev * 5 for x in H1]
H2 = [x + elev * 4 for x in H2]
H3 = [x + elev * 3 for x in H3]
H4 = [x + elev * 2 for x in H4]
H5 = [x + elev for x in H5]
H6 = [x for x in H6]
#

# for i in range(1, n_tanks + 1):
# 	tanks[i]['height'] = tanks[i]['height'] + elev * np.ones_like(tanks[i]['height']) * (n_tanks - i)
# 	tanks[i]['velocity'] = tanks[i]['velocity'] + [0] * (2727 - len(tanks[i]['velocity']))
# # Create a dataframe: and write in the Excel file
# df = pd.DataFrame({'Time': iteration, 'H1': tanks[1]['height'], 'H2': tanks[2]['height'],
#                    'H3': tanks[3]['height'], 'H4': tanks[4]['height'], 'H5': tanks[5]['height'],
#                    'H6': tanks[6]['height'], 'V1': tanks[1]['velocity'], 'V2': tanks[2]['velocity'],
#                    'V3': tanks[3]['velocity'], 'V4': tanks[4]['velocity'], 'V5': tanks[5]['velocity']})
# df.to_excel('water_flow_240611_comp2.xlsx', index=False)
# print(df)


def plot_heights(tanks):
	N_tank = 6
	t_values = np.arange(len(tanks[list(tanks)[0]]['height']))
	# heights = np.zeros((t_end, N_tank))

	# for i, t in enumerate(t_values):
	# 	input = torch.tensor([[t]], dtype=torch.float32)
	# 	u_pred = net(input).squeeze().detach().numpy()
	# 	heights[i, :] = u_pred[N_tank - 1:]

	plt.figure(figsize=(10, 6))
	for tank_number in range(1, N_tank+1):
		plt.plot(t_values, tanks[tank_number]['height'], label=f'Tank {tank_number} Height')

	plt.xlabel('Time (seconds)', fontsize=20)
	plt.ylabel('Height', fontsize=20)
	plt.title('Tank Heights Over Time', fontsize=24)
	plt.legend(fontsize=16)

	plt.savefig(f'figures/MELCOR_final_heights.png', dpi=300)

	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.show()
	plt.close()

plot_heights(tanks)