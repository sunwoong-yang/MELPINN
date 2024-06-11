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

start = time.time()
# Define the parameters:

n_tanks = 6

# Initial values for the height of water in the tank:
H1, H2, H3, H4, H5, H6 = [2], [0], [0], [0], [0], [0]

# Initial values for the velocity of water in the tank:
V_1, V_2, V_3, V_4, V_5 = [0], [0], [0], [0], [0]

#initial values for the velocity of air in the tank:
V_a_1, V_a_2, V_a_3, V_a_4, V_a_5 = [0], [0], [0], [0], [0]

#Void fraction analysis
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

# Define the parameters for the calculation:
L = 0.1
L_2 = 0.1
K = 1
f = 1
g = 9.81
rho = 1000 # density of the water
rho_a = 1.225 # density of the air
t = 1
A_t = 50 # area of the tank
A_p = 0.0314 # area of the pipe
iter = 0
iteration = [0]


# Calculate the speed of the fluid:
def velocity_calculator_water(v_o, h, t, L, K, g, a_o) -> float:
    v_n_o = v_o
    v_n_p = v_n_o
    a_n = void_fraction(h)
    v_o_2 = v_o + (a_o - a_n) * (- v_o)

    while True:
        # calculate the water with regarding air
        v_n_n = (
                (
                        v_o_2
                        + t / (L * rho) * (rho * g * h + v_o_2 * rho * v_o_2 * 0.0314 / 50)
                        + K * t * (v_n_o * v_n_p) / (2 * L)
                ) / (
                        1
                        + K * t * (v_n_o + v_n_p) / (2 * L)
                        + a_n * f * t * L_2 / (L * rho)
                        + t / L * v_o_2 * 0.0314 / 50
                )
        )

        if abs(v_n_n - v_n_o) < 0.09:
            break
        v_n_o = v_n_n
        v_n_p = v_n_o
    return v_n_n # v_n_n is the new velocity of water in the tank, v_a is the velocity of air in the tank


def outer_iteration(v_o, t, L, K, g, a_n, m_d, m_r) -> tuple:
    h_d = m_d / (rho * A_t) + 1.8
    h_d1 = h_d - 1.8
    h_r = m_r / (rho * A_t)
    if h_r < 1.8:
        h = h_d - 1.8
    else:
        h = h_d - h_r
    a_o = void_fraction(h_d1)

    ############ Momentum conservation ############
    v_n_n = velocity_calculator_water(v_o, h, t, L, K, g, a_o)

    m_d_n = m_d - rho * v_n_n * t * A_p * (1 - a_o)
    m_r_n = m_r + rho * v_n_n * t * A_p * (1 - a_o)
    return m_d_n, m_r_n, a_n, v_n_n, v_n_a


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
        a = 1 - h/0.2
    else:
        a = 1
    return a

def update_flow(tank_from, tank_to, v_o, a_o, void_fraction, outer_iteration, rho, A_t, t, L, K, g):
    m_d = tank_from['mass'][-1]
    m_r = tank_to['mass'][-1]

    ############# Mass conservation ##############
    h = m_d / (rho * A_t)

    a_o = void_fraction(h) if a_o is None else a_o
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

tanks = {i: {'mass': [M_init[i-1][-1]], 'height': [H_init[i-1][-1]], 'velocity': [vel_init[i-1][-1]],
             'velocity_a': [vel_a_init[i-1][-1]], 'void_fraction': [void_init[i-1][-1]]}
         for i in range(1, n_tanks+1)}


iter = 0
while tanks[list(tanks)[-1]]['height'][-1] <= tanks[list(tanks)[-2]]['height'][-1] + 1.8:
    for i in range(1, n_tanks):
        v_o = tanks[i]['velocity'][-1] if tanks[i]['velocity'] else v_o1
        a_o = tanks[i]['void_fraction'][-1] if tanks[i]['void_fraction'] else a_o1
        m_d, m_r, a, v_n_n, v_n_a = update_flow(
            tanks[i], tanks[i + 1], v_o, a_o, void_fraction, outer_iteration, rho, A_t, t, L, K, g
        )

        if i == 1:
            print("1st tank velocity: ", v_n_n)

        tanks[i]['velocity'][-1] = v_n_n
        tanks[i]['velocity_a'][-1] = v_n_a

    # the water level in tank 6
    h6 = tanks[list(tanks)[-1]]['mass'][-1] / (rho * A_t)
    tanks[list(tanks)[-1]]['height'].append(h6)

    iter += 1
    iteration.append(iter)
    print(f"THIS IS ITERATION: {iter}\n")


end = time.time()

print(f"{end - start:.5f} sec")
# *****************END OF CALCULATION ***********************



#*****************PLOTTING THE GRAPH*************************
# # plot the graph for x = iteration y = height H1~H6
for i in range(1, n_tanks+1):
    plt.plot(iteration, tanks[i]['height'], label='H'+str(i))
plt.plot(iteration, np.ones(len(iteration))*1.8, label='1.8')
plt.legend(('H1', 'H2', 'H3', 'H4', 'H5', 'H6'), loc='right')
plt.title('Water Level in the Tank (YSW)')
plt.xlabel('Iteration')
plt.ylabel('Height')
plt.show()
plt.savefig("Height.png")


# # # plot the mass for x = iteration y = mass M1~M6
# # plt.plot(iteration, M_1, label='M1')
# # plt.plot(iteration, M_2, label='M2')
# # plt.plot(iteration, M_3, label='M3')
# # plt.plot(iteration, M_4, label='M4')
# # plt.plot(iteration, M_5, label = 'M5')
# # plt.plot(iteration, M_6, label = 'M6')
# # plt.legend(('M1', 'M2', 'M3', 'M4', 'M5', 'M6'), loc='upper right')
# # plt.title('Mass in the Tank')
# # plt.xlabel('Iteration')
# # plt.ylabel('Mass')
# # plt.show()

# plot the graph for x = iteration y = velocity V1~V6
for i in range(1, n_tanks):
    plt.plot(iteration, tanks[i]['velocity'], label='V'+str(i))
# plt.plot(iteration, V_1, label='V_1')
# plt.plot(iteration, V_2, label='V_2')
# plt.plot(iteration, V_3, label='V_3')
# plt.plot(iteration, V_4, label='V_4')
# plt.plot(iteration, V_5, label='V_5')
plt.legend(('V_1', 'V_2', 'V_3', 'V_4', 'V_5'), loc='upper right')
plt.title('Water Flow in the Tank (YSW)')
plt.xlabel('Iteration')
plt.ylabel('Velocity')
plt.show()
plt.savefig("Velocity.png")

# I would like to add 9 to all the values of H1
H1 = [x + 9 for x in H1]
H2 = [x + 7.2 for x in H2]
H3 = [x + 7.2-1.8 for x in H3]
H4 = [x + 7.2-1.8-1.8 for x in H4]
H5 = [x + 7.2-1.8-1.8-1.8 for x in H5]
H6 = [x for x in H6]
#
# Create a dataframe: and write in the Excel file
# df = pd.DataFrame({'Time' : iteration, 'H1': H1, 'H2' : H2, 'H3' : H3, 'H4' : H4, 'H5' : H5, 'H6' : H6,
#                    'V_1': V_1, 'V_2': V_2, 'V_3': V_3, 'V_4': V_4, 'V_5': V_5})
# df.to_excel('water_flow_240402_without_loss_coeff.xlsx', index=False)
