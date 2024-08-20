"""
the water flow is going to be calculated
% v_n_n = (v_o + t / L * (g * z) + K * t / (2 * L) * v_n_o ^ 2) / (1 + K * t / L * (v_n_o))
made by CW Kim on 2024/03/30
"""
import pandas as pd
import numpy as np
import math
from math import sqrt
import sklearn
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt

from typing import Tuple

start = time.time()
# Define the parameters:
# Initial values for the height of water in the tank:
H1, H2 = [2], [0]

# Initial values for the velocity of water in the tank:
V_1 = [0]

# initial values for the velocity of air in the tank:
V_a_1 = [0]

# Void fraction analysis
A_1 = [0]

# initial mass of the CV
m1, m2 = 100000, 0

M_1 = [m1]
M_2 = [m2]

h = 0
v_o = 0
v_o1 = 0

v_o_a1 = 0

v_n_a1 = 0

v_n_o = 0
v_n_p = v_o
v_n_n1 = 0

v_n_a = 0

a = 0
a_1, a_2 = 0, 1
a_o1, a_o2 = 0, 0

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
elev = 1.8
iter = 0
iteration = [0]


def velocity_calculator_water(v_o, h, t, L, K, g, a_o) -> float:
    v_n_o = v_o
    v_n_p = v_n_o
    a_n = void_fraction(h)
    v_o_2 = v_o + (a_o - a_n) * (- v_o)
    num = 0
    while True:
        v_n_n = ((v_o_2 + t / (L*rho) * ( rho*g * h + v_o_2*rho*v_o_2*A_p/A_t)+ K * t * (v_n_o * v_n_p)/(2 * L))
                 /(1+ K * t * (v_n_o+v_n_p)/(2*L)+ a_n*f*t*L_2/(L*rho)+ t/L*v_o_2*A_p/A_t))
        if abs(v_n_n-v_n_o) < 0.09*(v_n_o):
            break
        v_n_o = v_n_n
        v_n_p = v_n_o
        num += 1

    return v_n_n


def outer_iteration(v_o, t, L, K, g, a_n, m_d, m_r) -> Tuple:
    h_d = m_d/(rho*A_t) + elev
    if h_d - elev <= 0.0001:
        v_n_n = 0.0
        m_d_n = 0.0
        m_r_n = 0.0
        a_n = 1.0
    else:
        h_d1 = h_d-elev
        h_r = m_r/(rho*A_t)
        if h_r < elev:
            h = h_d - elev
        else:
            h = h_d - h_r
        a_o = void_fraction(h_d1)
        v_n_n = velocity_calculator_water(v_o, h, t, L, K, g, a_o)  # v_o, H, t, L, K, g, a_o, v_a)
        m_d_n = m_d-rho*v_n_n*t*A_p*(1-a_o)
        m_r_n = m_r+rho*v_n_n*t*A_p*(1-a_o)
    return m_d_n, m_r_n, a_n, v_n_n, v_n_a

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


while H2[-1] <= H1[-1]+elev:
    m_d1 = M_1[-1]
    m_r1 = M_2[-1]

    h1 = M_1[-1]/(rho*A_t)
    a_o1 = void_fraction(h1)
    m_d1, m_r1, a_1, v_n_n1, v_n_a1 = outer_iteration(v_o1, t, L, K, g, a_o1, m_d1, m_r1)
    v_o1 = v_n_n1
    a_o1 = a_1
    v_o_a1 = v_n_a1
    M_1 = add_list(M_1, m_d1)
    M_2 = add_list(M_2, m_r1)
    h1 = M_1[-1]/(rho*A_t)
    H1 = add_list(H1, h1)
    V_1 = add_list(V_1, v_n_n1)
    V_a_1 = add_list(V_a_1, v_n_a1)
    A_1 = add_list(A_1, a_1)

    h2 = M_2[-1]/(rho*A_t)
    H2 = add_list(H2, h2)

    iter = iter + 1

    iteration.append(iter)

print(f"Number of iterations: {iter}")

end = time.time()

print(f"{end - start:.5f} sec")
# *****************END OF CALCULATION ***********************


#*****************PLOTTING THE GRAPH*************************
# # plot the graph for x = iteration y = height H1~H6
# plt.plot(iteration, H1, label='H1')
# plt.plot(iteration, H2, label='H2')
# plt.plot(iteration, H3, label='H3')
# plt.plot(iteration, H4, label='H4')
# plt.plot(iteration, H5, label='H5')
# plt.plot(iteration, H6, label='H6')
# plt.plot(iteration, np.ones(len(iteration))*1.8, label='1.8')
# plt.legend(('H1', 'H2', 'H3', 'H4', 'H5', 'H6'), loc='right')
# plt.title('Water Level in the Tank')
# plt.xlabel('Iteration')
# plt.ylabel('Height')
# plt.show()
# #
# # Create a dataframe: and write in the excel file
# # plot the graph for x = iteration y = velocity V1~V6
plt.figure(figsize=(10, 6))

# Plotting the velocity with enhanced style
plt.plot(iteration, V_1, label='Tank 1 Velocity', alpha=0.6, linewidth=2.5)

# Labels and title
plt.xlabel('Time (seconds)', fontsize=20)
plt.ylabel('Velocity', fontsize=20)
plt.title('Tank Velocities Over Time', fontsize=24, fontweight='bold')

# Legend
plt.legend(fontsize=16, shadow=True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Save the figure
plt.savefig("figures/MELCOR_TwoTanks_velocity.png", dpi=300)

plt.show()
# I would like to add 9 to all the values of H1
H1 = [x + elev*1 for x in H1]
H2 = [x + elev*0 for x in H2]
#
# Create a dataframe: and write in the Excel file
df = pd.DataFrame({'Time' : iteration, 'H1': H1, 'H2' : H2,
                   'V_1': V_1})
# df.to_excel("NUTHOS222.xlsx")

