"""
the water flow is going to be calculated
% v_n_n = (v_o + t / L * (g * z) + K * t / (2 * L) * v_n_o ^ 2) / (1 + K * t / L * (v_n_o))
made by CW Kim on 2024/03/30
"""
import pandas as pd
import numpy as np
import math
from math import sqrt
import sympy as sp
import time
import matplotlib.pyplot as plt

start = time.time()
# Define the parameters:
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
        v_n_n = ((v_o_2
                  + t / (L*rho) * ( rho*g * h
                                    + v_o_2*rho*v_o_2*0.0314/50)
                  #+ K * t * (v_n_o * v_n_p)/(2 * L)
                 )
                 /
                 (1
                 #+ K * t * (v_n_o+v_n_p)/(2*L)
                 + a_n*f*t*L_2/(L*rho)
                 + t / L * v_o_2*0.0314/50
                 )
                 )

        if abs(v_n_n - v_n_o) < 0.09:
            break
        v_n_o = v_n_n
        v_n_p = v_n_o
    return v_n_n # v_n_n is the new velocity of water in the tank, v_a is the velocity of air in the tank


def outer_iteration(v_o, t, L, K, g, a_n, m_d, m_r) -> tuple:
    # m_d is donor mass, m_r is receiver mass, j is the mass transfer coefficient
    h = m_d/(rho*A_t)
    a_o = void_fraction(h)
    v_n_n = velocity_calculator_water(v_o, h, t, L, K, g, a_o)  # v_o, H, t, L, K, g, a_o, v_a)
    m_d_n = m_d-rho*v_n_n*t*A_p*(1-a_o)
    m_r_n = m_r+rho*v_n_n*t*A_p*(1-a_o)
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

#****************************Calculation point********************************
# Let's calculate the water flow in the tank
# calculate until the tank in the 6th is full
while H6[-1] <= H5[-1]+1.8:
    # flow path between 1 and 2
    # There should be a donor and receiver mass for this flow.
    # In this case, the donor mass is the mass of the control volume 1
    # and the receiver mass is the mass of the control volume 2
    # for each volume * = old
    m_d1 = M_1[-1]
    m_r1 = M_2[-1]

    h1 = M_1[-1]/(rho*A_t)
    a_o1 = void_fraction(h1)
    m_d1, m_r1, a_1, v_n_n1, v_n_a1 = outer_iteration(v_o1, t, L, K, g, a_o1, m_d1, m_r1)
    print("1st tank velocity: ", v_n_n1)
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


    # flow path between 2 and 3
    m_d2 = M_2[-1]
    m_r2 = M_3[-1]
    h2 = M_2[-1]/(rho*A_t)
    H2 = add_list(H2, h2)
    a_o2 = void_fraction(h2)
    m_d2, m_r2, a_2, v_n_n2, v_n_a2 = outer_iteration(v_o2, t, L, K, g, a_o2, m_d2, m_r2)
    v_o2 = v_n_n2
    a_o2 = a_2
    v_o_a2 = v_n_a2
    M_2 = update_list(M_2, m_d2)
    # print("THis is M_2", M_2)
    M_3 = add_list(M_3, m_r2)
    h2 = M_2[-1]/(rho*A_t)
    H2 = update_list(H2, h2)
    V_2 = add_list(V_2, v_n_n2)
    V_a_2 = add_list(V_a_2, v_n_a2)
    A_2 = add_list(A_2, a_2)


    # flow path between 3 and 4
    m_d3 = M_3[-1]
    m_r3 = M_4[-1]
    if iter == 0:
        a_o3 = a_3
    h3 = M_3[-1]/(rho*A_t)
    H3 = add_list(H3, h3)
    a_o3 = void_fraction(h3)
    m_d3, m_r3, a_3, v_n_n3, v_n_a3 = outer_iteration(v_o3, t, L, K, g, a_o3, m_d3, m_r3)
    # print(v_n_n3)
    v_o3 = v_n_n3
    v_o_a3 = v_n_a3
    M_3 = update_list(M_3, m_d3)
    M_4 = add_list(M_4, m_r3)
    h3 = M_3[-1]/(rho*A_t)
    H3 = update_list(H3, h3)
    V_3 = add_list(V_3, v_n_n3)
    V_a_3 = add_list(V_a_3, v_n_a3)
    A_3 = add_list(A_3, a_3)

    # flow path between 4 and 5

    m_d4 = M_4[-1]
    m_r4 = M_5[-1]
    h4 = M_4[-1]/(rho*A_t)
    H4 = add_list(H4, h4)
    a_o4 = void_fraction(h4)
    m_d4, m_r4, a_4, v_n_n4, v_n_a4 = outer_iteration(v_o4, t, L, K, g, a_o4, m_d4, m_r4)
    v_o4 = v_n_n4
    v_o_a4 = v_n_a4
    M_4 = update_list(M_4, m_d4)
    M_5 = add_list(M_5, m_r4)
    h4 = M_4[-1]/(rho*A_t)
    H4 = update_list(H4, h4)
    V_4 = add_list(V_4, v_n_n4)
    V_a_4 = add_list(V_a_4, v_n_a4)
    A_4 = add_list(A_4, a_4)

    # flow path between 5 and 6

    m_d5 = M_5[-1]
    m_r5 = M_6[-1]
    h5 = M_5[-1]/(rho*A_t)
    H5 = add_list(H5, h5)
    a_o5 = void_fraction(h5)
    m_d5, m_r5,  a_5, v_n_n5, v_n_a5 = outer_iteration(v_o5, t, L, K, g, a_o5, m_d5, m_r5)
    v_o5 = v_n_n5
    v_o_a5 = v_n_a5
    M_5 = update_list(M_5, m_d5)
    M_6 = add_list(M_6, m_r5)
    h5 = M_5[-1]/(rho*A_t)
    H5 = update_list(H5, h5)
    V_5 = add_list(V_5, v_n_n5)
    V_a_5 = add_list(V_a_5, v_n_a5)
    A_5 = add_list(A_5, a_5)

    # the water level in tank 6
    h6 = M_6[-1]/(rho*A_t)
    H6 = add_list(H6, h6)

    iter = iter + 1

    iteration.append(iter)

    print("THIS IS ITERATION : ", iter, "\n")

end = time.time()

print(f"{end - start:.5f} sec")
# *****************END OF CALCULATION ***********************



#*****************PLOTTING THE GRAPH*************************
# # plot the graph for x = iteration y = height H1~H6
plt.plot(iteration, H1, label='H1')
plt.plot(iteration, H2, label='H2')
plt.plot(iteration, H3, label='H3')
plt.plot(iteration, H4, label='H4')
plt.plot(iteration, H5, label='H5')
plt.plot(iteration, H6, label='H6')
plt.legend(('H1', 'H2', 'H3', 'H4', 'H5', 'H6'), loc='right')
plt.title('Water Level in the Tank')
plt.xlabel('Iteration')
plt.ylabel('Height')
plt.show()

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
plt.plot(iteration, V_1, label='V_1')
plt.plot(iteration, V_2, label='V_2')
plt.plot(iteration, V_3, label='V_3')
plt.plot(iteration, V_4, label='V_4')
plt.plot(iteration, V_5, label='V_5')
plt.legend(('V_1', 'V_2', 'V_3', 'V_4', 'V_5'), loc='upper right')
plt.title('Water Flow in the Tank')
plt.xlabel('Iteration')
plt.ylabel('Velocity')
plt.show()
# I would like to add 9 to all the values of H1
H1 = [x + 9 for x in H1]
H2 = [x + 7.2 for x in H2]
H3 = [x + 7.2-1.8 for x in H3]
H4 = [x + 7.2-1.8-1.8 for x in H4]
H5 = [x + 7.2-1.8-1.8-1.8 for x in H5]
H6 = [x for x in H6]
#
# Create a dataframe: and write in the Excel file
df = pd.DataFrame({'Time' : iteration, 'H1': H1, 'H2' : H2, 'H3' : H3, 'H4' : H4, 'H5' : H5, 'H6' : H6,
                   'V_1': V_1, 'V_2': V_2, 'V_3': V_3, 'V_4': V_4, 'V_5': V_5})
df.to_excel('water_flow_240402_without_loss_coeff.xlsx', index=False)
