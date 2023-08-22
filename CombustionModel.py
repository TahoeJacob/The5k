# Author: Jacob Saunders
# Date: 21/08/2023
# Description: Model flame temperatures for thermodynamic analysis of regeneratively cooled engine

# Librarys
import numpy as np
import matplotlib.pyplot as plt

# Constants
x = 2 # Number of moles of H2 reactant (Perfect stoichiometric mix) [moles]
y = 1 # Number of moles of O2 reactant (Perfect stoichiometric mix) [moles]
Ru = 8.3145 # Univeral gas constant [J/mol*K]

# Arrays
T_array = np.arange(0,4000,50) # Temperature array 0-4000K [K]

# Calculate enthalpy at each of the temperatures in T array

h_T = np.zeros(len(T_array)) # Pre-load enthalpy h_T array with zeros to reduce computational load [J/mol]

S_T = np.zeros(len(T_array)) # Pre-load entropy s_T array with zeros to reduce computational load [J/mol]
 
G_T = np.zeros(len(T_array)) # Pre-load Gibbs energy array with zeros to reduce computation load [K*J/mol]

for i in range(len(T_array)):
    if T_array[i] > 1000:
        a1 = 2.9328658
        a2 = 8.2660800E-4
        a3 = -1.4640200E-7
        a4 = 1.5410000E-11
        a5 = -6.8880400E-1
        b1 = -8.1306560E2
        b2 = -1.0243289
    else:
        a1 = 2.3443311
        a2 = 7.9805210E-3
        a3 = -1.9478000E-5
        a4 = 2.0157200E-8
        a5 = -7.3761200E-12
        b1 = -9.1793517E2
        b2 = 6.8301024E-1
    
    h_T[i] = (a1 + a2*T_array[i]/2 + a3*(T_array[i]^2)/3 + a4*(T_array[i]^3)/4 + a5*(T_array[i]^4)/5 + b1/T_array[i])*Ru*T_array[i]
    S_T[i] = (a1*np.log(T_array[i]) + a2*T_array[i] + a3*(T_array[i]^2)/2 + a4*(T_array[i]^3)/3 + a5*(T_array[i]^4)/4 + b2)*Ru
    G_T[i] = h_T[i] - (T_array[i] * S_T[i])

#print(len(G_T)-len(T_array))

# plot
fig, ax = plt.subplots()

ax.plot(T_array, G_T, linewidth=2.0)

# ax.set(xlim=(0, 4000), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()