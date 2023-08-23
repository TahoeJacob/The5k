# Author: Jacob Saunders
# Date: 21/08/2023
# Description: Model flame temperatures for thermodynamic analysis of regeneratively cooled engine
# Theory based off: https://www.cryo-rocket.com


# Librarys
import numpy as np
import matplotlib.pyplot as plt

# Constants
Ru = 8.3145 # Univeral gas constant [J/mol*K]

def calculateEnthalpy(T_array,thermoDictionary, b_element, r_element):
    # Function which calculates the enthalpy of a element based off given temperature range
    #--------------------------------------------------------------------------------------
    # Inputs: name - type - desc - units
    # T_array - array - temperature array - [K]
    # thermoDictionary - dictionary - dictionary containing thermal coefficients a1-5 & b1-2 - unitless
    # b_element - string - thermoDictionary elemental b reference - N/A
    # r_element - string - thermoDictionary elemetnal r reference - N/A
    #--------------------------------------------------------------------------------------
    #Outputs: name - type - desc - units
    # h_T - array - array of enthalpys of length T_array - [J/kg]
    #--------------------------------------------------------------------------------------
    bTemp = 1000 # Max temperature before thermal coefficients need to be changed, b for <=1000K & r > 1000K
    h_T = np.zeros(len(T_array)) # Pre-load enthalpy h_T array with zeros to reduce computational load [J/mol]

    for i in range(len(T_array)):

        # Determine if current temperature is above 1000K as coefficients will change
        if T_array[i] > bTemp:
           a1 = thermoDictionary[r_element][0] # extract a1 
           a2 = thermoDictionary[r_element][1] # extract a2
           a3 = thermoDictionary[r_element][2] # extract a3
           a4 = thermoDictionary[r_element][3] # extract a4
           a5 = thermoDictionary[r_element][4] # extraft a5
           b1 = thermoDictionary[r_element][5] # extract b1
        else:
           a1 = thermoDictionary[b_element][0] # extract a1
           a2 = thermoDictionary[b_element][1] # extract a2
           a3 = thermoDictionary[b_element][2] # extract a3
           a4 = thermoDictionary[b_element][3] # extract a4
           a5 = thermoDictionary[b_element][4] # extraft a5
           b1 = thermoDictionary[b_element][5] # extract b1

        # Calculate etropy - based off https://ntrs.nasa.gov/api/citations/19940013151/downloads/19940013151.pdf pg9 
        h_T[i] = (a1 + a2*T_array[i]/2 + a3*(T_array[i]^2)/3 + a4*(T_array[i]^3)/4 + a5*(T_array[i]^4)/5 + b1/T_array[i])*Ru*T_array[i]

    return h_T

def calculateEntropy(T_array, thermoDictionary, b_element, r_element):
    # Function which calculates the entropy of a element based off a givent temperature range
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: name - type - desc - units
    # T_array - array - temperature array - [K]
    # thermoDictionary - dictionary - contains thermal coefficients a1-5 & b1-2 - unitless 
    # b_element - string - thermoDictionary elemental b reference - N/A
    # r_element - string - thermoDictionary elemetnal r reference - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs: name - type - desc - units
    # S_T - array - array of entropies of length T_array - [J/kg]
    #----------------------------------------------------------------------------------------------------------------------------------------
    bTemp = 1000 # Max temperature before thermal coefficients need to be changed, b for <=1000K & r > 1000K
    S_T = np.zeros(len(T_array)) # Pre-load entropy s_T array with zeros to reduce computational load [J/mol]
    
    for i in range(len(T_array)):

        # Determine if current temperature is above 1000K as coefficients will change
        if T_array[i] > bTemp:
           a1 = thermoDictionary[r_element][0] # extract a1 
           a2 = thermoDictionary[r_element][1] # extract a2
           a3 = thermoDictionary[r_element][2] # extract a3
           a4 = thermoDictionary[r_element][3] # extract a4
           a5 = thermoDictionary[r_element][4] # extraft a5
           b2 = thermoDictionary[r_element][5] # extract b2
        else:
           a1 = thermoDictionary[b_element][0] # extract a1
           a2 = thermoDictionary[b_element][1] # extract a2
           a3 = thermoDictionary[b_element][2] # extract a3
           a4 = thermoDictionary[b_element][3] # extract a4
           a5 = thermoDictionary[b_element][4] # extraft a5
           b2 = thermoDictionary[b_element][5] # extract b2

        # Calculate etropy - based off https://ntrs.nasa.gov/api/citations/19940013151/downloads/19940013151.pdf pg9 
        S_T[i] = (a1*np.log(T_array[i]) + a2*T_array[i] + a3*(T_array[i]^2)/2 + a4*(T_array[i]^3)/3 + a5*(T_array[i]^4)/4 + b2)*Ru

    return S_T

def calculateGibbsEnergy(T_array, h_T, S_T):
    # Function which calculates the Gibbs Free Energy of a element based off a givent temperature range
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # T_array - temperature array [K]
    # h_T - array of enthalpies based on T_array of length T_array
    # S_T - array of entropies based on T_array of length T_array 
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs name - type - desc - units
    # G_T - array - array of entropies of length T_array [J]
    #----------------------------------------------------------------------------------------------------------------------------------------

    G_T = np.zeros(len(T_array)) # Pre-load Gibbs energy array with zeros to reduce computation load [K*J/mol]

    for i in range(len(T_array)):
        G_T[i] = h_T[i] - (T_array[i] * S_T[i])

    return G_T

def calculateDeltaGibbs(productCoefficients, productGibbs, reactantCoefficients, reactantGibbs):
    # Function which calculates the change in Gibbs Free Energy based off products and reactant coefficients
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # productCoefficients - array - array of product coeffiecients - N/A
    # productGibbs - array - reaction product gibbs energies, aligns with product coefficients - [J]
    # reactantCoefficients - array - array of reactant coeffiecients - N/A
    # reactantGibbs - array - reactant Gibbs Free Energy values - [J]
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # deltaG - array - array of delta gibbs energy for each temperature in T_array - [J]
    #----------------------------------------------------------------------------------------------------------------------------------------
   
    # TO BE FINISHED TOMORROW ADD METHOD TO CALCULATE DELTA G 
    # REMEBER COEFFIEINTS ARE KNOW (3.1.a x = 2 y =1, 3.1.b reactant: 1 product: 2, 3.1.c reactant: 1 prodcut: 2, 3.1.d reactant: 2 products: 1, 2) etc..

def main():
    # main function initiates execution of code

    # Dictionary containing thermodynamic coefficients for different species of elements, r is for >1000Kelvin and b is for temperatures <= 1000K 
    thermo_curve_dict = {
    "bH2": [2.3443311E+00, 7.9805210E-03, -1.9478000E-05, 2.0157200E-08, -7.3761200E-12, -9.1793517E+02, 6.8301024E-01],
    "bO2": [3.7824564E+00, -2.9967342E-03, 9.8473000E-06, -9.6812951E-09, 3.2437284E-12, -1.0639436E+03, 3.6576757E+00],
    "bH2O": [4.1986406E+00, -2.0364341E-03, 6.5204020E-06, -5.4879706E-09, 1.7719782E-12, -3.0293727E+04, -8.4903221E-01],
    "bOH": [3.9920154E+00, -2.4013175E-03, 4.6179380E-06, -3.8811333E-09, 1.3641147E-12, 3.6150806E+03, -1.0392546E-01],
    "bH": [2.5000000E+00, 0.0000000E+00, 0.0000000E+00, 0.0000000E+00, 0.0000000E+00, 2.5473660E+04, -4.4668290E-01],
    "bO": [3.1686710E+00, -3.2793190E-03, 6.6430600E-06, -6.1280660E-09, 2.1126600E-12, 2.9122260E+04, 2.0519330E+00],
    "rH2": [2.9328658E+00, 8.2660800E-04, -1.4640200E-07, 1.5410000E-11, -6.8880400E-01, -8.1306560E+02, -1.0243289E+00],
    "rO2": [3.6609608E+00, 6.5636552E-04, -1.4114949E-07, 2.0579766E-11, -1.2991325E-01, -1.2159773E+03, 3.4153618E+00],
    "rH2O": [2.6770379E+00, 2.9731833E-03, -7.7376969E-07, 9.4433669E-11, -4.2690096E-01, -2.9885894E+04, 6.8825557E+00],
    "rOH": [2.8386461E+00, 1.1072559E-03, -2.9391498E-07, 4.2052425E-11, -2.4216909E-01, 3.9439585E+03, 5.8445266E+00],
    "rH": [2.5000000E+00, -5.6533420E-09, 3.6325170E-12, -9.1994970E-16, 7.9526070E-02, 2.5473660E+04, -4.4669850E-01],
    "rO": [2.5436370E+00, -2.7316250E-05, -4.1902950E-09, 4.9548180E-12, -4.7955370E-01, 2.9226010E+04, 4.9222950E+00]}

    T_array = np.arange(1,4000,50) # Temperature array 0-4000K [K]

    # Calculate equilibrium constants for eq 3.1.a-d using partial pressure K_P
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Equation 3.1.a - 2H2 + O2 = 4H2O (Assuming x=2 y =1 (perfect stoichiometric mix))
    # Step 1 calculate enthalpy of required elements H2, O2, H2O for temperature range T_array
    h_O2 = calculateEnthalpy(T_array, thermo_curve_dict, "bO2", "rO2")
    h_H2 = calculateEnthalpy(T_array,thermo_curve_dict,"bH2","rH2")
    h_H2O = calculateEnthalpy(T_array, thermo_curve_dict, "bH2O", "rH2O")

    # Step 2 calculate entropy of required elements H2, O2, H2O for temperature range T_array
    S_O2 = calculateEntropy(T_array, thermo_curve_dict, "bO2", "rO2")
    S_H2 = calculateEntropy(T_array, thermo_curve_dict, "bH2", "rH2")
    S_H2O = calculateEntropy(T_array, thermo_curve_dict, "bH2O", "rH20")

    # Step 3 calculate Gibbs Energy for T_array
    G_O2 = calculateGibbsEnergy(T_array, h_O2, S_O2)
    G_H2 = calculateGibbsEnergy(T_array, h_H2, S_H2)
    G_H2O = calculateGibbsEnergy(T_array, h_H2O, S_H2O)

    # Step 4 calculate Delta Gibbs Free Energy for equation 3.1.a 
    deltaG_O2 = 

    print(G_O2)

main()

# # plot
# fig, ax = plt.subplots()

# ax.plot(T_array, G_T, linewidth=2.0)

# # ax.set(xlim=(0, 4000), xticks=np.arange(1, 8),
# #        ylim=(0, 8), yticks=np.arange(1, 8))

# plt.show()
