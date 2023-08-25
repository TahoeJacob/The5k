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

        # Calculate enthalpy - based off https://ntrs.nasa.gov/api/citations/19940013151/downloads/19940013151.pdf pg9 
        h_T[i] = a1 + a2*T_array[i]/2 + a3*(T_array[i]**2)/3 + a4*(T_array[i]**3)/4 + a5*(T_array[i]**4)/5 + b1/T_array[i]

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
           b2 = thermoDictionary[r_element][6] # extract b2
        else:
           a1 = thermoDictionary[b_element][0] # extract a1
           a2 = thermoDictionary[b_element][1] # extract a2
           a3 = thermoDictionary[b_element][2] # extract a3
           a4 = thermoDictionary[b_element][3] # extract a4
           a5 = thermoDictionary[b_element][4] # extraft a5
           b2 = thermoDictionary[b_element][6] # extract b2

        # Calculate etropy - based off https://ntrs.nasa.gov/api/citations/19940013151/downloads/19940013151.pdf pg9 
        S_T[i] = a1*np.log(T_array[i]) + a2*T_array[i] + a3*(T_array[i]**2)/2 + a4*(T_array[i]**3)/3 + a5*(T_array[i]**4)/4 + b2

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
        G_T[i] = (h_T[i] * Ru * T_array[i]) - (T_array[i] * S_T[i] * Ru)

    return G_T

def calculateDeltaG(productCoefficients, productGibbs, reactantCoefficients, reactantGibbs):
    # Function which calculates the change in Gibbs Free Energy based off products and reactant coefficients
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # productCoefficients - array - array of product coeffiecients - N/A
    # productGibbs - array - reaction product gibbs energies, aligns with product coefficients - [J]
    # reactantCoefficients - array - array of reactant coeffiecients - N/A
    # reactantGibbs - array - reactant Gibbs Free Energy values - [J]
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # deltaG - array - array of partial pressure equilibrium constants for each temperature in T_array - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    
    GibbsProductSum = 0 # Sum of all products for delta G 
    GibbsReactantSum = 0 # Sum of all reactants for delta G

    for i in range(len(productCoefficients)):
        GibbsProductSum = GibbsProductSum + (productCoefficients[i]*productGibbs[i])
    
    for x in range(len(reactantCoefficients)):
        GibbsReactantSum = GibbsReactantSum + (reactantCoefficients[x]*reactantGibbs[x])
    
    deltaG = GibbsProductSum - GibbsReactantSum # Definition of delta Gibbs Energy

    return deltaG

def calculatePartialPressure(T_array, deltaG):
    # Function which calculates the equilibrium partial pressure constant for range of temperature values T_array
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # T_array - array - array of temperatures - [K]
    # deltaG - array - change in gibbs energies - [J/mol]
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # log10Kp - array - array of partial pressure equilibrium constants for each temperature in T_array - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    
    log10Kp = np.zeros(len(T_array))

    for i in range(len(T_array)):
        log10Kp[i] = -1*deltaG[i]/(Ru*T_array[i])
    
    return log10Kp

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
    "rH2": [2.93286579E+00, 8.26607967E-04, -1.46402335E-07, 1.54100359E-11, -6.88804432E-016, -8.13065597E+02, -1.02432887E+00],
    "rO2": [3.66096083E+00, 6.56365523E-04, -1.41149485E-07, 2.05797658E-11, -1.29913248E-15, -1.21597725E+03, 3.41536184E+00],
    "rH2O": [2.67703787E+00, 2.97318329E-03, -7.7376969E-07, 9.44336689E-11, -4.26900959E-15, -2.98858938E+04, 6.88255571E+00],
    "rOH": [2.83864607E+00, 1.10725586E-03, -2.93914978E-07, 4.2052427E-11, -2.42169092E-15, 3.94395852E+03, 5.84452662E+00],
    "rH": [2.50000286E+00, -5.65334214E-09, 3.63251723E-12, -9.19949720E-16, 7.95260746E-20, 2.54736589E+04, -4.4669849E-01],
    "rO": [2.54363697E+00, -2.73162486E-05, -4.19029520E-09, 4.95481845E-12, -4.79553694E-16, 2.92260120E+04, 4.92229457E+00]}

    T_array = np.arange(300,4000,10) # Temperature array 0-4000K [K]

    # Calculate equilibrium constants for eq 3.1.a-d using partial pressure K_P
    #----------------------------------------------------------------------------------------------------------------------------------------
    

    # Lets just get one correct data point
    #choose 3.1b as our official test at 500K the Kp should be close to -120
    temp = 1500 # temp in K

    if temp > 1000:
        a1_H = 2.50000286E+00
        a2_H = -5.65334214E-09
        a3_H = 3.63251723E-12
        a4_H = -9.19949720E-16
        a5_H = 7.95260746E-20	
        b1_H = 2.54736589E+04
        b2_H = -4.46698494E-01

        a1_H2 = 2.93286579E+00
        a2_H2 = 8.26607967E-04
        a3_H2 = -1.46402335E-07
        a4_H2 = 1.54100359E-11	
        a5_H2 = -6.888044032E-016
        b1_H2 = -8.13065597E+02
        b2_H2 = -1.02432887E+00
    else:
        a1_H = 2.5000000E+00
        a2_H = 0.0000000E+00
        a3_H = 0.0000000E+00
        a4_H = 0.0000000E+00
        a5_H = 0.0000000E+00	
        b1_H = 2.5473660E+04
        b2_H = -4.4668290E-01

        a1_H2 = 2.3443311E+00
        a2_H2 = 7.9805210E-03
        a3_H2 = -1.9478000E-05
        a4_H2 = 2.0157200E-08	
        a5_H2 = -7.3761200E-12	
        b1_H2 = -9.1793517E+02	
        b2_H2 = 6.8301024E-01


    a1_BeOH = 1.35071590E-2
    a2_BeOH = -1.85316870E-5
    a3_BeOH = 1.29424710E-8
    a4_BeOH = -3.54389610E-12
    a5_BeOH = -1.48196830E4
    b1_BeOH = 1.09928304E1
    b2_BeOH = -1.37885210E4

    Cp = a1_H2 + a2_H2*temp + a3_H2*(temp**2) + a4_H2*(temp**3) + a5_H2*(temp**4)

    #print("Heat capacity at {0}K: {1} [J/mol K]".format(temp,Cp))
    

    h_H = a1_H + a2_H*temp/2 + a3_H*(temp**2)/3 + a4_H*(temp**3)/4 + a5_H*(temp**4)/5 + b1_H/temp
    h_H2 = a1_H2 + a2_H2*temp/2 + a3_H2*(temp**2)/3 + a4_H2*(temp**3)/4 + a5_H2*(temp**4)/5 + b1_H2/temp

    S_H = a1_H*np.log(temp) + a2_H*temp + a3_H*(temp**2)/2 +a4_H*(temp**3)/3 + a5_H*(temp**4)/4 + b2_H
    S_H2 = a1_H2*np.log(temp) + a2_H2*temp + a3_H2*(temp**2)/2 +a4_H2*(temp**3)/3 + a5_H2*(temp**4)/4 + b2_H2

    G_H = (h_H*Ru*temp) - (temp*(S_H*Ru))
    G_H2 = (h_H2*Ru*temp) - (temp*(S_H2*Ru))

    deltaG = 2*G_H - G_H2
    log10Kp = -1*deltaG/(Ru*temp)

    H2text = "Molecular Hydrgen [H2] \nEnthalpy {0}K = {1}[J/mol] \nEntropy = {2} [J/mol] \nGibbs = {3} [J/mol]".format(temp,h_H2, S_H2, G_H2)
    Htext = "Atomic Hydrogen [H] \nEnthalpy {0}K = {1}[J/mol] \nEntropy = {2} [J/mol] \nGibbs = {3} [J/mol] \ndeltaG = {4} [J/mol] \nKp = {5} ".format(temp,h_H, S_H, G_H, deltaG, log10Kp)
    print(H2text)
    print(Htext)


    # Step 1 calculate enthalpy of required elements H2, O2, H2O, H, OH for temperature range T_array
    h2_O2 = calculateEnthalpy(T_array, thermo_curve_dict, "bO2", "rO2")
    h2_H2 = calculateEnthalpy(T_array,thermo_curve_dict,"bH2","rH2")
    h2_H2O = calculateEnthalpy(T_array, thermo_curve_dict, "bH2O", "rH2O")
    h2_H = calculateEnthalpy(T_array,thermo_curve_dict,"bH","rH")
    h2_OH = calculateEnthalpy(T_array,thermo_curve_dict, "bOH", "rOH")
    h2_O = calculateEnthalpy(T_array, thermo_curve_dict, "bO", "rO")

    # Step 2 calculate entropy of required elements H2, O2, H2O for temperature range T_array
    S2_O2 = calculateEntropy(T_array, thermo_curve_dict, "bO2", "rO2")
    S2_H2 = calculateEntropy(T_array, thermo_curve_dict, "bH2", "rH2")
    S2_H2O = calculateEntropy(T_array, thermo_curve_dict, "bH2O", "rH2O")
    S2_H = calculateEntropy(T_array, thermo_curve_dict, "bH","rH")
    S2_OH = calculateEntropy(T_array,thermo_curve_dict, "bOH", "rOH")
    S2_O = calculateEntropy(T_array,thermo_curve_dict,"bO", "rO")

    # Step 3 calculate Gibbs Energy for T_array
    G2_O2 = calculateGibbsEnergy(T_array, h2_O2, S2_O2)
    G2_H2 = calculateGibbsEnergy(T_array, h2_H2, S2_H2)
    G2_H2O = calculateGibbsEnergy(T_array, h2_H2O, S2_H2O)
    G2_H = calculateGibbsEnergy(T_array, h2_H, S2_H)
    G2_OH = calculateGibbsEnergy(T_array,h2_OH, S2_OH)
    G2_O = calculateGibbsEnergy(T_array, h2_O, S2_O)

    # Step 4 calculate change in gibbs energy for equation 3.1b
    deltaG2_a = calculateDeltaG([2],[G2_H2O], [2,1],[G2_H2, G2_O2])
    deltaG2_b = calculateDeltaG([2], [G2_H], [1], [G2_H2]) #product coefficients, product gibbs, reactant coefficients, reactant gibbs
    deltaG2_c = calculateDeltaG([2],[G2_O],[1],[G2_O2])
    deltaG2_d = calculateDeltaG([1,2],[G2_H2,G2_OH],[2],[G2_H2O])    


    # Step 5 calculate Partial pressure equilibrium constant for equation 3.1b
    log10Kp_a = calculatePartialPressure(T_array,deltaG2_a)
    log10Kp_b = calculatePartialPressure(T_array,deltaG2_b)
    log10Kp_c = calculatePartialPressure(T_array,deltaG2_c)
    log10Kp_d = calculatePartialPressure(T_array,deltaG2_d)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(T_array, log10Kp_a,linewidth=2.0, label = "3.1a")
    ax.plot(T_array, log10Kp_b, linewidth=2.0, label = "3.1b")
    ax.plot(T_array,log10Kp_c, linewidth=2.0, label = "3.1c")
    ax.plot(T_array,log10Kp_d, linewidth=2.0, label = "3.1d")
    ax.legend()


    ax.set_xlim([0,4000])
    ax.set_ylim([-250,100])
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r'$\log_{10} (K_p)$')
    ax.set_title(r'$\log_{10} (K_p)$ vs Temperature')
    ax.grid()
    plt.show()

main()