# Author: Jacob Saunders
# Date: 21/08/2023
# Description: Model flame temperatures for thermodynamic analysis of regeneratively cooled engine
# Theory based off: https://www.cryo-rocket.com


# Librarys
import numpy as np
import matplotlib.pyplot as plt

# Constants
Ru = 8.3145 # Univeral gas constant [J/mol*K]

def calculateMassFlow(oxidiserMass, fuelMass, mixtureRatio):
    # Function which calculates the enthalpy of a element based off given temperature range
    #--------------------------------------------------------------------------------------
    # Inputs: name - type - desc - units
    # oxidiserMass - float - mass of oxidiser - [g/mol]
    # fuelMass - float - mass of fuel - [g/mol]
    # mixtureRatio - float - oxidiser to fuel mixture ratio - Unitless
    #--------------------------------------------------------------------------------------
    #Outputs: name - type - desc - units
    # massFlow - array - array of form [oxidiser mass flow rate, y, fuel mass flow rate, x] - [kg/sec]
    # mass flow rates are in kg/sec
    # x and y are the stoichiometric ratios in terms of moles for the fuel and oxidiser respectively
    #--------------------------------------------------------------------------------------

    # assume y = 1 (1 mole of oxidiser)
    y = 1
    x = oxidiserMass/(fuelMass*mixtureRatio)

    # Note this is only for H2 O2 reaction will need to re-calculate stoichiometric mixture for other oxidiser fuel mixtures
    massFlowOxidiser = oxidiserMass*y
    massFlowFuel = fuelMass*x
    
    massFlow = [massFlowOxidiser,y, massFlowFuel, x]
    return massFlow

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
        h_T[i] = h_T[i]*Ru*T_array[i]
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

    return S_T*Ru

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
        G_T[i] = (h_T[i]) - (T_array[i] * S_T[i])

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

    # Temperature array 
    T_array = np.arange(300,4000,10) # Temperature array 0-4000K [K]

    # Mixture ratio
    mixtureRatio = 6 # OF ratio

    # Molecular weights of reactants
    oxidiserMass = 32 # [g/mol]
    fuelMass = 2 # [g/mol] 

    # Calculate flow rates for oxidiser and fuel reactants
    mdot_O2, y, mdot_H2, x = calculateMassFlow(oxidiserMass, fuelMass, mixtureRatio)

    # Define mole fractions for each of the mixtures 3.1a-d
    # molFracDenominator = -y*a+x+y+b+c+d

    # Define Ci coefficients in terms of a,b,c,d,x,y for each of the elements
    # C_H2O = 2*y*a-2*d
    # C_H2 = x-2*y*a-b+d
    # C_O2 = y-ya-c
    # C_OH = 2*d
    # C_H = 2*b
    # C_O = 2*c

    # Z_O2 = C_O2/molFracDenominator
    # Z_H2O = C_H2O/molFracDenominator
    # Z_H2 = C_H2/molFracDenominator
    # Z_OH = C_OH/molFracDenominator
    # Z_H = C_H/molFracDenominator
    # Z_O = C_O/molFracDenominator

    # # Define mass generation rates for all product terms
    # Ndot_H2 = (C_H2/(x+y))*(mdot_H2+mdot_O2)
    # Ndot_O2 = (C_O2/(x+y))*(mdot_H2+mdot_O2)
    # Ndot_H2O = (C_H2O/(x+y))*(mdot_H2+mdot_O2)
    # Ndot_OH = (C_OH/(x+y))*(mdot_H2+mdot_O2)
    # Ndot_H = (C_H/(x+y))*(mdot_H2+mdot_O2)
    # Ndot_O = (C_O/(x+y))*(mdot_H2+mdot_O2)


    

    # Calculate equilibrium constants for eq 3.1.a-d using partial pressure K_P
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Step 1 calculate enthalpy of required elements H2, O2, H2O, H, OH for temperature range T_array
    h_O2 = calculateEnthalpy(T_array, thermo_curve_dict, "bO2", "rO2")
    h_H2 = calculateEnthalpy(T_array,thermo_curve_dict,"bH2","rH2")
    h_H2O = calculateEnthalpy(T_array, thermo_curve_dict, "bH2O", "rH2O")
    h_H = calculateEnthalpy(T_array,thermo_curve_dict,"bH","rH")
    h_OH = calculateEnthalpy(T_array,thermo_curve_dict, "bOH", "rOH")
    h_O = calculateEnthalpy(T_array, thermo_curve_dict, "bO", "rO")

    # Step 2 calculate entropy of required elements H2, O2, H2O for temperature range T_array
    S_O2 = calculateEntropy(T_array, thermo_curve_dict, "bO2", "rO2")
    S_H2 = calculateEntropy(T_array, thermo_curve_dict, "bH2", "rH2")
    S_H2O = calculateEntropy(T_array, thermo_curve_dict, "bH2O", "rH2O")
    S_H = calculateEntropy(T_array, thermo_curve_dict, "bH","rH")
    S_OH = calculateEntropy(T_array,thermo_curve_dict, "bOH", "rOH")
    S_O = calculateEntropy(T_array,thermo_curve_dict,"bO", "rO")

    # Step 3 calculate Gibbs Energy for T_array
    G_O2 = calculateGibbsEnergy(T_array, h_O2, S_O2)
    G_H2 = calculateGibbsEnergy(T_array, h_H2, S_H2)
    G_H2O = calculateGibbsEnergy(T_array, h_H2O, S_H2O)
    G_H = calculateGibbsEnergy(T_array, h_H, S_H)
    G_OH = calculateGibbsEnergy(T_array,h_OH, S_OH)
    G_O = calculateGibbsEnergy(T_array, h_O, S_O)

    # Step 4 calculate change in gibbs energy for equation 3.1b
    deltaG_a = calculateDeltaG([2],[G_H2O], [2,1],[G_H2, G_O2])
    deltaG_b = calculateDeltaG([2], [G_H], [1], [G_H2]) #product coefficients, product gibbs, reactant coefficients, reactant gibbs
    deltaG_c = calculateDeltaG([2],[G_O],[1],[G_O2])
    deltaG_d = calculateDeltaG([1,2],[G_H2,G_OH],[2],[G_H2O])    

    # Step 5 calculate Partial pressure equilibrium constant for equation 3.1b
    log10Kp_a = calculatePartialPressure(T_array,deltaG_a)
    log10Kp_b = calculatePartialPressure(T_array,deltaG_b)
    log10Kp_c = calculatePartialPressure(T_array,deltaG_c)
    log10Kp_d = calculatePartialPressure(T_array,deltaG_d)


    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1,1,1)

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

    # Moving on in the document:
    # want to calculte enthalpy of formation and enthalpy magnitude for formation of H2O


    T_array = [298.15, 400, 500] # Temperatures of interest

    # formation of H2O => H2 + 1/2 O2 = H2O
    
    h_H2 = calculateEnthalpy(T_array, thermo_curve_dict, 'bH2', 'rH2')
    h_O2 = calculateEnthalpy(T_array, thermo_curve_dict, 'bO2', 'rO2')
    h_H2O = calculateEnthalpy(T_array, thermo_curve_dict, 'bH2O','rH2O')

    heatFormationH2O = h_H2O - 0.5*h_O2 - h_H2
    enthalpyMagH2O = heatFormationH2O[0]+h_H2O[2] - h_H2O[0]

    # Display
    for i in range(len(T_array)):
        print("Temp [k]: {0} Enthalpy [J/mol]: {1} heatFormationH2O [J/mol]: {2}\n".format(T_array[i], h_H2O[i], heatFormationH2O[i]))

    print(enthalpyMagH2O)

    # Diagnosis texts for future use will need to change to align with what is needed
    # H2text = "Molecular Hydrgen [H2] \nEnthalpy {0}K = {1}[J/mol] \nEntropy = {2} [J/mol] \nGibbs = {3} [J/mol]".format(temp,h_H2, S_H2, G_H2)
    # Htext = "Atomic Hydrogen [H] \nEnthalpy {0}K = {1}[J/mol] \nEntropy = {2} [J/mol] \nGibbs = {3} [J/mol] \ndeltaG = {4} [J/mol] \nKp = {5} ".format(temp,h_H, S_H, G_H, deltaG, log10Kp)
    # print(H2text)
    # print(Htext)
main()