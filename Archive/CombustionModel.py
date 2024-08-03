# Author: Jacob Saunders
# Date: 21/08/2023
# Description: Model flame temperatures for thermodynamic analysis of regeneratively cooled engine
# Theory based off: https://www.cryo-rocket.com


# Librarys
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import scipy
import scipy.linalg
import pprint

# Constants
Ru = 8.3145 # Univeral gas constant [J/mol*K]
atm_to_pa = 101325 # 1 atm in pa
J_to_kJ = 1

def extractFileCSV(filename):
    # Function which takes in a csv file and extracts its infromation into a numpy array
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # filename - string - filename - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # fileArray - array - numpy array of the files lines - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    f.close()

    return data

# Take CSV of NIST data and return arrays of temperature [K], pressure [MPa], and density [kg/m^3]
def extractNISTData(filename):
    # Function which takes in a csv file and extracts NIST file data for temperature [K], pressure [MPa], and density [kg/m^3]
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # filename - string - filename - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # tempArray = array - array of temperatures - [K]
    # pressureArray - array - array of pressures - [MPa]
    # densityArray - array - array of density initial guesses - [kg/m^3]
    # enthalpyArray - array - array of enthalpies - [kJ/mol]
    #----------------------------------------------------------------------------------------------------------------------------------------
    dataNIST= extractFileCSV(filename)
     
    tempArray = []
    pressureArray = []
    denistyArray = []
    enthalpyArray = []
    for value in dataNIST[1::]:
        tempArray.append(float(value[0]))
        pressureArray.append(float(value[1]))
        denistyArray.append(float(value[2]))
        enthalpyArray.append(float(value[5]))

    return tempArray,pressureArray,denistyArray,enthalpyArray

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

#def calculateEnthalpy(T,thermoDictionary, b_element, r_element):
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
    h_T = 0# Pre-load enthalpy h_T array with zeros to reduce computational load [J/mol]


    # Determine if current temperature is above 1000K as coefficients will change
    if T > bTemp:
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
    h_T = a1 + a2*T/2 + a3*(T**2)/3 + a4*(T**3)/4 + a5*(T**4)/5 + b1/T
    h_T = h_T*Ru*T*J_to_kJ 
    return h_T

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
    # h_T - array - array of enthalpys of length T_array - [J/mol]
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

#def calculateEntropy(T, thermoDictionary, b_element, r_element):
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
    S_T = 0 # Pre-load entropy s_T array with zeros to reduce computational load [J/mol]
    

    # Determine if current temperature is above 1000K as coefficients will change
    if T > bTemp:
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
    S_T = a1*np.log(T) + a2*T + a3*(T**2)/2 + a4*(T**3)/3 + a5*(T**4)/4 + b2

    return S_T*Ru*J_to_kJ

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
        S_T[i] = S_T[i]*Ru

    return S_T

def calculateCp(T_array, thermoDictionary, b_element, r_element):
    # Function which calculates the enthalpy of a element based off given temperature range
    #--------------------------------------------------------------------------------------
    #Inputs: name - type - desc - units
    # T_array - array - temperature array - [K]
    # thermoDictionary - dictionary - dictionary containing thermal coefficients a1-5 & b1-2 - unitless
    # b_element - string - thermoDictionary elemental b reference - N/A
    # r_element - string - thermoDictionary elemetnal r reference - N/A
    #--------------------------------------------------------------------------------------
    #Outputs: name - type - desc - units
    # Cp_T - array - array of specific heat capacities of length T_array - [J/kg*K]
    #--------------------------------------------------------------------------------------
    bTemp = 1000 # Max temperature before thermal coefficients need to be changed, b for <=1000K & r > 1000K
    Cp_T = np.zeros(len(T_array)) # Pre-load Cp_T array with zeros to reduce computational load [J/mol*K]

    for i in range(len(T_array)):
        # Determine if current temperature is above 1000K as coefficients will change
        if T_array[i] > bTemp:
            a1 = thermoDictionary[r_element][0]
            a2 = thermoDictionary[r_element][1]
            a3 = thermoDictionary[r_element][2]
            a4 = thermoDictionary[r_element][3]
            a5 = thermoDictionary[r_element][4]
            b2 = thermoDictionary[r_element][6]
        else:
            a1 = thermoDictionary[b_element][0]
            a2 = thermoDictionary[b_element][1]
            a3 = thermoDictionary[b_element][2]
            a4 = thermoDictionary[b_element][3]
            a5 = thermoDictionary[b_element][4]
            b2 = thermoDictionary[b_element][6]
        
        # Calculate Cp - based off https://ntrs.nasa.gov/api/citations/19940013151/downloads/19940013151.pdf pg9
        Cp_T[i] = (a1 + a2*T_array[i] + a3*(T_array[i]**2) + a4*(T_array[i]**3) * a5*(T_array[i]**4)) *Ru 
    
    return Cp_T

#ef calculateGibbsEnergy(T, h_T, S_T):
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

    G_T = 0 # Pre-load Gibbs energy array with zeros to reduce computation load [K*J/mol]
    
    G_T = (h_T) - (T * S_T)

    return G_T

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

#def calculatePartialPressureV2(T, deltaG):
    # Function which calculates the equilibrium partial pressure constant for range of temperature values T_array
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # T_array - array - array of temperatures - [K]
    # deltaG - array - change in gibbs energies - [J/mol]
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # log10Kp - array - array of partial pressure equilibrium constants for each temperature in T_array - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    
    log10Kp = 0

    
    log10Kp = -1*deltaG/(Ru*T)
    
    return np.exp(log10Kp)

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
        log10Kp[i] = (-1*deltaG[i])/(Ru*T_array[i])
    
    return log10Kp

#----------------------------------------------------------------------------------------------------------------------------------------
# Create functions to solve the 5 simultaneous non-linear equations f1-f5
#----------------------------------------------------------------------------------------------------------------------------------------
# Calculate the first equation value with guesses a b c d 
def calculatef1(a, b, c, d, Kp, P, x, y):
    # Function which calculates the equilibrium partial pressure constant for range of temperature values T_array
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # a - float - combustion coefficient - [moles] 
    # b - float - combustion coefficient - [moles] 
    # c - float - combustion coefficient - [moles] 
    # d - float - combustion coefficient - [moles]     
    # Kp - float - Partial pressure of equation 3.1a at Temp T and pressure P - [Pa]
    # P - float - Pressure ratio from chamber to atmospheric - [Pa]
    # x - float - Number of moles of hydrogen - [moles]
    # y - float - Number of moles of oxygen - [moles]
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # f1 - float - solution to f1 outlined in cryo-rocket.com - N/A
    #---------------------------------------------------------------------------------------------------------------------------------------- 
    # Calculate C coeffecients for equation 3.1a
    C_H2O = 2*y*a-2*d
    C_H2 = x-2*y*a-b+d
    C_O2 = y-y*a-c
    molFracDenominator = -y*a+x+y+b+c+d

    # Calculate Mole fractions for products and reactants of equation 3.1a
    Z_H2 = C_H2/molFracDenominator
    Z_O2 = C_O2/molFracDenominator
    Z_H2O = C_H2O/molFracDenominator

    K1 = ((Z_H2O)**(2*y))/((Z_H2**(2*y))*(Z_O2**y))*(P**(-y))

    f1 = (Z_H2O**2)/((Z_H2**2)*Z_O2)* (P**(-y)) -Kp

    return f1

# Calculate the second equation value with guessses a b c d 
def calculatef2(a, b, c, d, Kp, P, x, y):
    # Function which calculates the equilibrium partial pressure constant for range of temperature values T_array
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # a - float - combustion coefficient - [moles] 
    # b - float - combustion coefficient - [moles] 
    # c - float - combustion coefficient - [moles] 
    # d - float - combustion coefficient - [moles]
    # Kp - float - Partial pressure of equation 3.1b at temp T and pressure P - [Pa]
    # P - float - Pressure ratio from chamber to atmospheric - [Pa]
    # x - float - Number of moles of hydrogen - [moles]
    # y - float - Number of moles of oxygen - [moles]
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # f2 - float - solution to f2 outlined in cryo-rocket.com - N/A
    #---------------------------------------------------------------------------------------------------------------------------------------- 

    # Calculate C coeffecients for equation 3.1a
    C_H = 2*b
    C_H2 = x-2*y*a-b+d
    molFracDenominator = -y*a+x+y+b+c+d

    # Calculate Mole fractions for products and reactants of equation 3.1b
    Z_H2 = C_H2/molFracDenominator
    Z_H = C_H/molFracDenominator

    K2 = ((Z_H)**2)/((Z_H2))*P
    f2 = (Z_H**2)/(Z_H2)*P-Kp


    return f2

# Calculate the third equation value with guessses a b c d 
def calculatef3(a, b, c, d, Kp, P, x, y):
    # Function which calculates the equilibrium partial pressure constant for range of temperature values T_array
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # a - float - combustion coefficient - [moles] 
    # b - float - combustion coefficient - [moles] 
    # c - float - combustion coefficient - [moles] 
    # d - float - combustion coefficient - [moles]
    # Kp - float - Partial pressure of equation 3.1b at temp T and pressure P - [Pa]
    # P - float - Pressure ratio from chamber to atmospheric - [Pa]
    # x - float - Number of moles of hydrogen - [moles]
    # y - float - Number of moles of oxygen - [moles]
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # f3 - float - solution to f2 outlined in cryo-rocket.com - N/A
    #---------------------------------------------------------------------------------------------------------------------------------------- 

    # Calculate C coeffecients for equation 3.1a
    C_O2 = y-y*a-c
    C_O = 2*c
    molFracDenominator = -y*a+x+y+b+c+d

    # Calculate Mole fractions for products and reactants of equation 3.1c
    Z_O2 = C_O2/molFracDenominator
    Z_O = C_O/molFracDenominator

    K3 = ((Z_O)**2)/((Z_O2))*P
    f3 = (Z_O**2)/(Z_O2)*P-Kp

    return f3

# Calculate the fourth equation value with guessses a b c d 
def calculatef4(a, b, c, d, Kp, P, x, y):
    # Function which calculates the equilibrium partial pressure constant for range of temperature values T_array
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # a - float - combustion coefficient - [moles] 
    # b - float - combustion coefficient - [moles] 
    # c - float - combustion coefficient - [moles] 
    # d - float - combustion coefficient - [moles]
    # Kp - float - Partial pressure of equation 3.1b at temp T and pressure P - [Pa]
    # P - float - Pressure ratio from chamber to atmospheric - [Pa]
    # x - float - Number of moles of hydrogen - [moles]
    # y - float - Number of moles of oxygen - [moles]
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # f4 - float - solution to f2 outlined in cryo-rocket.com - N/A
    #---------------------------------------------------------------------------------------------------------------------------------------- 

    # Calculate C coeffecients for equation 3.1a
    C_H2O = 2*y*a-2*d
    C_H2 = x-2*y*a-b+d
    C_OH = 2*d
    molFracDenominator = -y*a+x+y+b+c+d

    # Calculate Mole fractions for products and reactants of equation 3.1d
    Z_H2O = C_H2O/molFracDenominator
    Z_H2 = C_H2/molFracDenominator
    Z_OH = C_OH/molFracDenominator


    K4 = ((Z_H2*Z_OH**2))/((Z_H2O**2))*P
    f4 = (Z_H2 * Z_OH**2)/(Z_H2O**2)*P-Kp

    return f4

# Calculate the enthalpy of reaction (fifth equation)
def calculatef5(a, b, c, d, T, x, y, mdot_O2, mdot_H2, thermo_curve_dict):
    # Function which calculates the equilibrium partial pressure constant for range of temperature values T_array
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # a - float - combustion coefficient - [moles] 
    # b - float - combustion coefficient - [moles] 
    # c - float - combustion coefficient - [moles] 
    # d - float - combustion coefficient - [moles]
    # T - float - combustion chamber temperature - [K]
    # x - float - Number of moles of hydrogen - [moles]
    # y - float - Number of moles of oxygen - [moles]
    # enthalpyArrayProduct - array - array of entahlpies at the temperatures T_minus, T, T_plus - [J/mol]
    # mdot_O2 - float - mass flow rate of oxygen - [kg/s]
    # mdot_H2 - float - mass flow rate of hydrogen - [kg/s]
    # nistO2FileName - string - name of required oxygen NIST filename - N/A
    # nistH2FileName - string - name of required hydrogen NIST filename - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # f5 - float - solution to f2 outlined in cryo-rocket.com - N/A
    #---------------------------------------------------------------------------------------------------------------------------------------- 

    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate the enthalpy of reactants (LOX and LH2) based off pressure and temperature (note these are taken from NIST database)
    #----------------------------------------------------------------------------------------------------------------------------------------
    # tempArrayO2,pressureArrayO2,denistyArrayO2,enthalpyArrayO2 = extractNISTData(nistO2FileName)
    # tempArrayH2,pressureArrayH2,denistyArrayH2,enthalpyArrayH2 = extractNISTData(nistH2FileName)

    # # Based off temperature get the closest enthalpy from nist database
    # # Oxygen
    # closestTempNISTO2 = min(tempArrayO2, key=lambda x: abs((T) - x))
    # initialGuessIndex = tempArrayO2.index(closestTempNISTO2)
    # #enthalpyO2 = enthalpyArrayO2[initialGuessIndex]
    enthalpyO2liquid = -4277.8 # J/mol


    #Hydrogen
    # closestTempNISTH2 = min(tempArrayH2, key=lambda x: abs((T) - x))
    # initialGuessIndex = tempArrayH2.index(closestTempNISTH2)
    # #enthalpyH2 = enthalpyArrayH2[initialGuessIndex]
    enthalpyH2liquid = -7.1898 # J/mol

    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate mass flow rates of all elements (H2, H, H2O, O2, O, OH)
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Define Ci coefficients in terms of a,b,c,d,x,y for each of the elements
    C_H2O = 2*y*a-2*d
    C_H2 = x-2*y*a-b+d
    C_O2 = y-y*a-c
    C_OH = 2*d
    C_H = 2*b
    C_O = 2*c

    # Define mass generation rates for all product terms
    Ndot_H2_Prod = (C_H2/(x+y))*(mdot_H2+mdot_O2)
    Ndot_O2_Prod = (C_O2/(x+y))*(mdot_H2+mdot_O2)
    Ndot_H2O_Prod = (C_H2O/(x+y))*(mdot_H2+mdot_O2)
    Ndot_OH_Prod = (C_OH/(x+y))*(mdot_H2+mdot_O2)
    Ndot_O_Prod = (C_O/(x+y))*(mdot_H2+mdot_O2)
    Ndot_H_Prod = (C_H/(x+y))*(mdot_H2+mdot_O2)

    # Calculate enthalpies 

    T_array = [T,1000]

    h_H2 = calculateEnthalpy(T_array,thermo_curve_dict,"bH2","rH2")
    h_O2 = calculateEnthalpy(T_array, thermo_curve_dict, "bO2", "rO2")
    h_H2O = calculateEnthalpy(T_array, thermo_curve_dict, "bH2O", "rH2O")
    h_OH = calculateEnthalpy(T_array,thermo_curve_dict, "bOH", "rOH")
    h_O = calculateEnthalpy(T_array, thermo_curve_dict, "bO", "rO")
    h_H = calculateEnthalpy(T_array,thermo_curve_dict,"bH","rH")

    # Define mass generation rates for all reactant terms
    Ndot_H2_React = x # mdot_H2/fuelMass
    Ndot_O2_React = y # mdot_O2/oxidiserMass
    #NdotArray = [Ndot_H2, Ndot_O2, Ndot_H2O, Ndot_OH, Ndot_O, Ndot_H]

    #enthalpyArrayProducts = [h_H2, h_O2, h_H2O, h_OH, h_O, h_H] 
   
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate f5
    #----------------------------------------------------------------------------------------------------------------------------------------

    # sumProducts = 0
    # for i in range(6):
    #     sumProducts = sumProducts + (enthalpyArrayProducts[i]*NdotArray[i])
    
    f5 = (Ndot_H2_React*enthalpyH2liquid) + (Ndot_O2_React*enthalpyO2liquid) - (Ndot_H2_Prod*h_H2[0]) - (Ndot_O2_Prod*h_O2[0]) - (Ndot_H2O_Prod*h_H2O[0]) - (Ndot_OH_Prod*h_OH[0]) - (Ndot_O_Prod*h_O[0]) - (Ndot_H_Prod*h_H[0])



    return f5



def main():
    # main function initiates execution of code

    # Dictionary containing thermodynamic coefficients for different species of elements, r is for >1000Kelvin and b is for temperatures <= 1000K 
    thermo_curve_dict = {
    "bH2": [2.34433112E+00, 7.980521075E-03, -1.94781510E-05, 2.01572094E-08, -7.37611761E-12, -9.17935173E+02, 6.83010238E-01],
    "bO2": [3.78245636E+00, -2.99673415E-03, 9.84730200E-06, -9.68129508E-09, 3.24372836E-12, -1.06394356E+03, 3.65767573E+00],
    "bH2O": [4.19864056E+00, -2.0364341E-03, 6.52040211E-06, -5.48797062E-09, 1.77197817E-12, -3.02937267E+04, -8.49032208E-01],
    "bOH": [3.9920154E+00, -2.4013175E-03, 4.6179380E-06, -3.8811333E-09, 1.3641147E-12, 3.6150806E+03, -1.0392546E-01],
    "bH": [2.5000000E+00, 0.0000000E+00, 0.0000000E+00, 0.0000000E+00, 0.0000000E+00, 2.5473660E+04, -4.4668290E-01],
    "bO": [3.1686710E+00, -3.2793190E-03, 6.6430600E-06, -6.1280660E-09, 2.1126600E-12, 2.9122260E+04, 2.0519330E+00],
    "rH2": [2.93286579E+00, 8.26607967E-04, -1.46402335E-07, 1.54100359E-11, -6.88804432E-016, -8.13065597E+02, -1.02432887E+00],
    "rO2": [3.66096083E+00, 6.56365523E-04, -1.41149485E-07, 2.05797658E-11, -1.29913248E-15, -1.21597725E+03, 3.41536184E+00],
    "rH2O": [2.67703787E+00, 2.97318329E-03, -7.7376969E-07, 9.44336689E-11, -4.26900959E-15, -2.98858938E+04, 6.88255571E+00],
    "rOH": [2.83864607E+00, 1.10725586E-03, -2.93914978E-07, 4.2052427E-11, -2.42169092E-15, 3.94395852E+03, 5.84452662E+00],
    "rH": [2.50000286E+00, -5.65334214E-09, 3.63251723E-12, -9.19949720E-16, 7.95260746E-20, 2.54736589E+04, -4.4669849E-01],
    "rO": [2.54363697E+00, -2.73162486E-05, -4.19029520E-09, 4.95481845E-12, -4.79553694E-16, 2.92260120E+04, 4.92229457E+00]}

    loopNumber = 1

    

    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate flame temperature 
    #----------------------------------------------------------------------------------------------------------------------------------------

    # Pressure (known) atm (ratio of chamber pressure to atmospheric (200/1))
    P = 200

    # Nist datafiles based off pressure
    nistO2FileName = '20.265MpaO2.csv'
    nistH2FileName = '20.265MpaH2.csv'
    # Initial guess array for a,b,c,d,T
    A_matrix = np.array([[0.6], [0.4], [0.03], [0.05], [2000]]) # Array of initial guesses for sequence
    #A_matrix = np.array([[-12.86658251], [-122.93503193], [16.84325975], [159.21147758], [344.84876752]]) # Array of initial guesses for sequence
    
    for f in range(loopNumber):
        # Extract values from A_array
        a = A_matrix[0,0]
        b = A_matrix[1,0]
        c = A_matrix[2,0]
        d = A_matrix[3,0]
        T = A_matrix[4,0]

        # Define deltas
        delta_a = a/100
        delta_b = b/100
        delta_c = c/100
        delta_d = d/100
        delta_T = T/10

        print(A_matrix)

        
        # Define temperatures
        T_plus = T + delta_T
        T_minus = T - delta_T
        T_array = [T_minus, T, T_plus]

        # Mixture ratio
        mixtureRatio = 6 # OF ratio

        # Molecular weights of reactants
        oxidiserMass = 32 # [g/mol]
        fuelMass = 2 # [g/mol] 

        # Calculate flow rates for oxidiser and fuel reactants
        mdot_O2, y, mdot_H2, x = calculateMassFlow(oxidiserMass, fuelMass, mixtureRatio)

        #----------------------------------------------------------------------------------------------------------------------------------------
        # Calculate enthalpies of all products elements (H2, H, H2O, O2, O, OH) at temperature T
        #----------------------------------------------------------------------------------------------------------------------------------------

        # Step 1 calculate enthalpy of required elements H2, O2, H2O, H, OH for temperature range T (Note* Enthalpies are in J/kg so need to multiply by 1000 to align units)
        h_H2 = calculateEnthalpy(T_array,thermo_curve_dict,"bH2","rH2")
        h_O2 = calculateEnthalpy(T_array, thermo_curve_dict, "bO2", "rO2")
        h_H2O = calculateEnthalpy(T_array, thermo_curve_dict, "bH2O", "rH2O")
        h_OH = calculateEnthalpy(T_array,thermo_curve_dict, "bOH", "rOH")
        h_O = calculateEnthalpy(T_array, thermo_curve_dict, "bO", "rO")
        h_H = calculateEnthalpy(T_array,thermo_curve_dict,"bH","rH")

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
        Kp_a = np.exp(calculatePartialPressure(T_array,deltaG_a))
        Kp_b = np.exp(calculatePartialPressure(T_array,deltaG_b))
        Kp_c = np.exp(calculatePartialPressure(T_array,deltaG_c))
        Kp_d = np.exp(calculatePartialPressure(T_array,deltaG_d))
        #print(log10Kp_a)

        # Calculate df1s
        df1_da = (calculatef1(a-delta_a, b, c, d, Kp_a[1], P, x, y) + calculatef1(a+delta_a, b, c, d, Kp_a[1], P, x, y) )/(2*delta_a) # equation 3.1a, delta a
        df1_db = (calculatef1(a, b-delta_b, c, d, Kp_a[1], P, x, y) + calculatef1(a, b+delta_b, c, d, Kp_a[1], P, x, y))/(2*delta_b) # equation 3.1a, delta b
        df1_dc = (calculatef1(a, b, c-delta_c, d, Kp_a[1], P, x, y) + calculatef1(a, b, c+delta_c, d, Kp_a[1], P, x, y))/(2*delta_c) # equation 3.1a, delta c
        df1_dd = (calculatef1(a, b, c, d-delta_d, Kp_a[1], P, x, y) + calculatef1(a, b, c, d+delta_d, Kp_a[1], P, x, y))/(2*delta_d) # equation 3.1a, delta d
        df1_dT = (calculatef1(a, b, c, d, Kp_a[0], P, x, y) + calculatef1(a, b, c, d, Kp_a[2], P, x, y))/(2*delta_T) # equation 3.1a, delta T

        # Calculate df2s
        df2_da = (calculatef2(a-delta_a, b, c, d, Kp_b[1], P, x, y) + calculatef2(a+delta_a, b, c, d, Kp_b[1], P, x, y))/(2*delta_a) # equation 3.1b, delta a
        df2_db = (calculatef2(a, b-delta_b, c, d, Kp_b[1], P, x, y) + calculatef2(a, b+delta_b, c, d, Kp_b[1], P, x, y))/(2*delta_b) # equation 3.1b, delta b
        df2_dc = (calculatef2(a, b, c-delta_c, d, Kp_b[1], P, x, y) + calculatef2(a, b, c+delta_c, d, Kp_b[1], P, x, y))/(2*delta_c) # equation 3.1b, delta c
        df2_dd = (calculatef2(a, b, c, d-delta_d, Kp_b[1], P, x, y) + calculatef2(a, b, c, d+delta_d, Kp_b[1], P, x, y))/(2*delta_d) # equation 3.1b, delta d
        df2_dT = (calculatef2(a, b, c, d, Kp_b[0], P , x, y) +  calculatef2(a, b, c, d, Kp_b[2], P, x, y))/(2*delta_T) # equation 3.1b, delta T

        # Calculate df3s
        df3_da = (calculatef3(a-delta_a, b, c, d, Kp_c[1], P, x, y) + calculatef3(a+delta_a, b, c, d, Kp_c[1], P ,x, y))/(2*delta_a) # equation 3.1c, delta a
        df3_db = (calculatef3(a, b-delta_b, c, d, Kp_c[1], P, x, y) + calculatef3(a, b+delta_b, c, d, Kp_c[1], P ,x, y))/(2*delta_b) # equation 3.1c, delta b
        df3_dc = (calculatef3(a, b, c-delta_c, d, Kp_c[1], P, x, y) + calculatef3(a, b, c+delta_c, d, Kp_c[1], P ,x, y))/(2*delta_c) # equation 3.1c, delta c
        df3_dd = (calculatef3(a, b, c, d-delta_d, Kp_c[1], P, x, y) + calculatef3(a, b, c, d+delta_d, Kp_c[1], P ,x, y))/(2*delta_d) # equation 3.1c, delta d
        df3_dT = (calculatef3(a, b, c, d, Kp_c[0], P, x, y) + calculatef3(a, b, c, d, Kp_c[2], P ,x, y))/(2*delta_T) # equation 3.1c, delta b


        # Calculate df4s
        df4_da = (calculatef4(a-delta_a, b, c, d, Kp_d[1], P, x, y) + calculatef4(a+delta_a, b, c, d, Kp_d[1], P ,x, y))/(2*delta_a) # equation 3.1d, delta a
        df4_db = (calculatef4(a, b-delta_b, c, d, Kp_d[1], P, x, y) + calculatef4(a, b+delta_b, c, d, Kp_d[1], P ,x, y))/(2*delta_b) # equation 3.1d, delta b
        df4_dc = (calculatef4(a, b, c-delta_c, d, Kp_d[1], P, x, y) + calculatef4(a, b, c+delta_c, d, Kp_d[1], P ,x, y))/(2*delta_c) # equation 3.1d, delta c
        df4_dd = (calculatef4(a, b, c, d-delta_d, Kp_d[1], P, x, y) + calculatef4(a, b, c, d+delta_d, Kp_d[1], P ,x, y))/(2*delta_d) # equation 3.1d, delta d
        df4_dT = (calculatef4(a, b, c, d, Kp_d[0], P, x, y) + calculatef4(a, b, c, d, Kp_d[2], P ,x, y))/(2*delta_T) # equation 3.1d, delta b


        # Calculate df5s
        df5_da = (calculatef5(a-delta_a, b, c, d, T, x, y, mdot_O2, mdot_H2, thermo_curve_dict) + calculatef5(a+delta_a, b, c, d, T, x ,y, mdot_O2, mdot_H2, thermo_curve_dict))/(2*delta_a) 
        df5_db = (calculatef5(a, b-delta_b, c, d, T, x, y, mdot_O2, mdot_H2, thermo_curve_dict) + calculatef5(a, b+delta_b, c, d, T, x, y, mdot_O2, mdot_H2, thermo_curve_dict))/(2*delta_b)
        df5_dc = (calculatef5(a, b, c-delta_c, d, T, x, y, mdot_O2, mdot_H2, thermo_curve_dict) + calculatef5(a, b, c+delta_c, d, T, x, y, mdot_O2, mdot_H2, thermo_curve_dict))/(2*delta_c)
        df5_dd = (calculatef5(a, b, c, d-delta_d, T, x ,y, mdot_O2, mdot_H2, thermo_curve_dict) + calculatef5(a, b, c, d+delta_d, T, x, y, mdot_O2, mdot_H2, thermo_curve_dict))/(2*delta_d)
        df5_dT = (calculatef5(a, b, c, d, T_minus, x ,y, mdot_O2, mdot_H2, thermo_curve_dict) + calculatef5(a, b, c, d, T_plus, x ,y, mdot_O2, mdot_H2, thermo_curve_dict))/(2*delta_T)

        # Formulate the Fprime Matrix (aka A Matrix in Ax = B)
        F_prime = -1*np.array([[df1_da, df1_db, df1_dc, df1_dd, df1_dT],
                            [df2_da, df2_db, df2_dc, df2_dd, df2_dT],
                            [df3_da, df3_db, df3_dc, df3_dd, df3_dT],
                            [df4_da, df4_db, df4_dc, df4_dd, df4_dT],
                            [df5_da, df5_db, df5_dc, df5_dd, df5_dT]])
        #print(F_prime)
        #print('\n')
        # Defnine the F matrix (aka B matrix in Ax = B)
        F_matrix = -1*np.array([[calculatef1(a, b, c, d, Kp_a[1], P , x, y)],
                        [calculatef2(a, b, c, d, Kp_b[1], P, x, y)],
                        [calculatef3(a, b, c, d, Kp_c[1], P ,x, y)],
                        [calculatef4(a, b, c, d, Kp_d[1], P, x, y)],
                        [calculatef5(a, b, c, d, T, x, y, mdot_O2, mdot_H2, thermo_curve_dict)]])
        
        # implement LU decomp to solve for B (F_prime*B = F)
        P_matrix, L_matrix, U_matrix = scipy.linalg.lu(F_prime)
        # print(F_prime)
        D_matrix = np.linalg.solve(L_matrix, F_matrix) # create Z matrix (LZ = B(F))
        X_matrix = np.linalg.solve(U_matrix, D_matrix)
        A_matrix = A_matrix + X_matrix
        #print(B)

    
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate equilibrium constants for eq 3.1.a-d using partial pressure K_P
    T_array = np.arange(300,4000,0.1)
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Step 1 calculate enthalpy of required elements H2, O2, H2O, H, OH for temperature range T_array
    h_O2_test = calculateEnthalpy(T_array, thermo_curve_dict, "bO2", "rO2")
    h_H2_test = calculateEnthalpy(T_array,thermo_curve_dict,"bH2","rH2")
    h_H2O_test = calculateEnthalpy(T_array, thermo_curve_dict, "bH2O", "rH2O")
    h_H_test = calculateEnthalpy(T_array,thermo_curve_dict,"bH","rH")
    h_OH_test = calculateEnthalpy(T_array,thermo_curve_dict, "bOH", "rOH")
    h_O_test = calculateEnthalpy(T_array, thermo_curve_dict, "bO", "rO")


    # Step 2 calculate entropy of required elements H2, O2, H2O for temperature range T_array
    S_O2_test = calculateEntropy(T_array, thermo_curve_dict, "bO2", "rO2")
    S_H2_test = calculateEntropy(T_array, thermo_curve_dict, "bH2", "rH2")
    S_H2O_test = calculateEntropy(T_array, thermo_curve_dict, "bH2O", "rH2O")
    S_H_test = calculateEntropy(T_array, thermo_curve_dict, "bH","rH")
    S_OH_test = calculateEntropy(T_array,thermo_curve_dict, "bOH", "rOH")
    S_O_test = calculateEntropy(T_array,thermo_curve_dict,"bO", "rO")

    # Step 3 calculate Gibbs Energy for T_array
    G_O2_test = calculateGibbsEnergy(T_array, h_O2_test, S_O2_test)
    G_H2_test = calculateGibbsEnergy(T_array, h_H2_test, S_H2_test)
    G_H2O_test = calculateGibbsEnergy(T_array, h_H2O_test, S_H2O_test)
    G_H_test = calculateGibbsEnergy(T_array, h_H_test, S_H_test)
    G_OH_test = calculateGibbsEnergy(T_array,h_OH_test, S_OH_test)
    G_O_test = calculateGibbsEnergy(T_array, h_O_test, S_O_test)

    # Cp values for H2, O2, H2O, H, OH, O
    Cp_H2 = calculateCp(T_array, thermo_curve_dict, "bH2", "rH2")
    Cp_O2 = calculateCp(T_array, thermo_curve_dict, "bO2", "rO2")
    Cp_H2O = calculateCp(T_array, thermo_curve_dict, "bH2O", "rH2O")
    Cp_H = calculateCp(T_array, thermo_curve_dict, "bH", "rH")
    Cp_OH = calculateCp(T_array, thermo_curve_dict, "bOH", "rOH")
    Cp_O = calculateCp(T_array, thermo_curve_dict, "bO", "rO")

    x = 2
    y = 1

    # Step 4 calculate change in gibbs energy for equation 3.1b
    deltaG_a_test = calculateDeltaG([2*y, x-2*y],[G_H2O_test, G_H2_test], [x,y],[G_H2_test, G_O2_test])
    deltaG_b_test = calculateDeltaG([2], [G_H_test], [1], [G_H2_test]) #product coefficients, product gibbs, reactant coefficients, reactant gibbs
    deltaG_c_test = calculateDeltaG([2],[G_O_test],[1],[G_O2_test])
    deltaG_d_test = calculateDeltaG([1,2],[G_H2_test,G_OH_test],[2],[G_H2O_test])    

    # Step 5 calculate Partial pressure equilibrium constant for equation 3.1b
    log10Kp_a_test = calculatePartialPressure(T_array,deltaG_a_test)
    log10Kp_b_test = calculatePartialPressure(T_array,deltaG_b_test)
    log10Kp_c_test = calculatePartialPressure(T_array,deltaG_c_test)
    log10Kp_d_test = calculatePartialPressure(T_array,deltaG_d_test)

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1,1,1)

    ax.plot(T_array, log10Kp_a_test,linewidth=2.0, label = "3.1a")
    ax.plot(T_array, log10Kp_b_test, linewidth=2.0, label = "3.1b")
    ax.plot(T_array,log10Kp_c_test, linewidth=2.0, label = "3.1c")
    ax.plot(T_array,log10Kp_d_test, linewidth=2.0, label = "3.1d")
    ax.legend()


    ax.set_xlim([0,4000])
    ax.set_ylim([-250,100])
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r'$\log_{10} (K_p)$')
    ax.set_title(r'$\log_{10} (K_p)$ vs Temperature')
    ax.grid()

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1,1,1)

    ax2.plot(T_array, Cp_H2, linewidth=2.0, label = "H2")

    plt.show()

    # Moving on in the document:
    # want to calculte enthalpy of formation and enthalpy magnitude for formation of H2O


    T_array = [298.15, 250, 300] # Temperatures of interest

    # formation of H2O => H2 + 1/2 O2 = H2O
    
    h_H2 = calculateEnthalpy(T_array, thermo_curve_dict, 'bH2', 'rH2')
    h_O2 = calculateEnthalpy(T_array, thermo_curve_dict, 'bO2', 'rO2')
    h_H2O = calculateEnthalpy(T_array, thermo_curve_dict, 'bH2O','rH2O')

    heatFormationH2O = h_H2O - 0.5*h_O2 - h_H2
    enthalpyMagH2O = heatFormationH2O[0]+h_H2O[2] - h_H2O[0]

    # Display
    for i in range(len(T_array)):
        print("Temp [k]: {0} Enthalpy [kJ/mol]: {1} heatFormationH2O [J/mol]: {2}\n".format(T_array[i], h_H2O[i]/1000, heatFormationH2O[i]))

    print(h_H2)
    print(h_O2)
    print(h_H2O)
main()