# Author: Jacob Saunders
# Date: 21/08/2023
# Description: Model flame temperatures for thermodynamic analysis of regeneratively cooled engine
# Theory based off: https://www.cryo-rocket.com


# Librarys
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# Constants
Ru = 8.3145 # Univeral gas constant [J/mol*K]
atm_to_pa = 101325 # 1 atm in pa

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

def calculateEnthalpy(T,thermoDictionary, b_element, r_element):
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
        h_T = h_T*Ru*T * 1000
    return h_T



def calculateEnthalpyV2(T_array,thermoDictionary, b_element, r_element):
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
        h_T[i] = h_T[i]*Ru*T_array[i] * 1000
    return h_T

def calculateEntropy(T, thermoDictionary, b_element, r_element):
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

    return S_T*Ru*1000

def calculateEntropyV2(T_array, thermoDictionary, b_element, r_element):
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

    return S_T*Ru*1000

def calculateGibbsEnergy(T, h_T, S_T):
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

def calculateGibbsEnergyV2(T_array, h_T, S_T):
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

def calculatePartialPressure(T, deltaG):
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
    
    return log10Kp

def calculatePartialPressureV2(T_array, deltaG):
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


    f1 = (Z_H2O**2)/((Z_H2**2)*Z_O2)*P*np.exp(-y)-Kp

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

    f4 = (Z_H2 * Z_OH**2)/(Z_H2O**2)*P-Kp

    return f4

# Calculate the enthalpy of reaction (fifth equation)
def calculatef5(a, b, c, d, T, x, y, enthalpyArrayProducts, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict):
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
    # mdot_O2 - float - mass flow rate of oxygen - [kg/s]
    # mdot_H2 - float - mass flow rate of hydrogen - [kg/s]
    # nistO2FileName - string - name of required oxygen NIST filename - N/A
    # nistH2FileName - string - name of required hydrogen NIST filename - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # f4 - float - solution to f2 outlined in cryo-rocket.com - N/A
    #---------------------------------------------------------------------------------------------------------------------------------------- 

    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate the enthalpy of reactants (LOX and LH2) based off pressure and temperature (note these are taken from NIST database)
    #----------------------------------------------------------------------------------------------------------------------------------------
    tempArrayO2,pressureArrayO2,denistyArrayO2,enthalpyArrayO2 = extractNISTData(nistO2FileName)
    tempArrayH2,pressureArrayH2,denistyArrayH2,enthalpyArrayH2 = extractNISTData(nistH2FileName)

    # Based off temperature get the closest enthalpy from nist database
    # Oxygen
    closestTempNISTO2 = min(tempArrayO2, key=lambda x: abs((T) - x))
    initialGuessIndex = tempArrayO2.index(closestTempNISTO2)
    enthalpyO2 = enthalpyArrayO2[initialGuessIndex]

    #Hydrogen
    closestTempNISTH2 = min(tempArrayH2, key=lambda x: abs((T) - x))
    initialGuessIndex = tempArrayH2.index(closestTempNISTH2)
    enthalpyH2 = enthalpyArrayH2[initialGuessIndex]

    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate mass flow rates of all elements (H2, H, H2O, O2, O, OH)
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Define mole fractions for each of the mixtures 3.1a-d
    molFracDenominator = -y*a+x+y+b+c+d
    # Define Ci coefficients in terms of a,b,c,d,x,y for each of the elements
    C_H2O = 2*y*a-2*d
    C_H2 = x-2*y*a-b+d
    C_O2 = y-y*a-c
    C_OH = 2*d
    C_H = 2*b
    C_O = 2*c

    Z_O2 = C_O2/molFracDenominator
    Z_H2O = C_H2O/molFracDenominator
    Z_H2 = C_H2/molFracDenominator
    Z_OH = C_OH/molFracDenominator
    Z_H = C_H/molFracDenominator
    Z_O = C_O/molFracDenominator

    # Define mass generation rates for all product terms
    Ndot_H2 = (C_H2/(x+y))*(mdot_H2+mdot_O2)
    Ndot_O2 = (C_O2/(x+y))*(mdot_H2+mdot_O2)
    Ndot_H2O = (C_H2O/(x+y))*(mdot_H2+mdot_O2)
    Ndot_OH = (C_OH/(x+y))*(mdot_H2+mdot_O2)
    Ndot_O = (C_O/(x+y))*(mdot_H2+mdot_O2)
    Ndot_H = (C_H/(x+y))*(mdot_H2+mdot_O2)
    NdotArray = [Ndot_H2, Ndot_O2, Ndot_H2O, Ndot_OH, Ndot_O, Ndot_H]
   
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate f5
    #----------------------------------------------------------------------------------------------------------------------------------------

    sumProducts = 0
    for i in range(6):
        sumProducts = sumProducts + enthalpyArrayProducts[i]*NdotArray[i]
    
    f5 = (Ndot_H2*enthalpyH2) + (Ndot_O2*enthalpyO2) - sumProducts


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

    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate flame temperature 
    #----------------------------------------------------------------------------------------------------------------------------------------

    # Pressure (known) atm (convert to Pa)
    P = 200 * atm_to_pa

    # Nist datafiles based off pressure
    nistO2FileName = '20.265MpaO2.csv'
    nistH2FileName = '20.265MpaH2.csv'
    # Initial guess array for a,b,c,d,T
    A_array = np.array([[0.1], [0.2], [0.3], [0.5], [3360]]) # Array of initial guesses for sequence

    # Extract values from A_array
    a = A_array[0,0]
    b = A_array[1,0]
    c = A_array[2,0]
    d = A_array[3,0]
    T = A_array[4,0]

    # Define deltas
    delta_a = a/1000
    delta_b = b/1000
    delta_c = c/1000
    delta_d = d/1000
    delta_T = T/1000

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
    h_H2 = calculateEnthalpy(T,thermo_curve_dict,"bH2","rH2")
    h_O2 = calculateEnthalpy(T, thermo_curve_dict, "bO2", "rO2")
    h_H2O = calculateEnthalpy(T, thermo_curve_dict, "bH2O", "rH2O")
    h_OH = calculateEnthalpy(T,thermo_curve_dict, "bOH", "rOH")
    h_O = calculateEnthalpy(T, thermo_curve_dict, "bO", "rO")
    h_H = calculateEnthalpy(T,thermo_curve_dict,"bH","rH")

    # Step 2 calculate entropy of required elements H2, O2, H2O for temperature range T_array
    S_O2 = calculateEntropy(T, thermo_curve_dict, "bO2", "rO2")
    S_H2 = calculateEntropy(T, thermo_curve_dict, "bH2", "rH2")
    S_H2O = calculateEntropy(T, thermo_curve_dict, "bH2O", "rH2O")
    S_H = calculateEntropy(T, thermo_curve_dict, "bH","rH")
    S_OH = calculateEntropy(T,thermo_curve_dict, "bOH", "rOH")
    S_O = calculateEntropy(T,thermo_curve_dict,"bO", "rO")

    # Step 3 calculate Gibbs Energy for T_array
    G_O2 = calculateGibbsEnergy(T, h_O2, S_O2)
    G_H2 = calculateGibbsEnergy(T, h_H2, S_H2)
    G_H2O = calculateGibbsEnergy(T, h_H2O, S_H2O)
    G_H = calculateGibbsEnergy(T, h_H, S_H)
    G_OH = calculateGibbsEnergy(T,h_OH, S_OH)
    G_O = calculateGibbsEnergy(T, h_O, S_O)

    # Step 4 calculate change in gibbs energy for equation 3.1b
    deltaG_a = calculateDeltaG([2],[G_H2O], [2,1],[G_H2, G_O2])
    deltaG_b = calculateDeltaG([2], [G_H], [1], [G_H2]) #product coefficients, product gibbs, reactant coefficients, reactant gibbs
    deltaG_c = calculateDeltaG([2],[G_O],[1],[G_O2])
    deltaG_d = calculateDeltaG([1,2],[G_H2,G_OH],[2],[G_H2O])    

    # Step 5 calculate Partial pressure equilibrium constant for equation 3.1b
    log10Kp_a = calculatePartialPressure(T,deltaG_a)
    log10Kp_b = calculatePartialPressure(T,deltaG_b)
    log10Kp_c = calculatePartialPressure(T,deltaG_c)
    log10Kp_d = calculatePartialPressure(T,deltaG_d)
    enthalpyArrayProducts = [h_H2, h_O2, h_H2O, h_OH, h_O, h_H] 
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate enthalpies of all products elements (H2, H, H2O, O2, O, OH) at temperature T + delta_T
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Step 1 calculate enthalpy of required elements H2, O2, H2O, H, OH for temperature range T (Note* Enthalpies are in J/kg so need to multiply by 1000 to align units)
    h_H2_deltaTplus = calculateEnthalpy(T + delta_T,thermo_curve_dict,"bH2","rH2")
    h_O2_deltaTplus = calculateEnthalpy(T + delta_T, thermo_curve_dict, "bO2", "rO2")
    h_H2O_deltaTplus = calculateEnthalpy(T + delta_T, thermo_curve_dict, "bH2O", "rH2O")
    h_OH_deltaTplus = calculateEnthalpy(T + delta_T,thermo_curve_dict, "bOH", "rOH")
    h_O_deltaTplus = calculateEnthalpy(T + delta_T, thermo_curve_dict, "bO", "rO")
    h_H_deltaTplus = calculateEnthalpy(T + delta_T,thermo_curve_dict,"bH","rH")

    # Step 2 calculate entropy of required elements H2, O2, H2O for temperature range T_array
    S_O2_deltaTplus = calculateEntropy(T + delta_T, thermo_curve_dict, "bO2", "rO2")
    S_H2_deltaTplus = calculateEntropy(T + delta_T, thermo_curve_dict, "bH2", "rH2")
    S_H2O_deltaTplus = calculateEntropy(T + delta_T, thermo_curve_dict, "bH2O", "rH2O")
    S_H_deltaTplus = calculateEntropy(T + delta_T, thermo_curve_dict, "bH","rH")
    S_OH_deltaTplus = calculateEntropy(T + delta_T,thermo_curve_dict, "bOH", "rOH")
    S_O_deltaTplus = calculateEntropy(T + delta_T,thermo_curve_dict,"bO", "rO")

    # Step 3 calculate Gibbs Energy for T_array
    G_O2_deltaTplus = calculateGibbsEnergy(T + delta_T, h_O2_deltaTplus, S_O2_deltaTplus)
    G_H2_deltaTplus = calculateGibbsEnergy(T + delta_T, h_H2_deltaTplus, S_H2_deltaTplus)
    G_H2O_deltaTplus = calculateGibbsEnergy(T + delta_T, h_H2O_deltaTplus, S_H2O_deltaTplus)
    G_H_deltaTplus = calculateGibbsEnergy(T + delta_T, h_H_deltaTplus, S_H_deltaTplus)
    G_OH_deltaTplus = calculateGibbsEnergy(T + delta_T,h_OH_deltaTplus, S_OH_deltaTplus)
    G_O_deltaTplus = calculateGibbsEnergy(T + delta_T, h_O_deltaTplus, S_O_deltaTplus)

    # Step 4 calculate change in gibbs energy for equation 3.1b
    deltaG_a_deltaTplus = calculateDeltaG([2],[G_H2O_deltaTplus], [2,1],[G_H2_deltaTplus, G_O2_deltaTplus])
    deltaG_b_deltaTplus = calculateDeltaG([2], [G_H_deltaTplus], [1], [G_H2_deltaTplus]) #product coefficients, product gibbs, reactant coefficients, reactant gibbs
    deltaG_c_deltaTplus = calculateDeltaG([2],[G_O_deltaTplus],[1],[G_O2_deltaTplus])
    deltaG_d_deltaTplus = calculateDeltaG([1,2],[G_H2_deltaTplus,G_OH_deltaTplus],[2],[G_H2O_deltaTplus])    

    # Step 5 calculate Partial pressure equilibrium constant for equation 3.1b
    log10Kp_a_deltaTplus = calculatePartialPressure(T,deltaG_a_deltaTplus)
    log10Kp_b_deltaTplus = calculatePartialPressure(T,deltaG_b_deltaTplus)
    log10Kp_c_deltaTplus = calculatePartialPressure(T,deltaG_c_deltaTplus)
    log10Kp_d_deltaTplus = calculatePartialPressure(T,deltaG_d_deltaTplus)
    enthalpyArrayProducts_deltaTplus = [h_H2_deltaTplus, h_O2_deltaTplus, h_H2O_deltaTplus, h_OH_deltaTplus, h_O_deltaTplus, h_H_deltaTplus]

    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate enthalpies of all products elements (H2, H, H2O, O2, O, OH) at temperature T + delta_T
    #----------------------------------------------------------------------------------------------------------------------------------------
       # Step 1 calculate enthalpy of required elements H2, O2, H2O, H, OH for temperature range T (Note* Enthalpies are in J/kg so need to multiply by 1000 to align units)
    h_H2_deltaTminus = calculateEnthalpy(T - delta_T,thermo_curve_dict,"bH2","rH2")
    h_O2_deltaTminus = calculateEnthalpy(T - delta_T, thermo_curve_dict, "bO2", "rO2")
    h_H2O_deltaTminus = calculateEnthalpy(T - delta_T, thermo_curve_dict, "bH2O", "rH2O")
    h_OH_deltaTminus = calculateEnthalpy(T - delta_T,thermo_curve_dict, "bOH", "rOH")
    h_O_deltaTminus = calculateEnthalpy(T - delta_T, thermo_curve_dict, "bO", "rO")
    h_H_deltaTminus = calculateEnthalpy(T - delta_T,thermo_curve_dict,"bH","rH")

    # Step 2 calculate entropy of required elements H2, O2, H2O for temperature range T_array
    S_O2_deltaTminus = calculateEntropy(T - delta_T, thermo_curve_dict, "bO2", "rO2")
    S_H2_deltaTminus = calculateEntropy(T - delta_T, thermo_curve_dict, "bH2", "rH2")
    S_H2O_deltaTminus = calculateEntropy(T - delta_T, thermo_curve_dict, "bH2O", "rH2O")
    S_H_deltaTminus = calculateEntropy(T - delta_T, thermo_curve_dict, "bH","rH")
    S_OH_deltaTminus = calculateEntropy(T - delta_T,thermo_curve_dict, "bOH", "rOH")
    S_O_deltaTminus = calculateEntropy(T - delta_T,thermo_curve_dict,"bO", "rO")

    # Step 3 calculate Gibbs Energy for T_array
    G_O2_deltaTminus = calculateGibbsEnergy(T - delta_T, h_O2_deltaTminus, S_O2_deltaTminus)
    G_H2_deltaTminus = calculateGibbsEnergy(T - delta_T, h_H2_deltaTminus, S_H2_deltaTminus)
    G_H2O_deltaTminus = calculateGibbsEnergy(T - delta_T, h_H2O_deltaTminus, S_H2O_deltaTminus)
    G_H_deltaTminus = calculateGibbsEnergy(T - delta_T, h_H_deltaTminus, S_H_deltaTminus)
    G_OH_deltaTminus = calculateGibbsEnergy(T - delta_T,h_OH_deltaTminus, S_OH_deltaTminus)
    G_O_deltaTminus = calculateGibbsEnergy(T - delta_T, h_O_deltaTminus, S_O_deltaTminus)

    # Step 4 calculate change in gibbs energy for equation 3.1b
    deltaG_a_deltaTminus = calculateDeltaG([2],[G_H2O_deltaTminus], [2,1],[G_H2_deltaTminus, G_O2_deltaTminus])
    deltaG_b_deltaTminus = calculateDeltaG([2], [G_H_deltaTminus], [1], [G_H2_deltaTminus]) #product coefficients, product gibbs, reactant coefficients, reactant gibbs
    deltaG_c_deltaTminus = calculateDeltaG([2],[G_O_deltaTminus],[1],[G_O2_deltaTminus])
    deltaG_d_deltaTminus = calculateDeltaG([1,2],[G_H2_deltaTminus,G_OH_deltaTminus],[2],[G_H2O_deltaTminus])    

    # Step 5 calculate Partial pressure equilibrium constant for equation 3.1b
    log10Kp_a_deltaTminus = calculatePartialPressure(T,deltaG_a_deltaTminus)
    log10Kp_b_deltaTminus = calculatePartialPressure(T,deltaG_b_deltaTminus)
    log10Kp_c_deltaTminus = calculatePartialPressure(T,deltaG_c_deltaTminus)
    log10Kp_d_deltaTminus = calculatePartialPressure(T,deltaG_d_deltaTminus)
    enthalpyArrayProducts_deltaTminus = [h_H2_deltaTminus, h_O2_deltaTminus, h_H2O_deltaTminus, h_OH_deltaTminus, h_O_deltaTminus, h_H_deltaTminus] 

    # Calculate df1s
    df1_da = (calculatef1(a+delta_a, b, c, d, log10Kp_a, P, x, y) - calculatef1(a-delta_a, b, c, d, log10Kp_a, P, x, y))/(2*delta_a) # equation 3.1a, delta a
    df1_db = (calculatef1(a, b+delta_b, c, d, log10Kp_a, P, x, y) - calculatef1(a, b-delta_b, c, d, log10Kp_a, P, x, y))/(2*delta_b) # equation 3.1a, delta b
    df1_dc = (calculatef1(a, b, c+delta_c, d, log10Kp_a, P, x, y) - calculatef1(a, b, c-delta_c, d, log10Kp_a, P, x, y))/(2*delta_c) # equation 3.1a, delta c
    df1_dd = (calculatef1(a, b, c, d+delta_d, log10Kp_a, P, x, y) - calculatef1(a, b, c, d-delta_d, log10Kp_a, P, x, y))/(2*delta_d) # equation 3.1a, delta d
    df1_dT = (calculatef1(a, b, c, d, log10Kp_a_deltaTplus, P, x, y) - calculatef1(a, b, c, d, log10Kp_a_deltaTminus, P, x, y))/(2*delta_T) # equation 3.1a, delta T

    # Calculate df2s
    df2_da = (calculatef2(a+delta_a, b, c, d, log10Kp_b, P, x, y) - calculatef2(a-delta_a, b, c, d, log10Kp_b, P, x, y))/(2*delta_a) # equation 3.1b, delta a
    df2_db = (calculatef2(a, b+delta_b, c, d, log10Kp_b, P, x, y) - calculatef2(a, b-delta_b, c, d, log10Kp_b, P, x, y))/(2*delta_b) # equation 3.1b, delta b
    df2_dc = (calculatef2(a, b, c+delta_c, d, log10Kp_b, P, x, y) - calculatef2(a, b, c-delta_c, d, log10Kp_b, P, x, y))/(2*delta_c) # equation 3.1b, delta c
    df2_dd = (calculatef2(a, b, c, d+delta_d, log10Kp_b, P, x, y) - calculatef2(a, b, c, d+delta_d, log10Kp_b, P, x, y))/(2*delta_d) # equation 3.1b, delta d
    df2_dT = (calculatef2(a, b, c, d, log10Kp_b_deltaTplus, P , x, y) - calculatef2(a, b, c, d, log10Kp_b_deltaTminus, P, x, y))/(2*delta_T) # equation 3.1b, delta T

    # Calculate df3s
    df3_da = (calculatef3(a+delta_a, b, c, d, log10Kp_c, P, x, y) - calculatef3(a-delta_a, b, c, d, log10Kp_c, P ,x, y))/(2*delta_a) # equation 3.1c, delta a
    df3_db = (calculatef3(a, b+delta_b, c, d, log10Kp_c, P, x, y) - calculatef3(a, b-delta_b, c, d, log10Kp_c, P ,x, y))/(2*delta_b) # equation 3.1c, delta b
    df3_dc = (calculatef3(a, b, c+delta_c, d, log10Kp_c, P, x, y) - calculatef3(a, b, c-delta_c, d, log10Kp_c, P ,x, y))/(2*delta_c) # equation 3.1c, delta c
    df3_dd = (calculatef3(a, b, c, d+delta_d, log10Kp_c, P, x, y) - calculatef3(a, b, c, d-delta_d, log10Kp_c, P ,x, y))/(2*delta_d) # equation 3.1c, delta d
    df3_dT = (calculatef3(a, b, c, d, log10Kp_c_deltaTplus, P, x, y) - calculatef3(a, b, c, d, log10Kp_c_deltaTminus, P ,x, y))/(2*delta_T) # equation 3.1c, delta b

    # Calculate df4s
    df4_da = (calculatef4(a+delta_a, b, c, d, log10Kp_d, P, x, y) - calculatef4(a-delta_a, b, c, d, log10Kp_d, P ,x, y))/(2*delta_a) # equation 3.1d, delta a
    df4_db = (calculatef4(a, b+delta_b, c, d, log10Kp_d, P, x, y) - calculatef4(a, b-delta_b, c, d, log10Kp_d, P ,x, y))/(2*delta_b) # equation 3.1d, delta b
    df4_dc = (calculatef4(a, b, c+delta_c, d, log10Kp_d, P, x, y) - calculatef4(a, b, c-delta_c, d, log10Kp_d, P ,x, y))/(2*delta_c) # equation 3.1d, delta c
    df4_dd = (calculatef4(a, b, c, d+delta_d, log10Kp_d, P, x, y) - calculatef4(a, b, c, d-delta_d, log10Kp_d, P ,x, y))/(2*delta_d) # equation 3.1d, delta d
    df4_dT = (calculatef4(a, b, c, d, log10Kp_d_deltaTplus, P, x, y) - calculatef4(a, b, c, d, log10Kp_d_deltaTminus, P ,x, y))/(2*delta_T) # equation 3.1d, delta b

    # Calculate df5s
    df5_da = (calculatef5(a+delta_a, b, c, d, T, x, y, enthalpyArrayProducts, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict) - calculatef5(a-delta_a, b, c, d, T, x ,y, enthalpyArrayProducts, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict))/(2*delta_a) 
    df5_db = (calculatef5(a, b+delta_b, c, d, T, x, y, enthalpyArrayProducts, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict) - calculatef5(a, b-delta_b, c, d, T, x, y, enthalpyArrayProducts, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict))/(2*delta_b)
    df5_dc = (calculatef5(a, b, c+delta_c, d, T, x, y, enthalpyArrayProducts, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict) - calculatef5(a, b, c-delta_c, d, T, x, y, enthalpyArrayProducts, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict))/(2*delta_c)
    df5_dd = (calculatef5(a, b, c, d+delta_d, T, x ,y, enthalpyArrayProducts, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict) - calculatef5(a, b, c, d-delta_d, T, x, y, enthalpyArrayProducts, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict))/(2*delta_d)
    df5_dT = (calculatef5(a, b, c, d, T+delta_T, x ,y, enthalpyArrayProducts_deltaTplus, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict) - calculatef5(a, b, c, d, T-delta_T, x ,y, enthalpyArrayProducts_deltaTminus, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict))/(2*delta_T)

    # Formulate the Fprime Matrix
    F_prime = np.array([[df1_da, df1_db, df1_dc, df1_dd, df1_dT],
                        [df2_da, df2_db, df2_dc, df2_dd, df2_dT],
                        [df3_da, df3_db, df3_dc, df3_dd, df3_dT],
                        [df4_da, df4_db, df4_dc, df4_dd, df4_dT],
                        [df5_da, df5_db, df5_dc, df5_dd, df5_dT]])

    # Defnine the F matrix
    F = -1*np.array([[calculatef1(a, b, c, d, log10Kp_a, P , x, y)],
                     [calculatef2(a, b, c, d, log10Kp_b, P, x, y)],
                     [calculatef3(a, b, c, d, log10Kp_c, P ,x, y)],
                     [calculatef4(a, b, c, d, log10Kp_d, P, x, y)],
                     [calculatef5(a, b, c, d, T, x, y, enthalpyArrayProducts, mdot_O2, mdot_H2, nistO2FileName, nistH2FileName, thermo_curve_dict)]])
    
    # implement LU decomp to solve for B (F_prime*B = F)
    

    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate equilibrium constants for eq 3.1.a-d using partial pressure K_P
    T_array = np.arange(300,4000,10)
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Step 1 calculate enthalpy of required elements H2, O2, H2O, H, OH for temperature range T_array
    h_O2_test = calculateEnthalpyV2(T_array, thermo_curve_dict, "bO2", "rO2")
    h_H2_test = calculateEnthalpyV2(T_array,thermo_curve_dict,"bH2","rH2")
    h_H2O_test = calculateEnthalpyV2(T_array, thermo_curve_dict, "bH2O", "rH2O")
    h_H_test = calculateEnthalpyV2(T_array,thermo_curve_dict,"bH","rH")
    h_OH_test = calculateEnthalpyV2(T_array,thermo_curve_dict, "bOH", "rOH")
    h_O_test = calculateEnthalpyV2(T_array, thermo_curve_dict, "bO", "rO")

    # Step 2 calculate entropy of required elements H2, O2, H2O for temperature range T_array
    S_O2_test = calculateEntropyV2(T_array, thermo_curve_dict, "bO2", "rO2")
    S_H2_test = calculateEntropyV2(T_array, thermo_curve_dict, "bH2", "rH2")
    S_H2O_test = calculateEntropyV2(T_array, thermo_curve_dict, "bH2O", "rH2O")
    S_H_test = calculateEntropyV2(T_array, thermo_curve_dict, "bH","rH")
    S_OH_test = calculateEntropyV2(T_array,thermo_curve_dict, "bOH", "rOH")
    S_O_test = calculateEntropyV2(T_array,thermo_curve_dict,"bO", "rO")

    # Step 3 calculate Gibbs Energy for T_array
    G_O2_test = calculateGibbsEnergyV2(T_array, h_O2_test, S_O2_test)
    G_H2_test = calculateGibbsEnergyV2(T_array, h_H2_test, S_H2_test)
    G_H2O_test = calculateGibbsEnergyV2(T_array, h_H2O_test, S_H2O_test)
    G_H_test = calculateGibbsEnergyV2(T_array, h_H_test, S_H_test)
    G_OH_test = calculateGibbsEnergyV2(T_array,h_OH_test, S_OH_test)
    G_O_test = calculateGibbsEnergyV2(T_array, h_O_test, S_O_test)

    # Step 4 calculate change in gibbs energy for equation 3.1b
    deltaG_a_test = calculateDeltaG([2],[G_H2O_test], [2,1],[G_H2_test, G_O2_test])
    deltaG_b_test = calculateDeltaG([2], [G_H_test], [1], [G_H2_test]) #product coefficients, product gibbs, reactant coefficients, reactant gibbs
    deltaG_c_test = calculateDeltaG([2],[G_O_test],[1],[G_O2_test])
    deltaG_d_test = calculateDeltaG([1,2],[G_H2_test,G_OH_test],[2],[G_H2O_test])    

    # Step 5 calculate Partial pressure equilibrium constant for equation 3.1b
    log10Kp_a_test = calculatePartialPressureV2(T_array,deltaG_a_test)
    log10Kp_b_test = calculatePartialPressureV2(T_array,deltaG_b_test)
    log10Kp_c_test = calculatePartialPressureV2(T_array,deltaG_c_test)
    log10Kp_d_test = calculatePartialPressureV2(T_array,deltaG_d_test)


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

    plt.show()

    # Moving on in the document:
    # want to calculte enthalpy of formation and enthalpy magnitude for formation of H2O


    T_array = [298.15, 400, 500] # Temperatures of interest

    # formation of H2O => H2 + 1/2 O2 = H2O
    
    h_H2 = calculateEnthalpyV2(T_array, thermo_curve_dict, 'bH2', 'rH2')
    h_O2 = calculateEnthalpyV2(T_array, thermo_curve_dict, 'bO2', 'rO2')
    h_H2O = calculateEnthalpyV2(T_array, thermo_curve_dict, 'bH2O','rH2O')

    heatFormationH2O = h_H2O - 0.5*h_O2 - h_H2
    enthalpyMagH2O = heatFormationH2O[0]+h_H2O[2] - h_H2O[0]

    # Display
    for i in range(len(T_array)):
        print("Temp [k]: {0} Enthalpy [J/mol]: {1} heatFormationH2O [J/mol]: {2}\n".format(T_array[i], h_H2O[i], heatFormationH2O[i]))

    print(enthalpyMagH2O)
main()