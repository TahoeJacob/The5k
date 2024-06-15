# Oxygen Phase model for low temperature enthalpy calculations
# By: Jacob Saunders
# Date: 26/08/2023
# Based on data from https://www.cryo-rocket.com/about/equation-of-state/2.2-hydrogen-model/#ref2Newton2 and relevant references from this document


# Background: 
# In order to simulate enthalpy of O2 at low temperatures (liquid) a real equation of state is required
# we will use the Reduced Helmholtz correlation equation of state which uses partial pressure, partial density, and partial temperature
# The reduced Helmholtz equation can be outlined in section 2 Equation of State

#imports
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# define global constants
critDensity = 436.1 # Critical Density [kg/m^3]
critTemp = 154.6 # Critical Temperature [K]
critPressure = 5 # Critical pressure [MPa]
univeralGasConstant = 8.31446261815324 # Universal gas constant [J/Kmol]

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
    #----------------------------------------------------------------------------------------------------------------------------------------
    dataNIST= extractFileCSV(filename)
     
    tempArray = []
    pressureArray = []
    denistyArray = []
    for value in dataNIST[1::]:
        tempArray.append(float(value[0]))
        pressureArray.append(float(value[1]))
        denistyArray.append(float(value[2]))

    return tempArray,pressureArray,denistyArray

def calcPressure(temp, density, coeff):
    # Function which takes in temperature [K] and density [kg/m^3] and retruns pressure [MPa]
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # temp - float - temperature value - [K]
    # density - float - density value - [kg/m^3]
    # gasConstnat - float - specific gas constant - [J/kgK]
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # pressure - float - pressure result- [MPa]
    #----------------------------------------------------------------------------------------------------------------------------------------

    delta = density/critDensity
    tau = temp/critTemp
    alpha = calcAlphaR_delta(tau,delta,coeff)

    pressure = temp*density*univeralGasConstant*(1+delta*alpha)

    return pressure

#----------------------------------------------------------------------------------------------------------------------------------------
# CALCULATE RESIDUAL PARTIAL DERRIVATIVES
#----------------------------------------------------------------------------------------------------------------------------------------

# Calculate the first partial derrivative w.r.t to delta
def calcAlphaR_delta(tau, delta, coeff):
    # Function which takes in coefficient dictionary and calculates the first derrivate of alpha R w.r.t delta
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # tau - float - reduced temperature - unitless
    # delta - float - reduced density - unitless 
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # alphaR_delta - float -  result - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------

    # Calculate dr_byddelata
    alphaSum = 0

    # First derrivative of the relative helmholtz equation w.r.t delta
    for i in range(32):
        if i <=12:
            alphaSum = alphaSum + (coeff['n'][i]*coeff['r'][i]*(delta**(coeff['r'][i]-1))*(tau**(coeff['s'][i])))
            #print("i is {3} \n ----------\nNi: {0} \n Si {1} \n Ri {2}".format(coeff['n'][i],coeff['s'][i],coeff['r'][i], i+1))
        if i >= 13 and i <=23:
            alphaSum = alphaSum + ( np.exp(-delta**2)*(coeff['n'][i]*(coeff['r'][i]*delta**(coeff['r'][i]-1) - 2*delta**(coeff['r'][i]+1))*tau**(coeff['s'][i])))
            #print("i is {3} \n ----------\nNi: {0} \n Si {1} \n Ri {2}".format(coeff['n'][i],coeff['s'][i],coeff['r'][i], i+1))
        if i >= 24 and i <=32:
            alphaSum = alphaSum + ( np.exp(-delta**4)*(coeff['n'][i]*(coeff['r'][i]*delta**(coeff['r'][i]-1) - 4*delta**(coeff['r'][i]+3))*tau**(coeff['s'][i])))
            #print("i is {3} \n ----------\nNi: {0} \n Si {1} \n Ri {2}".format(coeff['n'][i],coeff['s'][i],coeff['r'][i], i+1))
    return alphaSum

# Calculate the second partial derrivative w.r.t to delta and delta
def calcAlphaR_delta_delta(tau, delta, coeff):
    # Calculate the second partial derrivative of the residual component of Helmholtz equation w.r.t delta and tau
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # tau - float - reduced temperature - unitless
    # delta - float - reduced density - unitless 
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # alphaR_delta_delta - float -ideal component first derrivate w.r.t tau - [J/kg]
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate 
    alphaSum = 0

    # Second partial derrivative of the relative helmholtz equation w.r.t delta & delta
    for i in range(32):
        if i <=12:
            alphaSum = alphaSum + (coeff['n'][i]*coeff['r'][i]*(coeff['r'][i]-1)*delta**(coeff['r'][i]-2) * tau**(coeff['s'][i]))
        if i >= 13 and i <=23:
            alphaSum = alphaSum + np.exp(-delta**2)*(coeff['n'][i]*(coeff['r'][i]*(coeff['r'][i]-1)*delta**(coeff['r'][i]-2) - 2*(2*coeff['r'][i]+1)*delta**(coeff['r'][i]) + 4*delta**(coeff['r'][i]+2))*tau**(coeff['s'][i]))
        if i >= 24 and i <=32:
            alphaSum = alphaSum + np.exp(-delta**4)*(coeff['n'][i]*(coeff['r'][i]*(coeff['r'][i]-1)*delta**(coeff['r'][i]-2) - 4*(2*coeff['r'][i] + 3)*delta**(coeff['r'][i]+2) + 16*delta**(coeff['r'][i]+6))*tau**(coeff['s'][i]))

    #alphaR_delta_delta = sum(sumOne) + np.exp(-delta**2) * sum(sumTwo) + np.exp(-delta**4) * sum(sumThree)

    return alphaSum


def calcAlphaR_tau(tau, delta, coeff):
    # Calculate the first partial derrivative of the residual component of Helmholtz equation w.r.t tau
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # tau - float - reduced temperature - unitless
    # delta - float - reduced density - unitless 
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # alphaSum - float -ideal component first derrivate w.r.t tau - [J/kg]
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate  
    alphaSum = 0

    # First derrivative of the relative helmholtz equation w.r.t delta
    for i in range(32):
        if i <=12:
            alphaSum = alphaSum + ( (coeff['n'][i]) * (coeff['s'][i]) * (delta**(coeff['r'][i])) * (tau**( (coeff['s'][i]) - 1 ) ))
        if i >= 13 and i <=23:
            alphaSum = alphaSum + np.exp(-delta**(2)) * ( (coeff['n'][i]) * (coeff['s'][i]) * (delta**(coeff['r'][i])) * (tau**(coeff['s'][i]-1))) 
        if i >= 24 and i <=32:
            alphaSum = alphaSum +  np.exp(-delta**4) * (coeff['n'][i]*coeff['s'][i]*delta**(coeff['r'][i])*tau**(coeff['s'][i]-1))

    return alphaSum

# Calculate the second partial derrivative w.r.t tau and tau
def calcAlphaR_tau_tau(tau, delta, coeff):
    # Calculate the second partial derrivative of the residual component of Helmholtz equation w.r.t tau and tau
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # tau - float - reduced temperature - unitless
    # delta - float - reduced density - unitless 
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # alphaR_tau_tau - float -ideal component first derrivate w.r.t tau - [J/kg]
    #----------------------------------------------------------------------------------------------------------------------------------------

    # Calculate 
    alphaSum = 0

    # First derrivative of the relative helmholtz equation w.r.t delta
    for i in range(32):
        if i <=12:
            alphaSum = alphaSum + ( (coeff['n'][i]) * (coeff['s'][i]) * ( (coeff['s'][i]) - 1 ) * (delta**(coeff['r'][i])) * (tau**( (coeff['s'][i]) - 2 ) )) 
        if i >= 13 and i <=23:
            alphaSum = alphaSum + np.exp(-delta**(2)) * ( (coeff['n'][i]) * (coeff['s'][i]) * ( (coeff['s'][i]) - 1) * delta**(coeff['r'][i]) * (tau**(coeff['s'][i]-2))) 
        if i >= 24 and i <=32:
            alphaSum = alphaSum +  np.exp(-delta**4) * (coeff['n'][i]*coeff['s'][i]*(coeff['s'][i]-1)*delta**(coeff['r'][i])*tau**(coeff['s'][i]-2))


    return alphaSum

# Calculate the second partial derrivative w.r.t delta and tau
def calcAlphaR_delta_tau(tau, delta, coeff):
    # Calculate the second partial derrivative of the residual component of Helmholtz equation w.r.t delta and tau
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # tau - float - reduced temperature - unitless
    # delta - float - reduced density - unitless 
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # alphaR_delta_tau - float -ideal component first derrivate w.r.t tau - [J/kg]
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate 
    alphaSum = 0

    # First derrivative of the relative helmholtz equation w.r.t delta
    for i in range(32):
        if i <=12:
            alphaSum = alphaSum + (coeff['n'][i]*coeff['r'][i]*coeff['s'][i]*delta**(coeff['r'][i]-1)*tau**(coeff['s'][i]-1))
        if i >= 13 and i <=23:
            alphaSum = alphaSum + np.exp(-delta**2)*(coeff['n'][i]*(coeff['r'][i]*delta**(coeff['r'][i]-1) - 2*delta**(coeff['r'][i]+1))*coeff['s'][i]*tau**(coeff['r'][i]-1))
        if i >= 24 and i <=32:
            alphaSum = alphaSum + np.exp(-delta**4)*(coeff['n'][i]*(coeff['r'][i]*delta**(coeff['r'][i]-1) - 4*delta**(coeff['r'][i]+3))*coeff['s'][i]*tau**(coeff['s'][i]-1))

    return alphaSum


#----------------------------------------------------------------------------------------------------------------------------------------
# CALCULATE IDEAL PARTIAL DERRIVATIVES
#----------------------------------------------------------------------------------------------------------------------------------------

# Calculate the first partial derrivative w.r.t tau
def calcAlphaIdeal_tau(tau, coeff):
    # Calculate the first derrivative of the ideal component of Helmholtz equation w.r.t tau
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # tau - float - reduced temperature - unitless
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # alphaIdeal_Tau - float -ideal component first derrivate w.r.t tau - [J/kg]
    #----------------------------------------------------------------------------------------------------------------------------------------

    alphaIdeal_Tau = 1.5*coeff['k'][0]*tau**(0.5) - 2*coeff['k'][1]*tau**(-3) + coeff['k'][2]*tau**(-1) + coeff['k'][3] + coeff['k'][4]*((coeff['k'][6]*np.exp(coeff['k'][6]*tau))/(np.exp(coeff['k'][6]*tau)-1)) - coeff['k'][5]*((0.66*coeff['k'][7]*np.exp(-coeff['k'][7]*tau))/(1+0.66*np.exp(-coeff['k'][7]*tau)))

    return alphaIdeal_Tau

# Calculate the second partial derrivative w.r.t tau and tau
def calcAlphaIdeal_tau_tau(tau,coeff):
    # Calculate the second partial derrivative of the ideal component of Helmholtz equation w.r.t tau and tau
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # tau - float - reduced temperature - unitless
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # alphaIdeal_Tau_Tau - float -ideal component first derrivate w.r.t tau - [J/kg]
    #----------------------------------------------------------------------------------------------------------------------------------------

    alphaIdeal_Tau_Tau = 0.75*coeff['k'][0]*tau**(-0.5) + 6*coeff['k'][1]*tau**(-4) - coeff['k'][2]*tau**(-2) - coeff['k'][4]*(((coeff['k'][6]**(2))*np.exp(coeff['k'][6]*tau))/((np.exp(coeff['k'][6]*tau)-1)**2)) + coeff['k'][5]*((0.66*(coeff['k'][7]**2)*np.exp(-coeff['k'][7]*tau))/((1+0.66*np.exp(-coeff['k'][7]*tau))**2))

    return alphaIdeal_Tau_Tau

#----------------------------------------------------------------------------------------------------------------------------------------
# CALCULATE ISOCHORIC HEAT CAPACITY [J/kg K]
#----------------------------------------------------------------------------------------------------------------------------------------
def calcIsochoricHeatCap(tau, delta, coeff):
    # Function which takes in target reduced temperature array, NIST temperature array [K], NIST density array [kg/m^3], and coefficient dict to calculate Isochoric Heat Capicity [J/kg K]
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # targetTempArray -  array - array of desired reduced temperature values - [unitless]
    # NISTtempArray - array - array of temperature values from the NIST database - [K]
    # NISTdensityArray - array - specific gas constant - [kg/m^3]
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # isochoricHeatCapArray - float - array of predicted isochoric heat capcities based off known temperature and pressures - [J/kgK]
    #----------------------------------------------------------------------------------------------------------------------------------------

    # for targetTemp in targetTempArray:

    #     # find the index of closest temperature and density from NIST data
    #     closestTempNIST = min(NISTtempArray, key=lambda x: abs((targetTemp*critTemp) - x))
    #     initialGuessIndex = NISTtempArray.index(closestTempNIST)
    #     tempNIST = NISTtempArray[initialGuessIndex]
    #     densityNIST = NISTdensityArray[initialGuessIndex] 

    #     delta = densityNIST/critDensity
    #     tau = tempNIST/critTemp

    # Calculate the second partial derrivative of the ideal component w.r.t to tau and tau
    alphaIdeal_tau_tau = calcAlphaIdeal_tau_tau(tau, coeff)

    # Calculate the second partial derrivative of the residual component w.r.t tau and tau
    alphaR_tau_tau = calcAlphaR_tau_tau(tau, delta, coeff)

    # we now have to calculate the isochoric heat capacity
    isochoricHeatCap = (-tau**(2)*(alphaIdeal_tau_tau + alphaR_tau_tau))

    return isochoricHeatCap


#----------------------------------------------------------------------------------------------------------------------------------------
# CALCULATE ISOBARIC HEAT CAPACITY [J/kg K]
#----------------------------------------------------------------------------------------------------------------------------------------
def calcIsobaricHeatCap(targetTempArray, NISTtempArray, NISTdensityArray, coeff):
    # Function which takes in target reduced temperature array, NIST temperature array [K], NIST density array [kg/m^3], and coefficient dict to calculate Isobaric Heat Capicity [J/kg K]
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # targetTempArray -  array - array of desired reduced temperature values - [unitless]
    # NISTtempArray - array - array of temperature values from the NIST database - [K]
    # NISTdensityArray - array - specific gas constant - [kg/m^3]
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # isobaricHeatCapArray - array - array of predicted isochoric heat capcities based off known temperature and pressures - [J/kgK]
    #----------------------------------------------------------------------------------------------------------------------------------------


    isobaricHeatCapArray = []

    for targetTemp in targetTempArray:

        # find the index of closest temperature and density from NIST data
        closestTempNIST = min(NISTtempArray, key=lambda x: abs((targetTemp*critTemp) - x))
        initialGuessIndex = NISTtempArray.index(closestTempNIST)
        tempNIST = NISTtempArray[initialGuessIndex]
        densityNIST = NISTdensityArray[initialGuessIndex] 

        delta = densityNIST/critDensity
        tau = tempNIST/critTemp

        # Calculate isochoric heat capacity for the selected temperature and density
        isochoricHeatCap = calcIsochoricHeatCap(tau, delta, coeff)

        # Calculate residual first partial derrivative w.r.t delta
        alphaR_delta = calcAlphaR_delta(tau, delta, coeff)

        # Calculate residual second partial derrivative w.r.t delta and tau
        alphaR_delta_tau = calcAlphaR_delta_tau(tau, delta, coeff)

        #Calculate residual second partial derrivative w.r.t delta and delta
        alphaR_delta_delta = calcAlphaR_delta_delta(tau, delta, coeff)


        isobaricHeatCapArray.append(isochoricHeatCap + (((1 + delta*alphaR_delta - delta*tau*alphaR_delta_tau)**2)/(1+2*delta*alphaR_delta + delta**(2)*alphaR_delta_delta)) )

    return isobaricHeatCapArray

#----------------------------------------------------------------------------------------------------------------------------------------
# CALCULATE DENSITY OF OXYGEN BASED OFF GIVEN TEMPERATURE AND PRESSURE (UN-USED FUNCTION JUST USING NIST DATA)
#----------------------------------------------------------------------------------------------------------------------------------------
def calcPredictedResidualDensity(targetTempArray, NISTtempArray, NISTdensityArray, coeff):
    # Function which takes in target reduced temperature array, NIST temperature array [K], NIST density array [kg/m^3], and coefficient dict
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Inputs: 
    # targetTempArray -  array - array of desired reduced temperature values - [unitless]
    # NISTtempArray - array - array of temperature values from the NIST database - [K]
    # NISTdensityArray - array - specific gas constant - [kg/m^3]
    # coeff - dictionary - dictionary of coefficients for helmholtz equation - N/A
    #----------------------------------------------------------------------------------------------------------------------------------------
    #Outputs
    # predictedDensityArray - array - array of predicted reduced densities based off known temperature and pressures - [unitless]
    #----------------------------------------------------------------------------------------------------------------------------------------

    predictedDensityArray = []

    # Calculate the predicted density at each temperature for set pressures.
    for targetTemp in targetTempArray:
        # find nearrest temp from NIST data that aligns with target temp
        closestTempNIST = min(NISTtempArray, key=lambda x: abs((targetTemp*critTemp) - x))
        initialGuessIndex = NISTtempArray.index(closestTempNIST)

        # now we have Ti, Pi, and index to get pi can start newtonian calc
        densityGuess = NISTdensityArray[initialGuessIndex]
        deltaDensity = densityGuess/1000

        changeInDensity = 1

        while changeInDensity > 0.01:
            # calculate pressure at 1 deltaPressure deviation above
            pressurePlus = calcPressure(closestTempNIST,densityGuess + deltaDensity, coeff)
            
            # calculate pressure at 1 deltaPressure deviation below
            pressureMinus = calcPressure(closestTempNIST,densityGuess - deltaDensity, coeff)

            # calculate partial derrivative of density w.r.t pressure
            dp_dP = (pressurePlus-pressureMinus)/(2*deltaDensity)

            # Calculate the pressure with the currenty desnity guess
            # PressureGuess = calcPressure(closestTempNIST, densityGuess, coeff)

            # Record set the current density to the previous density as we are going to re-calculate in next step
            prevDensityGuess = densityGuess

            PressureGuess = np.exp(5.428*((closestTempNIST/critTemp)-1))*critPressure
            
            # Calculate the next density guess
            #densityGuess = densityGuess - (((PressureGuess-targetPressure) * 2*deltaDensity)+targetPressure)/(pressurePlus-pressureMinus)
            densityGuess = densityGuess - (PressureGuess)/dp_dP
            
            # Re-calculate deltaDensity based off new densityGuess
            deltaDensity = densityGuess/1000

            # Calculate the difference between new density and previous density. (should decrease everytime)
            changeInDensity = abs(densityGuess- prevDensityGuess) 

        # Display currenty density guess:
        print("Density: {0:.3f}   Reduce Density: {1:0.3f}  Delta Density{2} Target Temp {3}".format(densityGuess, densityGuess/critDensity, changeInDensity, targetTemp))  
        #print(densityGuess-densityArray[initialGuessIndex]) 
        predictedDensityArray.append(densityGuess/critDensity)

    return predictedDensityArray

#----------------------------------------------------------------------------------------------------------------------------------------
# CALCULATE ENTHALPY 
#----------------------------------------------------------------------------------------------------------------------------------------
def calcEnthalpy (targetTemp, tau, delta, coeff):


    alphaIdeal_tau = calcAlphaIdeal_tau(tau, coeff)

    alphaR_tau = calcAlphaR_tau(tau, delta, coeff)

    alphaR_delta = calcAlphaR_delta(tau, delta, coeff)

    enthalpy = (1 + tau*(alphaIdeal_tau + alphaR_tau) + delta * alphaR_delta) * univeralGasConstant * (targetTemp*critTemp)

    return enthalpy

#----------------------------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION
#----------------------------------------------------------------------------------------------------------------------------------------
def main():
    # Main function self explainatory
    coeff = {
        'r':[1,	1,	1,	2,	2,	2,	3,	3,	3,	6,	7,	7,	8,	1,	1,	2,	2,	3,	3,	5,	6,	7,	8,	10,	2,	3,	3,	4,	4,	5,	5,	5],
        's':[0,	1.5,	2.5,	-0.5,	1.5,	2,	0,	1,	2.5,	0,	2,	5,	2,	5,	6,	3.5,	5.5,	3,	7,	6,	8.5,	4,	6.5,	5.5, 22,	11,	18,	11,	23,	17,	18,	23],
        'n':[0.3983768749,	-0.1846157454E-1,	0.4183473197,	0.2370620711E-1,	0.9771730573E-1,	0.3017891294E-1,	0.2273353212E-1,	0.1357254086E-1,	- 0.4052698943E-1,	0.5454628515E-3,	0.5113182277E-3,	0.2953466883E-6,	-0.8687645072E-4,	-0.2127082589,	0.8735941958E-1 ,	0.1275509190,	-0.9067701064E-1,	-0.3540084206E-1,	-0.3623278059E-1,	0.1327699290E-1,	-0.3254111865E-3, 0.8313582932E-2,	0.2124570559E-2, -0.8325206232E-3,	-0.2626173276E-4,	0.2599581482E-2, 0.9984649663E-2,	0.2199923153E-2,	-0.2591350486E-1,	-0.1259630848,	0.1478355637,	-0.1011251078E-1],
        'k':[-0.740775E-3,	-0.664930E-4,	0.250042E1,	-0.214487E2,	0.101258E1,	- 0.944365,	0.145066E2,	0.749148E2 ,	0.414817E1]
    }

    # Create a figure
    fig, ax = plt.subplots()
    # Range of desired temperature ranges
    targetTempArray = np.arange(0.5,6,0.1)

    # go through each NIST array and plot the calculated densities 
    NISTData = ['1.25PcOx.csv']#,'1.25PcOx.csv', '0.75PcOx.csv']
   
    for fileRef in NISTData:
        # NIST temperature, pressure, density arrays
        NISTtempArray, NISTpressureArray, NISTdensityArray = extractNISTData(fileRef)

        #Extract file name
        fileName = Path(fileRef).stem

        # Calculate the predicted density at each temperature for set pressures.
        #predictedDensityArray = calcPredictedResidualDensity(targetTempArray, NISTtempArray, NISTdensityArray, coeff)  

        # Calculate the isobaric heat capacity
        isobaricHeatCapacityArray = calcIsobaricHeatCap(targetTempArray, NISTtempArray, NISTdensityArray, coeff)

        #target reduced temp
        targetTemp = 3

        closestTempNIST = min(NISTtempArray, key=lambda x: abs((targetTemp*critTemp) - x))
        initialGuessIndex = NISTtempArray.index(closestTempNIST)
        tempNIST = NISTtempArray[initialGuessIndex]
        densityNIST = NISTdensityArray[initialGuessIndex] 

        delta = densityNIST/critDensity
        tau = tempNIST/critTemp

        # Calculate the isochoric heat capacity
        isochoricHeatCap = calcIsochoricHeatCap(tau,delta,coeff)

        # Diagnose error
        alpha_tau_tau = calcAlphaIdeal_tau_tau(tau,coeff)
        #isochoricHeatCap = calcIsochoricHeatCap(tau, delta, coeff)
        print("tau: {0:.3f} delta: {1:.3f} alpha_tau_tau {2:.3f}".format(tau, delta, alpha_tau_tau))


        alphaR_tau_tau = calcAlphaR_tau_tau(tau,delta,coeff)
        alphaIdeal_tau_tau = calcAlphaIdeal_tau_tau(tau,coeff)
        alphaR_delta = calcAlphaR_delta(tau, delta, coeff)
        alphaR_delta_delta = calcAlphaR_delta_delta(tau, delta, coeff)
        alphaR_delta_tau = calcAlphaR_delta_tau(tau, delta, coeff)

        index = 28

        enthalpy = calcEnthalpy(targetTemp, tau, delta, coeff)
        print(enthalpy)

        #print("n[{4}]:{0} s[{4}]:{1} r[{4}]:{2} alphaIdeal_tau_tau: {3} ".format(coeff['n'][index],coeff['s'][index],coeff['r'][index],alphaR_tau_tau, index))
        #$print(isochoricHeatCap)

        #print("alphaR_delta {0} \n alphaR_delta_delta {1} \n alphaR_delta_tau {2} \n alphaR_tau_tau: {3} \n alphaIdeal_tau_tau {4}".format(alphaR_delta, alphaR_delta_delta, alphaR_delta_tau, alphaR_tau_tau, alphaIdeal_tau_tau))
        

        # # Plot results
        tempArrayReduced = [x/critTemp for x in NISTtempArray]
        densityArrayReduced = [x/critDensity for x in NISTdensityArray]

        # Plot NIST Reduced Temp vs Reduced Density
        plt.figure(2)
        plt.plot(tempArrayReduced, densityArrayReduced,linewidth=2.0, label = "P={0}NIST".format(fileName))
        plt.xlabel("Reduced Temperature")
        plt.ylabel("Reduced Density")
        plt.title("Reduced Density vs Reduced Temperature")
        plt.grid()

        #  Plot Reduced Temp vs Isochoric Heat Capacity
        ax.plot(targetTempArray, isobaricHeatCapacityArray,linewidth=2.0, label = "P={0}NIST".format(fileName))

        
    ax.legend()
    #ax2.legend()

    # Plot reduced temp vs reduced density
    
    # Plot reduced temp vs heat capacity
    ax.legend
    ax.set_xlabel("Reduced Temperature")
    ax.set_ylabel("Isobaric Heat Capacity")
    ax.set_title("Reduced Temperature vs Isobaric Heat Capacity")
    ax.grid()
    #plt.show()
    
    return None

main()


