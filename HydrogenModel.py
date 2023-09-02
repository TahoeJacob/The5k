# Hydrogen Phase model for low temperature enthalpy calculations
# By: Jacob Saunders
# Date: 26/08/2023
# Based on data from https://www.cryo-rocket.com/about/equation-of-state/2.2-hydrogen-model/#ref2Newton2 and relevant references from this document


# Background: 
# In order to simulate enthalpy of H2 at low temperatures (liquid) a real equation of state is required
# we will use the Reduced Helmholtz correlation equation of state which uses partial pressure, partial density, and partial temperature
# The reduced Helmholtz equation can be outlined in section 2 Equation of State

#imports
import numpy as np
import csv
import matplotlib.pyplot as plt


# Define global constants
critical_density = 31.112 #[kg/m^3]
criitcal_temperature = 32.938 #[K]
critical_pressure = 1.284 #[MPa] Note 1MPa = 10Bar

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

    
# Takin CSV of parahydrogen percentages at varying temperatures from 10K to 400K
def paraPercentFunction():
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # IMPORT TEMPERTAURE TO PARA PERCENTAGE FILE
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    dataParaPercent = extractFileCSV('HydrogenOrthoPara.csv')

    paraPercentage = np.array(dataParaPercent[1::], dtype=float) # creates array of Temp and corresponding para percentages
    paraTemp = []
    paraPercent = []
   
    for value in paraPercentage:
        paraTemp.append(value[0])
        paraPercent.append(value[1])

    fig1 = plt.figure(1)
    
    ax = fig1.add_subplot(1,1,1)

    ax.plot(paraTemp, paraPercent)


    ax.set_xlim([0,400])
    ax.set_ylim([0,120])
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel('Parahydrogen Percentage')
    ax.set_title('Hydrogen Composition')
    ax.grid()

    #plt.show()

    return paraTemp,paraPercent

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

# calculate the first derrivate w.r.t delta of the relative component of helmholtz equation
def alphaR_delta(tau, delta, coeff):
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
    sumOne = [] # first sum is from i = 1  to 7 so need a total of 7 zeros pre-loaded
    sumTwo = [] # second sum is from i=8 to 9 so only need total of 3 zeros pre-loaded
    sumThree = []  # third sum is from i=10 to 14 so only need a total of 4 zeros pre-loaded

    # First derrivative of the relative helmholtz equation w.r.t delta

    for i in range(14):
        if i < 7:
            sumOne.append( coeff['N_i'][i] * coeff['d_i'][i] * (delta**(coeff['d_i'][i]-1)) * (tau**(coeff['t_i'][i])) )
            
        elif i >= 7 and i < 9:
            sumTwo.append(coeff['N_i'][i] * np.exp(-delta**(coeff['p_i'][i]))* (tau**(coeff['t_i'][i])) * (-coeff['p_i'][i]*(delta**(coeff['p_i'][i]-1)) + coeff['d_i'][i]*delta**(coeff['d_i'][i]-1)))
            
        elif i >= 9 :
            sumThree.append(coeff['N_i'][i] * (tau**(coeff['t_i'][i])) * np.exp(coeff['phi_i'][i]*((delta-coeff['D_i'][i])**2) + coeff['Beta_i'][i]*((tau-coeff['gamma_i'][i])**2)) * (2*coeff['phi_i'][i]*(delta-coeff['D_i'][i])+coeff['d_i'][i]*delta**(coeff['d_i'][i]-1)) )

    dr_by_ddelta =    sum(sumOne) + sum(sumTwo) + sum(sumThree)

    return dr_by_ddelta


# Function which takes in density and temperature and calculates pressure based off reisdual components of the helmholtz equation
def calcPressure(temp, density, gasConstant, coeff):
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

    # Known variables
    tau = temp/criitcal_temperature
    delta = density/critical_density

    # calculate first derrivative of residual component w.r.t delta
    dalpha_by_ddelta = alphaR_delta(tau, delta,coeff)

    pressure = density*gasConstant*temp*(1+delta*dalpha_by_ddelta)

    return pressure
def main():
    # Function which runs all required models and wraps up information neatly
    #--------------------------------------------------------------------------------------
    # Inputs: name - type - desc - units
    # None
    #--------------------------------------------------------------------------------------
    #Outputs: name - type - desc - units
    # None
    #--------------------------------------------------------------------------------------

    # Create a dictionary of Helmholtz coeefficients 
    para_coeff = {
        'a_k':[-1.4485891134, 1.884521239, 4.30256, 13.0289,-47.7365, 50.0013, -18.6261, 0.993973, 0.536078],
        'b_k':[0,0,-15.1496751472, -25.0925982148,-29.4735563787,-35.405914117,-40.724998482,-163.7925799988,-309.2173173842],
        'N_i':[-7.33375,0.01,2.60375,4.66279,0.682390,-1.47078,0.135801,-1.05327,0.328239,-0.0577833,0.0449743,0.0703464,-0.0401766,0.119510],
        't_i':[0.6855,1,1,0.489,0.774,1.133,1.386,1.619,1.162,3.96,5.276,0.99,6.791,3.19],
        'd_i':[1,4,1,1,2,2,3,1,3,2,1,3,1,1],
        'p_i':[0,0,0,0,0,0,0,1,1,0,0,0,0,0],
        'phi_i':[0,0,0,0,0,0,0,0,0,-1.7437,-0.5516,-0.0634,-2.1341,-1.777],
        'Beta_i':[0,0,0,0,0,0,0,0,0,-0.194,-0.2019,-0.0301,-0.2383,-0.3253],
        'gamma_i':[0,0,0,0,0,0,0,0,0,0.8048,1.5248,0.6648,0.6832,1.493],
        'D_i':[0,0,0,0,0,0,0,0,0,1.5487,0.1785,1.28,0.6319,1.7104]
    }

    ortho_coeff = {
        'a_k':[-1.4675442336,1.8845068862,2.54151,-2.3661,1.00365,1.22447,0,0,0],
        'b_k':[0,0,-25.7676098736,-43.4677904877,-66.0445514750,-209.7531607465,0,0,0],
        'N_i':[-6.83148,0.01,2.11505,4.38353,0.211292,-1.00939,0.142086,-0.87696,0.804927,-0.710775,0.0639688,0.0710858,-0.087654,0.647088],
        't_i':[0.7333,1,1.1372,0.5136,0.5638,1.6248,1.829,2.404,2.105,4.1,7.658,1.259,7.589,3.946],
        'd_i':[1,4,1,1,2,2,3,1,3,2,1,3,1,1],
        'p_i':[0,0,0,0,0,0,0,1,1,0,0,0,0,0],
        'phi_i':[0,0,0,0,0,0,0,0,0,-1.169,-0.894,-0.04,-2.072,-1.306],
        'Beta_i':[0,0,0,0,0,0,0,0,0,-0.4555,-0.4046,-0.0869,-0.4415,-0.5743],
        'gamma_i':[0,0,0,0,0,0,0,0,0,1.5444,0.6627,0.763,0.6587,1.4327],
        'D_i':[0,0,0,0,0,0,0,0,0,0.6366,0.3876,0.9437,0.3976,0.9626]
    }

    # Gas Constants
    hydrogenGasConstant = 8.3144626  #4124.2 # [J/kgK] https://www.engineeringtoolbox.com/individual-universal-gas-constant-d_588.html

    paraTemp, paraPercent = paraPercentFunction()
 
    # OKAY steps going forward, need to ensure the FDS actually works when calculating density, goal for today is to match figure 2.2.3 

    # Get Temperature and density data from NIST for hydrogen at 2x critical pressure 
    targetPressure = 2*critical_pressure
    tempHigh = 6*criitcal_temperature
    print("Target is to calculate reduced density of H2 at {0:.3} MPa which is {1:.3f} reduced pressure, and over a range of {2:.3f} to {3:.3f} reduced temperature".format(targetPressure,targetPressure/critical_pressure,13.957, 6*criitcal_temperature))


    tempArrayCp2, pressureArrayCp2, densityArrayCp2 = extractNISTData('2Pc.csv')
    
    # test1 find reduced density at 2x critical pressure and 1.7 critical temp

    # target temp

    targetReducedTemp = 1.7
    targetTemp = targetReducedTemp*criitcal_temperature

    # find the closest value in NIST temp array takes in array and target value
    closestTempNIST = min(tempArrayCp2, key=lambda x: abs(targetTemp - x))
    initialGuessIndex = tempArrayCp2.index(closestTempNIST)


    # determine ortho para percentages based off temp
    closestTemp = min(paraTemp,key=lambda x: abs(targetTemp-x))
    paraIndex = paraTemp.index(closestTemp)
    percentagePara = (paraPercent[paraIndex]/100)
    percentageOrtho = 1-percentagePara
    

    # we have NIST data for pressure of interest which in this test case is 2x the critical pressure
    pressureOfIntrest = 2*critical_pressure

    #initial density guess based off NIST data
    densityGuess = densityArrayCp2[initialGuessIndex]
    densityGuessPara = densityArrayCp2[initialGuessIndex]
    densityGuessOrtho = densityArrayCp2[initialGuessIndex]

    # Do i iterations of the finite difference scheme to calculate the predicted density at said temp and pressure

    # Density step size (delta Density)
    deltaDensityPara = densityGuess/1000
    deltaDensityOrtho = densityGuess/1000
    for i in range(2):
        #------------------------- Para Density Calculation------------------------------------------------------------
        pressurePlusPara = calcPressure(targetTemp,densityGuessPara+deltaDensityPara, hydrogenGasConstant, para_coeff)
        pressureMinusPara = calcPressure(targetTemp,densityGuessPara-deltaDensityPara, hydrogenGasConstant, para_coeff)

        drho_by_dP_Para = (pressurePlusPara-pressureMinusPara)/(2*deltaDensityPara)

        PGuessPara = calcPressure(targetTemp, densityGuessPara, hydrogenGasConstant, para_coeff)

        densityGuessPara = densityGuessPara+((PGuessPara - pressureOfIntrest)/drho_by_dP_Para)

        
        deltaDensityPara = densityGuessPara/1000

        #------------------------- Ortho Density Calculation------------------------------------------------------------

        pressurePlusOrtho = calcPressure(targetTemp,densityGuessOrtho+deltaDensityOrtho, hydrogenGasConstant, ortho_coeff)
        pressureMinusOrtho = calcPressure(targetTemp,densityGuessOrtho-deltaDensityOrtho, hydrogenGasConstant, ortho_coeff)

        drho_by_dP_Ortho = (pressurePlusOrtho-pressureMinusOrtho)/(2*deltaDensityOrtho)

        PGuessOrtho = calcPressure(targetTemp, densityGuessOrtho, hydrogenGasConstant, ortho_coeff)

        densityGuessOrtho = densityGuessOrtho+((PGuessOrtho - pressureOfIntrest)/drho_by_dP_Ortho)

        
        deltaDensityOrtho = densityGuessOrtho/1000
        
        
        #------------------------- Density Calculation------------------------------------------------------------------
        densityGuessPrev = densityGuess
        densityGuess = (percentageOrtho*densityGuessOrtho) + (percentagePara*densityGuessPara)

        deltaGuess = densityGuess-densityGuessPrev
        print("Para Density: {0:.5f}\n Ortho Density: {1:.5f}\n Total Density: {2:.5f}\n Change In Density {3:.5f}".format(densityGuessPara, densityGuessOrtho, densityGuess, deltaGuess))


    paraN = 9 # number of steps for k if parahydrogen
    orthoN = 6 # number of steps for k if orthohydrogen

    N = np.arange(3,paraN,1) # Array of values for sum of 3 to N where N is dependant on if its para or orth
    

    return None


main()




    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Calculate the ideal componenet of the Helmholtz correlation
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # idealSum = np.zeros(len(N)) # Sum of k=3 to N value

    # for k in N:
    #     if len(N) == paraN-3:
    #         idealSum[k] = idealSum[k] + para_coeff['a_k'][k]*np.log[1-np.exp(para_coeff['b_k'][k]*tau)]
    #     else:
    #         idealSum[k] = idealSum[k]
    # ideal_component = np.log(delta) + 1.5*np.log(tau) + para_coeff['a_k'][0] + para_coeff['a_k'][1]*tau + idealSum
