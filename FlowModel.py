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
import math
plt.close('all')

def function5_5_1(MachNum_Exit, areaRatio, specificHeatRatio):


    MachNum_Throat = 1
    

    functionAns = (MachNum_Throat/MachNum_Exit)*((1+((specificHeatRatio-1)/2)*MachNum_Exit**2)/(1+((specificHeatRatio-1)/2)*MachNum_Throat**2))**((specificHeatRatio+1)/(2*(specificHeatRatio-1))) - areaRatio

    return functionAns

def calculateMachNumberThroat(areaRatioArray, specificHeatRatio):
    # Function which will calculate the throat mach number using Netwonian Raphson scheme
    #--------------------------------------------------------------------------------------
    # Inputs: 
    # Area Ratio - array floats - Area ratio
    # specificHeatRatio - float - Specific heat ratio of inlet
    #--------------------------------------------------------------------------------------
    # Outputs: None
    #--------------------------------------------------------------------------------------

    MachNum_Exit_Array = []
    for i in range(len(areaRatioArray)):
        # Calculate the function f(MachNum_Exit) see poage 5.1 equation 5.1.1 from cryo-rocket.com 

        MachNum_Exit = 1
        functionSol = -1
        while(functionSol <= 0.000001):
            functionSol = function5_5_1(MachNum_Exit, areaRatioArray[i], specificHeatRatio) # Calculate function value at current estimated guess for Exit Mach Number
            
            MachNum_Exit = MachNum_Exit + 0.001
        MachNum_Exit_Array.append(MachNum_Exit)
    return MachNum_Exit_Array

def calcualteExitPressure(stagnationPressure, MachNum_Exit_Array, specificHeatRatio):
    # Function which will calculate the exit pressure
    #--------------------------------------------------------------------------------------
    # Inputs: 
    # stagnationPressure - float - Injector face pressure
    # MachNum_Exit_Array - array floats - Exit Mach Numbers
    # specificHeatRatio - float - Specific heat ratio of inlet
    #--------------------------------------------------------------------------------------
    # Outputs: ExitPressure_Array - array floats - Array of Exit Pressure 
    #--------------------------------------------------------------------------------------

    ExitPressure_Array = []
    for MachNum_Exit in MachNum_Exit_Array:
        # Based off equation 5.1.2
        ExitPressure_Array.append(stagnationPressure*(1+((specificHeatRatio-1)/2)*MachNum_Exit**2)**(-specificHeatRatio/(specificHeatRatio-1)))

    return ExitPressure_Array

def calculateThrustCoefficient(stagnationPressure, Pressure_Exit_Array, areaRatioArray, specificHeatRatio):
    # Function which will calculate the thrust coefficient
    #--------------------------------------------------------------------------------------
    # Inputs: 
    # stagnationPressure - float - Injector face pressure
    # Pressure_Exit_Array - array float - Exit Pressures  based off areaRatioArray
    # areaRatioArray - array floats - Thrust chamber area ratios
    # specificHeatRatio - float - Specific heat ratio of inlet
    #--------------------------------------------------------------------------------------
    # Outputs: Thrust_Coefficient_Array - array floats - Array of Thrust Coefficients based off areaRatioArray
    #--------------------------------------------------------------------------------------
    Thrust_Coefficient_Array = []

    for i in range(len(areaRatioArray)):
        # Calculation based off equationb 3.4.2
        Thrust_Coefficient_Array.append(np.sqrt((2*specificHeatRatio**2)/(specificHeatRatio-1) * ((2/(specificHeatRatio+1))**((specificHeatRatio+1)/(specificHeatRatio-1))) * (1 - (Pressure_Exit_Array[i]/stagnationPressure)**((specificHeatRatio-1)/specificHeatRatio))) + areaRatioArray[i]*(Pressure_Exit_Array[i]/stagnationPressure))

    return Thrust_Coefficient_Array

def calculateThroatArea(ThrustVacuum, Thrust_Coeffecient_Array, stagnationPressure):
    # Function which will calculate the thrust coefficient
    #--------------------------------------------------------------------------------------
    # Inputs: 
    # ThrustVacuum - float - desired thrust in vacuum
    # Thrust_Coefficient_Array - array float - Thrust coefficients based off areaRatioArray
    # stagnationPressure - float - Injector face pressure
    #--------------------------------------------------------------------------------------
    # Outputs: Throat_Area_Array - array floats - Array of Thrust Coefficients based off areaRatioArray
    #--------------------------------------------------------------------------------------
    Throat_Area_Array = []
    for ThrustCoeff in Thrust_Coeffecient_Array:
        Throat_Area_Array.append(ThrustVacuum/(ThrustCoeff*stagnationPressure))

    return Throat_Area_Array

def calculateMassFlow(Throat_Area_Array, specificHeatRatio, tempInjector, pressureInjector, specificGasConstant):
    # Function which will calculate the mass flow rate of the engine
    #--------------------------------------------------------------------------------------
    # Inputs: 
    # specificHeatRatio - float - Specific heat ratio of inlet
    #--------------------------------------------------------------------------------------
    # Outputs: Mass_Flow_Array - array floats - Array of mass flow rates
    #--------------------------------------------------------------------------------------

    # Constants:
    MachNum_Throat = 1 # [Mach]
    

    # Calculate Temperature Throat using Eqtn 5.1.4
    tempThroat = tempInjector*(1+(specificHeatRatio-1)/2 * MachNum_Throat)**(-1)

    # Calculate Pressure in Throat using Eqtn 5.1.5
    pressureThroat = pressureInjector*(1+(specificHeatRatio-1)/2 * 1)**((-specificHeatRatio)/(specificHeatRatio-1))

    # Calculate density at the Throat
    throatDensity = pressureThroat/(specificGasConstant*tempThroat)

    # Calculate sonic velocity at Throat
    sonicVelocityThroat = np.sqrt(specificHeatRatio*specificGasConstant*tempThroat)

    # Calculate mass flow rate
    Mass_Flow_Array = []
    for ThroatArea in Throat_Area_Array:
        Mass_Flow_Array.append(throatDensity*sonicVelocityThroat*ThroatArea)
    

    return Mass_Flow_Array

def main():
    # Main Function, no inputs. Controls software at a high level sending commands to sub-functions/scripts
    #--------------------------------------------------------------------------------------
    # Inputs: None
    #--------------------------------------------------------------------------------------
    #Outputs: None
    #--------------------------------------------------------------------------------------

    # Constants
    specificHeatRatio = 1.19346 
    areaRatioArray = np.arange(1,80,0.1)
    tempInjector = 3560 # [K]
    pressureInjector = 18.61E6 # [Pa]
    specificGasConstant = 613.525 # Unitless

     # Step 1: Calculate the Cross Sectional Area of the Throat
    indexPos = np.where(areaRatioArray >= 69.5)
    stagnationPressure = 2700*6894.7572931783 # [pa] this is injector pressure
    ThrustVacuum = 491900*4.4482189159 # [N] Vacuum Thrust 

    # Calculate the gas mach number at chamber exit
    MachNum_Exit_Array = calculateMachNumberThroat(areaRatioArray, specificHeatRatio)

    # Calculate the gas pressure at chamber exit
    Pressure_Exit_Array = calcualteExitPressure(stagnationPressure, MachNum_Exit_Array, specificHeatRatio)
    
    # Calculate Thrust Coeffiecient based off area ratio
    Thrust_Coeffecient_Array = calculateThrustCoefficient(stagnationPressure, Pressure_Exit_Array, areaRatioArray, specificHeatRatio)

    # Calculate throat area as function of area ratio
    Throat_Area_Array = calculateThroatArea(ThrustVacuum, Thrust_Coeffecient_Array, stagnationPressure)

    # Calculate Exit Area as function of throat area and area ratio
    Exit_Area_Array = Throat_Area_Array*areaRatioArray

    # Calculate mass flow rate based off area ratio, Throat Area and Exit Area
    Mass_Flow_Array = calculateMassFlow(Throat_Area_Array, specificHeatRatio, tempInjector, pressureInjector, specificGasConstant)
    #print(Throat_Area_Array[indexPos[0][0]])

    # Display Data
    print(r'Throat Area A33.1 $[m^2]$:''{0} \n 'r'Exit Area A34 $[m^2]$: ''{1} \n'r'Mass Flow Rate [kg/sec]: {2}'.format(Throat_Area_Array[indexPos[0][0]], Exit_Area_Array[indexPos[0][0]], Mass_Flow_Array[indexPos[0][0]]))

    # Plot Results
    ydata_array = [MachNum_Exit_Array, Pressure_Exit_Array, Thrust_Coeffecient_Array, Throat_Area_Array, Exit_Area_Array, Mass_Flow_Array]
    xlabel_array = ['Area Ratio', 'Area Ratio', 'Area Ratio', 'Area Ratio', 'Area Ratio', 'Area Ratio']
    ylabel_array = [r'Exit Mach Number $M_{34}$', r'Exit Pressure, $P_{34}$ [Pa]', 'Thrust Coefficient [Pa]', r'$A_{33.1}$ $[m^2]$', r'$A_{34}$ $[m^2]$', r'Mass Flow Rate $[kg/sec]$']    
    title_array = ['Exit Mach Number vs Area Ratio', 'Exit Pressure vs Area Ratio', ' Thrust Coefficient vs Area Ratio', r'$A_{33.1}$ vs Area Ratio', r'$A_{34}$ vs Area Ratio', 'Mass Flow Rate vs Area Ratio (F=491,000lbs)']
    xlim_array = [[0,80],[0,80], [0, 80], [0,80], [0,80], [0,80]]
    ylim_array = [[1,5],[0,12E6], [1.2, 2], [0.055, 0.09], [0,5], [450,800]]
    

  
    for i in range(len(ydata_array)):
        plt.figure(i)
        plt.plot(areaRatioArray, ydata_array[i])
        plt.xlim(xlim_array[i])
        plt.ylim(ylim_array[i])
        plt.xlabel(xlabel_array[i])
        plt.ylabel(ylabel_array[i])
        plt.title(title_array[i])
        plt.grid()

    
    plt.tight_layout()
    plt.show()
    return None

main()
