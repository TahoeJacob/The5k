# Hydrogen model V2 used to calculate density of liquid Hydrogen
# By Jacob Saunders
# Based off www.cryo-rocket.com section 2.2

import numpy as np
import matplotlib.pyplot as plt
import csv


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

    dataParaPercent = extractFileCSV('Archive\HydrogenOrthoPara.csv')

    paraPercentage = np.array(dataParaPercent[1::], dtype=float) # creates array of Temp and corresponding para percentages
    paraTemp = []
    paraPercent = []
   
    for value in paraPercentage:
        paraTemp.append(value[0])
        paraPercent.append(value[1])

    # fig1 = plt.figure(1)
    
    # ax = fig1.add_subplot(1,1,1)

    # ax.plot(paraTemp, paraPercent)


    # ax.set_xlim([0,400])
    # ax.set_ylim([0,120])
    # ax.set_xlabel("Temperature [K]")
    # ax.set_ylabel('Parahydrogen Percentage')
    # ax.set_title('Hydrogen Composition')
    # ax.grid()

    return paraTemp,paraPercent


def para_fraction(temperature):
  """
  Calculates the para fraction based on the closest temperature input.

  Args:
    temperature: The input temperature.

  Returns:
    The para fraction.
  """

  temperatures = np.array([10, 20, 20.39, 30, 33.1, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 298.16, 300, 350, 400, 500])
  para_percentages = np.array([99.9999, 99.821, 99.789, 97.021, 95.034, 88.727, 77.054, 65.569, 55.991, 48.537, 42.882, 38.62, 32.959, 28.603, 25.974, 25.264, 25.075, 25.072, 25.019, 25.005, 25])

  # Use np.interp for interpolation
  para_fraction = np.interp(temperature, temperatures, para_percentages)
  return para_fraction

def create_plot(x_axis,y_axis, x_label, y_label, title):
    # Inputs:
    # x_axis - type: array - x axis values
    # y_axis - type: array - y axis values
    # x_label - type: string - x axis label
    # y_label - type: string - y axis label
    # title - type: string - title of the plot

    plt.figure()  # Create a new figure
    plt.plot(x_axis, y_axis)
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'{y_label}')
    plt.title(f'{title}')
    #plt.gca().yaxis.set_major_formatter('{:.1e}'.format)
    stringStart = 'X:{:.2f}, Y:{:.3f}'.format(float(x_axis[0]), float(y_axis[0]))
    stringEng = 'X:{:.2f}, Y:{:.3f}'.format(float(x_axis[-1]), float(y_axis[-1]))
    plt.annotate(stringStart,
                xy=(x_axis[0], y_axis[0]), xycoords="data",
                xytext=(100,10), textcoords="offset points",
                va="center", ha="center",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"))
    plt.annotate(stringEng,
                xy=(x_axis[-1], y_axis[-1]), xycoords="data",
                xytext=(-100,10), textcoords="offset points",
                va="center", ha="center",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"))
    plt.grid(True)

def calc_alpha0(tau, delta, c, coef):
    # Function to calculate the ideal component of the Helmholtz derivative
    # Inputs
    # tau: recipricol reduced temperature
    # delta: reduced density 
    # c: if we are using Para or Ortho hydrogen (c = "P" or "O" respectively)
    # coef: dictionary containing the coefficients for the model

    return ( np.log(delta) + 1.5*np.log(tau) + coef[c+'a'][1] + coef[c+'a'][2]*tau + [coef[c+'a'][k]*np.log(1-np.exp(coef[c+'b'][k]*tau)) for k in range(3,len(coef[c+'a'])) ] )

def calc_alpha_r(tau, delta, c, coef):
    # Function to calculate the real component of the Helmholtz derivative
    # Inputs:
    # tau: reciprocal reduced temperature 
    # delta: reduced density
    # c: if we are using Para or Ortho hydrogen (c = "P" or "O" respectively)
    # coef: dictionary containing the coefficients for the model
    alpha_r = 0
    sumTwo = 0
    sumThree = 0 
    for i in range(len(coef[c+'N'])):

        if i >= 1 and i <= 7:
            alpha_r += (coef[c+'N'][i]*delta**(coef[c+'d'][i]) * tau**(coef[c+'t'][i]) )
    
        if i >= 8 and i <= 9:
            alpha_r +=  ( coef[c+'N'][i]*delta**(coef[c+'d'][i]) * tau**(coef[c+'t'][i]) * np.exp(-delta**(coef[c+'p'][i])) )
            
        if i >= 10 and i <= 14:
            alpha_r += ( coef[c+'N'][i]*delta**(coef[c+'d'][i]) * tau**(coef[c+'t'][i]) * np.exp(coef[c+'phi'][i]*(delta-coef[c+'D'][i])**2 + coef[c+'beta'][i]*(tau - coef[c+'gam'][i])**2) ) 
    
    return alpha_r

# Define derivates of the above helmholtz functions
def calc_alpha_r_delta(tau, delta, c, coef):
    # First partial derivative of alpha_r with respect to delta
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model

    alpha_r_delta = 0

    for i in range(len(coef[c+'N'])):
        
        # iteratre through the 3 sum functions
        if i >= 1 and i <= 7:
            alpha_r_delta += ( coef[c+'d'][i]*coef[c+'N'][i] * delta**(coef[c+'d'][i]-1) * tau**(coef[c+'t'][i]) )
        if i >= 8 and i <= 9:
            alpha_r_delta += ( (coef[c+'N'][i] * delta**(coef[c+'d'][i]) * tau**(coef[c+'t'][i]) * np.exp(-delta**(coef[c+'p'][i])) * (coef[c+'d'][i] - coef[c+'p'][i]*delta**(coef[c+'p'][i])) )/delta )
        if i >= 10 and i <= 14: 
            alpha_r_delta += ( coef[c+'N'][i]*delta**(coef[c+'d'][i]-1) * tau**(coef[c+'t'][i]) * np.exp(coef[c+'phi'][i]*(delta-coef[c+'D'][i])**2 + coef[c+'beta'][i]*(tau - coef[c+'gam'][i])**2) * (2*coef[c+'phi'][i]*delta*(delta-coef[c+'D'][i]) + coef[c+'d'][i]) )

    return alpha_r_delta #+ sumTwo*np.exp(-delta**2) + sumThree*np.exp(-delta**4)

def calc_alpha_r_tau(tau, delta, c, coef):
    # First partial derivative of alpha_r with respect to tau
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model

    alpha_r_tau = 0
   

    for i in range(len(coef[c+'N'])):

        if i >= 1 and i <= 7:
            alpha_r_tau += ( coef[c+'N'][i] * delta**(coef[c+'d'][i]) * tau**(coef[c+'t'][i]-1) * coef[c+'t'][i] )
        if i >= 8 and i <= 9:
            alpha_r_tau += ( coef[c+'N'][i] * delta**(coef[c+'d'][i]) * tau**(coef[c+'t'][i]-1) * coef[c+'t'][i] * np.exp(-delta**(coef[c+'p'][i])) )
        if i >= 10 and i <= 14:
            alpha_r_tau += ( coef[c+'N'][i]*delta**(coef[c+'d'][i])*tau**(coef[c+'t'][i]-1) * np.exp(coef[c+'phi'][i]*(delta-coef[c+'D'][i])**2 + coef[c+'beta'][i]*(tau - coef[c+'gam'][i])**2) ) * (2*coef[c+'beta'][i]*tau*(tau - coef[c+'gam'][i]) + coef[c+'t'][i] )
    return alpha_r_tau 

def calc_alpha0_tau(tau, delta, c, coef):
    # First partial derivative of alpha0 with respect to tau
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model
    sumOne = 0
    for k in range(3,len(coef[c+'a'])):
        sumOne += ( (-coef[c+'a'][k]*coef[c+'b'][k]*np.exp(coef[c+'b'][k]*tau))/(1-np.exp(coef[c+'b'][k]*tau)) )

    return ( (1.5/tau) + coef[c+'a'][2] + sumOne )

def calc_alpha0_tau_tau(tau, delta, c, coef):
    # Function to calculate the second partial derivative of alpha0 with respect to tau
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model
    sumOne = 0
    for k in range(3,len(coef[c+'a'])):
        a = coef[c+'a'][k]
        b = coef[c+'b'][k]
        sumOne += ( (a*(b**2)*np.exp(b*tau))/(np.exp(b*tau)-1) - (a*(b**2)*np.exp(2*b*tau))/((np.exp(b*tau)-1)**2) )
        #sumOne += ( ( (-coef[c+'a'][k] * (coef[c+'b'][k]**2) * np.exp(coef[c+'b'][k]*tau))/(1-np.exp(coef[c+'b'][k]*tau))) - ((coef[c+'a'][k] * (coef[c+'b'][k]**2)* (np.exp(coef[c+'b'][k]*tau)**2) )/((1-np.exp(coef[c+'b'][k]*tau))**2)))
    return ( -3/(2*tau**2) + sumOne )

def calc_alpha_r_tau_tau(tau, delta, c, coef):
    # Function to calculate the second partial derivative of alpha_r with respect to tau
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model

    alpha_r_tau_tau = 0
    sumTwo = 0
    sumThree = 0
    for i in range(len(coef[c+'N'])):
        N = coef[c+'N'][i]
        d = coef[c+'d'][i]
        t = coef[c+'t'][i]
        gamma = coef[c+'gam'][i]
        phi = coef[c+'phi'][i]
        beta = coef[c+'beta'][i]
        D = coef[c+'D'][i]


        if i >= 1 and i <= 7:
            alpha_r_tau_tau += (N * (delta**d) * t * (t - 1) * (tau**(t - 2)) )
        if i >= 8 and i <= 9:
            p = coef[c+'p'][i]
            alpha_r_tau_tau += N*(delta**d)*t*(tau**(t - 2))*np.exp(-delta**p)*(t - 1)
        if i >= 10 and i <= 14:
            alpha_r_tau_tau += 2*N*beta*(delta**d)*(tau**t)*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2)) + N*(beta**2)*(delta**d)*(tau**t)*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2))*((2*gamma - 2*tau)**2) + N*(delta**d)*t*(tau**(t - 2))*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2))*(t - 1) - 2*N*beta*(delta**d)*t*(tau**(t - 1))*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2))*(2*gamma - 2*tau)
            
    return alpha_r_tau_tau #+ sumTwo*np.exp(-delta**2) + sumThree*np.exp(-delta**4)

def calc_alpha_r_delta_delta(tau, delta, c, coef):
    # Function to calculate the second partial derivative of alpha_r with respect to delta
    #   Inputs:
    #   tau: reciprocal reduced temperature
    #   delta: reduced density
    #   coef: dictionary containing the coefficients for the model
    # coef['n'][i]*coef['r'][i]*(coef['r'][i]-1)* delta**(coef['r'][i]-2) * tau**(coef['s'][i])

    alpha_r_delta_delta = 0
    sumTwo = 0
    sumThree = 0
    for i in range(len(coef[c+'N'])):
        N = coef[c+'N'][i]
        d = coef[c+'d'][i]
        t = coef[c+'t'][i]
        gamma = coef[c+'gam'][i]
        phi = coef[c+'phi'][i]
        beta = coef[c+'beta'][i]
        D = coef[c+'D'][i]

        if i >= 1 and i <= 7:
            alpha_r_delta_delta += ( N * d * (delta**(d-2)) * (tau**t) * (d - 1))
            #alpha_r_delta_delta += ( coef[c+'N'][i] * delta**(coef[c+'d'][i] - 2) * tau**(coef[c+'t'][i]) * coef[c+'d'][i] * (coef[c+'d'][i] - 1) )
        if i >= 8 and i <= 9:
            p = coef[c+'p'][i]  
            alpha_r_delta_delta += ( N*d*(delta**(d - 2))*(tau**t)*np.exp(-delta**p)*(d - 1) + N*(delta**d)*(delta**(2*p - 2))*(p**2)*(tau**t)*np.exp(-delta**p) - 2*N*d*(delta**(d - 1))*(delta**(p - 1))*p*(tau**t)*np.exp(-delta**p) - N*(delta**d)*(delta**(p - 2))*p*(tau**t)*np.exp(-delta**p)*(p - 1)
 )
            #alpha_r_delta_delta += ( coef[c+'N'][i] * tau**(coef[c+'t'][i]) *delta**(coef[c+'d'][i] - 2) * np.exp(-delta**(coef[c+'p'][i])) * (delta**(2*coef[c+'p'][i]) * coef[c+'p'][i]**2 + (-2*coef[c+'d'][i]*coef[c+'p'][i] - coef[c+'p'][i]**2 + coef[c+'p'][i])*delta**(coef[c+'p'][i]) + coef[c+'d'][i]**2 - coef[c+'d'][i] ) )
        if i >= 10 and i <= 14:
            alpha_r_delta_delta += ( 2*N*(delta**d)*phi*(tau**t)*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2)) + N*(delta**d)*(phi**2)*(tau**t)*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2))*((2*D - 2*delta)**2) + N*d*(delta**(d - 2))*(tau**t)*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2))*(d - 1) - 2*N*d*(delta**(d - 1))*phi*(tau**t)*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2))*(2*D - 2*delta)
)
            #alpha_r_delta_delta += ( coef[c+'N'][i] * delta**(coef[c+'d'][i] - 2) * tau**(coef[c+'t'][i]) * np.exp(coef[c+'phi'][i]*(delta-coef[c+'D'][i])**2 + coef[c+'beta'][i]*(tau - coef[c+'gam'][i])**2) * ( (4*coef[c+'phi'][i]*(delta**2)*((delta-coef[c+'D'][i])**2) * 2*(delta - coef[c+'D'][i]) ) + (4*coef[c+'phi'][i]*(delta-coef[c+'D'][i])*coef[c+'d'][i]*delta) + (2*coef[c+'phi'][i]*(delta-coef[c+'D'][i])*delta**2) + (4*(delta-coef[c+'D'][i])*delta**2) + coef[c+'d'][i]**2 - coef[c+'d'][i] ) )
    
    return alpha_r_delta_delta #+ sumTwo*np.exp(-delta**2) + sumThree*np.exp(-delta**4)

def calc_alpha_r_delta_tau(tau, delta, c, coef):
    # Function to calculate the mixed partial derivative of alpha_r with respect to delta and tau
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model

    alpha_r_delta_tau = 0
    sumTwo = 0
    sumThree = 0

    for i in range(len(coef[c + 'N'])):
        N = coef[c+'N'][i]
        d = coef[c+'d'][i]
        t = coef[c+'t'][i]
        gamma = coef[c+'gam'][i]
        phi = coef[c+'phi'][i]
        beta = coef[c+'beta'][i]
        D = coef[c+'D'][i]

        if i >= 1 and i <= 7:
            alpha_r_delta_tau += (N*d*(delta**(d - 1))*t*(tau**(t - 1)))
            #alpha_r_delta_tau += ( coef[c+'N'][i] * delta**(coef[c+'d'][i] -1) * tau**(coef[c+'t'][i] -1) * coef[c+'d'][i] * coef[c+'t'][i] )
        if i >= 8 and i <= 9:
            p = coef[c+'p'][i]
            alpha_r_delta_tau += N*d*(delta**(d - 1))*t*(tau**(t - 1))*np.exp(-delta**p) - N*(delta**d)*(delta**(p - 1))*p*t*(tau**(t - 1))*np.exp(-delta**p)

            #alpha_r_delta_tau += ( coef[c+'N'][i] * delta**(coef[c+'d'][i] -1) * tau**(coef[c+'t'][i] -1) * coef[c+'t'][i] * np.exp(-delta**(coef[c+'p'][i])) * (coef[c+'d'][i] - coef[c+'p'][i]*delta**(coef[c+'p'][i])))
        if i >= 10 and i <= 14:
            alpha_r_delta_tau += N*d*(delta**(d - 1))*t*(tau**(t - 1))*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2)) - N*(delta**d)*phi*t*(tau**(t - 1))*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2))*(2*D - 2*delta) - N*beta*d*(delta**(d - 1))*(tau**t)*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2))*(2*gamma - 2*tau) + N*beta*(delta**d)*phi*(tau**t)*np.exp(phi*((D - delta)**2) + beta*((gamma - tau)**2))*(2*D - 2*delta)*(2*gamma - 2*tau)

            #alpha_r_delta_tau += (4*tau**(coef[c+'t'][i] -1)*np.exp(coef[c+'phi'][i]*(delta-coef[c+'D'][i])**2 + coef[c+'beta'][i]*(tau - coef[c+'gam'][i])**2) * (coef[c+'phi'][i]*(delta-coef[c+'D'][i])*delta + coef[c+'d'][i]/2)*coef[c+'N'][i] * delta**(coef[c+'d'][i]-1) * (coef[c+'beta'][i]*(tau-coef[c+'gam'][i])*tau + coef[c+'t'][i]/2) )

    return alpha_r_delta_tau #+ sumTwo*np.exp(-delta**2) + sumThree*np.exp(-delta**4)

def Helmholtz(rho_guess, T, Tc, rho_c, Rs, c, coefficients):
    #  Helmholtz(rho, T, coeficients) is my function that uses all the oxygen coeficients
    #         % to calculate thermo properties.  You will need to replace this with
    #         % your version.  All of the outputs are stored in
    #         % helmholtz_output. For example:
    #         % helmholtz_output =[h,P,T,S, rho, Cp, Cv, alpha_r_delta]
    #         % where
    #         % h = Enthalpy [kJ/kg]
    #         % P = Pressure [Pa]
    #         % T = Temperature [K]
    #         % S = entropy [kJ/(kg * K)]
    #         % rho = density [kg/m^3]
    #         % Cp = Specific heat at constant pressure [kJ/(kg * K)]
    #         % Cv = Specific heat at constant volume [kJ/(kg * K)]
    #         % alpha_r_delta = helmholtz derivative of real component w.r.t delta    
    #         % Adjust as needed to fit your code.

    # Constants
    p0 = 101325  # Reference pressure [Pa]
    T0 = 298.15  # Reference temperature [K]

    # Convert temperature and density to reduced variables
    delta = rho_guess / rho_c
    tau = Tc / T

    # Calculate alpha0 
    alph0 = calc_alpha0(tau, delta, c, coefficients)

    # Calculate alpha0_tau
    alph0T = calc_alpha0_tau(tau, delta, c, coefficients)

    # Calculate alpha0_tau_tau
    alph0TT = calc_alpha0_tau_tau(tau, delta, c, coefficients)

    # Calculate alpha_r
    alphR = calc_alpha_r(tau, delta, c, coefficients)

    # Calculate alpha_r_delta
    alphRD = calc_alpha_r_delta(tau, delta, c, coefficients)

    # Calculate alpha_r_tau
    alphRT = calc_alpha_r_tau(tau, delta, c, coefficients)
    
    # Calculate alpha_r_tau_tau
    alphRTT = calc_alpha_r_tau_tau(tau, delta, c, coefficients)
    #changeIn = 1E-2
    #alphRTT = (calc_alpha_r(tau+changeIn, delta, c, coefficients) - 2*calc_alpha_r(tau, delta, c, coefficients) + calc_alpha_r(tau-changeIn,delta, c, coefficients))/(changeIn**2)#calc_alpha_r_tau_tau(tau, delta, c, coefficients)

    # Calculate alpha_r_delta_delta
    alphRDD = calc_alpha_r_delta_delta(tau, delta, c, coefficients)

    # Calculate alpha_r_delta_tau
    alphRDT = calc_alpha_r_delta_tau(tau, delta, c, coefficients)


    # Calculate enthalpy [kJ/kg]
    h = (1 + tau*(alph0T + alphRT) + delta*alphRD) * Rs*T 

    # Calculate pressure [Pa]
    P = rho_guess * Rs * T * (1 + delta * alphRD)

    # Calculate entropy [kJ/kgK]
    s = (tau*(alph0T + alphRT) - alph0 - alphR)* Rs 

    # Calculate isochoric heat capactiy [kJ/kg K]
    Cv = -(tau**2) * (alph0TT + alphRTT)

    # Calculate isobaric heat capactiy [kJ/kg K]
    Cp =  ((1 + delta*alphRD - delta*tau*alphRDT)**2)/(1 + 2*delta*alphRD + (delta**2) * alphRDD) + Cv

    return [h, P, T, s, rho_guess, Cp*Rs*0.001, Cv*Rs*0.001, alphRD, alphRDT, alphRDD, alph0TT, alphRTT, alph0T]

def hydrogen_debugging(coefficients):
    # Constants
    Tc = 32.938  # H2 critical temperature [K]
    Pc = 1.284E6  # H2 critical pressure [Pa]
    rho_c = 31.262  # critical density [kg/m^3]
    M = 2.016  # H2 molecular weight [g/mol]
    R = 8314.472  # universal gas constant
    Rs = R / M  # Specific gas constant

    # Desired pressure
    # P_desired = 1.25 * Pc
    P_array = [ 0.2E6]

    # Initial guess for rho
    rho_guesses = [80]  # initial guess for density [kg/m^3]
    d_rho = 0.0001  # delta rho used for derivative


    for P_desired, rho_guess in zip(P_array, rho_guesses):
        plot_T = []
        plot_Cp = []
        plot_h = []
        plot_rho = []
        plot_alpha_r_delta = []
        plot_alpha_r_delta_delta = []
        plot_alpha_r_delta_tau = []
        plot_alpha_r_tau_tau = []
        plot_alpha0_tau_tau = []
        plot_Cv = []
        
        plot_delta = []
        for T in range(14, int(6 * Tc) + 1):

            # For this Temp grab the ortho/para percentage
            paraPercent = para_fraction(T)/100

            # Newton solver at each temperature of interest 
            error = 999
            while error > 1E-10:
            #for i in range(200):
                helmholtz_output_para = Helmholtz(rho_guess, T, Tc, rho_c, Rs, 'P', coefficients)
                helmholtz_output_ortho = Helmholtz(rho_guess, T, Tc, rho_c, Rs, 'O', coefficients)
                
                alpha_r_delta = helmholtz_output_para[7]*paraPercent + helmholtz_output_ortho[7]*(1-paraPercent)
                delta = rho_guess / rho_c
                P_guess = rho_guess * Rs * T * (1 + delta * alpha_r_delta)

                rho_high = rho_guess + d_rho
                helmholtz_output_para = Helmholtz(rho_high, T, Tc, rho_c, Rs, 'P', coefficients)
                helmholtz_output_ortho = Helmholtz(rho_high, T, Tc, rho_c, Rs, 'O', coefficients)
                alpha_r_delta = helmholtz_output_para[7]*paraPercent + helmholtz_output_ortho[7]*(1-paraPercent)
                delta = rho_high / rho_c
                P_guess_high = rho_high * Rs * T * (1 + delta * alpha_r_delta)

                rho_low = rho_guess - d_rho
                helmholtz_output_para = Helmholtz(rho_low, T, Tc, rho_c, Rs, 'P', coefficients)
                helmholtz_output_ortho = Helmholtz(rho_low, T, Tc, rho_c, Rs, 'O', coefficients)
                alpha_r_delta = helmholtz_output_para[7]*paraPercent + helmholtz_output_ortho[7]*(1-paraPercent)
                delta = rho_low / rho_c
                P_guess_low = rho_low * Rs * T * (1 + delta * alpha_r_delta)

                derivative = (P_guess_high - P_guess_low) / (2 * d_rho)

                rho_new = rho_guess + (P_desired - P_guess) / derivative
                error = abs(rho_guess - rho_new)
                rho_guess = rho_new
               

            helmholtz_output_para = Helmholtz(rho_guess, T, Tc, rho_c, Rs, 'P', coefficients)
            helmholtz_output_ortho = Helmholtz(rho_guess, T, Tc, rho_c, Rs, 'O', coefficients)
            Cp = helmholtz_output_para[5]*paraPercent + helmholtz_output_ortho[5]*(1-paraPercent)
            Cv = helmholtz_output_para[6]*paraPercent + helmholtz_output_ortho[6]*(1-paraPercent)
            # h, P, T, s, rho_guess, Cp, Cv, alphRD, alphRDD, alphRDT, alphRT, alphRTT, alph0, alpha0T, alph0TT, alphR
            plot_Cp.append(Cp)
            plot_T.append(T)
            plot_rho.append(rho_guess)
            plot_h.append(helmholtz_output_para[0]*paraPercent + helmholtz_output_ortho[0]*(1-paraPercent))
            plot_alpha_r_delta.append(helmholtz_output_para[7]*paraPercent + helmholtz_output_ortho[7]*(1-paraPercent) )
            plot_delta.append(rho_guess)
            plot_Cv.append(Cv)
            plot_alpha_r_delta_delta.append(helmholtz_output_para[9]*paraPercent + helmholtz_output_ortho[9]*(1-paraPercent))
            plot_alpha_r_delta_tau.append(helmholtz_output_para[8]*paraPercent + helmholtz_output_ortho[8]*(1-paraPercent))
            plot_alpha_r_tau_tau.append(helmholtz_output_para[11]*paraPercent + helmholtz_output_ortho[11]*(1-paraPercent))
            plot_alpha0_tau_tau.append(helmholtz_output_para[10]*paraPercent + helmholtz_output_ortho[10]*(1-paraPercent))
        
  
        create_plot(plot_T, plot_Cp, 'Temperature [K]', 'Isobaric Heat Capacity [J/(kg*K)]', 'Isobaric Heat Capacity vs. Temperature')
        create_plot(plot_T, plot_Cv, 'Temperature [K]', 'Isochoric Heat Capacity [J/(kg*K)]', 'Isochoric Heat Capacity vs. Temperature')
        create_plot(plot_T, plot_rho, 'Temperature [K]', 'Density [kg/m^3]', 'Denisty vs. Temperature')
        # create_plot(plot_T, plot_alpha_r_delta_delta, 'Temperature [K]', r'$\alpha_{\delta\delta}^r$', r'$\alpha_{\delta\delta}^r$ vs. Temperature')
        # create_plot(plot_T, plot_alpha_r_delta_tau, 'Temperature [K]', r'$\alpha_{\delta\tau}^r$', r'$\alpha_{\delta\tau}^r$ vs. Temperature')
        # create_plot(plot_T, plot_alpha_r_delta, 'Temperature [K]', r'$\alpha_{\delta}^r$', r'$\alpha_{\delta}^r$ vs. Temperature')
        # create_plot(plot_T, plot_alpha_r_tau_tau, 'Temperature [K]', r'$\alpha_{\tau\tau}^r$', r'$\alpha_{\tau\tau}^r$ vs. Temperature')
        # create_plot(plot_T, plot_alpha0_tau_tau, 'Temperature [K]', r'$\alpha_{\tau\tau}^0$', r'$\alpha_{\tau\tau}^0$ vs. Temperature')
        # plt.figure(3)
        # plt.plot(plot_T, plot_h, '-b', linewidth=2)
        # plt.xlabel('Temperature [K]')
        # plt.ylabel('Enthalpy [kJ/kg]')
        # plt.title('Enthalpy [kJ/kg] vs. Temperature')
        # plt.legend(['P=1.25Pc'])
        # plt.grid(True)

        #create_plot(plot_T, plot_delta, 'Temperature [K]', r'$\delta$', r'$\delta$ vs. Temperature')
        # plt.figure(5)
        # plt.plot(plot_T, plot_Cv, '-b', linewidth=2)
        # plt.xlabel('Temperature [K]')
        # plt.ylabel('Isobaric Heat Capacity [J/(kg*K)]')
        # plt.title('Isobaric Heat Capacity vs. Temperature')
        # plt.legend(['P=1.25Pc'])
        # plt.grid(True)


def hydrogen_thermodynamics(P_desired, rho_guess, paraPercent, T):
    # Constants
    Tc = 32.938  # H2 critical temperature [K]
    Pc = 1.284E6  # H2 critical pressure [Pa]
    rho_c = 31.262  # critical density [kg/m^3]
    M = 2.016  # H2 molecular weight [g/mol]
    R = 8314.472  # universal gas constant
    Rs = R / M  # Specific gas constant
    d_rho = 0.0001  # delta rho used for derivative

    # Define the coefficients from the oxygen model table 2.3.2 in the book ignore first element place holder to shift dict index by 1
    coefficients = {
        "Pa": [1000000, -1.4485891134, 1.884521239, 4.30256, 13.0289, -47.7365, 50.0013, -18.6261, 0.993973, 0.536078],
        "Pb": [1000000, 0, 0, -15.1496751472, -25.0925982148,  -29.4735563787,-35.4059141417, -40.724998482, -163.7925799988, -309.2173173842],
        "Oa": [1000000, -1.4675442336, 1.8845068862,2.54151, -2.3661,  1.00365,  1.22447],
        "Ob": [1000000, 0, 0, -25.7676098736,  -43.4677904877, -66.0445514750, -209.7531607465],

        "PN": [1000000, -7.33375,  0.01, 2.60375, 4.66279, 0.682390, -1.47078,  0.135801, -1.05327, 0.328239, -0.0577833, 0.0449743, 0.0703464, -0.0401766,  0.119510],
        "Pt": [1000000, 0.6855   , 1   , 1      , 0.489  , 0.774   , 1.133   ,  1.386   , 1.619   , 1.162   , 3.96      , 5.276    , 0.99     , 6.791     , 3.19     ],
        "Pd": [1000000, 1, 4, 1, 1, 2, 2, 3, 1 ,3, 2, 1, 3, 1, 1],
        "Pp": [1000000, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        "ON": [1000000, -6.83148, 0.01, 2.11505, 4.38353, 0.211292, -1.00939, 0.142086, -0.87696, 0.804927, -0.710775, 0.0639688, 0.0710858, -0.087654, 0.647088],
        "Ot": [1000000, 0.7333  , 1   , 1.1372 ,  0.5136, 0.5638  ,  1.6248 , 1.829   , 2.404   , 2.105   , 4.1      , 7.658    , 1.259    , 7.589    , 3.946  ],
        "Od": [1000000, 1, 4, 1 ,1 ,2 ,2 ,3 ,1 ,3, 2, 1, 3, 1, 1],
        "Op": [1000000, 0, 0, 0, 0, 0, 0 ,0, 1, 1],
        "PD": [1000000, 0,0,0,0,0,0,0,0,0,  1.5487, 0.1785,  1.28, 0.6319,  1.7104],
        "Pgam": [1000000, 0,0,0,0,0,0,0,0,0,  0.8048,  1.5248,   0.6648, 0.6832,  1.493],
        "Pbeta": [1000000, 0,0,0,0,0,0,0,0,0,  -0.194,  -0.2019,  -0.0301, -0.2383,  -0.3253],
        "Pphi": [1000000, 0,0,0,0,0,0,0,0,0,  -1.7437,  -0.5516,  -0.0634, -2.1341,  -1.777],
        "OD": [1000000, 0,0,0,0,0,0,0,0,0,  0.6366, 0.3876,  0.9437, 0.3976,  0.9626],
        "Ogam": [1000000, 0,0,0,0,0,0,0,0,0,  1.5444,   0.6627, 0.763, 0.6587,  1.4327],
        "Obeta": [1000000, 0,0,0,0,0,0,0,0,0, -0.4555,  -0.4046,  -0.0869, -0.4415,  -0.5743],
        "Ophi": [1000000, 0,0,0,0,0,0,0,0,0,  -1.169,  -0.894,  -0.04, -2.072,  -1.306]}

    # Newton solver at each temperature of interest 
    error = 999
    while error > 0.01:
    #for i in range(200):
        helmholtz_output_para = Helmholtz(rho_guess, T, Tc, rho_c, Rs, 'P', coefficients)
        helmholtz_output_ortho = Helmholtz(rho_guess, T, Tc, rho_c, Rs, 'O', coefficients)
        
        alpha_r_delta = helmholtz_output_para[7]*paraPercent + helmholtz_output_ortho[7]*(1-paraPercent)
        delta = rho_guess / rho_c
        P_guess = rho_guess * Rs * T * (1 + delta * alpha_r_delta)

        rho_high = rho_guess + d_rho
        helmholtz_output_para = Helmholtz(rho_high, T, Tc, rho_c, Rs, 'P', coefficients)
        helmholtz_output_ortho = Helmholtz(rho_high, T, Tc, rho_c, Rs, 'O', coefficients)
        alpha_r_delta = helmholtz_output_para[7]*paraPercent + helmholtz_output_ortho[7]*(1-paraPercent)
        delta = rho_high / rho_c
        P_guess_high = rho_high * Rs * T * (1 + delta * alpha_r_delta)

        rho_low = rho_guess - d_rho
        helmholtz_output_para = Helmholtz(rho_low, T, Tc, rho_c, Rs, 'P', coefficients)
        helmholtz_output_ortho = Helmholtz(rho_low, T, Tc, rho_c, Rs, 'O', coefficients)
        alpha_r_delta = helmholtz_output_para[7]*paraPercent + helmholtz_output_ortho[7]*(1-paraPercent)
        delta = rho_low / rho_c
        P_guess_low = rho_low * Rs * T * (1 + delta * alpha_r_delta)

        derivative = (P_guess_high - P_guess_low) / (2 * d_rho)

        rho_new = rho_guess + (P_desired - P_guess) / derivative
        error = abs(rho_guess - rho_new)
        rho_guess = rho_new
        

    helmholtz_output_para = Helmholtz(rho_guess, T, Tc, rho_c, Rs, 'P', coefficients)
    helmholtz_output_ortho = Helmholtz(rho_guess, T, Tc, rho_c, Rs, 'O', coefficients)
    Cp = (helmholtz_output_para[5]*paraPercent + helmholtz_output_ortho[5]*(1-paraPercent))*1E3   # Specific heat at constant pressure [J/(kg*K)]
    Cv = (helmholtz_output_para[6]*paraPercent + helmholtz_output_ortho[6]*(1-paraPercent))*1E3    # Specific heat at constant volume [J/(kg*K)]
    h = helmholtz_output_para[0]*paraPercent + helmholtz_output_ortho[0]*(1-paraPercent)    # Enthalpy [kJ/kg]
    #S = helmholtz_output_para[3]*paraPercent + helmholtz_output_ortho[3]*(1-paraPercent)    # Entropy [kJ/(kg * K)]
    

    return [h, rho_guess, Cp, Cv]

def main():
    # Call the required function 
    #"n": [1, 0.398377, -1.84616, 0.418347, 0.023706, 0.097717, 0.030179, 0.022734, 0.013573, -0.04053, 0.000545, 0.000511,	2.95E-07, -8.7E-05,	-0.21271, 0.087359, 0.127551, -0.09068, -0.0354, -0.03623, 0.013277, -0.00033, -0.00831, 0.002125, -0.00083, -2.6E-05, 0.0026, 0.009985, 0.0022, 0.025914, -0.12596, 0.147836, -0.01011],
    #"k": [1, -0.00074, -6.6E-05, 2.50042, -21.4487, 1.01258, -0.94437, 14.5066, 74.9148, 4.14817]

    


    #hydrogen_debugging(coeff)

    # Call the required function
    P_desired = 20E6 # Pa
    T = 988.50 # K
    rho_guess = 80 # kg/m^3

    # For this Temp grab the ortho/para percentage
    paraPercent = para_fraction(T)/100

    h, rho, Cp, Cv = hydrogen_thermodynamics(P_desired, rho_guess, paraPercent, T)

    print(f'Temperature: {T} K \n Pressure: {P_desired} Pa \n Density: {rho} kg/m^3 \n Enthalpy: {h} kJ/kg  \n Isobaric Heat Capacity: {Cp} kJ/(kg*K) \n Isochoric Heat Capacity: {Cv} kJ/(kg*K)')
    #paraTemp, paraPercent = paraPercentFunction()
    
    # print(para_fraction(150))
    
    plt.show()

    return None


main()