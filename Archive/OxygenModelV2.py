# Oxygen model V2 used to calculate density of liquid Hydrogen
# By Jacob Saunders
# Based off www.cryo-rocket.com section 2.3

import numpy as np
import matplotlib.pyplot as plt



def calc_alpha0(tau, delta, delta0, coef):
    # Function to calculate the ideal component of the Helmholtz derivative
    # Inputs
    # tau: recipricol reduced temperature
    # delta: reduced density 
    # coef: dictionary containing the coefficients for the model

    return (coef["k"][1]*(tau**1.5) + coef['k'][2]*(1/(tau**2)) + coef['k'][3]*np.log(tau) + coef['k'][4]*tau + coef['k'][5]*np.log(np.exp(coef['k'][7]*tau) - 1) + coef['k'][9] + np.log(delta/delta0))

def calc_alpha_r(tau, delta, coef):
    # Function to calculate the real component of the Helmholtz derivative
    # Inputs:
    # tau: reciprocal reduced temperature 
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model
    alpha_r = 0
    sumTwo = 0
    sumThree = 0 
    for i in range(len(coef['r'])):

        if i >= 1 and i <= 13:
            alpha_r += (coef['n'][i]*(delta**coef['r'][i])*(tau**coef['s'][i])  )
    
        if i >= 14 and i <= 24:
            alpha_r +=  np.exp(-delta**2)*(coef['n'][i]*(delta**coef['r'][i])*(tau**coef['s'][i]) )
            
        if i >= 25 and i <= 35:
            alpha_r += np.exp(-delta**4)*(coef['n'][i]*(delta**coef['r'][i])*(tau**coef['s'][i]))
    
    return alpha_r

# Define derivates of the above helmholtz functions
def calc_alpha_r_delta(tau, delta, coef):
    # First partial derivative of alpha_r with respect to delta
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model

    alpha_r_delta = 0
    sumTwo = 0
    sumThree = 0
    for i in range(len(coef['r'])):
        
        # iteratre through the 3 sum functions
        if i >= 1 and i <= 13:
            alpha_r_delta += ( coef['n'][i]*coef['r'][i]*(delta**(coef['r'][i]-1))*(tau**coef['s'][i]) )
        if i >= 14 and i <= 24:
            alpha_r_delta += np.exp(-delta**2)*( coef['n'][i] * (coef['r'][i] * (delta**(coef['r'][i]-1)) - 2*(delta**(coef['r'][i]+1))) * (tau**coef['s'][i]) )
        if i >= 25 and i <= 32: 
            alpha_r_delta += np.exp(-delta**4)*( coef['n'][i] * (coef['r'][i] * (delta**(coef['r'][i] -1)) - 4*(delta**(coef['r'][i]+3)) ) * (tau**coef['s'][i]))

    return alpha_r_delta #+ sumTwo*np.exp(-delta**2) + sumThree*np.exp(-delta**4)

def calc_alpha_r_tau(tau, delta, coef):
    # First partial derivative of alpha_r with respect to tau
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model

    alpha_r_tau = 0
    sumTwo = 0
    sumThree = 0

    for i in range(len(coef['r'])):

        if i >= 1 and i <= 13:
            alpha_r_tau += ( coef['n'][i] * coef['s'][i] * (delta**coef['r'][i]) * (tau**(coef['s'][i]-1)) )
        if i >= 14 and i <= 24:
            sumTwo += ( coef['n'][i] * coef['s'][i] * (delta**coef['r'][i]) * (tau**(coef['s'][i]-1)) )
        if i >= 25 and i <= 35:
            sumThree += ( coef['n'][i] * coef['s'][i] * (delta**coef['r'][i]) * (tau**(coef['s'][i]-1)) )
    return alpha_r_tau + sumTwo*np.exp(-delta**2) + sumThree*np.exp(-delta**4)

def calc_alpha0_tau(tau, delta, coef):
    # First partial derivative of alpha0 with respect to tau
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model

    return ( 1.5*coef['k'][1]*(tau**0.5) - 2*coef['k'][2]*(1/tau**3) + coef['k'][3]*(1/tau) + coef['k'][4] + coef['k'][5] * ((coef['k'][7]*np.exp(coef['k'][7]*tau))/(np.exp(coef['k'][7]*tau) - 1)) - coef['k'][6]*((0.66*coef['k'][8]*np.exp(-coef['k'][8]*tau))/(1+0.66*np.exp(-coef['k'][8]*tau))) )

def calc_alpha0_tau_tau(tau, delta, coef):
    # Function to calculate the second partial derivative of alpha0 with respect to tau
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model

    return ( 0.75*coef['k'][1]*(tau**(-0.5)) + 6*coef['k'][2]*(tau**(-4)) - coef['k'][3]*(tau**(-2)) - coef['k'][5]*(( (coef['k'][7]**2)*np.exp(coef['k'][7]*tau))/((np.exp(coef['k'][7]*tau)-1)**2)) + coef['k'][6]*((0.66*coef['k'][8]**2*np.exp(-coef['k'][8]*tau))/((1+0.66*np.exp(-coef['k'][8]*tau))**2)) )

def calc_alpha_r_tau_tau(tau, delta, coef):
    # Function to calculate the second partial derivative of alpha_r with respect to tau
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model

    alpha_r_tau_tau = 0
    sumTwo = 0
    sumThree = 0
    for i in range(len(coef['r'])):

        if i >= 1 and i <= 13:
            alpha_r_tau_tau += ( coef['n'][i]*coef['s'][i]*(coef['s'][i]-1)* delta**(coef['r'][i]) * tau**(coef['s'][i]-2)  )
        if i >= 14 and i <= 24:
            alpha_r_tau_tau += np.exp(-delta**2)* ( coef['n'][i] * coef['s'][i] * (coef['s'][i]-1) * delta**(coef['r'][i]) * tau**(coef['s'][i]-2)  )
        if i >= 25 and i <= 35:
            alpha_r_tau_tau += np.exp(-delta**4)*( coef['n'][i] * coef['s'][i] * (coef['s'][i]-1) * delta**(coef['r'][i]) * tau**(coef['s'][i]-2))
    
    return alpha_r_tau_tau #+ sumTwo*np.exp(-delta**2) + sumThree*np.exp(-delta**4)

def calc_alpha_r_delta_delta(tau, delta, coef):
    # Function to calculate the second partial derivative of alpha_r with respect to delta
    #   Inputs:
    #   tau: reciprocal reduced temperature
    #   delta: reduced density
    #   coef: dictionary containing the coefficients for the model
    # coef['n'][i]*coef['r'][i]*(coef['r'][i]-1)* delta**(coef['r'][i]-2) * tau**(coef['s'][i])

    alpha_r_delta_delta = 0
    sumTwo = 0
    sumThree = 0
    for i in range(len(coef['r'])):

        if i >= 1 and i <= 13:
            alpha_r_delta_delta += ( coef['n'][i] * coef['r'][i] * (coef['r'][i] - 1) * delta**(coef['r'][i] -2) * tau**(coef['s'][i]) )
        if i >= 14 and i <= 24:
            alpha_r_delta_delta += np.exp(-delta**2)* (coef['n'][i] * (coef['r'][i] * (coef['r'][i] - 1) * delta**(coef['r'][i] - 2) - 2*(2*coef['r'][i] + 1) * delta**(coef['r'][i]) + 4*delta**(coef['r'][i] +2)) * tau**(coef['s'][i]))
        if i >= 25 and i <= 35:
            alpha_r_delta_delta += np.exp(-delta**4)*( coef['n'][i] * (coef['r'][i] * (coef['r'][i] - 1) * delta**(coef['r'][i] - 2) - 4*(2*coef['r'][i] + 3)*delta**(coef['r'][i] + 2) + 16*delta**(coef['r'][i] + 6)) * tau**(coef['s'][i]))
    
    return alpha_r_delta_delta #+ sumTwo*np.exp(-delta**2) + sumThree*np.exp(-delta**4)

def calc_alpha_r_delta_tau(tau, delta, coef):
    # Function to calculate the mixed partial derivative of alpha_r with respect to delta and tau
    # Inputs:
    # tau: reciprocal reduced temperature
    # delta: reduced density
    # coef: dictionary containing the coefficients for the model

    alpha_r_delta_tau = 0
    sumTwo = 0
    sumThree = 0

    for i in range(len(coef['r'])):
        if i >= 1 and i <= 13:
            alpha_r_delta_tau += ( coef['n'][i]*coef['r'][i]*coef['s'][i]*delta**(coef['r'][i]-1) * tau**(coef['s'][i] -1) )
        if i >= 14 and i <= 24:
            alpha_r_delta_tau += np.exp(-delta**2)*( coef['n'][i]* (coef['r'][i]*delta**(coef['r'][i] - 1) - 2*delta**(coef['r'][i] + 1)) * coef['s'][i] * tau**(coef['s'][i] - 1) )
        if i >= 25 and i <= 35:
            alpha_r_delta_tau += np.exp(-delta**4)*( coef['n'][i] * (coef['r'][i]*delta**(coef['r'][i]-1) - 4*delta**(coef['r'][i] + 3)) * coef['s'][i]*tau**(coef['s'][i] - 1) )

    return alpha_r_delta_tau #+ sumTwo*np.exp(-delta**2) + sumThree*np.exp(-delta**4)

def Helmholtz(rho_guess, T, Tc, rho_c, Rs, coefficients):
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
    delta0 = p0/(Rs*T0)

    # Convert temperature and density to reduced variables
    delta = rho_guess / rho_c
    tau = Tc / T

    # Calculate alpha0 
    alph0 = calc_alpha0(tau, delta, delta0, coefficients)

    # Calculate alpha0_tau
    alpha0T = calc_alpha0_tau(tau, delta, coefficients)

    # Calculate alpha0_tau_tau
    alph0TT = calc_alpha0_tau_tau(tau, delta, coefficients)

    # Calculate alpha_r
    alphR = calc_alpha_r(tau, delta, coefficients)

    # Calculate alpha_r_delta
    alphRD = calc_alpha_r_delta(tau, delta, coefficients)

    # Calculate alpha_r_tau
    alphRT = calc_alpha_r_tau(tau, delta, coefficients)
    
    # Calculate alpha_r_tau_tau
    alphRTT = calc_alpha_r_tau_tau(tau, delta, coefficients)

    # Calculate alpha_r_delta_delta
    alphRDD = calc_alpha_r_delta_delta(tau, delta, coefficients)

    # Calculate alpha_r_delta_tau
    alphRDT = calc_alpha_r_delta_tau(tau, delta, coefficients)


    # Calculate enthalpy [kJ/kg]
    h = (1 + tau*(alpha0T + alphRT) + delta*alphRD) * Rs*T 

    # Calculate pressure [Pa]
    P = rho_guess * Rs * T * (1 + delta * alphRD)

    # Calculate entropy [kJ/kgK]
    s = (tau*(alpha0T + alphRT) - alph0 - alphR)* Rs 

    # Calculate isochoric heat capactiy [kJ/kg K]
    Cv = -(tau**2) * (alph0TT + alphRTT)

    # Calculate isobaric heat capactiy [kJ/kg K]
    Cp =  ((1 + delta*alphRD - delta*tau*alphRDT)**2)/(1 + 2*delta*alphRD + (delta**2) * alphRDD) + Cv

    return [h, P, T, s, rho_guess, Cp*Rs*0.001, Cv*Rs*0.001, alphRD, alph0TT, alphRTT, alphRDD, alphRDT]

def oxygen_debugging(coefficients):
    # Constants
    Tc = 154.581  # O2 critical temperature [K]
    Pc = 5043.0 * 1000  # O2 critical pressure [Pa]
    rho_c = 13.63 * 31.9988  # critical density [kg/m^3]
    M = 31.9988  # O2 molecular weight [g/mol]
    R = 8314.472  # universal gas constant
    Rs = R / M  # Specific gas constant

    # Desired pressure
    # P_desired = 1.25 * Pc
    P_array = [1.25*Pc]

    # Initial guess for rho
    rho_guess = 1200
    d_rho = 0.0001  # delta rho used for derivative

    plot_T = []
    plot_Cp = []
    plot_h = []
    plot_alpha_r_delta = []
    plot_Cv = []
    plot_alpha0_tau_tau = []
    plot_alpha_r_tau_tau = []
    plot_alpha_r_delta_delta = []
    plot_alpha_r_delta_tau = []
    
    plot_delta = []
    for P_desired in P_array:
        for T in range(54, int(6 * Tc) + 1):

            # Newton solver at each temperature of interest 
            error = 999
            while error > 1E-10:
            #for i in range(200):
                helmholtz_output = Helmholtz(rho_guess, T, Tc, rho_c, Rs, coefficients)

                alpha_r_delta = helmholtz_output[7]
                delta = rho_guess / rho_c
                P_guess = rho_guess * Rs * T * (1 + delta * alpha_r_delta)

                rho_high = rho_guess + d_rho
                helmholtz_output = Helmholtz(rho_high, T, Tc, rho_c, Rs, coefficients)
                alpha_r_delta = helmholtz_output[7]
                delta = rho_high / rho_c
                P_guess_high = rho_high * Rs * T * (1 + delta * alpha_r_delta)

                rho_low = rho_guess - d_rho
                helmholtz_output = Helmholtz(rho_low, T, Tc, rho_c, Rs, coefficients)
                alpha_r_delta = helmholtz_output[7]
                delta = rho_low / rho_c
                P_guess_low = rho_low * Rs * T * (1 + delta * alpha_r_delta)

                derivative = (P_guess_high - P_guess_low) / (2 * d_rho)

                rho_new = rho_guess + (P_desired - P_guess) / derivative
                error = abs(rho_guess - rho_new)
                rho_guess = rho_new

            helmholtz_output = Helmholtz(rho_guess, T, Tc, rho_c, Rs, coefficients)
            Cp = helmholtz_output[5]
            # h, P, T, s, rho_guess, Cp, Cv, alphRD, alphRDD, alphRDT, alphRT, alphRTT, alph0, alpha0T, alph0TT, alphR
            plot_Cp.append(Cp)
            plot_T.append(T)
            plot_h.append(helmholtz_output[0])
            plot_alpha_r_delta.append(helmholtz_output[7])
            plot_alpha0_tau_tau.append(helmholtz_output[8])
            plot_alpha_r_tau_tau.append(helmholtz_output[9])
            plot_delta.append(rho_guess)
            plot_Cv.append(helmholtz_output[6])
            plot_alpha_r_delta_delta.append(helmholtz_output[10])
            plot_alpha_r_delta_tau.append(helmholtz_output[11])

        plt.figure(1)
        plt.plot(plot_T, plot_Cp, '-b', linewidth=2)
        plt.xlabel('Temperature [K]')
        plt.ylabel('Isobaric Heat Capacity [J/(kg*K)]')
        plt.title('Isobaric Heat Capacity vs. Temperature')
        plt.legend(['P=1.25Pc'])
        plt.grid(True)
        

        plt.figure(2)
        plt.plot(plot_T, plot_alpha_r_delta, '-b', linewidth=2)
        plt.xlabel('Temperature [K]')
        plt.ylabel(r'$\alpha_\delta^r$')
        plt.title(r'$\alpha_\delta^r$ vs. Temperature')
        plt.legend(['P=1.25Pc'])
        plt.grid(True)

        plt.figure(3)
        plt.plot(plot_T, plot_h, '-b', linewidth=2)
        plt.xlabel('Temperature [K]')
        plt.ylabel('Enthalpy [kJ/kg]')
        plt.title('Enthalpy [kJ/kg] vs. Temperature')
        plt.legend(['P=1.25Pc'])
        plt.grid(True)

        plt.figure(4)
        plt.plot(plot_T, plot_delta, '-b', linewidth=2)
        plt.xlabel('Reduced Temperature')
        plt.ylabel('Reduced Density')
        plt.title('Reduced Density vs. Reduced Temperature')
        plt.legend(['P=1.25Pc'])
        plt.grid(True)

        plt.figure(5)
        plt.plot(plot_T, plot_Cv, '-b', linewidth=2)
        plt.xlabel('Temperature [K]')
        plt.ylabel('Isobaric Heat Capacity [J/(kg*K)]')
        plt.title('Isobaric Heat Capacity vs. Temperature')
        plt.legend(['P=1.25Pc'])
        plt.grid(True)

        plt.figure(6)
        plt.plot(plot_T, plot_alpha0_tau_tau, '-b', linewidth=2)
        plt.xlabel('Temperature [K]')
        plt.ylabel('alpha0_tau_tau')
        plt.title('alpha0_tau_tau vs. Temperature')
        plt.legend(['P=1.25Pc'])
        plt.grid(True)

        plt.figure(7)
        plt.plot(plot_T, plot_alpha_r_tau_tau, '-b', linewidth=2)
        plt.xlabel('Temperature [K]')
        plt.ylabel('alpha_r_tau_tau')
        plt.title('alpha_r_tau_tau vs. Temperature')
        plt.legend(['P=1.25Pc'])
        plt.grid(True)

        plt.figure(8)
        plt.plot(plot_T, plot_alpha_r_delta_delta, '-b', linewidth=2)
        plt.xlabel('Temperature [K]')
        plt.ylabel('alpha_r_delta_delta')
        plt.title('alpha_r_delta_delta vs. Temperature')
        plt.legend(['P=1.25Pc'])
        plt.grid(True)

        plt.figure(9)
        plt.plot(plot_T, plot_alpha_r_delta_tau, '-b', linewidth=2)
        plt.xlabel('Temperature [K]')
        plt.ylabel('alpha_r_delta_tau')
        plt.title('alpha_r_delta_tau vs. Temperature')
        plt.legend(['P=1.25Pc'])
        plt.grid(True)

# Example call to the function
# coefficients = ...  # Define or load the coefficients
# oxygen_debugging(coefficients)

def main():
    # Call the required function 
    #"n": [1, 0.398377, -1.84616, 0.418347, 0.023706, 0.097717, 0.030179, 0.022734, 0.013573, -0.04053, 0.000545, 0.000511,	2.95E-07, -8.7E-05,	-0.21271, 0.087359, 0.127551, -0.09068, -0.0354, -0.03623, 0.013277, -0.00033, -0.00831, 0.002125, -0.00083, -2.6E-05, 0.0026, 0.009985, 0.0022, 0.025914, -0.12596, 0.147836, -0.01011],
    #"k": [1, -0.00074, -6.6E-05, 2.50042, -21.4487, 1.01258, -0.94437, 14.5066, 74.9148, 4.14817]}

    # Define the coefficients from the oxygen model table 2.3.2 in the book ignore first element place holder to shift dict index by 1
    coeff = {
        "r": [1, 1, 1, 1,	2,	2,	2,	3,	3,	3,6,	7,	7,	8,	1,	1,	2,	2,	3,	3,	5,	6,	7,	8,	10,	2,	3,	3,	4,	4,	5,	5,	5],
        "s": [1, 0, 1.5, 2.5, -0.5, 1.5, 2, 0, 1, 2.5, 0, 2, 5, 2, 5, 6, 3.5, 5.5, 3, 7, 6, 8.5, 4, 6.5, 5.5, 22, 11, 18, 11, 23, 17, 18, 23],  
    
        "n": [1, 0.3983768749, -0.1846157454E01, 0.4183473197, 0.2370620711E-1, 0.9771730573E-1, 0.3017891294E-1, 0.2273353212E-1, 0.1357254086E-1, -0.4052698943E-1, 0.5454628515E-3, 0.5113182277E-3,	0.2953466883E-6, -0.8687645072E-4,	-0.2127082589, 0.8735941958E-1, 0.1275509190, -0.9067701064E-1, -0.3540084206E-1, -0.3623278059E-1, 0.1327699290E-1, -0.3254111865E-3, -0.8313582932E-2, 0.2124570559E-2, -0.8325206232E-3, -0.2626173276E-4, 0.2599581482E-2, 0.9984649663E-2, 0.2199923153E-2, -0.2591350486E-1, -0.1259630848, 0.1478355637, -0.1011251078E-1],
       
        "k": [1, -0.740775E-3, -0.664930E-4, 2.50042, -0.214487E2, 0.101258E1, -0.944365, 0.145066E2, 74.9148, 4.14817]}


    oxygen_debugging(coeff)
    plt.show()

    return None


main()