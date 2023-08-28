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


# Define global constants
critical_density = 31.112 #[kg/m^3]
criitcal_temperature = 32.938 #[K]
critical_pressure = 1.284 #[MPa] Note 1MPa = 10Bar


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
    # Dictionary contants coefficient name as key follow by list of values for Parahydrogen 
    # Note if value is 0 indicates it does not apply
    para_coeff = {
        'a_k':[-1.4485891134, 1.884521239, 4.30256, 13.0289,-47.7365, 50.0013, -18.6261, 0.993973, 0.536078],
        'b_k':[0,0,-15.1496751472, -25.0925982148,-29.4735563787,-35.405914117,-40.724998482,-163.7925799988,-309.2173173842],
        'N_i':[-7.33375,0.01,2.60375,4.66279,0.682390,-1.47078,0.135801,-1.05327,0.328239,-0.0577833,0.0449743,0.0703464,-0.0401766,0.119510],
        't_i':[0.6855,1,1,0.489,0.774,1.133,1.386,1.619,1.162,3.96,5.276,0.99,6.791,3.19],
        'd_i':[1,4,1,1,2,2,3,1,3,2,1,3,1,1],
        'p_i':[0,0,0,0,0,0,0,1,1,0,0,0,0,0],
        'phi_i':[-1.7437,-0.5516,-0.0634,-2.1341,-1.777],
        'Beta_i':[-0.194,-0.2019,-0.0301,-0.2383,-0.3253],
        'gamma_i':[0.8048,1.5248,0.6648,0.6832,1.493],
        'D_i':[1.5487,0.1785,1.28,0.6319,1.7104]
    }

    ortho_coeff = {
        'a_k':[-1.4675442336,1.8845068862,2.54151,-2.3661,1.00365,1.22447,0,0,0],
        'b_k':[0,0,-25.7676098736,-43.4677904877,-66.0445514750,-209.7531607465,0,0,0],
        'N_i':[-6.83148,0.01,2.11505,4.38353,0.211292,-1.00939,0.142086,-0.87696,0.804927,-0.710775,0.0639688,0.0710858,-0.087654,0.647088],
        't_i':[0.7333,1,1.1372,0.5136,0.5638,1.6248,1.829,2.404,2.105,4.1,7.658,1.259,7.589,3.946],
        'd_i':[1,4,1,1,2,2,3,1,3,2,1,3,1,1],
        'p_i':[0,0,0,0,0,0,0,1,1,0,0,0,0,0],
        'phi_i':[-1.169,-0.894,-0.04,-2.072,-1.306],
        'Beta_i':[-0.4555,-0.4046,-0.0869,-0.4415,-0.5743],
        'gamma_i':[1.5444,0.6627,0.763,0.6587,1.4327],
        'D_i':[0.6366,0.3876,0.9437,0.3976,0.9626]
    }


    tau = temp/criitcal_temperature
    delta = density/critical_density

    paraN = 9 # number of steps for k if parahydrogen
    orthoN = 6 # number of steps for k if orthohydrogen

    N = np.arange(3,paraN,1)

    idealSum = 0 # Sum of k=3 to N value

    for k in N:
        if len(N) == paraN-3:
            idealSum[k] = idealSum[k] + para_coeff['a_k'][k]*np.log[1-np.exp(para_coeff['b_k'][k]*tau)]
        else:
            idealSum[k] = idealSum[k]

    ideal_component = np.log(delta) + 1.5*np.log(tau) + para_coeff['a_k'][0] + para_coeff['a_k'][1]*tau + idealSum





    return None


main()




