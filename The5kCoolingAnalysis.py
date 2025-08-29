import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as scipy
from scipy.interpolate import interp1d
from adjustText import adjust_text
from CoolProp.CoolProp import PropsSI
import cantera as ct
from HydrogenModelV2 import hydrogen_thermodynamics
from HydrogenModelV2 import para_fraction

# This is based off the MixtureOptimization.py code but modified directly to work with a new engine targetting 5kN LOX/RP-1 

# Import excel file
# df = pd.read_excel('CEAParsed.xlsx')

# Globals
Ru = 8.31446261815324 # Univeral gas constant [J/mol*K]
specificGasConstant = Ru/0.013551 # Specifc gas constant [J/Kmol] (named wrong but cant be arsed to change it)

# Function to create plots
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
    plt.grid(True)
    texts = [
        plt.annotate(f"{x_axis[0]}, {y_axis[0]}", xy=(x_axis[0], y_axis[0]), ha='center', bbox=dict(boxstyle="round", fc="w")),
        plt.annotate(f"{x_axis[-1]}, {y_axis[-1]}", xy=(x_axis[-1], y_axis[-1]), ha='center', bbox=dict(boxstyle="round", fc="w"))
    ]
    adjust_text(texts)

# Function to Calculate the mach number at the exit of the nozzle
def calc_exit_mach_num(area_ratio, gam, M_e):
    # Inputs:
    # area_ratio_array - type: array - the area ratio of the nozzle (Ae/At)
    # gam - type: float - specific heat ratio
    # M_e - type: float - exit mach number initial guess

    def equation(Me, gam, area_ratio):
        #return ((specific_heat_ratio+1)/2)**(-c) * ((1 + ( (specific_heat_ratio-1)/2) * (M_e**2))/M_e)**c - area_ratio
        # return  (((gam+1)/2)**(-((gam+1)/(gam-1)/2)))* (1 / Me) * (1 + Me**2 * (gam-1)/2)**((gam+1)/(gam-1)/2) - area_ratio
        return 1/Me*( (1 + ((gam-1)/2) * Me**2)/(1 + (gam-1)/2) ) ** ((gam+1)/(2*(gam-1))) - area_ratio
        
    return scipy.fsolve(equation, M_e, args=(gam, area_ratio))           
    
# Function to calculate the exit pressure of the nozzle
def calc_exit_pressure(P_c, gam, M_e):
    # Inputs:
    # P_c - type: float - chamber stagnation pressure [Pa]
    # gam - type: float - specific heat ratio
    # M_e - type: float - exit mach number

    return ( P_c * (1 + ( (gam-1)/2 ) * (M_e**2))**(-gam/(gam-1)) )

# Function to calculate the thrust coefficient of the nozzle
def calc_thrust_coeff(P_c, P_e, gam, area_ratio):
    # Inputs:
    # P_c - type: float - chamber pressure [Pa]
    # P_e - type: float - exit pressure [Pa]
    # gam - type: float - specific heat ratio
    # M_e - type: float - exit mach number

    return np.sqrt( (2*(gam**2))/(gam-1) * (2/(gam+1))**((gam+1)/(gam-1)) * (1 - (P_e/P_c)**((gam-1)/gam)) ) + area_ratio*(P_e/P_c) 

# Function to calculate the nozzle throat area
def calc_nozzle_throat_area(F_Vac, C_F, P_c):
    # Inputs:
    # F - type: float - thrust force [N]
    # C_F_array - type: array - vacuum thrust coefficient [Pa]
    # P_c - type: float - chamber pressure [Pa]

    return (F_Vac / (C_F * P_c))

# Function to calculate nozzle throat temperature 
def calc_nozzle_throat_temp(T_c, gam):
    # Inputs:
    # T_c - type: float - chamber temperature [K]
    # gam - type: float - specific heat ratio

    return (T_c * (1 + ((gam-1)/2))**-1)

# Function to calculate nozzle throat pressure
def calc_nozzle_throat_pressure(P_c, gam):
    # Inputs:
    # P_c - type: float - chamber pressure [Pa]
    # gam - type: float - specific heat ratio

    return (P_c * (1 + ((gam-1)/2))**(-gam/(gam-1)))

# Function to calculate chamber area
def calc_chamber_area(A_t):
    # Inputs:
    # A_t - type: float - throat area [m^2]
    # Need to convert to cm^2
    A_t_cm = A_t * 100**2

    D_t_cm = 2*np.sqrt(A_t_cm/np.pi)
    
    # Retrun chamber area in m^2
    A_c_cm = (8*(D_t_cm**(-0.6)) + 1.25)*A_t_cm
    A_c = A_c_cm / 100**2 # Convert to m^2 # Convert to mm^2

    return A_c

# Function to calculate chamber volume m3
def calc_chamber_volume(A_t, L_star):
    # Inputs:
    # A_t - type: float - throat area [m^2]
    # L_star - type: float - characteristic length [m]
  
    return A_t*L_star

# Function to calculate chamber length
def calc_chamber_length(A_c, V_c):
    # Inputs:   
    # A_c - type: float - chamber area [m^2]
    # V_c - type: float - chamber volume [m^3]
    return V_c/A_c

# Function which calculates general engine geometry based off CEA data
def engine_geometry(gam, P_c, P_amb, T_c, F_Vac, expExit, contChamber, L_star, Ru, Rd, alpha, vacuum_opt, showPlots):
    # Calculate the geometry of a rocket engine based off key parameters from CEA
    # Inputs:
    # df - type: data frame - dataframe containing the CEA data
    # gam - type: float - specific heat ratio
    # P_c - type: float - chamber stagnation pressure [Pa]
    # T_c - type: float - chamber stagnation temperature [K]
    # F_Vac - type: float - desired vacuum thrust [N]
    # targetPos - type: float - position in the area ratio array to target (e.g. 69.5 for RS25)
    #showPlots - type: bool - whether to show plots or not

    #Output:
    # A_c - type: float - chamber area [m^2]
    # A_t - type: float - throat area [m^2]
    # A_e - type: float - exit area [m^2]
    # L_c - type: float - chamber length [m]
    # Note: This function is based off the RS25 engine geometry and will need to be modified for other engines
    # create area_ratio_array from 2 to 80 in steps of 0.1
    area_ratio_array = np.arange(1.1, 80, 0.1)

    # First step is to calculate mach numbers in the exit
    M_e_array = [] # Create array to store exit mach numbers
    M_e = 1.2 # Initial guess for exit mach number

    # Second step is to calculate exit pressures
    P_e_array = [] # Create array to store exit pressures

    # Third step is to calculate the nozzle’s vacuum thrust coefficient
    C_F_array = [] # Create array to store vacuum thrust coefficients

    # Thrust coefficient at sea level
    # C_F_array_SL = [] # Create array to store sea level thrust coefficients

    # Fourth step is to calculate the nozzle's throat area 
    A_t_array = [] # Create array to store throat areas

    # Fifth step is to calculate the nozzle's exit area
    A_e_array = [] # Create array to store exit areas

    # Last step is to calculate mass flow rate with respect to area ratio
    m_dot_array = [] # Create array to store mass flow rates

    # Iterate through all area ratios 
    for area_ratio in area_ratio_array:
        # Calculate exit mach number for given area ratio
        M_e = calc_exit_mach_num(area_ratio, gam, M_e)
        M_e_array.append(M_e)

        # Calculate exit pressure for given area ratio
        P_e_array.append(calc_exit_pressure(P_c, gam, M_e))

        # Calculate vacuum thrust coeffiecient
        C_F_array.append(calc_thrust_coeff(P_c, P_e_array[-1], gam, area_ratio))

        # Calculate sea level thrust coefficient
        # C_F_array_SL.append(C_F_array[-1] - (P_amb/P_c)*expExit)

        # Calculate throat area
        # if vacuum_opt:
        A_t_array.append(calc_nozzle_throat_area(F_Vac, C_F_array[-1], P_c))
        # else:
            # A_t_array.append(calc_nozzle_throat_area(F_Vac, C_F_array_SL[-1], P_c))

        # Calculate exit area
        A_e_array.append(A_t_array[-1] * area_ratio)

        # Calculate throat temperature
        T_t = calc_nozzle_throat_temp(T_c, gam)

        # Calculate throat pressure
        P_t = calc_nozzle_throat_pressure(P_c, gam)

        # Calculate density at throat
        rho_t = P_t / (specificGasConstant * T_t)

        # Calculate sonic velocity at throat
        V_t = np.sqrt(gam * specificGasConstant * T_t)

        # Calculate mass flow rate
        m_dot_array.append(rho_t*V_t*A_t_array[-1])
    
    # Find the area ratio for the RS25 engine
    position_65 = np.where(np.isclose(area_ratio_array, expExit))[0] # 69.5 is area ratio for RS25 engine select based off new engine design
    
    A_t = A_t_array[position_65[0]] # Throat area for engine [m^2]
    R_t = np.sqrt(A_t/np.pi) # Throat Radius [m]


    # Determine chamber volume 
    V_c = A_t*(L_star*0.0254) # Calculate chamber volume [m^3] ()
    A_c = A_t * contChamber #calc_chamber_area(A_t) # m^2
    L_c = V_c/A_c #calc_chamber_length(A_c, V_c) # Calculate chamber length [m]
    # Determine chamber area

    # Determine chamber length from Injector to throat

    # Determine length from throat to exit plane
    R_U = R_t*Ru
    L_e = 0.8*(R_t * (np.sqrt(expExit)-1) + R_U*(1/np.cos(np.deg2rad(alpha-1))))/(np.tan(np.deg2rad(alpha)))

    # Determine throat area for RS25
    A_t = A_t_array[position_65[0]]
    
    # Calculate L*
    # L_star = V_c/A_t

    # Determine exit area for RS25
    A_e = A_e_array[position_65[0]]

    # Determine the massflow rate for RS25
    m_dot = m_dot_array[position_65[0]]

    debugInfo = False # Set to true to print debug info
    if debugInfo:
        print(f'INPUTS: \n Gamma {gam} \n Chamber Pressure: {P_c} [Pa] \n Chamber Temperature: {T_c} [K] \n Vacuum Thrust: {F_Vac} [N]')
        print(f' Target Expansion Ratio {expExit} \n Exit Diamter: {2*np.sqrt(A_e/np.pi)} m \n Throat Diameter {2*np.sqrt(A_t/np.pi)} m \n Chamber Diameter: {2*np.sqrt(A_c/np.pi)} m \n Chamber Length: {L_c} m \n L* {L_star} \n Chamber Volume: {V_c} m3')
        # print(f'Exit Mach Num: {M_e_array[position_65[0]]} \n Exit Pressure: {P_e_array[position_65[0]]} [Pa] \n Thrust Coefficient Vac: {C_F_array[position_65[0]]} [Pa] \n Throat Area: {A_t_array[position_65[0]]} m2 \n Exit Area: {A_e_array[position_65[0]]} m2 \n Mass Flow Rate: {m_dot_array[position_65[0]]} [Kg/s]')
    showPlots = False # Set to true to show plots
    if showPlots:
        create_plot(area_ratio_array, M_e_array, 'Area Ratio', 'Exit Mach Number', 'Exit Mach Number vs Area Ratio')
        create_plot(area_ratio_array, P_e_array, 'Area Ratio', 'Exit Pressure [Pa]', 'Exit Pressure vs Area Ratio')
        create_plot(area_ratio_array, C_F_array, 'Area Ratio', 'Thrust Coefficient [Pa]', 'Thrust Coefficient vs Area Ratio')
        create_plot(area_ratio_array, A_t_array, 'Area Ratio', 'Throat Area [m^2]', 'Throat Area vs Area Ratio')
        create_plot(area_ratio_array, A_e_array, 'Area Ratio', 'Exit Area [m^2]', 'Exit Area vs Area Ratio')
        create_plot(area_ratio_array, m_dot_array, 'Area Ratio', 'Mass Flow Rate [Kg/s]', 'Mass Flow Rate vs Area Ratio')
    return (A_c, A_t, A_e, L_c, L_e, m_dot, V_c) 

# Function to display engine geometry 2D
def displayEngineGeometry2(x_array, A_c, A_t, A_e, L_c):
    # Inputs:
    # x - type: float - distance from injector [m]
    # A_c - type: float - chamber area [m^2]
    # A_t - type: float - throat area [m^2]
    # A_e - type: float - exit area [m^2]
    # L_c - type: float - chamber length [m]

    # Call calc_radius to calculate radius at each x
    r_array = [] # Create array to store chamber radius
    for x in x_array:
        r_array.append(calc_radius(x, A_c, A_t, A_e, L_c))  
    
    
    r_array = np.array(r_array)
    x_array = np.array(x_array)
    create_plot(x_array[:len(r_array)], r_array, 'Distance from Injector [m]', 'Radius [m]', 'Radius vs Distance from Injector')
    return None


# Function to calculate raduys of chamber based off x
def calc_radius(x, L_star, A_c, A_t, A_e, L_c, Ru, Rd, theta_1, theta_D, alpha):
    # Inputs:
    # x - type: float - distance from injector [m]
    # A_c - type: float - chamber area [m^2]
    # A_t - type: float - throat area [m^2]
    # A_e - type: float - exit area [m^2]
    # L_c - type: float - chamber length [m]
    # Ru - type: float - radius of contraction [m]
    # Rd - type: float - radius of throat curve [m]
    # theta1 - type: float - angle of contraction [degrees]
    # thetaD - type: float - angle of expansion [degrees]
    # thetaE - type: float - angle of exit [degrees]

    # Output:
    # r - type: float - radius of chamber at x [inch]

    D_c = 2*np.sqrt(A_c/np.pi) # Calculate chamber diameter [m]
    D_t = 2*np.sqrt(A_t/np.pi) # Calculate throat diameter [m]
    D_e = 2*np.sqrt(A_e/np.pi) # Calculate exit diameter [m]

    # Convert L_c to inch

    # Calculate expansion ratio
    expansion_ratio = A_e/A_t

    contraction_ratio = A_c/A_t # Calculate contraction ratio
    R_T = D_t/2 # Radius of throat [m]
    R_C = D_c/2 # Radius of chamber [m]

    # Calculate geometty
    R_U = R_T * Ru # Radius of contraction [m]
    R_D = R_T * Rd # Radius of throat curve [m]
    R_1 = R_T * Ru # Radius of contraction [m]
    R_E = np.sqrt(expansion_ratio)*R_T # Radius of expansion [m]

    # Calculate volume of chamber
    V_c = A_t*L_star # Volume of chamber [m^3]


    # calculate the length of the cone section 
    L_cone = (R_T*(np.sqrt(contraction_ratio)-1) + R_U*((1/np.cos(np.deg2rad(theta_1)))-1))/np.tan(np.deg2rad(theta_1))

    # Using cone frustume volume formula calculate the approximate cone volume
    V_cone = (np.pi/3) * L_cone*(R_C**2 + R_T**2 + R_T*R_C) # Volume of cone frustum [m^3]

    # Calculate the required volume for the culindrical chamber section
    V_cyl = V_c - V_cone # Volume of cylindrical section [m^3]

    # Calculate the length of the cylindrical section
    L_e = V_cyl/(contraction_ratio*A_t)

    L_c = L_cone + L_e # Total length of chamber from injector to throat [m]


    # Define key constants for RS25 engine @TODO need to replace with actual values for own engine
    # L_e = 5.339 # Length before chamber starts contracting [m]
    #theta_E = 5.3738 # Angle of exit [degrees]
    # R_1 = 1.73921 * R_T # Radius of contraction [m]
    #R_U = 0.494 * R_T # Radius of contraction [m]
    # R_U = 5.1527 # Small Throat RU
    # R_D  = 2.019 # Radius of sma;;throat curve [m]
    #R_D = 0.2 * R_T # Radius of throat curve [m]

    print(f'A_t [m^2]: {A_t} \n D_t [m] {D_t} \n R_t [m]: {R_T} \n D_e: {D_e} \n R_E [m]: {R_E} \n V_c [m^3]: {V_c} \n D_c [m]: {D_c} \n R_C [m]: {R_C} \n L_cone [m]: {L_cone} \n V_cone [m^3]: {V_cone} \n V_cyl [m^3]: {V_cyl} \n L_e [m]: {L_e} \n L_c [m]: {L_c}')
    # Calculate length from throat to exit plane
    L = 0.8*(R_T * (np.sqrt(expansion_ratio)-1) + R_U*(1/np.cos(np.deg2rad(alpha-1))))/(np.tan(np.deg2rad(15)))
    #L_N = 0.8 * ((np.sqrt(expansion_ratio)-1)*D_t_inch/2)/(np.tan(np.deg2rad(15)))
     
    # Constant def
    m = (R_T + R_U - D_c/2 + R_1 - (R_U+R_1)*np.cos(np.deg2rad(theta_1)))/(L_c - L_e - (R_U + R_1)*np.sin(np.deg2rad(theta_1)))
    
    if x <= L_e:
        r = D_c/2
    elif (L_e < x) and (x <= L_e + R_1*np.sin(np.deg2rad(theta_1))):
       
        r = np.sqrt(R_1**2 - (x-L_e)**2) + D_c/2 - R_1
    elif (L_e + R_1*np.sin(np.deg2rad(theta_1)) < x) and (x <= L_c - R_U*np.sin(np.deg2rad(theta_1))):
        r = (m*x + D_c/2 - R_1 + R_1*np.cos(np.deg2rad(theta_1)) - m*(L_e + R_1*np.sin(np.deg2rad(theta_1))))
    elif (L_c - R_U*np.sin(np.deg2rad(theta_1)) < x) and (x <= L_c):
        r = -np.sqrt(R_U**2 - (x-L_c)**2) + R_T + R_U
    elif (L_c < x) and (x <= L_c + R_D*np.sin(np.deg2rad(theta_D))):
        r = -np.sqrt(R_D**2 - (x - L_c)**2) + R_T + R_D
    else:
        # Calculate key constants
        N_x = R_D * np.cos(np.deg2rad(theta_D) - np.pi/2) 
        N_y = R_D * np.sin(np.deg2rad(theta_D) - np.pi/2) + R_D + R_T
        E_x = 0.8*(R_T * (np.sqrt(expansion_ratio)-1) + R_U*(1/np.cos(np.deg2rad(alpha-1))))/(np.tan(np.deg2rad(15))) # 80% Rau nozzle length
        E_y = R_E # Radius of exit
        Q_x = (E_y - np.tan(np.deg2rad(alpha))*E_x - N_y + np.tan(np.deg2rad(theta_D))*N_x)/(np.tan(np.deg2rad(theta_D)) - np.tan(np.deg2rad(alpha)))
        Q_y = (np.tan(np.deg2rad(theta_D))*(D_e/2 - np.tan(np.deg2rad(alpha))*E_x) - np.tan(np.deg2rad(alpha))*(N_y - np.tan(np.deg2rad(theta_D))*N_x))/(np.tan(np.deg2rad(theta_D)) - np.tan(np.deg2rad(alpha)))

        # Calculate t(x)
        t_x = (-2*Q_x + np.sqrt(4*Q_x**2 - 4*(N_x - x + 15.444)*(-N_x - 2*Q_x + E_x)))/(2*(-N_x - 2*Q_x + E_x))
        r_t = ((1-t_x)**2)*N_y + 2*(t_x - t_x**2)*Q_y + (t_x**2) * R_E
        r = r_t
    return r # Radius in [m]
    
  
# Function to calculate the central finite difference
def central_finite_difference(func, x, h, *args):
    # Inputs:
    # func - type: function - function to calculate
    # x - type: float - x value
    # h - type: float - step size
    # args - type: list - arguments to pass to function

    return (func(x+h, *args) - func(x-h, *args))/(2*h)

# Function which outlines ODE for local mach number squarred
def dN_dx (x, y, h, m, keydata):
    # Inputs:
    # x - type: float - current distance from injector
    # y - type: list - dependent variables [N, P, T]
    # h = type: float - step size
    # m - type: int - iteration counter
    # keydata - type: list - key data to pass to ODE solver contains [A_c, A_t, A_e, L_c, gam, Cp]

    # Unpack inputs
    N = y[0] # Local mach number squared
    P = y[1] # Local pressure
    T = y[2] # Local temperature

    # Key data
    A_c = keydata[0] # Area injector [m^2]
    A_t = keydata[1] # Area throat [m^2]
    A_e = keydata[2] # Area exit [m^2]
    L_c = keydata[3] # Chamber length [inch]
    gam = keydata[4] # Specific heat ratio
    Cp = keydata[5]  # Specific heat at constant pressure
    dF_dx = keydata[6][m]
    dQ_dx = keydata[7][m]

    # Calculate area at x 
    A = np.pi* (calc_radius(x, A_c, A_t, A_e, L_c))**2
    dA_dx = (np.pi*calc_radius(x+h, A_c, A_t, A_e, L_c)**2 - np.pi*calc_radius(x-h, A_c, A_t, A_e, L_c)**2)/(2*h)

    dN_dx = ((N/(1-N)) * ((1+gam*N)/(Cp*T)) * dQ_dx  +  (N/(1-N))*((2 + (gam-1)*N)/(specificGasConstant*T))*dF_dx - (N/(1-N))*((2 + (gam-1)*N)/A)*dA_dx)

    return dN_dx

# Function which outlines ODE for local pressure
def dP_dx (x, y, h, m, keydata):
        # Inputs:
    # x - type: float - current distance from injector
    # y - type: list - dependent variables [N, P, T]
    # h = type: float - step size
    # m - type: int - iteration counter
    # keydata - type: list - key data to pass to ODE solver contains [A_c, A_t, A_e, L_c, gam, Cp]

    # Unpack inputs
    N = y[0] # Local mach number squared
    P = y[1] # Local pressure
    T = y[2] # Local temperature

    # Key data
    A_c = keydata[0] # Area injector [m^2]
    A_t = keydata[1] # Area throat [m^2]
    A_e = keydata[2] # Area exit [m^2]
    L_c = keydata[3] # Chamber length [inch]
    gam = keydata[4] # Specific heat ratio
    Cp = keydata[5]  # Specific heat at constant pressure
    dF_dx = keydata[6][m] # Derivative of force with respect to x
    dQ_dx = keydata[7][m] # Derivative of heat with respect to x
    
    # Calculate area at x 
    A = np.pi* (calc_radius(x, A_c, A_t, A_e, L_c))**2
    dA_dx = (np.pi*calc_radius(x+h, A_c, A_t, A_e, L_c)**2 - np.pi*calc_radius(x-h, A_c, A_t, A_e, L_c)**2)/(2*h)

    dP_dx = (-(P/(1-N))*((gam*N)/(Cp*T))*dQ_dx - (P/(1-N))*((1+ (gam-1)*N)/(specificGasConstant*T))*dF_dx + (P/(1-N))*((gam*N)/A)*dA_dx)
    return dP_dx

# Function which outlines ODE for local temperature
def dT_dx (x, y, h, m, keydata):
        # Inputs:
    # x - type: float - current distance from injector [inch]
    # y - type: list - dependent variables [N, P, T]
    # h = type: float - step size
    # m - type: int - iteration counter
    # keydata - type: list - key data to pass to ODE solver contains A_c, A_t, A_e, L_c, gam, Cp, dF_dx, dQ_dx

    # Unpack inputs
    N = y[0] # Local mach number squared
    P = y[1] # Local pressure
    T = y[2] # Local temperature

    # Key data
    A_c = keydata[0] # Area injector [m^2]
    A_t = keydata[1] # Area throat [m^2]
    A_e = keydata[2] # Area exit [m^2]
    L_c = keydata[3] # Chamber length [m]
    gam = keydata[4] # Specific heat ratio
    Cp = keydata[5]  # Specific heat at constant pressure
    dF_dx = keydata[6][m] # Derivative of force with respect to x
    dQ_dx = keydata[7][m] # Derivative of heat with respect to x

    # Calculate area at x 
    A = np.pi* (calc_radius(x, A_c, A_t, A_e, L_c))**2
    dA_dx = (np.pi*calc_radius(x+h, A_c, A_t, A_e, L_c)**2 - np.pi*calc_radius(x-h, A_c, A_t, A_e, L_c)**2)/(2*h)


    
  
    
    dT_dx = ( (T/(1-N))*((1-gam*N)/(Cp*T))*dQ_dx - (T/(1-N))*(((gam-1)*N)/(specificGasConstant*T))*dF_dx + (T/(1-N))*(((gam-1)*N)/A)*dA_dx)
    return dT_dx

# Function which defines derrivatives of ODEs 
def derivs(x, y, h, m, keydata):
    # Inputs
    # x: Independent variable
    # y: Dependent variables (list)
    # h: Step size
    # m: Iteration counter
    # keydata: list - key data to pass to ODE solver contains array of radius and specific heat ratio
    # Outputs
    # Output k values for RK4 method of length n 
    k = [] # K values for RK4 method of length n 
    
    # ODEs 
    # dy1/dx 
    k.append(dN_dx(x, y, h, m, keydata))
    
    #dy2/dx 
    k.append(dP_dx(x, y, h, m, keydata))

    #dy3/dx
    k.append(dT_dx(x, y, h, m, keydata))

    return k

# RK4 function
def rk4(x, y, n, h, m, keydata):
    # Inputs
    # x: Independent variable
    # y: Dependent variables (list)
    # n: Number of ODEs
    # h: Step size
    # m: Iteration counter
    # keydata: list - key data to pass to ODE solver contains array of radius and specific heat ratio
    
    # Outputs
    k = [] # K values for RK4 method of length n has tuples
    ym = [0] * len(y) # midpoint values of y
    ye = [0] * len(y)# endpoint values of y
    k.append(derivs(x, y, h, m, keydata))
    
    for i in range(n):
        ym[i] = y[i] + k[0][i]*h/2
    k.append(derivs(x+h/2, ym, h, m, keydata))
    for i in range(n):
        ym[i] = y[i] + k[1][i]*h/2
    k.append(derivs(x+h/2, ym, h, m, keydata))
    for i in range(n):
        ye[i] = y[i] + k[2][i]*h
    
    k.append(derivs(x+h, ye, h, m, keydata))
    for i in range(n):
        y[i] = y[i] + h*(k[0][i] + 2*(k[1][i] + k[2][i]) + k[3][i])/6
    x = x+h # increment x by step size
    
    return x, y


# Function to solve ODEs using RK4 method
def integrator(x, y, n, h, xend, m, keydata):
    # Inputs 
    # x: Independent variable
    # y: Dependent variables (list)
    # n: Number of ODEs
    # h: Step size
    # xend: Final value of independent variable
    # m: Iteration counter
    # keydata: list - key data to pass to ODE solver contains array of radius and specific heat ratio

    while True:
        if (xend - x < h):
            h = xend - x
        x, y = rk4(x, y, n, h, m, keydata)
        if (x >= xend):
            break
    return x, y

def calc_flow_data(xi, xf, dx, M_c, P_c, T_c, keydata):
    #Inputs:
    # xi - type: float - initial distance from injector [m]
    # xf - type: float - final distance from injector [m]
    # dx - type: float - step size [m]
    # M_c - type: float - chamber mach number
    # P_c - type: float - chamber pressure [Pa]
    # T_c - type: float - chamber temperature [K]
    # keydata - type: list - key data to pass to ODE solver contains [A_c (m^2), A_t (m^2), A_e (m^2), L_c (m), gam, Cp, dF_dx, dQ_dx] 
    # Outputs:
    # dx - type: float - step size [m]
    # xp_m - type: list - list of x values at each iteration [m]
    # yp_m - type: list - list of y values at each iteration [N (mach^2), P (Pa), T (K)]


    # Calculate key flow data Pressure, Temperature, Mach number
    
    # Will use fourth order runge kutta to solve the following ODEs (worked example in testfile.py)
    # we will let y be a list of [N, P, T] where N is the local mach number squared, P is the local pressure, T is the local temperature 
    # Define initial conditions of RK4 algorithm 
    n = 3 # Number of ODEs 
    yi = [ M_c**2, P_c, T_c] # Initial conditions of n dependent variables [N, P, T] remebering N is Mach number squarred
    xout = dx # Output interval
    x = xi # Working x value (set to initial condition)
    m = 0 # Iteration counter
    xp_m = [] # Track all x values through out iteration process

    yp_m = [] # Copy of y values for all iterations
    
    y = yi # working y values
    
    while True:
        xend = x + xout
        if (xend > xf):
            xend = xf
        h = dx
        x, y = integrator(x, y, n, h, xend, m, keydata)
        m += 1 # Increment m as we have integrated 
        
        xp_m.append(x)
        yp_m.append(y.copy())        
        if (x>=xf):
            break
    # # Output results
    # for i in range(m+1):
    #    print("x = ", xp_m[i]*0.0254, "y = ", yp_m[i]) 
    # Create plots of pressure, temperature, mach number
    # print(f'Pressure at end of engine: {yp_m[-1][1]} [Pa] \n Temperature at end of engine: {yp_m[-1][2]} [K] \n Mach number at end of engine: {np.sqrt(yp_m[-1][0])}')
    # create_plot([xp*0.0254 for xp in xp_m], [np.sqrt(y[0]) for y in yp_m], "Distance from Injector [m]", "Mach Number", "Mach Number vs Distance from Injector")
    return dx, xp_m, yp_m


#-------------------------------------------------------------------
# FUNCTION TO CALCULATE HEAT TRANSFER THROUGH COOLANT CHANNELS
# ------------------------------------------------------------------
# Function to calculate gas transport properties 
def calc_tranport_properties(T, molecules, molecular_mass=None, molecular_fraction=None):
    # Allow passing a single combustion_molecules dict as the second argument
    # If molecules is a dict, extract arrays from it
    if isinstance(molecules, dict):
        # molecules: dict of {species: [fraction, mass]}
        molecular_fraction = [v[0] for v in molecules.values()]
        molecular_mass = [v[1] for v in molecules.values()]
        molecules = list(molecules.keys())
    # Otherwise, expect arrays as before

    # Create dictionary of transport property coefficients for each molecule where b is the low temperature range and r is the high temperature range
    tranport_coeff_dict = {
        "bH2": [0.68887644E00, 0.48727168E01, -0.59565053E03, 0.55569577E00, 0.93724945E00, 0.19013311E03, -0.19701961E05, 0.17545108E01],
        "bO2": [0.61936357E00, -0.44608607E02, -0.13460714E04, 0.19597562E01, 0.81595343E00, -0.34366856E02, 0.22785080E04, 0.10050999E01],
        "bH2O": [0.78387780E00, -0.38260408E03, 0.49040158E05, 0.85222785E00, 0.15541443E01, 0.66106305E02, 0.55969886E04, -0.39259598E01],
        "bOH": [0.78530133E00, -0.16524903E03, 0.12621544E05, 0.69788972E00, 0.10657500E01, 0.45300526E02, -0.37257802E04, -0.49894757E00],
        "bH": [0.58190587E00, 0.46941424E02, -0.68759582E04, 0.91591909E00, 0.58190587E00, 0.46941424E02, -0.68759582E04, 0.43477961E01],
        "bO": [0.73101989E00, 0.60468346E01, 0.35630372E04, 0.10955772E01, 0.73824503E00, 0.11221345E02, 0.31668244E04, 0.17085307E01],
        "rH2": [0.70504381E00, 0.36287686E02, -0.72255550E04, 0.41921607E00, 0.74368397E00, -0.54941898E03, 0.25676376E06, 0.35553997E01],
        "rO2": [0.63839563E00, -0.12344438E01, -0.22885810E05, 0.18056937E01, 0.80805788E00, 0.11982181E03, -0.47335931E05, 0.95189193E00],
        "rH2O": [0.50714993E00, -0.68966913E03, 0.87454750E05, 0.30285155E01, 0.79349503E00, -0.13340063E04, 0.37884327E06, 0.23591474E01],
        "rOH": [0.58936635E00, -0.36223418E03, 0.23355306E05, 0.22363455E01, 0.58415552E00, -0.87533541E03, 0.20830503E06, 0.35371017E01],
        "rH": [0.51631898E00, -0.14613202E04, 0.71446141E06, 0.21559015E01, 0.51631898E00, -0.14613202E04, 0.71446141E06, 0.55877786E01],
        "rO": [0.79832550E00, 0.18039626E03, -0.53243244E05, 0.51131026E00, 0.79819261E00, 0.17970493E03, -0.52900889E05, 0.11797640E01]
    }

    viscosity_array = []
    thermal_conductivity_array = []

    for element in molecules:
        if T <= 1000:
            A_visc = tranport_coeff_dict[f'b{element}'][0]
            B_visc = tranport_coeff_dict[f'b{element}'][1]
            C_visc = tranport_coeff_dict[f'b{element}'][2]
            D_visc = tranport_coeff_dict[f'b{element}'][3]
            A_thermal = tranport_coeff_dict[f'b{element}'][4]
            B_thermal = tranport_coeff_dict[f'b{element}'][5]
            C_thermal = tranport_coeff_dict[f'b{element}'][6]
            D_thermal = tranport_coeff_dict[f'b{element}'][7]
        else:
            A_visc = tranport_coeff_dict[f'r{element}'][0]
            B_visc = tranport_coeff_dict[f'r{element}'][1]
            C_visc = tranport_coeff_dict[f'r{element}'][2]
            D_visc = tranport_coeff_dict[f'r{element}'][3]
            A_thermal = tranport_coeff_dict[f'r{element}'][4]
            B_thermal = tranport_coeff_dict[f'r{element}'][5]
            C_thermal = tranport_coeff_dict[f'r{element}'][6]
            D_thermal = tranport_coeff_dict[f'r{element}'][7]

        viscosity_array.append(np.exp(A_visc*np.log(T) + (B_visc/T) + (C_visc/(T**2)) + D_visc))
        thermal_conductivity_array.append(np.exp(A_thermal*np.log(T) + (B_thermal/T) + (C_thermal/(T**2)) + D_thermal))

    viscocisty_mix = 0
    thermal_mix = 0

    for i in range(len(molecules)):
        visc_den_sum = 0
        therm_den_sum = 0
        for j in range(len(molecules)):
            sigma_ij = 0.25*((1+(viscosity_array[i]/viscosity_array[j])**(0.5) * (molecular_mass[j]/molecular_mass[i])**(0.25))**2) * ((2*molecular_mass[j])/(molecular_mass[i] + molecular_mass[j]))
            vi_ij = sigma_ij*(1 + ((2.41*(molecular_mass[i]-molecular_mass[j])*(molecular_mass[i] - 0.142*molecular_mass[j])) / (molecular_mass[i] + molecular_mass[j])**2))
            visc_den_sum += molecular_fraction[j]*sigma_ij
            therm_den_sum += molecular_fraction[j]*vi_ij

        viscocisty_mix += (molecular_fraction[i]*viscosity_array[i])/(molecular_fraction[i] + visc_den_sum)
        thermal_mix += (molecular_fraction[i]*thermal_conductivity_array[i])/(molecular_fraction[i] + therm_den_sum)

    return viscocisty_mix, thermal_mix

# Function to calculate the specific heat of the gas
def calc_gas_specific_heat(T, molecules, molecular_mass, molecular_fraction):


    # Dictionary containing thermodynamic coefficients for different species of elements, c, b, r for 100 - 700, 700 - 2000, 2000 - 6000 K respectively
    # Containes coefficients A, B, C, D, E for calculating Cp 
    thermo_curve_dict = {  
    "bO2": [-3.425563420E+04, 4.847000970E+02, 1.119010961E+00, 4.293889240E-03, -6.836300520E-07, -2.023372700E-09, 1.039040018E-12 ],
    "rO2": [-1.037939022E+06, 2.344830282E+03, 1.819732036E+00, 1.267847582E-03, -2.188067988E-07, 2.053719572E-11, -8.193467050E-16],
    "bOH": [-1.998858990E+03, 9.300136160E+01, 3.050854229E+00, 1.529529288E-03, -3.157890998E-06, 3.315446180E-09, -1.138762683E-12],
    "rOH": [ 1.017393379E+06, -2.509957276E+03, 5.116547860E+00, 1.305299930E-04, -8.284322260E-08, 2.006475941E-11, -1.556993656E-15],
    "bO": [-7.953611300E+03, 1.607177787E+02, 1.966226438E+00, 1.013670310E-03, -1.110415423E-06, 6.517507500E-10, -1.584779251E-13],
    "rO": [ 2.619020262E+05, -7.298722030E+02, 3.317177270E+00, -4.281334360E-04, 1.036104594E-07, -9.438304330E-12, 2.725038297E-16],
    "bH2O": [-3.947960830E+04, 5.755731020E+02, 9.317826530E-01, 7.222712860E-03, -7.342557370E-06, 4.955043490E-09, -1.336933246E-12],
    "rH2O": [1.034972096E+06, -2.412698562E+03, 4.646110780E+00, 2.291998307E-03, -6.836830480E-07, 9.426468930E-11, -4.822380530E-15],
    "bH2": [4.078323210E+04, -8.009186040E+02, 8.214702010E+00, -1.269714457E-02, 1.753605076E-05, -1.202860270E-08, 3.368093490E-12],
    "rH2": [ 5.608128010E+05, -8.371504740E+02, 2.975364532E+00, 1.252249124E-03, -3.740716190E-07, 5.936625200E-11, -3.606994100E-15],
    "bH": [0.000000000E+00, 0.000000000E+00, 2.500000000E+00, 0.000000000E+00, 0.000000000E+00, 0.000000000E+00 ,0.000000000E+00],
    "rH":[6.078774250E+01, -1.819354417E-01, 2.500211817E+00, -1.226512864E-07,  3.732876330E-11,-5.687744560E-15, 3.410210197E-19]}



    # Calculate the specific heat of gas for each element in the mixture
    element_Cp = [] # Specific heat of each element in the mixture [J/molK]
    for element in molecules:
        # Determine if current temperature is above 1000K as coefficients will change
        if T >= 1000:
            a1 = thermo_curve_dict[f'r{element}'][0] # extract a1 
            a2 = thermo_curve_dict[f'r{element}'][1] # extract a2
            a3 = thermo_curve_dict[f'r{element}'][2] # extract a3
            a4 = thermo_curve_dict[f'r{element}'][3] # extract a4
            a5 = thermo_curve_dict[f'r{element}'][4] # extraft a5
            a6 = thermo_curve_dict[f'r{element}'][5] # extract a6
            a7 = thermo_curve_dict[f'r{element}'][6] # extract a7
        else:
            a1 = thermo_curve_dict[f'b{element}'][0] # extract a1
            a2 = thermo_curve_dict[f'b{element}'][1] # extract a2
            a3 = thermo_curve_dict[f'b{element}'][2] # extract a3
            a4 = thermo_curve_dict[f'b{element}'][3] # extract a4
            a5 = thermo_curve_dict[f'b{element}'][4] # extraft a5
            a6 = thermo_curve_dict[f'r{element}'][5] # extract a6
            a7 = thermo_curve_dict[f'r{element}'][6] # extract a7

        element_Cp.append((a1*(1/(T**2)) + a2*(1/(T)) + a3 + a4*T + a5*(T**2) + a6*(T**3) + a7*(T**4))*Ru) # Calculate specific heat of element in mixture
    
    # Now calculate the specific heat of the mixture
    Cp_mix = 0 # Specific heat of the mixture [kJ/KgK] 
    for i in range(len(molecules)):
        Cp_mix += (molecular_fraction[i]*element_Cp[i]) # Calculate specific heat of mixture in kJ/KgK

    return Cp_mix # Return specific heat of mixture in J/molK
# Function which outputs chamber inner surface area
def calc_A_gas(dx, x, engine_info):
    # Inputs
    # dx - type: float - step size [m]
    # x - type: float - distance from injector [m]
    # engine_info - type: EngineInfo - engine information object containing gam, M_c, P_c, T_c, Cp, F_Vac, Ncc, combustion_molecules, A_c, A_t, A_e, L_c, x_j, chan_land, chan_w, chan_h, chan_t
    # Outputs
    # A_gas - type: float - chamber inner surface area [m^2]
    # print(f'x {x} [m] \n radius [m] {engine_info.get_radius(x)} \n dx [m] {dx} \n Ncc {engine_info.Ncc}')

    return ((2*np.pi*engine_info.get_radius(x))*dx)/engine_info.Ncc 

def calc_h_gas(x, y, dx, T_hw, T_star, engine_info):
    # Description:
    # Calculate the heat transfer coefficient of the gas using the Bartz equation
    # The Bartz equation is given by:
    # h_gas = 0.023 * (k_gas / d) * (Re**0.8) * (Pr**0.3)
    # where k_gas is the thermal conductivity of the gas, d is the hydraulic diameter, Re is the Reynolds number and Pr is the Prandtl number.
    # Inputs:
    # x - type: float - distance from injector [m]
    # y - type: list - dependent variables [N (mach^2), P (Pa), T (K)]
    # dx - type: float - step size [m]
    # T_star - type: float - reference temperature [K]
    # engine_info - type: EngineInfo - engine information object containing gam, M_c, P_c, T_c, Cp, F_Vac, Ncc, combustion_molecules, A_c, A_t, A_e, L_c, x_j, chan_land, chan_w, chan_h, chan_t
    # Outputs:
    # h_gas - type: float - heat transfer coefficient of the gas [W/m^2K]

    # Unpack y Input
    T = y[2] # Local static temperature [K] at pos X
    P = y[1] # Local static pressure [Pa] at pos X
    N = y[0] # Local mach number squared at pos X
    
    # Define variables
    Dt = 2*np.sqrt(engine_info.A_t/np.pi) # Throat diameter [m]
    P_c = engine_info.P_c # Chamber stagnation pressure [Pa]
    C_star = engine_info.C_star # Characteristic velocity [m/s]
    Ru = (0.494*0.0254)*Dt/2 # Radius at the throat [m]
    Ru = engine_info.RU#*0.0254 #* 0.0254 # [m]
    RD = engine_info.RD#*0.0254 #* 0.0254 # Radius of curvature at the throat [m]
    area = engine_info.calc_area(x) # Cross sectional Area at pos X [m^2]
    A_t = engine_info.A_t # Throat cross sectional area [m^2]
    T_s = engine_info.T_c # Chamber stagnation temperature [K]
    gam = engine_info.gam # Specific heat ratio
    mach = np.sqrt(N) # Local mach number at pos X
    
    # Calculate reference temperature to calcualte transport properties
    #T_star = T_s / (1 + ((gam - 1)/2)*N) # Reference temperature [K]
    #T_star = (1 + 0.032*N+0.58*(T_hw/T-1))*T

    # Calculate transport properties
    eta, llambda, Cp, Pr  = engine_info.calc_transport_properties(T_star, P)
    #print(f'Pressure [Pa] {P} \n Temperature [K] {T_star} \n eta [Pa.s] {eta} \n Thermal Conductivity [W/mK] {llambda} \n Cp [J/KgK] {Cp} \n Prantl Number {Pr}\n Dt [m] {Dt} \n P_c [Pa] {P_c} \n C_star [m/s] {C_star} \n Ru [m] {Ru} \n Area [m^2] {area} \n A_t [m^2] {A_t} \n T_s [K] {T_s} \n Gamma {gam} \n Local Mach Number {mach} \n Reference Temperature [K] {T_star}')
    # NOTE: Mistake on cryo-rocket barts equation is 0.026/(Dt**0.2) not 0.023/(Dt**2)

    sigma = 1 / ( (0.5*(T_hw/T_s) * (1+((gam-1)/2)*N) + 0.5)**(0.68) * (1 + ((gam-1)/2)*N )**(0.12) )
    h_gas = (0.026/(Dt**0.2)) * ( ((eta**0.2) * Cp)/(Pr**0.6) ) * (P_c/C_star)**(0.8) * (Dt/(Ru*0.5 + RD*0.5))**(0.1) * (A_t/area)**(0.9) * sigma
    # print(f'sigma{sigma} \n T_hw/T_c {T_hw/T_s} \n h_gas [W/m^2K] {h_gas}')
    # Debug Print
    # print(f'Throat Diameter [m] {Dt}\n Stagnation Pressure [Pa] {P_c}\n Static Pressure [Pa] {P} \n C_star [m/s] {C_star} \n Throat Radiu of curvature [m] {Ru} \n Local cross-sectional area [m^2] {area} \n Throat Area [m^2] {A_t} \n Stagnation Temperature [K] {T_s} \n Gamma {gam} \n Local Mach Number {mach} \n Reference Temperature [K] {T_star} \n Prantl Number {Pr} \n Cp [J/KgK] {Cp} \n Viscosity [Pa.s] {eta} \n Thermal Conductivity [W/mK] {llambda} \n Sigma {sigma} \n Heat Transfer Coefficient [W/m^2K] {h_gas}')

    return h_gas

def calc_T_aw(x, y, T_hw, T_star, engine_info):
    # Calculate adiabatic wall temperature
    # Inputs:
    # x - type: float - distance from injector [m]
    # y - type: list - dependent variables [N (mach^2), P (Pa), T (K)]
    # T_hw - type: float - wall temperature [K]
    # engine_info - type: EngineInfo - engine information object containing gam, M_c, P_c, T_c, Cp, F_Vac, Ncc, combustion_molecules, A_c, A_t, A_e, L_c, x_j, chan_land, chan_w, chan_h, chan_t
    # Outputs:
    # A_aw - type: float - adiamatic wall temperature [m^2]

    # Unpack y Input
    T = y[2] # Local static temperature [K] at pos X
    P = y[1] # Local static pressure [Pa] at pos X
    N = y[0] # Local mach number squared at pos X

    # Define variables
    T_s = engine_info.T_c # Chamber stagnation temperature [K]
    P_c = engine_info.P_c # Chamber stagnation pressure [Pa]

    gam = engine_info.gam # Specific heat ratio
    mach = np.sqrt(N) # Local mach number at pos X

    # Calculate reference temperature to calcualte transport properties
    #T_star = T_s / (1 + ((gam - 1)/2)*N) # Reference temperature [K]
    #T_star = (1 + 0.032 * N + 0.58 * ( T_hw/T - 1) ) * T # no change
    
    # Calculate transport properties
    eta, llambda, Cp, Pr  = engine_info.calc_transport_properties(T_star, P)

    # Calculate the adiabatic wall temperature
    T_aw = T_s * ( (1 + Pr**0.33 * ((gam-1)/2)*N)/(1+ (gam-1)/2 * N))
    
    return T_aw

# function which calculates the heat transfer through the walls and coolant channels 
def calc_heat_transfer(x, y, h, Ncc, keydata):
    # Inputs
    # x - type: array - distance from injector
    # y - type: array - dependent variables [N, P, T]
    # h - type: float - step size
    # Ncc - type: float - number of coolant channels
    # keydata - type: list - key data to pass to ODE solver contains array of radius and specific heat ratio A_c, A_t, A_e, L_c, gam, Cp, dF_dx, dQ_dx

    # Solve for Thw and Tcw using a multivariable newtonian method


    # q_gas = h_gas*A_gas*(T_aw - T_hw)
    # q_H2 = h_H2*A_H2*(Tcw - T_H2)
    # q_wall = k*A_gas*dT/dy

    A_gas = calc_A_gas(x, y, h, Ncc, keydata)


    # Define the functions 
    # f1 = q_H2 - q_gas 
    # f2 = q_H2 - q_wall 

    return None



def calc_q_gas(dx, x, y, T_hw, engine_info):
    # Description:
    # Solve for q_gas, equation is h_gas*A_gas*(T_aw - T_hw) where h_gas is the heat transfer coefficient of the gas, A_gas is the chamber inner surface area, T_aw is the wall temperature and T_hw is the gas temperature at the wall.
    # Inputs:
    # dx - type: float - step size [m]
    # x - type: float - distance along combustion chamber w.r.t inject being 0 [m]
    # y - type: list - list of y values at each iteration [N (mach^2), P (Pa), T (K)]
    # T_hw - type: float - wall temperature [K]
    # engine_info - type: EngineInfo - engine information object containing gam, M_c, P_c, T_c, Cp, F_Vac, Ncc, combustion_molecules, A_c, A_t, A_e, L_c, x_j, chan_land, chan_w, chan_h, chan_t
    # Outputs:
    # q_gas - type: list - list of heat transfer through gas [W]
    chan_w, chan_h, chan_t, chan_land = engine_info.get_geo_interp(x, dx)

    # Unpack y Input
    T = y[2] # Local static temperature [K] at pos X
    P = y[1] # Local static pressure [Pa] at pos X
    N = y[0] # Local mach number squared at pos X
    T_s = engine_info.T_c # Chamber stagnation temperature [K]
    gam = engine_info.gam # Specific heat ratio
    
    T_star = (1 + 0.032*N+0.58*(T_hw/T-1))*T

    # Calculate surface area
    A_gas = calc_A_gas(dx, x, engine_info) # [m^2]
    
    # Calculate h_gas using Bartz Equation
    h_gas = calc_h_gas(x, y, dx, T_hw, T_star, engine_info) # [W/m^2K]
    # Calculate T_aw 
    T_aw = calc_T_aw(x, y, T_hw, T_star, engine_info) # [m^2]

    R_hot = 1/(h_gas*(chan_w+chan_land)*dx) # [K/W] - thermal resistance of the gas

    # q_gas = h_gas * A_gas * (T_aw - T_hw)  # [W]
    q_gas = (T_aw - T_hw) / R_hot  # [W] - using thermal resistance to calculate heat transfer

    # print(f'Calculated q_gas: {q_gas} [W] at x = {x} [m] with T_hw = {T_hw} [K] \n radius = {engine_info.get_radius(x)} [m] \n h_gas = {h_gas} [W/m^2K] \n A_gas = {A_gas} [m^2] \n T_aw = {T_aw} [K] \n T_aw - T_hw = {T_aw - T_hw} [K] \n Q_gas = {q_gas/(2*np.pi*engine_info.get_radius(x)*dx)} [W/m]')
    heatflux= q_gas / A_gas  # [W/m^2]


    return q_gas, heatflux, h_gas, A_gas, R_hot, (chan_w+chan_land)*dx

# Calculate the Reynolds number
def calc_reynoldsNumber(rho, u, D, v):
    """
    Calculate the reynolds number
    Inputs
    rho - type: float - density of fluid [kg/m^3]
    u - type: float - viscosity of fluid [Pa*s]
    D - type: float - hydraulic diameter of pipe [m]
    v - type: float - velocity of fluid [m/s]
    """

    return (rho*v*D)/u

# Calculate the friction factor using a Newtonian root solver
def calc_frictionFactor(Re, Dh, initial_guess, e):
    """
    Calculates the friction factor using a Newtonian root solver.

    Parameters:
    - Re (float): Reynolds number.[unitless]
    - Dh (float): Hydraulic diameter. [m]
    - initial_guess (float): Initial guess for the friction factor. [unitless]
    - e (float): Roughness height. [m]

    Returns:
    - frictionFactor (float): The calculated friction factor.
    """
    # Define the function to solve for the friction factor
    def g(f, e, Dh, Re):
        #print("G(f):", 1/np.sqrt(f) + 2*np.log10((e/(3.7*Dh)) + (2.51/(Re*np.sqrt(f)))))
        return 1/np.sqrt(f) + 2*np.log10((e/(3.7065*Dh)) + (2.5226/(Re*np.sqrt(f))))
    
    def gprime(f, e, Dh, Re):
        h = 1e-5
        #print("GPrime:",(g(f+h, e, D, Re) - g(f-h, e, D, Re))/(2*h))
        return (g(f+h, e, Dh, Re) - g(f-h, e, Dh, Re))/(2*h)
    
    # Initialize the friction factor
    frictionFactor = scipy.fsolve(g, initial_guess, args = (e, Dh, Re))
    
    return frictionFactor[0]

#-----------------------------------------------------------------------------------
# FUNCTION TO CALCULATE SECTION 2.6 Values (Thermodynamics) THROUGH COOLANT CHANNELS
#-----------------------------------------------------------------------------------

# Function to calculate dilute gas component of heat transfer coefficient (Eqtn 2.6.1 from cryo-rocket.com)
def calc_dilute_gas_component(tau, delta, coef, c):
    # Inputs
    # tau | type: float | Desc: reduced temperature T/Tc
    # delta | type: float | Desc: reduced density rho/rhoc
    # coef | type: Dictionary | Desc: Dictionary of coefficients for the component
    # c | type: string | Desc: What type of hydrogen (normal vs para)

    # Outputs:
    # gamma0 | type: float | Desc: dilute gas component float

    gamma0nummerator = 0 # Initialize gamma0nummerator
    gamma0denominator = 0 # Initialize gamma0denominator

    # Initialize summation for top of equation
    for i in range(len(coef[c+'A1'])):
        #print(c,i,coef[c+'A1'][i])
        A1 = coef[c+'A1'][i]
        gamma0nummerator += (A1)*(tau**(i+1))

    for i in range(len(coef[c+'A2'])):
        #print(i,coef[c+'A2'][i])

        gamma0denominator += coef[c+'A2'][i]*(tau**(i+1))
        # if c == 'N':
        #     print(gamma0denominator)
    return  gamma0nummerator/gamma0denominator

# Function to calculate excess conductivity (Eqtn 2.6.2 from cryo-rocket.com)
def calc_excess_conductivity(tau, delta, coef, c):
    # Inputs
    # tau | type: float | Desc: reduced temperature T/Tc
    # delta | type: float | Desc: reduced density rho/rhoc
    # coef | type: Dictionary | Desc: Dictionary of coefficients for the component
    # c | type: string | Desc: What type of hydrogen (normal vs para)

    # Outputs:
    # deltaGamma | type: float | Desc: excess conductivity float

    deltaGamma = 0 # Initialize excess conductivity sum tracker

    for i in range(len(coef[c+'B1'])):
        # print(i,coef[c+'B1'][i], coef[c+'B2'][i])
        deltaGamma += (coef[c+'B1'][i] + coef[c+'B2'][i] * tau) * (delta**(i+1))
    return  deltaGamma

# Function to calculate critical enhancement factor (Eqtn 2.6.3 from cryo-rocket.com)
def calc_critical_enhancement_factor(tau, delta, C1, C2, C3):
    # Inputs
    # tau | type: float | Desc: reduced temperature T/Tc
    # delta | type: float | Desc: reduced density rho/rhoc
    # coef | type: Dictionary | Desc: Dictionary of coefficients for the component
    # c | type: string | Desc: What type of hydrogen (normal vs para)

    # Output
    # gammaC | type: float | Desc: critical enhancement factor float

    
    delta_rho_c = delta - 1
    delta_T_c = tau - 1

    gammaC = (C1/(C2 + abs(delta_T_c))) * np.exp(-((C3*delta_rho_c)**2))
    
    #print(np.exp(-(C3*delta_rho_c)**2))
    return gammaC

# Function to calculate Normal hydrogen thermal conductivity
def calc_normal_hydrogen_thermal_conductivity(T, P_desired):
    # Inputs 
    # tau | type: float | Desc: reduced temperature T/Tc
    # P_desired | type: float | Desc: pressure

    # Outputs
    # gamma_total_normal | type: float | Desc: total thermal conductivity for normal hydrogen

    # #  Define coefficients for thermal conductivity 

    thermal_coef_dict = {
        'NA1': [-3.40976E-01, 4.58820E+00, -1.45080E+00, 3.26394E-01, 3.16939E-03, 1.90592E-04, -1.13900E-06], # units [W/m*K]
        'NA2': [1.38497E+02, -2.21878E+01, 4.57151E+00, 1.00000E+00],
        'NB1': [3.63081E-02, -2.07629E-02, 3.14810E-02, -1.43097E-02, 1.74980E-03],# units [W/m*K]
        'NB2': [1.83370E-03, -8.86716E-03, 1.58260E-02, -1.06283E-02, 2.80673E-03],# units [W/m*K]
        }

    Tc = 33.145  # Critical temperature in K
    rho_c = 31.262 # Critical density in kg/m^3
    
    tau = T/Tc # Reduced temperature

    # Constants 
    C1 = 6.24E-4 # C1 constant
    C2 = -2.58E-7 # C2 constant
    C3 = 0.837 # C3 constant
    rho_initial = 80 # kgm^3

    # Get the density of liquid hydrogen at said temperature and pressure 25% para percent as this calc is for normal hydrogen 
    h, rho_guess, Cp, Cv = hydrogen_thermodynamics(P_desired, rho_initial, 0.25, T)
    delta = rho_guess/rho_c # Reduced density

    # Calculate thermal transfer for normal hydrogen
    gamma0 = calc_dilute_gas_component(tau, delta, thermal_coef_dict, 'N')
    deltaGamma = calc_excess_conductivity(tau, delta, thermal_coef_dict, 'N')
    gammaC = calc_critical_enhancement_factor(tau, delta, C1, C2,C3)
    gamma_total_normal = gamma0 + deltaGamma + gammaC

    return gamma_total_normal

def calc_para_hydrogen_thermal_conductivity(T, P_desired):


    thermal_coef_dict = {
        'PA1': [-1.24500E+00, 3.10212E+02, -3.31004E+02, 2.46016E+02, -6.57810E+01, 1.08260E+01, -5.19659E-01, 1.439790E-02], # units [W/m*K]
        'PA2': [1.42304E+04, -1.93922E+04, 1.58379E+04, -4.81812E+03, 7.28639E+02, -3.57365E+01, 1],
        'PB1': [2.65975E-02, -1.33826E-03,  1.30219E-02, -5.67678E-03,  -9.23380E-05],# units [W/m*K]
        'PB2': [-1.21727E-03, 3.66663E-03, 3.88715E-03, -9.21055E-03, 4.00723E-03],# units [W/m*K]
        }

    Tc = 32.938  # Critical temperature in K
    rho_c = 31.323 # Critical density in kg/m^3
    rho_initial = 80 # G    # Rho guess is 80 kg/m^3 just becuase this works for a wide range of temperatures and pressures
    tau = T/Tc
    C1 = 3.57E-4 # C1 Constant para-hydrogen
    C2 = -2.46E-2 # C2 Constant para-hydrogen
    C3 = 0.2 # C3 Constant para-hydrogen

    # Get the density of liquid hydrogen at said temperature and pressure  
    h, rho_guess, Cp, Cv = hydrogen_thermodynamics(P_desired, rho_initial, 1, T) # 100% paraPercent
    # Using density from above calculate delta and tau
    delta = rho_guess/rho_c # Reduced density
    

    # Calculate thermal transfer for para hydrogen
    gamma0 = calc_dilute_gas_component(tau, delta, thermal_coef_dict, 'P')
    deltaGamma = calc_excess_conductivity(tau, delta, thermal_coef_dict, 'P')
    gammaC = calc_critical_enhancement_factor(tau, delta, C1, C2, C3)
    gamma_total_para = gamma0 + deltaGamma + gammaC

    return gamma_total_para

# Function to calculate hydrogen viscosity
def calc_hydrogen_viscosity(T, P_desired):
    # Inputs
    # T - type: float - Temperature in K
    # P_desired - type: float - Pressure in Pa

    # Outputs
    # viscosity - type: float - Viscosity in [Pa*s]

    #Constants
    rho_initial = 80 # Rho guess is 80 kg/m^3 just becuase this works for a wide range of temperatures and pressures

    # Calculate para percent
    paraPercent = para_fraction(T)/100

    # Calculate density of hydrogen at said temperature and pressure
    h, rho_guess, Cp, Cv = hydrogen_thermodynamics(P_desired, rho_initial, paraPercent, T)


    rho = rho_guess/1000 # Convert to g/cm^3

    # Constants
    C1 = 8.5558
    C2 = 650.39
    C3 = 1175.9
    C4 = 19.55

    # Calculate eta_0
    eta_0 = C1 * (T**(3/2)) / (T + C4) * (T + C2) / (T + C3)

    # Calculate A using equation (2.6.10)
    A = np.exp(5.7694 + np.log(rho) + 65*rho**(3/2) - 6e-6 * np.exp(127.2*rho))

    # Calculate B using equation (2.6.11)
    B = 10 + 7.2 * ((rho / 0.07)**6 - (rho / 0.07)**(3/2)) - 17.63 * np.exp(-58.75 * (rho / 0.07)**3)

    # Calculate delta_eta using equation (2.6.8)
    delta_eta = A * np.exp(B / T)

    # Calculate eta_tot using equation (2.6.9) [Pa Sec]
    eta_tot = (eta_0 + delta_eta) * 1e-7

    return eta_tot

# Function to calculate liquid transport properties of hydrogen
def calc_liquid_hydrogen_transport_properties(T, P_desired):
    # Inputs
    # T - type: float - Temperature in K
    # P_desired - type: float - Pressure in Pa

    # Outputs
    # thermal_conductivity - type: float - Thermal Conductivity in [W/m*K]
    # viscosity - type: float - Viscosity in [Pa*s]

    # Calculate normal hydrogen thermal conductivity
    gamma_total_normal = calc_normal_hydrogen_thermal_conductivity(T, P_desired)

    # Calculate para hydrogen thermal conductivity
    gamma_total_para = calc_para_hydrogen_thermal_conductivity(T, P_desired)
    
    # Calculate ortho thermal conductivity
    gamma_total_ortho = (gamma_total_normal - 0.25* gamma_total_para)/0.75
    
    # Calculate para percent at current temperature
    paraPercent = para_fraction(T)/100

    # Calculate total thermal conductivity for entire mixture
    thermal_conductivity = paraPercent*gamma_total_para + (1-paraPercent)*gamma_total_ortho
    
    # Calculate viscosity
    viscosity = calc_hydrogen_viscosity(T, P_desired)
    # print("Visc", viscosity)


    return thermal_conductivity, viscosity

# Function to calculate the heat transfer through the coolant channels
def calc_q_h2(dx, x, y, s, T_cw, T_LH2, P_LH2, engine_info):
    # Description:
    # Solve for q_h2, equation is h_h2*A_h2*(T_cw - T_hw) where h_h2 is the heat transfer coefficient of the coolant, A_h2 is the coolant channel surface area, T_cw is the coolant temperature and T_hw is the wall temperature.
    # Inputs:
    # dx - type: float - step size [m]
    # x - type: float - distance along combustion chamber w.r.t inject being 0 [m]
    # y - type: list - list of y values at each iteration [N (mach^2), P (Pa), T (K)]
    # T_cw - type: float - coolant temperature [K]
    # T_LH2 - type: float - liquid hydrogen bulk temperature [K]
    # P_LH2 - type: float - liquid hydrogen bulk pressure [Pa]
    # engine_info - type: EngineInfo - engine information object containing gam, M_c, P_c, T_c, Cp, F_Vac, Ncc, combustion_molecules, A_c, A_t, A_e, L_c, x_j, chan_land, chan_w, chan_h, chan_t
    # Outputs:
    # q_h2 - type: list - list of heat transfer through coolant [W]

    # Get channel dimensions for the current x position
    chan_w, chan_h, chan_t, chan_land = engine_info.get_geo_interp(x, dx)
    chan_w_next, chan_h_next, chan_t_next, chan_land_next = engine_info.get_geo_interp(x+dx, dx)


    # Calculate hydraluic diameter of cooling channels at x
    #Dh = (2*chan_w*chan_h)/(chan_w + chan_h) # m
    #Dh_next = (2*chan_w_next*chan_h_next)/(chan_w_next + chan_h_next) # m

    # Calcualte channel area
    chan_area = chan_w * chan_h # m^2
    chan_area_next = chan_w_next * chan_h_next # m^2

    # Calculate wetted perimeter
    P = 2*(chan_w + chan_h) # m
    P_next = 2*(chan_w_next + chan_h_next) # m

    # Calculate hydraulic diameter
    Dh = (4*chan_area)/P
    Dh_next = (4*chan_area_next)/P_next
    

    # print(f'Channel Width: {chan_w} [m]\n Channel Height: {chan_h} [m]\n Channel Thickness: {chan_t} [m]\n Channel Land: {chan_land} [m]\n Radius {engine_info.get_radius(x)} [m] at x = {x} [m] \n T_cw {T_cw} [K] \n T_LH2 {T_LH2} [K] \n P_LH2 {P_LH2} [Pa] \n dx {dx} [m] \n DH {Dh} [m]')

    # Calculate coolant reference temperature [K] Equation 6.1.5b
    #T_LH2 = (T_cw + T_LH2)/2

    # Get the viscosity of LH2 @ T 
    fluid = "Hydrogen"

    # Get properties
    rho_LH2 = PropsSI("D", "T", T_LH2, "P", P_LH2, fluid)                           # kg/m^3
    eta_LH2 = PropsSI("VISCOSITY", "T", T_LH2, "P", P_LH2, fluid)                 # Pa·s
    therm_LH2 = PropsSI("CONDUCTIVITY", "T", T_LH2, "P", P_LH2, fluid)   # W/m·K
    Cp = PropsSI("CPMASS", "T", T_LH2, "P", P_LH2, fluid)                           # J/kg·K
    Pr_LH2 = PropsSI("PRANDTL", "T", T_LH2, "P", P_LH2, fluid)                     # dimensionless
    h = PropsSI("H", "T", T_LH2, "P", P_LH2, fluid)/1000                             # J/kg Enthalpy [kJ/kg]

    # Calculate LH2 viscosity and thermal conductivity # [W/m*K] and [Pa*s]
    # therm_LH2, eta_LH2 = calc_liquid_hydrogen_transport_properties(T_LH2, P_LH2) # TESTED agains NIST and MINI-REFPROP PASSED @  38.9MPa 296K
    
    # paraPercent = para_fraction(T_LH2)/100

    # Calculate the density of LH2 @ T_f
    # h = [kJ/kg], rho = [kg/m^3], Cp = [J/g*K], Cv = [J/g*K]
    # h, rho_LH2, Cp, Cv = hydrogen_thermodynamics(P_LH2, 80, paraPercent, T_LH2) # Density [kg/m^3] # TESTED against NIST and MINI-REFPROP PASSED @  38.9MPa 296K
    # print(f'\nrho_LH2: {rho_LH2} [kg/m^3] \n'
    #       f'eta_LH2: {eta_LH2} [Pa*s] \n'
    #       f'therm_LH2: {therm_LH2} [W/m*K] \n'
    #       f'Cp: {Cp} [J/kg*K] \n'
    #       f'Pr_LH2: {Pr_LH2} [Unitless] \n'
    #       f'T_LH2: {T_LH2} [K] \n'
    #       f'P_LH2: {P_LH2} [Pa] \n')
    # Calcualte LH2 prandtl number
    # Pr_LH2 = (eta_LH2 * Cp)/therm_LH2
    #print(f'LH2 Prandtl Number: {Pr_LH2}')

    # Calculate velocity of coolant in channel NOTE mass flow rate needs to be scalled down by number of channels cross sectional area is channel area allready in m 
    
    v = (engine_info.mdot_LH2/engine_info.Ncc)/(rho_LH2*chan_area) # m/s
    # Calculate Reynolds number 
    # Re = calc_reynoldsNumber(rho_LH2, eta_LH2, Dh, v)
    Re = rho_LH2*v*Dh/eta_LH2 # [unitless] - Reynolds number

    # print(Re, Dh, engine_info.e)
    # Calculate friction factor
    f = calc_frictionFactor(Re, Dh, 0.001, engine_info.e)  

    # Calculate Xi (weird squiqqly greek letter) smooth rought friction factor to smooth friction factor
    xi = f/calc_frictionFactor(Re, Dh, 0.001, 0)

    # Calculate epsilon 
    epsilon_star = Re * (engine_info.e/Dh)*((f/8)**0.5)
    
    # Calculate B(epsilon_star)
    if epsilon_star >= 7:
        B = 4.7*(epsilon_star)**0.2
    else:
        B = 4.5 + 0.57*(epsilon_star)**0.75
    
    
    #C_eta = (1+1.5*bulk.Pr^(-1/6)*bulk.Re^(-1/8)*(bulk.Pr-1))/(1+1.5*bulk.Pr^(-1/6)*bulk.Re^(-1/8)*(bulk.Pr*eta-1))*eta;% Nino Equation 10
    # Calculate C1 coefficient
    C1 = (1 + 1.5*Pr_LH2**(-1/6) * Re**(-1/8) * (Pr_LH2-1))/(1+1.5*Pr_LH2**(-1/6)*Re**(-1/8)*(Pr_LH2*xi - 1))*xi

    # Calculate C2 coefficient
    C2 = 1 + (s/Dh)**(-0.7) * (T_cw/T_LH2)**(0.1)
    # Calculate C3 Eqtn 6.1.14 
    radius, radiusType = engine_info.get_r_value(x)

    if radius != 0:
        if radiusType != 'Ru':
            concavity = -0.05
            C3 = (Re*((0.25*Dh)/(-1*radius))**2 )** concavity
        else:
            concavity = 0.05
            # Calculate C3 coefficient
            C3 = (Re*((0.25*Dh)/(radius))**2 )** concavity
    else:
        C3 = 1
    
    # Calculate the Nusselt Number
    #rocket_correction = 2.7 # Test 
    Nu = ( (f/8)*Re*Pr_LH2*(T_LH2/T_cw)**0.55)/( 1 + (f/8)**0.5 * (B - 8.48))  * C2 * C3 #* C1
    
    # Calculate h_H2 the heat transfer coefficient for the cooling channels
    h_H2 = (Nu * therm_LH2)/Dh # [W/m^2K]

    # print(
    #     f'Calculated h_H2: {h_H2} [W/m^2K] (Nu*thermL_H2)/Dh\n'
    #     f'Radius: {radius} [m] \n'
    #     f'friction factor {f}\n'
    #     f'Reynolds Number: {Re}\n'
    #     f'Mass Flow Rate: {engine_info.mdot_LH2} [kg/s]\n'
    #     f'Ncc (Number of coolant channels): {engine_info.Ncc}\n'
    #     f'Dh (Hydraluic Diameter): {Dh}\n'
    #     f'v (fluid velocity): {v}\n'
    #     f'Nu (Local nusselt number): {Nu}\n'
    #     f'Nu ratio: {Nu_ratio}\n'
    #     f'C1 (correction factor): {C1}\n'
    #     f'C2 (correction factor): {C2}\n'
    #     f'C3 (correction factor): {C3}\n'
    #     f'epsilon_star: {epsilon_star}\n'
    #     f'xi: {xi}\n'
    #     f'B: {B}\n'
    #     f'e:{ engine_info.e}\n'
    #     f'sqrt(f/8): {((f/8)**0.5)}\n'
    #     f'e/Dh: {(engine_info.e/Dh)}')
    
    
    # print(f'Calculated h_H2: {h_H2} [W/m^2K]\n Nu = {Nu} [unitless] \n Dh = {Dh} [m] \n therm_LH2 = {therm_LH2} [W/m*K] \n T_cw = {T_cw} [K] \n T_LH2 = {T_LH2} [K]')
    # Calcualte heat transfer efficieny through the channels

    # Equation 6.1.18
    perimiter = 2*(chan_land + dx) # Calculate the perimeter of the fin [m]
    fin_area = chan_land * dx # Calculate the cross sectional area of the channel [m^2]

    # m = np.sqrt(2*h_H2)/(engine_info.k * chan_land)
    m = np.sqrt(h_H2*perimiter)/(engine_info.k * fin_area) # From https://ia801300.us.archive.org/5/items/bzbzbzHeatTrans/Heat%20and%20Mass%20Transfer/Bergman%2C%20Incropera/Introduction%20to%20Heat%20Transfer%206e%20c.2011%20-%20Bergman%2C%20Incropera.pdf Table 3.4 bottom
    M = np.sqrt((h_H2 * perimiter * engine_info.k*fin_area))*(T_cw - T_LH2)
    
    # Equation 6.1.19
    L_c = chan_h + chan_land/2 # + chan_t/2

    q_fin  = M * np.tanh(m*L_c) # Calculate the fin heat transfer [W] From https://ia801300.us.archive.org/5/items/bzbzbzHeatTrans/Heat%20and%20Mass%20Transfer/Bergman%2C%20Incropera/Introduction%20to%20Heat%20Transfer%206e%20c.2011%20-%20Bergman%2C%20Incropera.pdf table 3.4 Eqtn 3.81

    q_base = h_H2 * (dx*chan_w)*(T_cw - T_LH2) # Calculate the base heat transfer [W] using Newtons law of cooling

    # Fin thermal resistance
    R_fin = (T_cw - T_LH2)/q_fin # Calculate the fin thermal resistance [K/W]
    R_base = (T_cw - T_LH2)/q_base # Calculate the base thermal resistance [K/W]
    R_cold = 1/(1/R_fin + 1/R_base) # Calculate the cold thermal resistance [K/W]

    #Equation 6.1.17
    #eta_fin = np.tanh(m*L_c)/(m*L_c)

    # Calculate A_H2 the wetted surface area of the cahnnel interior
    #A_H2 = (2*eta_fin*chan_h + chan_w)*dx
   
    #m = np.sqrt(h_H2 * A_H2)/(engine_info.k*)

    # Calculate q_H2
    #q_h2 = h_H2 * A_H2 * (T_cw - T_LH2) # W
    q_h2 = (T_cw - T_LH2)/R_cold # Calculate the heat transfer through the coolant channels [W]

    #print(f'Calculated q_h2: {q_h2} [W] at x = {x} [m] with T_cw = {T_cw} [K] \n h_H2 = {h_H2} [W/m^2K] \n A_H2 = {A_H2} [m^2] \n T_LH2 = {T_LH2} [K] \n T_cw - T_LH2 = {T_cw - T_LH2} [K]')

    # Calculate next station values 
    # TEMPERTAURE CHANGE
    # print(f'q_h2 [W]: {q_h2} \n mdot = {engine_info.mdot_LH2} [kg/s] \n Ncc = {engine_info.Ncc} [unitless] \n Cp = {Cp} [J/kg*K] \n T_LH2 = {T_LH2} [K] \n P_L2 = {P_LH2} [Pa]')

    

    #deltaT =  q_h2/((engine_info.mdot_LH2/engine_info.Ncc)*Cp) # Calculate the change in temperature of the coolant based on the heat transfer through the coolant channels
    
    # Calculate next station values
    #T_LH2_new = T_LH2 + deltaT

    # PRESSURE CHANGE
    majorLosses = (f*rho_LH2*(v**2)*dx)/(2*Dh) # Calculate the major losses in the coolant channels
    
    
    if chan_area_next > chan_area: # expansion
        # Expansion
        K = ((Dh / Dh_next)**2 - 1)**2
    
    elif chan_area_next < chan_area: # contraction
        # Contraction
        K = 0.5 - 0.167 * (Dh_next/Dh) - 0.125 * (Dh_next / Dh)**2 - 0.208*(Dh_next/Dh)**3
    else:
        # No change
        K = 0
   
    minorLosses = (K*rho_LH2*v**2)/2

    #fluidLosses = (2/(engine_info.calc_area(x)*engine_info.Ncc)+(engine_info.calc_area(x+dx)*engine_info.Ncc))*( (1/(rho_LH2*engine_info.calc_area(x)*engine_info.Ncc)) - (1/(rho_LH2*engine_info.calc_area(x+dx)*engine_info.Ncc)))# Calculate the fluid losses in the coolant channels
    
    fluidLosses = (2/((chan_area*engine_info.Ncc) + (chan_area_next*engine_info.Ncc))) * engine_info.mdot_LH2 ** 2 * ((1/(rho_LH2*chan_area*engine_info.Ncc)) - (1/(rho_LH2*chan_area_next*engine_info.Ncc)))
    
    pressureLoss = minorLosses + majorLosses + fluidLosses # Calculate the total pressure loss in the coolant channels
    
    P_LH2_new = P_LH2 - pressureLoss # Calculate the new pressure of the coolant channels

    h_next = h + (q_h2/1000) / (engine_info.mdot_LH2/engine_info.Ncc) # Convert q_h2 to kJ/s and divide by mass flow rate to get enthalpy change [kJ/kg]

    # Calculate deltaT based off enthalpy change
    T_LH2_new = PropsSI("T", "H", h_next*1000, "P", P_LH2_new, fluid) # Convert h_next to J/kg and get new temperature [K] using NIST REFPROP

    # Calculate dF_dx for this x value
    dF_dx = (f*dx*v**2)/(2*Dh) # Calculate the frictional losses in the coolant channels
    
    # Debug Print
    # print(f'Pressure Loss: {pressureLoss} [Pa] \n Major Losses: {majorLosses} [Pa] \n Minor Losses: {minorLosses} [Pa] \n Fluid Losses: {fluidLosses} [Pa] \n Dh: {Dh} [m] \n Dh_next: {Dh_next} [m] \n K: {K} [unitless] \n rho_LH2: {rho_LH2} [kg/m^3] \n v: {v} [m/s] \n eta_fin: {eta_fin} [unitless]')
    # print(f'Temperature Change: {deltaT} [K] at x = {x} [m] with q_h2 = {q_h2} [W] and Cp = {Cp} [J/kg*K] and mdot_LH2 = {engine_info.mdot_LH2} [kg/s]')
    # Debug Print
    # print(f'Calculated q_h2: {q_h2} [W] at x = {x} [m] with T_cw = {T_cw} [K] \n h_H2 = {h_H2} [W/m^2K] \n A_H2 = {A_H2} [m^2] \n T_LH2 = {T_LH2} [K] \n T_cw - T_LH2 = {T_cw - T_LH2} [K]')

    return q_h2, T_LH2_new, P_LH2_new, dF_dx, h_H2, C3, v, rho_LH2, chan_area, Re, Nu, Dh, therm_LH2

# Function to calculate the heat transfer through the walls
def calc_q_wall(dx, x, y, T_hw, T_cw, engine_info):
    # Descrioption:
    # Solve for q_wall, equation is k*A_gas*dT/dy where k is the thermal conductivity of the wall, A_gas is the chamber inner surface area, dT/dy is the temperature gradient across the wall.
    # Inputs:
    # dx - type: float - step size [m]
    # x - type: float - distance along combustion chamber w.r.t inject being 0 [m]
    # y - type: list - list of y values at each iteration [N (mach^2), P (Pa), T (K)]
    # T_hw - type: float - wall temperature [K]
    # T_cw - type: float - coolant temperature [K]
    # engine_info - type: EngineInfo - engine information object containing gam, M_c, P_c, T_c, Cp, F_Vac, Ncc, combustion_molecules, A_c, A_t, A_e, L_c, x_j, chan_land, chan_w, chan_h, chan_t
    # Outputs:
    # q_wall - type: list - list of heat transfer through wall [W]

    # Get channel dimensions for the current x position
    chan_w, chan_h, chan_t, chan_land = engine_info.get_geo(x)

    # Calculate A_gas
    A_gas = calc_A_gas(dx, x, engine_info) # [m^2]

    # Calculate thermal temperature gradient of the wall
    T_grade = (T_hw - T_cw)/chan_t # [K/m]
    #           W/m-K * m^2 * K/m = W

    R_wall = chan_t/(engine_info.k * dx * (chan_w + chan_land)) # [K/W] - Thermal resistance of the wall


    q_wall = engine_info.k * A_gas * T_grade # [W]
    q_wall = (T_hw - T_cw)/R_wall # Calculate the heat transfer through the wall [W]

    # Debug Print
    #print(f'Calculated q_wall: {q_wall} [W] at x = {x} [m] with T_hw = {T_hw} [K] \n T_cw = {T_cw} [K] \n A_gas = {A_gas} [m^2] \n T_grade = {T_grade} [K/m]')
    return q_wall


# Define a class which holds the engine information
class EngineInfo:
    def __init__(self, gam, C_star, M_c, P_c, T_c, Cp, F_Vac, Ncc, combustion_molecules, A_c, A_t, A_e, L_c, x_j, chan_land, chan_w, chan_h, chan_t, gas, mdot_LH2, e, k, mdot_chamber, RD, RU, R1, theta1, thetaD, thetaE):
        # Initialize the engine information
        # Inputs:
        self.gam = gam  # Specific heat ratio
        self.C_star = C_star  # Characteristic velocity in [m/s]
        self.M_c = M_c  # Injector mach number
        self.P_c = P_c  # Chamber pressure in [Pa]
        self.T_c = T_c  # Chamber temperature in [K]
        self.Cp = Cp  # Specific heat at constant pressure in [J/KgK]
        self.F_Vac = F_Vac  # Vacuum thrust in [N]
        self.Ncc = Ncc  # Number of coolant channels
        self.combustion_molecules = combustion_molecules  # Mole fractions of combustion gases
        self.A_c = A_c  # Combustion chamber area [m^2]
        self.A_t = A_t  # Throat area [m^2]
        self.A_e = A_e  # Exit area [m^2]
        self.L_c = L_c  # Chamber length [m]
        self.x_j = x_j  # Array of x values from throat which will be used to calculate the channel geometry [m]
        self.chan_land = chan_land  # Channel land [m]
        self.chan_w = chan_w  # Channel width [m]
        self.chan_h = chan_h  # Channel height [m]
        self.chan_t = chan_t  # Channel thickness [m]
        self.gas = gas # Gas object from Cantera
        self.mdot_LH2 = mdot_LH2  # Mass flow rate of LH2 [kg/s]
        self.mdot_chamber = mdot_chamber # Mass flow rate of the combustion chamber [kg/s]
        self.e = e  # Roughness height of the channel material [m]
        self.k = k # Thermal conductivity of the channel material [W/m*K]
        self.RD = RD # Radius of the throat curve [m]
        self.RU = RU # Radius of the contraction [m]
        self.R1 = R1 # Radius of the expansion [m]
        self.theta1 = theta1
        self.thetaD = thetaD  # Angle of the contraction [rad]  
        self.thetaE = thetaE  # Angle of the expansion [rad]


    # Function to find the closest channel geometry dimensions at a given x value
    def get_geo(self, x):
        # Function which will return the curren channel width, height, and thickness at a given x value
        # Inputs
        # x - type: float - distance from injector [m]
        # Outputs
        # chan_w - type: float - channel width [m]
        # chan_h - type: float - channel height [m]
        # chan_t - type: float - channel thickness [m]

        # Find the closest x value in the array
        idx = (np.abs(np.array(self.x_j) - x)).argmin()
        return self.chan_w[idx], self.chan_h[idx], self.chan_t[idx], self.chan_land[idx] 

    def get_geo_interp(self, x, dx):
        # Function which will return the current channel width, height, and thickness at a given x value
        # Inputs
        # x - type: float - distance from injector [m]
        # Outputs
        # chan_w - type: float - channel width [m]
        # chan_h - type: float - channel height [m]
        # chan_t - type: float - channel thickness [m]

        # Create interpolation functions (linear or cubic as needed)
        land_interp = interp1d(self.x_j, self.chan_land, kind='linear', bounds_error=False, fill_value="extrapolate")
        w_interp = interp1d(self.x_j, self.chan_w, kind='linear', bounds_error=False, fill_value="extrapolate")
        h_interp = interp1d(self.x_j, self.chan_h, kind='linear', bounds_error=False, fill_value="extrapolate")
        t_interp = interp1d(self.x_j, self.chan_t, kind='linear', bounds_error=False, fill_value="extrapolate")

        # Now get interpolated values on your fine grid
        chan_land_fine = land_interp(x)
        chan_w_fine = w_interp(x)
        chan_h_fine = h_interp(x)
        chan_t_fine = t_interp(x)

        return chan_w_fine, chan_h_fine, chan_t_fine, chan_land_fine

    # Function to plot engine channel geometry
    def displayChannelGeometry(self):
        # Inputs:
        # x_j - type: array - Array of x values from throat which will be used to calculate the channel geometry [m]
        # chan_land - type: array - Channel land [m]
        # chan_w - type: array - Channel width [m]
        # chan_h - type: array - Channel height [m]
        # chan_t - type: array - Channel thickness [m]

        plt.figure(figsize=(10, 6))
        plt.plot(np.array(self.x_j), np.array(self.chan_w)*1000, label='Channel Width', color='orange')
        plt.plot(self.x_j, np.array(self.chan_h)*1000, label='Channel Height', color='green')
        plt.plot(self.x_j, np.array(self.chan_t)*1000, label='Channel Thickness', color='red')
        plt.plot(self.x_j, np.array(self.chan_land)*1000, label='Channel Land', color='blue')
        plt.xlabel('Distance from Injector (m)')
        plt.ylabel('Channel Geometry (mm)')
        plt.ylim(0, 7)
        plt.title('Engine Channel Geometry')
        plt.legend()
        plt.grid()
        
    # Function to get radius at a given x
    def get_radius(self, x):
        # Inputs:
        # x - type: float - distance from injector [m]
        # Outputs:
        # radius - type: float - radius at given x [m]
        
        return calc_radius(x, self.A_c, self.A_t, self.A_e, self.L_c)

    # Function to calculate transport properties of the gas
    def calc_transport_properties(self, T, P):
        # Inputs:
        # T - type: float - temperature [K]
        # P - type: float - pressure [Pa]
        # Outputs:
        # Cp_mix - type: float - specific heat of the mixture [J/KgK]
        # viscosity - type: float - viscosity of the mixture [Pa·s]
        # thermal_conductivity - type: float - thermal conductivity of the mixture [W/m-K]
        mole_fractions = {k: v[0] for k, v in self.combustion_molecules.items()}
        self.gas.TPX = T, P, mole_fractions 

        viscosity = self.gas.viscosity                        # Pa·s
        thermal_conductivity = self.gas.thermal_conductivity  # W/m-K
        Cp = self.gas.cp_mass                               # J/kg-K
        Prandtl_number = (Cp * viscosity)/thermal_conductivity # dimensionless

        return viscosity, thermal_conductivity, Cp, Prandtl_number

    # Function to calculate the radius at a given x
    def get_r_value(self, x):
        # Function which will return the current radius at a given x value
        # Inputs
        # x - type: float - distance from injector [m]
        # Outputs
        # radius - type: float - either Ru, RI, RD i
        # Convert to inch
        x = x * 39.3701 # Convert x to inch
        D_t = 2*np.sqrt(self.A_t/np.pi)*39.3701# Calculate throat diameter [inch]
        L_c = self.L_c * 39.3701

        # Determine radius of throat in [inch]
        R_T = D_t/2

        # Define key constants for RS25 engine @TODO need to replace with actual values for own engine
        L_e = 5.339 # Length before chamber starts contracting [inch]
        theta_1 = 25.4157 # Angle of contraction [degrees]
        theta_D = 37 # Angle of expansion [degrees]
        #theta_E = 5.3738 # Angle of exit [degrees]
        R_1 = 1.73921 * R_T # Radius of contraction [inch]
        R_U = 5.1527 # Radius of contraction [inch]
        R_D = 2.019 # Radius of throat curve [inch]


        # Calculate length from throat to exit plane
        if (L_e < x) and (x <= L_e + R_1*np.sin(np.deg2rad(theta_1))):
            radius = R_1
            r_type = 'R1'
        elif (L_c - R_U*np.sin(np.deg2rad(theta_1)) < x) and (x <= L_c):
            radius = R_U
            r_type = 'Ru'
        elif (L_c < x):
            radius = R_D
            r_type = 'Rd'
        else:
            radius = 0
            r_type = 'none'
        return radius * 0.0254, r_type # Convert radius to meters

    # Function to calculate the engine area based on current x location
    def calc_area(self, x):
        # Inputs:
        # x - type: float - distance from injector [m]
        # Outputs:
        # area - type: float - area at given x [m^2]
        
        return np.pi * (self.get_radius(x)**2)  # Calculate area using A = πr²
    

def newton_solve_temperatures(dx, x, y, s, T_LH2, P_LH2, engine_info, 
                              T_hw_init, T_cw_init, 
                              tol=0.1, max_iter=50):
    """
    Solves for T_hw and T_cw using Newton-Raphson method.
    """
    def F(T):
        T_hw, T_cw = T
        q_gas, heatflux, h_gas, A_gas, R_hot, areasurface = calc_q_gas(dx, x, y, T_hw, engine_info)
        q_h2, T_LH2_new, P_LH2_new, dF_dx, h_H2, C3, v, rho_LH2, chan_area, Re, Nu, Dh, therm_LH2 = calc_q_h2(dx, x, y, s, T_cw, T_LH2, P_LH2, engine_info)
        q_wall = calc_q_wall(dx, x, y, T_hw, T_cw, engine_info)

        f1 = q_wall - q_gas
        f2 = q_gas - q_h2
        return np.array([f1, f2])

    def jacobian(T, h=1e-3):
        T_hw, T_cw = T
        J = np.zeros((2, 2))
        for i in range(2):
            T1 = np.array(T)
            T2 = np.array(T)
            T1[i] -= h
            T2[i] += h
            f1 = F(T1)
            f2 = F(T2)
            J[:, i] = (f2 - f1) / (2 * h)
        return J

    # Initial guess
    T = np.array([T_hw_init, T_cw_init])

    for iteration in range(max_iter):
        F_val = F(T)
        #print(f"Iteration {iteration}: F(T) = {F_val} T = {T}")
        if np.linalg.norm(F_val, ord=2) < tol:
            # print(f"Converged in {iteration} iterations.")
            return T

        J = jacobian(T)
        try:
            delta = np.linalg.solve(J, -F_val)
        except np.linalg.LinAlgError:
            raise RuntimeError("Jacobian is singular. Try different initial guesses.")
        
        T += delta

    raise RuntimeError("Newton-Raphson did not converge.")

def plot_bell_nozzle_rrs(
    Rt,
    epsilon,
    Rc,                 # chamber radius (meters)
    Lc,                 # injector-to-throat axial length (meters)
    l_percent=80,
    theta_c_deg=30,
    theta_n_deg=30,
    theta_e_deg=15,
    r_conv_mult=1.75,         # large converging radius (f2) = r_conv_mult*Rt
    r_throat_conv_mult=1.5,   # converging-side throat fillet (f5)
    r_throat_div_mult=0.382,  # diverging-side throat fillet (f4)
    num_points=200,
    display_units='mm'         # 'm' or 'mm' for plotting/printing only
):
    # Unit scaling for display (internal calculations remain in meters)
    if str(display_units).lower() == 'mm':
        sf = 1000.0
        unit_str = 'mm'
    else:
        sf = 1.0
        unit_str = 'm'

    th_c = np.deg2rad(theta_c_deg)
    th_n = np.deg2rad(theta_n_deg)
    th_e = np.deg2rad(theta_e_deg)

    Re = np.sqrt(epsilon) * Rt
    Lrao = ((np.sqrt(epsilon) - 1.0) * Rt) / np.tan(np.deg2rad(15.0))
    Ln   = (l_percent/100.0) * Lrao

    # Throat fillets around x ≈ 0
    r1 = r_throat_conv_mult * Rt
    ang1 = np.linspace(-(90+theta_c_deg), -90, num_points) * np.pi/180.0
    x_f5 = r1*np.cos(ang1)
    y_f5 = r1*np.sin(ang1) + (r1 + Rt)
    xP4, yP4 = x_f5[0], y_f5[0]

    r2 = r_throat_div_mult * Rt
    ang2 = np.linspace(-90, theta_n_deg-90, num_points) * np.pi/180.0
    x_f4 = r2*np.cos(ang2)
    y_f4 = r2*np.sin(ang2) + (r2 + Rt)
    Nx, Ny = x_f4[-1], y_f4[-1]         # start of bell
    Ex, Ey = Ln, Re                     # exit

    # Converging geometry: big arc (f2) tangent to straight chamber and to -theta_c line at P4
    rc = r_conv_mult * Rt
    mc = -np.tan(th_c)
    bc = yP4 - mc*xP4
    yc = Rc - rc
    s = np.sqrt(mc*mc + 1.0)
    xc_plus  = (yc - bc + rc*s)/mc
    xc_minus = (yc - bc - rc*s)/mc
    xc = xc_minus if xc_minus < xc_plus else xc_plus  # upstream center x

    # Projection (tangency) point P3 on the -theta_c line (not used for f1 length, but used for the short connector)
    S = mc*xc - yc + bc
    xP3 = xc - mc * S / (mc*mc + 1.0)
    yP3 = yc + S / (mc*mc + 1.0)

    # Circle param for f2 from top point (P2) down to P3
    phi_top = np.pi/2.0                    # top point is tangent to the straight (y=Rc), at x = xc
    psi_c   = np.arctan(mc)
    phi3    = psi_c + np.pi/2.0
    phi_f2 = np.linspace(phi_top, phi3, num_points)
    x_f2 = xc + rc*np.cos(phi_f2)
    y_f2 = yc + rc*np.sin(phi_f2)

    # Short straight connector P3 -> P4 along -theta_c
    x_f3 = np.linspace(xP3, xP4, max(2, num_points//8))
    y_f3 = mc*(x_f3 - xP4) + yP4

    # IMPORTANT: fix straight-chamber length to satisfy injector-to-throat Lc
    # throat plane is at x=0, tangency (start of f2) is at x=xc (negative)
    L_conv = -xc                      # axial distance from P2 to throat
    L_straight = Lc - L_conv          # what remains for the straight section
    if L_straight > 1e-12:
        x_f1 = np.linspace(-Lc, xc, max(2, num_points//3))
        y_f1 = np.full_like(x_f1, Rc)
    else:
        # No room left for a straight section; start directly with the converging arc
        x_f1 = np.array([])
        y_f1 = np.array([])

    # Bell (quadratic Bézier) from N to E with imposed slopes theta_n and theta_e
    m1, m2 = np.tan(th_n), np.tan(th_e)
    C1 = Ny - m1*Nx
    C2 = Ey - m2*Ex
    Qx = (C2 - C1) / (m1 - m2)
    Qy = (m1*C2 - m2*C1) / (m1 - m2)
    t = np.linspace(0, 1, num_points)
    x_f6 = (1 - t)**2 * Nx + 2*(1 - t)*t * Qx + t**2 * Ex
    y_f6 = (1 - t)**2 * Ny + 2*(1 - t)*t * Qy + t**2 * Ey

    # Compose polyline (meters)
    x = np.concatenate([x_f1, x_f2, x_f3, x_f5, x_f4, x_f6])
    y = np.concatenate([y_f1, y_f2, y_f3, y_f5, y_f4, y_f6])

    # Plot using display units (stretch y-axis)
    fig, ax = plt.subplots(figsize=(9, 7))  # taller figure to stretch y
    if x_f1.size:
        ax.plot(x_f1*sf, y_f1*sf, '-r', lw=2, label='f1: straight chamber')
    ax.plot(x_f2*sf, y_f2*sf, '-b', lw=2, label='f2: converging arc (rc)')
    ax.plot(x_f3*sf, y_f3*sf, '-c', lw=2, label='(short line to throat)')
    ax.plot(x_f5*sf, y_f5*sf, '-m', lw=2, label=f'f5: throat fillet (conv) r={r_throat_conv_mult}Rt')
    ax.plot(x_f4*sf, y_f4*sf, '-g', lw=2, label=f'f4: throat fillet (div) r={r_throat_div_mult}Rt')
    ax.plot(x_f6*sf, y_f6*sf, '-y', lw=2, label='f6: bell (Bézier)')

    # Reference
    ax.scatter([0], [Rt*sf], c='k', s=25, zorder=5, label='Throat (Rt)')
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.6)

    # Let y stretch freely (avoid equal aspect squashing)
    ax.set_aspect('equal', adjustable='datalim')
    ax.autoscale()
    ax.margins(y=0.01)  # a bit more vertical padding

    ax.set_xlabel(f'Axial (x) [{unit_str}]')
    ax.set_ylabel(f'Radius (y) [{unit_str}]')
    ax.set_title(f'RRS-style Bell Nozzle Contour ({unit_str} display)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()

    # Report lengths in display units
    print(f"Lc (injector to throat): {Lc*sf:.3f} {unit_str}")
    print(f"Converging axial length (P2->throat): {L_conv*sf:.3f} {unit_str}")
    print(f"Straight-chamber length used: {max(0.0, L_straight)*sf:.3f} {unit_str}")
    print(f"Bell length (N->E): {Ln*sf:.3f} {unit_str}")
    print(f"Overall nozzle length: {(Ln + max(0.0, L_straight) + L_conv)*sf:.3f} {unit_str}")
    if L_straight <= 0:
        print("Note: Lc is shorter than the converging length; straight section omitted.")

    # Return original (meters) arrays to keep API stable
    return x, y

def main():
    # Main function

    # Variable to show plots
    showPlots = False

    # Constants RS25
    # gam = 1.19346 #1.168 # Specific heat ratio  - CEA data
    # C_star = 2300 #1783 # Characteristic velocity in [m/s] - CEA data
    # P_c = 18.23E6 #2E6 # Chamber stagnation pressure in [Pa] - CEA data
    # T_c = 3542 #3254 # Chamber stagnation temperature in [K] - CEA data
    # Cp = 3.71330E3	 # Specific heat at constant pressure in [J/KgK] - CEA data
    # F_Vac = 2184076.8131#5200 # Vacuum thrust in [N] - CEA data
    # M_c = 0.26419 # Injector mach number - calculated by trial and error, testfile.py has a good example of how to do this
    # Ncc = 390#430.0 # Number of coolant channels - guessed 
    # e = 2.5E-7 # Channel wall roughness of Narloy Z [m]
    # k = 316  # Thermal conductivity of Narloy Z [W/m*K] 
    # mdot_LH2 = 13.2 # kg/s mass flow rate of the LH2 in the coolant channels. 
    # meltingpoint = 1000 # Melting point of inconel Z [K]
    # expRatio = 69.5 # Nozzle expansion ratio
    # contChamber = 2.7 # Chamber contraction ratio
    # L_star = 36 # L*
    # RU = 1.5 # Radius of contraction 
    # RD = 0.385 # Radius of throat curve
    # R1 = 1.73921 # Radius of entry curve
    # theta1 = 25 #25.4157 # Radius of throat angle 
    # thetaD = 37 # Angle of expansion [degrees]
    # thetaE =  5.3738 # half-angle of the nozzle [degrees]

    # Constants 5kN Enginer LOX/RP1
    gam = 1.168 # Specific heat ratio  - CEA data
    C_star = 1783 # Characteristic velocity in [m/s] - CEA data
    P_c = 2E6 # Chamber stagnation pressure in [Pa] - CEA data
    T_c = 3254 # Chamber stagnation temperature in [K] - CEA data
    Cp = 3.71330E3	 # Specific heat at constant pressure in [J/KgK] - CEA data
    F_Vac = 5200 # Vacuum thrust in [N] - CEA data
    M_c = 0.26419 # Injector mach number - calculated by trial and error, testfile.py has a good example of how to do this
    Ncc = 390#430.0 # Number of coolant channels - guessed 
    e = 2.5E-7 # Channel wall roughness of Narloy Z [m]
    k = 316  # Thermal conductivity of Narloy Z [W/m*K] 
    mdot_LH2 = 13.2 # kg/s mass flow rate of the LH2 in the coolant channels. 
    meltingpoint = 1000 # Melting point of inconel Z [K]
    expRatio = 9 # Nozzle expansion ratio
    contChamber = 7 # Chamber contraction ratio
    L_star = 43 # L*
    RU = 1.5 # Radius of contraction 
    RD = 0.385 # Radius of throat curve
    R1 = 1.73921 # Radius of entry curve
    theta1 = 25 #25.4157 # Radius of throat angle 
    thetaD = 37 # Angle of expansion [degrees]
    thetaE =  5.3738 # half-angle of the nozzle [degrees]
    theta_c_deg = 30 # nozzle contraction half angle





    # Define the molecules in the combustion gases and their molecular masses (Get this from CEA data)
    combustion_molecules = {'H2' : [0.2517, 2.01588E-3], 'O2' : [0.0074,31.9988E-3], 'H2O' :[0.6724,18.01528E-3], 'OH' : [0.036,17.00734E-3], 'H' : [0.02829,1.00794E-3], 'O' : [0.004,15.9994E-3]} # Mole fractions of combustion gases from CEA data
    
     # Create gas object
    gas = ct.Solution('gri30.yaml')


    # Calculate the engine geometry based off CEA data 
    A_c, A_t, A_e, L_c, L_e, mdot_chamber, Vc = engine_geometry(gam, P_c, 101325, T_c, F_Vac, expRatio, contChamber, L_star, RU, RD, thetaE, False, False)

    D_t = np.sqrt((4*A_t[0])/np.pi) # Calculate throat diameter [m]
    R_t = D_t/2 # Calculate throat radius [m]
    D_c = np.sqrt((4*A_c[0])/np.pi) # Calculate chamber diameter [m]
    R_c = D_c/2 # Calculate chamber radius [m]
    D_e = np.sqrt((4*A_e[0])/np.pi) # Calculate exit diameter [m]
    R_e = D_e/2 # Calculate exit radius [m]

    L_cone = (R_t*(np.sqrt(contChamber)-1) + (RU*R_t)*(1/np.cos(np.deg2rad(theta_c_deg)) -1))/(np.tan(np.deg2rad(theta_c_deg))) # Convergent cone length
    V_cone = np.pi/3 * L_cone*(R_c**2 + R_t**2 + (R_t*R_c))
    V_chamber = Vc[0] - V_cone
    L_chamber = V_chamber/(contChamber*A_t[0])
    L_c = L_chamber + L_cone # Total chamber length
   
    # Plot rocket geometry and get x-axis and radius values
    x, y = plot_bell_nozzle_rrs(R_t, expRatio, R_c, L_c, l_percent=80, theta_c_deg=20, theta_n_deg=30, theta_e_deg=12, r_conv_mult=1.75, r_throat_conv_mult=1.5, r_throat_div_mult=0.382, num_points=120)
    
    print(f'Calculated Engine Geometry: \n A_c: {A_c[0]} [m^2] D_c {D_c} [m] R_c {R_c} [m] \n A_t: {A_t} [m^2] D_t {D_t} [m] Rt {R_t} [m] \n A_e: {A_e[0]} [m^2] D_e: {D_e} [m] R_e: {R_e} [m] \n L_c: {L_c} [m] \n L_e [m] {L_e[0]} \n mdot_chamber: {mdot_chamber[0]} [kg/s]')
    
    

    

    # Define chamber channel geometry  
    # Loading geometry data into the engine channels classx
    #x_j = [-35.56, -34.29, -32.41, -30.48, -27.94, -25.4, -22.86, -20.32, -17.78, -15.24, -12.7, -10.16, -8.89, -7.62, -6.35, -5.08, -3.81, -3.048, -2.54, -1.27, 0, 2.54, 5.08, 7.62, 10.16, 12.7, 15.24, 17.78, 20.32, 24.71] # Array of x values from throat which will be used to calculate the channel geometry [m]
    # x_j_subtracted = [(x+35.56)/100 for x in x_j]
    # print(x_j_subtracted)
    x_j =       [0.0,      0.0127,   0.0315,   0.0508,   0.0762,   0.1016,   0.127,    0.1524,   0.1778,   0.2032,   0.2286,   0.254,    0.2667,   0.2794,   0.2921,   0.3048,   0.3175,   0.32512,  0.3302,   0.3429,   0.3556,   0.381,    0.4064,   0.4318,   0.4572,   0.4826,   0.508,    0.5334,   0.5588,   0.6027]#[0.000000, 0.012700, 0.031500, 0.050800, 0.076200, 0.101600, 0.127000, 0.152400, 0.177800, 0.203200, 0.228600, 0.241300, 0.254000, 0.266700, 0.279400, 0.292100, 0.304800, 0.317500, 0.325120, 0.330200, 0.342900, 0.355600, 0.381000, 0.406400, 0.431800, 0.457200, 0.482600, 0.508000, 0.533400, 0.558800, 0.602700] # Array of x values from throat which will be used to calculate the channel geometry [m]
    # Channel  land [m]
    chan_land = [0.002068, 0.002068, 0.002068, 0.002068, 0.002068, 0.002045, 0.001976, 0.001857, 0.001748, 0.001844, 0.001847, 0.001653, 0.001562, 0.001463, 0.001361, 0.001275, 0.001196, 0.001261, 0.001143, 0.001113, 0.001105, 0.001209, 0.001516, 0.001603, 0.001554, 0.001844, 0.002131, 0.002405, 0.002685, 0.003155] # Array of channel land widths [m]
    # Channel width [m]
    chan_w =    [0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001509, 0.001217, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001227, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575]  # Array of channel widths [m] 
    # Channel height [m]
    chan_h =    [0.002489, 0.002489, 0.002489, 0.002489,  0.003175, 0.003175, 0.003175, 0.003175, 0.003284, 0.003792, 0.004338, 0.003231, 0.003987, 0.002728, 0.002812, 0.002977, 0.002982, 0.003117, 0.003147, 0.003307, 0.003442, 0.004318, 0.004953, 0.005352, 0.005093, 0.005474, 0.005817, 0.006096, 0.006096, 0.006096] # Array of channel heights [m]
    # Channel thickness [m]
    chan_t =    [0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889]  # Array of channel thicknesses from hot side to cold side [m]

    # Channel width [m]
    chan_w =    [0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001509, 0.001217, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001227, 0.001575, 0.001575, 0.001575]  # Array of channel widths [m] 
    # Channel height [m]
    chan_h =    [0.002489, 0.002489, 0.002489, 0.002489,  0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.003442, 0.004953, 0.004953, 0.004953, 0.004953, 0.005352, 0.006096, 0.006096, 0.006096] # Array of channel heights [m]
    # Channel thickness [m]
    chan_t =    [0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 00.000711, 0.000711, 0.0007119, 0.000889, 0.000889]  # Array of channel thicknesses from hot side to cold side [m]

    # Create an instance of the EngineInfo class to store all engine information in a single object NOTE: All values are in SI units
    engine_info = EngineInfo(gam, C_star, M_c, P_c, T_c, Cp, F_Vac, Ncc, combustion_molecules, A_c[0], A_t[0], A_e[0], L_c, x_j, chan_land, chan_w, chan_h, chan_t, gas, mdot_LH2, e, k, mdot_chamber, RD, RU, R1, theta1, thetaD, thetaE)
    
    # Display the channel geometry
    engine_info.displayChannelGeometry() # Display the engine channel geometry
    
    # Calculate flow data (calcualte flow data throughout entire engine at each x value (distance in m from injector) and step size in m)
    # xi = 0 # Start of chamber [m]
    # xf = 24.29*0.0254 #135.5 # End of chamber [m]
    # dx = (24.29/1000)*0.0254 # Step size [m]

    # array_length = int(xf/dx) # Length of the arrays to hold the flow data
    # Initial conditons for thermal analysis set F and Q to zero for isentropic condition.
    # dF_dx = np.zeros(array_length)  # Thrust gradient array
    # dQ_dx = np.zeros(array_length) # Heat transfer gradient 
    # Key data to pass to ODE solver
    # keydata = [engine_info.A_c, engine_info.A_t, engine_info.A_e, engine_info.L_c, engine_info.gam, engine_info.Cp, dF_dx, dQ_dx] # Key data to pass to ODE solver

    # dx, xp_m, yp_m = calc_flow_data(xi, xf, dx, engine_info.M_c, engine_info.P_c, engine_info.T_c, keydata) # returns pressure, temperature, mach number throughout  entire engine at each x value (distance in m from injector) and step size in m
    

    # create_plot([xp for xp in xp_m], [np.sqrt(y[0]) for y in yp_m], "Distance from Injector [m]", "Mach Number", "Mach Number vs Distance from Injector")
    # create_plot([xp for xp in xp_m], [y[1] for y in yp_m], "Distance from Injector [m]", "Pressure [Pa]", "Pressure vs Distance from Injector")
    # create_plot([xp for xp in xp_m], [y[2] for y in yp_m], "Distance from Injector [m]", "Temperature [K]", "Temperature vs Distance from Injector")

    # xp_m = np.arange(xi, xf, dx)  # Create an array of x values from xi to xf with step size dx

    # displayEngineGeometry2(xp_m, A_c, A_t, A_e, L_c)
    plt.show()

    
# #----------------------------------------------------------------------------------------------
#     # Define start conditions for thermal analysis
#     T_hw_init = 500.0 # Wall temperature at the injector [K] - This is a guess, can be changed later
#     T_cw_init = 300.0 # Coolant temperature at the injector [K] - This is a guess, can be changed later
#     T_LH2_init = 52 # Liquid hydrogen temperature in Kelvin at inlet of coolant channel
#     P_LH2_init = 4.482E7 #38.93E6 # Liquid hydrogen pressure in Pascals at inlet of coolant channel
#     #----------------------------------------------------------------------------------------------
    
#     # Initialize arrays with initial conditions for thermal analysis
#     T_hw_array = np.zeros(array_length) # Wall temperature at the injector [K]
#     T_cw_array = np.zeros(array_length) # Coolant temperature at the injector [K]
#     T_LH2_array = np.zeros(array_length) # Liquid hydrogen temperature in Kelvin at inlet of coolant channel
#     P_LH2_array = np.zeros(array_length) # Liquid hydrogen pressure in Pascals at inlet of coolant channel   
#     h_H2_array = np.zeros(array_length) # Heat transfer coefficient for liquid hydrogen in [W/m^2K]    
    
#     heatflux_hotside = np.zeros(array_length) # Heat flux on the hot side of the wall [W/m^2]
#     q_h2_array = np.zeros(array_length) # Heat transfer through coolant channels [W]
#     q_gas_array = np.zeros(array_length) # Heat transfer from gas to wall [W]
#     C3_array = np.zeros(array_length) # Heat transfer coefficient for liquid hydrogen in [W/m^2K]
#     h_gas_array = np.zeros(array_length) # Heat transfer coefficient for gas in [W/m^2K]
#     v_fluid_array = np.zeros(array_length) # Fluid velocity in the coolant channels [m/s]
#     rho_LH2_array = np.zeros(array_length) # Liquid hydrogen temperature in Kelvin at inlet of coolant channel
#     chan_area_array = np.zeros(array_length) # Channel area in the coolant channels [m^2]
#     ReynoldsNum_array = np.zeros(array_length) # Reynolds number in the coolant channels [dimensionless]
#     NusseltCold_array = np.zeros(array_length) # Nusselt number in the coolant channels [dimensionless]
#     A_gas_array = np.zeros(array_length) # Gas area in the coolant channels [m^2]
#     R_hot_array = np.zeros(array_length) # Hot side radius in the coolant channels [m]
#     areasurface_array = np.zeros(array_length) # Surface area in the coolant channels [m^2]
#     T_hw_array_prev = np.zeros(array_length) # Previous wall temperature at the injector [K]
#     T_cw_array_prev = np.zeros(array_length) # Previous coolant temperature at the injector [K]
#     error_T_hw_array = np.zeros(array_length) # Error in wall temperature at the injector [K]
#     error_T_cw_array = np.zeros(array_length) # Error in coolant temperature at the injector [K]
#     Dh_array = np.zeros(array_length) # Hydraulic diameter in the coolant channels [m]
#     therm_LH2_array = np.zeros(array_length) # Thermal conductivity of liquid hydrogen in the coolant channels [W/mK]
#     melting_point_array = np.full(array_length, meltingpoint)  # Melting point array [K]
#     T_hw_matrix = [[]]
#     # Initialize the first values of T_hw, T_cw, T_LH2, P_LH2
    

#     s = dx # Initialize s, the distance along the coolant channel, to zero
#     # x = xp_m[-1]
#     # y = yp_m[-1]  # Get the last y value for the last x value

#     # q_gas, heatflux = calc_q_gas(dx, x, y, T_hw, engine_info) # Calculate heat transfer from gas to wall
#     # q_h2, T_LH2, P_LH2, dF_dx_val, h_h2 = calc_q_h2(dx, x, y, s, T_cw, T_LH2, P_LH2, engine_info) # Calculate heat transfer through coolant channels
#     # q_wall = calc_q_wall(dx, x, y, T_hw, T_cw, engine_info) # Calculate heat transfer through the wall

#     # print(f'q_gas {q_gas} [W] \n q_h2 = {q_h2} [W] \n q_wall = {q_wall} [W] \n at x = {x} [m] with T_hw = {T_hw} [K] with T_cw = {T_cw} [K] \n T_LH2_new = {T_LH2} [K] \n P_LH2_new = {P_LH2} [Pa] \n dF_dx_val = {dF_dx_val} [N/m]')

#     # T = newton_solve_temperatures(dx, x, y, s, T_LH2, P_LH2, engine_info, T_hw, T_cw, tol=0.1, max_iter=50)

#     # T_hw, T_cw = T
#     T_cw_error = 100 # Initialize the error in the coolant temperature array
#     T_hw_error = 100 # Initialize the error in the wall temperature array
#     # q_gas, heatflux = calc_q_gas(dx, x, y, T_hw, engine_info) # Calculate heat transfer from gas to wall
#     # q_h2, T_LH2, P_LH2, dF_dx_val, h_h2 = calc_q_h2(dx, x, y, s, T_cw, T_LH2, P_LH2, engine_info) # Calculate heat transfer through coolant channels
#     # q_wall = calc_q_wall(dx, x, y, T_hw, T_cw, engine_info) # Calculate heat transfer through the wall
#     iter = 0
#     # print(f'q_gas {q_gas} [W] \n q_h2 = {q_h2} [W] \n q_wall = {q_wall} [W] \n at x = {x} [m] with T_hw = {T_hw} [K] with T_cw = {T_cw} [K] \n T_LH2_new = {T_LH2} [K] \n P_LH2_new = {P_LH2} [Pa] \n dF_dx_val = {dF_dx_val} [N/m]')
#     while T_cw_error > 10 and T_hw_error > 10: # Run until the error in the coolant temperature array and wall temperature array is less than 0.1 K

#         T_hw = T_hw_init # Initialize the hot wall temperature
#         T_cw = T_cw_init
#         T_LH2 = T_LH2_init # Initialize the liquid hydrogen temperature
#         P_LH2 = P_LH2_init
#         T_cw_array_prev = T_cw_array.copy() # Store the previous coolant temperature array
#         T_hw_array_prev = T_hw_array.copy() # Store the previous wall temperature array

#         for i, (x,y) in enumerate(zip(reversed(xp_m), reversed(yp_m))):
#             # Store the values for the current x value
            
#             # Run the thermal analysis for current x value, run until q_gas, q_h2, and q_wall have converged by adjusting T_hw, T_cw
#             T = newton_solve_temperatures(dx, x, y, s, T_LH2, P_LH2, engine_info, T_hw, T_cw, tol=0.1, max_iter=50)

#             # Unpack the new temperatures
#             T_hw, T_cw = T
            

#             # Calculate Heat Transfer Information based off convergence values, overwrite T_LH2 and P_LH2 with new values
#             q_gas,heatflux, h_gas, A_gas, R_hot, areasurface = calc_q_gas(dx, x, y, T_hw, engine_info)
#             q_h2, T_LH2, P_LH2, dF_dx_val, h_H2,C3,v, rho_LH2, chan_area, Re, Nu, Dh, therm_LH2 = calc_q_h2(dx, x, y, s, T_cw, T_LH2, P_LH2, engine_info) # Calculate heat transfer through coolant channels
#             q_wall = calc_q_wall(dx, x, y, T_hw, T_cw, engine_info) # Calculate heat transfer through the wall
            
#             # Calculate dF_dx and dQ_dx based on the heat transfer through the coolant channels
#             dQ_dx[i] = dQ_dx[-1] + (q_gas/mdot_chamber)
#             dF_dx[i] = dF_dx_val
#             s+=dx
            

#             T_LH2_array[i] = T_LH2
#             T_hw_array[i] = T_hw
#             T_cw_array[i] = T_cw
#             P_LH2_array[i] = P_LH2
#             heatflux_hotside[i] = heatflux # Store the heat flux on the hot side of the wall [W/m^2]
#             v_fluid_array[i] = v # Store the fluid velocity in the coolant channels [m/s]
#             q_h2_array[i] = q_h2 # Store the heat transfer through coolant channels [W]
#             h_H2_array[i] = h_H2 # Store the heat transfer coefficient for liquid hydrogen in [W/m^2K]
#             C3_array[i] = C3 # Store the heat transfer coefficient for liquid hydrogen in [W/m^2K]
#             q_gas_array[i] = q_gas # Store the heat transfer from gas to wall [W]
#             h_gas_array[i] = h_gas # Store the heat transfer coefficient for gas in [W/m^2K]
#             rho_LH2_array[i] = rho_LH2 # Store the density of liquid hydrogen in [kg/m^3]
#             chan_area_array[i] = chan_area # Store the channel area in the coolant channels [m^2]
#             ReynoldsNum_array[i] = Re # Store the Reynolds number in the coolant channels [dimensionless]
#             NusseltCold_array[i] = Nu # Store the Nusselt number in the coolant channels [dimensionless]
#             A_gas_array[i] = A_gas # Store the gas area in the coolant channels [m^2]
#             R_hot_array[i] = R_hot # Store the hot side radius in the coolant channels [m]
#             areasurface_array[i] = areasurface # Store the surface area in the coolant channels [m^2]
#             Dh_array[i] = Dh # Store the hydraulic diameter in the coolant channels [m]
#             therm_LH2_array[i] = therm_LH2 # Store the thermal conductivity of liquid hydrogen in the coolant channels [W/mK]
#             # print(f'q_gas: {q_gas} [W] \n q_h2 = {q_h2} [W] \n q_wall = {q_wall} [W] \n at x = {x} [m] with T_hw = {T_hw} [K] with T_cw = {T_cw} [K] \n T_LH2_new = {T_LH2} [K] \n P_LH2_new = {P_LH2} [Pa] \n T_LH2_new = {T_LH2 - T_LH2_array[-1] } [K] \n P_LH2_new = {P_LH2_array[-1] - P_LH2} [Pa] \n dF_dx = {dF_dx[i]} [N/m] \n dQ_dx = {dQ_dx[i]} [W-s/kg] \n')
#             # Re-run the flow data calculate with updated dF_dx and dQ_dx values
        

#         for b in range(len(T_cw_array)):
#             error_T_cw_array[b] = T_cw_array[b] - T_cw_array_prev[b] # Calculate the error in the coolant temperature array
#             error_T_hw_array[b] = T_hw_array[b] - T_hw_array_prev[b] # Calculate the error in the wall temperature array
#         T_cw_error = np.mean(np.abs(error_T_cw_array)) # Calculate the mean error in the coolant temperature array
#         T_hw_error = np.mean(np.abs(error_T_hw_array)) # Calculate the mean error in the wall temperature array
        
#         print(f'Iteration {iter+1} complete, T_cw error: {np.mean(np.abs(error_T_cw_array))} [K], T_hw error: {np.mean(np.abs(error_T_hw_array))} [K]') # Print the error in the coolant and wall temperatures
#         keydata = [engine_info.A_c, engine_info.A_t, engine_info.A_e, engine_info.L_c, engine_info.gam, engine_info.Cp, dF_dx, dQ_dx] # Key data to pass to ODE solver
#         dx, xp_m, yp_m = calc_flow_data(xi, xf, dx, engine_info.M_c, engine_info.P_c, engine_info.T_c, keydata) # returns pressure, temperature, mach number throughout  entire engine at each x value (distance in m from injector) and step size in m
#         iter += 1

#     create_plot([x for x in xp_m], [T_LH2 for T_LH2 in reversed(T_LH2_array)], "Distance from Injector [m]", "T_LH2 [K]", "T_LH2 vs Distance from Injector")
#     create_plot([x for x in xp_m], [P_LH2 for P_LH2 in reversed(P_LH2_array)], "Distance from Injector [m]", "P_LH2 [Pa]", "P_LH2vs Distance from Injector")
#     create_plot([x for x in xp_m], [h_H2 for h_H2 in reversed(h_H2_array)], "Distance from Injector [m]", "T_cw [K]", "h_cold vs x Distance from Injector")
#     create_plot([x for x in xp_m], [therm_LH2 for therm_LH2 in reversed(therm_LH2_array)], "Distance from Injector [m]", "Thermal Conductivity [W/mK]", "Thermal Conductivity vs Distance from Injector")
#     # create_plot([x for x in xp_m], [A_gas for A_gas in reversed(A_gas_array)], "Distance from Injector [m]", "A_gas [m^2]", "A_gas vs Distance from Injector")
#     # create_plot([x for x in xp_m], [R_hot for R_hot in reversed(R_hot_array)], "Distance from Injector [m]", "R_hot [W/m-K]", "R_hot vs Distance from Injector")
#     # create_plot([x for x in xp_m], [areasurface for areasurface in reversed(areasurface_array)], "Distance from Injector [m]", "Surface Area [m^2]", "Surface Area vs Distance from Injector")
#     # create_plot([x for x in xp_m], [v for v in reversed(v_fluid_array)], "Distance from Injector [m]", "LH2 Channel Velocity [m/s]", "Fluid Velocity vs Distance from Injector")
#     # create_plot([x for x in xp_m], [chan_area for chan_area in reversed(chan_area_array)], "Distance from Injector [m]", "Channel Area [m^2]", "Channel Area vs Distance from Injector")
#     # create_plot([x for x in xp_m], [Re for Re in reversed(ReynoldsNum_array)], "Distance from Injector [m]", "Reynolds Number", "Reynolds Number vs Distance from Injector")
#     create_plot([x for x in xp_m], [Nu for Nu in reversed(NusseltCold_array)], "Distance from Injector [m]", "Nusselt Number", "Nusselt Number vs Distance from Injector")
#     create_plot([x for x in xp_m], [Dh for Dh in reversed(Dh_array)], "Distance from Injector [m]", "Hydraulic Diameter [m]", "Hydraulic Diameter vs Distance from Injector")
#     # create_plot([x for x in xp_m], [rho_LH2 for rho_LH2 in reversed(rho_LH2_array)], "Distance from Injector [m]", "LH2 Density [kg/m^3]", "LH2 Density vs Distance from Injector")
#     # create_plot([x for x in xp_m], [C3 for C3 in reversed(C3_array)], "Distance from Injector [m]", "C3 [W/m^2K]", "C3 vs Distance from Injector")
#     # Plot T_hw and T_cw on the same plot for comparison
#     plt.figure()
#     plt.plot([x for x in xp_m], [T_cw for T_cw in reversed(T_cw_array)], label="T_cw [K]")
#     plt.plot([x for x in xp_m], [T_hw for T_hw in reversed(T_hw_array)], label="T_hw [K]")
#     plt.plot([x for x in xp_m], [meltingpoint for meltingpoint in reversed(melting_point_array)], label="Melting Point [K]", linestyle='--', color='red')
#     plt.xlabel("Distance from Injector [m]")
#     plt.ylabel("Temperature [K]")
#     plt.title("T_hw and T_cw vs Distance from Injector")
#     plt.legend()
#     plt.grid(True)
#     create_plot([x for x in xp_m], [qflux_hot for qflux_hot in reversed(heatflux_hotside)], "Distance from Injector [m]", "heatflux [w/m^2]", "heatflux vs Distance from Injector")
#     # create_plot([x for x in xp_m], [q_LH2 for q_LH2 in reversed(q_h2_array)], "Distance from Injector [m]", "q_h2 [W]", "q_h2 vs Distance from Injector")
#     create_plot([xp for xp in xp_m], [np.sqrt(y[0]) for y in yp_m], "Distance from Injector [m]", "Mach Number", "Mach Number vs Distance from Injector")
#     create_plot([xp for xp in xp_m], [y[1] for y in yp_m], "Distance from Injector [m]", "Pressure [Pa]", "Pressure vs Distance from Injector")
#     create_plot([xp for xp in xp_m], [y[2] for y in yp_m], "Distance from Injector [m]", "Temperature [K]", "Temperature vs Distance from Injector")
#     # create_plot([x for x in xp_m], [q_gas for q_gas in reversed(q_gas_array)], "Distance from Injector [m]", "q_gas [W]", "q_gas vs Distance from Injector")
#     create_plot([x for x in xp_m], [h_gas for h_gas in reversed(h_gas_array)], "Distance from Injector [m]", "h_gas [W/m^2K]", "h_gas vs Distance from Injector")
#     plt.show()

#     print("Thermal analysis complete.")
#     print('-' * 50)
#     print(f'Peak Hot Wall Temperature: {max(T_hw_array):.2f} K')
#     print(f'Peak Cold Wall Temperature: {max(T_cw_array):.2f} K')
#     # Print peak heat flux on hot side in engineering notation (e.g., 1.23E+5)
#     print(f'Peak heatflux on hot side: {max(heatflux_hotside):.4E} W/m^2')
#     print(f'Peak hot wall heat transfer coefficient: {max(h_gas_array):.2f} W/m^2K')
#     print(f'Delta P: {P_LH2_array[0] - P_LH2_array[-1]:.2E} Pa')
#     print(f'Delta T: {T_LH2_array[-1] - T_LH2_array[0]:.2f} K')
    
        
plt.show()

main()