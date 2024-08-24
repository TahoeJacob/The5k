import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as scipy
from HydrogenModelV2 import hydrogen_thermodynamics
from HydrogenModelV2 import para_fraction

# Import formatted data in excel format. Use this to complete data analaysis


# Import excel file
df = pd.read_excel('CEAParsed.xlsx')

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

# Function to Calculate the mach number at the exit of the nozzle
def calc_exit_mach_num(area_ratio, gam, M_e):
    # Inputs:
    # area_ratio_array - type: array - the area ratio of the nozzle (Ae/At)
    # gam - type: float - specific heat ratio
    # M_e - type: float - exit mach number initial guess

    def equation(Me, gam, area_ratio):
        #return ((specific_heat_ratio+1)/2)**(-c) * ((1 + ( (specific_heat_ratio-1)/2) * (M_e**2))/M_e)**c - area_ratio
        return  (((gam+1)/2)**(-((gam+1)/(gam-1)/2)))* (1 / Me) * (1 + Me**2 * (gam-1)/2)**((gam+1)/(gam-1)/2) - area_ratio
        
    return scipy.fsolve(equation, M_e, args=(gam, area_ratio))           
    
# Function to calculate the exit pressure of the nozzle
def calc_exit_pressure(P_c, gam, M_e):
    # Inputs:
    # P_c - type: float - chamber pressure
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
    A_c = A_c_cm / 100**2 # Convert to m^2

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
def engine_geometry(gam, P_c, T_c, F_Vac, showPlots):
    """
    Calculate the geometry of a rocket engine based on key parameters from CEA.

    Parameters:
    gam (float): Specific heat ratio.
    P_c (float): Chamber pressure.
    T_c (float): Chamber temperature.
    F_Vac (float): Vacuum thrust.
    showPlots (bool): Flag to indicate whether to show plots.

    Returns:
    tuple: A tuple containing the chamber area, throat area, exit area, and chamber length units in [m].
    """

    # create area_ratio_array from 2 to 80 in steps of 0.1
    area_ratio_array = np.arange(1.1, 80, 0.1)

    # First step is to calculate mach numbers in the exit
    M_e_array = [] # Create array to store exit mach numbers
    M_e = 1.2 # Initial guess for exit mach number

    # Second step is to calculate exit pressures
    P_e_array = [] # Create array to store exit pressures

    # Third step is to calculate the nozzle’s vacuum thrust coefficient
    C_F_array = [] # Create array to store vacuum thrust coefficients

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

        # Calculate throat area
        A_t_array.append(calc_nozzle_throat_area(F_Vac, C_F_array[-1], P_c))

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
    position_65 = np.where(np.isclose(area_ratio_array, 69.5))[0] # 69.5 is area ratio for RS25 engine select based off new engine design
    if showPlots:
        print(f'Exit Mach Num: {M_e_array[position_65[0]]} \n Exit Pressure: {P_e_array[position_65[0]]} [Pa] \n Thrust Coefficient: {C_F_array[position_65[0]]} [Pa] \n Throat Area: {A_t_array[position_65[0]]} m2 \n Exit Area: {A_e_array[position_65[0]]} m2 \n Mass Flow Rate: {m_dot_array[position_65[0]]} [Kg/s]')

    # Calculate expansion ratio
    expansion_ratio = A_e_array[position_65[0]]/A_t_array[position_65[0]]

    # Determine chamber volume 
    V_c = calc_chamber_volume(A_t_array[position_65[0]], 36)

    # Determine chamber area
    A_c = calc_chamber_area(A_t_array[position_65[0]])

    # Determine chamber length
    L_c = calc_chamber_length(A_c, V_c)*0.0254 # L_c in m

    # Determine throat area for RS25
    A_t = A_t_array[position_65[0]]
    
    # Determine exit area for RS25
    A_e = A_e_array[position_65[0]]
    if showPlots:
        print(f'Exit Diamter: {2*A_e/np.pi} m \n Throat Diameter {2*np.sqrt(A_t/np.pi)} m \n Chamber Diameter: {2*np.sqrt(A_c/np.pi)} m \n Chamber Length: {L_c} m \n Chamber Volume: {V_c} m3')
    

    if showPlots:
        create_plot(area_ratio_array, M_e_array, 'Area Ratio', 'Exit Mach Number', 'Exit Mach Number vs Area Ratio')
        create_plot(area_ratio_array, P_e_array, 'Area Ratio', 'Exit Pressure [Pa]', 'Exit Pressure vs Area Ratio')
        create_plot(area_ratio_array, C_F_array, 'Area Ratio', 'Thrust Coefficient [Pa]', 'Thrust Coefficient vs Area Ratio')
        create_plot(area_ratio_array, A_t_array, 'Area Ratio', 'Throat Area [m^2]', 'Throat Area vs Area Ratio')
        create_plot(area_ratio_array, A_e_array, 'Area Ratio', 'Exit Area [m^2]', 'Exit Area vs Area Ratio')
        create_plot(area_ratio_array, m_dot_array, 'Area Ratio', 'Mass Flow Rate [Kg/s]', 'Mass Flow Rate vs Area Ratio')
    return (A_c, A_t, A_e, L_c) 

# Function to display engine geometry 
def display_engine_geometry(x_array, A_c, A_t, A_e, L_c):
    # Inputs:
    # x - type: float - distance from injector
    # A_c - type: float - chamber area [m^2]
    # A_t - type: float - throat area [m^2]
    # A_e - type: float - exit area [m^2]
    # L_c - type: float - chamber length [inch]

    # Convert to inch
    D_c = 2*np.sqrt(A_c/np.pi)*39.3701 # Calculate chamber diameter [m]
    D_t = 2*np.sqrt(A_t/np.pi)*39.3701# Calculate throat diameter [m]
    D_e = 2*np.sqrt(A_e/np.pi)*39.3701# Calculate exit diameter [m]

    # Calculate expansion ratio
    expansion_ratio = A_e/A_t

    # Determine radius of throat in [m]
    R_T = D_t/2
    
    # Create geometry plots

    # For each distance from injector, calculate the chamber radius and add to radius array
    r_array = [] # Create array to store chamber radius

    # Define key constants for RS25 engine
    L_e = 5.339 # Length before chamber starts contracting [m]
    theta_1 = 25.4157 # Angle of contraction [degrees]
    theta_D = 37 # Angle of expansion [degrees]
    alpha =  5.3738 # half-angle of the nozzle [degrees]
    #theta_E = 5.3738 # Angle of exit [degrees]
    R_1 = 1.73921 * R_T # Radius of contraction [m]
    R_U = 0.494 * R_T # Radius of contraction [m]
    R_D = 0.2 * R_T # Radius of throat curve [m]
    R_E = np.sqrt(69.5)*R_T # Radius of expansion [m]
    

    # Calculate length from throat to exit plane
    L = 0.8*(R_T * (np.sqrt(expansion_ratio)-1) + R_U*(1/np.cos(np.deg2rad(alpha-1))))/(np.tan(np.deg2rad(15)))
    #L_N = 0.8 * ((np.sqrt(expansion_ratio)-1)*D_t_inch/2)/(np.tan(np.deg2rad(15)))

    for x in x_array:
        if x <= L_e:
            r_array.append(D_c/2)
        elif (L_e < x) and (x <= L_e + R_1*np.sin(np.deg2rad(theta_1))):
            r_array.append(np.sqrt(R_1**2 - (x-L_e)**2) + D_c/2 - R_1)
        elif (L_e + R_1*np.sin(np.deg2rad(theta_1)) < x) and (x <= L_c - R_U*np.sin(np.deg2rad(theta_1))):
            r_array.append( (R_T + R_U - D_c/2 + R_1 - (R_U + R_1)*np.cos(np.deg2rad(theta_1)))/(L_c - L_e - (R_U + R_1)*np.sin(np.deg2rad(theta_1)))*x + D_c/2 - R_1 + R_1*np.cos(np.deg2rad(theta_1)) - (R_T + R_U - D_c/2 + R_1 - (R_U + R_1)*np.cos(np.deg2rad(theta_1)))/(L_c - L_e - (R_U + R_1)*np.sin(np.deg2rad(theta_1)))*(L_e + R_1*np.sin(np.deg2rad(theta_1))) )  
        elif (L_c - R_U*np.sin(np.deg2rad(theta_1)) < x) and (x <= L_c):
            r_array.append(-np.sqrt(R_U**2 - (x-L_c)**2) + R_T + R_U)
        elif (L_c < x) and (x <= L_c + R_D*np.sin(np.deg2rad(theta_D))):
            r_array.append(-np.sqrt(R_D**2 - (x - L_c)**2) + R_T + R_D)
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
            r_array.append(r_t)
    
    # Convert r_array and x_array to meters
    r_array = np.array(r_array) * 0.0254
    x_array = np.array(x_array) * 0.0254
    create_plot(x_array[:len(r_array)], r_array, 'Distance from Injector [m]', 'Radius [m]', 'Radius vs Distance from Injector')
    return None

# Function to calculate radius of chamber based off x
def calc_radius(x, A_c, A_t, A_e, L_c):
    # Inputs:
    # x - type: float - distance from injector [inch]
    # A_c - type: float - chamber area [m^2]
    # A_t - type: float - throat area [m^2]
    # A_e - type: float - exit area [m^2]
    # L_c - type: float - chamber length [inch]

    # Convert to inch
    D_c = 2*np.sqrt(A_c/np.pi)*39.3701 # Calculate chamber diameter [inch]
    D_t = 2*np.sqrt(A_t/np.pi)*39.3701# Calculate throat diameter [inch]
    D_e = 2*np.sqrt(A_e/np.pi)*39.3701# Calculate exit diameter [inch]

    # Convert L_c to inch
    L_c = L_c * 39.3701

    # Calculate expansion ratio
    expansion_ratio = A_e/A_t

    # Determine radius of throat in [m]
    R_T = D_t/2

    # Define key constants for RS25 engine @TODO need to replace with actual values for own engine
    L_e = 5.339 # Length before chamber starts contracting [m]
    theta_1 = 25.4157 # Angle of contraction [degrees]
    theta_D = 37 # Angle of expansion [degrees]
    alpha =  5.3738 # half-angle of the nozzle [degrees]
    #theta_E = 5.3738 # Angle of exit [degrees]
    R_1 = 1.73921 * R_T # Radius of contraction [m]
    R_U = 0.494 * R_T # Radius of contraction [m]
    R_D = 0.2 * R_T # Radius of throat curve [m]
    R_E = np.sqrt(expansion_ratio)*R_T # Radius of expansion [m]


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
    return r
    
  
    #return r*0.0254 # Return in m 

# Function which outlines ODE for local mach number squarred
def dN_dx (x, y, h, keydata):
    # Inputs:
    # x - type: float - current distance from injector
    # y - type: list - dependent variables [N, P, T]
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
    dF_dx = keydata[6]
    dQ_dx = keydata[7]

    # Calculate area at x 
    A = np.pi* (calc_radius(x, A_c, A_t, A_e, L_c))**2
    dA_dx = (np.pi*calc_radius(x+h, A_c, A_t, A_e, L_c)**2 - np.pi*calc_radius(x-h, A_c, A_t, A_e, L_c)**2)/(2*h)

    dN_dx = ((N/(1-N)) * ((1+gam*N)/(Cp*T)) * dQ_dx  +  (N/(1-N))*((2 + (gam-1)*N)/(specificGasConstant*T))*dF_dx - (N/(1-N))*((2 + (gam-1)*N)/A)*dA_dx)

    return dN_dx

# Function which outlines ODE for local pressure
def dP_dx (x, y, h, keydata):
        # Inputs:
    # x - type: float - current distance from injector
    # y - type: list - dependent variables [N, P, T]
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
    dF_dx = keydata[6]
    dQ_dx = keydata[7]
    
    # Calculate area at x 
    A = np.pi* (calc_radius(x, A_c, A_t, A_e, L_c))**2
    dA_dx = (np.pi*calc_radius(x+h, A_c, A_t, A_e, L_c)**2 - np.pi*calc_radius(x-h, A_c, A_t, A_e, L_c)**2)/(2*h)

    dP_dx = (-(P/(1-N))*((gam*N)/(Cp*T))*dQ_dx - (P/(1-N))*((1+ (gam-1)*N)/(specificGasConstant*T))*dF_dx + (P/(1-N))*((gam*N)/A)*dA_dx)
    return dP_dx

# Function which outlines ODE for local temperature
def dT_dx (x, y, h, keydata):
        # Inputs:
    # x - type: float - current distance from injector [inch]
    # y - type: list - dependent variables [N, P, T]
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
    dF_dx = keydata[6]
    dQ_dx = keydata[7]

    # Calculate area at x 
    A = np.pi* (calc_radius(x, A_c, A_t, A_e, L_c))**2
    dA_dx = (np.pi*calc_radius(x+h, A_c, A_t, A_e, L_c)**2 - np.pi*calc_radius(x-h, A_c, A_t, A_e, L_c)**2)/(2*h)


    
  
    
    dT_dx = ( (T/(1-N))*((1-gam*N)/(Cp*T))*dQ_dx - (T/(1-N))*(((gam-1)*N)/(specificGasConstant*T))*dF_dx + (T/(1-N))*(((gam-1)*N)/A)*dA_dx)
    return dT_dx

# Function which defines derrivatives of ODEs 
def derivs(x, y, h, keydata):
    # Inputs
    # x: Independent variable
    # y: Dependent variables (list)
    # h: Step size
    # keydata: list - key data to pass to ODE solver contains array of radius and specific heat ratio
    # Outputs
    # Output k values for RK4 method of length n 
    k = [] # K values for RK4 method of length n 
    
    # ODEs 
    # dy1/dx 
    k.append(dN_dx(x, y, h, keydata))
    
    #dy2/dx 
    k.append(dP_dx(x, y, h, keydata))

    #dy3/dx
    k.append(dT_dx(x, y, h, keydata))

    return k

# RK4 (runga kutta 4th order) function
def rk4(x, y, n, h, keydata):
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
    k.append(derivs(x, y, h, keydata))
    
    for i in range(n):
        ym[i] = y[i] + k[0][i]*h/2
    k.append(derivs(x+h/2, ym, h, keydata))
    for i in range(n):
        ym[i] = y[i] + k[1][i]*h/2
    k.append(derivs(x+h/2, ym, h, keydata))
    for i in range(n):
        ye[i] = y[i] + k[2][i]*h
    
    k.append(derivs(x+h, ye, h, keydata))
    for i in range(n):
        y[i] = y[i] + h*(k[0][i] + 2*(k[1][i] + k[2][i]) + k[3][i])/6
    x = x+h # increment x by step size
    return x, y

# Function to solve ODEs using RK4 method
def integrator(x, y, n, h, xend, keydata):
    """
    Numerical integrator for solving ordinary differential equations (ODEs).

    Parameters:
    - x: Independent variable
    - y: Dependent variables (list)
    - n: Number of ODEs
    - h: Step size
    - xend: Final value of independent variable
    - keydata: List of key data to pass to ODE solver (contains array of radius and specific heat ratio)

    Returns:
    - x: Final value of independent variable
    - y: Final values of dependent variables (list)
    """
    while True:
        if (xend - x < h):
            h = xend - x
        x, y = rk4(x, y, n, h, keydata)
        if (x >= xend):
            break
    return x, y

def calc_flow_data(M_c, P_c, T_c, keydata):
    """
    Calculate key flow data including Pressure, Temperature, and Mach number.
    Will use fourth order runge kutta (RK4) to solve the following ODEs (worked example in testfile.py)

    Args:
        M_c (float): Chamber Mach number.
        P_c (float): Chamber pressure [Pa].
        T_c (float): Chamber temperature [K].
        keydata: Additional key data of form [A_c, A_t, A_e, L_c, gam, Cp, dF_dx, dQ_dx].

    Returns:
        tuple: A tuple containing the step size (dx) [Inch], a list of x values (xp_m) [Inch], and a list of y values (yp_m) contains Mach Number, Pressure [Pa], Temperature [K].
    """    
    
    # Define initial conditions of RK4 algorithm 
    n = 3 # Number of ODEs 
    yi = [ M_c**2, P_c, T_c] # Initial conditions of n dependent variables [N, P, T] remebering N is Mach number squarred
    xi = 0 # [Inch] Initial value of independent variable
    xf = 135.5 # [inch] Distance from injector to end of engine in inch (has to be inch for radius function to work)
    dx = 25.19/200 # Step size [inch]
    xout = dx # [Inch] Output interval
    x = xi # [Inch] Working x value (set to initial condition)
    m = 0 # Iteration counter
    xp_m = [] # Track all x values through out iteration process

    yp_m = [] # Copy of y values for all iterations
    
    y = yi # working y values
    
    while True:
        xend = x + xout
        if (xend > xf):
            xend = xf
        h = dx
        x, y = integrator(x, y, n, h, xend, keydata)
        m += 1 # Increment m as we have integrated 
        xp_m.append(x)
        yp_m.append(y.copy())        
        if (x>=xf):
            break
    
    # Output results
    # for i in range(m+1):
    #    print("x = ", xp_m[i]*0.0254, "y = ", yp_m[i]) 
    # Create plots of pressure, temperature, mach number
    #print(f'Pressure at end of engine: {yp_m[-1][1]} [Pa] \n Temperature at end of engine: {yp_m[-1][2]} [K] \n Mach number at end of engine: {np.sqrt(yp_m[-1][0])}')
    # create_plot([xp*0.0254 for xp in xp_m], [np.sqrt(y[0]) for y in yp_m], "Distance from Injector [m]", "Mach Number", "Mach Number vs Distance from Injector")
    return dx, xp_m, yp_m


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
    
    print(np.exp(-(C3*delta_rho_c)**2))
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

    # Calculate eta_tot using equation (2.6.9)
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
    print("Visc", viscosity)


    return thermal_conductivity, gamma_total_normal, gamma_total_para, viscosity

# Function to calculate gas transport properties 
def calc_gas_tranport_properties(T, molecules, molecular_mass, molecular_fraction):
    # Inputs 
    # T - float - Temperature in K
    # molecules - array - Array of molecules in combustion gases 
    # molecular_mass - array - Array of molecular masses of combustion gases in same order as molecules
    # molecular_fraction - array - Array of molecular fractions of combustion gases in same order as molecules

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
    "rO": [0.79832550E00, 0.18039626E03, -0.53243244E05, 0.51131026E00, 0.79819261E00, 0.17970493E03, -0.52900889E05, 0.11797640E01]}

    # Array of transport properties of length molecules 
    viscosity_array = [] # Viscosity of each molecule
    thermal_conductivity_array = [] # Thermal conductivity of each molecule

    # Calculate the viscosity and thermal conductivity of each molecule
    for element in molecules:
        # Extract elements coefficients
        if T < 1000:
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

        # Using the above coefficients calculate the viscosity and thermal conductivity
        viscosity_array.append(np.exp(A_visc*np.log(T) + (B_visc/T) + (C_visc/(T**2)) + D_visc))
        thermal_conductivity_array.append(np.exp(A_thermal*np.log(T) + (B_thermal/T) + (C_thermal/(T**2)) + D_thermal))
    
    viscocisty_mix = 0 # Viscosity of the mixture
    thermal_mix = 0 # Thermal conductivity of the mixture

    # Calculate the viscosity and thermal conductivity of the mixture
    for i in range(len(molecules)):

        visc_den_sum = 0 # sum of viscosity denominator resets every iteration of i
        therm_den_sum = 0 # sum of thermal conductivity denominator resets every iteration of i

        # Calculate denominator values
        for j in range(len(molecules)):
            if j == i:
                continue
            else:
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
    "bH": [0.000000000E+00, 0.000000000E+00, 2.500000000E+00, 0.000000000E+00, 0.000000000E+00, 0 ,0.000000000E+00],
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
            a6 = thermo_curve_dict[f'b{element}'][5] # extract a6
            a7 = thermo_curve_dict[f'b{element}'][6] # extract a7

        element_Cp.append((a1*(1/(T**2)) + a2*(1/(T)) + a3 + a4*T + a5*(T**2) + a6*(T**3) + a7*(T**4))*Ru) # Calculate specific heat of element in mixture
    # Now calculate the specific heat of the mixture
    Cp_mix = 0 # Specific heat of the mixture [kJ/KgK] 
    for i in range(len(molecules)):
        Cp_mix += (molecular_fraction[i]*element_Cp[i]) # Calculate specific heat of mixture in kJ/KgK

    return Cp_mix # Return specific heat of mixture in J/molK

#-------------------------------------------------------------------
# FUNCTION TO CALCULATE HEAT TRANSFER THROUGH COOLANT CHANNELS
# ------------------------------------------------------------------

# Function which outputs chamber inner surface area
def calc_A_gas(x, y, dx, Ncc, keydata):
    # Inputs
    # x - type: float - distance from injector
    # y - type: array - dependent variables [N, P, T]
    # h - type: float - step size
    # Ncc - type: float - number of coolant channels
    # keydata - type: list - key data to pass to ODE solver contains array of radius and specific heat ratio A_c, A_t, A_e, L_c, gam, Cp, dF_dx, dQ_dx

    # Unpack keydata
    A_c = keydata[0] # Area injector [m^2]
    A_t = keydata[1] # Area throat [m^2]
    A_e = keydata[2] # Area exit [m^2]
    L_c = keydata[3] # Chamber length [inch]

    return (2*np.pi*calc_radius(x,A_c, A_t, A_e, L_c)*dx)/Ncc

def calc_h_gas(x, y, h, keydata):
    # Inputs
    # x - type: float - distance from injector
    # y - type: array - dependent variables [N, P, T]
    # h - type: float - step size
    # Ncc - type: float - number of coolant channels
    # keydata - type: list - key data to pass to ODE solver contains array of radius and specific heat ratio A_c, A_t, A_e, L_c, gam, Cp, dF_dx, dQ_dx

    return None

# function which calculates the heat transfer through the walls and coolant channels 
def calc_heat_transfer(x, y, dx, Ncc, gam, engine_geometry):
    """
    Calculate the heat transfer in a mixture optimization process.
    Parameters:
    x (array): Distance from injector.
    y (array): Dependent variables [N, P, T] (Section 5.3 results).
    dx (float): Step size.
    Ncc (float): Number of coolant channels.
    gam (float): Specific heat ratio calculated from CEA.
    engine_geometry: Engine geometry data.
    Returns:
    tuple which contains T_cw, T_hw, q_H2, q_wall, q_gas.
    """
    
    # Solve for Thw and Tcw using a multivariable newtonian method


    # q_gas = h_gas*A_gas*(T_aw - T_hw)
    # q_H2 = h_H2*A_H2*(Tcw - T_H2)
    # q_wall = k*A_gas*dT/dy

    A_gas = calc_A_gas(x, y, dx, Ncc, keydata)


    # Define the functions 
    # f1 = q_H2 - q_gas 
    # f2 = q_H2 - q_wall 

    return None


class EngineChannels:
    """
    Class representing engine channels for mixture optimization.
    Attributes:
        x_j (list): Array of x values from injector which will be used to calculate the channel geometry [m]
        chan_w (list): Array of channel widths [m]
        chan_h (list): Array of channel heights [m]
        chan_t (list): Array of channel thicknesses from hot side to cold side [m]
    Methods:
        get_geo(x):
            Function which will return the current channel width, height, and thickness at a given x value
            Parameters:
                x (float): Distance from injector [m]
            Returns:
                tuple: A tuple containing the channel width, height, thickness, and index of the closest x value in the array
    """

    def __init__(self, x_j, A_c, A_t, A_e, L_c, chan_w, chan_h, chan_t):
        self.x_j = x_j # Array of x values from injector which will be used to calculate the channel geometry [m]
        self.A_c = A_c # Area injector [m^2]
        self.A_t = A_t # Area throat [m^2]
        self.A_e = A_e # Area exit [m^2]
        self.L_c = L_c # Chamber length [m]
        self.chan_w = chan_w # Array of channel widths [m]
        self.chan_h = chan_h # Array of channel heights [m]
        self.chan_t = chan_t # Array of channel thicknesses from hot side to cold side [m]

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
        return self.chan_w[idx], self.chan_h[idx], self.chan_t[idx], idx


def main():
    # Main function
    # Constants
    gam = 1.19346 # Specific heat ratio
    M_c = 0.2279 # Injector mach number
    P_c = 18.23E6 # Chamber pressure in Pascals
    T_c = 3542 # Chamber temperature in Kelvin
    Cp = 1	 # Specific heat at constant pressure in J/KgK NOTE THIS IS A PLACEHOLDER CALCULATED BY HydrogenModelV2.py
    F_Vac = 2184076.8131 # Vacuum thrust in lbs 
    M_c = 0.26419 # Injector mach number
    Ncc = 430.0 # Number of coolant channels 
    showPlots = False # Boolean to show plots
    # Loading geometry data into the engine channels class
    # x_j = [-35.56, -34.29, -32.41, -30.48, -27.94, -25.4, -22.86, -20.32, -17.78, -15.24, -12.7, -11.43, -10.16, -8.89, -7.62, -6.35, -5.08, -3.81, -3.048, -2.54, -1.27, 0, 2.54, 5.08, 7.62, 10.16, 12.7, 15.24, 17.78, 20.32, 24.71] # Array of x values from throat which will be used to calculate the channel geometry [m]
    # x_j_subtracted = [(x+35.56)/100 for x in x_j]
    x_j =     [0.000000, 0.012700, 0.031500, 0.050800, 0.076200, 0.101600, 0.127000, 0.152400, 0.177800, 0.203200, 0.228600, 0.241300, 0.254000, 0.266700, 0.279400, 0.292100, 0.304800, 0.317500, 0.325120, 0.330200, 0.342900, 0.355600, 0.381000, 0.406400, 0.431800, 0.457200, 0.482600, 0.508000, 0.533400, 0.558800, 0.602700] # Array of x values from throat which will be used to calculate the channel geometry [m]
    # Channel width [m]
    chan_w =  [0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001509, 0.001217, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001227, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575]  # Array of channel widths [m] 
    # Channel height [m]
    chan_h =  [0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002512, 0.002619, 0.002845, 0.002535, 0.002337, 0.002139, 0.002217, 0.002352, 0.002477, 0.002448, 0.002609, 0.002741, 0.002870, 0.004318, 0.004953, 0.005004, 0.005100, 0.005476, 0.005817, 0.006096, 0.006096, 0.006069] # Array of channel heights [m]
    # Channel thickness [m]
    chan_t =  [0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889]  # Array of channel thicknesses from hot side to cold side [m]
    
    # Calculate the engine geometry based off CEA data
    A_c, A_t, A_e, L_c = engine_geometry(gam, P_c, T_c, F_Vac, showPlots) # returns area of injector, throat, exit, and chamber length all in [m^2, m^2, m^2 m] respectively
    
    # Create engine channels object
    engine_channels = EngineChannels(x_j,A_c, A_t, A_e, L_c, chan_w, chan_h, chan_t)

    # Initial conditons for thermal analysis set F and Q to zero for isentropic condition.
    dF_dx = 0 # Thrust gradient
    dQ_dx = 0 # Heat transfer gradient 

    # Key data
    keydata = [A_c, A_t, A_e, L_c, gam, Cp, dF_dx, dQ_dx] # Key data to pass to ODE solver
    
    # Calculate flow data
    dx, xp_m, yp_m = calc_flow_data(M_c, P_c, T_c, keydata) # returns pressure, temperature, mach number throughout  entire engine

    # Calculate heat transfer based off flow data
    #calc_heat_transfer(xp_m, yp_m, dx, Ncc, keydata)


    # Test transport propertie stuff
    create_plot([xp*0.0254 for xp in xp_m], [np.sqrt(y[0]) for y in yp_m], "Distance from Injector [m]", "Mach Number", "Mach Number vs Distance from Injector")
    create_plot([xp*0.0254 for xp in xp_m], [y[1] for y in yp_m], "Distance from Injector [m]", "Pressure [Pa]", "Pressure vs Distance from Injector")
    create_plot([xp*0.0254 for xp in xp_m], [y[2] for y in yp_m], "Distance from Injector [m]", "Temperature [K]", "Temperature vs Distance from Injector")
    
    if showPlots:

        molecules = ['H2', 'O2', 'H2O', 'OH', 'H', 'O'] # Molecules in combustion gases
        molecular_mass = [2.01588E-3, 31.9988E-3, 18.01528E-3, 17.00734E-3, 1.00794E-3, 15.9994E-3] # Molecular masses of combustion gases
        molecular_fraction = [0.24738, 0.00219, 0.68580, 0.03688, 0.02568, 0.00206] # Molecular fractions of combustion gases

        # Calculate transport properties
        visc = [] # Viscosity of the mixture
        therm = [] # Thermal conductivity of the mixture
        for y in yp_m:
            T = y[2]
            v, t = calc_gas_tranport_properties(T, molecules, molecular_mass, molecular_fraction)
            visc.append(v)
            therm.append(t)
        create_plot([xp*0.0254 for xp in xp_m], [v*1E-7 for v in visc], "Distance from Injector [m]", "Viscosity [Pa s]", "Viscosity vs Distance from Injector")
        create_plot([xp*0.0254 for xp in xp_m], [t*1E-4 for t in therm], "Distance from Injector [m]", "Thermal Conductivity [W/mK]", "Thermal Conductivity vs Distance from Injector")

        print(calc_gas_tranport_properties(3595.63, molecules, molecular_mass, molecular_fraction))
        print(calc_gas_specific_heat(3595.63, molecules, molecular_mass, molecular_fraction))
        # Calculate specific heat of gas
        molecules = ['H2', 'O2', 'H2O', 'OH', 'H', 'O'] # Molecules in combustion gases
        molecular_mass = [2.01588E-3, 31.9988E-3, 18.01528E-3, 17.00734E-3, 1.00794E-3, 15.9994E-3] # Molecular masses of combustion gases
        molecular_fraction = [0.24738, 0.00219, 0.68580, 0.03688, 0.02568, 0.00206] # Molecular fractions of combustion gases

        Cp_array = [] # Specific heat of the gas
        T_array = np.arange(200, 5000, 0.1)
        for T in T_array:
       
            Cp_array.append(calc_gas_specific_heat(T, molecules, molecular_mass, molecular_fraction))
       
        create_plot(T_array,  Cp_array, "Temperature [K]", "Specific Heat [J/KGK]", "Specific Heat vs Temperature")

    plt.show()

main()