import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as scipy

# Import formatted data in excel format. Use this to complete data analaysis


# Import excel file
df = pd.read_excel('CEAParsed.xlsx')

# Globals
univeralGasConstant = 8.31446261815324/0.013551 # Specifc gas constant [J/Kmol] (named wrong but cant be arsed to change it)

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
    # Calculate the geometry of a rocket engine based off key parameters from CEA
    # Inputs:
    # df - type: data frame - dataframe containing the CEA data
    


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
        rho_t = P_t / (univeralGasConstant * T_t)

        # Calculate sonic velocity at throat
        V_t = np.sqrt(gam * univeralGasConstant * T_t)

        # Calculate mass flow rate
        m_dot_array.append(rho_t*V_t*A_t_array[-1])
    
    # Find the area ratio for the RS25 engine
    position_65 = np.where(np.isclose(area_ratio_array, 69.5))[0] # 69.5 is area ratio for RS25 engine select based off new engine design
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


# Function to calculate raduys of chamber based off x
def calc_radius(x, A_c, A_t, A_e, L_c):
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
    
  
    return r*0.0254 # Return in m 

# Function to calculate the central finite difference
def central_finite_difference(func, x, h, *args):
    # Inputs:
    # func - type: function - function to calculate
    # x - type: float - x value
    # h - type: float - step size
    # args - type: list - arguments to pass to function

    return (func(x+h, *args) - func(x-h, *args))/(2*h)

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
    
    # Calculate area at x 
    A = np.pi* (calc_radius(x, A_c, A_t, A_e, L_c))**2
    dA_dx = (np.pi*calc_radius(x+h, A_c, A_t, A_e, L_c)**2 - np.pi*calc_radius(x-h, A_c, A_t, A_e, L_c)**2)/(2*h)



    dQ_dx = 0
    dF_dx = 0


    dN_dx = ((N/(1-N)) * ((1+gam*N)/(Cp*T)) * dQ_dx  +  (N/(1-N))*((2 + (gam-1)*N)/(univeralGasConstant*T))*dF_dx - (N/(1-N))*((2 + (gam-1)*N)/A)*dA_dx)

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
    
    # Calculate area at x 
    A = np.pi* (calc_radius(x, A_c, A_t, A_e, L_c))**2
    dA_dx = (np.pi*calc_radius(x+h, A_c, A_t, A_e, L_c)**2 - np.pi*calc_radius(x-h, A_c, A_t, A_e, L_c)**2)/(2*h)


    dQ_dx = 0
    dF_dx = 0


    dP_dx = (-(P/(1-N))*((gam*N)/(Cp*T))*dQ_dx - (P/(1-N))*((1+ (gam-1)*N)/(univeralGasConstant*T))*dF_dx + (P/(1-N))*((gam*N)/A)*dA_dx)
    return dP_dx


# Function which outlines ODE for local temperature
def dT_dx (x, y, h, keydata):
        # Inputs:
    # x - type: float - current distance from injector [inch]
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
    L_c = keydata[3] # Chamber length [m]
    gam = keydata[4] # Specific heat ratio
    Cp = keydata[5]  # Specific heat at constant pressure
    
    # Calculate area at x 
    A = np.pi* (calc_radius(x, A_c, A_t, A_e, L_c))**2
    dA_dx = (np.pi*calc_radius(x+h, A_c, A_t, A_e, L_c)**2 - np.pi*calc_radius(x-h, A_c, A_t, A_e, L_c)**2)/(2*h)


    dQ_dx = 0
    dF_dx = 0
    
    dT_dx = ( (T/(1-N))*((1-gam*N)/(Cp*T))*dQ_dx - (T/(1-N))*(((gam-1)*N)/(univeralGasConstant*T))*dF_dx + (T/(1-N))*(((gam-1)*N)/A)*dA_dx)
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

# RK4 function
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
        x, y = rk4(x, y, n, h, keydata)
        if (x >= xend):
            break
    return x, y

def calc_flow_data(A_c, A_t, A_e, L_c, gam, Cp, M_c, P_c, T_c, F_Vac):
    #Inputs:
    # x_array - type: array - distance from injector in inches
    # r_array - type: array - radius of chamber at x
    # gam - type: float - specific heat ratio
    # M_c - type: float - chamber mach number
    # P_c - type: float - chamber pressure [Pa]
    # T_c - type: float - chamber temperature [K]
    # F_Vac - type: float - vacuum thrust [N] 

    # Calculate key flow data Pressure, Temperature, Mach number
    # Key data
    keydata = [A_c, A_t, A_e, L_c, gam, Cp] # Key data to pass to ODE solver

    # Will use fourth order runge kutta to solve the following ODEs (worked example in testfile.py)
    # we will let y be a list of [N, P, T] where N is the local mach number squared, P is the local pressure, T is the local temperature 
    # Define initial conditions of RK4 algorithm 
    n = 3 # Number of ODEs 
    M = 0.26415 # Injector mach number
    yi = [ M**2, P_c, T_c] # Initial conditions of n dependent variables [N, P, T] remebering N is Mach number squarred
    xi = 0 # Initial value of independent variable
    xf = 135.5 # Distance from injector to end of engine in inch (has to be inch for radius function to work)
    dx = 0.01 # Step size 
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
        x, y = integrator(x, y, n, h, xend, keydata)
        m += 1 # Increment m as we have integrated 
        xp_m.append(x)
        yp_m.append(y.copy())        
        if (x>=xf):
            break
    
    # # Output results
    # for i in range(m+1):
    #    print("x = ", xp_m[i]*0.0254, "y = ", yp_m[i]) 
    # Create plots of pressure, temperature, mach number
    print(f'Pressure at end of engine: {yp_m[-1][1]} [Pa] \n Temperature at end of engine: {yp_m[-1][2]} [K] \n Mach number at end of engine: {np.sqrt(yp_m[-1][0])}')
    create_plot([xp*0.0254 for xp in xp_m], [np.sqrt(y[0]) for y in yp_m], "Distance from Injector [m]", "Mach Number", "Mach Number vs Distance from Injector")
    return None

def main():
    # Main function

    # Variable to show plots
    showPlots = False
    # Constants
    gam = 1.19346 # Specific heat ratio
    M_c = 0.2279 # Injector mach number
    P_c = 18.23E6 # Chamber pressure in Pascals
    T_c = 3542 # Chamber temperature in Kelvin
    Cp = 3.7848E3	 # Specific heat at constant pressure in J/KgK
    F_Vac = 2184076.8131 # Vacuum thrust in lbs 
    

    # Calculate the engine geometry based off CEA data
    A_c, A_t, A_e, L_c = engine_geometry(gam, P_c, T_c, F_Vac, showPlots) # returns x_array (inches from injector) and r_array (radius of chamber at x) 



    # Calculate flow data
    calc_flow_data(A_c, A_t, A_e, L_c, gam, Cp, M_c, P_c, T_c, F_Vac) # returns pressure, temperature, mach number throughout  entire engine


    # result = (df['CF'] * df['Cstar']) / 9.81
    # create_plot(df['O/F'], df['T_Chamber'], 'O/F', 'T Chamber', 'T Chamber vs O/F')
    # create_plot(df['O/F'], df['Gamma_Chamber'], 'O/F', 'Gamma Chamber', 'Gamma Chamber vs O/F')
    # create_plot(df['O/F'], df['Cstar'], 'O/F', 'CStar', 'CStar vs O/F')
    # create_plot(df['O/F'], df['Isp'], 'O/F', 'Isp', 'Isp vs O/F')
    # create_plot(df['O/F'], result, 'O/F', 'Isp [Sec]', 'Isp vs O/F')
    plt.show()

main()