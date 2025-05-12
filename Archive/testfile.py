# Implement RK4 method for solving ODEs simultaneously 
import numpy as np
import matplotlib.pyplot as plt

# THIS WORKS FOR A SYSTEM OF ODEs



# Function which defines derrivatives of ODEs 
def derivs(x, y):
    # Inputs
    # x: Independent variable
    # y: Dependent variables (list)
    # Outputs
    # Output k values for RK4 method of length n 
    k = [] # K values for RK4 method of length n 
    
    # ODEs 
    # dy1/dx 
    k.append(y[1])
    
    #dy2/dx 
    k.append(-16.1*y[0])

    #dy3/dx
    k.append(y[3])

    #dy4/dx
    k.append(-16.1*np.sin(y[2]))

    return k


# RK4 function
def rk4(x, y, n, h):
    # Inputs
    # x: Independent variable
    # y: Dependent variables (list)
    # n: Number of ODEs
    # h: Step size
    
    # Outputs
    k = [] # K values for RK4 method of length n has tuples
    ym = [0] * len(y) # midpoint values of y
    ye = [0] * len(y)# endpoint values of y
    k.append(derivs(x, y))
    
    for i in range(n):
        ym[i] = y[i] + k[0][i]*h/2
    k.append(derivs(x+h/2, ym))
    for i in range(n):
        ym[i] = y[i] + k[1][i]*h/2
    k.append(derivs(x+h/2, ym))
    for i in range(n):
        ye[i] = y[i] + k[2][i]*h
    
    k.append(derivs(x+h, ye))
    for i in range(n):
        y[i] = y[i] + h*(k[0][i] + 2*(k[1][i] + k[2][i]) + k[3][i])/6
    x = x+h # increment x by step size
    return x, y

# Function to solve ODEs using RK4 method
def integrator(x, y, n, h, xend):
    # Inputs 
    # x: Independent variable
    # y: Dependent variables (list)
    # n: Number of ODEs
    # h: Step size
    # xend: Final value of independent variable
    while True:
        if (xend - x < h):
            h = xend - x
        x, y = rk4(x, y, n, h)
        if (x >= xend):
            break
    return x, y

def main():
    # Main function define constants and initial conditions
    n = 3 # Number of ODEs 
    yi = [0.785398, 0, 0.785398] # Initial conditions of n dependent variables
    xi = 0 # Initial value of independent variable
    xf = 4 # Final value of independent variable
    dx = 0.001 # Step size
    xout = dx # Output interval

    x = xi # Working x value (set to initial condition)
    m = 0 # Iteration counter
    xp_m = [x] # Track all x values through out iteration process

    yp_m = [] # Copy of y values for all iterations
    yp_m.append(yi) # Initial condition
    y = yi # working y values
    
    while True:
        xend = x + xout
        if (xend > xf):
            xend = xf
        h = dx
        x, y = integrator(x, y, n, h, xend)
        m += 1 # Increment m as we have integrated 
        xp_m.append(x)
        yp_m.append(y.copy())        
        if (x>=xf):
            break
    
    # Output results
    for i in range(m+1):
        print("x = ", xp_m[i], "y = ", yp_m[i])


    #++++++++++++++++++++++++++++++++++++++++++++
    # SECTION 1
    #++++++++++++++++++++++++++++++++++++++++++++
    # Code which calculates the injector mach number which works best for initial isnetropic solution
    # Calculate flow data (Section 5.3 results) 
    vale = np.arange(0.22790, 0.27001, 0.00001)
    potential_sol = []
    vale = [0.26085]
    # Initial loop which will be used to calculate first heat transfer data
    for M_c_RS25 in vale:
        # Assuming calc_flow_data function is defined elsewhere and returns the relevant values
        dx, xp_m, yp_m = calc_flow_data(M_c_RS25, P_c, T_c, keydata)  # Corrected the function call with M_c_RS25
        print(M_c_RS25)
        if np.sqrt(yp_m[np.argmax([t[0] for t in yp_m])][0]) > 1.0:  # If the maximum Mach number is greater than 1
            index = np.argmax([t[0] for t in yp_m])  # Extract the index of largest value from the yp_m tuple array
            print("above Mach 1", M_c_RS25, index)
            if index > len(xp_m) - 30:  # Check if the index is within the last 30 values
                potential_sol.append(M_c_RS25)

    print(potential_sol)


    #++++++++++++++++++++++++++++++++++++++++++++
    # SECTION 2
    #++++++++++++++++++++++++++++++++++++++++++++
    # Ensure calc_radius is defined and all variables (x_j, chan_w, etc.) are of the same length
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot the channel geometry data on the left y-axis
    ax1.plot(x_j, [w*1000 for w in chan_w], label="Channel Width", color="blue")
    ax1.plot(x_j, [h*1000 for h in chan_h], label="Channel Height", color="green")
    ax1.plot(x_j, [t*1000 for t in chan_t], label="Channel Thickness", color="orange")
    ax1.plot(x_j, [l*1000 for l in chan_land], label="Channel Land", color="red")
    ax1.set_ylim(0, 7)
    ax1.set_xlabel("Distance from Injector [m]")
    ax1.set_ylabel("Channel Geometry [m]")
    ax1.set_title("Channel Geometry vs Distance from Injector")

    # Create a second y-axis for the radius plot
    ax2 = ax1.twinx()
    ax2.plot(x_j, [2.54*calc_radius(x*39.37, A_c, A_t, A_e, L_c) for x in x_j], label="Radius", color="purple")
    ax2.set_ylabel("Radius [m]")
    ax2.set_ylim(0, 50)

    # Combine legends from both axes
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Show grid and display the plot
    ax1.grid(True)
    ax1.set_xlim(0, 0.6)
    return None

main()