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


    return None

main()