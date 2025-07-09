import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI
import cantera as ct
from rocketcea.cea_obj import CEA_Obj
import pandas as pd
import re



# Global Constants
g = 9.81 # Gravity in m/s/s^2 (sea level)
Ru = 8.31446261815324 # Univeral gas constant [J/mol*K]
specificGasConstant = Ru/0.013551 # Specifc gas constant [J/Kmol] (named wrong but cant be arsed to change it)


class RocketEngine:
    def __init__(self, oxidizer, fuel, chamber_pressure, expansion_ratio, burn_time, thrust, num_engines=1):
        self.oxidizer = oxidizer
        self.fuel = fuel
        self.chamber_pressure_bar = chamber_pressure
        self.expansion_ratio = expansion_ratio
        self.burn_time = burn_time
        self.thrust = thrust
        self.num_engines = num_engines

    def fullCeaOutput(self, of_ratio):
        # Return the full CEA output
        cea = CEA_Obj(oxName=self.oxidizer, fuelName =self.fuel)

        # Convert chamber pressure in bar to psi
        chamber_pressure_psi = self.chamber_pressure_bar * 14.5038  # Convert bar to psi

        fullOutput = cea.get_full_cea_output(Pc = chamber_pressure_psi, MR = of_ratio, eps=self.expansion_ratio)
        return fullOutput

    def calculate_performance(self, of_ratio):
        # Using RocketCEA calcualte key engine peroformance parameters
        cea = CEA_Obj(oxName=self.oxidizer, fuelName=self.fuel)

        chamber_pressure_psi = self.chamber_pressure_bar * 14.5038  # Convert bar to psi

        t_comb_r = cea.get_Tcomb(Pc=chamber_pressure_psi, MR=of_ratio)
        chamberTemp = t_comb_r * (5.0 / 9.0)  # Convert Rankine to Kelvin

        # Get molar fractions as a dictionary for each O/F ratio
        mole_frac = cea.get_SpeciesMoleFractions(Pc=chamber_pressure_psi, MR=of_ratio)

        # Get specific heat ratio for the mixture
        gamma = cea.get_Chamber_MolWt_gamma(Pc=chamber_pressure_psi, MR=of_ratio)[1]

        # Get C_star for the mixture
        cstar = cea.get_Cstar(Pc=chamber_pressure_psi, MR=of_ratio) * 0.3048  # Convert ft/s to m/s

        # Get coefficient of thrust Cf
        Cf  = cea.getFrozen_PambCf(Pamb=14.7, Pc=chamber_pressure_psi, MR=of_ratio, eps=self.expansion_ratio)
        
        cstar = np.sqrt(gamma*specificGasConstant*chamberTemp)/(gamma*np.sqrt( (2/gamma+1)**((gamma+1)/(gamma-1)) ))
       
        #Cf = np.sqrt( (2*gamma**2)/(gamma-1) * (2/(gamma + 1))**((gamma+1)/(gamma-1)) * (1-(1)))
        # Get Isp for given expansion ratio
        if self.expansion_ratio is None:
            isp = cea.get_Isp(Pc=chamber_pressure_psi, MR=of_ratio)
        else:
            isp = cea.get_Isp(Pc=chamber_pressure_psi, MR=of_ratio, eps=self.expansion_ratio)
       
        return chamberTemp, mole_frac, gamma, cstar, isp, Cf[0]

    def calculate_thrust(self):


        pass

    def calculate_total_impulse(self):
        # Calculate total impulse using thrust and burn time
        total_impulse = self.thrust * self.burn_time  # in N*s
        return total_impulse

    def display_MR_curves(self, of_ratios, engine_name=None):
        # Display plots of chamber temperature, mole fractions, specific heat ratios, C*, and Isp for given range of O/F ratios

        of_data = self.extract_cea_output(of_ratios)  # Extract CEA output for the given O/F ratios
        # Plot Chamber Temperature vs O/F ratio (cryo-rocket.com figure 3.4.1)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, [data['T_c'] for data in of_data.values()], label='Flame Temperature (K)')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Flame Temperature (K)')
        plt.title(f'{engine_name} Flame Temperature vs O/F Ratio')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 4000)
        plt.xlim(0, of_ratios[-1])
        plt.tight_layout()

        # Plot specific heat ratio (cryo-rocket.com figure 3.4.4)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, [data['gamma'] for data in of_data.values()])
        plt.xlabel('O/F Ratio')
        plt.ylabel('Specific Heat Ratio (γ)')
        plt.title(f'{engine_name} Specific Heat Ratio vs O/F Ratio')
        plt.grid(True)
        plt.tight_layout()

        # Plot c* (cryo-rocket.com figure 3.4.5)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, [data['Cstar_c'] for data in of_data.values()])
        plt.xlabel('O/F Ratio')
        plt.ylabel('C* (m/s)')
        plt.title(f'{engine_name} Characteristic Velocity (C*) vs O/F Ratio')
        plt.grid(True)
        plt.tight_layout()

        # Plot Isp for fixed expansion and infinite expansion ratio (cryo-rocket.com figure 3.4.6)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, [data['Isp_t'] for data in of_data.values()], label='Isp')
        #plt.plot(of_ratios, Cf_values_infinite, label='Infinite Expansion Ratio')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Isp (sec)')
        plt.title(f'{engine_name} Isp vs O/F Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        I = self.burn_time * self.thrust * self.num_engines  # Total impulse in N*s

        for data in of_data.values():
            tot_mass = I / (data['Isp_t'] * 9.81)
            fuel_mass = tot_mass / (1 + data['OF_ratio'])
            ox_mass = (data['OF_ratio']*tot_mass) / (data['OF_ratio'] + 1)
            data['tot_mass'] = tot_mass
            data['ox_mass'] = ox_mass
            data['fuel_mass'] = fuel_mass
            data['ox_volume'] = ox_mass / 1141
            data['fuel_volume'] = fuel_mass / 70
            data['propellant_volume'] = data['ox_volume'] + data['fuel_volume']

                  
        # Plot propellant masses 
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, [data['ox_mass'] for data in of_data.values()], label='Oxidizer Mass (kg)')
        plt.plot(of_ratios, [data['fuel_mass'] for data in of_data.values()], label='Fuel Mass (kg)')
        plt.plot(of_ratios, [data['tot_mass'] for data in of_data.values()], label='Total Propellant Mass (kg)')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Mass (kg)')
        plt.xlim(0, 8)
        plt.ylim(0, 1.2E6)
        plt.title(f'{engine_name} Propellant Masses vs O/F Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Plot propellant volume
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, [data['propellant_volume'] for data in of_data.values()], label='Propellant Volume (m^3)')
        plt.plot(of_ratios, [data['ox_volume'] for data in of_data.values()], label='Oxidizer Volume (m^3)')
        plt.plot(of_ratios, [data['fuel_volume'] for data in of_data.values()], label='Fuel Volume (m^3)')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Volume (m^3)')
        plt.xlim(0, 8)
        plt.title(f'{engine_name} Propellant Volume vs O/F Ratio')
        plt.legend()
        plt.grid(True)
    
        plt.show()


        pass

    def extract_cea_output(self, of_ratios):
        # FUNCTION WORKSPACE
        
        # Create a dictionary to store the results for each O/F ratio
        of_data = {}

        # This function will extract the CEA output from the full output string and return it as a dictionary
        for of in of_ratios:

            fulloutput = self.fullCeaOutput(of)

            fulloutput=fulloutput.splitlines()
            for i,line in enumerate(fulloutput):
                if line.startswith(' P, ATM'):
                    P_c = float(line.split()[2])*101325 # Chamber pressure in Pa (Originally in atm)
                    P_t = float(line.split()[3])*101325 # Throat pressure in atm
                    P_e = float(line.split()[4])*101325 # Exit pressure in atm
                if line.startswith(' T, K'):
                    T_c = float(line.split()[2]) # Chamber temperature in K
                    T_t = float(line.split()[3]) # Throat temperature in K
                    T_e = float(line.split()[4]) # Exit temperature in K
                if line.startswith(' RHO, G/CC'):
                    def fix_sci_notation(s):
                        return re.sub(r'([0-9])\-([0-9])', r'\1e-\2', s)
                    Rho_c = float(fix_sci_notation(line.split()[2]))*0.001  # Chamber density in kg/m^3 (Was in g/cc (grams/cubic centimeter) )
                    Rho_t = float(fix_sci_notation(line.split()[3]))*0.001  # Throat density in kg/m^3 (Was in g/cc (grams/cubic centimeter) )
                    Rho_e = float(fix_sci_notation(line.split()[4]))*0.001  # Exit density in kg/m^3 (Was in g/cc (grams/cubic centimeter) )
                if line.startswith(' H, CAL/G'):
                    H_c = float(line.split()[2]) * 4184.6  # Chamber enthalpy in J/Kg (Was in cal/g)
                    H_t = float(line.split()[3]) * 4184.6  # Throat enthalpy in J/Kg (Was in cal/g)
                    H_e = float(line.split()[4]) * 4184.6  # Exit enthalpy in J/Kg (Was in cal/g)
                if line.startswith(' GAMMAs'):
                    gamma = float(line.split()[2])  # Specific heat ratio
                if line.startswith(' VISC,MILLIPOISE'):
                    visc_c = float(line.split()[1]) * 0.001  # Viscosity in Pa.s (Was in millipoise)
                    visc_t = float(line.split()[2]) * 0.001  # Viscosity in Pa.s (Was in millipoise)
                    visc_e = float(line.split()[3]) * 0.001  # Viscosity in Pa.s (Was in millipoise)
                if line.startswith('  WITH EQUILIBRIUM'):
                    # Extract the equilibrium reactions block
                    eq_reactions = fulloutput[i+1:i+6]   
                    # Extract Cp Equilibrium values
                    Cp_Equil_c = float(eq_reactions[1].split()[2]) * 4184.6  # Cp in J/Kg.K (Was in cal/g.K)
                    Cp_Equil_t = float(eq_reactions[1].split()[3]) * 4184.6  # Cp in J/Kg.K (Was in cal/g.K)
                    Cp_Equil_e = float(eq_reactions[1].split()[4]) * 4184.6  # Cp in J/Kg.K (Was in cal/g.K)

                    # Extract Conductivity Equilibrium values
                    cond_Equil_c = float(eq_reactions[2].split()[1])
                    cond_Equil_t = float(eq_reactions[2].split()[2])
                    cond_Equil_e = float(eq_reactions[2].split()[3])

                    # Extract Prantdl Number Equilibrium values
                    Pr_Equil_c = float(eq_reactions[3].split()[2])
                    Pr_Equil_t = float(eq_reactions[3].split()[3])
                    Pr_Equil_e = float(eq_reactions[3].split()[4])
                if line.startswith('  WITH FROZEN'):
                    froze_reactions = fulloutput[i+1:i+6]  # Extract the frozen reactions block
                    # Extract Cp Equilibrium values
                    Cp_Froze_c = float(froze_reactions[1].split()[2]) #* 4184.6  # Cp in J/Kg.K (Was in cal/g.K)
                    Cp_Froze_t = float(froze_reactions[1].split()[3]) #* 4184.6  # Cp in J/Kg.K (Was in cal/g.K)
                    Cp_Froze_e = float(froze_reactions[1].split()[4]) #* 4184.6  # Cp in J/Kg.K (Was in cal/g.K)

                    # Extract Conductivity Equilibrium values
                    cond_Froze_c = float(froze_reactions[2].split()[1])
                    cond_Froze_t = float(froze_reactions[2].split()[2])
                    cond_Froze_e = float(froze_reactions[2].split()[3])

                    # Extract Prantdl Number Equilibrium values
                    Pr_Froze_c = float(froze_reactions[3].split()[2])
                    Pr_Froze_t = float(froze_reactions[3].split()[3])
                    Pr_Froze_e = float(froze_reactions[3].split()[4])
                if line.startswith(' Ae/At'):
                    Ae_At_c = float(line.split()[1])
                    Ae_At_e = float(line.split()[2])
                if line.startswith(' CSTAR'):
                    Cstar_c = float(line.split()[2]) * 0.3048 # Convert C* from ft/s to m/s
                    Cstar_t = float(line.split()[3]) * 0.3048 # Convert C* from ft/s to m/s
                if line.startswith(' CF'):
                    CF_c = float(line.split()[1])  # Thrust coefficient at chamber conditions
                    CF_t = float(line.split()[2])  # Thrust coefficient at throat conditions
                if line.startswith(' Ivac'):
                    Isp_vac_c = float(line.split()[1]) # Isp at vacuum conditions Lb-sec/lb
                    Isp_vac_t = float(line.split()[2]) # Isp at throat conditions Lb-sec/lb
                if line.startswith(' Isp'):
                    Isp_c = float(line.split()[2]) # Isp at chamber conditions Lb-sec/lb
                    Isp_t = float(line.split()[3]) # Isp at throat conditions Lb-sec/lb
                if line.startswith(' MOLE FRACTIONS'):
                    # Extract the mole fractions block
                    molar_fractions = [] # Store the mole fractions
                    while not line.startswith('  * THERMODYNAMIC'):
                        line = fulloutput[i+1] 
                        if line.split() == []: # Ignore empty lines
                            pass
                        elif line.startswith('  * THERMODYNAMIC'): # Ignore last line
                            break
                        else: # Add all other lines as these are molar fractions
                            molar_fractions.append(line.split())
                        i += 1

            # Store the extracted data in the of_data dictionary
            of_data[of] = {
                'OF_ratio': of,
                'P_c': P_c,
                'P_t': P_t,
                'P_e': P_e,
                'T_c': T_c,
                'T_t': T_t,
                'T_e': T_e,
                'Rho_c': Rho_c,
                'Rho_t': Rho_t,
                'Rho_e': Rho_e,
                'H_c': H_c,
                'H_t': H_t,
                'H_e': H_e,
                'gamma': gamma,
                'visc_c': visc_c,
                'visc_t': visc_t,
                'visc_e': visc_e,
                'Cp_Equil_c': Cp_Equil_c,
                'Cp_Equil_t': Cp_Equil_t,
                'Cp_Equil_e': Cp_Equil_e,
                'cond_Equil_c': cond_Equil_c,
                'cond_Equil_t': cond_Equil_t,
                'cond_Equil_e': cond_Equil_e,
                'Pr_Equil_c': Pr_Equil_c,
                'Pr_Equil_t': Pr_Equil_t,
                'Pr_Equil_e': Pr_Equil_e,
                'Cp_Froze_c': Cp_Froze_c,
                'Cp_Froze_t': Cp_Froze_t,
                'Cp_Froze_e': Cp_Froze_e,
                'cond_Froze_c': cond_Froze_c,
                'cond_Froze_t': cond_Froze_t,
                'cond_Froze_e': cond_Froze_e,
                'Pr_Froze_c': Pr_Froze_c,
                'Pr_Froze_t': Pr_Froze_t,
                'Pr_Froze_e': Pr_Froze_e, 
                'Ae_At_c': Ae_At_c, 
                'Ae_At_e': Ae_At_e, 
                'Cstar_c' : Cstar_c, 
                'Cstar_t' : Cstar_t, 
                'CF_c' : CF_c, 
                'CF_t' : CF_t, 
                'Isp_vac_c' : Isp_vac_c, 
                'Isp_vac_t' : Isp_vac_t, 
                'Isp_c' : Isp_c,
                'Isp_t' : Isp_t,
                'molar_fractions': molar_fractions}
        

        return of_data

def RS25_Design():
    # FUNCTION WORKSPACE
    # This function will be used to run through the steps to design the RS25 rocket engine

    expansion_ratio = 69  # Fixed expansion ratio for RS
    Pc = 200 # Chamber pressure in bar (200 bar for RS25)
    fuel_stroage_density = 70.85  # Density of LH2 in kg/m^3
    ox_storage_density = 1140.0  # Density of LOX in kg/m^3
    burn_time = 520  # Estimated burn time in seconds
    engine_thrust_lb = 491E3  # Estimated thrust sea level  [lbf]
    engine_thrust_n = engine_thrust_lb * 4.44822  # Convert lbf to N
    number_of_engines = 3  # Number of RS25 engines
    # Function to run rocket engine calculations/design
    RS25 = RocketEngine('LOX', 'LH2', Pc, expansion_ratio, burn_time, engine_thrust_n, num_engines=number_of_engines) # Verification Engine: RS25 - Cryo-rocket.com
    engine_name = 'RS25'  # Name of the engine for display purposes

    of_ratios = np.arange(0.8, 21, 0.1)  # Define O/F ratios for the RS25 engine

    # of_ratios =[6]
    # Display the performance curves for the RS25 engine
    #RS25.display_MR_curves(of_ratios, fuel_stroage_density, ox_storage_density, engine_name=engine_name)

    # Loop through each O/F ratio and extract the CEA output
    of_data = RS25.extract_cea_output(of_ratios)

    RS25.display_MR_curves(of_ratios, engine_name=engine_name)  # Display the performance curves for the RS25 engine

    # Plot temperature
    plt.figure()
    plt.plot(of_ratios, [data['T_c'] for data in of_data.values()], label='Chamber Temperature (K)')
    plt.xlabel('O/F Ratio')
    plt.ylabel('Temperature (K)')
    plt.title(f'{engine_name} Temperature vs O/F Ratio')
    plt.ylim(0, 4000)  # Limit y-axis to 4000 K
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    pass

def The5k():
    # FUNCTION WORKSPACE
    # This function will be used to run through the steps to design the 5k rocket engine
    
    engine_name = 'The5k'  # Name of the engine for display purposes

    # Define Rocket Key Information
    rocketMass = 90.0 # Estimated mass of the rocket in kg
    rocketThrust = 5000 # Estimated thrust sea level  [N]
    Pc = 20 # Chamber presssure in bar (desired)
    fuel_stroage_density = 737.9  # Density of RP-1 in kg/m^3 (approximate)
    ox_storage_density = 1140.0  # Density of LOX in kg/m^3 (approximate)
    burn_time = 10  # Estimated burn time in seconds
    
    
    
    # Calculate thrust to weight ratio
    thrust_to_weight_ratio = rocketThrust / (rocketMass * g)  # Thrust to weight ratio

    # o/f Ratio array to find the optimal O/F ratio
    of_ratios = np.arange(0.5, 7, 0.1)  # Define O/F ratios for the 5k engine
    expansion_ratio = None  # No fixed expansion ratio for the 5k engine yet... 

    # Function to run rocket engine calculations/design
    The5k = RocketEngine('LOX', 'RP-1', Pc, expansion_ratio, burn_time, rocketThrust) # Verification Engine: RS25 - Cryo-rocket.com
    The5k.display_MR_curves(of_ratios, fuel_stroage_density, ox_storage_density, engine_name=engine_name) # Display the performance curves for the 5k engine to determine optimal O/F


    pass

def  main():
    # Call functions and run code here


    RS25_Design()

    # The5k()


    



main()

