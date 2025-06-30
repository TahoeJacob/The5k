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

    def display_MR_curves(self, of_ratios, fuel_stroage_density, ox_storage_density, engine_name=None):
        # Display plots of chamber temperature, mole fractions, specific heat ratios, C*, and Isp for given range of O/F ratios
        
        # Initialize lists to store performance parameters
        chamberTemperatures = []
        mole_fractions = []
        specific_heat_ratios = []
        cstar_values = []
        Isp_values_fixed = [] # Isp values expansion ratio fixed at 69
        ox_mass = []
        fuel_mass = []
        total_propellant_mass = []
        propellent_volume = []
        propellant_volume_ox = []
        propellant_volume_fuel = []
        Cf_values = [] # Thrust Coefficient values

        I = self.calculate_total_impulse()  # Calculate total impulse for the engine


        # Loop through each O/F ratio and calculate the performance parameters
        for of in of_ratios:
            chamberTemp, mole_frac, gamma, cstar, isp, Cf = self.calculate_performance(of)
            chamberTemperatures.append(chamberTemp)
            mole_fractions.append(mole_frac)
            specific_heat_ratios.append(gamma)
            Cf_values.append(Cf) # Thrust Coefficient
            cstar_values.append(cstar)
            Isp_values_fixed.append(isp)
            m_tot = I/isp # Total required propellant mass for the engine
            m_H2 = m_tot / (1 + of) # Mass of fuel (LH2)
            m_O2 = m_tot/ ((1/of) + 1)
            ox_mass.append(m_O2)  # Mass of oxidizer
            fuel_mass.append(m_H2) # Mass of fuel
            total_propellant_mass.append(m_tot)  # Total propellant mass
            ox_volume = m_O2 / ox_storage_density  # Volume of oxidizer
            fuel_volume = m_H2 / fuel_stroage_density
            propellant_volume_ox.append(ox_volume)  # Volume of oxidizer
            propellant_volume_fuel.append(fuel_volume)
            propellent_volume.append(ox_volume+fuel_volume)  # Total volume of propellant



        # Extract all unique species from the mole fractions
        all_species = set()
        for mf in mole_fractions:    
            all_species.update(mf[1].keys()) # for each reactant, add it to the set of all species

        all_species = sorted(all_species)

        # Prepare data for plotting
        species_mole_fractions = {species_molar_fraction: [] for species_molar_fraction in all_species}
        mixture_molecular_weight = []
        for mf in mole_fractions:
            species_mole_weight = []
            for species in all_species:
                molar_fraction = mf[1].get(species)
                molar_weight = mf[0].get(species)
                if molar_fraction is not None: # If species exists in the mixture
                    species_mole_weight.append(molar_weight*molar_fraction[0])
                    species_mole_fractions[species].append(molar_fraction[0])
                else:
                    species_mole_fractions[species].append(0.0)
            mixture_molecular_weight.append(sum(species_mole_weight))

        # Plot Chamber Temperature vs O/F ratio (cryo-rocket.com figure 3.4.1)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, chamberTemperatures, label='Flame Temperature (K)')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Flame Temperature (K)')
        plt.title(f'{engine_name} Flame Temperature vs O/F Ratio')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 4000)
        plt.xlim(0, of_ratios[-1])
        plt.tight_layout()

        # Plot mole fraction for each species (cryo-rocket.com figure 3.4.2)
        plt.figure(figsize=(10, 6))
        for species in all_species:
            plt.plot(of_ratios, species_mole_fractions[species], label=species)
        plt.xlabel('O/F Ratio')
        plt.ylabel('Mole Fraction')
        plt.title(f'{engine_name} Species Mole Fractions vs O/F Ratio')
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize='small')
        plt.grid(True)
        plt.tight_layout()

        # Plot mixture molecular weight (cryo-rocket.com figure 3.4.3)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, mixture_molecular_weight)
        plt.xlabel('O/F Ratio')
        plt.ylabel('Mixture Molecular Weight (g/mol)')
        plt.title(f'{engine_name} Mixture Molecular Weight vs O/F Ratio')
        plt.grid(True)
        plt.tight_layout()

        # Plot specific heat ratio (cryo-rocket.com figure 3.4.4)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, specific_heat_ratios)
        plt.xlabel('O/F Ratio')
        plt.ylabel('Specific Heat Ratio (γ)')
        plt.title(f'{engine_name} Specific Heat Ratio vs O/F Ratio')
        plt.grid(True)
        plt.tight_layout()

        # Plot c* (cryo-rocket.com figure 3.4.5)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, cstar_values)
        plt.xlabel('O/F Ratio')
        plt.ylabel('C* (m/s)')
        plt.title(f'{engine_name} Characteristic Velocity (C*) vs O/F Ratio')
        plt.grid(True)
        plt.tight_layout()

        # Plot Thrust Coefficient Cf for fixed expansion and infinite expansion ratio (cryo-rocket.com figure 3.4.6)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, Cf_values, label=f'Cf of EPS {self.expansion_ratio}')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Isp (sec)')
        plt.title(f'{engine_name} Isp vs O/F Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Plot Isp for fixed expansion and infinite expansion ratio (cryo-rocket.com figure 3.4.6)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, Isp_values_fixed, label='Isp')
        #plt.plot(of_ratios, Cf_values_infinite, label='Infinite Expansion Ratio')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Isp (sec)')
        plt.title(f'{engine_name} Isp vs O/F Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Plot propellant masses 
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, ox_mass, label='Oxidizer Mass (kg)')
        plt.plot(of_ratios, fuel_mass, label='Fuel Mass (kg)')
        plt.plot(of_ratios, total_propellant_mass, label='Total Propellant Mass (kg)')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Mass (kg)')
        plt.title(f'{engine_name} Propellant Masses vs O/F Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Plot propellant volume
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, propellent_volume, label='Propellant Volume (m^3)')
        plt.plot(of_ratios, propellant_volume_ox, label='Oxidizer Volume (m^3)')
        plt.plot(of_ratios, propellant_volume_fuel, label='Fuel Volume (m^3)')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Volume (m^3)')
        plt.title(f'{engine_name} Propellant Volume vs O/F Ratio')
        plt.grid(True)
    
        plt.show()


        pass

def extract_cea_output(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    # Define the blocks to extract
    blocks_to_extract = [
        ("Pinf/P", ["TRANSPORT PROPERTIES (GASES ONLY)", "WITH EQUILIBRIUM REACTIONS", "WITH FROZEN REACTIONS", "PERFORMANCE PARAMETERS", "MOLE FRACTIONS"]),
        ("VISC,MILLIPOISE", ["WITH EQUILIBRIUM REACTIONS", "WITH FROZEN REACTIONS", "PERFORMANCE PARAMETERS", "MOLE FRACTIONS"]),
        ("WITH EQUILIBRIUM REACTIONS", ["WITH FROZEN REACTIONS", "PERFORMANCE PARAMETERS", "MOLE FRACTIONS"]),
        ("WITH FROZEN REACTIONS", ["PERFORMANCE PARAMETERS", "MOLE FRACTIONS"]),
        ("PERFORMANCE PARAMETERS", ["MOLE FRACTIONS"]),
        ("MOLE FRACTIONS", ["NOTE.", "PRODUCTS WHICH WERE CONSIDERED"])
    ]

    results = {}
    for start_pat, end_pats in blocks_to_extract:
        block = extract_block(lines, start_pat, end_pats)
        if block:
            results[start_pat] = block

    return results

def RS25_Design():
    # FUNCTION WORKSPACE
    # This function will be used to run through the steps to design the RS25 rocket engine

    expansion_ratio = 69  # Fixed expansion ratio for RS
    Pc = 200 # Chamber pressure in bar (200 bar for RS25)
    fuel_stroage_density = 70.85  # Density of LH2 in kg/m^3
    ox_storage_density = 1140.0  # Density of LOX in kg/m^3
    burn_time = 500  # Estimated burn time in seconds
    engine_thrust_lb = 450000  # Estimated thrust sea level  [lbf]
    engine_thrust_n = engine_thrust_lb * 4.44822  # Convert lbf to N
    number_of_engines = 3  # Number of RS25 engines

    # Function to run rocket engine calculations/design
    RS25 = RocketEngine('LOX', 'LH2', Pc, expansion_ratio, burn_time, engine_thrust_n, num_engines=number_of_engines) # Verification Engine: RS25 - Cryo-rocket.com
    engine_name = 'RS25'  # Name of the engine for display purposes

    of_ratios = np.arange(0.5, 21, 0.1)  # Define O/F ratios for the RS25 engine

    of_ratios =[6]
    # Display the performance curves for the RS25 engine
    #RS25.display_MR_curves(of_ratios, fuel_stroage_density, ox_storage_density, engine_name=engine_name)
    for of in of_ratios:

        fulloutput = RS25.fullCeaOutput(of)

        fulloutput=fulloutput.splitlines()
        for i,line in enumerate(fulloutput):
            print(i, line)
            if line.startswith(' P, ATM'):
                Pc = float(line.split()[2])*101325 # Chamber pressure in Pa (Originally in atm)
                Pt = float(line.split()[3])*101325 # Throat pressure in atm
                Pe = float(line.split()[4])*101325 # Exit pressure in atm
            if line.startswith(' T, K'):
                Tc = float(line.split()[2]) # Chamber temperature in K
                Tt = float(line.split()[3]) # Throat temperature in K
                Te = float(line.split()[4]) # Exit temperature in K
            if line.startswith(' RHO, G/CC'):
                def fix_sci_notation(s):
                    return re.sub(r'([0-9])\-([0-9])', r'\1e-\2', s)
                Rho_c = float(fix_sci_notation(line.split()[2]))*0.001  # Chamber density in kg/m^3 (Was in g/cc (grams/cubic centimeter) )
                Rho_t = float(fix_sci_notation(line.split()[3]))*0.001  # Throat density in kg/m^3 (Was in g/cc (grams/cubic centimeter) )
                Rho_e = float(fix_sci_notation(line.split()[4]))*0.001  # Exit density in kg/m^3 (Was in g/cc (grams/cubic centimeter) )
            if line.startswith(' H, CAL/G'):
                Hc = float(line.split()[2]) * 4184.6  # Chamber enthalpy in J/Kg (Was in cal/g)
                Ht = float(line.split()[3]) * 4184.6  # Throat enthalpy in J/Kg (Was in cal/g)
                He = float(line.split()[4]) * 4184.6  # Exit enthalpy in J/Kg (Was in cal/g)
            if line.startswith(' GAMMAs'):
                gamma = float(line.split()[2])  # Specific heat ratio
            if line.startswith(' VISC, MILLIPOISE'):
                print(line.split())

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

    #The5k()


    



main()

