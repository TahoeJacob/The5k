import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI
import cantera as ct
from rocketcea.cea_obj import CEA_Obj
import pandas as pd
import re



# Global Constants
g = 9.81 # Gravity in m/s/s^2 (sea level)
c = 295 # Speed of sound in air at sea level in m/s (approximate, varies with temperature and pressure)
Ru = 8.31446261815324 # Univeral gas constant [J/mol*K]
specificGasConstant = Ru/0.0275787914406925 # Specifc gas constant [J/Kmol] (named wrong but cant be arsed to change it)


class RocketEngine:
    def __init__(self, oxidizer, fuel, chamber_pressure, expansion_ratio, burn_time, thrust, num_engines=1, Cd = 0.21):
        self.oxidizer = oxidizer
        self.fuel = fuel
        self.chamber_pressure_bar = chamber_pressure
        self.expansion_ratio = expansion_ratio
        self.burn_time = burn_time
        self.thrust = thrust
        self.num_engines = num_engines
        self.Cd = Cd  # Drag coefficient of the rocket body

    def fullCeaOutput(self, of_ratio):
        # Return the full CEA output
        cea = CEA_Obj(oxName=self.oxidizer, fuelName =self.fuel)

        # Convert chamber pressure in bar to psi
        chamber_pressure_psi = self.chamber_pressure_bar * 14.5038  # Convert bar to psi

        fullOutput = cea.get_full_cea_output(Pc = self.chamber_pressure_bar, MR = of_ratio, eps=self.expansion_ratio, short_output=1, pc_units='bar', show_mass_frac=1, output='siunits' )
        return fullOutput

    def calculate_total_impulse(self):
        # Calculate total impulse using thrust and burn time
        total_impulse = self.thrust * self.burn_time  # in N*s
        return total_impulse

    def display_MR_curves(self, of_ratios, fuel_storage_density, ox_storage_density, engine_name=None, target_of=2):
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
        plt.plot(of_ratios, [data['gamma_t'] for data in of_data.values()], label='Throat Conditions (γ)')
        plt.plot(of_ratios, [data['gamma_c'] for data in of_data.values()], label='Chamber Conditions (γ)')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Specific Heat Ratio (γ)')
        plt.title(f'{engine_name} Specific Heat Ratio vs O/F Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Plot c* (cryo-rocket.com figure 3.4.5)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, [data['Cstar_t'] for data in of_data.values()])
        plt.xlabel('O/F Ratio')
        plt.ylabel('C* (m/s)')
        plt.title(f'{engine_name} Characteristic Velocity (C*) vs O/F Ratio')
        plt.grid(True)
        plt.tight_layout()

        # Plot Cf (cryo-rocket.com figure 3.4.7)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, [data['CF_t'] for data in of_data.values()], label='Throat Conditions (CF)')
        plt.xlabel('O/F Ratio')
        plt.ylabel('Thrust Coefficient (CF)')
        plt.title(f'{engine_name} Thrust Coefficient vs O/F Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()


        # Plot Isp for fixed expansion and infinite expansion ratio (cryo-rocket.com figure 3.4.6)
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, [data['Isp_e'] for data in of_data.values()], label='Isp')
        plt.plot(of_ratios, [data['Isp_vac_e'] for data in of_data.values()], label='Isp vac (Exit)')
        # Example: Get the index of a specific O/F ratio in of_data
        # target_of = 2  # Replace with the O/F ratio you want to find
        of_keys = [round(float(of), 1) for of in of_data.keys()]
        if target_of in of_keys:
            idx = of_keys.index(target_of)
        # Annotate the plot at the target O/F ratio
        isp_value = [data['Isp_e'] for data in of_data.values()][idx]
        plt.annotate(f'O/F={target_of:.1f}\nIsp={isp_value:.1f}',
                     xy=(target_of, isp_value),
                     xytext=(target_of + 0.2, isp_value + 10),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
        plt.xlabel('O/F Ratio')
        plt.ylabel('Isp vac (sec)' )
        plt.title(f'{engine_name} Isp vs O/F Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        

        I = self.burn_time * self.thrust * self.num_engines  # Total impulse in N*s
        ox_volumes = []
        fuel_volumes = []
        tot_masses = []
        tot_volumes = []
        of_array = []
    
        for data in of_data.values():
            tot_mass = I / (data['Isp_e'] * g)  
            fuel_mass = tot_mass / (1 + data['OF_ratio'])
            ox_mass = (data['OF_ratio']*tot_mass) / (data['OF_ratio'] + 1)
            data['tot_mass'] = tot_mass
            data['ox_mass'] = ox_mass
            data['fuel_mass'] = fuel_mass
            data['ox_volume'] = ox_mass / ox_storage_density
            data['fuel_volume'] = fuel_mass / fuel_storage_density
            data['propellant_volume'] = data['ox_volume'] + data['fuel_volume']
            of_array.append(data['OF_ratio'])
            fuel_volumes.append(data['fuel_volume'])
            ox_volumes.append(data['ox_volume'])
            tot_volumes.append(data['ox_volume'] + data['fuel_volume'])
            tot_masses.append(tot_mass)


                  
        # Plot propellant masses 
        plt.figure(figsize=(8, 5))
        plt.plot(of_ratios, [data['ox_mass'] for data in of_data.values()], label='Oxidizer Mass (kg)')
        plt.plot(of_ratios, [data['fuel_mass'] for data in of_data.values()], label='Fuel Mass (kg)')
        plt.plot(of_ratios, [data['tot_mass'] for data in of_data.values()], label='Total Propellant Mass (kg)')
        # Annotate the plot at the target O/F ratio
        mass_value = [data['tot_mass'] for data in of_data.values()][idx]
        plt.annotate(f'O/F={target_of:.1f}\nTot Mass={mass_value:.1f}',
                     xy=(target_of, mass_value),
                     xytext=(target_of + 0.2, mass_value + 10),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
        of_min_target = of_array[tot_masses.index(min(tot_masses))]  # Get the O/F ratio corresponding to the minimum total mass
        plt.annotate(f'O/F={of_min_target:.1f}\nTot Mass={min(tot_masses):.1f}',
                     xy=(of_min_target, min(tot_masses)),
                     xytext=(of_min_target + 0.2, min(tot_masses) + 10),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
        plt.xlabel('O/F Ratio')
        plt.ylabel('Mass (kg)')
        plt.xlim(0, 8)
        # plt.ylim(0, 1.2E6)
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
    

        of_data = self.print_OF_info(of_data, target_of)  # Display CEA output data for O/F ratio of 2
        
        plt.show()

        return of_data

    def print_OF_info(self, of_data, target_of):
        # Inputs:
        # of_ratios: List of O/F ratios to display data for
        # target_of: O/F ratio to display data for (used for annotation)

         # Calculate length of pipe based off volumes and diameters
        d = 0.152 # Daiameter of pipe [m] (from steel and tube)
        A = np.pi * (d/2)**2  # Cross-sectional area of the pipe [m^2]
        
        # Function which displays CEA data for given O/F ratio
        temp_array = [] # Create an array to store the chamber temperatures for each of the O/F ratios
        ISP_array = [] # Create an array to store the Isp for each of the O/F ratios
        ISP_vac_array = [] # Create an array to store the Isp at vacuum for each of the O/F ratios
        fuel_mass_array = [] # Create an array to store the fuel mass for each of the O/F ratios
        ox_mass_array = [] # Create an array to store the oxidizer mass for each of the O/F ratios
        tot_mass_array = [] # Create an array to store the total mass for each of the O/F ratios
        of_array = [] # Create an array to store the O/F ratios for each of the O/F ratios
        fuel_volume_array = [] # Create an array to store the fuel volume for each of the O/F ratios
        ox_volume_array = [] # Create an array to store the oxidizer volume for each of the O/F ratios
        tot_volume_array = [] # Create an array to store the total volume for each of the O/F ratios
        gamma_array = [] # Create an array to store the specific heat ratio for each of the O/F ratios
        c_star_array = [] # Create an array to store the characteristic velocity for each of the O/F ratios
        chamber_pressure_array = [] # Create an array to store the chamber pressure for each of the O/F ratios
        Cp_array = [] # Create an array to store the specific heat at constant pressure for each of the O/F ratios

        for data in of_data.values():
            temp_array.append(data['T_c'])
            ISP_array.append(data['Isp_e'])
            ISP_vac_array.append(data['Isp_vac_e'])
            fuel_mass_array.append(data['fuel_mass'])
            ox_mass_array.append(data['ox_mass'])
            tot_mass_array.append(data['tot_mass'])
            of_array.append(round(float(data['OF_ratio']), 1))
            fuel_volume_array.append(data['fuel_volume'])
            ox_volume_array.append(data['ox_volume'])
            tot_volume_array.append(data['propellant_volume'])
            gamma_array.append(data['gamma_t'])  # Specific heat ratio at throat conditions
            data['tank_length'] = round(data['propellant_volume'] / A, 2)  # Length of tank [m]
            data['ox_tank_length'] = round(data['ox_volume'] / A, 2)  # Length of oxidizer tank [m]
            data['fuel_tank_length'] = round(data['fuel_volume'] / A, 2)  # Length of fuel tank [m]
            c_star_array.append(data['Cstar_t'])  # Characteristic velocity at throat conditions
            chamber_pressure_array.append(data['P_c']) # Chamber pressure in Pa
            Cp_array.append(data['Cp_Equil_t'])

        

        length = [round(volume / A, 2) for volume in tot_volume_array]
        ox_tank_length = round(ox_volume_array[of_array.index(target_of)] / A, 2)  # Length of oxidizer tank [m]
        fuel_tank_length = round(fuel_volume_array[of_array.index(target_of)] / A, 2)

        # Print the CEA data for the target O/F ratio    
        # Prepare table data
        def get_of_for_extreme(arr, of_arr, func):
            idx = arr.index(func(arr))
            return of_arr[idx]

        table_data = [
            ["Parameter", "Value", "Min/Max", "Delta"],
            [
            "Chamber Temperature (K)",
            f"{temp_array[of_array.index(target_of)]:.2f}",
            f"Max: {max(temp_array):.2f} @ O/F={get_of_for_extreme(temp_array, of_array, max):.2f}",
            f"{max(temp_array) - temp_array[of_array.index(target_of)]:.2f}"
            ],
            [
            "Isp (sec)",
            f"{ISP_array[of_array.index(target_of)]:.2f}",
            f"Max: {max(ISP_array):.2f} @ O/F={get_of_for_extreme(ISP_array, of_array, max):.2f}",
            f"{max(ISP_array) - ISP_array[of_array.index(target_of)]:.2f}"
            ],
            [
            "Isp at vacuum (sec)",
            f"{ISP_vac_array[of_array.index(target_of)]:.2f}",
            f"Max: {max(ISP_vac_array):.2f} @ O/F={get_of_for_extreme(ISP_vac_array, of_array, max):.2f}",
            f"{max(ISP_vac_array) - ISP_vac_array[of_array.index(target_of)]:.2f}"
            ],
            [
            "Fuel Mass (kg)",
            f"{fuel_mass_array[of_array.index(target_of)]:.2f}",
            f"Min: {min(fuel_mass_array):.2f} @ O/F={get_of_for_extreme(fuel_mass_array, of_array, min):.2f}",
            f"{abs(min(fuel_mass_array) - fuel_mass_array[of_array.index(target_of)]):.2f}"
            ],
            [
            "Oxidizer Mass (kg)",
            f"{ox_mass_array[of_array.index(target_of)]:.2f}",
            f"Min: {min(ox_mass_array):.2f} @ O/F={get_of_for_extreme(ox_mass_array, of_array, min):.2f}",
            f"{abs(min(ox_mass_array) - ox_mass_array[of_array.index(target_of)]):.2f}"
            ],
            [
            "Total Mass (kg)",
            f"{tot_mass_array[of_array.index(target_of)]:.2f}",
            f"Min: {min(tot_mass_array):.2f} @ O/F={get_of_for_extreme(tot_mass_array, of_array, min):.2f}",
            f"{abs(min(tot_mass_array) - tot_mass_array[of_array.index(target_of)]):.2f}"
            ],
            [
            "Fuel Volume (m^3)",
            f"{fuel_volume_array[of_array.index(target_of)]:.2f}",
            f"Min: {min(fuel_volume_array):.2f} @ O/F={get_of_for_extreme(fuel_volume_array, of_array, min):.2f}",
            f"{abs(min(fuel_volume_array) - fuel_volume_array[of_array.index(target_of)]):.2f}"
            ],
            [
            "Oxidizer Volume (m^3)",
            f"{ox_volume_array[of_array.index(target_of)]:.2f}",
            f"Min: {min(ox_volume_array):.2f} @ O/F={get_of_for_extreme(ox_volume_array, of_array, min):.2f}",
            f"{abs(min(ox_volume_array) - ox_volume_array[of_array.index(target_of)]):.2f}"
            ],
            [
            "Total Volume (m^3)",
            f"{tot_volume_array[of_array.index(target_of)]:.2f}",
            f"Min: {min(tot_volume_array):.2f} @ O/F={get_of_for_extreme(tot_volume_array, of_array, min):.2f}",
            f"{abs(min(tot_volume_array) - tot_volume_array[of_array.index(target_of)]):.2f}"
            ],
            ["Chamber Pressure (Bar)",
            f"{chamber_pressure_array[of_array.index(target_of)]:.2f}",
            f"Max: {max(chamber_pressure_array):.2f} @ O/F={get_of_for_extreme(chamber_pressure_array, of_array, max):.2f}",
            f"{max(chamber_pressure_array) - chamber_pressure_array[of_array.index(target_of)]:.2f}"
            ],
            ["Specific Heat Ratio (γ)", 
            f"{gamma_array[of_array.index(target_of)]:.4f}",
            f"Max: {max(gamma_array):.4f} @ O/F={get_of_for_extreme(gamma_array, of_array, max):.2f}",
            f"{max(gamma_array) - gamma_array[of_array.index(target_of)]:.4f}"
            ],
            ["Characteristic Velocity (C*)",    
            f"{c_star_array[of_array.index(target_of)]:.2f} m/s",
            f"Max: {max(c_star_array):.2f} m/s @ O/F={get_of_for_extreme(c_star_array, of_array, max):.2f}",
            f"{max(c_star_array) - c_star_array[of_array.index(target_of)]:.2f} m/s"
            ],
            ["Specific Heat at Constant Pressure (Cp)",
            f"{Cp_array[of_array.index(target_of)]:.2f} J/Kg.K",
            f"Max: {max(Cp_array):.2f} J/Kg.K @ O/F={get_of_for_extreme(Cp_array, of_array, max):.2f}",
            f"{max(Cp_array) - Cp_array[of_array.index(target_of)]:.2f} J/Kg.K"
            ],
        ]

       
        print(f"Pipe Length for O/F ratio of {target_of}: {length[of_array.index(target_of)]} m, Oxidizer Tank Length: {ox_tank_length} m, Fuel Tank Length: {fuel_tank_length} m") 
        # Based off length calculate mass of pipe
        mass_per_meter = 6  # Mass of steel per meter [kg/m] (from steel and tube)
        print(f"Pipe Mass for O/F ratio of {target_of}: {round(length[of_array.index(target_of)] * mass_per_meter, 2)} kg")

        # Calculate column widths for alignment
        col_widths = [max(len(str(row[i])) for row in table_data) for i in range(4)]
        sep = " | "
        total_width = sum(col_widths) + len(sep) * 3

        print("=" * total_width)
        print(f"Cea Data for O/F ratio of {target_of}:")
        print(f"Specific Heat Ratio at Throat Conditions: {gamma_array[of_array.index(target_of)]:.4f}")
        print("=" * total_width)
        # Print table header
        print(
            f"{table_data[0][0]:<{col_widths[0]}}{sep}"
            f"{table_data[0][1]:<{col_widths[1]}}{sep}"
            f"{table_data[0][2]:<{col_widths[2]}}{sep}"
            f"{table_data[0][3]:<{col_widths[3]}}"
        )
        print("-" * total_width)
        # Print table rows
        for row in table_data[1:]:
            print(
            f"{row[0]:<{col_widths[0]}}{sep}"
            f"{row[1]:<{col_widths[1]}}{sep}"
            f"{row[2]:<{col_widths[2]}}{sep}"
            f"{row[3]:<{col_widths[3]}}"
            )
        print("=" * total_width)
        print()

        return of_data
    
    def extract_cea_output(self, of_ratios):
        # FUNCTION WORKSPACE
        
        # Create a dictionary to store the results for each O/F ratio
        of_data = {}

        # This function will extract the CEA output from the full output string and return it as a dictionary
        for of in of_ratios:

            fulloutput = self.fullCeaOutput(of)

            fulloutput=fulloutput.splitlines()
            for i,line in enumerate(fulloutput):
                if line.startswith(' P, BAR'):
                    P_c = float(line.split()[2]) # Chamber pressure in Pa (Originally in bar)
                    P_t = float(line.split()[3]) # Throat pressure in bar
                    # P_e = float(line.split()[4])*101325 # Exit pressure in atm
                if line.startswith(' T, K'):
                    T_c = float(line.split()[2]) # Chamber temperature in K
                    T_t = float(line.split()[3]) # Throat temperature in K
                    # T_e = float(line.split()[4]) # Exit temperature in K
                if line.startswith(' RHO, KG/CU M'):
                    def fix_sci_notation(s):
                        return re.sub(r'([0-9])\-([0-9])', r'\1e-\2', s)
                    Rho_c = float(fix_sci_notation(line.split()[3]))  # Chamber density in kg/m^3 (Was in g/cc (grams/cubic centimeter) )
                    Rho_t = float(fix_sci_notation(line.split()[5]))  # Throat density in kg/m^3 (Was in g/cc (grams/cubic centimeter) )
                    # Rho_c = float(line.split()[3]) * (10**float(line.split()[4]))
                    # Rho_t = float(line.split()[5]) * (10**float(line.split()[6]))  # Throat density in kg/m^3 (Was in g/cc (grams/cubic centimeter) )
           
                if line.startswith(' H, KJ/KG'):
                    H_c = float(line.split()[2])*1000            # Chamber enthalpy in J/Kg 
                    H_t = float(line.split()[3])*1000           # Throat enthalpy in J/Kg 
                    # H_e = float(line.split()[4]) * 4184.6  # Exit enthalpy in J/Kg (Was in cal/g)
                if line.startswith(' GAMMAs'):
                    gamma_c = float(line.split()[1])  # Specific heat ratio at chamber conditions
                    gamma_t = float(line.split()[2])  # Specific heat ratio


                if line.startswith(' VISC,MILLIPOISE'):
                    visc_c = float(line.split()[1]) * 0.001  # Viscosity in Pa.s (Was in millipoise)
                    visc_t = float(line.split()[2]) * 0.001  # Viscosity in Pa.s (Was in millipoise)
                    # visc_e = float(line.split()[3]) * 0.001  # Viscosity in Pa.s (Was in millipoise)
                
                if line.startswith('  WITH EQUILIBRIUM'):
                    # Extract the equilibrium reactions block
                    eq_reactions = fulloutput[i+1:i+6]   
                    # Extract Cp Equilibrium values
                    Cp_Equil_c = float(eq_reactions[1].split()[2])*1000 # Cp in J/Kg.K (Was in cal/g.K)
                    Cp_Equil_t = float(eq_reactions[1].split()[3])*1000  # Cp in J/Kg.K (Was in cal/g.K)
                    
                    # Cp_Equil_e = float(eq_reactions[1].split()[4]) * 4184.6  # Cp in J/Kg.K (Was in cal/g.K)

                    # Extract Conductivity Equilibrium values
                    cond_Equil_c = float(eq_reactions[2].split()[1])*1000
                    cond_Equil_t = float(eq_reactions[2].split()[2])*1000
                    # cond_Equil_e = float(eq_reactions[2].split()[3])
                    # Extract Prantdl Number Equilibrium values
                    Pr_Equil_c = float(eq_reactions[3].split()[2])
                    Pr_Equil_t = float(eq_reactions[3].split()[3])
                    # Pr_Equil_e = float(eq_reactions[3].split()[4])
                if line.startswith('  WITH FROZEN'):
                    froze_reactions = fulloutput[i+1:i+6]  # Extract the frozen reactions block
                    # Extract Cp Equilibrium values
                    Cp_Froze_c = float(froze_reactions[1].split()[2])*1000 #* 4184.6  # Cp in J/Kg.K (Was in cal/g.K)
                    Cp_Froze_t = float(froze_reactions[1].split()[3])*1000 #* 4184.6  # Cp in J/Kg.K (Was in cal/g.K)
                    # Cp_Froze_e = float(froze_reactions[1].split()[4]) #* 4184.6  # Cp in J/Kg.K (Was in cal/g.K)

                    # Extract Conductivity Equilibrium values
                    cond_Froze_c = float(froze_reactions[2].split()[1])
                    cond_Froze_t = float(froze_reactions[2].split()[2])
                    # cond_Froze_e = float(froze_reactions[2].split()[3])

                    # Extract Prantdl Number Equilibrium values
                    Pr_Froze_c = float(froze_reactions[3].split()[2])
                    Pr_Froze_t = float(froze_reactions[3].split()[3])
                    # Pr_Froze_e = float(froze_reactions[3].split()[4])
                if line.startswith(' Ae/At'):
                    Ae_At_c = float(line.split()[1])
                    # Ae_At_e = float(line.split()[2])
                if line.startswith(' CSTAR'):
                    # Cstar_c = float(line.split()[2]) * 0.3048 # Convert C* from ft/s to m/s
                    Cstar_t = float(line.split()[2])  # Convert C* from ft/s to m/s
                if line.startswith(' CF'):
                    # CF_c = float(line.split()[1])  # Thrust coefficient at chamber conditions
                    CF_t = float(line.split()[1])  # Thrust coefficient at throat conditions
                if line.startswith(' Ivac'):
                    # Isp_vac_c = float(line.split()[1]) # Isp at vacuum conditions Lb-sec/lb
                    Isp_vac_t = float(line.split()[2])/g
                    Isp_vac_e = float(line.split()[3])/g # Isp at throat conditions Lb-sec/lb
                if line.startswith(' Isp'):
                    # Isp_c = float(line.split()[2]) # Isp at chamber conditions Lb-sec/lb
                    Isp_t = float(line.split()[2])/g # Isp at throat conditions M//SEC
                    Isp_e = float(line.split()[3])/g # Isp at exit conditions Lb-sec/lb
                if line.startswith(' MASS FRACTIONS'):
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
                # 'P_e': P_e,
                'T_c': T_c,
                'T_t': T_t,
                # 'T_e': T_e,
                'Rho_c': Rho_c,
                'Rho_t': Rho_t,
                # 'Rho_e': Rho_e,
                'H_c': H_c,
                'H_t': H_t,
                # 'H_e': H_e,
                'gamma_c': gamma_c,
                'gamma_t': gamma_t,
                'visc_c': visc_c,
                'visc_t': visc_t,
                # 'visc_e': visc_e,
                'Cp_Equil_c': Cp_Equil_c,
                'Cp_Equil_t': Cp_Equil_t,
                # 'Cp_Equil_e': Cp_Equil_e,
                'cond_Equil_c': cond_Equil_c,
                'cond_Equil_t': cond_Equil_t,
                # 'cond_Equil_e': cond_Equil_e,
                'Pr_Equil_c': Pr_Equil_c,
                'Pr_Equil_t': Pr_Equil_t,
                # 'Pr_Equil_e': Pr_Equil_e,
                'Cp_Froze_c': Cp_Froze_c,
                'Cp_Froze_t': Cp_Froze_t,
                # 'Cp_Froze_e': Cp_Froze_e,
                'cond_Froze_c': cond_Froze_c,
                'cond_Froze_t': cond_Froze_t,
                # 'cond_Froze_e': cond_Froze_e,
                'Pr_Froze_c': Pr_Froze_c,
                'Pr_Froze_t': Pr_Froze_t,
                # 'Pr_Froze_e': Pr_Froze_e, 
                'Ae_At_c': Ae_At_c, 
                # 'Ae_At_e': Ae_At_e, 
                # 'Cstar_c' : Cstar_c, 
                'Cstar_t' : Cstar_t, 
                # 'CF_c' : CF_c, 
                'CF_t' : CF_t, 
                'Isp_vac_e' : Isp_vac_e, 
                'Isp_vac_t' : Isp_vac_t, 
                'Isp_e' : Isp_e,
                'Isp_t' : Isp_t,
                'molar_fractions': molar_fractions}
        

        return of_data

    def calc_gravity(self, h):
        R_earth = 6.371e6  # meters
        g0 = 9.81  # m/s^2
        return g0 * (R_earth / (R_earth + h)) ** 2

    def calc_air_density(self, h): # from https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
        if h  < 11000:  # Below 11 km, use the barometric formula
            T = 15.05 - 0.00649*h # Temperature in Celsius
            p = 101.29 * ((T+273.15)/288.08)**(5.256)
        elif h < 25000 and h >= 11000: # Above 11km less than 25km
            T = -56.46  # Temperature in Celsius
            p = 22.65 * np.exp(1.73 - 0.000157*h)
        else:  # Above 25 km, use a constant pressure
            T = -131.21 + 0.00299*h  # Temperature in Celsius
            p = 2.488 * ((T + 273.15)/216.6)**(-11.388)
                               
        rho = p/(0.2869*(T+273.15))  # Density in kg/m^3
        return rho


    def estimate_altitude(self, of_ratios, target_of, dt = 0.1, recovery_mass = 5, engine_mass = 10, payload_mass = 2, tank_mass_per_meter = 6):
        # Function to calculate altitute rocket will get reach
        # Inputs:
        # of_ratios: List of O/F ratios to display data for
        # target_of: O/F ratio to display data for (used for annotation)
        # dt: Time step for the simulation in seconds
        # recovery_mass: Mass of the recovery system in kg
        # engine_mass: Mass of the rocket engine in kg
        # payload_mass: Mass of the payload in kg
        # tank_mass_per_meter: Mass of the tank per meter in kg/m

        # Returns:
        # altitude: Estimated altitude the rocket will reach in meters

        # First thing is to setup arrays for acceleration, velocity, and altitude
        acceleration = [] # Acceleration in m/s
        velocities = []  # Velocity in m/s
        machs = []  # Mach number
        altitudes = []  # Altitude in m

        v = 0 # Initial velocity in m/s
        h = 0 # Initial altitude in m


        # Caluclate initial mass of rocket
        of_array = []  # Array to store O/F ratios
        tot_masses = []  # Array to store total masses
        fuel_masses = []  # Array to store fuel masses
        ox_masses = []  # Array to store oxidizer masses
        tank_lengths = []  # Array to store tank lengths
        fuel_mdot_array = [] # Array to store fuel mass flow rates
        ox_mdot_array = []  # Array to store oxidizer mass flow rates
        for data in of_ratios.values(): # Extract CEA output for each O/F ratio
            of_array.append(round(float(data['OF_ratio']), 1))
            tot_masses.append(data['tot_mass'])
            fuel_masses.append(data['fuel_mass'])
            ox_masses.append(data['ox_mass'])
            tank_lengths.append(data['tank_length'])  # Length of tank [m]
            fuel_mdot_array.append(data['fuel_mass'] / self.burn_time)  # Fuel mass flow rate in kg/s
            ox_mdot_array.append(data['ox_mass'] / self.burn_time)  # Oxidizer mass flow rate in kg/s
        
        # Get the index of the target O/F ratio
        total_ox_mass = ox_masses[of_array.index(target_of)]  # Total oxidizer mass in kg
        total_fuel_mass = fuel_masses[of_array.index(target_of)]  # Total fuel mass in kg
        tank_length = tank_lengths[of_array.index(target_of)]  # Length of tank [m]
        tank_mass = (tank_length * tank_mass_per_meter)+2  # Mass of the tank plus some extra mass for the tank structure (end caps, welds, internal pipes) [kg]
        ox_mdot = ox_mdot_array[of_array.index(target_of)]  # Oxidizer mass flow rate in kg/s for target O/F ratio
        fuel_mdot = fuel_mdot_array[of_array.index(target_of)]  # Fuel mass flow rate in kg/s for target O/F ratio
        
        rocket_mass = total_ox_mass + total_fuel_mass + recovery_mass + engine_mass + payload_mass + tank_mass  # Total mass of the rocket in kg
        
        # Setup time array
        times = np.arange(0, self.burn_time, dt)
        A = np.pi * (0.152 / 2) ** 2  # Cross-sectional area of the rocket in m^2 (diameter of 0.152 m)


        for t in times:
            m = rocket_mass - (ox_mdot + fuel_mdot) * t
            g = self.calc_gravity(h)
            rho = self.calc_air_density(h)
            drag = 0.5 * rho * self.Cd * A * v**2
            if v < 0: drag *= -1  # drag always opposes motion

            a = (self.thrust - m * g - drag) / m
            v += a * dt
            h += v * dt
            acceleration.append(a)
            velocities.append(v)
            altitudes.append(h)
            machs.append(v/c)
            # print(f"Time: {t:.2f} s, Velocity: {v:.2f} m/s mach {v/c}, Altitude: {h:.2f} m, Mass: {m:.2f} kg")  # Print the time, velocity, altitude, and mass at each time step

        # After burnout (coasting, no thrust)
        v_burnout = velocities[-1]
        h_burnout = altitudes[-1]
        h = h_burnout
        v = v_burnout

        while v > 0:
            g = self.calc_gravity(h)
            rho = self.calc_air_density(h)
            drag = 0.5 * rho * self.Cd * A * v**2
            a = -(g + drag / m)
            v += a * dt
            h += v * dt
            t += dt
            times = np.append(times, t)  # Append the time to the times array
            altitudes.append(h)  # Append the altitude to the altitudes array
            acceleration.append(a)  # Append the acceleration to the acceleration array
            velocities.append(v)  # Append the velocity to the velocities array
            machs.append(v/c) # Append the Mach number to the machs array

            # print(f"Time: {t:.2f} s, Velocity: {v:.2f} m/s, mach {v/c}, Altitude: {h:.2f} m, Mass: {m:.2f} kg")  # Print the time, velocity, altitude, and mass at each time step


        print(f"Final Time: {t:.2f} s, Final Velocity: {v:.2f} m/s, Final Altitude: {h:.2f} m, Final Mass: {m:.2f} kg")  # Print the final time, velocity, altitude, and mass
        # Display data
        print(f"Engine propellant mass flow rate {(ox_mdot + fuel_mdot):.2f} kg/s, Oxidizer mass flow rate {ox_mdot:.2f} kg/s, Fuel mass flow rate {fuel_mdot:.2f} kg/s")  # Print the propellant mass flow rate
        print(f"Total wet mass of the rocket: {rocket_mass:0.2f} kg, Total dry mass: {recovery_mass + engine_mass + payload_mass + tank_mass} for O/F ratio of {target_of}")  # Print the total mass of the rocket
        print(f"Total oxidizer mass: {total_ox_mass:0.2f} kg \nTotal fuel mass: {total_fuel_mass:.2f} kg \nTank Length: {tank_length:.2f} m \nTank Mass: {tank_mass:.2f} kg \nEngine Mass: {engine_mass} kg \nPayload: {payload_mass} kg")  # Print the total oxidizer and fuel mass
        print(f"Estimated altitude reached: {h:.2f} m")  # Print the estimated altitude reached by the rocket
        plt.figure(figsize=(8, 5))
        plt.plot(times, altitudes, label='Altitude (m)')
        plt.xlabel('Time (s)')
        plt.ylabel('Altitude (m)')
        plt.title(f'Altitude vs Time for O/F={target_of}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(times, machs, label='mach')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title(f'Velocity vs Time for O/F={target_of}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(times, acceleration, label='Acceleration (m/s²)')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.title(f'Acceleration vs Time for O/F={target_of}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        pass



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
    RS25 = RocketEngine('LOX', 'LH2', Pc, expansion_ratio, burn_time, engine_thrust_n, num_engines=number_of_engines, Cd = 0.2) # Verification Engine: RS25 - Cryo-rocket.com
    engine_name = 'RS25'  # Name of the engine for display purposes

    of_ratios = np.arange(0.8, 21, 0.1)  # Define O/F ratios for the RS25 engine

    # of_ratios =[6]
    # Display the performance curves for the RS25 engine
    #RS25.display_MR_curves(of_ratios, fuel_stroage_density, ox_storage_density, engine_name=engine_name)

    # Loop through each O/F ratio and extract the CEA output
    of_data = RS25.extract_cea_output(of_ratios)

    RS25.display_MR_curves(of_ratios, fuel_stroage_density, ox_storage_density, engine_name=engine_name)  # Display the performance curves for the RS25 engine

    # Display CEA output data for OF ratio of 6
    # RS25.extract_OF_info(6)  # Display CEA output data for O/F ratio of 6
 
    pass

def The5k():
    # FUNCTION WORKSPACE
    # This function will be used to run through the steps to design the 5k rocket engine
    
    engine_name = 'The5k'  # Name of the engine for display purposes

    # Define Rocket Key Information
    rocketMass = 90.0 # Estimated mass of the rocket in kg
    desired_engine_thrust = 5000 # Estimated thrust sea level  [N]
    Pc = 20 # Chamber presssure in bar (desired)
    fuel_stroage_density = 810 #795.6  # Density of RP-1 in kg/m^3 (approximate)
    ox_storage_density = 1140.0  # Density of LOX in kg/m^3 (approximate)
    burn_time = 23  # Estimated burn time in seconds
    expansion_ratio = 69  # No fixed expansion ratio for the 5k engine yet... 

    # Calculate thrust to weight ratio
    thrust_to_weight_ratio = desired_engine_thrust / (rocketMass * g)  # Thrust to weight ratio

    # o/f Ratio array to find the optimal O/F ratio
    of_ratios = np.arange(0.5, 7, 0.1)  # Define O/F ratios for the 5k engine

    # Function to run rocket engine calculations/design
    The5k = RocketEngine('LOX', 'RP-1', Pc, expansion_ratio, burn_time, desired_engine_thrust, num_engines=1, Cd = 0.4) # Verification Engine: RS25 - Cryo-rocket.com
    of_data = The5k.display_MR_curves(of_ratios, fuel_stroage_density, ox_storage_density, engine_name=engine_name, target_of=2) # Display the performance curves for the 5k engine to determine optimal O/F

    The5k.estimate_altitude(of_data, target_of=2, dt=0.1, recovery_mass=5, engine_mass=10, payload_mass=2, tank_mass_per_meter=6)  # Estimate altitude the rocket will reach with the optimal O/F ratio
    pass

def  main():
    # Call functions and run code here


    # RS25_Design()

    The5k()



    



main()

