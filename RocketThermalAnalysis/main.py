"""
main.py
Top-level entry point. Edit the EngineConfig below to define your engine,
then run:  python main.py
"""

import numpy as np
import matplotlib.pyplot as plt

from config import EngineConfig
from cea_interface import get_cea_for_analysis
from geometry import size_engine, plot_contour, nozzle_radius
from flow_solver import solve_flow
from heat_transfer import ChannelGeometry, solve_thermal, plot_thermal
from film_cooling import compute_film_taw


# =============================================================================
# ENGINE CONFIG — edit this section
# =============================================================================
config = EngineConfig(
    # Propellants
    fuel        = "RP-1",
    oxidizer    = "LOX",
    coolant     = "RP1",        # CoolProp name (used later by coolant_props)

    # Performance
    P_c   = 20.0e5,             # 60 bar [Pa]
    F_vac = 5500.0,             # [N]

    # O/F — uncomment / set both for sweep + analysis
    OF        = 2.0,
    OF_sweep  = (1.5, 8, 0.1),

    # CEA
    frozen = False,

    # Geometry
    exp_ratio   = 8.0,
    cont_ratio  = 6.0,
    L_star      = 1.143,        # 45 inch — typical RP-1/LOX

    # Nozzle contour
    theta1  = 30.0,
    thetaD  = 30.0,
    thetaE  = 12.0,
    R1_mult = 1.5,
    RU_mult = 1.5,
    RD_mult = 0.382,

    # Wall material (6061-T6 Al)
    wall_k         = 167.0,     # [W/m*K]
    wall_roughness = 6.3e-6,    
    wall_melt_T    = 855.0,     # [K]

    # Coolant inlet (counter-current from nozzle exit)
    T_coolant_inlet = 290.0,    # [K]
    P_coolant_inlet = 35.0e5,   # [Pa]
    mdot_coolant    = None,     # computed from OF + mdot_total

    # Channels
    N_channels = 36,
    dx         = 6e-3,

    # Film Cooling 
    film_fraction  = 0.15,   # 5% of fuel flow as film
    film_inject_x  = 0.0,    # inject at injector face
    film_coolant   = "RP1",
    film_T_inlet   = 400.0,
    film_Kt        = 0.0013, # turbulent mixing intensity (Vasiliev 1993, range 0.0005-0.002)

    wall_2d=True,
    use_integral_bl=True,
    C_bartz=0.023,          # Thick BL calibration (Bartz 1965 Fig 10)
)



# =============================================================================
# RS25 ENGINE CONFIG — Same config as MixtureOptimazation.py Output aligns relatively with Betti/Wang/CryoRocket.com
# =============================================================================
# config = EngineConfig(
#     fuel="LH2", oxidizer="LOX", coolant="Hydrogen",
#     P_c=18.23E6,
#     F_vac=2184076.8131,        # N (100% RPL)
#     OF=6.0,
#     exp_ratio=69.5,
#     cont_ratio=2.699,       # A_c/A_t = (R_c/R_t)^2 = (8.9416/5.4416)^2
#     L_star=0.914,           # 36 inches — RS25 LH2/LOX
#     theta1=25.4167,         # Hardware convergence half-angle [deg]
#     thetaD=37.0,
#     thetaE=5.3738,
#     R1_mult=0.3196,         # R_1/R_t = 1.73921/5.4416 (hardware, dimensionless)
#     RU_mult=0.9469,         # R_U/R_t = 5.1527/5.4416
#     RD_mult=0.3711,         # R_D/R_t = 2.019/5.4416
#     wall_k=316.0,           # Copper alloy (NARloy-Z) — matches MixtureOptimization.py
#     wall_roughness=2.5e-7,  # Milled / electroformed
#     wall_melt_T=1356.0,     # Cu melting point [K]
#     T_coolant_inlet=52.0,   # Coolant Inlet [K]
#     P_coolant_inlet=44.82e6,# Coolant Inlet Pressure [Pa] — 4.482e7 per Wang & Luong
#     mdot_coolant=14.31,     # [kg/s] STMCC circuit only: 31.54 lb/s per Wang & Luong (1994)
#                             # RS25 splits total LH2 flow — only ~21% goes through STMCC channels
#     N_channels=390,
#     dx=1e-3,                # Step size for 1-D analysis [m]
#     wall_2d=True,           # 2-D wall conduction (Betti quasi-2D method)
# )

# =============================================================================
# BETTI FPL VALIDATION — Betti, Pizzarelli & Nasuti (J. Prop. Power, 2014)
# SSME MCC at FPL (109% rated thrust), standard throat, ε=5
# =============================================================================
# config = EngineConfig(
#     fuel="LH2", oxidizer="LOX", coolant="Hydrogen",
#     P_c=22.587e6,               # 225.87 bar — FPL chamber pressure
#     F_vac=2015429.0,            # [N] — calibrated for D_t=261.75mm
#     OF=6.0,
#     exp_ratio=5.0,              # MCC only (not full nozzle)
#     cont_ratio=3.0,             # Betti Sec III
#     L_star=0.914,               # 36 inches
#     theta1=25.4167,
#     thetaD=37.0,
#     thetaE=10.0,                # Wider exit angle for short MCC nozzle
#     R1_mult=0.3196,
#     RU_mult=0.9469,
#     RD_mult=0.3711,
#     wall_k=316.0,               # NARloy-Z at 533 K
#     wall_roughness=2.3e-7,      # 0.23 μm — Betti "rough" case
#     wall_melt_T=1356.0,
#     T_coolant_inlet=53.89,      # Betti Sec III
#     P_coolant_inlet=44.547e6,   # 445.47 bar — Betti Sec III
#     mdot_coolant=14.306,        # Betti Sec III
#     N_channels=390,
#     dx=1e-3,
#     wall_2d=True,
#     use_integral_bl=True,
#     C_bartz=0.023,          # Thick BL calibration (Bartz 1965 Fig 10)
# )


# =============================================================================
# END CONFIG
# =============================================================================


def run():
    # --- Step 1: CEA ---
    cea_result = get_cea_for_analysis(config)
    if cea_result is None:
        # Only a sweep was requested — plots shown, nothing further to do
        return

    # --- Step 2: Engine geometry ---
    geom = size_engine(config, cea_result)
    plot_contour(geom, dx=5e-4)

    # Derive coolant mass flow if not set (assume fuel-cooled)
    if config.mdot_coolant is None:
        config.mdot_coolant = geom.mdot_fuel
        print(f"\nDerived mdot_coolant = mdot_fuel = {geom.mdot_fuel:.4f} kg/s")

    # Define Cooling Channel Geometry
    x_j = np.arange(0, geom.L_c + geom.L_nozzle, config.dx) # Create slices at each dx
    chan_h = np.full(shape=(len(x_j),), fill_value=2e-3) # 2mm constant height channel at each slice
    chan_t = np.full(shape=(len(x_j),), fill_value=1e-3) # 1mm Constant thickness channel at each slice
    chan_land = np.full(shape=(len(x_j),), fill_value= 1.5e-3) # 1.5mm Constant land width channel at each slice
    chan_w = np.zeros(len(x_j)) # Pre-fill in chan_w for each slcie with a zero

    # Calculate width at each slice based off land width
    for i, x in enumerate(x_j):
        r = nozzle_radius(x, geom, config.dx) # Calculate the radius of the slice
        circ = 2*np.pi*r # Calculate the circumference of the slice
        avail_width = circ/config.N_channels # Calculate available width at each slice
        chan_w[i] = avail_width - chan_land[i] # Calculate width at each slice
   
    
    # Display channel geometry just like in RPA
    # for i, x in enumerate(x_j):
    #     print(f"x: {x:.4f} [m] hc: {chan_h[i]*1000:.4f} [mm] a: {chan_w[i]*1000:.4f} b: {chan_land[i]*1000:.4f} [mm]")
 
    chan_geom = ChannelGeometry(x_j, chan_w, chan_h, chan_t, chan_land,)

    # chan_geom = ChannelGeometry(
    #     x_j       = np.array([0.0, 0.0127, 0.0315, 0.0508, 0.0762, 0.1016, 0.127, 0.1524, 0.1778, 0.2032, 0.2286, 0.254, 0.2667, 0.2794, 0.2921, 0.3048, 0.3175, 0.32512, 0.3302, 0.3429, 0.3556, 0.381, 0.4064, 0.4318, 0.4572, 0.4826, 0.508, 0.5334, 0.5588, 0.6027,]),
    #     chan_w    = np.array([0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001509, 0.001217, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001227, 0.001575, 0.001575, 0.001575,]),   # 1.5 mm channel width
    #     chan_h    = np.array([0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.003442, 0.004953, 0.004953, 0.004953, 0.004953, 0.005352, 0.006096, 0.006096, 0.006096,]),   # 3.0 mm channel height
    #     chan_t    = np.array([0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000889, 0.000889,]),   # 0.8 mm wall thickness
    #     chan_land = np.array([0.002068, 0.002068, 0.002068, 0.002068, 0.002068, 0.002045, 0.001976, 0.001857, 0.001748, 0.001844, 0.001847, 0.001653, 0.001562, 0.001463, 0.001361, 0.001275, 0.001196, 0.001261, 0.001143, 0.001113, 0.001105, 0.001209, 0.001516, 0.001603, 0.001554, 0.001844, 0.002131, 0.002405, 0.002685, 0.003155,]),   # 1.0 mm land width
    # )


    # --- Step 3: Adiabatic flow solution (isentropic first pass) ---
    # Limit to the axial range covered by the cooling channel geometry
    flow = solve_flow(geom, cea_result, config, xf=chan_geom.x_j[-1])

    # --- Step 4: Film cooling T_aw correction (if configured) ---
    T_aw_eff = compute_film_taw(flow, geom, cea_result, config)

    # --- Step 5: Thermal analysis ---
    thermal = solve_thermal(flow, geom, cea_result, chan_geom, config,
                            T_aw_eff=T_aw_eff if config.film_fraction > 0.0 else None)

    # --- Step 5b: Iterate film temp if using regen exit as injection ---
    if config.film_fraction > 0.0 and config.film_T_from_regen:
        for film_iter in range(5):
            T_regen_exit = float(thermal.T_coolant[0])   # index 0 = injector face
            if abs(T_regen_exit - config.film_T_inlet) < 1.0:
                print(f"  Film T_inlet converged at {T_regen_exit:.1f} K "
                      f"(iter {film_iter + 1})")
                break
            print(f"\n--- Film iteration {film_iter + 1}: "
                  f"T_film_inlet {config.film_T_inlet:.1f} → {T_regen_exit:.1f} K ---")
            config.film_T_inlet = T_regen_exit
            T_aw_eff = compute_film_taw(flow, geom, cea_result, config)
            thermal = solve_thermal(flow, geom, cea_result, chan_geom, config,
                                    T_aw_eff=T_aw_eff)

    plot_thermal(thermal, geom, config)

    plt.show()  # Keep all plot windows open


if __name__ == "__main__":
    run()
