"""
validate_flow.py
Visual and numerical validation of the flow_solver isentropic solver.

Run with:
    python validate_flow.py

Produces:
  - 4-panel plot: M, N=M², P, T vs axial position for the 5kN RP-1/LOX engine
  - RS25 Mc comparison against isentropic hand-calculation
"""

import numpy as np
import matplotlib.pyplot as plt

from config import EngineConfig
from cea_interface import get_cea_for_analysis
from geometry import size_engine, exit_mach
from flow_solver import solve_flow, isentropic_Mc, integrate


# -----------------------------------------------------------------------
# Engine configurations
# -----------------------------------------------------------------------
RP1_CFG = EngineConfig(
    fuel="RP-1", oxidizer="LOX", coolant="RP1",
    P_c=20e5, F_vac=5500,
    OF=2.4, exp_ratio=8.0, cont_ratio=6.0, L_star=1.143,
    theta1=30.0, thetaD=30.0, thetaE=12.0,
    R_chamber_mult=0.5, R_throat_conv_mult=1.5, R_throat_div_mult=0.382,
    wall_k=167.0,           # 6061-T6 Al
    wall_roughness=6.3e-6,  # SLM 3D-print
    wall_melt_T=855.0,
    T_coolant_inlet=290.0,  # RP-1 at ambient
    P_coolant_inlet=30e5,   # 30 bar
    N_channels=60,
    dx=1e-3,                # 1mm increments
)

RS25_CFG = EngineConfig(
    fuel="LH2", oxidizer="LOX", coolant="Hydrogen",
    P_c=18.23E6,
    F_vac=2184076.8131,        # N (100% RPL)
    OF=6.0,
    exp_ratio=77.5,
    cont_ratio=2.699,       # A_c/A_t = (R_c/R_t)^2 = (8.9416/5.4416)^2
    L_star=0.914,           # 36 inches — RS25 LH2/LOX
    theta1=25.4167,
    thetaD=37.0,
    thetaE=5.3738,
    R_chamber_mult=0.3196,     # hardware, dimensionless (chamber-side arc)
    R_throat_conv_mult=0.9469, # R_U/R_t = 5.1527/5.4416
    R_throat_div_mult=0.3711,  # R_D/R_t = 2.019/5.4416
    wall_k=350.0,           # Copper alloy (NARloy-Z)
    wall_roughness=1.6e-6,  # Milled / electroformed
    wall_melt_T=1356.0,     # Cu melting point [K]
    T_coolant_inlet=52.0,   # Coolant Inlet [K]
    P_coolant_inlet=44.83e6,# Coolant Inlet Pressure [Pa]
    N_channels=390,
    dx=5e-3,                # Step size for 1-D analysis [m]
)


# -----------------------------------------------------------------------
# Helper: compute isentropic stagnation P and T
# -----------------------------------------------------------------------
def stag_P(P, M, gam):
    return P * (1 + (gam - 1) / 2 * M**2)**(gam / (gam - 1))


def stag_T(T, M, gam):
    return T * (1 + (gam - 1) / 2 * M**2)


# -----------------------------------------------------------------------
# Validate RP-1/LOX 5kN with plots
# -----------------------------------------------------------------------
def validate_rp1():
    print("=" * 60)
    print("5kN RP-1/LOX at O/F = 2.4")
    print("=" * 60)

    cfg = RP1_CFG
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)
    sol = solve_flow(geom, cea, cfg)

    gam = cea.gamma_c
    x_mm = (sol.x - geom.L_c) * 1000   # relative to throat [mm]

    # --- Verify isentropic stagnation conditions are constant ---
    P0_arr = stag_P(sol.P, sol.M, gam)
    T0_arr = stag_T(sol.T, sol.M, gam)
    P0_err = (P0_arr - cea.P_c) / cea.P_c * 100
    T0_err = (T0_arr - cea.T_c) / cea.T_c * 100

    print(f"\n  Stagnation consistency check:")
    print(f"    P0 max error: {np.max(np.abs(P0_err)):.4f} %  (should be ~0)")
    print(f"    T0 max error: {np.max(np.abs(T0_err)):.4f} %  (should be ~0)")

    # --- Check key stations ---
    i_throat = int(np.argmin(sol.A))
    i_exit   = -1
    M_exit_theory = exit_mach(cfg.exp_ratio, gam)

    print(f"\n  Key stations:")
    print(f"    Injector    x = {sol.x[0]*1000:.1f} mm  M = {sol.M[0]:.5f}  "
          f"P = {sol.P[0]/1e5:.3f} bar  T = {sol.T[0]:.1f} K")
    print(f"    Throat      x = {sol.x[i_throat]*1000:.1f} mm  M = {sol.M[i_throat]:.5f}  "
          f"P = {sol.P[i_throat]/1e5:.3f} bar  T = {sol.T[i_throat]:.1f} K")
    print(f"    Exit        x = {sol.x[i_exit]*1000:.1f} mm  M = {sol.M[i_exit]:.5f}  "
          f"P = {sol.P[i_exit]/1e5:.3f} bar  T = {sol.T[i_exit]:.1f} K")
    print(f"\n  M_exit: {sol.M[-1]:.5f}  (theory {M_exit_theory:.5f}  "
          f"error {abs(sol.M[-1]-M_exit_theory)/M_exit_theory*100:.3f}%)")

    # --- Isentropic throat relations ---
    P_t_theory = cea.P_c * (2 / (gam + 1))**(gam / (gam - 1))
    T_t_theory = cea.T_c * (2 / (gam + 1))
    print(f"\n  Throat vs isentropic critical conditions:")
    print(f"    P_throat:  {sol.P[i_throat]/1e5:.4f} bar   "
          f"theory: {P_t_theory/1e5:.4f} bar   "
          f"err: {abs(sol.P[i_throat]-P_t_theory)/P_t_theory*100:.3f}%")
    print(f"    T_throat:  {sol.T[i_throat]:.2f} K   "
          f"theory: {T_t_theory:.2f} K   "
          f"err: {abs(sol.T[i_throat]-T_t_theory)/T_t_theory*100:.3f}%")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"RP-1/LOX  5kN  Pc={cfg.P_c/1e5:.0f} bar  OF={cfg.OF}  ER={cfg.exp_ratio}\n"
        f"Isentropic first pass  (x relative to throat)",
        fontsize=11)

    # Throat and chamber-start markers
    def add_markers(ax):
        ax.axvline(0,         color='r', ls='--', lw=0.9, label='Throat')
        ax.axvline(-geom.L_c * 1000, color='g', ls=':', lw=0.9, label='Injector')
        ax.grid(True, alpha=0.3)

    # Mach number
    ax = axes[0, 0]
    ax.plot(x_mm, sol.M, 'b-', lw=1.5)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, label='M = 1')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('Mach number M')
    ax.set_title('Mach Number')
    ax.legend(fontsize=8)

    # N = M²
    ax = axes[0, 1]
    ax.plot(x_mm, sol.M**2, 'purple', lw=1.5)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, label='N = 1 (M = 1)')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('N = M²')
    ax.set_title('N = M²  (ODE state variable)')
    ax.legend(fontsize=8)

    # Pressure
    ax = axes[1, 0]
    ax.plot(x_mm, sol.P / 1e5, 'r-', lw=1.5)
    ax.axhline(cea.P_c / 1e5, color='gray', ls=':', lw=0.8, label=f'P_c = {cea.P_c/1e5:.0f} bar')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('Static pressure [bar]')
    ax.set_title('Static Pressure')
    ax.legend(fontsize=8)

    # Temperature
    ax = axes[1, 1]
    ax.plot(x_mm, sol.T, 'darkorange', lw=1.5)
    ax.axhline(cea.T_c, color='gray', ls=':', lw=0.8, label=f'T_c = {cea.T_c:.0f} K')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('Static temperature [K]')
    ax.set_title('Static Temperature')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show(block=False)

    return cea, geom, sol


# -----------------------------------------------------------------------
# Validate RS25 Mc against hand-calculated value
# -----------------------------------------------------------------------
def validate_rs25_mc_bruteforce(): 
    """
    Function to replicate the brute force method to find injector velocity (mc) based off MixtureOptimization

    Will run through the typical cea and engine sizing. Will then scrape through specified array of mach numbers.

    It will then produce plots of all viable solutions so user can determine which is the best (note plots will be limited to the first 10 solutions)

    Inputs:
    None
    Outputs:
    None
    """

    print()
    print("=" * 60)
    print("RS25 Injection Mach Validation - Brute Force Method")
    print("=" * 60)

    cfg = RS25_CFG
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg,cea)

    xf = 0.6027 # End of STMCC chamber in [m]

    mc_arr = np.arange(0.24, 0.26, 0.00001) # Specify injector arrays
    mc_arr = [0.269]
    for mc in mc_arr:
        sol = integrate(mc, cea.P_c, cea.T_c, geom, cea, cfg.dx,)
    
    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"LH2/LOX  RS25  Pc={cfg.P_c/1e5:.0f} bar  OF={cfg.OF}  ER={cfg.exp_ratio}\n"
        f"Isentropic first pass  (x relative to throat)",
        fontsize=11)

    # Throat and chamber-start markers
    def add_markers(ax):
        ax.axvline(sol.x[0],         color='r', ls='--', lw=0.9, label='Throat')
        ax.axvline(sol.x[-1], color='g', ls=':', lw=0.9, label='Injector')
        ax.grid(True, alpha=0.3)

    # Mach number
    ax = axes[0, 0]
    ax.plot(sol.x, sol.M, 'b-', lw=1.5)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, label='M = 1')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('Mach number M')
    ax.set_title('Mach Number')
    ax.legend(fontsize=8)

    # N = M²
    ax = axes[0, 1]
    ax.plot(sol.x, sol.M**2, 'purple', lw=1.5)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, label='N = 1 (M = 1)')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('N = M²')
    ax.set_title('N = M²  (ODE state variable)')
    ax.legend(fontsize=8)

    # Pressure
    ax = axes[1, 0]
    ax.plot(sol.x, sol.P / 1e5, 'r-', lw=1.5)
    ax.axhline(cea.P_c / 1e5, color='gray', ls=':', lw=0.8, label=f'P_c = {cea.P_c/1e5:.0f} bar')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('Static pressure [bar]')
    ax.set_title('Static Pressure')
    ax.legend(fontsize=8)

    # Temperature
    ax = axes[1, 1]
    ax.plot(sol.x, sol.T, 'darkorange', lw=1.5)
    ax.axhline(cea.T_c, color='gray', ls=':', lw=0.8, label=f'T_c = {cea.T_c:.0f} K')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('Static temperature [K]')
    ax.set_title('Static Temperature')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show(block=False)

    return None


def validate_rs25_mc():
    print()
    print("=" * 60)
    print("RS25 Injection Mach Validation")
    print("=" * 60)

    cfg = RS25_CFG
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)

    gam  = cea.gamma_c
    CR   = geom.A_c / geom.A_t    # actual contraction ratio from sizing
    Mc   = isentropic_Mc(CR, gam)

    # Also run the full flow solution
    sol = solve_flow(geom, cea, cfg, xf=0.604)
    x_mm = (sol.x - geom.L_c) * 1000   # relative to throat [mm]

    # Isentropic Mc for EXACTLY the nominal CR=2.17 with the CEA gamma
    Mc_nominal_CR = isentropic_Mc(cfg.cont_ratio, gam)

    print(f"\n  CEA results:")
    print(f"    γ_c      = {gam:.5f}")
    print(f"    T_c      = {cea.T_c:.1f} K")
    print(f"    P_c      = {cea.P_c/1e5:.2f} bar")

    print(f"\n  Geometry:")
    print(f"    CR (from sizing) = {CR:.5f}  (configured {cfg.cont_ratio})")
    print(f"    D_throat = {2*geom.R_t*1000:.2f} mm   (hardware ~263 mm)")

    print(f"\n  Injection Mach:")
    print(f"    Mc (from sized CR={CR:.4f}, γ={gam:.4f}) = {Mc:.6f}")
    print(f"    Mc (from nominal CR={cfg.cont_ratio:.2f},   γ={gam:.4f}) = {Mc_nominal_CR:.6f}")
    print(f"    Mc from old code (hand-tuned brute-force)  = 0.264000  (for reference)")
    print(f"    Mc from flow solution at x=0               = {sol.Mc:.6f}")

    # Verify using hand formula: A/A* = (1/Mc)*((1+(γ-1)/2*Mc²)/((γ+1)/2))^((γ+1)/(2(γ-1)))
    AR_check = ((1.0 / Mc)
                * ((1.0 + (gam - 1.0) / 2.0 * Mc**2)
                   / ((gam + 1.0) / 2.0))**((gam + 1.0) / (2.0 * (gam - 1.0))))
    print(f"\n  Self-check:  isentropic_Mc({CR:.4f}) = {Mc:.6f}")
    print(f"    → A/A* = {AR_check:.6f}   (should equal {CR:.6f})")

    # Isentropic injection velocity
    R_sp = cea.R_specific
    v_c  = Mc * np.sqrt(gam * R_sp * cea.T_c)
    print(f"\n  Injection velocity at Mc = {Mc:.5f}:")
    print(f"    v_c = Mc * sqrt(γ R T_c) = {Mc:.5f} * sqrt({gam:.4f} * {R_sp:.2f} * {cea.T_c:.0f})")
    print(f"        = {v_c:.2f} m/s")

    # RS25 exit Mach theory
    M_exit_theory = exit_mach(cfg.exp_ratio, gam)
    print(f"\n  Exit Mach check:")
    print(f"    M_exit (isentropic, AR={cfg.exp_ratio}) = {M_exit_theory:.4f}")
    print(f"    M_exit from flow solution               = {sol.M[-1]:.4f}")
    print(f"    error = {abs(sol.M[-1]-M_exit_theory)/M_exit_theory*100:.4f}%")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"LH2/LOX  RS25  Pc={cfg.P_c/1e5:.0f} bar  OF={cfg.OF}  ER={cfg.exp_ratio}\n"
        f"Isentropic first pass  (x relative to throat)",
        fontsize=11)

    # Throat and chamber-start markers
    def add_markers(ax):
        ax.axvline(0,         color='r', ls='--', lw=0.9, label='Throat')
        ax.axvline(-geom.L_c * 1000, color='g', ls=':', lw=0.9, label='Injector')
        ax.grid(True, alpha=0.3)

    # Mach number
    ax = axes[0, 0]
    ax.plot(x_mm, sol.M, 'b-', lw=1.5)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, label='M = 1')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('Mach number M')
    ax.set_title('Mach Number')
    ax.legend(fontsize=8)

    # N = M²
    ax = axes[0, 1]
    ax.plot(x_mm, sol.M**2, 'purple', lw=1.5)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, label='N = 1 (M = 1)')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('N = M²')
    ax.set_title('N = M²  (ODE state variable)')
    ax.legend(fontsize=8)

    # Pressure
    ax = axes[1, 0]
    ax.plot(x_mm, sol.P / 1e5, 'r-', lw=1.5)
    ax.axhline(cea.P_c / 1e5, color='gray', ls=':', lw=0.8, label=f'P_c = {cea.P_c/1e5:.0f} bar')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('Static pressure [bar]')
    ax.set_title('Static Pressure')
    ax.legend(fontsize=8)

    # Temperature
    ax = axes[1, 1]
    ax.plot(x_mm, sol.T, 'darkorange', lw=1.5)
    ax.axhline(cea.T_c, color='gray', ls=':', lw=0.8, label=f'T_c = {cea.T_c:.0f} K')
    add_markers(ax)
    ax.set_xlabel('x from throat [mm]')
    ax.set_ylabel('Static temperature [K]')
    ax.set_title('Static Temperature')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show(block=False)

    return cea, geom, sol


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # cea_rp1, geom_rp1, sol_rp1 = validate_rp1()
    # cea_rs25, geom_rs25, sol_rs25 = validate_rs25_mc()
    validate_rs25_mc_bruteforce()
    plt.show()
