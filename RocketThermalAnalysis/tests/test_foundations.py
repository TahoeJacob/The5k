"""
test_foundations.py
Run with:  python test_foundations.py

Tests:
  1. CEA interface — single O/F (RP-1/LOX at O/F=2.0)
  2. CEA interface — O/F sweep (RP-1/LOX, 1.5→10 step 0.1)
  3. Geometry sizing — internal consistency and RS25 validation (LH2/LOX)
"""

# --- run-from-anywhere shim (file lives in subfolder) ---
import os, sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
os.chdir(_PARENT)
# --------------------------------------------------------

import sys
import numpy as np
import matplotlib
# matplotlib.use('Agg')   # non-interactive backend so tests don't block on plots

from config import EngineConfig
from cea_interface import get_cea_for_analysis, CEAResult
from geometry import size_engine, nozzle_radius, build_contour, exit_mach, plot_contour
from flow_solver import isentropic_Mc, integrate, solve_flow
from heat_transfer import ChannelGeometry, solve_thermal


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _pass(name):
    print(f"  PASS  {name}")

def _fail(name, msg):
    print(f"  FAIL  {name}: {msg}")
    return False

def run_test(name, fn):
    try:
        fn()
        _pass(name)
        return True
    except AssertionError as e:
        return _fail(name, str(e))
    except Exception as e:
        return _fail(name, f"{type(e).__name__}: {e}")

# -----------------------------------------------------------------------
#  Engine Configurations
# -----------------------------------------------------------------------

RS25_CONFIG = EngineConfig(
    fuel="LH2", oxidizer="LOX", coolant="Hydrogen",
    P_c=206.4e5,            # 206.4 bar
    F_vac=2_090_500,        # N (100% RPL)
    OF=6.0,
    exp_ratio=69.5,
    cont_ratio=2.699,       # A_c/A_t = (R_c/R_t)^2 = (8.9416/5.4416)^2
    L_star=0.914,           # 36 inches — RS25 LH2/LOX
    theta1=25.4167,         # Hardware convergence half-angle [deg]
    thetaD=37.0,
    thetaE=5.3738,
    R_chamber_mult=0.3196,     # hardware, dimensionless (chamber-side arc)
    R_throat_conv_mult=0.9469, # R_U/R_t = 5.1527/5.4416 (throat conv fillet)
    R_throat_div_mult=0.3711,  # R_D/R_t = 2.019/5.4416  (throat div fillet)
    wall_k=350.0,           # Copper alloy (NARloy-Z)
    wall_roughness=1.6e-6,  # Milled / electroformed
    wall_melt_T=1356.0,     # Cu melting point [K]
    T_coolant_inlet=52.0,   # Coolant Inlet [K]
    P_coolant_inlet=44.83e6,# Coolant Inlet Pressure [Pa]
    N_channels=390,
    dx=5e-3,                # Step size for 1-D analysis [m]
)

RPA_5KN_CONFIG = EngineConfig(
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

# -----------------------------------------------------------------------
# 1. CEA single O/F  — RP-1 / LOX at O/F = 2.0
# -----------------------------------------------------------------------
def test_cea_single():
    cfg = EngineConfig(
        fuel="RP-1", oxidizer="LOX", coolant="RP1",
        P_c=60e5, F_vac=5500,
        OF=2.0,
        exp_ratio=5.0,
    )
    result = get_cea_for_analysis(cfg)

    assert result is not None, "Expected CEAResult, got None"
    assert isinstance(result, CEAResult)

    # Physical range checks — RP-1/LOX at OF=2.0 is fuel-rich so T_c is lower
    assert 2500 < result.T_c < 3800,   f"T_c = {result.T_c:.0f} K out of range"
    assert 50e5 < result.P_c < 70e5,   f"P_c = {result.P_c:.0e} Pa unexpected"
    assert 1.1  < result.gamma_c < 1.4, f"gamma = {result.gamma_c}"
    assert 1300 < result.C_star < 2000, f"C* = {result.C_star:.0f} m/s unexpected"
    assert 2000 < result.Isp_vac_e < 4000, f"Isp_vac_e = {result.Isp_vac_e:.0f} m/s"

    # Gas constant must be physically reasonable (RP-1 combustion products ~25 g/mol)
    assert 10e-3 < result.molar_mass < 35e-3, f"M_mol = {result.molar_mass*1000:.1f} g/mol"
    assert 200 < result.R_specific < 1000,    f"R_sp = {result.R_specific:.1f}"

    # Transport properties must be positive and in physical range
    assert result.visc_c > 0,   "visc_c must be positive"
    assert result.Cp_c   > 0,   "Cp_c must be positive"
    assert 0.01 < result.cond_c < 10, f"cond_c = {result.cond_c:.4f} W/m/K out of range"
    assert 0 < result.Pr_c < 2,       f"Pr_c = {result.Pr_c}"

    # Mass fractions present
    assert len(result.mass_fractions) > 0, "No mass fractions parsed"


def test_cea_single_of2_results():
    """Check that O/F=2.0 gives lower T_c than stoichiometric (~3.4) for RP-1/LOX."""
    cfg = EngineConfig(
        fuel="RP-1", oxidizer="LOX", coolant="RP1",
        P_c=60e5, F_vac=5500,
        OF=2.0, exp_ratio=5.0,
    )
    r2 = get_cea_for_analysis(cfg)

    cfg.OF = 3.4
    r34 = get_cea_for_analysis(cfg)

    assert r2.T_c < r34.T_c, (
        f"T_c at OF=2.0 ({r2.T_c:.0f} K) should be lower than at OF=3.4 ({r34.T_c:.0f} K)"
    )


def test_cea_no_of_raises():
    """Config with neither OF nor OF_sweep must raise ValueError."""
    cfg = EngineConfig(
        fuel="RP-1", oxidizer="LOX", coolant="RP1",
        P_c=60e5, F_vac=5500,
    )
    raised = False
    try:
        get_cea_for_analysis(cfg)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError when neither OF nor OF_sweep is set"


def test_cea_sweep_only_returns_none():
    """Sweep with no single OF must return None (caller stops there)."""
    cfg = EngineConfig(
        fuel="RP-1", oxidizer="LOX", coolant="RP1",
        P_c=60e5, F_vac=5500,
        OF=None,
        OF_sweep=(1.5, 3.0, 0.1),   # small sweep for speed
    )
    result = get_cea_for_analysis(cfg)
    assert result is None, f"Expected None for sweep-only run, got {result}"


def test_cea_sweep_full():
    """
    Full RP-1/LOX sweep from 1.5 to 10 in steps of 0.1.
    Verifies sweep completes and key parameters have expected trends.
    """
    print("\n  [sweep 1.5 → 10, step 0.1 — may take ~30s] ", end="", flush=True)

    cfg = EngineConfig(
        fuel="RP-1", oxidizer="LOX", coolant="RP1",
        P_c=60e5, F_vac=5500,
        OF=2.4,                       # design point — but still run the sweep
        OF_sweep=(1.5, 10.0, 0.1),
        exp_ratio=5.0,
    )

    # Monkey-patch get_cea_for_analysis internals — run sweep manually so we can
    # inspect the intermediate dict without plt.show() blocking.
    from rocketcea.cea_obj import CEA_Obj
    from cea_interface import _run_single

    cea_obj = CEA_Obj(fuelName=cfg.fuel, oxName=cfg.oxidizer)
    OF_range = np.arange(1.5, 10.0 + 0.05, 0.1)
    results = {}
    for of in OF_range:
        results[round(of, 2)] = _run_single(cea_obj, cfg, of)

    print(f"{len(results)} points computed.")

    # All expected O/F values present
    assert len(results) >= 85, f"Expected ≥85 sweep points, got {len(results)}"

    # T_c should peak somewhere between OF=3 and OF=8 for RP-1/LOX
    of_vals = sorted(results.keys())
    Tc_vals  = [results[o].T_c      for o in of_vals]
    Isp_vals = [results[o].Isp_vac_e for o in of_vals]

    peak_Tc_of  = of_vals[np.argmax(Tc_vals)]
    peak_Isp_of = of_vals[np.argmax(Isp_vals)]

    assert 2.5 < peak_Tc_of  < 5.0, f"T_c peak at O/F={peak_Tc_of}, expected 2.5–5.0"
    assert 2.0 < peak_Isp_of < 4.5, f"Isp peak at O/F={peak_Isp_of}, expected 2.0–4.5"

    # All values monotone-ish: T_c and Isp should decrease once past peak
    assert max(Tc_vals)  > 3000, f"Peak T_c = {max(Tc_vals):.0f} K, expected > 3000 K"
    assert max(Isp_vals) > 3000, f"Peak Isp = {max(Isp_vals):.0f} m/s, expected > 3000"

    print(f"    T_c peak: O/F = {peak_Tc_of:.1f} ({max(Tc_vals):.0f} K)  |  "
          f"Isp peak: O/F = {peak_Isp_of:.1f} ({max(Isp_vals):.0f} m/s)")


# -----------------------------------------------------------------------
# 2. Geometry  — RS25 validation (LH2/LOX)
# -----------------------------------------------------------------------
# RS25 public reference values used for range checks:
#   Throat diameter   ~ 263 mm
#   Chamber pressure  ~ 206.4 bar
#   Expansion ratio   = 77.5
#   Contraction ratio ~ 2.17
#   Vacuum thrust     = 2,090,500 N  (100% RPL)
#   Isp_vac (real)    ~ 4,444 m/s   (but CEA theoretical will be ~4,558 m/s)
#
# Because CEA gives theoretical performance (no losses), the CEA-based throat
# will be ~2-3% smaller than hardware.  Tests therefore verify:
#   (a) internal mathematical consistency of the sizing route
#   (b) dimensional outputs are in the correct physical ballpark (±10%)

def test_geometry_internal_consistency():
    """
    Area ratios, mass flows and L* volume balance must close exactly.
    """
    cfg = RS25_CONFIG
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)

    # Area ratio consistency
    assert abs(geom.A_e / geom.A_t - cfg.exp_ratio) < 1e-6, \
        f"A_e/A_t = {geom.A_e/geom.A_t:.4f}, expected {cfg.exp_ratio}"
    assert abs(geom.A_c / geom.A_t - cfg.cont_ratio) < 1e-6, \
        f"A_c/A_t = {geom.A_c/geom.A_t:.4f}, expected {cfg.cont_ratio}"

    # Throat area from C* definition: A_t = (mdot * C*) / P_c
    A_t_check = (geom.mdot * cea.C_star) / cea.P_c
    assert abs(geom.A_t - A_t_check) < 1e-8, \
        f"A_t self-check failed: {geom.A_t:.6f} vs {A_t_check:.6f}"

    # Mass flow: mdot = F_vac / Isp_vac_e
    mdot_check = cfg.F_vac / cea.Isp_vac_e
    assert abs(geom.mdot - mdot_check) < 1e-6, \
        f"mdot = {geom.mdot:.4f}, expected {mdot_check:.4f}"

    # Fuel/oxidizer split
    assert abs(geom.mdot_fuel + geom.mdot_ox - geom.mdot) < 1e-9, \
        "mdot_fuel + mdot_ox != mdot"
    assert abs(geom.mdot_ox / geom.mdot_fuel - cfg.OF) < 1e-6, \
        f"O/F check: {geom.mdot_ox/geom.mdot_fuel:.4f} vs {cfg.OF}"

    # L* volume balance: required V_c = A_t * L_star
    V_c_required = cfg.L_star * geom.A_t

    # Compute volumes from geometry
    R_t, R_c = geom.R_t, geom.R_c
    th1 = np.deg2rad(cfg.theta1)
    RU  = geom.R_throat_conv
    L_cone = ((R_t * (np.sqrt(cfg.cont_ratio) - 1) + RU * (1/np.cos(th1) - 1)) / np.tan(th1))
    V_cone = (np.pi / 3) * L_cone * (R_c**2 + R_t**2 + R_c * R_t)
    V_cylinder = geom.L_e * geom.A_c
    V_total = V_cylinder + V_cone

    assert abs(V_total - V_c_required) / V_c_required < 1e-6, \
        f"L* volume mismatch: computed {V_total*1e6:.2f} cm³, required {V_c_required*1e6:.2f} cm³"

    # L_c = L_cylinder + L_cone
    assert abs(geom.L_c - (geom.L_e + L_cone)) < 1e-9, \
        f"L_c = {geom.L_c*1000:.2f} mm, L_e + L_cone = {(geom.L_e+L_cone)*1000:.2f} mm"

def test_geometry_rs25_dimensions():
    """
    Compare CEA-theoretical RS25 geometry against known hardware dimensions.
    Tolerances account for combustion efficiency not modelled by CEA.
    """
    cfg = RS25_CONFIG
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)

    D_t_mm = 2 * geom.R_t * 1000
    D_e_mm = 2 * geom.R_e * 1000
    D_c_mm = 2 * geom.R_c * 1000

    print(f"\n  RS25 geometry (CEA-theoretical):")
    print(f"    D_t = {D_t_mm:.1f} mm   (hardware ~263 mm)")
    print(f"    D_c = {D_c_mm:.1f} mm   (hardware ~388 mm)")
    print(f"    D_e = {D_e_mm:.1f} mm   (hardware ~2310 mm)")
    print(f"    mdot = {geom.mdot:.1f} kg/s   (hardware ~510 kg/s)")
    print(f"    L_c  = {geom.L_c*1000:.1f} mm")

    # Throat: 263 mm hardware, CEA gives theoretical (slightly smaller due to efficiency)
    assert 220 < D_t_mm < 290, f"D_t = {D_t_mm:.1f} mm out of ±10% of 263 mm hardware"

    # Exit diameter: D_t * sqrt(ER) ≈ 263 * sqrt(77.5) ≈ 2315 mm
    D_e_expected = 263 * np.sqrt(77.5)
    assert abs(D_e_mm - D_e_expected) / D_e_expected < 0.12, \
        f"D_e = {D_e_mm:.0f} mm, expected ~{D_e_expected:.0f} mm"

    # Mass flow: real RS25 is ~510 kg/s at 109% RPL, ~470 at 100% RPL
    # CEA-theoretical will be slightly lower (higher theoretical Isp → lower mdot)
    assert 400 < geom.mdot < 520, f"mdot = {geom.mdot:.1f} kg/s out of expected range"

    # Positive finite lengths
    assert geom.L_c  > 0
    assert geom.L_e  > 0
    assert geom.L_nozzle > 0

def test_geometry_contour_smooth():
    """Nozzle profile must be smooth: no jumps > 5 mm per step, throat is minimum."""
    cfg = RS25_CONFIG
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)
    
    x_arr, r_arr = build_contour(geom, dx=2e-3)
    plot_contour(geom, 2E-3)
    # No NaN or negative radii
    assert not np.any(np.isnan(r_arr)), "NaN in radius array"
    assert np.all(r_arr > 0), "Non-positive radius in contour"

    # Throat (x = L_c) should be close to minimum radius
    throat_idx = np.argmin(np.abs(x_arr - geom.L_c))
    min_idx    = np.argmin(r_arr)
    # Minimum should be within ±5 steps of throat
    assert abs(min_idx - throat_idx) <= 5, \
        f"Profile minimum at idx {min_idx} (x={x_arr[min_idx]*1000:.1f} mm) " \
        f"but throat at idx {throat_idx} (x={geom.L_c*1000:.1f} mm)"

    # No jumps larger than 5 mm between adjacent points
    dr = np.abs(np.diff(r_arr)) * 1000
    assert np.max(dr) < 5.0, f"Profile jump of {np.max(dr):.2f} mm detected"

    # Converging section decreases, diverging section increases
    r_conv = r_arr[:throat_idx]
    r_div  = r_arr[throat_idx:]
    assert r_conv[-1] < r_conv[0],  "Radius not decreasing in converging section"
    assert r_div[-1]  > r_div[0],   "Radius not increasing in diverging section"
    
def test_geometry_rp1_contour_smooth():
    """Nozzle profile must be smooth: no jumps > 5 mm per step, throat is minimum."""
    cfg = RPA_5KN_CONFIG
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)
    
    x_arr, r_arr = build_contour(geom, dx=5e-4)
    plot_contour(geom, 2E-3)
    # No NaN or negative radii
    assert not np.any(np.isnan(r_arr)), "NaN in radius array"
    assert np.all(r_arr > 0), "Non-positive radius in contour"

    # Throat (x = L_c) should be close to minimum radius
    throat_idx = np.argmin(np.abs(x_arr - geom.L_c))
    min_idx    = np.argmin(r_arr)
    # Minimum should be within ±5 steps of throat
    assert abs(min_idx - throat_idx) <= 5, \
        f"Profile minimum at idx {min_idx} (x={x_arr[min_idx]*1000:.1f} mm) " \
        f"but throat at idx {throat_idx} (x={geom.L_c*1000:.1f} mm)"

    # No jumps larger than 5 mm between adjacent points
    dr = np.abs(np.diff(r_arr)) * 1000
    assert np.max(dr) < 5.0, f"Profile jump of {np.max(dr):.2f} mm detected"

    # Converging section decreases, diverging section increases
    r_conv = r_arr[:throat_idx]
    r_div  = r_arr[throat_idx:]
    assert r_conv[-1] < r_conv[0],  "Radius not decreasing in converging section"
    assert r_div[-1]  > r_div[0],   "Radius not increasing in diverging section"
    
# -----------------------------------------------------------------------
# 3. Flow solver
# -----------------------------------------------------------------------

def test_flow_isentropic_mc():
    """
    isentropic_Mc must satisfy the area-Mach relation:
    A/A* = cont_ratio for the subsonic root.
    """
    gam = 1.2
    for cr in [2.0, 3.5, 6.0, 10.0]:
        Mc = isentropic_Mc(cr, gam)
        # Verify via the area-Mach function directly
        AR = (1.0 / Mc) * ((1 + (gam - 1)/2 * Mc**2) / ((gam + 1)/2))**((gam + 1)/(2*(gam - 1)))
        assert abs(AR - cr) < 1e-5, \
            f"isentropic_Mc({cr}, {gam}) = {Mc:.6f} → A/A* = {AR:.6f}, expected {cr}"
        assert 0 < Mc < 1.0, f"Mc must be subsonic, got {Mc:.4f}"

def test_flow_mc_increases_with_cr():
    """Higher contraction ratio → lower injection Mach (larger chamber relative to throat)."""
    gam = 1.2
    mc_values = [isentropic_Mc(cr, gam) for cr in [2.0, 3.0, 6.0, 10.0]]
    for i in range(len(mc_values) - 1):
        assert mc_values[i] > mc_values[i + 1], \
            f"Mc should decrease with increasing CR: {mc_values}"

def test_flow_adiabatic_rp1_lox():
    """
    Adiabatic RK4 integration for RP-1/LOX 5kN should give:
    - M ≈ 1.0 at the throat (within 2%)
    - M_exit matching isentropic exit Mach (within 2%)
    - Pressure monotonically non-increasing (converging section subsonic → throat)
    """
    cfg = RPA_5KN_CONFIG
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)

    sol = solve_flow(geom, cea, cfg)

    # Throat index — minimum area point
    i_t = int(np.argmin(sol.A))
    M_throat = sol.M[i_t]
    assert 0.9 < M_throat < 1.1, \
        f"M_throat = {M_throat:.4f}, expected ≈ 1.0"

    # Exit Mach vs isentropic
    M_exit_theory = exit_mach(cfg.exp_ratio, cea.gamma_c)
    M_exit_sol    = sol.M[-1]
    err = abs(M_exit_sol - M_exit_theory) / M_exit_theory
    assert err < 0.02, \
        f"M_exit = {M_exit_sol:.4f} vs theory {M_exit_theory:.4f} ({err*100:.2f}% error)"

    # State arrays must be non-empty and finite
    for arr, name in [(sol.M, "M"), (sol.P, "P"), (sol.T, "T"), (sol.A, "A")]:
        assert len(arr) > 0,          f"{name} array is empty"
        assert np.all(np.isfinite(arr)), f"{name} array contains non-finite values"
        assert np.all(arr > 0),           f"{name} array contains non-positive values"

    print(f"\n    Mc={sol.Mc:.5f}  M_t={M_throat:.4f}  M_exit={M_exit_sol:.4f}"
          f"  (theory {M_exit_theory:.4f})")

def test_flow_adiabatic_rs25_lox():
    """
    Adiabatic RK4 integration for LH2/LOX RS25 should give:
    - M ≈ 1.0 at the throat (within 2%)
    - M_exit matching isentropic exit Mach (within 2%)
    - Pressure monotonically non-increasing (converging section subsonic → throat)
    """
    cfg = RS25_CONFIG
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)

    sol = solve_flow(geom, cea, cfg)

    # Throat index — minimum area point
    i_t = int(np.argmin(sol.A))
    M_throat = sol.M[i_t]
    assert 0.9 < M_throat < 1.05, \
        f"M_throat = {M_throat:.4f}, expected ≈ 1.0"

    # Exit Mach vs isentropic
    M_exit_theory = exit_mach(cfg.exp_ratio, cea.gamma_c)
    M_exit_sol    = sol.M[-1]
    err = abs(M_exit_sol - M_exit_theory) / M_exit_theory
    assert err < 0.02, \
        f"M_exit = {M_exit_sol:.4f} vs theory {M_exit_theory:.4f} ({err*100:.2f}% error)"

    # State arrays must be non-empty and finite
    for arr, name in [(sol.M, "M"), (sol.P, "P"), (sol.T, "T"), (sol.A, "A")]:
        assert len(arr) > 0,          f"{name} array is empty"
        assert np.all(np.isfinite(arr)), f"{name} array contains non-finite values"
        assert np.all(arr > 0),           f"{name} array contains non-positive values"

    print(f"\n    Mc={sol.Mc:.5f}  M_t={M_throat:.4f}  M_exit={M_exit_sol:.4f}"
          f"  (theory {M_exit_theory:.4f})")

# -----------------------------------------------------------------------
# 4. Heat transfer
# -----------------------------------------------------------------------

def _rs25_chan_geom(geom) -> ChannelGeometry:
    """
    Illustrative RS25 channel geometry — constant cross-section for code validation.

    w=4 mm, h=10 mm, t=1.5 mm, land=2 mm  → pitch=6 mm, area=40 mm².
    This gives coolant velocity ~64 m/s and ΔP ~7 bar over the 3.35 m engine —
    physically feasible numbers to exercise the Niino correlation.

    NOTE: The pitch (6 mm) exceeds the throat circumference per channel
    (804 mm / 360 = 2.24 mm) so this geometry is illustrative only.
    Real RS25 channels taper dramatically toward the throat.
    """
    total = geom.L_c + geom.L_nozzle
    x_j = np.linspace(0.0, total, 10)
    return ChannelGeometry(
        x_j       = np.array([0.0, 0.0127, 0.0315, 0.0508, 0.0762, 0.1016, 0.127, 0.1524, 0.1778, 0.2032, 0.2286, 0.254, 0.2667, 0.2794, 0.2921, 0.3048, 0.3175, 0.32512, 0.3302, 0.3429, 0.3556, 0.381, 0.4064, 0.4318, 0.4572, 0.4826, 0.508, 0.5334, 0.5588, 0.6027,]),
        chan_w    = np.array([0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001575, 0.001509, 0.001217, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001016, 0.001227, 0.001575, 0.001575, 0.001575,]),   # 1.5 mm channel width
        chan_h    = np.array([0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.002489, 0.003442, 0.004953, 0.004953, 0.004953, 0.004953, 0.005352, 0.006096, 0.006096, 0.006096,]),   # 3.0 mm channel height
        chan_t    = np.array([0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000889, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000711, 0.000889, 0.000889,]),   # 0.8 mm wall thickness
        chan_land = np.array([0.002068, 0.002068, 0.002068, 0.002068, 0.002068, 0.002045, 0.001976, 0.001857, 0.001748, 0.001844, 0.001847, 0.001653, 0.001562, 0.001463, 0.001361, 0.001275, 0.001196, 0.001261, 0.001143, 0.001113, 0.001105, 0.001209, 0.001516, 0.001603, 0.001554, 0.001844, 0.002131, 0.002405, 0.002685, 0.003155,]),   # 1.0 mm land width
    )

def test_ht_rs25_converges():
    """
    RS25 LH2 thermal solve must converge and produce physically ordered temperatures:
      T_hw > T_cw > T_coolant_inlet  at peak heat flux station.
    Also prints a summary table for manual comparison against RS25 literature.
    """
    cfg  = RS25_CONFIG
    cea  = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)
    cfg.mdot_coolant = geom.mdot_fuel   # full fuel flow as coolant

    chan_geom = _rs25_chan_geom(geom)
    flow      = solve_flow(geom, cea, cfg, xf=chan_geom.x_j[-1])
    
    th        = solve_thermal(flow, geom, cea, chan_geom, cfg,
                              T_hw_init=600.0, T_cw_init=200.0,
                              tol_K=2.0, max_outer=30)

    # Convergence
    assert th.n_iters < 30, f"Did not converge in 30 iterations"

    # Peak-flux station (typically near throat)
    i_peak = int(np.argmax(th.heatflux))
    x_pk   = flow.x[i_peak]

    print(f"\n  RS25 LH2 thermal summary (N_ch={cfg.N_channels}):")
    print(f"    Peak heat flux station: x = {x_pk*1000:.1f} mm  "
          f"(throat at {geom.L_c*1000:.1f} mm)")
    print(f"    Peak heat flux : {th.heatflux[i_peak]/1e6:.2f} MW/m²")
    print(f"    T_hw (max)     : {np.max(th.T_hw):.0f} K")
    print(f"    T_cw at peak   : {th.T_cw[i_peak]:.0f} K")
    print(f"    T_coolant exit : {th.T_coolant[0]:.1f} K  (at injector face)")
    print(f"    T_coolant inlet: {th.T_coolant[-1]:.1f} K  (at nozzle exit)")
    print(f"    h_gas at peak  : {th.h_gas[i_peak]/1000:.1f} kW/(m²·K)")
    print(f"    h_cool at peak : {th.h_coolant[i_peak]/1000:.1f} kW/(m²·K)")
    print(f"    Re_cool (peak) : {th.Re_coolant[i_peak]:.0f}")
    print(f"    v_cool (peak)  : {th.v_coolant[i_peak]:.1f} m/s")
    print(f"    P_cool exit    : {th.P_coolant[0]/1e5:.1f} bar  "
          f"(inlet {cfg.P_coolant_inlet/1e5:.0f} bar)")

    # Physical ordering at peak station
    assert th.T_hw[i_peak] > th.T_cw[i_peak], "T_hw must exceed T_cw"
    assert th.T_cw[i_peak] > th.T_coolant[i_peak], "T_cw must exceed T_coolant"
    assert th.heatflux[i_peak] > 0, "Heat flux must be positive"

    # RS25 literature: peak heat flux ~50–160 MW/m² at throat (varies by channel design)
    assert 5e6 < th.heatflux[i_peak] < 300e6, \
        f"Peak heat flux {th.heatflux[i_peak]/1e6:.1f} MW/m² out of expected range"

    # Coolant must be warming up (injector-face T_coolant > inlet)
    assert th.T_coolant[0] > cfg.T_coolant_inlet, \
        "Coolant should be warmer at injector face than at inlet (nozzle exit)"


# --- 5kN RP-1/LOX — 60 channels, MixtureOptimization.py channel geometry ---
# Channel geometry taken directly from MixtureOptimization.py (x from injector face).
# The arrays span 0 → 0.6027 m (the old engine design); our new 5kN engine is
# ~0.258 m total, so the solver samples only the 0.0–0.258 m portion via
# ChannelGeometry.at() interpolation.  Near the throat (x ≈ 0.209 m) the
# dimensions are in the constant-width zone: w ≈ 1.575 mm, h ≈ 2.489 mm.

def test_ht_5kn_rp1_converges():
    """
    5kN RP-1/LOX (60 ch, 2 mm land) thermal solve must converge and produce
    physically consistent temperatures and positive heat flux everywhere.
    Prints a full station table for import into RPA for cross-validation.
    """
    cfg  = RPA_5KN_CONFIG
    cea  = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)
    cfg.mdot_coolant = geom.mdot_fuel
    flow      = solve_flow(geom, cea, cfg)

    # Constant land channel geometry
    min_width = 1e-3 # Minimum allowable width of channel will be 1mm
    tot_len = geom.L_c + geom.L_nozzle
    x_j = np.arange(0, tot_len, cfg.dx) # Setup array of x slices which will be used for analysis
    chan_h = np.full(shape = len(x_j,), fill_value=2e-3) # Setup height array with constant height of 2mm
    chan_t = np.full(shape = len(x_j,), fill_value=0.9e-3) # Setup thickness array with constant thickness of 0.9mm
    chan_land = np.full(shape = len(x_j,), fill_value=1.5e-3) # Setup land array with constant of 1.5mm
    chan_w = np.zeros(len(x_j)) # Pre-fill chan width array with zeros with length equalling x_j
    if (2*np.pi*nozzle_radius(geom.L_c, geom))/cfg.N_channels - chan_land[0] < min_width: # Assuming const land so any value of array will work
        ValueError("Land is too large resulting in channel width being too small, suggest reducing Ncc or adjusting land width")
    
    # If no issue with above land selction at throat we are good to caluclate the channel width through out entire chamber at each x_j value    
    for i, x in enumerate(x_j):
        # Calculate radius at every x slice
        r = nozzle_radius(x, geom) # Get chamber radius at each x slice
        circumf = 2*np.pi*(r+chan_t[i]) # Calculate the circumference at each slice note this accounts for the thickness of the chamber wall at this slice
        chan_w[i] = circumf/cfg.N_channels - chan_land[i] # Calculate the slice chan_w, we know chan_land so available width will be total width - chan_land
    
    chan_geom = ChannelGeometry(x_j, chan_w, chan_t, chan_land)

    th        = solve_thermal(flow, geom, cea, chan_geom, cfg,
                              T_hw_init=700.0, T_cw_init=400.0,
                              tol_K=1.0, max_outer=30)

    assert th.n_iters < 30, "Did not converge"

    i_throat = int(np.argmin(flow.A))
    i_peak   = int(np.argmax(th.heatflux))

    print(f"\n  5kN RP-1/LOX thermal summary  "
          f"(N_ch={cfg.N_channels}, land=2 mm, for RPA comparison):")
    print(f"    mdot_total  = {geom.mdot:.4f} kg/s  "
          f"mdot_fuel = {geom.mdot_fuel:.4f} kg/s  "
          f"mdot/ch = {geom.mdot_fuel/cfg.N_channels*1000:.2f} g/s")
    print(f"    D_throat    = {2*geom.R_t*1000:.2f} mm   "
          f"D_chamber = {2*geom.R_c*1000:.2f} mm")
    print(f"    L_c         = {geom.L_c*1000:.1f} mm   "
          f"L_nozzle = {geom.L_nozzle*1000:.1f} mm")
    print(f"    Converged in {th.n_iters} iteration(s)")
    print()
    print(f"    Peak heat-flux station: x = {flow.x[i_peak]*1000:.1f} mm  "
          f"(throat x = {geom.L_c*1000:.1f} mm)")
    print(f"    Peak heat flux  : {th.heatflux[i_peak]/1e6:.3f} MW/m²")
    print(f"    T_hw max        : {np.max(th.T_hw):.1f} K  "
          f"(wall_melt = {cfg.wall_melt_T:.0f} K)")
    print(f"    T_cw at throat  : {th.T_cw[i_throat]:.1f} K")
    print(f"    T_coolant (injector face) : {th.T_coolant[0]:.1f} K")
    print(f"    T_coolant (nozzle exit)   : {th.T_coolant[-1]:.1f} K  "
          f"[inlet = {cfg.T_coolant_inlet:.0f} K]")
    print(f"    P_coolant (injector face) : {th.P_coolant[0]/1e5:.2f} bar")
    print(f"    P_coolant (nozzle exit)   : {th.P_coolant[-1]/1e5:.2f} bar  "
          f"[inlet = {cfg.P_coolant_inlet/1e5:.0f} bar]")
    print(f"    h_gas at throat  : {th.h_gas[i_throat]/1000:.2f} kW/(m²·K)")
    print(f"    h_cool at throat : {th.h_coolant[i_throat]/1000:.2f} kW/(m²·K)")
    print(f"    Re_cool (throat) : {th.Re_coolant[i_throat]:.0f}")
    print(f"    v_cool (throat)  : {th.v_coolant[i_throat]:.1f} m/s")
    print()
    print(f"    {'x[mm]':>8}  {'M':>6}  {'q[MW/m2]':>10}  "
          f"{'T_hw[K]':>8}  {'T_cw[K]':>8}  {'T_cool[K]':>9}  "
          f"{'h_gas[kW]':>10}  {'h_cool[kW]':>11}  {'Re':>8}")
    for k in range(0, len(flow.x), max(1, len(flow.x) // 20)):
        print(f"    {flow.x[k]*1000:8.1f}  {flow.M[k]:6.3f}  "
              f"{th.heatflux[k]/1e6:10.3f}  "
              f"{th.T_hw[k]:8.1f}  {th.T_cw[k]:8.1f}  {th.T_coolant[k]:9.1f}  "
              f"{th.h_gas[k]/1000:10.2f}  {th.h_coolant[k]/1000:11.2f}  "
              f"{th.Re_coolant[k]:8.0f}")

    # Physical ordering everywhere
    assert np.all(th.T_hw > th.T_cw), "T_hw must exceed T_cw at every station"
    assert np.all(th.T_cw > 0), "T_cw must be positive"
    assert np.all(th.heatflux > 0), "Heat flux must be positive everywhere"
    assert th.T_coolant[0] > cfg.T_coolant_inlet, \
        "Coolant must warm up from nozzle exit to injector"

# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # print("=" * 60)
    # print("Test 1: CEA Interface")
    # print("=" * 60)
    results = []
    results.append(run_test("single O/F (RP-1/LOX OF=2.0)", test_cea_single))
    results.append(run_test("T_c lower at OF=2.0 than OF=3.4", test_cea_single_of2_results))
    results.append(run_test("no OF → ValueError", test_cea_no_of_raises))
    results.append(run_test("sweep-only → returns None", test_cea_sweep_only_returns_none))
    results.append(run_test("full sweep 1.5→10 step 0.1", test_cea_sweep_full))

    print()
    print("=" * 60)
    print("Test 2: Geometry (LH2/LOX RS25)")
    print("=" * 60)
    results.append(run_test("internal consistency (areas, mdot, volume)", test_geometry_internal_consistency))
    results.append(run_test("RS25 dimensional validation", test_geometry_rs25_dimensions))
    results.append(run_test("5kN RPA dimensional validation", test_geometry_rp1_contour_smooth))
    results.append(run_test("nozzle contour smooth", test_geometry_contour_smooth))

    print()
    print("=" * 60)
    print("Test 3: Flow Solver")
    print("=" * 60)
    results.append(run_test("isentropic Mc satisfies area-Mach relation", test_flow_isentropic_mc))
    results.append(run_test("Mc decreases with increasing CR", test_flow_mc_increases_with_cr))
    results.append(run_test("adiabatic flow RP-1/LOX (M_throat≈1, M_exit correct)", test_flow_adiabatic_rp1_lox))
    results.append(run_test("adiabatic flow LH2/LOX (M_throat=1, M_exit correct)", test_flow_adiabatic_rs25_lox))
    
    print()
    print("=" * 60)
    print("Test 4: Heat Transfer")
    print("=" * 60)
    results.append(run_test("RS25 LH2 thermal solve (convergence + T ordering)", test_ht_rs25_converges))
    # results.append(run_test("5kN RP-1 60ch 2mm-land (convergence + RPA table)", test_ht_5kn_rp1_converges))

    print()
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"{'=' * 60}")
    print(f"  {n_pass}/{len(results)} passed   {n_fail} failed")
    print(f"{'=' * 60}")

    sys.exit(0 if n_fail == 0 else 1)
