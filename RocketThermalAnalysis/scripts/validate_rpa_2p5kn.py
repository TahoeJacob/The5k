"""
validate_rpa_2p5kn.py
Compare our code against RPA 2.5kN results for both 0% and 10% film cooling.

RPA parameters (from user):
  F_vac = 2.97kN, F_SL = 2.5kN, P_c = 20 bar, OF = 2.0
  Dc = 84.3mm, Dt = 34.42mm, Lcyl = 123.05mm, Lc = 194.65mm, Le = 64.20mm
  Ae/At = 5.0, Ac/At = 6.0, L* = 1000mm
  Wall: k=16 W/mK, t=0.9mm
  Channels: 1.5mm width, 1.5mm height, constant
  N=60 (0-130mm), N=30 (130-240mm), N=60 (240-end)
  Coolant inlet: ~298K at nozzle exit, P=3.5 MPa
"""

# --- run-from-anywhere shim (file lives in subfolder) ---
import os, sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
os.chdir(_PARENT)
# --------------------------------------------------------

from pathlib import Path

import numpy as np
from dataclasses import replace

from cea_interface import get_cea_for_analysis
from geometry import size_engine
from flow_solver import solve_flow
from heat_transfer import solve_thermal
from film_cooling import compute_film_taw
from cea_lookup import build_of_lut
from channel_builder import build_channel_geometry
from config_loader import load_config


# Config matching RPA exactly — see configs/rpa_2p5kn_validation.toml
RPA_CONFIG_PATH = Path(_PARENT) / "configs" / "rpa_2p5kn_validation.toml"


def parse_rpa_file(path):
    """Parse RPA thermal results into arrays."""
    import re
    x, r, h_gas, q_conv, Twg, Tc, N_ch = [], [], [], [], [], [], []
    with open(path) as f:
        lines = f.readlines()
    in_data = False
    for line in lines:
        if line.strip().startswith("Location, mm"):
            in_data = True
            continue
        if not in_data:
            continue
        parts = line.strip().split('\t')
        if len(parts) < 14:
            continue
        try:
            x.append(float(parts[0]))
            r.append(float(parts[1]))
            h_gas.append(float(parts[2]))
            q_conv.append(float(parts[3]))
            Twg.append(float(parts[6]))
            Tc.append(float(parts[9]))
            N_ch.append(float(parts[13]))
        except (ValueError, IndexError):
            continue
    return {
        'x': np.array(x), 'r': np.array(r),
        'h_gas': np.array(h_gas), 'q_conv': np.array(q_conv),
        'Twg': np.array(Twg), 'Tc': np.array(Tc),
        'N': np.array(N_ch),
    }


def run_comparison(film_frac, rpa_file, label):
    """Run our code and compare with RPA."""
    config = load_config(RPA_CONFIG_PATH)
    config.film_fraction = film_frac
    config.mdot_coolant = None  # auto-compute

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Film fraction = {film_frac*100:.0f}%")
    print(f"{'='*70}")

    # Engine setup
    cea = get_cea_for_analysis(config)
    geom = size_engine(config, cea)

    # Set coolant mass flow = fuel minus film
    mdot_fuel = geom.mdot_fuel
    config.mdot_coolant = mdot_fuel * (1.0 - film_frac)
    print(f"  mdot_fuel = {mdot_fuel*1000:.1f} g/s, "
          f"mdot_coolant = {config.mdot_coolant*1000:.1f} g/s")

    # Channel geometry comes from configs/rpa_2p5kn_validation.toml
    # (1.5 mm × 1.5 mm constant, 0.9 mm hot-wall, 30→60 bifurcation).
    chan_geom, _ = build_channel_geometry(config, geom)

    # Flow solution
    flow = solve_flow(geom, cea, config, xf=chan_geom.x_j[-1])

    # Film cooling
    T_aw_eff = OF_eff = phase_code = cea_per_station = None
    if film_frac > 0:
        T_aw_eff, OF_eff, phase_code = compute_film_taw(
            flow, geom, cea, config)
        of_lut = build_of_lut(config, verbose=False)
        cea_per_station = []
        for i, pc in enumerate(phase_code):
            if pc == 3:
                props = of_lut.at(float(OF_eff[i]))
                cea_per_station.append(replace(
                    cea, T_c=props["T_c"], visc_c=props["visc_c"],
                    Cp_froz_c=props["Cp_c"], Pr_froz_c=props["Pr_c"],
                    gamma_c=props["gamma_c"], C_star=props["C_star"]))
            else:
                cea_per_station.append(cea)

    # Thermal solve
    thermal = solve_thermal(
        flow, geom, cea, chan_geom, config,
        T_aw_eff=T_aw_eff if film_frac > 0 else None,
        cea_per_station=cea_per_station,
        phase_code=phase_code if film_frac > 0 else None)

    # Parse RPA results
    rpa = parse_rpa_file(rpa_file)

    # Compare at key stations
    t_mm = thermal.x * 1000
    print(f"\n  Station-by-station comparison (our code vs RPA):")
    print(f"  {'x[mm]':>7} {'T_hw_us':>8} {'T_hw_RPA':>9} {'err':>6} "
          f"{'h_g_us':>8} {'h_g_RPA':>8} {'err':>6} "
          f"{'q_us':>8} {'q_RPA':>8} {'err':>6}")
    print(f"  {'-'*90}")

    # Sample at RPA stations
    for ri in range(len(rpa['x'])):
        rx = rpa['x'][ri]
        # Find nearest our station
        j = int(np.argmin(np.abs(t_mm - rx)))
        if abs(t_mm[j] - rx) > 2.0:
            continue

        our_Thw = thermal.T_hw[j]
        rpa_Thw = rpa['Twg'][ri]
        our_hg  = thermal.h_gas[j] / 1000  # W/m²K → kW/m²K
        rpa_hg  = rpa['h_gas'][ri]
        our_q   = thermal.heatflux[j] / 1000  # W/m² → kW/m²
        rpa_q   = rpa['q_conv'][ri]

        err_T = (our_Thw - rpa_Thw) / rpa_Thw * 100 if rpa_Thw > 0 else 0
        err_h = (our_hg - rpa_hg) / rpa_hg * 100 if rpa_hg > 0 else 0
        err_q = (our_q - rpa_q) / rpa_q * 100 if rpa_q > 0 else 0

        marker = ""
        if abs(t_mm[j] - geom.L_c * 1000) < 3:
            marker = " <-- THROAT"

        print(f"  {rx:7.1f} {our_Thw:8.0f} {rpa_Thw:9.0f} {err_T:+5.0f}% "
              f"{our_hg:8.2f} {rpa_hg:8.2f} {err_h:+5.0f}% "
              f"{our_q:8.0f} {rpa_q:8.0f} {err_q:+5.0f}%{marker}")

    # Summary
    peak_us  = float(thermal.T_hw.max())
    peak_rpa = float(rpa['Twg'].max())
    pk_x_us  = float(t_mm[np.argmax(thermal.T_hw)])
    pk_x_rpa = float(rpa['x'][np.argmax(rpa['Twg'])])
    dT_us    = float(thermal.T_coolant.max() - thermal.T_coolant.min())
    dT_rpa   = float(rpa['Tc'].max() - rpa['Tc'].min())

    print(f"\n  SUMMARY:")
    print(f"    Peak T_hw:  Ours = {peak_us:.0f} K @ x={pk_x_us:.0f}mm  |  "
          f"RPA = {peak_rpa:.0f} K @ x={pk_x_rpa:.0f}mm  |  "
          f"err = {(peak_us-peak_rpa)/peak_rpa*100:+.1f}%")
    print(f"    ΔT coolant: Ours = {dT_us:.0f} K  |  RPA = {dT_rpa:.0f} K")

    if film_frac > 0 and T_aw_eff is not None:
        # Show film T_aw_eff at a few stations
        # Compute T_aw from flow (recovery temperature)
        gamma = cea.gamma_c
        r_factor = 0.9  # typical Pr^0.33
        T_aw_bare = flow.T * (1 + r_factor * (gamma - 1) / 2 * flow.M**2)
        print(f"\n  Film T_aw_eff at key stations:")
        for xc in [0, 10, 30, 50, 80, 100, 130, 150, 170, 190, 194]:
            j = int(np.argmin(np.abs(t_mm - xc)))
            pc = phase_code[j] if phase_code is not None else 0
            phase_name = {0: "none", 1: "liquid", 2: "vapour", 3: "gaseous"}
            reduction = T_aw_bare[j] - T_aw_eff[j]
            print(f"    x={xc:3d}mm: T_aw_eff={T_aw_eff[j]:.0f}K  "
                  f"T_aw_bare={T_aw_bare[j]:.0f}K  "
                  f"reduction={reduction:.0f}K  "
                  f"phase={phase_name.get(pc, '?')}")


if __name__ == "__main__":
    # Run 0% film first
    run_comparison(0.0,
                   "../References/2.5kNRPA0%FilmCoolingResults.txt",
                   "0% Film Cooling — Baseline")

    # Then 10% film
    run_comparison(0.10,
                   "../References/RPA2.5kN_10%FilmInconelResults.txt",
                   "10% Film Cooling")
