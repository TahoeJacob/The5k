"""
main.py
Top-level entry point.

Usage:
    python main.py                                 # default config
    python main.py configs/launcher_e1.toml       # any TOML in configs/

Add new engines as TOML files in configs/ — see config_loader.py for the
schema. The default config (configs/the5k_2p5kn_cucrzr.toml) is the 2.5 kN
RP-1/LOX CuCrZr build that matches RPA CuCrZr_2.5kNEnging.txt.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cea_interface import get_cea_for_analysis
from geometry import (size_engine, plot_contour,
                      export_csv, export_dxf, _segment_breakpoints)
from flow_solver import solve_flow
from heat_transfer import solve_thermal, plot_thermal
from film_cooling import compute_film_taw
from cea_lookup import build_of_lut
from channel_builder import build_channel_geometry, report_channels
from config_loader import load_config
from dataclasses import replace


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "the5k_2p5kn_cucrzr.toml"


def _resolve_config_path(argv: list[str]) -> Path:
    """Pick a config file: CLI arg if given, else DEFAULT_CONFIG."""
    if len(argv) > 1:
        p = Path(argv[1])
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p
    return DEFAULT_CONFIG


config = load_config(_resolve_config_path(sys.argv))
print(f"Loaded engine config: {_resolve_config_path(sys.argv)}")


def run(plot=True):
    # --- Step 1: CEA ---
    cea_result = get_cea_for_analysis(config)
    if cea_result is None:
        # Only a sweep was requested — plots shown, nothing further to do
        return None

    # --- Step 2: Engine geometry ---
    geom = size_engine(config, cea_result)
    if plot:
        plot_contour(geom, dx=5e-4)

    # Derive coolant mass flow if not set
    if config.mdot_coolant is None:
        if config.coolant == "Oxygen":
            config.mdot_coolant = geom.mdot_ox
            print(f"\nDerived mdot_coolant = mdot_ox = "
                  f"{config.mdot_coolant:.4f} kg/s  (LOX regen)")
        else:
            config.mdot_coolant = geom.mdot_fuel * (1.0 - config.film_fraction)
            print(f"\nDerived mdot_coolant = mdot_fuel × (1 - film) = "
                  f"{config.mdot_coolant:.4f} kg/s  "
                  f"(film = {config.film_fraction*100:.0f}%)")

    # Build the cooling channel geometry from config.channels (delegated to
    # channel_builder so validation scripts use the identical code path).
    chan_geom, info = build_channel_geometry(config, geom)
    report_channels(chan_geom, info, geom)

    N_throat     = info["N_throat"]
    x_split_up   = info["x_split_up"]
    x_split_down = info["x_split_down"]
    half_trans   = info["half_trans"]

    # --- Export geometry for OnShape (throat-centered, x=0 at throat) ---
    # Build key station list for channel_dimensions.csv
    L_total = geom.L_c + geom.L_nozzle

    # Pull contour breakpoints (in injector-face coords) so we can mark the
    # chamber-arc start (= end of straight chamber section).
    _segs = _segment_breakpoints(geom)
    # _segs order: f1 straight, f2 chamber arc, f3 linear taper,
    #              f4 throat conv fillet, f5 throat div fillet, f6 Bezier bell
    f2 = _segs[1]   # chamber arc R_chamber
    f4 = _segs[3]   # throat conv fillet (ends at throat)
    f5 = _segs[4]   # throat div fillet  (starts at throat)
    f6 = _segs[5]   # Bezier bell

    key_stations = [("Injector face", 0.0)]

    # Chamber arc R_chamber: start / mid / end
    key_stations.append(("Chamber arc START", f2[1]))
    key_stations.append(("Chamber arc MID",   0.5 * (f2[1] + f2[2])))
    key_stations.append(("Chamber arc END",   f2[2]))

    # Throat-conv fillet: start / mid / end (end == throat, added later)
    key_stations.append(("Throat conv fillet START", f4[1]))
    key_stations.append(("Throat conv fillet MID",   0.5 * (f4[1] + f4[2])))

    if x_split_up is not None:
        key_stations.append(("Upstream split START", x_split_up - half_trans))
        key_stations.append(("Upstream split MID",   x_split_up))
        key_stations.append(("Upstream split END",   x_split_up + half_trans))

    key_stations.append(("Throat", geom.L_c))

    # Throat-div fillet: start (== throat) / mid / end
    key_stations.append(("Throat div fillet MID", 0.5 * (f5[1] + f5[2])))
    key_stations.append(("Throat div fillet END", f5[2]))

    if x_split_down is not None:
        key_stations.append(("Downstream split START", x_split_down - half_trans))
        key_stations.append(("Downstream split MID",   x_split_down))
        key_stations.append(("Downstream split END",   x_split_down + half_trans))

    # Bezier bell: every 5 mm from start to exit (skip endpoints to avoid
    # duplicating the throat-div END and Nozzle exit entries)
    bell_start, bell_end = f6[1], f6[2]
    bell_dx = 5e-3
    n_bell = int(np.floor((bell_end - bell_start) / bell_dx))
    for k in range(1, n_bell + 1):
        x_bell = bell_start + k * bell_dx
        if x_bell >= bell_end - 1e-6:
            break
        key_stations.append((f"Bell {k*5} mm", x_bell))

    key_stations.append(("Nozzle exit", L_total))
    # Clamp to valid range and sort by axial position
    key_stations = [(lbl, max(0.0, min(x, L_total))) for lbl, x in key_stations]
    key_stations.sort(key=lambda kv: kv[1])

    export_csv(geom, chan_geom, out_dir='exports',
               key_stations=key_stations, key_stations_N_throat=N_throat)
    export_dxf(geom, chan_geom, out_dir='exports')

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
    T_aw_eff, OF_eff, phase_code = compute_film_taw(flow, geom, cea_result, config)

    # --- Step 4b: Surface-layer CEA lookup for Bartz h_gas ---
    # RPA evaluates Bartz on the fuel-rich surface layer only AFTER the film
    # has vaporised and mixed into the gas (phase_code == 3).  In the liquid
    # (1) and vapour (2) phases the film is a wall-adhered layer and the gas
    # surface layer is unchanged from the unfilmed core — use frozen core CEA.
    cea_per_station = None
    if config.film_fraction > 0.0:
        of_lut = build_of_lut(config)
        cea_per_station = []
        for i, pc in enumerate(phase_code):
            if pc == 3:
                # Gaseous mixing — equilibrium CEA at shifted OF_eff
                props = of_lut.at(float(OF_eff[i]))
                cea_per_station.append(replace(
                    cea_result,
                    T_c       = props["T_c"],
                    visc_c    = props["visc_c"],
                    Cp_froz_c = props["Cp_c"],    # equilibrium
                    Pr_froz_c = props["Pr_c"],    # equilibrium
                    gamma_c   = props["gamma_c"],
                    C_star    = props["C_star"]))
            else:
                # No film, liquid, or vapour — unchanged core (frozen)
                cea_per_station.append(cea_result)

    # --- Step 5: Thermal analysis ---
    thermal = solve_thermal(flow, geom, cea_result, chan_geom, config,
                            T_aw_eff=T_aw_eff if config.film_fraction > 0.0 else None,
                            cea_per_station=cea_per_station,
                            phase_code=phase_code if config.film_fraction > 0.0 else None)

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
            T_aw_eff, OF_eff, phase_code = compute_film_taw(flow, geom, cea_result, config)
            cea_per_station = []
            for i, pc in enumerate(phase_code):
                if pc == 3:
                    props = of_lut.at(float(OF_eff[i]))
                    cea_per_station.append(replace(
                        cea_result,
                        T_c       = props["T_c"],
                        visc_c    = props["visc_c"],
                        Cp_froz_c = props["Cp_c"],
                        Pr_froz_c = props["Pr_c"],
                        gamma_c   = props["gamma_c"],
                        C_star    = props["C_star"]))
                else:
                    cea_per_station.append(cea_result)
            thermal = solve_thermal(flow, geom, cea_result, chan_geom, config,
                                    T_aw_eff=T_aw_eff,
                                    cea_per_station=cea_per_station,
                                    phase_code=phase_code)

    if plot:
        plot_thermal(thermal, geom, config)

    return {
        "config":    config,
        "cea":       cea_result,
        "geom":      geom,
        "chan_geom": chan_geom,
        "flow":      flow,
        "thermal":   thermal,
    }


if __name__ == "__main__":
    run()
    plt.show()  # Keep all plot windows open
