"""
main.py
Top-level entry point. Edit the EngineConfig below to define your engine,
then run:  python main.py
"""

import numpy as np
import matplotlib.pyplot as plt

from config import EngineConfig
from cea_interface import get_cea_for_analysis
from geometry import (size_engine, plot_contour, nozzle_radius,
                      export_csv, export_dxf, _segment_breakpoints)
from flow_solver import solve_flow
from heat_transfer import ChannelGeometry, solve_thermal, plot_thermal
from film_cooling import compute_film_taw
from cea_lookup import build_of_lut
from dataclasses import replace


# =============================================================================
# ENGINE CONFIG — edit this section
# =============================================================================
config = EngineConfig(
    # Propellants
    fuel        = "RP-1",
    oxidizer    = "LOX",
    coolant     = "RP1",

    # Performance — matches RPA CuCrZr_2.5kNEnging.txt
    P_c   = 20.0e5,             # 20 bar
    F_vac = 3074.0,             # 2.5 kN SL (3.074 kN vac)

    # O/F
    OF        = 2.0,
    OF_sweep  = None,

    # CEA
    frozen = False,

    # Geometry — matches RPA
    exp_ratio   = 6.0,          # Ae/At (RPA)
    cont_ratio  = 8.0,          # Ac/At (RPA)
    L_star      = 1.27,         # 1270 mm (RPA)

    # Nozzle contour
    theta1  = 30.0,
    thetaD  = 30.0,
    thetaE  = 12.0,
    R_chamber_mult     = 1.5,
    R_throat_conv_mult = 1.5,
    R_throat_div_mult  = 0.382,

    # Wall material — Copper (CuCrZr) — operational limit, not melting
    wall_k         = 300.0,     # [W/m·K]
    wall_roughness = 6.3e-6,    # SLM
    wall_melt_T    = 1073.0,    # [K] CuCrZr strength limit (800°C, Waugh 2021)

    # Coolant inlet (counter-current from nozzle exit)
    T_coolant_inlet = 298.0,    # [K]
    P_coolant_inlet = 35.0e5,   # 35 bar [Pa]
    mdot_coolant    = None,

    # Channels — bifurcating: 53 at throat, 106 in chamber (RPA)
    N_channels             = 53,
    N_channels_throat      = 53,
    N_channels_chamber     = 106,
    channel_split_r_ratio  = 2.0,
    channel_split_transition = 10e-3,
    dx                     = 1e-3,

    # Tapered channel height — matches RPA 3-section design
    chan_h_throat  = 1.1e-3,    # Sec2 throat
    chan_h_chamber = 1.3e-3,    # Sec1 chamber
    chan_h_exit    = 1.5e-3,    # Sec3 exit

    # Film cooling — 5% (matches RPA film slot at injector face)
    film_fraction  = 0.05,
    film_inject_x  = 0.0,
    film_coolant   = "RP1",
    film_T_inlet   = 298.0,
    film_Kt        = 0.0013,
    film_model     = "simon",

    wall_2d=True,
    use_integral_bl=False,
    C_bartz=0.026,
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
#     R_chamber_mult=0.3196,     # hardware, dimensionless (chamber-side arc)
#     R_throat_conv_mult=0.9469, # R_U/R_t = 5.1527/5.4416
#     R_throat_div_mult=0.3711,  # R_D/R_t = 2.019/5.4416
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
#     R_chamber_mult=0.3196,
#     R_throat_conv_mult=0.9469,
#     R_throat_div_mult=0.3711,
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

    # Derive coolant mass flow if not set (fuel minus film)
    if config.mdot_coolant is None:
        config.mdot_coolant = geom.mdot_fuel * (1.0 - config.film_fraction)
        print(f"\nDerived mdot_coolant = mdot_fuel × (1 - film) = "
              f"{config.mdot_coolant:.4f} kg/s  "
              f"(film = {config.film_fraction*100:.0f}%)")

    # Define Cooling Channel Geometry
    x_j = np.arange(0, geom.L_c + geom.L_nozzle, config.dx) # Create slices at each dx
    CHAN_W = 1.0e-3  # Constant channel width [m] — matches RPA throat (Sec2)
    chan_t = np.full(shape=(len(x_j),), fill_value=0.9e-3) # 0.9mm wall thickness
    chan_w = np.full(shape=(len(x_j),), fill_value=CHAN_W)  # 1.5mm constant width
    chan_land = np.zeros(len(x_j)) # land = circ/N - width (computed below)
    chan_h = np.zeros(len(x_j))
    n_chan_per_station = np.zeros(len(x_j), dtype=int)

    # Bifurcation setup: throat-region channel count, splitting to chamber count
    # wherever the local wall radius exceeds split_r_ratio · R_t
    N_throat  = config.N_channels_throat  or config.N_channels
    N_chamber = config.N_channels_chamber or N_throat
    split_r   = config.channel_split_r_ratio * geom.R_t
    x_throat  = geom.L_c   # throat axial location

    # Find the two axial locations where r(x) = split_r — one upstream of the
    # throat (chamber → throat) and one downstream (throat → exit).  Across a
    # configurable transition band the channel count is ramped linearly to
    # mimic a real Y-cusp split rather than an instantaneous jump.
    r_arr_for_split = np.array([nozzle_radius(x, geom, config.dx) for x in x_j])
    split_above = r_arr_for_split > split_r
    x_split_up   = None
    x_split_down = None
    for i in range(1, len(x_j)):
        if split_above[i] != split_above[i - 1]:
            # linear interp for crossing position
            r0, r1 = r_arr_for_split[i - 1], r_arr_for_split[i]
            frac = (split_r - r0) / (r1 - r0) if r1 != r0 else 0.0
            x_cross = x_j[i - 1] + frac * (x_j[i] - x_j[i - 1])
            if x_cross < x_throat and x_split_up is None:
                x_split_up = x_cross
            # Nozzle side intentionally left unsplit — keep N_throat channels
            # all the way from the throat to the exit (simpler SLM geometry,
            # no Y-merge in the diverging section).

    half_trans = 0.5 * config.channel_split_transition

    def n_local_at(x: float) -> float:
        """Effective (possibly non-integer) channel count at axial station x.
        Equals N_throat between the two crossings, N_chamber outside, and
        ramps linearly across a band of width channel_split_transition
        centered on each crossing."""
        if x_split_up is not None and x < x_split_up - half_trans:
            return float(N_chamber)
        if x_split_up is not None and x < x_split_up + half_trans:
            f = (x - (x_split_up - half_trans)) / max(2.0 * half_trans, 1e-12)
            return float(N_chamber + (N_throat - N_chamber) * f)
        if x_split_down is None or x < x_split_down - half_trans:
            return float(N_throat)
        if x < x_split_down + half_trans:
            f = (x - (x_split_down - half_trans)) / max(2.0 * half_trans, 1e-12)
            return float(N_throat + (N_chamber - N_throat) * f)
        return float(N_chamber)

    # Tapered channel height (linear interp chamber → throat → exit).
    # Falls back to a constant 2 mm if any taper value is unset.
    if (config.chan_h_throat is not None
            and config.chan_h_chamber is not None
            and config.chan_h_exit is not None):
        x_taper = np.array([0.0, x_throat, geom.L_c + geom.L_nozzle])
        h_taper = np.array([config.chan_h_chamber,
                            config.chan_h_throat,
                            config.chan_h_exit])
        chan_h[:] = np.interp(x_j, x_taper, h_taper)
    else:
        chan_h[:] = 2e-3

    # Calculate land at each slice: land = pitch - width
    n_chan_float = np.zeros(len(x_j))
    for i, x in enumerate(x_j):
        r = nozzle_radius(x, geom, config.dx)
        N_local = n_local_at(x)
        n_chan_float[i] = N_local
        n_chan_per_station[i] = int(round(N_local))
        circ = 2*np.pi*r
        chan_land[i] = circ/N_local - CHAN_W

    # Report bifurcation
    if N_chamber != N_throat:
        print(f"\n--- Bifurcating channels ---")
        print(f"  N_throat  = {N_throat}, N_chamber = {N_chamber}")
        print(f"  Split radius = {split_r*1000:.2f} mm  (= {config.channel_split_r_ratio:.2f} · R_t)")
        if x_split_up is not None:
            print(f"  Upstream split   x ≈ {x_split_up*1000:.1f} mm")
        if x_split_down is not None:
            print(f"  Downstream split x ≈ {x_split_down*1000:.1f} mm")
        print(f"  Y-cusp transition length: {config.channel_split_transition*1000:.1f} mm")
        print(f"  Channel width range:  {chan_w.min()*1000:.3f} – {chan_w.max()*1000:.3f} mm")
        print(f"  Channel height range: {chan_h.min()*1000:.3f} – {chan_h.max()*1000:.3f} mm")

    # ------------------------------------------------------------------
    # Segment summary for RPA entry — dimensions at each transition point
    # ------------------------------------------------------------------
    def _dims_at(x: float):
        """Return (N, w, h, land, t, r) at axial station x [m]."""
        N = n_local_at(x)
        r = nozzle_radius(x, geom, config.dx)
        h = float(np.interp(x, x_j, chan_h))
        ld = float(np.interp(x, x_j, chan_land))
        t  = float(np.interp(x, x_j, chan_t))
        w = (2.0 * np.pi * r / N) - ld
        return N, w, h, ld, t, r

    # Build segment boundary list
    L_total = geom.L_c + geom.L_nozzle
    boundaries = [("Injector face", 0.0)]
    if x_split_up is not None:
        boundaries.append(("Upstream split START", x_split_up - half_trans))
        boundaries.append(("Upstream split END",   x_split_up + half_trans))
    boundaries.append(("Throat", x_throat))
    if x_split_down is not None:
        boundaries.append(("Downstream split START", x_split_down - half_trans))
        boundaries.append(("Downstream split END",   x_split_down + half_trans))
    boundaries.append(("Nozzle exit", L_total))
    # Filter out any boundaries that fall outside the engine
    boundaries = [(lbl, max(0.0, min(x, L_total))) for lbl, x in boundaries]

    print(f"\n--- Segment dimensions for RPA entry ---")
    print(f"  {'Station':<26} {'x[mm]':>7} {'r[mm]':>7} {'N':>6} "
          f"{'w[mm]':>7} {'h[mm]':>7} {'land[mm]':>9} {'t_w[mm]':>8}")
    print("  " + "-"*82)
    for lbl, x in boundaries:
        N, w, h, ld, t, r = _dims_at(x)
        print(f"  {lbl:<26} {x*1000:7.1f} {r*1000:7.2f} {N:6.1f} "
              f"{w*1000:7.3f} {h*1000:7.3f} {ld*1000:9.3f} {t*1000:8.3f}")
    print()
    print(f"  Wall thickness (chan_t): {chan_t.min()*1000:.3f} mm (constant)")
    print(f"  Engine total length: {L_total*1000:.1f} mm  "
          f"(L_c={geom.L_c*1000:.1f} mm, L_nozzle={geom.L_nozzle*1000:.1f} mm)")
   
    
    # Display channel geometry just like in RPA
    # for i, x in enumerate(x_j):
    #     print(f"x: {x:.4f} [m] hc: {chan_h[i]*1000:.4f} [mm] a: {chan_w[i]*1000:.4f} b: {chan_land[i]*1000:.4f} [mm]")
 
    chan_geom = ChannelGeometry(x_j, chan_w, chan_h, chan_t, chan_land,
                                n_chan=n_chan_float)

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

    plot_thermal(thermal, geom, config)

    plt.show()  # Keep all plot windows open


if __name__ == "__main__":
    run()
