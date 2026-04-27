"""
channel_designer.py
Edit the control point tables below, then run:  python channel_designer.py

Shows: channel width, height, land width (derived), coolant velocity.
Optionally runs thermal analysis to overlay T_hw.

Control points are interpolated linearly onto the 1mm solver grid.
x values in mm — use any spacing you like (5mm increments work well).
"""

# --- run-from-anywhere shim (file lives in subfolder) ---
import os, sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
os.chdir(_PARENT)
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from config import EngineConfig
from cea_interface import get_cea_for_analysis
from geometry import size_engine, nozzle_radius
from flow_solver import solve_flow
from heat_transfer import ChannelGeometry, solve_thermal
from film_cooling import compute_film_taw
from cea_lookup import build_of_lut
from main import config

# =====================================================================
# EDIT THESE TABLES — x in [mm], dimensions in [mm]
# =====================================================================

# Channel height [mm] at control points along x [mm]
#   x=0 is injector face, x=throat is ~214mm for current 3kN config
HEIGHT = {
    #  x_mm:  h_mm
       0:    1.50,
      50:    1.50,
     100:    1.50,
     150:    1.50,
     200:    1.50,
     215:    1.50,    # throat
     230:    1.50,
     260:    1.50,
     305:    1.50,    # exit
}

# Channel width [mm] at control points along x [mm]
#   Land width is auto-computed: land = circumference/N_channels - width
#   Warnings printed if land < 1.5mm or < 1.2mm (SLM printability)
WIDTH = {
    #  x_mm:  w_mm
       0:    1.30,
      50:    1.30,
     100:    1.30,
     150:    1.30,
     200:    1.20,
     215:    1.20,    # throat — narrow channels, high velocity
     230:    1.20,
     260:    1.30,
     305:    1.30,    # exit
}

# Wall thickness [mm] — constant for now
WALL_T_MM = 0.9

# Set True to run full thermal analysis (adds ~5s)
RUN_THERMAL = True

# =====================================================================
# END EDIT SECTION
# =====================================================================


def build_geometry():
    """Build engine, compute channels from control points, plot results."""
    # Engine setup
    cea = get_cea_for_analysis(config)
    geom = size_engine(config, cea)
    if config.mdot_coolant is None:
        config.mdot_coolant = geom.mdot_fuel

    x_throat_mm = geom.L_c * 1000
    L_total_mm  = (geom.L_c + geom.L_nozzle) * 1000

    # Solver grid
    x_j = np.arange(0, geom.L_c + geom.L_nozzle, config.dx)
    x_mm = x_j * 1000

    # Interpolate height control points onto solver grid
    h_ctrl_x = np.array(sorted(HEIGHT.keys()), dtype=float)
    h_ctrl_y = np.array([HEIGHT[k] for k in sorted(HEIGHT.keys())])
    chan_h = np.interp(x_mm, h_ctrl_x, h_ctrl_y) * 1e-3  # mm → m

    # Interpolate width control points onto solver grid
    w_ctrl_x = np.array(sorted(WIDTH.keys()), dtype=float)
    w_ctrl_y = np.array([WIDTH[k] for k in sorted(WIDTH.keys())])
    chan_w = np.interp(x_mm, w_ctrl_x, w_ctrl_y) * 1e-3  # mm → m

    chan_t = np.full_like(x_j, WALL_T_MM * 1e-3)

    # Bifurcation logic (same as main.py)
    N_throat  = config.N_channels_throat or config.N_channels
    N_chamber = config.N_channels_chamber or N_throat
    split_r   = config.channel_split_r_ratio * geom.R_t
    half_trans = 0.5 * config.channel_split_transition

    r_arr = np.array([nozzle_radius(x, geom, config.dx) for x in x_j])
    split_above = r_arr > split_r
    x_split_up = x_split_down = None
    for i in range(1, len(x_j)):
        if split_above[i] != split_above[i - 1]:
            r0, r1 = r_arr[i - 1], r_arr[i]
            frac = (split_r - r0) / (r1 - r0) if r1 != r0 else 0.0
            xc = x_j[i - 1] + frac * (x_j[i] - x_j[i - 1])
            if xc < geom.L_c and x_split_up is None:
                x_split_up = xc
            elif xc > geom.L_c:
                x_split_down = xc

    def n_local(x):
        if x_split_up is not None and x < x_split_up - half_trans:
            return float(N_chamber)
        if x_split_up is not None and x < x_split_up + half_trans:
            f = (x - (x_split_up - half_trans)) / max(2 * half_trans, 1e-12)
            return float(N_chamber + (N_throat - N_chamber) * f)
        if x_split_down is None or x < x_split_down - half_trans:
            return float(N_throat)
        if x < x_split_down + half_trans:
            f = (x - (x_split_down - half_trans)) / max(2 * half_trans, 1e-12)
            return float(N_throat + (N_chamber - N_throat) * f)
        return float(N_chamber)

    n_chan = np.array([n_local(x) for x in x_j])

    # Derive land width: land = circumference/N - width
    chan_land = np.zeros_like(x_j)
    for i in range(len(x_j)):
        circ = 2 * np.pi * r_arr[i]
        chan_land[i] = circ / n_chan[i] - chan_w[i]

    # Derived quantities
    A_flow = chan_w * chan_h
    mdot_per_ch = config.mdot_coolant / n_chan
    rho_cool = 750.0  # kg/m³ approx RP-1
    v_cool = mdot_per_ch / (rho_cool * A_flow)

    # ── Land width warnings ─────────────────────────────────────────
    land_mm = chan_land * 1000
    has_warning = False
    for xc in sorted(WIDTH.keys()):
        j = int(np.argmin(np.abs(x_mm - xc)))
        lw = land_mm[j]
        if lw < 1.2:
            print(f"  *** CRITICAL: land = {lw:.2f} mm at x = {xc} mm "
                  f"-- BELOW 1.2 mm SLM minimum ***")
            has_warning = True
        elif lw < 1.5:
            print(f"  * WARNING:  land = {lw:.2f} mm at x = {xc} mm "
                  f"-- below 1.5 mm (tight for SLM)")
            has_warning = True

    # Check every station (not just control points)
    land_min_idx = int(np.argmin(chan_land))
    land_min = land_mm[land_min_idx]
    if land_min < 1.2:
        print(f"  *** CRITICAL: minimum land = {land_min:.2f} mm "
              f"at x = {x_mm[land_min_idx]:.0f} mm ***")
        has_warning = True
    elif land_min < 1.5:
        print(f"  * WARNING:  minimum land = {land_min:.2f} mm "
              f"at x = {x_mm[land_min_idx]:.0f} mm")
        has_warning = True
    if not has_warning:
        print(f"  Land OK -- minimum = {land_min:.2f} mm "
              f"at x = {x_mm[land_min_idx]:.0f} mm")

    # Print table at control points
    all_ctrl_x = sorted(set(list(HEIGHT.keys()) + list(WIDTH.keys())))
    print(f"\n{'x[mm]':>7} {'N':>5} {'w[mm]':>7} {'h[mm]':>7} "
          f"{'land[mm]':>9} {'v[m/s]':>8} {'A[mm²]':>8}")
    print("-" * 62)
    for xc in all_ctrl_x:
        j = int(np.argmin(np.abs(x_mm - xc)))
        flag = ""
        lw = land_mm[j]
        if lw < 1.2:
            flag = "  *** <1.2"
        elif lw < 1.5:
            flag = "  *   <1.5"
        print(f"{xc:7.0f} {n_chan[j]:5.0f} {chan_w[j]*1000:7.3f} "
              f"{chan_h[j]*1000:7.3f} {chan_land[j]*1000:9.3f} "
              f"{v_cool[j]:8.2f} {A_flow[j]*1e6:8.4f}{flag}")
    print(f"\n  Throat at x = {x_throat_mm:.1f} mm")
    print(f"  Peak velocity = {v_cool.max():.1f} m/s "
          f"at x = {x_mm[np.argmax(v_cool)]:.0f} mm")

    # Build ChannelGeometry for thermal solver
    chan_geom = ChannelGeometry(x_j, chan_w, chan_h, chan_t, chan_land, n_chan=n_chan)

    # ── Optional thermal run ────────────────────────────────────────
    thermal = None
    if RUN_THERMAL:
        flow = solve_flow(geom, cea, config, xf=x_j[-1])
        T_aw_eff = OF_eff = phase_code = cea_per_station = None
        if config.film_fraction > 0.0:
            T_aw_eff, OF_eff, phase_code = compute_film_taw(flow, geom, cea, config)
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
        thermal = solve_thermal(
            flow, geom, cea, chan_geom, config,
            T_aw_eff=T_aw_eff, cea_per_station=cea_per_station,
            phase_code=phase_code)

    # ── Plot ────────────────────────────────────────────────────────
    n_panels = 5 if thermal else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.4 * n_panels),
                              sharex=True)
    fig.subplots_adjust(hspace=0.08, top=0.95, bottom=0.07)

    for ax in axes:
        ax.axvline(x_throat_mm, color='grey', ls='--', lw=0.8, alpha=0.5)

    # Panel: Nozzle contour
    ax = axes[0]
    ax.fill_between(x_mm, r_arr * 1000, -r_arr * 1000,
                     color='#ddd', alpha=0.5)
    ax.plot(x_mm, r_arr * 1000, 'k-', lw=1)
    ax.plot(x_mm, -r_arr * 1000, 'k-', lw=1)
    ax.set_ylabel('Radius [mm]')
    ax.set_title('Channel Designer -- edit HEIGHT / WIDTH tables and re-run')

    # Panel: Channel width + height + land
    ax = axes[1]
    ax.plot(x_mm, chan_w * 1000, 'c-', lw=1.5, label='Channel width')
    ax.plot(x_mm, chan_h * 1000, 'g-', lw=1.5, label='Channel height')
    ax.plot(x_mm, chan_land * 1000, 'm-', lw=1, alpha=0.7, label='Land width')
    # Land warning zones
    ax.axhline(1.5, color='m', ls=':', lw=0.8, alpha=0.4)
    ax.axhline(1.2, color='r', ls=':', lw=0.8, alpha=0.4)
    # Mark land violations
    land_warn = land_mm < 1.5
    land_crit = land_mm < 1.2
    if np.any(land_crit):
        ax.fill_between(x_mm, 0, chan_land * 1000,
                         where=land_crit, color='red', alpha=0.15)
    if np.any(land_warn & ~land_crit):
        ax.fill_between(x_mm, 0, chan_land * 1000,
                         where=(land_warn & ~land_crit), color='orange', alpha=0.1)
    # Control point markers
    for xc in HEIGHT:
        j = int(np.argmin(np.abs(x_mm - xc)))
        ax.plot(xc, chan_h[j] * 1000, 'go', ms=7, zorder=5)
    for xc in WIDTH:
        j = int(np.argmin(np.abs(x_mm - xc)))
        ax.plot(xc, chan_w[j] * 1000, 'c^', ms=7, zorder=5)
    ax.set_ylabel('Dimension [mm]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel: N channels
    ax = axes[2]
    ax.plot(x_mm, n_chan, 'k-', lw=1.5)
    ax.set_ylabel('N channels')
    ax.grid(True, alpha=0.3)

    # Panel: Coolant velocity
    ax = axes[3]
    ax.plot(x_mm, v_cool, 'b-', lw=1.5)
    ax.set_ylabel('v_coolant [m/s]')
    ax.set_xlabel('Axial position [mm]')
    ax.grid(True, alpha=0.3)

    # Panel: Thermal (if run)
    if thermal:
        ax = axes[4]
        t_mm = thermal.x * 1000
        ax.plot(t_mm, thermal.T_hw, 'r-', lw=1.5, label='T_hw')
        ax.axhline(config.wall_melt_T, color='r', ls=':', lw=0.8,
                    label=f'T_melt = {config.wall_melt_T:.0f} K')
        ax.set_ylabel('T_hw [K]')
        ax.set_xlabel('Axial position [mm]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # q_gas on twin axis
        ax2 = ax.twinx()
        ax2.plot(t_mm, thermal.heatflux / 1e6, 'b-', lw=1, alpha=0.5)
        ax2.set_ylabel('q_gas [MW/m²]', color='b')

    plt.show()


if __name__ == "__main__":
    build_geometry()
