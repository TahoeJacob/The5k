"""
wall_field_diagnostic.py
Re-run the 2-D wall solve at user-selected axial stations and visualize the
full temperature field T(x, y) in the unit cell so you can see what the
channel ceiling, side, and land midpoint temperatures look like.

Run:  python wall_field_diagnostic.py
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

from main import config
from cea_interface import get_cea_for_analysis
from geometry import size_engine, nozzle_radius
from flow_solver import solve_flow
from film_cooling import compute_film_taw
from heat_transfer import (ChannelGeometry, solve_thermal,
                           _build_wall_2d, _make_wall_grid,
                           _bartz_h, _coolant_htc, _T_aw)


def run_field_at(x_query, label, flow, geom, cea, chan_geom,
                 thermal, T_aw_eff_arr, closeout_t=0.0,
                 h_ext=0.0, T_ext=300.0):
    """Re-solve the 2-D wall at axial station x_query and return T_field grid."""
    # Find nearest index in thermal arrays
    i = int(np.argmin(np.abs(thermal.x - x_query)))
    x       = thermal.x[i]
    M       = flow.M[i]
    A       = np.pi * nozzle_radius(x, geom, config.dx)**2
    T_cool  = thermal.T_coolant[i]
    P_cool  = thermal.P_coolant[i]
    h_cool  = thermal.h_coolant[i]
    h_gas   = thermal.h_gas[i]
    T_aw    = T_aw_eff_arr[i] if T_aw_eff_arr is not None else _T_aw(M, cea)

    # Channel geometry at this station
    chan_w, chan_h, chan_t, chan_land = chan_geom.at(x)
    chan_w_half = chan_w / 2.0
    land_half   = chan_land / 2.0

    x_nodes, y_nodes = _make_wall_grid(chan_w_half, chan_t, chan_h, land_half,
                                       closeout_t=closeout_t)

    T_field, node_map, solid_nodes, xn, yn, is_void = _build_wall_2d(
        x_nodes, y_nodes, chan_w_half, chan_t,
        h_gas, T_aw, h_cool, T_cool, config.wall_k,
        chan_h=chan_h, h_ext=h_ext, T_ext=T_ext)

    # Reconstruct full grid (NaN in void)
    T_grid = np.full((len(yn), len(xn)), np.nan)
    for (ii, jj), eq in node_map.items():
        T_grid[jj, ii] = T_field[eq]

    return {
        'label':   label,
        'x':       x,
        'xn':      xn, 'yn': yn,
        'T_grid':  T_grid,
        'chan_w':  chan_w,  'chan_h': chan_h,
        'chan_t':  chan_t,  'chan_land': chan_land,
        'closeout_t': closeout_t,
        'h_gas':   h_gas,   'h_cool': h_cool,
        'T_aw':    T_aw,    'T_cool': T_cool,
        'T_hw':    thermal.T_hw[i],
        'T_cw':    thermal.T_cw[i],
    }


def report(field):
    yn = field['yn']
    xn = field['xn']
    T  = field['T_grid']
    chan_t = field['chan_t']
    chan_h = field['chan_h']
    closeout_t = field['closeout_t']
    chan_w_half = field['chan_w'] / 2.0

    j_ceil  = int(np.argmin(np.abs(yn - (chan_t + chan_h))))   # channel ceiling
    j_outer = len(yn) - 1                                      # outer surface

    # Hot wall row
    T_hw_row = T[0, :]
    # Channel ceiling row (above the channel only)
    ceil_mask = xn < chan_w_half + 1e-12
    T_ceil_row = T[j_ceil, ceil_mask]
    # Land midpoint at ceiling level (above the land)
    land_mask = xn >= chan_w_half - 1e-12
    T_land_ceil = T[j_ceil, land_mask]
    # Channel side wall column
    i_side = int(np.argmin(np.abs(xn - chan_w_half)))
    T_side_col = T[(yn > chan_t) & (yn < chan_t + chan_h), i_side]
    # Outer surface row (top of closeout)
    T_outer_row = T[j_outer, :]

    print(f"\n=== {field['label']} (x = {field['x']*1000:.1f} mm) ===")
    print(f"  Geometry: w={field['chan_w']*1000:.3f} mm  h={chan_h*1000:.3f} mm  "
          f"t_w={chan_t*1000:.3f} mm  land={field['chan_land']*1000:.3f} mm  "
          f"closeout={closeout_t*1000:.2f} mm")
    print(f"  BCs: h_gas={field['h_gas']/1000:.2f} kW/m²K  T_aw={field['T_aw']:.0f} K  "
          f"h_cool={field['h_cool']/1000:.2f} kW/m²K  T_cool={field['T_cool']:.0f} K")
    print(f"  Hot wall (gas-side):     T_max = {np.nanmax(T_hw_row):.1f} K  "
          f"T_min = {np.nanmin(T_hw_row):.1f} K  ΔT = {np.nanmax(T_hw_row)-np.nanmin(T_hw_row):.1f} K")
    print(f"  Channel ceiling (above ch.):  T_max = {np.nanmax(T_ceil_row):.1f} K  "
          f"T_avg = {np.nanmean(T_ceil_row):.1f} K")
    if T_side_col.size:
        print(f"  Channel side wall:       T_max = {np.nanmax(T_side_col):.1f} K  "
              f"T_min = {np.nanmin(T_side_col):.1f} K")
    print(f"  Land midpoint (ceiling): T = {T_land_ceil[-1]:.1f} K")
    print(f"  Outer surface (closeout top): T_max = {np.nanmax(T_outer_row):.1f} K  "
          f"T_min = {np.nanmin(T_outer_row):.1f} K  "
          f"ΔT_azi = {np.nanmax(T_outer_row)-np.nanmin(T_outer_row):.1f} K")
    print(f"  6061-T6 strength gates:  573 K (300°C, ~50% yield)  "
          f"623 K (350°C, ~25% yield)  673 K (400°C, ~zero yield)")


def plot_field(field, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    xn = field['xn'] * 1000  # mm
    yn = field['yn'] * 1000
    T  = field['T_grid']
    pc = ax.pcolormesh(xn, yn, T, shading='auto', cmap='inferno', vmin=300, vmax=900)
    plt.colorbar(pc, ax=ax, label='T [K]')

    # Iso-T contours with labels in K, spanning the actual field range so
    # each slice shows the local distribution clearly.  Step size adapts to
    # the local ΔT so nearly-uniform fields still get ~6 contours.
    import matplotlib.patheffects as pe
    Tmin = float(np.nanmin(T))
    Tmax = float(np.nanmax(T))
    dT = Tmax - Tmin
    if dT > 0.05:
        # Choose a "nice" step that gives ~6 levels across the range
        target_n = 6
        raw_step = dT / target_n
        nice_steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 25, 50, 100]
        step = next((s for s in nice_steps if s >= raw_step), 100)
        lo = step * np.ceil(Tmin / step)
        hi = step * np.floor(Tmax / step)
        levels = np.arange(lo, hi + 0.5*step, step)
        if len(levels) > 0:
            fmt = '%.1f K' if step < 1 else '%.0f K'
            cs_iso = ax.contour(xn, yn, T, levels=levels,
                                colors='white', linewidths=1.0, alpha=0.9)
            labels = ax.clabel(cs_iso, fmt=fmt, fontsize=8, inline_spacing=2)
            for txt in labels:
                txt.set_color('white')
                txt.set_path_effects([pe.withStroke(linewidth=2.0,
                                                    foreground='black')])

    # Critical 6061-T6 strength gates: 300°C / 350°C / 400°C
    gates = [573, 623, 673]
    gate_levels = [g for g in gates if Tmin < g < Tmax]
    if gate_levels:
        cs_gate = ax.contour(xn, yn, T, levels=gate_levels,
                             colors=['cyan', 'lime', 'red'][:len(gate_levels)],
                             linewidths=1.6)
        gate_fmt = {573: '300°C', 623: '350°C', 673: '400°C'}
        ax.clabel(cs_gate, fmt=gate_fmt, fontsize=8)

    # Outline the channel cavity (now bounded above by closeout)
    chan_w_half = field['chan_w']/2 * 1000
    chan_t = field['chan_t'] * 1000
    chan_h = field['chan_h'] * 1000
    ax.plot([0,             chan_w_half,    chan_w_half,        0],
            [chan_t,        chan_t,         chan_t + chan_h,    chan_t + chan_h],
            'w-', linewidth=1.5)

    ax.set_xlabel('x (azimuthal half-cell) [mm]')
    ax.set_ylabel('y (radial: gas → coolant → outer wall) [mm]')
    ax.set_title(f"{field['label']}  (x={field['x']*1000:.0f} mm)\n"
                 f"T_hw={field['T_hw']:.0f} K  T_cw={field['T_cw']:.0f} K")
    ax.set_aspect('equal')
    return ax


def main():
    cea = get_cea_for_analysis(config)
    geom = size_engine(config, cea)

    if config.mdot_coolant is None:
        config.mdot_coolant = geom.mdot_fuel

    # Re-build chan_geom the same way main.py does
    x_j = np.arange(0, geom.L_c + geom.L_nozzle, config.dx)
    chan_t = np.full(len(x_j), 0.9e-3)
    chan_land = np.full(len(x_j), 1.0e-3)

    N_throat  = config.N_channels_throat  or config.N_channels
    N_chamber = config.N_channels_chamber or N_throat
    split_r   = config.channel_split_r_ratio * geom.R_t
    x_throat  = geom.L_c

    r_arr = np.array([nozzle_radius(x, geom, config.dx) for x in x_j])
    above = r_arr > split_r
    x_split_up = x_split_down = None
    for i in range(1, len(x_j)):
        if above[i] != above[i-1]:
            r0, r1 = r_arr[i-1], r_arr[i]
            frac = (split_r - r0) / (r1 - r0) if r1 != r0 else 0.0
            xc = x_j[i-1] + frac * (x_j[i] - x_j[i-1])
            if xc < x_throat and x_split_up is None:
                x_split_up = xc
            elif xc > x_throat:
                x_split_down = xc
    half = 0.5 * config.channel_split_transition

    def n_at(x):
        if x_split_up is not None and x < x_split_up - half:    return float(N_chamber)
        if x_split_up is not None and x < x_split_up + half:
            f = (x - (x_split_up - half)) / (2*half); return N_chamber + (N_throat-N_chamber)*f
        if x_split_down is None or x < x_split_down - half:     return float(N_throat)
        if x < x_split_down + half:
            f = (x - (x_split_down - half)) / (2*half); return N_throat + (N_chamber-N_throat)*f
        return float(N_chamber)

    chan_h = np.interp(x_j, [0.0, x_throat, geom.L_c + geom.L_nozzle],
                       [config.chan_h_chamber, config.chan_h_throat, config.chan_h_exit])
    chan_w = np.zeros(len(x_j))
    n_chan_float = np.zeros(len(x_j))
    for i, x in enumerate(x_j):
        r = nozzle_radius(x, geom, config.dx)
        N = n_at(x)
        n_chan_float[i] = N
        chan_w[i] = 2*np.pi*r/N - chan_land[i]

    chan_geom = ChannelGeometry(x_j, chan_w, chan_h, chan_t, chan_land, n_chan=n_chan_float)

    flow = solve_flow(geom, cea, config, xf=x_j[-1])
    T_aw_eff = compute_film_taw(flow, geom, cea, config)
    thermal = solve_thermal(flow, geom, cea, chan_geom, config, T_aw_eff=T_aw_eff)

    # Stations to inspect: throat (worst), upstream of split (chamber side, hot),
    # downstream of split (exit side, cooler)
    stations = [
        (0.0,                     'Injector face'),
        (x_split_up - 0.005,      'Just before upstream split'),
        (x_throat,                'THROAT (peak T_hw)'),
        (x_split_down + 0.005,    'Just after downstream split'),
        (geom.L_c + geom.L_nozzle - 1e-3, 'Nozzle exit'),
    ]

    # Two modes: top row = default (no closeout, RPA-comparable), bottom row =
    # with structural closeout above the channel ceiling.
    # Outer-surface convection to ambient air (still air on hot Al ~5–10
    # W/m²K natural conv + ~3–8 W/m²K linearized radiation for ε≈0.3 polished
    # Al at 600–700 K → use ~15 W/m²K total).
    h_ext_ambient = 15.0
    T_ext_ambient = 300.0

    modes = [
        (0.0,    0.0,            'No closeout (default)'),
        (3.3e-3, h_ext_ambient,  f'Closeout 3.3 mm + air (h={h_ext_ambient:.0f})'),
    ]

    fields_by_mode = []
    for closeout_t, h_ext, mode_label in modes:
        print(f"\n########## MODE: {mode_label} ##########")
        row = []
        for x, lbl in stations:
            f = run_field_at(x, f"{lbl}\n[{mode_label}]", flow, geom, cea,
                             chan_geom, thermal, T_aw_eff,
                             closeout_t=closeout_t,
                             h_ext=h_ext, T_ext=T_ext_ambient)
            row.append(f)
            report(f)
        fields_by_mode.append(row)

    n_rows = len(modes)
    n_cols = len(stations)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    for r, row in enumerate(fields_by_mode):
        for c, f in enumerate(row):
            plot_field(f, ax=axes[r, c])
    plt.tight_layout()
    plt.savefig('exports/wall_field_throat.png', dpi=120)
    print("\nSaved: exports/wall_field_throat.png")
    plt.show()


if __name__ == '__main__':
    main()
