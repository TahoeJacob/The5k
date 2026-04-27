"""
optimize_channels.py
Sweep N_throat, channel height, and film fraction for a copper-lined
2.5kN RP-1/LOX engine with SLM depowdering constraints.

All channel dimensions (width, height, land) must be >= MIN_DIM for
depowdering.  Bifurcation: N_chamber = 2 × N_throat.

For each N_throat the channel width is set to maximise cooling:
  width = circ_throat / N_throat - MIN_DIM   (land = MIN_DIM at throat)

Usage:  python optimize_channels.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import replace

from config import EngineConfig
from cea_interface import get_cea_for_analysis
from geometry import size_engine, nozzle_radius
from flow_solver import solve_flow
from heat_transfer import ChannelGeometry, solve_thermal
from film_cooling import compute_film_taw
from cea_lookup import build_of_lut


# ── Copper 2.5kN engine config ─────────────────────────────────────
config = EngineConfig(
    fuel        = "RP-1",
    oxidizer    = "LOX",
    coolant     = "RP1",

    P_c   = 20.0e5,          # 20 bar
    F_vac = 2971.0,           # 2.5kN sea-level class

    OF       = 2.0,
    OF_sweep = None,
    frozen   = False,

    exp_ratio  = 5.0,
    cont_ratio = 6.0,
    L_star     = 1.0,         # 1000 mm

    theta1  = 30.0,
    thetaD  = 30.0,
    thetaE  = 12.0,
    R1_mult = 1.5,
    RU_mult = 1.5,
    RD_mult = 0.382,

    # Copper liner
    wall_k         = 300.0,       # W/m·K  (copper alloy)
    wall_roughness = 6.3e-6,      # SLM
    wall_melt_T    = 1350.0,      # K  (CuCrZr approximate)

    # Coolant circuit
    T_coolant_inlet = 298.0,
    P_coolant_inlet = 35.0e5,     # 35 bar
    mdot_coolant    = None,

    # Channels (overridden per sweep point)
    N_channels             = 30,
    N_channels_throat      = 30,
    N_channels_chamber     = 60,
    channel_split_r_ratio  = 2.0,
    channel_split_transition = 10e-3,
    dx = 1e-3,

    # Film cooling (overridden per sweep point)
    film_fraction  = 0.0,
    film_inject_x  = 0.0,
    film_coolant   = "RP1",
    film_T_inlet   = 298.0,
    film_model     = "simon",
    film_Kt        = 0.0013,

    wall_2d        = True,
    use_integral_bl = False,
    C_bartz        = 0.026,
)


# =====================================================================
# SWEEP PARAMETERS
# =====================================================================
WALL_T  = 0.9e-3         # Wall thickness [m]
MIN_DIM = 1.0e-3         # Minimum width and land [m] (SLM depowdering)
MIN_H   = 1.3e-3         # Minimum channel height [m]

# Channel heights to sweep [m]
CHAN_HEIGHTS = [1.0e-3, 1.3e-3, 1.5e-3, 2.0e-3]

# Film fractions to sweep (ascending — goal is 0%)
FILM_FRACTIONS = [0.00, 0.05]

# N_throat range is auto-computed from geometry + MIN_DIM
# =====================================================================


def _build_n_chan_array(x_j, r_arr, geom, N_throat, N_chamber):
    """Build smoothly-transitioning N_channels array with bifurcation."""
    split_r = config.channel_split_r_ratio * geom.R_t
    half_trans = 0.5 * config.channel_split_transition

    split_above = r_arr > split_r
    x_split_up = x_split_down = None
    x_throat = geom.L_c

    for i in range(1, len(x_j)):
        if split_above[i] != split_above[i - 1]:
            r0, r1 = r_arr[i - 1], r_arr[i]
            frac = (split_r - r0) / (r1 - r0) if r1 != r0 else 0.0
            xc = x_j[i - 1] + frac * (x_j[i] - x_j[i - 1])
            if xc < x_throat and x_split_up is None:
                x_split_up = xc
            elif xc > x_throat:
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

    return np.array([n_local(x) for x in x_j])


def run_sweep():
    print("Copper 2.5kN optimisation: N_throat × height × film%")
    print(f"  Target: T_hw < {config.wall_melt_T:.0f} K  (copper)")
    print(f"  SLM depowdering: width, land >= {MIN_DIM*1000:.1f} mm,  height >= {MIN_H*1000:.1f} mm")
    print(f"  P_c = {config.P_c/1e5:.0f} bar,  P_coolant = {config.P_coolant_inlet/1e5:.0f} bar")
    print(f"  wall_k = {config.wall_k:.0f} W/m·K,  wall_t = {WALL_T*1000:.1f} mm")

    # Engine geometry (fixed)
    cea = get_cea_for_analysis(config)
    geom = size_engine(config, cea)

    R_t = geom.R_t
    circ_throat = 2 * np.pi * R_t
    print(f"  R_t = {R_t*1000:.2f} mm,  circ_throat = {circ_throat*1000:.1f} mm")
    print(f"  mdot_fuel = {geom.mdot_fuel*1000:.1f} g/s")

    # N_throat range: pitch >= width_min + land_min = 2 × MIN_DIM
    N_thr_max = int(circ_throat / (2 * MIN_DIM))
    N_thr_min = max(6, int(circ_throat / (10e-3 + MIN_DIM)))  # width <= 10mm
    print(f"  N_throat range: {N_thr_min} – {N_thr_max}")

    x_j = np.arange(0, geom.L_c + geom.L_nozzle, config.dx)
    r_arr = np.array([nozzle_radius(x, geom, config.dx) for x in x_j])

    # Results: (N_thr, N_ch, W_mm, H_mm, land_mm, film%, peak_Thw,
    #           peak_x_mm, dT_cool, dP_bar)
    results = []
    done = 0

    for N_thr in range(N_thr_min, N_thr_max + 1):
        N_ch = 2 * N_thr
        pitch_throat = circ_throat / N_thr
        W = pitch_throat - MIN_DIM   # maximise width, land = MIN_DIM
        land_throat = MIN_DIM

        # Enforce width >= MIN_DIM
        if W < MIN_DIM:
            continue

        n_chan = _build_n_chan_array(x_j, r_arr, geom, N_thr, N_ch)

        # Check feasibility everywhere (land must stay >= 0)
        feasible = True
        min_land = 1.0
        for i in range(len(x_j)):
            circ = 2 * np.pi * r_arr[i]
            land = circ / n_chan[i] - W
            if land < 0:
                feasible = False
                break
            min_land = min(min_land, land)
        if not feasible:
            continue

        print(f"\n{'='*70}")
        print(f"  N_throat = {N_thr}  N_chamber = {N_ch}  |  "
              f"W = {W*1000:.2f} mm  land_throat = {land_throat*1000:.2f} mm")
        print(f"{'='*70}")

        for H in CHAN_HEIGHTS:
            for ff in FILM_FRACTIONS:
                config.film_fraction = ff
                config.mdot_coolant = geom.mdot_fuel * (1.0 - ff)
                config.N_channels = N_thr
                config.N_channels_throat = N_thr
                config.N_channels_chamber = N_ch

                # Build channel geometry
                chan_w    = np.full_like(x_j, W)
                chan_h    = np.full_like(x_j, H)
                chan_t    = np.full_like(x_j, WALL_T)
                chan_land = np.zeros_like(x_j)
                for i in range(len(x_j)):
                    circ = 2 * np.pi * r_arr[i]
                    chan_land[i] = circ / n_chan[i] - W

                chan_geom = ChannelGeometry(x_j, chan_w, chan_h, chan_t,
                                            chan_land, n_chan=n_chan)

                # Flow solution
                flow = solve_flow(geom, cea, config, xf=x_j[-1])

                # Film cooling
                T_aw_eff = OF_eff = phase_code = cea_per_station = None
                if ff > 0:
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

                done += 1
                try:
                    thermal = solve_thermal(
                        flow, geom, cea, chan_geom, config,
                        T_aw_eff=T_aw_eff,
                        cea_per_station=cea_per_station,
                        phase_code=phase_code)
                except Exception as e:
                    print(f"    FAILED N={N_thr} H={H*1000:.1f} "
                          f"film={ff*100:.0f}%: {e}")
                    continue

                peak_idx = int(np.argmax(thermal.T_hw))
                peak_Thw = float(thermal.T_hw[peak_idx])
                peak_x   = float(thermal.x[peak_idx]) * 1000
                dT_cool  = float(thermal.T_coolant.max() -
                                  thermal.T_coolant.min())
                dP_bar   = float(thermal.P_coolant.max() -
                                  thermal.P_coolant.min()) / 1e5

                results.append((N_thr, N_ch, W * 1000, H * 1000,
                                land_throat * 1000, ff * 100, peak_Thw,
                                peak_x, dT_cool, dP_bar))

                status = "OK" if peak_Thw < config.wall_melt_T else "OVER"
                print(f"  [{done:3d}] N={N_thr}/{N_ch} W={W*1000:.1f} "
                      f"H={H*1000:.1f}  film={ff*100:2.0f}%  "
                      f"T_hw={peak_Thw:7.0f}K  dT={dT_cool:5.0f}K  "
                      f"dP={dP_bar:5.1f}bar  {status}")

    if not results:
        print("\nNo feasible configurations found!")
        return

    # Sort by film% then peak T_hw
    results.sort(key=lambda r: (r[5], r[6]))

    # Print all results
    print(f"\n{'='*110}")
    print(f"  ALL RESULTS (sorted by film%, then peak T_hw)")
    print(f"{'='*110}")
    print(f"  {'N_thr':>5} {'N_ch':>4} {'W_mm':>5} {'H_mm':>5} {'land':>5} "
          f"{'film%':>5} {'T_hw':>7} {'x_pk':>6} {'dT':>5} {'dP':>6}")
    print(f"  {'-'*70}")
    for r in results:
        Nt, Nc, W_mm, H_mm, lnd, ff, Thw, xpk, dT, dP = r
        flag = " ***" if Thw > config.wall_melt_T else " OK"
        print(f"  {Nt:5d} {Nc:4d} {W_mm:5.1f} {H_mm:5.1f} {lnd:5.2f} "
              f"{ff:5.0f} {Thw:7.0f} {xpk:6.0f} {dT:5.0f} {dP:6.1f}{flag}")

    # Surviving configs
    survivors = [r for r in results if r[6] < config.wall_melt_T]
    if survivors:
        best = min(survivors, key=lambda r: (r[5], r[6]))
        print(f"\n  {len(survivors)} configurations SURVIVE "
              f"(T_hw < {config.wall_melt_T:.0f} K)")
        print(f"  BEST (min film): N={best[0]}/{best[1]}, W={best[2]:.1f}mm, "
              f"H={best[3]:.1f}mm, film={best[5]:.0f}%")
        print(f"        Peak T_hw = {best[6]:.0f} K at x = {best[7]:.0f} mm, "
              f"dT_cool = {best[8]:.0f} K, dP = {best[9]:.1f} bar")

        # Best at each film fraction
        print(f"\n  Best at each film fraction:")
        for ff_val in sorted(set(r[5] for r in survivors)):
            sub = [r for r in survivors if r[5] == ff_val]
            b = min(sub, key=lambda r: r[6])
            print(f"    film={b[5]:2.0f}%: N={b[0]}/{b[1]} W={b[2]:.1f}mm "
                  f"H={b[3]:.1f}mm  T_hw={b[6]:.0f}K  dP={b[9]:.1f}bar")

        # At 0% film: list all surviving configs
        zero_film = [r for r in survivors if r[5] == 0]
        if zero_film:
            print(f"\n  0% FILM SURVIVORS ({len(zero_film)} configs):")
            for r in sorted(zero_film, key=lambda r: r[6]):
                print(f"    N={r[0]}/{r[1]} W={r[2]:.1f}mm H={r[3]:.1f}mm  "
                      f"T_hw={r[6]:.0f}K  dT={r[8]:.0f}K  dP={r[9]:.1f}bar")
        else:
            # Show closest at 0%
            zero_all = [r for r in results if r[5] == 0]
            if zero_all:
                b = min(zero_all, key=lambda r: r[6])
                print(f"\n  No 0% film configs survive. Closest: "
                      f"N={b[0]}/{b[1]} W={b[2]:.1f}mm H={b[3]:.1f}mm  "
                      f"T_hw={b[6]:.0f}K  (need {b[6]-config.wall_melt_T:.0f}K reduction)")
    else:
        best = min(results, key=lambda r: r[6])
        print(f"\n  NO configurations survive T_limit = {config.wall_melt_T:.0f} K")
        print(f"  Closest: N={best[0]}/{best[1]} W={best[2]:.1f}mm "
              f"H={best[3]:.1f}mm film={best[5]:.0f}%")
        print(f"           Peak T_hw = {best[6]:.0f} K  "
              f"(need {best[6] - config.wall_melt_T:.0f} K more reduction)")

    # ── Plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: T_hw vs N_throat for each height (0% film only)
    ax1 = axes[0]
    for H_mm in sorted(set(r[3] for r in results)):
        sub = [r for r in results if abs(r[3] - H_mm) < 0.01 and r[5] == 0]
        if sub:
            sub.sort(key=lambda r: r[0])
            ns = [r[0] for r in sub]
            thws = [r[6] for r in sub]
            ax1.plot(ns, thws, 'o-', ms=5, label=f'H = {H_mm:.1f} mm')

    ax1.axhline(config.wall_melt_T, color='r', ls=':', lw=1,
                label=f'T_limit = {config.wall_melt_T:.0f} K')
    ax1.set_xlabel('N_throat')
    ax1.set_ylabel('Peak T_hw [K]')
    ax1.set_title('0% Film: Peak T_hw vs N_throat')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: T_hw vs film% for each N_throat (best height)
    ax2 = axes[1]
    for N_thr in sorted(set(r[0] for r in results)):
        best_per_ff = {}
        for r in results:
            if r[0] == N_thr:
                ff = r[5]
                if ff not in best_per_ff or r[6] < best_per_ff[ff][6]:
                    best_per_ff[ff] = r
        if best_per_ff:
            ffs = sorted(best_per_ff.keys())
            thws = [best_per_ff[f][6] for f in ffs]
            W_mm = best_per_ff[ffs[0]][2]
            ax2.plot(ffs, thws, 'o-', ms=5,
                     label=f'N={N_thr} W={W_mm:.1f}mm')

    ax2.axhline(config.wall_melt_T, color='r', ls=':', lw=1,
                label=f'T_limit = {config.wall_melt_T:.0f} K')
    ax2.set_xlabel('Film cooling [%]')
    ax2.set_ylabel('Peak T_hw [K]')
    ax2.set_title('Peak T_hw vs Film % (best height)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel 3: dP vs N_throat for each height (0% film)
    ax3 = axes[2]
    for H_mm in sorted(set(r[3] for r in results)):
        sub = [r for r in results if abs(r[3] - H_mm) < 0.01 and r[5] == 0]
        if sub:
            sub.sort(key=lambda r: r[0])
            ns = [r[0] for r in sub]
            dps = [r[9] for r in sub]
            ax3.plot(ns, dps, 'o-', ms=5, label=f'H = {H_mm:.1f} mm')

    ax3.set_xlabel('N_throat')
    ax3.set_ylabel('Coolant ΔP [bar]')
    ax3.set_title('0% Film: Pressure drop vs N_throat')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('copper_optimization.png', dpi=150)
    print(f"\n  Plot saved to copper_optimization.png")


if __name__ == "__main__":
    run_sweep()
