"""
sweep_channels.py
Sweep cooling-channel throat geometry — N_throat, h_throat, w_throat — and
report thermal performance + SLM printability metrics for each point.

Designed to find the sweet spot between:
  - thermal: peak T_hw < CuCrZr strength limit (1073 K), and ΔP within budget
  - SLM:     D_h ≥ 0.8 mm (depowdering), aspect ratio ≤ 5 (no ceiling sag),
             land width ≥ 1.0 mm (printable rib), wall ≥ 0.6 mm

Usage:  python scripts/sweep_channels.py [config.toml]
        Default config: configs/the5k_2p5kn_cucrzr.toml
"""
import os, sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
os.chdir(_PARENT)

import io
import contextlib
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from cea_interface import get_cea_for_analysis
from geometry import size_engine
from flow_solver import solve_flow
from heat_transfer import solve_thermal
from channel_builder import build_channel_geometry
from config_loader import load_config


# Parameter grid — change here to refocus the sweep
N_THROATS  = [36, 42, 48, 53]                 # bifurcation: N_chamber = 2 * N_throat
H_THROATS  = [1.1e-3, 1.3e-3, 1.5e-3, 1.8e-3] # channel height at throat [m]
W_THROATS  = [1.0e-3, 1.2e-3, 1.4e-3]         # channel width (constant axially) [m]

# SLM and thermal limit thresholds — change here to retune flags
T_HW_LIMIT_K  = 1073.0   # CuCrZr 800 °C strength limit
DP_BUDGET_BAR = 25.0     # pump-head budget; current inlet is 35 bar
LAND_MIN_MM   = 1.0
DH_MIN_MM     = 0.80
AR_MAX        = 5.0
WALL_MIN_MM   = 0.60     # we keep wall_t at config default; report only


def _flag(value, ok_predicate, fail_marker="!"):
    """Return one-char marker if value violates the predicate."""
    return "" if ok_predicate(value) else fail_marker


def _solve_one(cfg, n_throat, h_throat, w_throat):
    """Run one thermal-solve point. Return diagnostics dict."""
    # Mutate the channels in place (we make a deep-ish copy of the design)
    ch = cfg.channels
    cfg.channels = replace(
        ch,
        n_throat        = int(n_throat),
        n_chamber       = int(2 * n_throat),
        # Chamber/exit heights stay as configured; only throat varies
        height_taper    = [(s, h_throat if s == "throat" else v)
                           for s, v in ch.height_taper],
        width_taper     = [(s, w_throat) for s, _ in ch.width_taper],
        wall_t_taper    = ch.wall_t_taper,
    )
    # Mirror the channel design into legacy fields (config.__post_init__ would
    # do this on construction; we re-mirror after our mutation).
    cd = cfg.channels
    cfg.N_channels         = cd.n_throat
    cfg.N_channels_throat  = cd.n_throat
    cfg.N_channels_chamber = cd.n_chamber

    with contextlib.redirect_stdout(io.StringIO()):
        cea  = get_cea_for_analysis(cfg)
        geom = size_engine(cfg, cea)
        if cfg.mdot_coolant is None:
            cfg.mdot_coolant = geom.mdot_fuel
        chan, _ = build_channel_geometry(cfg, geom)
        flow    = solve_flow(geom, cea, cfg, xf=chan.x_j[-1])
        thermal = solve_thermal(flow, geom, cea, chan, cfg)

    i_thr = int(np.argmin(np.abs(chan.x_j - geom.L_c)))
    w = float(chan.chan_w[i_thr])
    h = float(chan.chan_h[i_thr])
    D_h = 2.0 * w * h / (w + h)
    AR  = h / w
    land  = float(chan.chan_land[i_thr])
    v_thr = float(thermal.v_coolant[i_thr]) if hasattr(thermal, "v_coolant") else float("nan")

    return dict(
        N_throat   = n_throat,
        h_mm       = h * 1000,
        w_mm       = w * 1000,
        T_hw_max_K = float(thermal.T_hw.max()),
        q_max_MW   = float(thermal.heatflux.max()) / 1e6,
        T_out_K    = float(thermal.T_coolant[0]),
        dP_bar     = (cfg.P_coolant_inlet - float(thermal.P_coolant[0])) / 1e5,
        land_mm    = land * 1000,
        D_h_mm     = D_h * 1000,
        AR         = AR,
        v_thr_m_s  = v_thr,
    )


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/the5k_2p5kn_cucrzr.toml"
    print(f"Sweeping channels for {cfg_path}")
    print(f"  N_throat  ∈ {N_THROATS}")
    print(f"  h_throat  ∈ {[f'{h*1000:.1f}' for h in H_THROATS]} mm")
    print(f"  w_throat  ∈ {[f'{w*1000:.1f}' for w in W_THROATS]} mm")
    print(f"  Total: {len(N_THROATS) * len(H_THROATS) * len(W_THROATS)} points")
    print()

    rows = []
    for n, h, w in product(N_THROATS, H_THROATS, W_THROATS):
        cfg = load_config(cfg_path)
        try:
            r = _solve_one(cfg, n, h, w)
        except Exception as exc:
            r = dict(N_throat=n, h_mm=h*1000, w_mm=w*1000,
                     T_hw_max_K=float("nan"), q_max_MW=float("nan"),
                     T_out_K=float("nan"), dP_bar=float("nan"),
                     land_mm=float("nan"), D_h_mm=float("nan"),
                     AR=float("nan"), v_thr_m_s=float("nan"),
                     err=str(exc)[:40])
        rows.append(r)

    # Header
    hdr = (f"  {'N':>3}  {'h':>4}  {'w':>4}    "
           f"{'T_hw':>5} {'q_w':>5} {'T_out':>5} {'ΔP':>4}    "
           f"{'land':>5} {'D_h':>4} {'AR':>4} {'v_thr':>5}    flags")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print("  " + " "*22 + "[K]   [MW]  [K]   [bar]"
          + " "*4 + "[mm]  [mm]      [m/s]")
    print(sep)

    # Sort by T_hw ascending (cooler first), so the safest designs sit at top
    rows.sort(key=lambda r: (float("inf") if np.isnan(r.get("T_hw_max_K", np.nan))
                              else r["T_hw_max_K"]))

    for r in rows:
        if "err" in r:
            print(f"  {r['N_throat']:>3}  {r['h_mm']:>4.1f}  {r['w_mm']:>4.1f}"
                  f"    ERROR: {r['err']}")
            continue
        flags = ""
        flags += _flag(r["T_hw_max_K"], lambda v: v <= T_HW_LIMIT_K, "T")
        flags += _flag(r["dP_bar"],     lambda v: v <= DP_BUDGET_BAR, "P")
        flags += _flag(r["land_mm"],    lambda v: v >= LAND_MIN_MM, "L")
        flags += _flag(r["D_h_mm"],     lambda v: v >= DH_MIN_MM, "D")
        flags += _flag(r["AR"],         lambda v: v <= AR_MAX, "A")
        ok = "✓" if flags == "" else flags
        print(f"  {r['N_throat']:>3}  {r['h_mm']:>4.2f}  {r['w_mm']:>4.2f}    "
              f"{r['T_hw_max_K']:>5.0f} {r['q_max_MW']:>5.1f} "
              f"{r['T_out_K']:>5.0f} {r['dP_bar']:>4.1f}    "
              f"{r['land_mm']:>5.2f} {r['D_h_mm']:>4.2f} "
              f"{r['AR']:>4.2f} {r['v_thr_m_s']:>5.1f}    {ok}")

    print()
    print("  Flag legend:  T = T_hw exceeds CuCrZr strength limit (1073 K)")
    print("                P = ΔP exceeds pump-budget proxy (25 bar)")
    print("                L = land width below SLM floor (1.0 mm)")
    print("                D = hydraulic diameter below depowder floor (0.8 mm)")
    print("                A = aspect ratio above ceiling-sag limit (5.0)")
    print("                ✓ = all constraints met")


if __name__ == "__main__":
    main()
