"""
validate_lumen.py
Compare model output against the DLR LUMEN regen-cooling design papers.

Sources:
  [1] Haemisch et al., "LUMEN — Design of the Regenerative Cooling System
      for an Expander Bleed Cycle Engine Using Methane",
      Space Propulsion 2020+1 SP2020_00068.
      References/210217_SP_Haemisch_final.pdf
  [2] Dresia et al., "Improved Wall Temperature Prediction for the LUMEN
      Rocket Combustion Chamber with Neural Networks",
      MDPI Aerospace 10(5):450, 2023.
      doi:10.3390/aerospace10050450
      References/aerospace-10-00450-v3.pdf

Channel geometry in configs/lumen_25kn.toml is now ALL paper-REPORTED.
Engine envelope (D_t, L_c, L_n) is still solver-derived from F+P_c+CEA;
absolute axial positions of h2/h3 in our model differ slightly from the
paper because we don't import the exact CAD contour.
"""
import os, sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
os.chdir(_PARENT)

import numpy as np
import matplotlib
matplotlib.use("Agg")

from cea_interface import get_cea_for_analysis
from geometry import size_engine, _segment_breakpoints
from flow_solver import solve_flow
from heat_transfer import solve_thermal
from channel_builder import build_channel_geometry
from config_loader import load_config


# Paper-reported targets (paper [1] tables 1, 4, 5 + paper [2] §4.3)
PAPER = {
    "config":          "configs/lumen_25kn.toml",
    "P_c_bar":         60.0,
    "OF":              3.4,
    "thrust_class_kN": 25.0,
    "T_in_K":          120.0,
    "P_out_bar":       80.0,        # paper [2] nominal (paper [1] used 68.4)
    "mdot_kg_s":       2.6,         # paper [2] nominal (paper [1] used 2.35)
    "n_channels":      86,
    "channel_width_mm":  1.0,
    "hotwall_t_mm":      1.0,
    "roughness_um":      5.0,
    # Paper [1] table 5 (CFD, design point, paper [1] mass flow / Pout)
    "T_w_h2_K":        874.1,       # peak wall T near injector cylinder zone
    "T_w_h4_K":        879.1,       # peak wall T near throat
    "T_out_K":         408.6,       # coolant outlet T at injector face
    "delta_P_bar":     23.7,        # cooling-channel ΔP
    # Paper [2] §4.3
    "q_max_MW_m2":     51.0,        # peak heat flux from ROCFLAM-III profile
    # Paper [1] table 4 — channel heights at 5 control points
    "h_taper_mm":      {"h1": 8.2, "h2": 4.0, "h3": 4.0, "h4": 1.8, "h5": 4.1},
    "h_taper_distances_mm": {"h1_to_h2": 60, "h2_to_h3": 247.2, "h3_to_h4": 119},
}


def _run_thermal(use_integral_bl: bool):
    """Run the full thermal solve once and return (cfg, geom, chan, info, thermal)."""
    cfg = load_config(PAPER["config"])
    cfg.use_integral_bl = use_integral_bl
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)
    if cfg.mdot_coolant is None:
        cfg.mdot_coolant = geom.mdot_fuel
    chan, info = build_channel_geometry(cfg, geom)
    flow = solve_flow(geom, cea, cfg, xf=chan.x_j[-1])
    thermal = solve_thermal(flow, geom, cea, chan, cfg)
    return cfg, geom, chan, info, thermal


def _summarize(thermal, geom, label):
    i_thr = int(np.argmin(np.abs(thermal.x - geom.L_c)))
    return dict(
        label   = label,
        q_max   = float(thermal.heatflux.max())/1e6,
        Thw_max = float(thermal.T_hw.max()),
        Thw_inj = float(thermal.T_hw[0]),
        Thw_thr = float(thermal.T_hw[i_thr]),
        T_out   = float(thermal.T_coolant[0]),
        dP      = (thermal.P_coolant.max() - thermal.P_coolant[0]) / 1e5,
    )


def main():
    print("LUMEN cooling-channel validation — Bartz simplified vs integral-BL")
    print("Sources: Haemisch SP2020+1 [1] + Dresia MDPI 2023 [2]")
    print("="*78)

    print("\n--- Run 1: simplified Bartz (default) ---")
    cfg, geom, chan, info, thermal_simple = _run_thermal(use_integral_bl=False)
    s_simple = _summarize(thermal_simple, geom, "simplified")

    print("\n--- Run 2: Bartz 1965 integral-BL method ---")
    cfg2, geom2, chan2, info2, thermal_intbl = _run_thermal(use_integral_bl=True)
    s_intbl = _summarize(thermal_intbl, geom2, "integral-BL")

    # Use integral-BL run for the channel/envelope summary table
    chan, info, thermal = chan2, info2, thermal_intbl

    # ----------------------------------------------------------------
    # 1. Engine envelope sanity
    # ----------------------------------------------------------------
    print("\n--- Engine envelope ---")
    print(f"  D_chamber : ours {2*geom.R_c*1000:6.2f} mm   "
          f"paper 80.00 mm   diff {(2*geom.R_c*1000-80)/80*100:+.1f}%")
    print(f"  D_throat  : ours {2*geom.R_t*1000:6.2f} mm   paper not stated")
    print(f"  L_c       : ours {geom.L_c*1000:6.1f} mm   "
          f"paper L_cyl + transitions ≈ "
          f"{sum(PAPER['h_taper_distances_mm'].values()):.1f} mm")
    print(f"  L_nozzle  : ours {geom.L_nozzle*1000:6.1f} mm  "
          f"(cooled MCC; paper has ~150 mm cooled before extension)")
    print(f"  ε exit    : ours {cfg.exp_ratio:.1f}      "
          f"(approximate — sized so cooled L_n matches paper)")

    # ----------------------------------------------------------------
    # 2. Channel-level: confirm taper exactly matches paper
    # ----------------------------------------------------------------
    print("\n--- Channels (REPORTED in paper [1] table 4) ---")
    print(f"  n_channels: ours {info['N_throat']}   paper {PAPER['n_channels']}")
    print(f"  Width     : ours {chan.chan_w[0]*1000:.2f}–"
          f"{chan.chan_w[-1]*1000:.2f} mm   paper 1.00 mm constant")
    print(f"  Wall t    : ours {chan.chan_t[0]*1000:.2f} mm  "
          f"paper {PAPER['hotwall_t_mm']:.2f} mm")
    print(f"  Roughness : ours {cfg.wall_roughness*1e6:.1f} µm   "
          f"paper {PAPER['roughness_um']:.1f} µm")
    print(f"  Heights @ paper control points:")
    i_inj = 0
    i_thr = int(np.argmin(np.abs(chan.x_j - geom.L_c)))
    i_ext = -1
    print(f"    h1 (injector): ours {chan.chan_h[i_inj]*1000:.2f} mm   "
          f"paper {PAPER['h_taper_mm']['h1']:.2f} mm")
    print(f"    h4 (throat):   ours {chan.chan_h[i_thr]*1000:.2f} mm   "
          f"paper {PAPER['h_taper_mm']['h4']:.2f} mm")
    print(f"    h5 (exit):     ours {chan.chan_h[i_ext]*1000:.2f} mm   "
          f"paper {PAPER['h_taper_mm']['h5']:.2f} mm")

    # ----------------------------------------------------------------
    # 3. Side-by-side: simplified Bartz vs integral-BL Bartz vs paper
    # ----------------------------------------------------------------
    def _row(label, paper_val, units, fmt, simple_val, intbl_val):
        e_s = (simple_val - paper_val) / paper_val * 100 if paper_val else 0
        e_i = (intbl_val  - paper_val) / paper_val * 100 if paper_val else 0
        print(f"  {label:<26} {paper_val:>{8}{fmt}} {units:<6} "
              f"  simple {simple_val:>{8}{fmt}} ({e_s:+5.0f}%)  "
              f"  int-BL {intbl_val:>{8}{fmt}} ({e_i:+5.0f}%)")

    print("\n--- Side-by-side: paper [1] table 5 vs both Bartz variants ---")
    print(f"  {'metric':<26} {'paper':>14}    "
          f"{'simplified Bartz':>26}    {'integral-BL Bartz':>26}")
    print(f"  {'-'*98}")
    _row("Peak q_w",          PAPER["q_max_MW_m2"], "MW/m²", ".2f",
         s_simple["q_max"],   s_intbl["q_max"])
    _row("Peak T_hw",         PAPER["T_w_h4_K"],    "K",     ".0f",
         s_simple["Thw_max"], s_intbl["Thw_max"])
    _row("T_hw @ injector",   PAPER["T_w_h2_K"],    "K",     ".0f",
         s_simple["Thw_inj"], s_intbl["Thw_inj"])
    _row("T_hw @ throat",     PAPER["T_w_h4_K"],    "K",     ".0f",
         s_simple["Thw_thr"], s_intbl["Thw_thr"])
    _row("Coolant T_out",     PAPER["T_out_K"],     "K",     ".1f",
         s_simple["T_out"],   s_intbl["T_out"])
    _row("Coolant ΔP",        PAPER["delta_P_bar"], "bar",   ".1f",
         s_simple["dP"],      s_intbl["dP"])

    # ----------------------------------------------------------------
    # 4. Diagnostics
    # ----------------------------------------------------------------
    # cylindrical region average (where r ≈ R_c)
    segs = _segment_breakpoints(geom)
    cyl_end = segs[1][1]
    i_cyl_end = int(np.searchsorted(thermal.x, cyl_end, side="right")) - 1
    print(f"\n--- Cylindrical region (x = 0 → {cyl_end*1000:.0f} mm) ---")
    print(f"  q_w avg : {float(np.mean(thermal.heatflux[:i_cyl_end]))/1e6:.2f} MW/m²")
    print(f"  T_hw    : {thermal.T_hw[:i_cyl_end].min():.0f} → "
          f"{thermal.T_hw[:i_cyl_end].max():.0f} K")
    print(f"  h_gas   : {thermal.h_gas[:i_cyl_end].min()/1e3:.1f} → "
          f"{thermal.h_gas[:i_cyl_end].max()/1e3:.1f} kW/(m²·K)")
    print(f"  T_cool  : {thermal.T_coolant[:i_cyl_end].min():.0f} → "
          f"{thermal.T_coolant[:i_cyl_end].max():.0f} K")


if __name__ == "__main__":
    main()
