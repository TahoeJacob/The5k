"""
validate_harcc.py
Compare model output against the DLR HARCC subscale paper data.

Paper:
    J. Haemisch, D. Suslov, M. Oschwald,
    "Experimental and Numerical Investigation of Heat Transfer Processes in
     Rocket Engine Cooling Channels Operated with Cryogenic Hydrogen and
     Methane at Supercritical Conditions",
    32nd International Symposium on Space Technology and Science, 2019.
    PDF: https://elib.dlr.de/128226/1/Paper_a90042.pdf

Caveat — HARCC is NOT a complete engine
----------------------------------------
The HARCC test article is a 200 mm cylindrical combustion-chamber segment
bolted between an injector + a 200 mm "Standard-Segment" upstream and a
separate cooled nozzle downstream.  The test article has NO throat or
nozzle of its own.  Our solver builds a chamber+throat+bell from F+P_c+CEA,
so the *peak* values our model produces are at the throat — which doesn't
exist in the paper's test article.

This script therefore restricts the comparison to the CYLINDRICAL chamber
region of our model (where wall radius ≈ R_c).  Quantities reported:
  - Averaged q_w over the cylindrical region (vs paper's calorimetric S1)
  - q_w(x), T_hw(x), h_gas(x) sampled at the paper's four thermocouple
    positions (P1=52, P2=85, P3=119, P4=152 mm from HARCC inlet), mapped
    to our cylindrical region as a fraction of cylinder length.
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

import matplotlib
matplotlib.use("Agg")

from cea_interface import get_cea_for_analysis
from geometry import size_engine, _segment_breakpoints
from flow_solver import solve_flow
from heat_transfer import solve_thermal
from channel_builder import build_channel_geometry
from config_loader import load_config


# Paper data — Sector S1 (AR 1.7), reported in tables 4 & 5 + figs 8a, 9a
PAPER_DATA = {
    "LH2": {
        "config":          "configs/dlr_harcc_lh2.toml",
        "P_c_bar":         49.1,
        "OF":              3.9,
        "F_kN":            17.0,
        "T_in_K":          63.5,
        "P_out_bar":       155.3,
        "mdot_per_ch_g_s": 7.5,
        # Calorimetric (mean over 200 mm) and inverse-method (linear at HARCC
        # inlet/outlet).  Paper Table 4.
        "q_w_calo_MW_m2":  26.3,
        "q_w_inv_in_MW_m2":  20.1,
        "q_w_inv_out_MW_m2": 25.2,
        # APPROXIMATE — eyeballed from paper fig 8a (S1 AR 1.7, hydrogen).
        # Y-axis shown in fig is roughly 200–550 K. Values are estimates;
        # if you have the figure handy, replace these with read-off values.
        "T_w_paper_K_at_TC": [(52, 285), (85, 305), (119, 320), (152, 335)],
    },
    "LCH4": {
        "config":          "configs/dlr_harcc_lch4.toml",
        "P_c_bar":         50.2,
        "OF":              2.0,
        "F_kN":            17.8,
        "T_in_K":          138.6,
        "P_out_bar":       78.1,
        "mdot_per_ch_g_s": 20.1,
        "q_w_calo_MW_m2":  14.5,
        "q_w_inv_in_MW_m2":  13.7,
        "q_w_inv_out_MW_m2": 14.7,
        # APPROXIMATE — eyeballed from paper fig 9a (S1 AR 1.7, methane).
        # Y-axis shown in fig is roughly 200–800 K. Values are estimates;
        # if you have the figure handy, replace these with read-off values.
        "T_w_paper_K_at_TC": [(52, 410), (85, 510), (119, 595), (152, 670)],
    },
}


def cylindrical_region_indices(thermal_x: np.ndarray, geom) -> tuple[int, int]:
    """Return (i_start, i_end) bounding the cylindrical chamber region.

    The cylinder runs from injector face (x=0) to the start of the chamber
    arc — _segment_breakpoints[1] gives (label, x_start, x_end, color) for
    the chamber arc, so the cylinder ends at segs[1][1].
    """
    segs = _segment_breakpoints(geom)
    cyl_end = segs[1][1]   # x where chamber arc starts
    i_start = 0
    i_end   = int(np.searchsorted(thermal_x, cyl_end, side="right")) - 1
    return i_start, i_end


def _solve(cfg_path, use_integral_bl):
    cfg = load_config(cfg_path)
    cfg.use_integral_bl = use_integral_bl
    cea = get_cea_for_analysis(cfg)
    geom = size_engine(cfg, cea)
    if cfg.mdot_coolant is None:
        cfg.mdot_coolant = geom.mdot_fuel
    chan, info = build_channel_geometry(cfg, geom)
    flow = solve_flow(geom, cea, cfg, xf=chan.x_j[-1])
    thermal = solve_thermal(flow, geom, cea, chan, cfg)
    return cfg, geom, chan, info, thermal


def run_one_case(label: str, paper: dict):
    print(f"\n{'='*76}")
    print(f"  HARCC validation — {label}  (simplified Bartz vs integral-BL)")
    print(f"  Paper: P_c = {paper['P_c_bar']} bar, OF = {paper['OF']}, "
          f"F = {paper['F_kN']} kN")
    print(f"{'='*76}")

    print("  Run 1: simplified Bartz")
    cfg_s, geom_s, chan_s, info_s, t_s = _solve(paper["config"], False)
    print("  Run 2: integral-BL Bartz")
    cfg, geom, chan, info, thermal = _solve(paper["config"], True)

    # --- restrict to the cylindrical region -----------------------------
    i0, i1 = cylindrical_region_indices(thermal.x, geom)
    cyl_x = thermal.x[i0:i1+1]
    cyl_qw  = thermal.heatflux[i0:i1+1]
    cyl_Thw = thermal.T_hw[i0:i1+1]
    cyl_hg  = thermal.h_gas[i0:i1+1]
    cyl_Tc  = thermal.T_coolant[i0:i1+1]
    cyl_Pc  = thermal.P_coolant[i0:i1+1]
    L_cyl_mm = (cyl_x[-1] - cyl_x[0]) * 1000.0
    # simplified-Bartz comparison arrays
    i0s, i1s = cylindrical_region_indices(t_s.x, geom_s)
    s_cyl_qw  = t_s.heatflux[i0s:i1s+1]
    s_cyl_Thw = t_s.T_hw[i0s:i1s+1]

    print(f"\n  Generated geometry:")
    print(f"    D_c = {2*geom.R_c*1000:.2f} mm    "
          f"(paper: 80.0 mm — diff {(2*geom.R_c*1000-80)/80*100:+.1f}%)")
    print(f"    Cylindrical region: x = {cyl_x[0]*1000:.1f} → "
          f"{cyl_x[-1]*1000:.1f} mm  (length {L_cyl_mm:.1f} mm)")
    print(f"    Paper measurement region: 200 mm of cylinder "
          f"(HARCC segment), 4 thermocouple positions at 52/85/119/152 mm "
          f"from HARCC inlet")

    # --- area-weighted average q_w over the cylinder --------------------
    # Strictly: integrate q_w · 2πr · dx then divide by surface area.
    # In the cylinder r is constant → simple arithmetic mean.
    qw_avg     = float(np.mean(cyl_qw))
    qw_avg_s   = float(np.mean(s_cyl_qw))
    Thw_avg    = float(np.mean(cyl_Thw))
    Thw_avg_s  = float(np.mean(s_cyl_Thw))
    paper_q    = paper["q_w_calo_MW_m2"]
    print(f"\n  q_w averaged over cylindrical region:")
    print(f"    Paper (S1, calorimetric, 200 mm avg): {paper_q:.2f} MW/m²")
    print(f"    Simplified Bartz : {qw_avg_s/1e6:6.2f} MW/m²  "
          f"({(qw_avg_s/1e6 - paper_q)/paper_q*100:+.0f}%)")
    print(f"    Integral-BL Bartz: {qw_avg/1e6:6.2f} MW/m²  "
          f"({(qw_avg/1e6   - paper_q)/paper_q*100:+.0f}%)")
    print(f"    Paper inverse-method linear: "
          f"{paper['q_w_inv_in_MW_m2']:.2f} (in) → "
          f"{paper['q_w_inv_out_MW_m2']:.2f} (out) MW/m²")
    print(f"  T_hw averaged over cylindrical region:")
    print(f"    Simplified Bartz : {Thw_avg_s:.0f} K")
    print(f"    Integral-BL Bartz: {Thw_avg:.0f} K")

    # --- Sample at paper's TC positions, mapped onto our cylinder -------
    # Paper TC positions are in [mm] from HARCC inlet, paper cylinder = 200mm.
    # Map: paper_frac = paper_x_mm / 200.0  → our_x = paper_frac * L_cyl
    print(f"\n  Sampled at thermocouple positions (mapped onto our "
          f"{L_cyl_mm:.0f} mm cylinder):")
    print(f"    {'TC':>4} {'paper_x':>9} {'our_x':>9} "
          f"{'q_w_us':>10} {'q_w_inv':>10} "
          f"{'T_hw_us':>10} {'T_w_paper':>11}  err")
    print(f"    {'':>4} {'[mm]':>9} {'[mm]':>9} "
          f"{'[MW/m²]':>10} {'[MW/m²]':>10} "
          f"{'[K]':>10} {'[K]':>11}")
    print(f"    " + "-"*78)
    for tc_idx, (paper_x_mm, paper_T_K) in enumerate(paper["T_w_paper_K_at_TC"], 1):
        frac = paper_x_mm / 200.0
        our_x = cyl_x[0] + frac * (cyl_x[-1] - cyl_x[0])
        j = int(np.argmin(np.abs(cyl_x - our_x)))
        # Linear interp the paper inverse-method q_w at this position
        q_inv = paper["q_w_inv_in_MW_m2"] + (
            paper["q_w_inv_out_MW_m2"] - paper["q_w_inv_in_MW_m2"]
        ) * frac
        err = (cyl_Thw[j] - paper_T_K) / paper_T_K * 100
        print(f"    P{tc_idx:<3} {paper_x_mm:9.1f} {our_x*1000:9.1f} "
              f"{cyl_qw[j]/1e6:10.2f} {q_inv:10.2f} "
              f"{cyl_Thw[j]:10.0f} {paper_T_K:11.0f}  {err:+5.0f}%")

    print(f"\n  Other diagnostics in the cylindrical region:")
    print(f"    h_gas     : {cyl_hg.min()/1e3:.2f} → {cyl_hg.max()/1e3:.2f}  kW/(m²·K)")
    print(f"    T_coolant : {cyl_Tc.min():.1f} → {cyl_Tc.max():.1f}  K  "
          f"(ΔT = {cyl_Tc.max() - cyl_Tc.min():.1f} K, "
          f"counter-flow — paper is co-flow)")
    print(f"    P_coolant : {cyl_Pc.min()/1e5:.2f} → {cyl_Pc.max()/1e5:.2f}  bar")

    return dict(qw_avg=qw_avg, L_cyl_mm=L_cyl_mm,
                paper_qw_calo=paper["q_w_calo_MW_m2"])


def main():
    print("HARCC subscale validation (cylindrical chamber region only).")
    print("Paper: Haemisch et al., DLR Lampoldshausen, 2019 — elib.dlr.de/128226")
    print("Caveat: HARCC has no throat/nozzle. Comparison restricted to where")
    print("        our model is in the cylindrical chamber section.")
    print("        Cooling direction is also reversed (we counter-flow,")
    print("        paper co-flow) — affects T_w(x) profile, not chamber-mean q_w.")
    for label, paper in PAPER_DATA.items():
        run_one_case(label, paper)


if __name__ == "__main__":
    main()
