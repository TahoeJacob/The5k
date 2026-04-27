"""
thermal_report.py
Generate a self-contained markdown report — engine config + computed
geometry + axial-station thermal table + relevant model excerpts —
suitable for sharing with another reviewer (e.g. another Claude) when
asking "is the thermal model over-predicting T_hw?"

Output: exports/thermal_report.md

Run from anywhere:
    python scripts/thermal_report.py
"""

# --- run-from-anywhere shim (file lives in subfolder) ---
import os, sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
os.chdir(_PARENT)
# --------------------------------------------------------

import inspect
import numpy as np

import main
import heat_transfer
from geometry import nozzle_radius


N_TABLE_STATIONS = 30
OUT_PATH = "exports/thermal_report.md"


def _select_indices(x_arr: np.ndarray, n: int, extra_x: list) -> list:
    """Pick ~n evenly-spaced indices, plus indices closest to extra_x values."""
    base = np.linspace(0, len(x_arr) - 1, n, dtype=int).tolist()
    for xv in extra_x:
        if xv is None:
            continue
        base.append(int(np.argmin(np.abs(x_arr - xv))))
    return sorted(set(base))


def _fmt_row(cells, widths):
    return "| " + " | ".join(f"{c:>{w}}" for c, w in zip(cells, widths)) + " |"


def main_report():
    print("Running pipeline (plot=False)...")
    res = main.run(plot=False)
    if res is None:
        print("CEA returned None (sweep mode?); aborting report.")
        return

    cfg       = res["config"]
    cea       = res["cea"]
    geom      = res["geom"]
    chan_geom = res["chan_geom"]
    flow      = res["flow"]
    th        = res["thermal"]

    L_total = geom.L_c + geom.L_nozzle
    x_arr   = th.x

    i_peak = int(np.argmax(th.T_hw))
    x_peak = float(x_arr[i_peak])

    # -- Pick rows for the table -------------------------------------------
    extra = [0.0, geom.L_c, L_total, x_peak]   # injector, throat, exit, T_hw peak
    idxs  = _select_indices(x_arr, N_TABLE_STATIONS, extra)

    # Compute r(x) for each selected station
    r_at = np.array([nozzle_radius(float(x_arr[i]), geom, cfg.dx) for i in idxs])

    # Compute interpolated flow values at thermal stations
    M_th = np.interp(x_arr, flow.x, flow.M)
    P_th = np.interp(x_arr, flow.x, flow.P)
    T_th = np.interp(x_arr, flow.x, flow.T)

    # T_aw at each thermal station (no film: pure recovery temperature)
    T_aw = np.array([heat_transfer._T_aw(float(M_th[i]), cea) for i in range(len(x_arr))])

    md = []

    # ===== Header ========================================================
    md.append("# Thermal Model Report — No Film Cooling Baseline")
    md.append("")
    md.append("**Question for reviewer:** Hot-wall temperature `T_hw` looks "
              "massively over-predicted for what should be a tractable "
              "no-film, copper-walled chamber. Please audit the model "
              "(formulation, boundary conditions, units) using the data "
              "and code excerpts below.")
    md.append("")
    md.append(f"- Pipeline entry point: `RocketThermalAnalysis/main.py` "
              f"→ `run(plot=False)`")
    md.append(f"- Solver: 1-D axial march coupled to a 2-D wall conduction "
              f"slice (`config.wall_2d = {cfg.wall_2d}`)")
    md.append(f"- Film cooling: `film_fraction = {cfg.film_fraction}` (none)")
    md.append("")

    # ===== Engine config ==================================================
    md.append("## 1. Engine config")
    md.append("")
    md.append("| Field | Value |")
    md.append("|---|---|")
    md.append(f"| Fuel / Oxidizer | {cfg.fuel} / {cfg.oxidizer} |")
    md.append(f"| Coolant | {cfg.coolant} |")
    md.append(f"| Chamber pressure P_c | {cfg.P_c/1e5:.2f} bar |")
    md.append(f"| Vacuum thrust F_vac | {cfg.F_vac:.1f} N |")
    md.append(f"| O/F ratio | {cfg.OF:.2f} |")
    md.append(f"| Expansion ratio Ae/At | {cfg.exp_ratio:.2f} |")
    md.append(f"| Contraction ratio Ac/At | {cfg.cont_ratio:.2f} |")
    md.append(f"| L* | {cfg.L_star*1000:.1f} mm |")
    md.append(f"| Nozzle θ1 / θD / θE | {cfg.theta1:.1f}° / "
              f"{cfg.thetaD:.1f}° / {cfg.thetaE:.1f}° |")
    md.append(f"| R_chamber, R_throat_conv, R_throat_div mults | "
              f"{cfg.R_chamber_mult:.3f}, {cfg.R_throat_conv_mult:.3f}, "
              f"{cfg.R_throat_div_mult:.3f} |")
    md.append(f"| CEA mode | {'frozen' if cfg.frozen else 'equilibrium'} |")
    md.append(f"| Bartz coefficient C | {cfg.C_bartz} |")
    md.append(f"| Use integral BL | {cfg.use_integral_bl} |")
    md.append("")

    # ===== Wall + coolant =================================================
    md.append("## 2. Wall material & coolant boundary")
    md.append("")
    md.append("| Field | Value |")
    md.append("|---|---|")
    md.append(f"| Wall material | {cfg.wall_material} |")
    md.append(f"| Wall conductivity k | {cfg.wall_k:.1f} W/(m·K) |")
    md.append(f"| Wall roughness | {cfg.wall_roughness*1e6:.1f} µm |")
    md.append(f"| Wall T limit (config.wall_melt_T) | "
              f"{cfg.wall_melt_T:.0f} K ({cfg.wall_melt_T-273.15:.0f}°C) |")
    md.append(f"| Coolant inlet T | {cfg.T_coolant_inlet:.1f} K |")
    md.append(f"| Coolant inlet P | {cfg.P_coolant_inlet/1e5:.2f} bar |")
    md.append(f"| Coolant mass flow (computed) | "
              f"{cfg.mdot_coolant:.4f} kg/s |")
    md.append(f"| Coolant flow direction | counter-current "
              f"(enters at nozzle exit, exits at injector face) |")
    md.append(f"| 2-D wall solve | `wall_2d = {cfg.wall_2d}` |")
    md.append(f"| 2-D wall stress | `wall_2d_stress = {cfg.wall_2d_stress}` |")
    md.append("")

    # ===== CEA outputs ====================================================
    md.append("## 3. CEA stagnation-state outputs")
    md.append("")
    md.append("| Field | Value | Units |")
    md.append("|---|---|---|")
    md.append(f"| T_c (chamber stag temp) | {cea.T_c:.1f} | K |")
    md.append(f"| γ_c | {cea.gamma_c:.4f} | — |")
    md.append(f"| C* | {cea.C_star:.1f} | m/s |")
    md.append(f"| Cp_froz_c | {cea.Cp_froz_c:.1f} | J/(kg·K) |")
    md.append(f"| Pr_froz_c | {cea.Pr_froz_c:.4f} | — |")
    md.append(f"| µ_c (visc) | {cea.visc_c:.4e} | Pa·s |")
    md.append(f"| Isp_vac (exit) | {cea.Isp_vac_e:.1f} | m/s |")
    md.append("")

    # ===== Geometry =======================================================
    md.append("## 4. Computed engine geometry")
    md.append("")
    md.append("| Quantity | Value | Units |")
    md.append("|---|---|---|")
    md.append(f"| Throat dia D_t | {2*geom.R_t*1000:.3f} | mm |")
    md.append(f"| Chamber dia D_c | {2*geom.R_c*1000:.3f} | mm |")
    md.append(f"| Exit dia D_e | {2*geom.R_e*1000:.3f} | mm |")
    md.append(f"| Throat area A_t | {geom.A_t*1e6:.3f} | mm² |")
    md.append(f"| Chamber area A_c | {geom.A_c*1e6:.3f} | mm² |")
    md.append(f"| Exit area A_e | {geom.A_e*1e6:.3f} | mm² |")
    md.append(f"| Chamber length L_c | {geom.L_c*1000:.2f} | mm |")
    md.append(f"| Nozzle length L_n | {geom.L_nozzle*1000:.2f} | mm |")
    md.append(f"| Total length | {L_total*1000:.2f} | mm |")
    md.append(f"| R_chamber arc | {geom.R_chamber*1000:.2f} | mm |")
    md.append(f"| R_throat_conv | {geom.R_throat_conv*1000:.2f} | mm |")
    md.append(f"| R_throat_div  | {geom.R_throat_div*1000:.2f} | mm |")
    md.append(f"| Total mass flow | {geom.mdot:.4f} | kg/s |")
    md.append(f"| Fuel mass flow | {geom.mdot_fuel:.4f} | kg/s |")
    md.append(f"| Ox mass flow | {geom.mdot_ox:.4f} | kg/s |")
    md.append(f"| Exit Mach | {geom.M_exit:.3f} | — |")
    md.append(f"| Exit static pressure | {geom.P_exit/1000:.2f} | kPa |")
    md.append("")

    # ===== Channel geometry summary ======================================
    md.append("## 5. Cooling channel layout")
    md.append("")
    md.append(f"- N_throat = {cfg.N_channels_throat}, "
              f"N_chamber = {cfg.N_channels_chamber}, "
              f"split radius = {cfg.channel_split_r_ratio} × R_t")
    md.append(f"- Channel width (constant): "
              f"{chan_geom.chan_w[0]*1000:.3f} mm")
    md.append(f"- Wall thickness (constant): "
              f"{chan_geom.chan_t[0]*1000:.3f} mm")
    md.append(f"- Channel height taper: "
              f"chamber {cfg.chan_h_chamber*1000:.2f} mm → "
              f"throat {cfg.chan_h_throat*1000:.2f} mm → "
              f"exit {cfg.chan_h_exit*1000:.2f} mm")
    md.append("")
    md.append("Channel dimensions at key stations:")
    md.append("")
    md.append("| Station | x [mm] | r_wall [mm] | N | w [mm] | h [mm] | "
              "land [mm] | t_wall [mm] |")
    md.append("|---|---|---|---|---|---|---|---|")
    for lbl, x_key in [("Injector face", 0.0),
                       ("Throat",        geom.L_c),
                       ("Nozzle exit",   L_total),
                       ("T_hw peak",     x_peak)]:
        ix   = int(np.argmin(np.abs(chan_geom.x_j - x_key)))
        r_x  = nozzle_radius(float(chan_geom.x_j[ix]), geom, cfg.dx)
        N_ix = chan_geom.n_chan[ix] if hasattr(chan_geom, "n_chan") else float("nan")
        md.append(f"| {lbl} | {chan_geom.x_j[ix]*1000:.1f} | "
                  f"{r_x*1000:.2f} | {N_ix:.1f} | "
                  f"{chan_geom.chan_w[ix]*1000:.3f} | "
                  f"{chan_geom.chan_h[ix]*1000:.3f} | "
                  f"{chan_geom.chan_land[ix]*1000:.3f} | "
                  f"{chan_geom.chan_t[ix]*1000:.3f} |")
    md.append("")

    # ===== Peak summary ===================================================
    md.append("## 6. Peak / extreme values")
    md.append("")
    md.append("| Quantity | Value | Location |")
    md.append("|---|---|---|")
    md.append(f"| max T_hw | {th.T_hw.max():.1f} K "
              f"({th.T_hw.max()-273.15:.0f}°C) | "
              f"x = {x_arr[i_peak]*1000:.1f} mm |")
    md.append(f"| max T_cw | {th.T_cw.max():.1f} K | "
              f"x = {x_arr[int(np.argmax(th.T_cw))]*1000:.1f} mm |")
    md.append(f"| max heat flux | {th.heatflux.max()/1e6:.2f} MW/m² | "
              f"x = {x_arr[int(np.argmax(th.heatflux))]*1000:.1f} mm |")
    md.append(f"| max h_gas | {th.h_gas.max():.1f} W/(m²·K) | "
              f"x = {x_arr[int(np.argmax(th.h_gas))]*1000:.1f} mm |")
    md.append(f"| max T_coolant | {th.T_coolant.max():.1f} K | "
              f"x = {x_arr[int(np.argmax(th.T_coolant))]*1000:.1f} mm |")
    md.append(f"| min P_coolant | {th.P_coolant.min()/1e5:.2f} bar | "
              f"x = {x_arr[int(np.argmin(th.P_coolant))]*1000:.1f} mm |")
    md.append(f"| ΔT_wall (T_hw-T_cw) max | "
              f"{(th.T_hw - th.T_cw).max():.1f} K | "
              f"x = {x_arr[int(np.argmax(th.T_hw - th.T_cw))]*1000:.1f} mm |")
    md.append(f"| Wall T limit (config) | {cfg.wall_melt_T:.0f} K | — |")
    md.append(f"| Margin (limit - max T_hw) | "
              f"{cfg.wall_melt_T - th.T_hw.max():.1f} K | — |")
    md.append("")

    # ===== Axial-station table ===========================================
    md.append("## 7. Axial-station table")
    md.append("")
    md.append("Counter-current coolant: T_coolant *decreases* with x (it enters "
              "at exit and leaves at injector). Indices marked * are the "
              "additional rows added for injector / throat / exit / T_hw peak.")
    md.append("")
    cols = ["x [mm]", "r_wall [mm]", "M", "P_gas [kPa]", "T_aw [K]",
            "h_gas [kW/m²K]", "q_w [MW/m²]", "T_hw [K]", "T_cw [K]",
            "T_cool [K]", "P_cool [bar]", "h_cool [kW/m²K]",
            "Re_cool [1e3]", "v_cool [m/s]"]
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    extra_idx = set(int(np.argmin(np.abs(x_arr - xv))) for xv in extra
                    if xv is not None)
    for i in idxs:
        marker = "*" if i in extra_idx else ""
        r_w = nozzle_radius(float(x_arr[i]), geom, cfg.dx)
        md.append(
            f"| {x_arr[i]*1000:6.1f}{marker} | {r_w*1000:6.2f} | "
            f"{M_th[i]:5.3f} | {P_th[i]/1000:7.1f} | {T_aw[i]:6.0f} | "
            f"{th.h_gas[i]/1000:6.2f} | {th.heatflux[i]/1e6:5.2f} | "
            f"{th.T_hw[i]:6.0f} | {th.T_cw[i]:6.0f} | "
            f"{th.T_coolant[i]:6.0f} | {th.P_coolant[i]/1e5:5.2f} | "
            f"{th.h_coolant[i]/1000:6.2f} | "
            f"{th.Re_coolant[i]/1000:6.1f} | {th.v_coolant[i]:5.1f} |"
        )
    md.append("")

    # ===== Model excerpts =================================================
    md.append("## 8. Model excerpts (verbatim from `heat_transfer.py`)")
    md.append("")
    md.append("### `_bartz_h` — gas-side HTC")
    md.append("")
    md.append("```python")
    md.append(inspect.getsource(heat_transfer._bartz_h).rstrip())
    md.append("```")
    md.append("")
    md.append("### `_T_aw` — adiabatic wall (recovery) temperature")
    md.append("")
    md.append("```python")
    md.append(inspect.getsource(heat_transfer._T_aw).rstrip())
    md.append("```")
    md.append("")

    # ===== Notes for reviewer ============================================
    md.append("## 9. What I want the reviewer to look at")
    md.append("")
    md.append("1. Is the `_bartz_h` formulation (Bartz 1957 with σ correction) "
              "correctly implemented? Are units consistent?")
    md.append("2. Recovery factor in `_T_aw` uses `r = Pr_froz^(1/3)` — "
              "appropriate for turbulent BL?")
    md.append("3. Coolant-side: is the Nu correlation a reasonable choice "
              "for RP-1 in narrow rectangular SLM channels at these Re?")
    md.append("4. Is the wall model self-consistent? With `wall_2d=True` "
              "the 2-D conduction should match a 1-D thin-wall estimate "
              "(`q = k(T_hw-T_cw)/t_wall`) at chamber stations where the "
              "fin effect is small.")
    md.append("5. Compare peak T_hw to physical intuition — for a copper "
              "(k=300 W/m·K, t=1 mm) wall at q≈30–60 MW/m² with subcooled "
              "RP-1 at ~300 K and h_cool ~10–30 kW/m²K, what range of "
              "T_hw is expected?")
    md.append("")
    md.append(f"_Generated: {os.popen('date').read().strip()}_")
    md.append("")

    text = "\n".join(md)
    with open(OUT_PATH, "w") as f:
        f.write(text)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    n_lines = text.count("\n")
    print(f"\nWrote {OUT_PATH}  ({size_kb:.1f} KB, {n_lines} lines)")
    print(f"  Peak T_hw: {th.T_hw.max():.1f} K at x = "
          f"{x_arr[i_peak]*1000:.1f} mm")
    print(f"  Wall T limit: {cfg.wall_melt_T:.0f} K  "
          f"(margin {cfg.wall_melt_T - th.T_hw.max():.1f} K)")


if __name__ == "__main__":
    main_report()
