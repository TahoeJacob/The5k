"""
film_cooling.py
Liquid + gaseous film cooling model.

Physics (following EUCASS2023-035, Barredo Juan & López Platero, and
NASA SP-8124 Appendix B):

Phase 1 — Liquid heating  (x_inject → x_sat)
    Film enters at T_film_inlet, is heated by gas convection.
    Heat flux to film:  q_film = h_film * (T_aw - T_film)
    Film heats up:      dT_film/dx = 2π·r·q_film / (m_film·Cp_film)
    Wall sees T_aw_eff = T_film  (gas cannot heat wall through liquid film)

Phase 2 — Vaporisation  (x_sat → x_vap)
    Film temperature locked at T_sat.
    Evaporation rate:   dm_film/dx = -2π·r·q_film / L_lv
    Wall sees T_aw_eff = T_sat

Phase 3 — Gaseous film  (x_vap → x_end)
    Vapour mixes with free-stream boundary layer.
    Film cooling efficiency η decays with distance via NASA Appendix B mixing:
        η(x) = 1 / (θ · (1 + W_E/W_e))
    Simplified here as exponential decay calibrated against typical K_t=0.001:
        η(x) = exp(-K_mix · (s/r_local))   where s = x - x_vap
    Modified adiabatic wall temperature (EUCASS Eq. 31):
        T_aw_eff = [H_aw - η·H_c,v + η·Cp_v·T_if + (1-η)·(Cp_s·T0g - h_g)]
                   / [η·Cp_v + (1-η)·Cp_s]
    Simplified (ideal gas, no dissociation):
        T_aw_eff = η · T_vap_partial + (1-η) · T_aw_gas

When film is exhausted or never injected, T_aw_eff = T_aw_gas (no correction).

Film HTC — Grisson (1966) correlation for annular liquid film:
    h_film = 0.025 · (k_l/D) · Re_film^0.8 · Pr_l^0.4
    Re_film = 4 · m_film / (2π·r · μ_l)    (film Reynolds based on perimeter)

Entry point
-----------
    compute_film_taw(flow, geom, cea, config) → np.ndarray  shape (n,)
    Returns T_aw_eff at every axial station in flow.x order.
    Stations with no film protection return the bare Bartz T_aw.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from flow_solver import FlowSolution
from geometry import EngineGeometry, nozzle_radius
from cea_interface import CEAResult
from config import EngineConfig
from coolant_props import get_coolant_props


# ---------------------------------------------------------------------------
# Bare Bartz adiabatic wall temperature (mirrors heat_transfer._T_aw)
# ---------------------------------------------------------------------------
def _T_aw_bare(M: float, cea: CEAResult) -> float:
    """Adiabatic wall temperature without film [K]."""
    gam = cea.gamma_c
    r   = cea.Pr_froz_c**(1.0 / 3.0)
    N   = M**2
    return cea.T_c * (1.0 + r * (gam - 1.0) / 2.0 * N) / (1.0 + (gam - 1.0) / 2.0 * N)


# ---------------------------------------------------------------------------
# Grisson (1966) film HTC
# ---------------------------------------------------------------------------
def _grisson_h_film(m_film: float, r_wall: float,
                    props_l) -> float:
    """
    Liquid film HTC [W/(m²·K)] via Grisson (1966).

    h = 0.025 · (k_l / D) · Re_film^0.8 · Pr_l^0.4

    where Re_film = 4·m_film / (circumference · μ_l)
    and D = 2·r_wall (hydraulic diameter ≈ 2× wall radius for thin film).
    """
    circ   = 2.0 * np.pi * r_wall
    D      = 2.0 * r_wall
    Re_f   = 4.0 * m_film / (circ * props_l.viscosity)
    Pr_f   = props_l.viscosity * props_l.Cp / props_l.conductivity
    Re_f   = max(Re_f, 1.0)
    h_film = 0.025 * (props_l.conductivity / D) * Re_f**0.8 * Pr_f**0.4
    return float(h_film)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute_film_taw(flow:   FlowSolution,
                     geom:   EngineGeometry,
                     cea:    CEAResult,
                     config: EngineConfig) -> np.ndarray:
    """
    Compute effective adiabatic wall temperature T_aw_eff [K] at every
    axial station, accounting for liquid + gaseous film cooling.

    Parameters
    ----------
    flow   : FlowSolution from solve_flow()
    geom   : EngineGeometry from size_engine()
    cea    : CEAResult from get_cea_for_analysis()
    config : EngineConfig — reads film_* fields

    Returns
    -------
    T_aw_eff : np.ndarray, shape (n,)
        At stations before injection or where film is spent, equals bare T_aw.
    """
    n   = len(flow.x)
    dx  = config.dx

    # Bare T_aw at each station (no film)
    T_aw_gas = np.array([_T_aw_bare(float(flow.M[i]), cea) for i in range(n)])
    T_aw_eff = T_aw_gas.copy()

    # Short-circuit if film cooling not configured
    if config.film_fraction <= 0.0:
        return T_aw_eff

    # Film mass flow [kg/s] — fraction of total fuel flow
    mdot_film = config.film_fraction * (config.mdot_coolant or 0.0)
    if mdot_film <= 0.0:
        return T_aw_eff

    fluid         = config.film_coolant
    T_film_inlet  = config.film_T_inlet
    x_inject      = config.film_inject_x

    # Find injection station index
    i_start = int(np.searchsorted(flow.x, x_inject))
    i_start = min(i_start, n - 1)

    # Get saturation temperature at chamber pressure
    # (approximate: use inlet pressure as representative; film is thin)
    T_sat = _get_T_sat(fluid, config.P_coolant_inlet)
    L_lv  = _get_L_lv(fluid, T_sat, config.P_coolant_inlet)

    T_film   = float(T_film_inlet)
    m_film   = float(mdot_film)
    phase    = "liquid"   # "liquid" → "vapour" → "gaseous" → "spent"
    x_vap    = None       # x at which vaporisation completes

    print(f"\n--- Film Cooling ---")
    print(f"  mdot_film = {mdot_film*1000:.2f} g/s  "
          f"({config.film_fraction*100:.1f}% of coolant)")
    print(f"  Injection at x = {x_inject*1000:.1f} mm  "
          f"T_inlet = {T_film_inlet:.1f} K  T_sat = {T_sat:.1f} K")

    for i in range(i_start, n):
        x   = float(flow.x[i])
        M   = float(flow.M[i])
        r   = nozzle_radius(x, geom, dx)
        T_aw_i = T_aw_gas[i]

        if phase == "spent":
            # No film — bare T_aw already set
            break

        if phase == "liquid":
            props = get_coolant_props(T_film, config.P_coolant_inlet, fluid)
            h_f   = _grisson_h_film(m_film, r, props)
            q_f   = h_f * (T_aw_i - T_film)
            q_f   = max(q_f, 0.0)

            # Wall sees film temperature, not T_aw
            T_aw_eff[i] = T_film

            # Heat film
            circ       = 2.0 * np.pi * r
            dT_film_dx = circ * q_f / (m_film * props.Cp)
            T_film    += dT_film_dx * dx

            if T_film >= T_sat:
                T_film = T_sat
                phase  = "vapour"

        elif phase == "vapour":
            props = get_coolant_props(T_sat, config.P_coolant_inlet, fluid)
            h_f   = _grisson_h_film(m_film, r, props)
            q_f   = h_f * (T_aw_i - T_sat)
            q_f   = max(q_f, 0.0)

            T_aw_eff[i] = T_sat

            # Evaporate film
            circ        = 2.0 * np.pi * r
            dm_film_dx  = -circ * q_f / L_lv
            m_film     += dm_film_dx * dx
            m_film      = max(m_film, 0.0)

            if m_film <= 0.0:
                x_vap = x
                phase = "gaseous"
                print(f"  Film fully vaporised at x = {x*1000:.1f} mm")

        elif phase == "gaseous":
            # Exponential decay of film efficiency with axial distance
            # K_mix calibrated to typical turbulent mixing rate from
            # Vasiliev & Kudryavtsev (1993) K_t = 0.001
            s     = x - x_vap
            K_mix = config.film_K_mix
            eta   = np.exp(-K_mix * s / max(r, 1e-6))
            eta   = float(np.clip(eta, 0.0, 1.0))

            # Modified T_aw: blend between vapour injection temp and bare T_aw
            # Simplified from EUCASS Eq. 31 for ideal gas (no dissociation)
            T_aw_eff[i] = eta * T_sat + (1.0 - eta) * T_aw_i

            if eta < 0.01:
                phase = "spent"
                print(f"  Gaseous film spent (η<1%) at x = {x*1000:.1f} mm")

    # Summary
    i_end_film = np.searchsorted(flow.x, x_inject + 1e-9)
    reduction  = np.mean(T_aw_gas[i_start:] - T_aw_eff[i_start:])
    print(f"  Mean T_aw reduction (film zone): {reduction:.1f} K")
    print(f"  Phase at exit: {phase}")

    return T_aw_eff


# ---------------------------------------------------------------------------
# CoolProp helpers for saturation properties
# ---------------------------------------------------------------------------
def _get_T_sat(fluid: str, P: float) -> float:
    """Saturation temperature at pressure P [Pa]."""
    try:
        import CoolProp.CoolProp as CP
        return float(CP.PropsSI('T', 'P', P, 'Q', 0, fluid))
    except Exception:
        # Fallback: approximate T_sat for RP-1 at ~20 bar ≈ 490 K
        _fallback = {"RP1": 490.0, "Methane": 180.0, "Hydrogen": 25.0}
        return _fallback.get(fluid, 400.0)


def _get_L_lv(fluid: str, T_sat: float, P: float) -> float:
    """Latent heat of vaporisation L_lv [J/kg] at saturation."""
    try:
        import CoolProp.CoolProp as CP
        h_liq = CP.PropsSI('H', 'P', P, 'Q', 0, fluid)
        h_vap = CP.PropsSI('H', 'P', P, 'Q', 1, fluid)
        return float(h_vap - h_liq)
    except Exception:
        _fallback = {"RP1": 250e3, "Methane": 510e3, "Hydrogen": 450e3}
        return _fallback.get(fluid, 300e3)
