"""
film_cooling.py
Liquid + gaseous film cooling model following the RPA thermal analysis
methodology (Ponomarenko 2012) and Vasiliev & Kudryavtsev (1993).

Phase 1 — Liquid heating  (x_inject → x_sat)
    Film enters at T_film_inlet, heated by gas-side convection (Bartz HTC).
    Includes film stability coefficient η(Re_f) per RPA Figure 2 / [1].
    Heating:  dTf = 2π·R·q / (η_stab · ṁf · c̄f) · dx
    Wall sees T_aw_eff = T_film

Phase 2 — Vaporisation  (x_sat → x_vap)
    Film locked at T_sat, gas heat evaporates liquid.
    Rate:  dṁf = 2π·R·q / Q_vap · dx
    Wall sees T_aw_eff = T_sat

Phase 3 — Gaseous film mixing  (x_vap → x_end)
    Vasiliev-Kudryavtsev (1993) turbulent mixing model [1]:
        ξ = 1 - exp(-M · s / Hs)
        M = Kt · m̄s / m̄f
    where:
        Kt   = turbulent mixing intensity, (0.05–0.20)×10⁻²  [1]
        m̄s  = surface layer relative mass flow (≈ 1 for small film fractions)
        m̄f  = film relative mass flow = ṁ_film / ṁ_total
        Hs   = surface layer thickness ≈ turbulent BL thickness
             = 0.37 · x / Re_x^0.2   (flat-plate turbulent BL estimate)

    Film effectiveness η = 1 - ξ = exp(-M · s / Hs)
    T_aw_eff = η · T_sat + (1-η) · T_aw_bare

    The 1/m̄f scaling in M is the key physics: more film mass → lower M
    → slower mixing → longer protection.  No curve-fitting required.

References
----------
[1] Vasiliev A.P., Kudryavtsev V.M. et al. "Basics of theory and analysis
    of liquid-propellant rocket engines", vol.2, 4th Ed., Moscow, 1993.
[2] Ponomarenko A. "RPA: Thermal Analysis of Thrust Chambers", June 2012.

Entry point
-----------
    compute_film_taw(flow, geom, cea, config) → np.ndarray  shape (n,)
"""

import numpy as np

from flow_solver import FlowSolution
from geometry import EngineGeometry, nozzle_radius
from cea_interface import CEAResult
from config import EngineConfig
from heat_transfer import _bartz_h


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
# Film stability coefficient η(Re_f) — RPA Figure 2 / Vasiliev [1]
# ---------------------------------------------------------------------------
def _film_stability(mdot_film: float, r_wall: float, mu_film: float) -> float:
    """
    Film stability coefficient η from RPA Figure 2.

    Accounts for film breakup at low Reynolds numbers — a thin, slow film
    is less stable and absorbs heat less effectively.

    Re_f = ṁ_film / (2π·R · μ_film)

    Fit to Figure 2:
        η ≈ 0.4 + 0.4·(1 - exp(-Re_f/1500))
        Range: 0.4 (laminar, unstable) → 0.8 (turbulent, stable)
    """
    circ = 2.0 * np.pi * r_wall
    Re_f = mdot_film / (circ * mu_film)
    Re_f = max(Re_f, 1.0)
    return 0.4 + 0.4 * (1.0 - np.exp(-Re_f / 1500.0))


# ---------------------------------------------------------------------------
# Turbulent BL thickness estimate
# ---------------------------------------------------------------------------
def _bl_thickness(x: float, cea: CEAResult, M_local: float) -> float:
    """
    Estimate turbulent boundary layer thickness at axial station x [m].

    Uses flat-plate turbulent BL:  δ = 0.37 · x / Re_x^0.2

    Local gas properties estimated from isentropic relations + CEA chamber
    conditions.
    """
    if x <= 0.0:
        return 1e-6

    gam   = cea.gamma_c
    R_sp  = cea.R_specific
    T_c   = cea.T_c
    P_c   = cea.P_c
    mu_c  = cea.visc_c

    # Local isentropic conditions
    fac     = 1.0 + (gam - 1.0) / 2.0 * M_local**2
    T_local = T_c / fac
    P_local = P_c * fac**(- gam / (gam - 1.0))
    rho     = P_local / (R_sp * T_local)
    a_local = np.sqrt(gam * R_sp * T_local)
    v_local = M_local * a_local

    # Viscosity: Sutherland-like scaling from chamber value
    # μ ∝ T^0.7 (approximate for combustion gases)
    mu_local = mu_c * (T_local / T_c)**0.7

    Re_x = rho * v_local * x / mu_local
    Re_x = max(Re_x, 1.0)

    Hs = 0.37 * x / Re_x**0.2
    return max(Hs, 1e-6)


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

    Implements the RPA thermal analysis film cooling model:
    - Liquid phases (heating + vaporisation) per Ponomarenko 2012 p.10
    - Gaseous mixing per Vasiliev & Kudryavtsev 1993, as in RPA p.8-9

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

    # Film mass flow [kg/s] — fraction of coolant flow
    mdot_film = config.film_fraction * (config.mdot_coolant or 0.0)
    if mdot_film <= 0.0:
        return T_aw_eff

    fluid         = config.film_coolant
    T_film_inlet  = config.film_T_inlet
    x_inject      = config.film_inject_x

    # Find injection station index
    i_start = int(np.searchsorted(flow.x, x_inject))
    i_start = min(i_start, n - 1)

    # Saturation properties at chamber pressure
    T_sat = _get_T_sat(fluid, config.P_coolant_inlet)
    L_lv  = _get_L_lv(fluid, T_sat, config.P_coolant_inlet)

    T_film   = float(T_film_inlet)
    m_film   = float(mdot_film)
    phase    = "liquid"
    x_vap    = None

    # Gaseous mixing parameters (Vasiliev-Kudryavtsev model)
    Kt      = config.film_Kt                    # turbulent mixing intensity
    m_bar_f = mdot_film / geom.mdot              # film relative mass flow
    m_bar_s = 1.0 - m_bar_f                      # surface layer ≈ rest of flow

    # Approximate liquid film viscosity for stability coefficient
    mu_film_liq = 3.0e-4   # RP-1 at ~400K [Pa·s] (order-of-magnitude)

    # Approximate liquid Cp
    Cp_film = 2100.0   # RP-1 [J/(kg·K)]

    print(f"\n--- Film Cooling ---")
    print(f"  mdot_film = {mdot_film*1000:.2f} g/s  "
          f"({config.film_fraction*100:.1f}% of coolant)")
    print(f"  Injection at x = {x_inject*1000:.1f} mm  "
          f"T_inlet = {T_film_inlet:.1f} K  T_sat = {T_sat:.1f} K")
    print(f"  Kt = {Kt:.4f}  m̄f = {m_bar_f:.4f}  "
          f"M = Kt·m̄s/m̄f = {Kt * m_bar_s / m_bar_f:.2f}")

    for i in range(i_start, n):
        x   = float(flow.x[i])
        M   = float(flow.M[i])
        r   = nozzle_radius(x, geom, dx)
        T_aw_i = T_aw_gas[i]

        if phase == "spent":
            break

        if phase == "liquid":
            # --- Phase 1: Liquid heating (RPA p.10) ---
            # Bartz HTC drives heat into film (dominant thermal resistance)
            A   = float(flow.A[i])
            h_g = _bartz_h(M, A, T_film, cea, geom)
            q_f = h_g * (T_aw_i - T_film)
            q_f = max(q_f, 0.0)

            # Wall sees film temperature
            T_aw_eff[i] = T_film

            # Film stability coefficient η (RPA Figure 2)
            eta_stab = _film_stability(m_film, r, mu_film_liq)

            # Heat film:  dTf = 2πR·q / (η·ṁf·Cp) · dx
            circ       = 2.0 * np.pi * r
            dT_film_dx = circ * q_f / (eta_stab * m_film * Cp_film)
            T_film    += dT_film_dx * dx

            if T_film >= T_sat:
                T_film = T_sat
                phase  = "vapour"

        elif phase == "vapour":
            # --- Phase 2: Vaporisation (RPA p.11) ---
            A   = float(flow.A[i])
            h_g = _bartz_h(M, A, T_sat, cea, geom)
            q_f = h_g * (T_aw_i - T_sat)
            q_f = max(q_f, 0.0)

            T_aw_eff[i] = T_sat

            # Evaporate:  dṁf = 2πR·q / Q_vap · dx
            circ       = 2.0 * np.pi * r
            dm_film_dx = -circ * q_f / L_lv
            m_film    += dm_film_dx * dx
            m_film     = max(m_film, 0.0)

            if m_film <= 0.0:
                x_vap = x
                phase = "gaseous"
                print(f"  Film fully vaporised at x = {x*1000:.1f} mm")

        elif phase == "gaseous":
            # --- Phase 3: Gaseous mixing (RPA p.8-9) ---
            # Vasiliev-Kudryavtsev (1993) turbulent mixing model:
            #
            #   ξ = 1 - exp(-M · s / Hs)
            #   M = Kt · m̄s / m̄f
            #   Hs = turbulent BL thickness at station x
            #
            # Film effectiveness η = 1 - ξ
            # More film → larger m̄f → smaller M → slower mixing

            s  = x - x_vap
            Hs = _bl_thickness(x, cea, M)

            M_mix = Kt * m_bar_s / m_bar_f
            xi    = 1.0 - np.exp(-M_mix * s / Hs)
            eta   = 1.0 - xi   # = exp(-M_mix · s / Hs)
            eta   = float(np.clip(eta, 0.0, 1.0))

            # Modified adiabatic wall temperature
            T_aw_eff[i] = eta * T_sat + (1.0 - eta) * T_aw_i

            if eta < 0.01:
                phase = "spent"
                print(f"  Gaseous film spent (η<1%) at x = {x*1000:.1f} mm")

    # Summary
    reduction = np.mean(T_aw_gas[i_start:] - T_aw_eff[i_start:])
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
