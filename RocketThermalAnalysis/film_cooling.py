"""
film_cooling.py
Liquid + gaseous film cooling model following the RPA thermal analysis
methodology (Ponomarenko 2012) and Vasiliev & Kudryavtsev (1993).

Phase 1 — Liquid heating  (x_inject → x_sat)
    Film enters at T_film_inlet, heated by gas-side convection (Bartz HTC
    evaluated at T_f).  Per Ponomarenko RPA_ThermalAnalysis.pdf p.10:
        "the only component of the total heat flux reaching the wall is the
         radiation heat flux, whereas the convective component heats the
         liquid coolant. The convective heat transfer from the liquid
         coolant to the wall may be neglected."
    Heating ODE:    dT_f/dx = 2π·R·q_w^{T_f} / (η · ṁ_f · c̄_f)
        q_w^{T_f} = h_g(T_f) · (T_aw_bare − T_f)
        η        = film stability from Fig 2 [1] (0.4–0.8)  — DIVIDES ṁ_f
                   (reflects effective liquid mass doing cooling work).
    Wall convective q set to ZERO for regen path (suppress mask).

Phase 2 — Vaporisation  (x_sat → x_vap)
    Film locked at T_sat, all gas-side flux goes into latent heat.
    Ponomarenko RPA p.11:
        dṁ_f/dx = 2π·R·q_w^{T_sat} / Q_vap
        q_w^{T_sat} = h_g(T_sat) · (T_aw_bare − T_sat)
    η not used here (per Ponomarenko Eq.).
    Wall convective q set to ZERO for regen path (suppress mask).

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
    compute_film_taw(flow, geom, cea, config) → (T_aw_eff, OF_eff, phase_code)
        T_aw_eff   : np.ndarray shape (n,)
                     Meaningful only where phase=0 or phase=3 (regen wall
                     sees this driving temperature).  Where phase ∈ {1,2}
                     the wall convective flux is suppressed entirely, so
                     this value is irrelevant at those stations (set to
                     bare T_aw for safety).
        OF_eff     : np.ndarray shape (n,) — surface-layer effective O/F for
                     Bartz property lookup (only meaningful where phase=3)
        phase_code : np.ndarray[int] shape (n,)
                     0 = no film (pre-injection or spent)
                     1 = liquid film on wall (wall q_conv = 0, film absorbs)
                     2 = vapour film on wall (wall q_conv = 0, vaporising)
                     3 = gaseous mixing (equilibrium CEA at OF_eff)
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
# Spalding B-number transpiration correction for vaporising films
# ---------------------------------------------------------------------------
def _transpiration_blowing(T_aw: float, T_sat: float,
                           Cp_gas: float, L_eff: float) -> float:
    """
    Transpiration correction h/h₀ for a vaporising liquid film.

    Uses the Spalding B-number formulation (standard in combustion science):

        h / h₀ = ln(1 + B) / B

        B = Cp_gas · (T_aw − T_sat) / L_eff

    where L_eff = L_vap + L_pyrolysis is the *effective* enthalpy sink at the
    film surface.  For kerosene-class fuels, endothermic cracking/pyrolysis
    absorbs 300–600 kJ/kg on top of the ~250 kJ/kg latent heat (Edwards 2003,
    Ward et al. 2004), giving L_eff ~ 550–850 kJ/kg.

    This replaces Grisson's K_M molecular weight correction which drives
    h/h₀ → 0.07 for RP-1 (M_vap=170, M_gas=21) — far too aggressive because
    RP-1 doesn't evaporate as a single heavy species; it cracks into light
    fragments (C2-C4) at the surface.  The Spalding B formulation sidesteps
    the MW issue entirely and captures the physics through L_eff.

    Parameters
    ----------
    T_aw   : bare adiabatic wall temperature [K]
    T_sat  : saturation temperature of the coolant [K]
    Cp_gas : gas-side Cp [J/(kg·K)]
    L_eff  : effective enthalpy sink = L_vap + L_pyrolysis [J/kg]

    Returns
    -------
    h_ratio : h/h₀ ∈ (0, 1].  Multiply h_gas_bare by this.
    """
    dT = T_aw - T_sat
    if dT <= 0.0 or L_eff <= 0.0:
        return 1.0

    B = Cp_gas * dT / L_eff

    if B < 1e-6:
        return 1.0
    return np.log(1.0 + B) / B


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
                     config: EngineConfig):
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

    # Core OF (propellant OF excluding film) — used as the baseline surface-layer
    # composition. In the absence of film mixing, OF_eff = OF_core everywhere.
    OF_core = float(config.OF) if config.OF is not None else 2.0
    OF_eff_arr = np.full(n, OF_core)
    phase_code = np.zeros(n, dtype=np.int8)   # 0 everywhere by default

    # Bare T_aw at each station (no film)
    T_aw_gas = np.array([_T_aw_bare(float(flow.M[i]), cea) for i in range(n)])
    T_aw_eff = T_aw_gas.copy()

    # Short-circuit if film cooling not configured
    if config.film_fraction <= 0.0:
        return T_aw_eff, OF_eff_arr, phase_code

    # Film mass flow [kg/s] — fraction of coolant flow
    mdot_film = config.film_fraction * (config.mdot_coolant or 0.0)
    if mdot_film <= 0.0:
        return T_aw_eff, OF_eff_arr, phase_code

    # Diagnostic trackers for film heating / vaporisation phase lengths
    x_heat_end = None   # x where T_f reaches T_sat
    x_vap_end  = None   # x where ṁ_f → 0

    # Surface-layer mass budget for RPA's "Relative thickness of near-wall layer"
    # knob.  ṁ_s = δ_rel · ṁ_total.  Film vapour mixing into this layer shifts
    # local OF toward fuel-rich, which changes Bartz property group μ^0.2·Cp/Pr^0.6.
    delta_rel    = max(float(config.film_BL_thickness), 1e-6)
    mdot_s       = delta_rel * geom.mdot
    mdot_film_0  = mdot_film   # initial film flow (kg/s)

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
    eta_stab_peak = 0.0     # track peak circumferential coverage during film lifetime

    # Gaseous mixing parameters (Vasiliev-Kudryavtsev model)
    Kt      = config.film_Kt                    # turbulent mixing intensity
    m_bar_f = mdot_film / geom.mdot              # film relative mass flow
    m_bar_s = 1.0 - m_bar_f                      # surface layer ≈ rest of flow

    # Approximate liquid film viscosity for stability coefficient
    mu_film_liq = 3.0e-4   # RP-1 at ~400K [Pa·s] (order-of-magnitude)

    # Approximate liquid Cp
    Cp_film = 2100.0   # RP-1 [J/(kg·K)]

    # Gas-side Cp for Spalding B transpiration parameter
    Cp_gas = cea.Cp_froz_c   # [J/(kg·K)]

    # Effective enthalpy sink: latent heat + endothermic pyrolysis
    L_eff = L_lv + config.film_L_pyrolysis

    print(f"\n--- Film Cooling ---")
    print(f"  mdot_film = {mdot_film*1000:.2f} g/s  "
          f"({config.film_fraction*100:.1f}% of coolant)")
    print(f"  Injection at x = {x_inject*1000:.1f} mm  "
          f"T_inlet = {T_film_inlet:.1f} K  T_sat = {T_sat:.1f} K")
    print(f"  Kt = {Kt:.4f}  m̄f = {m_bar_f:.4f}  "
          f"M = Kt·m̄s/m̄f = {Kt * m_bar_s / m_bar_f:.2f}")

    # Transpiration diagnostic (chamber-average values)
    _B_diag = Cp_gas * (cea.T_c * 0.98 - T_sat) / L_eff
    _hh0_diag = np.log(1.0 + _B_diag) / _B_diag if _B_diag > 1e-6 else 1.0
    print(f"  Spalding transpiration: L_eff={L_eff/1e3:.0f} kJ/kg  "
          f"(L_vap={L_lv/1e3:.0f} + L_pyro={config.film_L_pyrolysis/1e3:.0f})  "
          f"B={_B_diag:.1f}  h/h₀={_hh0_diag:.4f}")

    for i in range(i_start, n):
        x   = float(flow.x[i])
        M   = float(flow.M[i])
        r   = nozzle_radius(x, geom, dx)
        T_aw_i = T_aw_gas[i]

        if phase == "spent":
            break

        if phase == "liquid":
            # --- Phase 1: Liquid heating ---
            # Film enters subcooled, heated by gas-side convection.
            # η (film stability, RPA Fig.2) = fraction of circumference with
            # stable liquid film.  That fraction shields the wall; the rest
            # (1-η) sees bare T_aw.
            #
            # Film heating ODE (on the η-covered fraction):
            #   dT_f/dx = 2πR · η · h_g · (T_aw − T_f) / (η · ṁ_f · c̄_f)
            #           = 2πR · h_g · (T_aw − T_f) / (ṁ_f · c̄_f)
            #
            # Wall driving temperature (circumferential average):
            #   T_aw_eff = η·T_film + (1-η)·T_aw_bare
            eta_stab = _film_stability(m_film, r, mu_film_liq)
            eta_stab_peak = max(eta_stab_peak, eta_stab)

            A   = float(flow.A[i])
            h_g = _bartz_h(M, A, T_film, cea, geom)
            q_f = h_g * (T_aw_i - T_film)
            q_f = max(q_f, 0.0)

            # Wall sees blended T_aw: film-covered fraction at T_film,
            # exposed fraction at bare T_aw.
            T_aw_eff[i]   = eta_stab * T_film + (1.0 - eta_stab) * T_aw_i
            OF_eff_arr[i] = OF_core
            phase_code[i] = 1

            circ       = 2.0 * np.pi * r
            dT_film_dx = circ * q_f / (m_film * Cp_film)
            T_film    += dT_film_dx * dx

            if T_film >= T_sat:
                T_film     = T_sat
                x_heat_end = x
                phase      = "vapour"
                print(f"  Film reaches T_sat at x = {x*1000:.1f} mm")

        elif phase == "vapour":
            # --- Phase 2: Vaporisation ---
            # Film pinned at T_sat.  Gas-side heat on the η-covered fraction
            # goes into latent heat (vaporisation); the (1-η) fraction hits
            # the wall at bare T_aw.
            #
            # Spalding transpiration correction: h_eff = h_g · ln(1+B)/B.
            # Vapour leaving the film surface thickens the viscous sublayer.
            # B = Cp_gas·ΔT / L_eff, where L_eff = L_vap + L_pyrolysis.
            #
            # Vaporisation ODE (η-covered fraction):
            #   dṁ_f/dx = -2πR · η · h_eff · (T_aw − T_sat) / L_lv
            #
            # Wall driving temperature (circumferential average):
            #   T_aw_eff = η·T_sat + (1-η)·T_aw_bare
            eta_stab = _film_stability(m_film, r, mu_film_liq)
            eta_stab_peak = max(eta_stab_peak, eta_stab)

            A   = float(flow.A[i])
            h_g = _bartz_h(M, A, T_sat, cea, geom)

            # Spalding transpiration correction
            h_ratio = _transpiration_blowing(T_aw_i, T_sat, Cp_gas, L_eff)
            h_eff   = h_g * h_ratio

            q_f = h_eff * (T_aw_i - T_sat)
            q_f = max(q_f, 0.0)

            # Wall sees blended T_aw: film-covered at T_sat, exposed at bare T_aw.
            T_aw_eff[i]   = eta_stab * T_sat + (1.0 - eta_stab) * T_aw_i
            OF_eff_arr[i] = OF_core
            phase_code[i] = 2

            circ       = 2.0 * np.pi * r
            dm_film_dx = -circ * eta_stab * q_f / L_lv
            m_film    += dm_film_dx * dx
            m_film     = max(m_film, 0.0)

            if m_film <= 0.0:
                x_vap     = x
                x_vap_end = x
                phase     = "gaseous"
                print(f"  Film fully vaporised at x = {x*1000:.1f} mm  "
                      f"(η_stab_peak={eta_stab_peak:.3f})")

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

            M_mix   = Kt * m_bar_s / m_bar_f
            eta_gas = float(np.exp(-M_mix * s / Hs))
            # Gaseous film starts with the peak circumferential coverage
            # from the liquid/vapour phase (η_stab_peak) and decays via
            # turbulent mixing.  This ensures continuity: at s=0, gaseous
            # η matches the blending that was active during vaporisation.
            eta     = float(np.clip(eta_stab_peak * eta_gas, 0.0, 1.0))

            # Modified adiabatic wall temperature
            T_aw_eff[i] = eta * T_sat + (1.0 - eta) * T_aw_i

            # Surface-layer OF_eff:
            # Film vapour in the surface layer decays as η → 0 (Vasiliev
            # turbulent mixing dilutes film into the surface layer).  Capped at
            # the available layer mass ṁ_s so OF_eff can't go below the
            # pure-film limit.
            m_fv   = min(mdot_film_0 * eta, mdot_s)
            w      = m_fv / mdot_s                     # film mass fraction of layer
            denom  = (1.0 - w) + w * (1.0 + OF_core)
            OF_eff_arr[i] = max((1.0 - w) * OF_core / denom, 1.0)  # clamp to LUT min
            phase_code[i] = 3

            if eta < 0.01:
                phase = "spent"
                print(f"  Gaseous film spent (η<1%) at x = {x*1000:.1f} mm")

    # Summary
    reduction = np.mean(T_aw_gas[i_start:] - T_aw_eff[i_start:])
    print(f"  Mean T_aw reduction (film zone): {reduction:.1f} K")
    print(f"  Phase at exit: {phase}")
    of_min = float(np.min(OF_eff_arr[i_start:]))
    print(f"  Surface-layer OF_eff range (film zone): "
          f"{of_min:.3f} → {OF_core:.3f}  (δ_rel={delta_rel:.3f})")

    return T_aw_eff, OF_eff_arr, phase_code


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
