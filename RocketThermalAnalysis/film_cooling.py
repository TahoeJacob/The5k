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

Two models are available via `config.film_model`:

  "aedc" — Kutateladze/Stollery with virtual upstream edge, per
           AEDC-TR-91-1 Eq. 3.5 (Morren & Nejad 1991).  Closed-form,
           self-contained, no free Kt.  Derived from turbulent BL mass
           entrainment + calorimetric heat balance:

               η = [ 1 + K_M·(cpg/cpc)·(0.325·(Kt·X + X0)^0.8 − 1) ]^−1

           where X = K·x, K = G·μg^0.25·Mc^(−1.25), X_0 is the virtual
           upstream point from mass balance at injection, K_M is the
           foreign-gas correction (M_c/M_g)^0.14, and Kt = 1 + 10.2·e_t
           is the free-stream turbulence correction.

  "v_k"  — Vasiliev-Kudryavtsev (Ponomarenko 2012 p.9).  Evolves m̄_s,
           m̄_f, M along x.  Has a free Kt knob (published range
           (0.05–0.20)×10⁻²) and depends on a separately-specified
           surface-layer thickness H_s that Ponomarenko does not give
           a formula for.  Matches RPA qualitatively but quantitative
           agreement depends on H_s choice.

        m̄s(ξ) = m̄s⁰·(1−ξ/2) + m̄f⁰·(ξ/2)
        m̄f(ξ) = m̄s⁰·(ξ/2)   + m̄f⁰·(1−ξ/2)
        M(ξ)  = Kt · m̄s(ξ) / m̄f(ξ)          ← CURRENT, not initial
        ξ     = 1 − exp(−M(ξ) · x / Hs)     ← implicit in ξ

    where:
        Kt   = turbulent mixing intensity, (0.05–0.20)×10⁻²  [1]
        m̄s⁰  = initial surface-layer relative mass flow = 1 − m̄f⁰
        m̄f⁰  = film relative mass flow = ṁ_film / ṁ_total
        Hs   = surface-layer thickness ≈ turbulent BL thickness
             = 0.37 · x / Re_x^0.2   (flat-plate turbulent BL estimate)
        x    = distance from FILM INJECTION point (Ponomarenko p.10)

    Film effectiveness η = 1 − ξ
    T_aw_eff = η · T_sat + (1−η) · T_aw_bare

    The evolving M is the key physics: as film dilutes into the surface
    layer, local M drops, mixing slows down, and the film persists
    further downstream than a constant-M model predicts.  No curve
    fitting required — Kt is the only empirical knob and sits in the
    published range.

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
def _aedc_film_eta(s:              float,   # distance from injection [m]
                   M_c_dot:        float,   # coolant mass flow per circ [kg/(m·s)]
                   G_gas:          float,   # free-stream mass flux [kg/(m²·s)]
                   mu_gas:         float,   # free-stream viscosity [Pa·s]
                   cpg:            float,   # free-stream cp [J/(kg·K)]
                   cpc:            float,   # coolant cp  [J/(kg·K)]
                   M_cool_mol:     float,   # coolant molar mass [kg/mol]
                   M_gas_mol:      float,   # gas molar mass     [kg/mol]
                   et:             float,
                   x_i:            float = 0.0) -> float:
    """
    Gaseous film effectiveness η from AEDC-TR-91-1 Eq. 3.5
    (Morren & Nejad 1991 p.24, after Kutateladze & Stollery).

        η = [ 1 + K_M·(cpg/cpc)·(0.325·(Kt·X + X0)^0.8 − 1) ]^−1

    with
        X   = K · s                                 (s from injection)
        K   = G · μg^0.25 · Ṁc^(−1.25)              [1/m]
        G   = ρg · ug   free-stream mass flux       [kg/(m²·s)]
        Ṁc  = ṁ_film / (2π r_wall)                  coolant mass flow
              per unit circumference                [kg/(m·s)]
        X_0 = (3.08 + X_i^0.8)^1.25 − X_i           virtual upstream point
        X_i = K · x_i                               (0 when injection = origin)
        Kt  = 1 + 10.2·e_t                          (Eq. 3.1.2 turbulence correction)
        K_M = (M_cool_mol / M_gas_mol)^0.14         (Eq. 3.2 foreign-gas correction)

    All M_c_dot, G, μ appearing in K must be in SI — the resulting X is
    dimensionless.  For non-uniform free-stream flow (a rocket nozzle),
    AEDC §3.4 warns that Eq. 3.5 is strictly an integral for constant G;
    the caller should pass G evaluated at (or just downstream of) the
    INJECTION point, not the local value — otherwise η is non-monotonic
    across the throat.

    Returns η ∈ [0, 1].
    """
    if s <= 0.0 or M_c_dot <= 0.0 or G_gas <= 0.0:
        return 1.0

    K_scale = G_gas * (mu_gas**0.25) * (M_c_dot**(-1.25))   # [1/m]

    X    = K_scale * s
    X_i  = K_scale * max(x_i, 0.0)
    X_0  = (3.08 + X_i**0.8)**1.25 - X_i

    K_t  = 1.0 + 10.2 * et
    K_M  = (M_cool_mol / M_gas_mol)**0.14

    bracket = 0.325 * (K_t * X + X_0)**0.8 - 1.0
    denom   = 1.0 + K_M * (cpg / cpc) * bracket
    if denom <= 0.0:
        return 1.0
    eta = 1.0 / denom
    return float(np.clip(eta, 0.0, 1.0))


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

    # AEDC-model constants for the film coolant (vapour side, Phase 3)
    _coolant_props = {
        "RP1":      (0.170, 2500.0),  # (molar mass [kg/mol], cp_vapor [J/kgK])
        "Methane":  (0.01604, 2450.0),
        "Hydrogen": (0.002016, 14300.0),
    }
    M_coolant_kg_mol, cpc_vapor = _coolant_props.get(
        config.film_coolant, (0.170, 2500.0))

    # Free-stream conditions at the INJECTION station — frozen per AEDC §3.4
    # (Eq. 3.5 is an integral result for constant G; using local G across the
    # throat would spuriously jump η).
    M_inj       = float(flow.M[i_start])
    fac_inj     = 1.0 + (cea.gamma_c - 1.0) / 2.0 * M_inj**2
    T_inj       = cea.T_c / fac_inj
    P_inj       = cea.P_c * fac_inj**(-cea.gamma_c / (cea.gamma_c - 1.0))
    rho_inj     = P_inj / (cea.R_specific * T_inj)
    a_inj       = np.sqrt(cea.gamma_c * cea.R_specific * T_inj)
    u_inj       = M_inj * a_inj
    G_inj       = rho_inj * u_inj                              # [kg/m²s]
    mu_inj      = cea.visc_c * (T_inj / cea.T_c)**0.7          # Sutherland-like
    r_inj       = nozzle_radius(float(flow.x[i_start]), geom, dx)
    M_c_dot_inj = mdot_film / max(2.0 * np.pi * r_inj, 1e-12)  # [kg/(m·s)]

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
            # --- Phase 3: Gaseous mixing ---
            s = x - x_inject        # distance from INJECTION

            if config.film_model == "aedc":
                # AEDC-TR-91-1 Eq. 3.5 — closed form, no free Kt knob.
                # Free-stream G/μ frozen at the injection station per §3.4.
                eta = _aedc_film_eta(
                    s           = s,
                    M_c_dot     = M_c_dot_inj,
                    G_gas       = G_inj,
                    mu_gas      = mu_inj,
                    cpg         = cea.Cp_c,
                    cpc         = cpc_vapor,
                    M_cool_mol  = M_coolant_kg_mol,
                    M_gas_mol   = cea.molar_mass,
                    et          = config.film_aedc_et,
                    x_i         = 0.0,
                )
            else:
                # Vasiliev-Kudryavtsev (Ponomarenko 2012 p.9) — implicit ξ:
                #   m̄s(ξ) = m̄s⁰·(1−ξ/2) + m̄f⁰·(ξ/2)
                #   m̄f(ξ) = m̄s⁰·(ξ/2)   + m̄f⁰·(1−ξ/2)
                #   M(ξ)  = Kt · m̄s(ξ) / m̄f(ξ)
                #   ξ     = 1 − exp(−M(ξ) · s / Hs)
                Hs = _bl_thickness(x, cea, M)
                xi = 0.0
                for _ in range(20):
                    m_bar_s_cur = m_bar_s * (1.0 - xi/2.0) + m_bar_f * (xi/2.0)
                    m_bar_f_cur = m_bar_s * (xi/2.0)        + m_bar_f * (1.0 - xi/2.0)
                    if m_bar_f_cur < 1e-12:
                        xi = 1.0
                        break
                    M_mix   = Kt * m_bar_s_cur / m_bar_f_cur
                    xi_new  = 1.0 - np.exp(-M_mix * s / Hs)
                    if abs(xi_new - xi) < 1e-6:
                        xi = xi_new
                        break
                    xi = 0.5 * (xi + xi_new)
                eta = 1.0 - xi
                eta = float(np.clip(eta, 0.0, 1.0))

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
