"""
film_cooling.py
Liquid + gaseous film cooling model following the RPA thermal analysis
methodology (Ponomarenko 2012, RPA_ThermalAnalysis.pdf pp. 8-11).

Phase 1 — Liquid heating  (x_inject → x_sat)             [Ponomarenko p.10]
    Film enters at T_film_inlet, heated by gas-side convection at T_f.
    Stability coefficient η(Re_f) per RPA Figure 2 / Vasiliev [1].
    Wall sees T_aw_eff = T_f  (convective q = h_g·(T_aw − T_f))

Phase 2 — Vaporisation  (x_sat → x_vap)                  [Ponomarenko p.11]
    Film locked at T_sat, gas heat evaporates liquid.
    Wall sees T_aw_eff = T_sat.

Phase 3 — Gaseous mixing with surface layer              [Ponomarenko p.9]
    After the film is fully vaporised, the gaseous coolant mixes with the
    surface layer.  RPA does **not** compute a simple effectiveness η and
    blend T_aw — instead it tracks the mixture composition in the
    wall-adjacent layer and applies Ievlev's "similar conditions"
    correlation to scale the heat flux:

        q^(2) = q^(1) · S^(2) / S^(1)

        S = (I∞ − I_w)·T_e^0.425·μ_1000^0.15
            ────────────────────────────────────────────────────────────────
            R_1500^0.425 · (T_e + T_w)^0.595 · (3 T_e + T_w)^0.15

    where
        q^(1), S^(1) are computed for the bare free-stream gas
        q^(2), S^(2) are computed for the mixed surface layer

    Layer mixing follows Vasiliev/Ponomarenko p.9:

        m̄s(ξ) = m̄s⁰·(1−ξ/2) + m̄f⁰·(ξ/2)
        m̄f(ξ) = m̄s⁰·(ξ/2)   + m̄f⁰·(1−ξ/2)
        k_ff   = (m̄f⁰/m̄f)·(1−ξ/2)    ← fraction of f-layer that's film
        k_fs   = (m̄s⁰/m̄f)·(ξ/2)      ← fraction of f-layer that's gas
        ξ      = 1 − exp(−M · x/Hs),   M = Kt · m̄s/m̄f     (implicit in ξ)

    with Kt ∈ (0.05–0.20)×10⁻² per Vasiliev [1].

    Two closure choices — noted because the paper does NOT specify them:
        Hs  : "thickness of the surface layer".  No formula in the paper.
              We use the turbulent flat-plate BL thickness
              Hs = 0.37·s/Re_s^0.2
              with s = distance from injection.
        m̄s⁰ : initial surface-layer relative mass flow.  No formula in the
              paper.  Physical interpretation: ṁ_BL(x_inject)/ṁ_total — the
              gas already in the wall BL when the film is introduced.
              Default: 2·Hs_dev/R_chamber from a developed turbulent BL at
              x = L_chamber; override via config.film_m_bar_s0.

    Layer properties (the inputs to Ievlev's S) are computed as
    mass-weighted blends of the free-stream combustion gas and the film
    coolant vapour:
        I∞_mix = w_gas·I∞_g + w_film·I∞_f    (stagnation enthalpy)
        R_mix  = w_gas·R_g  + w_film·R_f
        μ_mix  = w_gas·μ_g  + w_film·μ_f
        T_e_mix: back-solved from the mixture stagnation enthalpy

    Since the thermal solver has a (T_aw_eff) interface, Phase 3 is
    expressed as an equivalent T_aw_eff that reproduces the Ievlev-scaled
    flux at a reference wall temperature T_w_ref:

        T_aw_eff = T_w_ref + (S^(2)/S^(1)) · (T_aw_bare − T_w_ref)

    The S^(2)/S^(1) ratio depends on T_w only through the
    (T_e+T_w) and (3T_e+T_w) factors and is weak — a fixed T_w_ref
    (default 1000 K) is adequate for the back-conversion.

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
def _ievlev_S(I_inf: float, I_w: float,
              T_e:   float, T_w: float,
              mu_1000: float, R_1500: float) -> float:
    """
    Ievlev's "similar conditions" parameter S (Ponomarenko p.8).

        S = (I∞ − I_w)·T_e^0.425·μ_1000^0.15
            ────────────────────────────────────────────────
            R_1500^0.425·(T_e + T_w)^0.595·(3T_e + T_w)^0.15

    All temperatures [K], enthalpies [J/kg], μ [Pa·s], R [J/(kg·K)].
    Used as a ratio (S^(2)/S^(1)) — absolute units cancel.
    """
    num   = max(I_inf - I_w, 1.0) * T_e**0.425 * mu_1000**0.15
    denom = (R_1500**0.425
             * max(T_e + T_w, 1.0)**0.595
             * max(3.0 * T_e + T_w, 1.0)**0.15)
    return num / denom


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

    # Approximate liquid film viscosity for stability coefficient
    mu_film_liq = 3.0e-4   # RP-1 at ~400K [Pa·s] (order-of-magnitude)

    # Approximate liquid Cp
    Cp_film = 2100.0   # RP-1 [J/(kg·K)]

    # Ponomarenko Phase 3 — free-stream and film vapour reference properties
    # for the mixed-layer calculation.  Film vapour numbers are from the
    # RPA Liquid Propellant Handbook for RP-1 at 500–1000 K; Methane and
    # Hydrogen are rough estimates at similar reference temperatures.
    _coolant_vapour_props = {
        # fluid    :  (R [J/kgK], μ_1000 [Pa·s], Cp_vap [J/kgK])
        "RP1":       (48.9,   2.5e-5, 2500.0),   # R = 8.314/0.170
        "Methane":   (518.0,  2.8e-5, 4200.0),
        "Hydrogen":  (4124.0, 2.0e-5, 14900.0),
    }
    R_film_vap, mu1000_film, cp_film_vap = _coolant_vapour_props.get(
        config.film_coolant, _coolant_vapour_props["RP1"])

    # Free-stream / core-gas reference properties (bare, un-mixed)
    R_gas        = cea.R_specific                     # [J/(kg·K)]
    cp_gas       = cea.Cp_c                           # [J/(kg·K)] at T_c
    mu1000_gas   = cea.visc_c * (1000.0 / cea.T_c)**0.7   # Sutherland-like
    I_gas_stag   = cp_gas * cea.T_c                   # stagnation enthalpy

    # Film stagnation enthalpy at the exit of Phase 2 (just vaporised at T_sat)
    I_film_vap   = cp_film_vap * T_sat

    # Reference wall temperature for T_aw_eff back-conversion.  The
    # S^(2)/S^(1) ratio depends on T_w only through mild (T_e+T_w) and
    # (3T_e+T_w) factors, so a fixed reference is adequate.  1000 K is
    # near the CuCrZr design range.
    T_w_ref      = 1000.0

    # Initial layer mass ratios.
    #
    # m̄_f⁰ = ṁ_film / ṁ_total (well-defined).
    #
    # m̄_s⁰ is the initial "surface layer" relative mass flow at injection.
    # Ponomarenko p.9 does not give a formula.  Physical interpretation: the
    # combustion-gas mass flow already in the wall BL at the injection
    # location, as a fraction of total chamber mass flow.  For a developed
    # turbulent BL in the chamber, ṁ_BL/ṁ_total ≈ 2·Hs/R_chamber (annular
    # approximation), typically 0.1–0.3 for injection past the injector face.
    #
    # Setting m̄_s⁰ = 1 − m̄_f⁰ (treating the *entire* outer flow as the
    # surface layer) makes the layer enormous relative to the film and the
    # asymptotic film fraction in the wall layer collapses to ~m̄_f⁰, i.e.
    # protection vanishes.  That over-predicts q_gas by ~50 % vs RPA.
    m_bar_f0     = m_bar_f                  # ṁ_film / ṁ_total
    if config.film_m_bar_s0 is not None:
        m_bar_s0 = float(config.film_m_bar_s0)
    else:
        # Physical default: turbulent flat-plate BL at the chamber-end
        # reference length, normalised by chamber radius.
        R_chamber = float(nozzle_radius(0.0, geom, dx))
        L_ref     = max(float(geom.L_c), 0.02)
        # Representative chamber Mach: sample ~80 % of the way from injection
        # to the throat, still subsonic (M ≈ 0.15–0.25 in typical chambers).
        i_ref  = min(i_start + int(0.8 * L_ref / dx), n - 1)
        M_ref  = max(float(flow.M[i_ref]), 0.1)
        Hs_dev = _bl_thickness(L_ref, cea, M_ref)
        m_bar_s0 = float(np.clip(2.0 * Hs_dev / R_chamber, 0.05, 0.5))

    print(f"\n--- Film Cooling ---")
    print(f"  mdot_film = {mdot_film*1000:.2f} g/s  "
          f"({config.film_fraction*100:.1f}% of coolant)")
    print(f"  Injection at x = {x_inject*1000:.1f} mm  "
          f"T_inlet = {T_film_inlet:.1f} K  T_sat = {T_sat:.1f} K")
    print(f"  Kt = {Kt:.4f}  m̄f⁰ = {m_bar_f0:.4f}  m̄s⁰ = {m_bar_s0:.4f}  "
          f"M₀ = Kt·m̄s⁰/m̄f⁰ = {Kt * m_bar_s0 / m_bar_f0:.2f}")

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
            # --- Phase 3: Gaseous mixing (Ponomarenko p.9) ---
            # Track mixture composition in the wall-adjacent "f" layer,
            # apply Ievlev's S^(2)/S^(1) ratio to scale q, and back-compute
            # an equivalent T_aw_eff.
            s  = x - x_inject        # distance from INJECTION (p.10)
            Hs = _bl_thickness(x, cea, M)

            # Implicit solve for ξ: M = Kt·m̄s(ξ)/m̄f(ξ), ξ = 1−exp(−M·s/Hs)
            xi = 0.0
            for _ in range(20):
                ms_cur = m_bar_s0 * (1.0 - xi/2.0) + m_bar_f0 * (xi/2.0)
                mf_cur = m_bar_s0 * (xi/2.0)       + m_bar_f0 * (1.0 - xi/2.0)
                if mf_cur < 1e-12:
                    xi = 1.0
                    break
                M_mix  = Kt * ms_cur / mf_cur
                xi_new = 1.0 - np.exp(-M_mix * s / Hs)
                if abs(xi_new - xi) < 1e-6:
                    xi = xi_new
                    break
                xi = 0.5 * (xi + xi_new)
            xi = float(np.clip(xi, 0.0, 1.0))

            # Wall-adjacent "f" layer composition after mixing:
            #   k_ff = fraction of f-layer that's original film
            #   k_fs = fraction of f-layer that's entrained gas
            mf_xi = m_bar_s0 * (xi/2.0) + m_bar_f0 * (1.0 - xi/2.0)
            if mf_xi > 1e-12:
                w_film = (m_bar_f0 * (1.0 - xi/2.0)) / mf_xi   # = k_ff
                w_gas  = (m_bar_s0 * (xi/2.0))       / mf_xi   # = k_fs
            else:
                w_film, w_gas = 0.0, 1.0

            # Mass-weighted mixed-layer properties
            R_mix      = w_gas * R_gas     + w_film * R_film_vap
            mu1000_mix = w_gas * mu1000_gas + w_film * mu1000_film
            I_inf_mix  = w_gas * I_gas_stag + w_film * I_film_vap
            cp_mix     = w_gas * cp_gas    + w_film * cp_film_vap
            T_e_mix    = I_inf_mix / max(cp_mix, 1.0)

            # Bare (S^(1)) free-stream stagnation enthalpy and edge T.
            # T_e_bare is the local recovery/edge temperature (bare T_aw).
            T_e_bare = T_aw_i
            I_inf_b  = I_gas_stag

            # Wall enthalpy reference — use gas Cp at T_w_ref (both layers).
            I_w_ref  = cp_gas * T_w_ref

            S1 = _ievlev_S(I_inf_b,   I_w_ref, T_e_bare, T_w_ref,
                           mu1000_gas, R_gas)
            S2 = _ievlev_S(I_inf_mix, I_w_ref, T_e_mix,  T_w_ref,
                           mu1000_mix, R_mix)

            f_q = S2 / S1 if S1 > 0.0 else 1.0
            f_q = float(np.clip(f_q, 0.0, 1.2))

            # Back-compute an equivalent T_aw_eff that reproduces the
            # Ievlev-scaled flux at the reference wall temperature:
            #     q = h·(T_aw_eff − T_w_ref)  ≡  f_q · h·(T_aw_bare − T_w_ref)
            T_aw_eff[i] = T_w_ref + f_q * (T_aw_i - T_w_ref)

            if (1.0 - f_q) < 0.005:
                phase = "spent"
                print(f"  Gaseous film spent (S²/S¹>99.5%) at x = {x*1000:.1f} mm")

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
