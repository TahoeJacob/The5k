"""
heat_transfer.py
1-D regenerative cooling thermal analysis.

Physics
-------
Gas-side HTC    : Bartz (1957) with σ correction — Sutton & Biblarz (2010) eq. 8-22
Adiabatic wall  : turbulent recovery  r = Pr^(1/3)
Wall conduction : 1-D flat-plate resistance  q = k·A·ΔT / t
Coolant HTC     : Gnielinski (1975) for RP1 / Methane
                  Niino (1982) for Hydrogen (cryogenic)
Fin efficiency  : rectangular fin, adiabatic tip
Pressure drop   : Colebrook-White (major) + Idelchik (minor at area changes)

Coupling
--------
The gas-side Mach / P / T profile comes from the isentropic FlowSolution
and is held fixed throughout the thermal iteration (decoupled approach).
Coolant flows counter-current: enters at nozzle exit, exits at injector.

Entry point
-----------
  solve_thermal(flow, geom, cea, chan_geom, config) → ThermalSolution

Supporting dataclasses
----------------------
  ChannelGeometry : user-defined channel cross-section arrays
  ThermalSolution : result arrays (injector → exit indexing)
"""

import numpy as np
import scipy.optimize as sp
from dataclasses import dataclass

from geometry import EngineGeometry
from cea_interface import CEAResult
from flow_solver import FlowSolution
from config import EngineConfig
from coolant_props import get_coolant_props, get_T_from_enthalpy


# -----------------------------------------------------------------------
# Channel geometry
# -----------------------------------------------------------------------
@dataclass
class ChannelGeometry:
    """
    Cooling channel cross-section geometry along the engine axis.

    All arrays are in injector → exit order (ascending x).
    Dimensions may vary axially (e.g. constant-width design, tapered throat).

    Attributes
    ----------
    x_j       : axial stations from injector face [m]
    chan_w    : channel width (azimuthal) [m]
    chan_h    : channel height / radial depth [m]
    chan_t    : hot-wall thickness (gas-side wall) [m]
    chan_land : land width (wall between adjacent channels) [m]
    """
    x_j:       np.ndarray
    chan_w:    np.ndarray
    chan_h:    np.ndarray
    chan_t:    np.ndarray
    chan_land: np.ndarray

    def at(self, x: float):
        """
        Linearly interpolate (w, h, t, land) at axial position x [m].
        Clips to boundary values outside the defined range.
        """
        w    = float(np.interp(x, self.x_j, self.chan_w,
                               left=self.chan_w[0],    right=self.chan_w[-1]))
        h    = float(np.interp(x, self.x_j, self.chan_h,
                               left=self.chan_h[0],    right=self.chan_h[-1]))
        t    = float(np.interp(x, self.x_j, self.chan_t,
                               left=self.chan_t[0],    right=self.chan_t[-1]))
        land = float(np.interp(x, self.x_j, self.chan_land,
                               left=self.chan_land[0], right=self.chan_land[-1]))
        return w, h, t, land


# -----------------------------------------------------------------------
# Thermal solution output
# -----------------------------------------------------------------------
@dataclass
class ThermalSolution:
    """
    1-D thermal analysis results.

    All arrays are ordered injector → exit (same as FlowSolution.x).
    """
    x:           np.ndarray   # Axial positions [m]
    T_hw:        np.ndarray   # Hot wall temperature [K]
    T_cw:        np.ndarray   # Cold wall temperature [K]
    T_coolant:   np.ndarray   # Coolant bulk temperature [K]
    P_coolant:   np.ndarray   # Coolant static pressure [Pa]
    heatflux:    np.ndarray   # Gas-side heat flux [W/m²]
    h_gas:       np.ndarray   # Gas-side HTC [W/(m²·K)]
    h_coolant:   np.ndarray   # Coolant-side HTC [W/(m²·K)]
    q_gas:       np.ndarray   # Gas heat rate per channel per step [W]
    q_coolant:   np.ndarray   # Coolant heat absorption per channel per step [W]
    Re_coolant:  np.ndarray   # Coolant Reynolds number
    Nu_coolant:  np.ndarray   # Coolant Nusselt number
    Dh:          np.ndarray   # Coolant hydraulic diameter [m]
    v_coolant:   np.ndarray   # Coolant velocity [m/s]
    rho_coolant: np.ndarray   # Coolant density [kg/m³]
    n_iters:     int          # Outer iterations required to converge


# -----------------------------------------------------------------------
# Internal: friction factor (Colebrook-White)
# -----------------------------------------------------------------------
def _colebrook_f(Re: float, Dh: float, roughness: float,
                 f_init: float = 0.02) -> float:
    """
    Moody friction factor from the Colebrook-White equation.

    Re < 2300  : Hagen-Poiseuille  f = 64 / Re
    Re ≥ 2300  : Colebrook implicit solve (brentq)
    roughness = 0 gives the smooth-pipe (Prandtl-Kármán) value.
    """
    if Re < 2300.0:
        return 64.0 / max(Re, 1.0)

    def g(f: float) -> float:
        return (1.0 / np.sqrt(f)
                + 2.0 * np.log10(roughness / (3.7065 * Dh)
                                 + 2.5226 / (Re * np.sqrt(f))))
    try:
        return float(sp.brentq(g, 1e-6, 1.0, xtol=1e-8))
    except ValueError:
        return f_init   # fallback if brentq brackets fail


# -----------------------------------------------------------------------
# Internal: radius of curvature (Niino C3 correction)
# -----------------------------------------------------------------------
def _curvature_at(x: float, geom: EngineGeometry):
    """
    Wall radius of curvature for the Niino C3 correction.

    Returns (radius [m], label) where label ∈ {'Ru', 'Rd', 'none'}.
    """
    if geom.L_e < x <= geom.L_c:
        return geom.RU, 'Ru'
    if x > geom.L_c:
        return geom.RD, 'Rd'
    return 0.0, 'none'


# -----------------------------------------------------------------------
# Gas side
# -----------------------------------------------------------------------
def _bartz_h(M: float, A: float, T_hw: float,
             cea: CEAResult, geom: EngineGeometry) -> float:
    """
    Bartz (1957) gas-side HTC [W/(m²·K)].

    Transport properties are evaluated at the chamber stagnation state
    (T_c, P_c) using FROZEN Cp and Pr per the Bartz reference-state
    convention.  Frozen (composition-fixed) properties are used because
    the boundary layer cannot equilibrate chemically on the residence-time
    scale — consistent with Cantera at fixed mole fractions (the method
    used in the reference MixtureOptimization.py code).  Using equilibrium
    Cp (which includes the ∂H/∂T contribution of reaction progress) would
    over-predict h_g by ~1.5–2× for hydrogen flames.

    σ correction per Sutton & Biblarz (2010) eq. 8-22:
      σ = 1 / [(½·(T_hw/T_c)·(1+(γ-1)/2·M²) + ½)^0.68 · (1+(γ-1)/2·M²)^0.12]
    """
    gam   = cea.gamma_c
    D_t   = 2.0 * geom.R_t
    R_cur = 0.5 * (geom.RU + geom.RD)   # mean curvature radius at throat

    fac   = 1.0 + (gam - 1.0) / 2.0 * M**2
    sigma = 1.0 / ((0.5 * (T_hw / cea.T_c) * fac + 0.5)**0.68 * fac**0.12)

    return (
        (0.026 / D_t**0.2)
        * (cea.visc_c**0.2 * cea.Cp_froz_c / cea.Pr_froz_c**0.6)
        * (cea.P_c / cea.C_star)**0.8
        * (D_t / R_cur)**0.1
        * (geom.A_t / A)**0.9
        * sigma
    )


def _T_aw(M: float, cea: CEAResult) -> float:
    """
    Adiabatic wall temperature [K].

    Recovery factor for turbulent flow:  r = Pr_froz^(1/3)
    T_aw = T_c · (1 + r·(γ-1)/2·M²) / (1 + (γ-1)/2·M²)

    Frozen Pr is used for the same reason as in _bartz_h — the recovery
    factor is a boundary-layer quantity evaluated at fixed composition.
    """
    gam = cea.gamma_c
    r   = cea.Pr_froz_c**(1.0 / 3.0)
    N   = M**2
    return cea.T_c * (1.0 + r * (gam - 1.0) / 2.0 * N) / (1.0 + (gam - 1.0) / 2.0 * N)


def _gas_heat(x: float, M: float, A: float, T_hw: float, dx: float,
              chan_geom: ChannelGeometry,
              cea: CEAResult, geom: EngineGeometry,
              T_aw_eff: float = None):
    """
    Gas-side heat quantities at axial station x.

    Parameters
    ----------
    T_aw_eff : float, optional
        Effective adiabatic wall temperature from film cooling model [K].
        If None, the bare Bartz T_aw is used.

    Returns
    -------
    q_gas [W]      : heat transferred to ONE channel over step dx
    heatflux [W/m²]: local heat flux
    h_gas [W/(m²·K)]: Bartz HTC
    """
    chan_w, _, _, chan_land = chan_geom.at(x)
    h_g    = _bartz_h(M, A, T_hw, cea, geom)
    T_aw_v = T_aw_eff if T_aw_eff is not None else _T_aw(M, cea)

    heatflux = h_g * (T_aw_v - T_hw)
    q_gas    = heatflux * (chan_w + chan_land) * dx
    return q_gas, heatflux, h_g


def _q_wall(T_hw: float, T_cw: float,
            chan_w: float, chan_land: float, chan_t: float,
            dx: float, k_wall: float) -> float:
    """
    Conductive heat through the hot wall [W] per channel over step dx.

    1-D flat-plate model: area = (chan_w + chan_land) · dx
    """
    return k_wall * (chan_w + chan_land) * dx * (T_hw - T_cw) / chan_t


# -----------------------------------------------------------------------
# Coolant side
# -----------------------------------------------------------------------
def _coolant_heat(x: float, T_cw: float, T_cool: float, P_cool: float,
                  s: float, dx: float,
                  chan_geom: ChannelGeometry,
                  geom: EngineGeometry, config: EngineConfig,
                  mdot_per_ch: float):
    """
    Coolant-side heat transfer and pressure drop at station x.

    Marches counter-current: T_cool and P_cool are the coolant state
    arriving at this station from the nozzle exit side.

    Parameters
    ----------
    x           : axial position [m]
    T_cw        : cold wall temperature [K]
    T_cool      : coolant bulk temperature entering this step [K]
    P_cool      : coolant pressure entering this step [Pa]
    s           : distance from coolant inlet (nozzle exit) [m]
    dx          : step size [m]
    chan_geom   : ChannelGeometry
    geom        : EngineGeometry
    config      : EngineConfig
    mdot_per_ch : mass flow per channel [kg/s]

    Returns
    -------
    q_cool [W], T_cool_new [K], P_cool_new [Pa],
    h_cool [W/(m²·K)], Re, Nu, Dh [m], v [m/s], rho [kg/m³]
    """
    fluid = config.coolant
    e     = config.wall_roughness
    k_w   = config.wall_k

    # Channel dimensions at current and next counter-current station
    chan_w, chan_h, _, chan_land = chan_geom.at(x)
    x_next                       = max(x - dx, chan_geom.x_j[0])
    chan_w_n, chan_h_n, _, _     = chan_geom.at(x_next)

    chan_area   = chan_w   * chan_h
    chan_area_n = chan_w_n * chan_h_n
    Dh          = 4.0 * chan_area   / (2.0 * (chan_w   + chan_h))
    Dh_n        = 4.0 * chan_area_n / (2.0 * (chan_w_n + chan_h_n))

    # Coolant thermodynamic properties
    props = get_coolant_props(T_cool, P_cool, fluid)
    rho, h_enth, visc, cond, Cp = (props.rho, props.h, props.viscosity,
                                    props.conductivity, props.Cp)

    v  = mdot_per_ch / (rho * chan_area)
    Re = rho * v * Dh / visc
    Pr = visc * Cp / cond
    f  = _colebrook_f(Re, Dh, e)

    # Nusselt number
    if fluid == "RP1":
        # Ponomarenko / Dobrovolsky kerosene correlation (used by RPA):
        #   Nu = 0.021 Re^0.8 Pr^0.4 (0.64 + 0.36 T_c/T_wc)
        # The temperature-ratio factor accounts for large property variation
        # between the bulk coolant and the hot wall — critical for RP-1 where
        # T_c << T_wc.  Omitting it (e.g. Gnielinski) over-predicts h_cool by
        # ~25% and artificially collapses T_hw toward T_coolant.
        if Re < 2300.0:
            Nu = 4.36
        else:
            T_ratio = T_cool / max(T_cw, T_cool + 1.0)   # clip to avoid div/0
            Nu = 0.021 * Re**0.8 * Pr**0.4 * (0.64 + 0.36 * T_ratio)
    elif fluid == "Methane":
        # Ponomarenko methane correlation (RPA):
        #   Nu = 0.0185 Re^0.8 Pr^0.4 (T_c/T_wc)^0.1
        if Re < 2300.0:
            Nu = 4.36
        else:
            T_ratio = T_cool / max(T_cw, T_cool + 1.0)
            Nu = 0.0185 * Re**0.8 * Pr**0.4 * T_ratio**0.1
    else:
        # Niino (1982) for cryogenic hydrogen
        f_sm    = _colebrook_f(Re, Dh, 0.0)       # smooth-pipe reference
        xi      = f / f_sm if f_sm > 0.0 else 1.0
        eps_s   = Re * (e / Dh) * np.sqrt(f / 8.0)
        B       = 4.7 * eps_s**0.2 if eps_s >= 7.0 else 4.5 + 0.57 * eps_s**0.75

        C1 = ((1.0 + 1.5 * Pr**(-1.0/6) * Re**(-1.0/8) * (Pr - 1.0))
              / (1.0 + 1.5 * Pr**(-1.0/6) * Re**(-1.0/8) * (Pr * xi - 1.0))) * xi
        C2 = 1.0 + xi**0.1 * (s / Dh)**(-0.7)

        r_curv, r_type = _curvature_at(x, geom)
        if r_curv > 0.0:
            sign      = 1.0  if r_type == 'Ru' else -1.0
            concavity = 0.05 if r_type == 'Ru' else -0.05
            C3 = (Re * ((0.25 * Dh) / (sign * r_curv))**2)**concavity
        else:
            C3 = 1.0

        Nu = ((f / 8.0) * Re * Pr * (T_cool / T_cw)**0.55
              / (1.0 + np.sqrt(f / 8.0) * (B - 8.48))) * C1 * C2 * C3

    h_cool = Nu * cond / Dh

    # Rectangular fin efficiency (land between channels, adiabatic tip).
    #
    # The land is a radial fin: base at T_cw, tip adiabatic, height = chan_h.
    # Its cross-section (perpendicular to radial direction): chan_land × dx.
    # Of the four faces, only the two AZIMUTHAL faces (dx tall × chan_h) are
    # exposed to coolant — the two AXIAL faces connect to adjacent land segments
    # and carry no independent convection in the 1-D marching model.
    #
    # Therefore:  P = 2·dx,   A_c = chan_land·dx
    #   → m = √(hP / kA_c) = √(2h / (k·chan_land))   [mesh-independent]
    #
    # This is identical to RPA's formulation (Ponomarenko 2012, p.17):
    #   ξ = √(2α_c / λ_w) · b/√δ  with b=chan_h, δ=chan_land
    #
    # Note: P = 2·(chan_land+dx) is WRONG for this geometry — it includes the
    # axial end faces and makes m² mesh-dependent (diverges as dx→0).
    DT = T_cw - T_cool
    if chan_land > 1e-9:
        m_fin  = np.sqrt(2.0 * h_cool / (k_w * chan_land))
        L_fin  = chan_h + chan_land / 2.0          # Incropera adiabatic-tip correction
        q_fin  = np.sqrt(2.0 * h_cool * k_w * chan_land) * dx * DT * np.tanh(
                     np.clip(m_fin * L_fin, 0.0, 50.0))
    else:
        q_fin = 0.0

    q_base = h_cool * chan_w * dx * DT
    q_cool = q_fin + q_base

    # Pressure drop: major (Darcy-Weisbach) + minor (area change)
    major = f * rho * v**2 * dx / (2.0 * Dh)
    if   chan_area_n > chan_area:                               # expansion
        K = ((Dh / Dh_n)**2 - 1.0)**2
    elif chan_area_n < chan_area:                               # contraction
        K = 0.5 - 0.167*(Dh_n/Dh) - 0.125*(Dh_n/Dh)**2 - 0.208*(Dh_n/Dh)**3
    else:
        K = 0.0
    minor      = K * rho * v**2 / 2.0
    P_cool_new = P_cool - (major + minor)

    # Temperature advance via enthalpy conservation
    h_next     = h_enth + q_cool / mdot_per_ch
    T_cool_new = get_T_from_enthalpy(h_next, P_cool_new, fluid)

    return q_cool, T_cool_new, P_cool_new, h_cool, Re, Nu, Dh, v, rho


# -----------------------------------------------------------------------
# Newton-Raphson: find T_hw, T_cw at one axial station
# -----------------------------------------------------------------------
def _newton_solve(x: float, M: float, A: float,
                  T_cool: float, P_cool: float, s: float, dx: float,
                  chan_geom: ChannelGeometry,
                  cea: CEAResult, geom: EngineGeometry,
                  config: EngineConfig, mdot_per_ch: float,
                  T_hw0: float, T_cw0: float,
                  tol: float = 0.1, max_iter: int = 50,
                  T_aw_eff: float = None):
    """
    Newton-Raphson solve for (T_hw, T_cw) satisfying the thermal circuit:

        q_gas(T_hw) = q_wall(T_hw, T_cw) = q_coolant(T_cw)

    Residuals:
        F1 = q_wall - q_gas  = 0    (hot-face energy balance)
        F2 = q_gas  - q_cool = 0    (global energy balance)

    Step damping: max |ΔT| capped at 50 K per Newton step.

    Returns (T_hw, T_cw).
    """
    chan_w, _, chan_t, chan_land = chan_geom.at(x)

    def F(T_vec):
        T_hw, T_cw = float(T_vec[0]), float(T_vec[1])
        q_g, *_  = _gas_heat(x, M, A, T_hw, dx, chan_geom, cea, geom,
                              T_aw_eff=T_aw_eff)
        q_w      = _q_wall(T_hw, T_cw, chan_w, chan_land, chan_t, dx, config.wall_k)
        q_c, *_  = _coolant_heat(x, T_cw, T_cool, P_cool, s, dx,
                                  chan_geom, geom, config, mdot_per_ch)
        return np.array([q_w - q_g, q_g - q_c])

    def jac_fd(T_vec, h_fd=1e-3):
        J = np.zeros((2, 2))
        for i in range(2):
            T_p = np.array(T_vec, dtype=float); T_p[i] += h_fd
            T_m = np.array(T_vec, dtype=float); T_m[i] -= h_fd
            J[:, i] = (F(T_p) - F(T_m)) / (2.0 * h_fd)
        return J

    T = np.array([T_hw0, T_cw0], dtype=float)
    for _ in range(max_iter):
        Fv = F(T)
        if np.linalg.norm(Fv) < tol:
            break
        J = jac_fd(T)
        try:
            delta = np.linalg.solve(J, -Fv)
        except np.linalg.LinAlgError:
            raise RuntimeError(f"Singular Jacobian at x = {x*1000:.1f} mm.")
        max_step = np.max(np.abs(delta))
        if max_step > 50.0:
            delta *= 50.0 / max_step
        T += delta

    return float(T[0]), float(T[1])


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------
def solve_thermal(flow: FlowSolution,
                  geom: EngineGeometry,
                  cea:  CEAResult,
                  chan_geom: ChannelGeometry,
                  config: EngineConfig,
                  T_hw_init: float = 700.0,
                  T_cw_init: float = 400.0,
                  tol_K: float = 1.0,
                  relax: float = 0.8,
                  max_outer: int = 30,
                  T_aw_eff: np.ndarray = None) -> ThermalSolution:
    """
    Coupled 1-D regenerative cooling thermal analysis.

    The isentropic flow profile (flow) is held fixed throughout the
    iteration — this is a decoupled (sequential) approach.  Coolant
    enters at the nozzle exit and flows counter-current toward the
    injector.

    Parameters
    ----------
    flow       : FlowSolution from solve_flow()
    geom       : EngineGeometry from size_engine()
    cea        : CEAResult from get_cea_for_analysis()
    chan_geom  : ChannelGeometry (user-defined cross-section arrays)
    config     : EngineConfig
    T_hw_init  : initial hot-wall temperature guess [K]
    T_cw_init  : initial cold-wall temperature guess [K]
    tol_K      : convergence criterion — mean |ΔT| < tol_K [K]
    relax      : under-relaxation factor  (0 < relax ≤ 1)
    max_outer  : maximum outer iterations

    Returns
    -------
    ThermalSolution (all arrays indexed injector → exit)
    """
    if config.mdot_coolant is None:
        raise ValueError(
            "config.mdot_coolant must be set before calling solve_thermal.  "
            "Set it explicitly or derive it from geom.mdot_fuel in main.py.")

    n           = len(flow.x)
    dx          = config.dx
    mdot_per_ch = config.mdot_coolant / config.N_channels

    # Output arrays (index 0 = injector, index n-1 = nozzle exit)
    T_hw_arr  = np.full(n, T_hw_init)
    T_cw_arr  = np.full(n, T_cw_init)
    T_cool_arr = np.full(n, config.T_coolant_inlet)
    P_cool_arr = np.full(n, config.P_coolant_inlet)
    hf_arr     = np.zeros(n)
    hg_arr     = np.zeros(n)
    hc_arr     = np.zeros(n)
    qg_arr     = np.zeros(n)
    qc_arr     = np.zeros(n)
    Re_arr     = np.zeros(n)
    Nu_arr     = np.zeros(n)
    Dh_arr     = np.zeros(n)
    v_arr      = np.zeros(n)
    rho_arr    = np.zeros(n)

    n_iters = 0

    # Build T_aw_eff lookup — None means bare Bartz T_aw used at each station
    _T_aw_eff_arr = T_aw_eff  # may be None (no film) or np.ndarray (film active)

    film_active = (_T_aw_eff_arr is not None)
    print(f"\n--- Thermal Solver ---")
    print(f"  Fluid: {config.coolant}  "
          f"N_ch: {config.N_channels}  "
          f"mdot/ch: {mdot_per_ch*1000:.2f} g/s  "
          f"T_in: {config.T_coolant_inlet:.0f} K  "
          f"P_in: {config.P_coolant_inlet/1e5:.1f} bar")
    if film_active:
        print(f"  Film cooling ACTIVE — T_aw_eff supplied ({config.film_fraction*100:.1f}% film)")

    for outer in range(max_outer):
        T_hw_prev = T_hw_arr.copy()
        T_cw_prev = T_cw_arr.copy()

        # Coolant enters at nozzle exit (index n-1)
        T_cool = config.T_coolant_inlet
        P_cool = config.P_coolant_inlet
        s      = dx    # distance from coolant inlet

        # Carry Newton initial guesses station-to-station for fast convergence
        T_hw_nwt = T_hw_arr[n - 1]
        T_cw_nwt = T_cw_arr[n - 1]

        for k in range(n - 1, -1, -1):   # exit → injector (counter-current)
            x = float(flow.x[k])
            M = float(flow.M[k])
            A = float(flow.A[k])

            # Film-corrected T_aw for this station (None = use bare Bartz)
            T_aw_k = float(_T_aw_eff_arr[k]) if film_active else None

            T_hw_nwt, T_cw_nwt = _newton_solve(
                x, M, A, T_cool, P_cool, s, dx,
                chan_geom, cea, geom, config, mdot_per_ch,
                T_hw_nwt, T_cw_nwt,
                T_aw_eff=T_aw_k)

            # Under-relaxation
            T_hw_arr[k] = relax * T_hw_nwt + (1.0 - relax) * T_hw_prev[k]
            T_cw_arr[k] = relax * T_cw_nwt + (1.0 - relax) * T_cw_prev[k]

            # Final evaluation with relaxed wall temperatures for result storage
            q_g, hf, h_g = _gas_heat(x, M, A, T_hw_arr[k], dx, chan_geom, cea, geom,
                                      T_aw_eff=T_aw_k)
            (q_c, T_cool_new, P_cool_new,
             h_c, Re, Nu, Dh, v, rho) = _coolant_heat(
                x, T_cw_arr[k], T_cool, P_cool, s, dx,
                chan_geom, geom, config, mdot_per_ch)

            T_cool_arr[k] = T_cool
            P_cool_arr[k] = P_cool
            hf_arr[k]     = hf
            hg_arr[k]     = h_g
            hc_arr[k]     = h_c
            qg_arr[k]     = q_g
            qc_arr[k]     = q_c
            Re_arr[k]     = Re
            Nu_arr[k]     = Nu
            Dh_arr[k]     = Dh
            v_arr[k]      = v
            rho_arr[k]    = rho

            # Advance coolant toward injector
            T_cool    = T_cool_new
            P_cool    = P_cool_new
            s        += dx

            # Update Newton guess for next (upstream) station
            T_hw_nwt  = T_hw_arr[k]
            T_cw_nwt  = T_cw_arr[k]

        err_hw  = float(np.mean(np.abs(T_hw_arr - T_hw_prev)))
        err_cw  = float(np.mean(np.abs(T_cw_arr - T_cw_prev)))
        n_iters = outer + 1

        print(f"  Iter {n_iters:2d}:  ΔT_hw = {err_hw:6.2f} K  ΔT_cw = {err_cw:6.2f} K"
              f"  T_hw_max = {np.max(T_hw_arr):.0f} K  T_cw_max = {np.max(T_cw_arr):.0f} K")

        if err_hw < tol_K and err_cw < tol_K:
            print(f"  Converged in {n_iters} iteration(s).")
            break
    else:
        print(f"  Warning: did not converge in {max_outer} iterations "
              f"(ΔT_hw = {err_hw:.2f} K  ΔT_cw = {err_cw:.2f} K).")

    T_hw_max = float(np.max(T_hw_arr))
    if T_hw_max > config.wall_melt_T:
        print(f"  *** WARNING: T_hw_max = {T_hw_max:.0f} K  "
              f"exceeds wall_melt_T = {config.wall_melt_T:.0f} K ***")

    return ThermalSolution(
        x           = flow.x,
        T_hw        = T_hw_arr,
        T_cw        = T_cw_arr,
        T_coolant   = T_cool_arr,
        P_coolant   = P_cool_arr,
        heatflux    = hf_arr,
        h_gas       = hg_arr,
        h_coolant   = hc_arr,
        q_gas       = qg_arr,
        q_coolant   = qc_arr,
        Re_coolant  = Re_arr,
        Nu_coolant  = Nu_arr,
        Dh          = Dh_arr,
        v_coolant   = v_arr,
        rho_coolant = rho_arr,
        n_iters     = n_iters,
    )


# -----------------------------------------------------------------------
# Thermal results plots
# -----------------------------------------------------------------------
def plot_thermal(thermal: ThermalSolution,
                 geom:    EngineGeometry,
                 config:  EngineConfig) -> None:
    """
    Six-panel figure of key thermal results, each with the engine
    half-profile overlaid as a light gray fill so peaks can be
    related to the throat, converging, and diverging sections.

    Panels
    ------
    1. Wall temperatures  (T_hw, T_cw, melting-point limit)
    2. Hot-wall heat flux  [MW/m²]
    3. Gas-side HTC  (Bartz)  [kW/(m²·K)]
    4. Coolant bulk temperature  [K]
    5. Coolant static pressure  [MPa]
    6. Coolant Reynolds number
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from geometry import nozzle_radius

    x_mm = thermal.x * 1000   # axial position [mm]

    # Engine half-profile at each thermal station
    r_wall_mm = np.array(
        [nozzle_radius(float(xi), geom, config.dx) * 1000 for xi in thermal.x]
    )

    x_throat_mm = geom.L_c * 1000   # throat axial position [mm]

    # ------------------------------------------------------------------
    # Helper: add geometry fill on a twin right-hand axis
    # scale controls how far the geometry extends up the axis
    # (scale=5 → geometry occupies bottom ~20 % of the panel)
    # ------------------------------------------------------------------
    def _geom_overlay(ax, scale: float = 5.0):
        ax2 = ax.twinx()
        ax2.fill_between(x_mm, 0, r_wall_mm,
                         color='lightsteelblue', alpha=0.35, zorder=0)
        ax2.plot(x_mm, r_wall_mm, color='steelblue', lw=0.8, zorder=1)
        ax2.set_ylim(0, r_wall_mm.max() * scale)
        ax2.set_ylabel('Wall radius [mm]', color='steelblue', fontsize=7)
        ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=7)
        return ax2

    def _markers(ax):
        ax.axvline(x_throat_mm, color='red', ls='--', lw=0.9,
                   label=f'Throat ({x_throat_mm:.0f} mm)')
        ax.grid(True, alpha=0.3, zorder=2)

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        f'{config.fuel}/{config.oxidizer}  '
        f'Pc = {config.P_c/1e5:.0f} bar  '
        f'O/F = {config.OF}  '
        f'Ncc = {config.N_channels}  '
        f'ṁ_cool = {config.mdot_coolant:.2f} kg/s',
        fontsize=12)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.42)

    # ---- 1: Wall temperatures ----------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_mm, thermal.T_hw, 'r-',  lw=1.5, label='T_hw (hot wall)', zorder=3)
    ax1.plot(x_mm, thermal.T_cw, 'b-',  lw=1.5, label='T_cw (cold wall)', zorder=3)
    ax1.axhline(config.wall_melt_T, color='k', ls=':', lw=1.2,
                label=f'T_melt = {config.wall_melt_T:.0f} K', zorder=3)
    _markers(ax1)
    ax1.set_xlabel('Axial position [mm]')
    ax1.set_ylabel('Temperature [K]')
    ax1.set_title('Wall Temperatures')
    ax1.legend(fontsize=7, loc='upper right')
    _geom_overlay(ax1)

    # ---- 2: Hot-wall heat flux ---------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_mm, thermal.heatflux / 1e6, 'r-', lw=1.5,
             label=f'peak {thermal.heatflux.max()/1e6:.1f} MW/m²', zorder=3)
    _markers(ax2)
    ax2.set_xlabel('Axial position [mm]')
    ax2.set_ylabel('Heat flux [MW/m²]')
    ax2.set_title('Hot-Wall Heat Flux')
    ax2.legend(fontsize=7)
    _geom_overlay(ax2)

    # ---- 3: Gas-side HTC (Bartz) ------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x_mm, thermal.h_gas / 1e3, color='darkorange', lw=1.5,
             label=f'peak {thermal.h_gas.max()/1e3:.1f} kW/(m²·K)', zorder=3)
    _markers(ax3)
    ax3.set_xlabel('Axial position [mm]')
    ax3.set_ylabel('HTC [kW/(m²·K)]')
    ax3.set_title('Gas-Side HTC (Bartz)')
    ax3.legend(fontsize=7)
    _geom_overlay(ax3)

    # ---- 4: Coolant bulk temperature --------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x_mm, thermal.T_coolant, 'b-', lw=1.5,
             label=f'ΔT = {thermal.T_coolant[0]-thermal.T_coolant[-1]:.1f} K',
             zorder=3)
    _markers(ax4)
    ax4.set_xlabel('Axial position [mm]')
    ax4.set_ylabel('Temperature [K]')
    ax4.set_title('Coolant Bulk Temperature')
    ax4.legend(fontsize=7)
    _geom_overlay(ax4)

    # ---- 5: Coolant pressure ----------------------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    dP_bar = (thermal.P_coolant[-1] - thermal.P_coolant[0]) / 1e5
    ax5.plot(x_mm, thermal.P_coolant / 1e6, color='purple', lw=1.5,
             label=f'ΔP = {abs(dP_bar):.1f} bar', zorder=3)
    _markers(ax5)
    ax5.set_xlabel('Axial position [mm]')
    ax5.set_ylabel('Pressure [MPa]')
    ax5.set_title('Coolant Static Pressure')
    ax5.legend(fontsize=7)
    _geom_overlay(ax5)

    # ---- 6: Coolant Reynolds number ---------------------------------
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(x_mm, thermal.Re_coolant / 1e3, color='teal', lw=1.5,
             label='Re', zorder=3)
    ax6.axhline(2.3, color='gray', ls=':', lw=0.8,
                label='Re = 2300 (laminar limit)')
    _markers(ax6)
    ax6.set_xlabel('Axial position [mm]')
    ax6.set_ylabel('Reynolds number [×10³]')
    ax6.set_title('Coolant Reynolds Number')
    ax6.legend(fontsize=7)
    _geom_overlay(ax6)

    # ------------------------------------------------------------------
    # Summary print
    # ------------------------------------------------------------------
    i_peak_hf = int(np.argmax(thermal.heatflux))
    i_peak_hw = int(np.argmax(thermal.T_hw))
    print(f"\n--- Thermal Results Summary ---")
    print(f"  Peak T_hw    = {thermal.T_hw.max():.0f} K  "
          f"at x = {thermal.x[i_peak_hw]*1000:.1f} mm  "
          f"({'EXCEEDS' if thermal.T_hw.max() > config.wall_melt_T else 'below'} "
          f"T_melt = {config.wall_melt_T:.0f} K)")
    print(f"  Peak T_cw    = {thermal.T_cw.max():.0f} K  "
          f"at x = {thermal.x[i_peak_hw]*1000:.1f} mm")
    print(f"  Peak q_gas   = {thermal.heatflux.max()/1e6:.2f} MW/m²  "
          f"at x = {thermal.x[i_peak_hf]*1000:.1f} mm")
    print(f"  Peak h_gas   = {thermal.h_gas.max()/1e3:.1f} kW/(m²·K)")
    print(f"  ΔT coolant   = {thermal.T_coolant[0]-thermal.T_coolant[-1]:.1f} K  "
          f"({thermal.T_coolant[-1]:.1f} → {thermal.T_coolant[0]:.1f} K)")
    print(f"  ΔP coolant   = {abs(dP_bar):.1f} bar  "
          f"({thermal.P_coolant[-1]/1e5:.1f} → {thermal.P_coolant[0]/1e5:.1f} bar)")

    plt.tight_layout()
