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
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import Optional

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
    n_chan:    Optional[np.ndarray] = None  # channel count at each station
                                             # (for bifurcating designs).
                                             # If None, falls back to
                                             # config.N_channels everywhere.

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
             cea: CEAResult, geom: EngineGeometry,
             C: float = 0.026) -> float:
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

    Parameters
    ----------
    C : float
        Bartz leading coefficient.  C = 0.026 (thin BL, default) or
        C = 0.023 (thick BL, Bartz 1965 Fig 10).
    """
    gam   = cea.gamma_c
    D_t   = 2.0 * geom.R_t
    R_cur = 0.5 * (geom.RU + geom.RD)   # mean curvature radius at throat

    fac   = 1.0 + (gam - 1.0) / 2.0 * M**2
    sigma = 1.0 / ((0.5 * (T_hw / cea.T_c) * fac + 0.5)**0.68 * fac**0.12)

    return (
        (C / D_t**0.2)
        * (cea.visc_c**0.2 * cea.Cp_froz_c / cea.Pr_froz_c**0.6)
        * (cea.P_c / cea.C_star)**0.8
        * (D_t / R_cur)**0.1
        * (geom.A_t / A)**0.9
        * sigma
    )


def solve_boundary_layer(flow: FlowSolution, cea: CEAResult,
                         geom: EngineGeometry, T_hw_arr: np.ndarray,
                         C_bartz: float = 0.026) -> np.ndarray:
    """
    Bartz (1965) integral boundary-layer method for gas-side HTC.

    Marches coupled ODEs for momentum thickness θ (Eq. 25) and energy
    thickness φ (Eq. 30) from the chamber inlet to the nozzle exit.
    Computes local skin-friction Cf (Eq. 37, film-temperature method)
    and Stanton number Ch (Eq. 38, von Kármán Reynolds analogy).

    This replaces the simplified Bartz (Eq. 50) which has no boundary-
    layer history and over-predicts h_gas in the divergent section.

    References
    ----------
    D. R. Bartz, "Turbulent Boundary-Layer Heat Transfer from Rapidly
    Accelerating Flow of Rocket Combustion Gases and of Heated Air,"
    Advances in Heat Transfer, Vol. 2, 1965, pp. 1–108.

    Parameters
    ----------
    flow     : FlowSolution — isentropic M, P, T, A arrays
    cea      : CEAResult — stagnation properties
    geom     : EngineGeometry — throat geometry
    T_hw_arr : np.ndarray — current hot-wall temperature guess at each station

    Returns
    -------
    h_gas : np.ndarray — gas-side HTC [W/(m²·K)] at each flow station
    """
    n   = len(flow.x)
    gam = cea.gamma_c
    Pr  = cea.Pr_froz_c
    Cp  = cea.Cp_froz_c
    mu_ref = cea.visc_c     # reference viscosity at T_c
    T_ref  = cea.T_c        # reference temperature for power-law scaling
    R_sp   = cea.R_specific

    h_gas = np.zeros(n)

    # --- Initial conditions ---
    # Flat-plate turbulent BL at chamber inlet (Bartz 1965 Sec. II.A.7)
    # θ₀ = 0.036 · L / Re_L^(1/5)  (Schlichting, turbulent flat plate)
    T_0   = float(flow.T[0])
    P_0   = float(flow.P[0])
    M_0   = float(flow.M[0])
    rho_0 = P_0 / (R_sp * T_0)
    a_0   = np.sqrt(gam * R_sp * T_0)
    U_0   = M_0 * a_0
    mu_0  = mu_ref * (T_0 / T_ref)**0.7

    # Use chamber length as effective run length for IC
    L_run = geom.L_c
    Re_L  = max(rho_0 * U_0 * L_run / mu_0, 1e3)
    theta = 0.036 * L_run / Re_L**0.2
    phi   = theta   # Reynolds analogy for initial φ ≈ θ

    H = 1.4   # Shape factor δ*/θ — turbulent BL (Coles profile, Appendix B)

    for i in range(n):
        M_i = float(flow.M[i])
        T_i = float(flow.T[i])
        P_i = float(flow.P[i])
        A_i = float(flow.A[i])
        r_i = np.sqrt(A_i / np.pi)

        # Local freestream properties
        mu_i  = mu_ref * (T_i / T_ref)**0.7      # Bartz Eq. 49 power-law
        rho_i = P_i / (R_sp * T_i)
        a_i   = np.sqrt(gam * R_sp * T_i)
        U_i   = M_i * a_i

        # Adiabatic wall temperature (recovery factor r = Pr^(1/3))
        rec = Pr**(1.0 / 3.0)
        fac = 1.0 + (gam - 1.0) / 2.0 * M_i**2
        T_aw_i = T_ref * (1.0 + rec * (gam - 1.0) / 2.0 * M_i**2) / fac
        T_w_i  = float(T_hw_arr[i])

        # --- Skin friction Cf (Eq. 37, film-temperature method) ---
        # Cf = 0.0256 / Rθ^(1/4) · [½(T_w/T + 1)]^(-0.6)
        Re_theta = rho_i * U_i * theta / mu_i
        Re_theta = max(Re_theta, 100.0)           # floor for numerical stability
        T_ratio  = 0.5 * (T_w_i / T_i + 1.0)
        Cf = 0.0256 / Re_theta**0.25 * T_ratio**(-0.6)

        # --- Stanton number Ch (Eq. 38, von Kármán Reynolds analogy) ---
        # Ch = (Cf/2) / [1 + 5·√(Cf/2)·(1 - Pr + ln((5Pr+1)/6))]
        sq = np.sqrt(max(Cf / 2.0, 1e-12))
        denom_vk = 1.0 + 5.0 * sq * (1.0 - Pr + np.log((5.0 * Pr + 1.0) / 6.0))
        Ch = (Cf / 2.0) / max(denom_vk, 1e-6)

        # h_gas from Stanton number: h = Ch · ρ · U · Cp
        h_gas[i] = Ch * rho_i * U_i * Cp

        # --- March ODEs to next station (forward Euler) ---
        if i < n - 1:
            dx_i = float(flow.x[i + 1] - flow.x[i])
            if dx_i <= 0:
                continue

            # Geometric derivatives (finite difference)
            r_next = np.sqrt(float(flow.A[i + 1]) / np.pi)
            M_next = float(flow.M[i + 1])
            drdz   = (r_next - r_i) / dx_i
            dMdz   = (M_next - M_i) / dx_i

            sqrt_wall = np.sqrt(1.0 + drdz**2)   # wall-length factor

            # Momentum thickness ODE (Eq. 25)
            # dθ/dz = (Cf/2)·√(1+(dr/dz)²)
            #       − θ·[(2−M²+H)/(M·(1+(γ−1)/2·M²))·dM/dz + (1/r)·dr/dz]
            if M_i > 1e-3 and r_i > 1e-6:
                M_coeff = (2.0 - M_i**2 + H) / (M_i * fac)
                dtheta = ((Cf / 2.0) * sqrt_wall
                          - theta * (M_coeff * dMdz + drdz / r_i))
            else:
                dtheta = (Cf / 2.0) * sqrt_wall

            # Energy thickness ODE (Eq. 30)
            # dφ/dz = Ch·(T_aw−T_w)/(T₀−T_w)·√(1+(dr/dz)²)
            #       − φ·[(1−M²)/(M·fac)·dM/dz + (1/r)·dr/dz − dT_w/dz/(T₀−T_w)]
            T_w_next = float(T_hw_arr[min(i + 1, n - 1)])
            dTw_dz   = (T_w_next - T_w_i) / dx_i
            dT_drive = T_ref - T_w_i       # T₀ − T_w
            if abs(dT_drive) < 1.0:
                dT_drive = 1.0              # avoid division by zero

            Ch_source = Ch * (T_aw_i - T_w_i) / dT_drive * sqrt_wall

            if M_i > 1e-3 and r_i > 1e-6:
                M_coeff_e = (1.0 - M_i**2) / (M_i * fac)
                dphi = (Ch_source
                        - phi * (M_coeff_e * dMdz + drdz / r_i
                                 - dTw_dz / dT_drive))
            else:
                dphi = Ch_source

            # Update with floor to prevent negative thickness
            theta = max(theta + dtheta * dx_i, 1e-8)
            phi   = max(phi + dphi * dx_i, 1e-8)

    # --- Normalize to calibrated simplified Bartz at the throat ---
    # The integral method gives the correct SHAPE (BL history effects) but
    # over-predicts the absolute magnitude at the throat because Re_θ ≪ Re_D*.
    # Following Bartz 1965 Sec II.B: use the calibrated simplified formula
    # (Eq. 50) at the throat as the reference magnitude and scale the
    # integral profile to match.  This preserves the physically-correct
    # decay in the divergent section while matching calibrated data.
    i_throat = int(np.argmin(flow.A))
    h_bl_throat = h_gas[i_throat]
    h_bartz_throat = _bartz_h(float(flow.M[i_throat]), float(flow.A[i_throat]),
                               float(T_hw_arr[i_throat]), cea, geom,
                               C=C_bartz)
    if h_bl_throat > 0.0:
        scale = h_bartz_throat / h_bl_throat
        h_gas *= scale

    return h_gas


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
              T_aw_eff: float = None,
              h_gas_override: float = None,
              C_bartz: float = 0.026,
              suppress: bool = False):
    """
    Gas-side heat quantities at axial station x.

    Parameters
    ----------
    T_aw_eff : float, optional
        Effective adiabatic wall temperature from film cooling model [K].
        If None, the bare Bartz T_aw is used.
    h_gas_override : float, optional
        Pre-computed h_gas from integral BL solver [W/(m²·K)].
        If None, the simplified Bartz (Eq. 50) is used.
    suppress : bool
        If True, return zero wall convective flux.  Used for stations
        shielded by an intact liquid or vaporising film (Ponomarenko
        RPA thermal p.10–11): all gas-side convective heat is absorbed
        by the film, none reaches the wall.  h_gas is still reported
        for diagnostics.

    Returns
    -------
    q_gas [W]      : heat transferred to ONE channel over step dx
    heatflux [W/m²]: local heat flux
    h_gas [W/(m²·K)]: gas-side HTC
    """
    chan_w, _, _, chan_land = chan_geom.at(x)
    h_g    = h_gas_override if h_gas_override is not None else _bartz_h(M, A, T_hw, cea, geom, C=C_bartz)

    if suppress:
        return 0.0, 0.0, h_g

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
# Coolant-side HTC helper (no heat calc / state advance)
# -----------------------------------------------------------------------
def _coolant_htc(x: float, T_cw: float, T_cool: float, P_cool: float,
                 s: float, chan_geom: ChannelGeometry,
                 geom: EngineGeometry, config: EngineConfig,
                 mdot_per_ch: float):
    """
    Compute coolant-side HTC and hydraulic quantities only.

    Returns (h_cool, Re, Nu, Dh, v, rho, f, props) where props is a
    CoolantState with .h (enthalpy), .Cp, etc.
    """
    fluid = config.coolant
    e     = config.wall_roughness

    chan_w, chan_h, _, chan_land = chan_geom.at(x)
    chan_area = chan_w * chan_h
    Dh       = 4.0 * chan_area / (2.0 * (chan_w + chan_h))

    props = get_coolant_props(T_cool, P_cool, fluid)
    rho, visc, cond, Cp = props.rho, props.viscosity, props.conductivity, props.Cp

    v  = mdot_per_ch / (rho * chan_area)
    Re = rho * v * Dh / visc
    Pr = visc * Cp / cond
    f  = _colebrook_f(Re, Dh, e)

    if fluid == "RP1":
        if Re < 2300.0:
            Nu = 4.36
        else:
            T_ratio = T_cool / max(T_cw, T_cool + 1.0)
            Nu = 0.021 * Re**0.8 * Pr**0.4 * (0.64 + 0.36 * T_ratio)
    elif fluid == "Methane":
        if Re < 2300.0:
            Nu = 4.36
        else:
            T_ratio = T_cool / max(T_cw, T_cool + 1.0)
            Nu = 0.0185 * Re**0.8 * Pr**0.4 * T_ratio**0.1
    else:
        f_sm  = _colebrook_f(Re, Dh, 0.0)
        xi    = f / f_sm if f_sm > 0.0 else 1.0
        eps_s = Re * (e / Dh) * np.sqrt(f / 8.0)
        B     = 4.7 * eps_s**0.2 if eps_s >= 7.0 else 4.5 + 0.57 * eps_s**0.75
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
    return h_cool, Re, Nu, Dh, v, rho, f, props


# -----------------------------------------------------------------------
# Coolant state advance helper (ΔP + enthalpy → T_new)
# -----------------------------------------------------------------------
def _advance_coolant(x: float, q_cool: float,
                     T_cool: float, P_cool: float, dx: float,
                     f: float, rho: float, v: float, Dh: float,
                     h_enth: float,
                     chan_geom: ChannelGeometry,
                     config: EngineConfig, mdot_per_ch: float):
    """Advance coolant state by one step given q_cool from the wall solver."""
    chan_w, chan_h, _, _ = chan_geom.at(x)
    x_next = max(x - dx, chan_geom.x_j[0])
    chan_w_n, chan_h_n, _, _ = chan_geom.at(x_next)

    chan_area   = chan_w   * chan_h
    chan_area_n = chan_w_n * chan_h_n
    Dh_n       = 4.0 * chan_area_n / (2.0 * (chan_w_n + chan_h_n))

    major = f * rho * v**2 * dx / (2.0 * Dh)
    if   chan_area_n > chan_area:
        K = ((Dh / Dh_n)**2 - 1.0)**2
    elif chan_area_n < chan_area:
        K = 0.5 - 0.167*(Dh_n/Dh) - 0.125*(Dh_n/Dh)**2 - 0.208*(Dh_n/Dh)**3
    else:
        K = 0.0
    minor      = K * rho * v**2 / 2.0
    P_cool_new = P_cool - (major + minor)

    h_next     = h_enth + q_cool / mdot_per_ch
    T_cool_new = get_T_from_enthalpy(h_next, P_cool_new, config.coolant)
    return T_cool_new, P_cool_new


# -----------------------------------------------------------------------
# 2-D wall conduction solver  (Betti / Pizzarelli quasi-2-D method)
# -----------------------------------------------------------------------
def _make_wall_grid(chan_w_half, chan_t, chan_h, land_half,
                    n_cw=8, n_lw=6, n_iw=8, n_ch=14):
    """
    Build aligned x/y node arrays for the unit cell.

    Returns x_nodes (nx,), y_nodes (ny,).
    """
    x_nodes = np.concatenate([
        np.linspace(0.0, chan_w_half, n_cw, endpoint=False),
        np.linspace(chan_w_half, chan_w_half + land_half, n_lw, endpoint=True),
    ])
    y_nodes = np.concatenate([
        np.linspace(0.0, chan_t, n_iw, endpoint=False),
        np.linspace(chan_t, chan_t + chan_h, n_ch, endpoint=True),
    ])
    return x_nodes, y_nodes


def _build_wall_2d(x_nodes, y_nodes, chan_w_half, chan_t,
                   h_gas, T_aw, h_cool, T_cool, k_w):
    """
    Assemble and solve the 2-D Laplace system for the wall unit cell.

    Returns T_field (N_solid,), node_map dict {(i,j): eq_index},
            solid_nodes list [(i,j), ...].
    """
    nx = len(x_nodes)
    ny = len(y_nodes)
    H  = y_nodes[-1]
    W  = x_nodes[-1]

    # Identify channel void: x < chan_w_half AND y > chan_t  (strict interior)
    # Boundary nodes AT chan_w_half or chan_t are SOLID (they carry BCs)
    is_void = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        for i in range(nx):
            if x_nodes[i] < chan_w_half - 1e-12 and y_nodes[j] > chan_t + 1e-12:
                is_void[j, i] = True

    # Map solid nodes to equation indices
    solid_nodes = []
    node_map = {}
    for j in range(ny):
        for i in range(nx):
            if not is_void[j, i]:
                node_map[(i, j)] = len(solid_nodes)
                solid_nodes.append((i, j))
    N = len(solid_nodes)

    # Build sparse system in COO format
    rows, cols, vals = [], [], []
    b = np.zeros(N)

    for eq, (i, j) in enumerate(solid_nodes):
        xi = x_nodes[i]
        yj = y_nodes[j]
        diag = 0.0

        # Helper: spacing
        dx_l = x_nodes[i] - x_nodes[i-1] if i > 0 else x_nodes[1] - x_nodes[0]
        dx_r = x_nodes[i+1] - x_nodes[i] if i < nx-1 else x_nodes[-1] - x_nodes[-2]
        dy_d = y_nodes[j] - y_nodes[j-1] if j > 0 else y_nodes[1] - y_nodes[0]
        dy_u = y_nodes[j+1] - y_nodes[j] if j < ny-1 else y_nodes[-1] - y_nodes[-2]
        dx_c = 0.5 * (dx_l + dx_r)
        dy_c = 0.5 * (dy_d + dy_u)

        def _add_coupling(ni, nj, d_near, d_center):
            """Add coupling to neighbor (ni,nj) or apply BC."""
            nonlocal diag
            coeff = k_w / (d_near * d_center)

            if (ni, nj) in node_map:
                # Normal interior coupling
                rows.append(eq); cols.append(node_map[(ni, nj)])
                vals.append(coeff)
                diag -= coeff
            elif 0 <= ni < nx and 0 <= nj < ny and is_void[nj, ni]:
                # Neighbor is channel void → Robin BC with h_cool, T_cool
                # -k dT/dn = h_cool * (T - T_cool)
                # Ghost: T_ghost = T_node - (h_cool * d_near / k) * (T_node - T_cool)
                # → coupling becomes: coeff * [T_node - h*d/k*(T_node - T_cool)]
                #                   = coeff * T_node * (1 - h*d/k) + coeff*h*d/k*T_cool
                Bi = h_cool * d_near / k_w
                diag -= coeff * Bi           # T_node coefficient
                b[eq] -= coeff * Bi * T_cool
            # else: neighbor is outside domain → check which boundary

        # --- Left neighbor (i-1) ---
        if i == 0:
            # x=0 boundary: adiabatic (symmetry) → reflect: T_{-1} = T_{+1}
            if (1, j) in node_map:
                coeff = k_w / (dx_l * dx_c)
                rows.append(eq); cols.append(node_map[(1, j)])
                vals.append(coeff)
                diag -= coeff
                # The left ghost mirrors the right neighbor, so we add coeff
                # to the right neighbor instead and the center gets -2*coeff
                # Actually: Laplace at boundary with ghost T_{-1}=T_{1}:
                # (T_{1} - 2T_0 + T_{-1})/dx² = (2T_1 - 2T_0)/dx²
                # So the left flux contributes coeff to T_1 and -coeff to T_0
                # We already added coeff to T_1 above.
                # The "normal" left contribution to diag is -coeff (ghost=T_1)
            else:
                pass  # i=0, j in void region — shouldn't happen since void is x<chan_w_half
        else:
            _add_coupling(i-1, j, dx_l, dx_c)

        # --- Right neighbor (i+1) ---
        if i == nx - 1:
            # x=W boundary: adiabatic (symmetry) → reflect
            if i > 0 and (i-1, j) in node_map:
                coeff = k_w / (dx_r * dx_c)
                rows.append(eq); cols.append(node_map[(i-1, j)])
                vals.append(coeff)
                diag -= coeff
            # Ghost mirror: adds coeff to T_{i-1} and -coeff to center
        else:
            _add_coupling(i+1, j, dx_r, dx_c)

        # --- Bottom neighbor (j-1) ---
        if j == 0:
            # y=0 boundary: Robin BC with h_gas, T_aw
            # -k dT/dy = h_gas * (T - T_aw)  [heat INTO wall when T < T_aw]
            # Ghost: T_{j-1} = T_{j+1} + 2*dy*h_gas/k * (T_aw - T_j)
            # Substitute: bottom flux = k*(T_0 - T_{-1})/dy/dy_c
            #   = k * [T_0 - T_1 - 2dy*h/k*(T_aw-T_0)] / (dy * dy_c)
            # The j+1 coupling is handled separately. Here add the ghost effect:
            Bi = h_gas * dy_d / k_w
            coeff_ghost = k_w / (dy_d * dy_c)
            # Ghost reflection: T_{-1} = T_{1} + 2*h_gas*dy/k * (T_aw - T_0)
            # In the stencil, the j-1 term contributes coeff_ghost * T_{-1}
            # = coeff_ghost * [T_1 + 2*Bi*(T_aw - T_0)]
            # = coeff_ghost * T_1 + coeff_ghost * 2*Bi*T_aw - coeff_ghost*2*Bi*T_0
            if (i, 1) in node_map:
                rows.append(eq); cols.append(node_map[(i, 1)])
                vals.append(coeff_ghost)
                diag -= coeff_ghost
            diag -= coeff_ghost * 2.0 * Bi
            b[eq] -= coeff_ghost * 2.0 * Bi * T_aw
        else:
            _add_coupling(i, j-1, dy_d, dy_c)

        # --- Top neighbor (j+1) ---
        if j == ny - 1:
            # y=H boundary: adiabatic → reflect T_{ny} = T_{ny-2}
            if j > 0 and (i, j-1) in node_map:
                coeff = k_w / (dy_u * dy_c)
                rows.append(eq); cols.append(node_map[(i, j-1)])
                vals.append(coeff)
                diag -= coeff
        else:
            _add_coupling(i, j+1, dy_u, dy_c)

        # Center node diagonal
        rows.append(eq); cols.append(eq); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    T_field = spla.spsolve(A, b)

    return T_field, node_map, solid_nodes, x_nodes, y_nodes, is_void


def _extract_wall_results(T_field, node_map, solid_nodes, x_nodes, y_nodes,
                          is_void, chan_w_half, chan_t, h_cool, T_cool):
    """
    Extract T_hw (avg/max at gas side), T_cw (avg over channel walls),
    and q_cool per unit axial length for the HALF-cell [W/m].
    """
    nx = len(x_nodes)
    ny = len(y_nodes)

    # --- T_hw: area-weighted average at y=0 ---
    T_hw_sum = 0.0
    T_hw_wt  = 0.0
    T_hw_max = 0.0
    for i in range(nx):
        if (i, 0) not in node_map:
            continue
        T_val = T_field[node_map[(i, 0)]]
        # Cell width for this node
        if i == 0:
            w = 0.5 * (x_nodes[1] - x_nodes[0])
        elif i == nx - 1:
            w = 0.5 * (x_nodes[-1] - x_nodes[-2])
        else:
            w = 0.5 * (x_nodes[i+1] - x_nodes[i-1])
        T_hw_sum += T_val * w
        T_hw_wt  += w
        T_hw_max  = max(T_hw_max, T_val)
    T_hw_avg = T_hw_sum / T_hw_wt if T_hw_wt > 0 else T_hw_max

    # --- T_cw: area-weighted average over channel-facing boundaries ---
    # Channel base: y=chan_t, x < chan_w_half (nodes at y=chan_t that border void above)
    # Channel side: x=chan_w_half, chan_t < y (nodes at x=chan_w_half that border void left)
    T_cw_sum = 0.0
    T_cw_wt  = 0.0
    q_cool_half = 0.0  # heat per unit axial length for half-cell [W/m]

    # Channel base (y index where y_nodes ≈ chan_t)
    j_base = np.searchsorted(y_nodes, chan_t - 1e-12)
    # Ensure we found the right index
    if j_base < ny and abs(y_nodes[j_base] - chan_t) < 1e-10:
        for i in range(nx):
            if x_nodes[i] >= chan_w_half - 1e-12:
                break  # only x < chan_w_half
            if (i, j_base) not in node_map:
                continue
            T_val = T_field[node_map[(i, j_base)]]
            if i == 0:
                w = 0.5 * (x_nodes[1] - x_nodes[0])
            else:
                w = 0.5 * (x_nodes[min(i+1, nx-1)] - x_nodes[max(i-1, 0)])
            T_cw_sum += T_val * w
            T_cw_wt  += w
            q_cool_half += h_cool * (T_val - T_cool) * w

    # Channel side (x index where x_nodes ≈ chan_w_half)
    i_side = np.searchsorted(x_nodes, chan_w_half - 1e-12)
    if i_side < nx and abs(x_nodes[i_side] - chan_w_half) < 1e-10:
        for j in range(ny):
            if y_nodes[j] <= chan_t + 1e-12:
                continue  # only y > chan_t
            if (i_side, j) not in node_map:
                continue
            T_val = T_field[node_map[(i_side, j)]]
            if j == ny - 1:
                w = 0.5 * (y_nodes[-1] - y_nodes[-2])
            else:
                w = 0.5 * (y_nodes[min(j+1, ny-1)] - y_nodes[max(j-1, 0)])
            T_cw_sum += T_val * w
            T_cw_wt  += w
            q_cool_half += h_cool * (T_val - T_cool) * w

    T_cw_avg = T_cw_sum / T_cw_wt if T_cw_wt > 0 else T_hw_avg

    return T_hw_avg, T_hw_max, T_cw_avg, q_cool_half


def _solve_wall_2d(x: float, M: float, A: float,
                   T_cool: float, P_cool: float, s: float, dx: float,
                   chan_geom: ChannelGeometry,
                   cea: CEAResult, geom: EngineGeometry,
                   config: EngineConfig, mdot_per_ch: float,
                   T_hw_guess: float, T_cw_guess: float,
                   T_aw_eff: float = None,
                   h_gas_override: float = None,
                   C_bartz: float = 0.026,
                   suppress: bool = False,
                   tol: float = 1.0, max_iter: int = 15):
    """
    2-D wall conduction solve at one axial station.

    Iterates Bartz h_gas(T_hw) and Niino h_cool(T_cw) with the 2-D
    Laplace solve until T_hw and T_cw converge.

    Parameters
    ----------
    h_gas_override : float, optional
        Pre-computed h_gas from integral BL solver [W/(m²·K)].
        If provided, Bartz σ iteration on h_gas is skipped — h_gas is
        held fixed and only h_cool iterates with T_cw.

    Returns
    -------
    T_hw, T_cw, T_hw_max, q_cool,
    h_gas, h_cool, Re, Nu, Dh, v, rho, f, h_enth
    """
    chan_w, chan_h, chan_t_val, chan_land = chan_geom.at(x)
    chan_w_half = chan_w / 2.0
    land_half  = chan_land / 2.0

    # Build grid once per station (geometry fixed)
    x_nodes, y_nodes = _make_wall_grid(chan_w_half, chan_t_val, chan_h, land_half)

    T_hw = T_hw_guess
    T_cw = T_cw_guess
    T_aw_v = T_aw_eff if T_aw_eff is not None else _T_aw(M, cea)

    h_g = 0.0
    h_c_out = 0.0
    Re_out = Nu_out = Dh_out = v_out = rho_out = f_out = h_enth_out = 0.0

    for it in range(max_iter):
        # Gas-side HTC (still reported for diagnostics even when suppressed)
        if h_gas_override is not None:
            h_g = h_gas_override
        else:
            h_g = _bartz_h(M, A, T_hw, cea, geom, C=C_bartz)

        # Coolant-side HTC
        h_c, Re_c, Nu_c, Dh_c, v_c, rho_c, f_c, props = _coolant_htc(
            x, T_cw, T_cool, P_cool, s, chan_geom, geom, config, mdot_per_ch)
        h_c_out, Re_out, Nu_out = h_c, Re_c, Nu_c
        Dh_out, v_out, rho_out, f_out = Dh_c, v_c, rho_c, f_c
        h_enth_out = props.h

        # When film shields the wall, convective gas-side coupling is zero.
        # Pass h_gas_eff=0 to the Laplace solve so the wall equilibrates
        # to the coolant only (adiabatic hot face).
        h_g_eff = 0.0 if suppress else h_g

        # Solve 2-D wall
        T_field, node_map, solid_nodes, xn, yn, is_void = _build_wall_2d(
            x_nodes, y_nodes, chan_w_half, chan_t_val,
            h_g_eff, T_aw_v, h_c, T_cool, config.wall_k)

        # Extract results
        T_hw_new, T_hw_max, T_cw_new, q_half = _extract_wall_results(
            T_field, node_map, solid_nodes, xn, yn, is_void,
            chan_w_half, chan_t_val, h_c, T_cool)

        if abs(T_hw_new - T_hw) < tol and abs(T_cw_new - T_cw) < tol:
            T_hw, T_cw = T_hw_new, T_cw_new
            break
        T_hw, T_cw = T_hw_new, T_cw_new

    # q_cool for one full channel over axial step dx
    # q_half is heat per unit axial length for half-cell → ×2 (symmetry) × dx
    q_cool = q_half * 2.0 * dx

    return (T_hw, T_cw, T_hw_max, q_cool,
            h_g, h_c_out, Re_out, Nu_out, Dh_out, v_out, rho_out, f_out,
            h_enth_out)


# -----------------------------------------------------------------------
# Newton-Raphson: find T_hw, T_cw at one axial station (1-D path)
# -----------------------------------------------------------------------
def _newton_solve(x: float, M: float, A: float,
                  T_cool: float, P_cool: float, s: float, dx: float,
                  chan_geom: ChannelGeometry,
                  cea: CEAResult, geom: EngineGeometry,
                  config: EngineConfig, mdot_per_ch: float,
                  T_hw0: float, T_cw0: float,
                  tol: float = 0.1, max_iter: int = 50,
                  T_aw_eff: float = None,
                  suppress: bool = False):
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
                              T_aw_eff=T_aw_eff, suppress=suppress)
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
                  T_aw_eff: np.ndarray = None,
                  cea_per_station: list = None,
                  phase_code: np.ndarray = None) -> ThermalSolution:
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

    # Per-station mdot per channel.  For bifurcating designs the channel
    # count varies along x.  Float values are kept (not rounded) so the
    # Y-cusp transition smooths continuously across the split band.
    if chan_geom.n_chan is not None:
        n_chan_at_flow = np.interp(flow.x, chan_geom.x_j,
                                   chan_geom.n_chan.astype(float))
        mdot_per_ch_arr = config.mdot_coolant / np.maximum(n_chan_at_flow, 1.0)
    else:
        n_chan_at_flow  = np.full(n, float(config.N_channels))
        mdot_per_ch_arr = np.full(n, config.mdot_coolant / config.N_channels)
    mdot_per_ch = float(mdot_per_ch_arr[int(np.argmin(flow.A))])  # throat value (for printout only)

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
    cea_override_active = (cea_per_station is not None)
    phase_active = (phase_code is not None)
    print(f"\n--- Thermal Solver ---")
    n_ch_min = int(round(float(n_chan_at_flow.min())))
    n_ch_max = int(round(float(n_chan_at_flow.max())))
    if n_ch_min == n_ch_max:
        n_ch_str = f"{n_ch_min}"
        mdot_str = f"{(config.mdot_coolant/n_ch_min)*1000:.2f} g/s"
    else:
        n_ch_str = f"{n_ch_min}–{n_ch_max} (bifurcating)"
        mdot_str = (f"{(config.mdot_coolant/n_ch_max)*1000:.2f}–"
                    f"{(config.mdot_coolant/n_ch_min)*1000:.2f} g/s")
    print(f"  Fluid: {config.coolant}  "
          f"N_ch: {n_ch_str}  "
          f"mdot/ch: {mdot_str}  "
          f"T_in: {config.T_coolant_inlet:.0f} K  "
          f"P_in: {config.P_coolant_inlet/1e5:.1f} bar")
    if film_active:
        print(f"  Film cooling ACTIVE — T_aw_eff supplied ({config.film_fraction*100:.1f}% film)")
    if cea_override_active:
        print(f"  Gas-side Bartz: per-station surface-layer CEA override "
              f"(δ_rel={config.film_BL_thickness:.3f})")
    if config.wall_2d:
        print(f"  Wall model: 2-D conduction (Betti quasi-2D)")
    else:
        print(f"  Wall model: 1-D flat-plate + fin")
    if config.use_integral_bl:
        print(f"  Gas-side HTC: Bartz 1965 integral boundary layer")
    else:
        print(f"  Gas-side HTC: Simplified Bartz (Eq. 50)")

    for outer in range(max_outer):
        T_hw_prev = T_hw_arr.copy()
        T_cw_prev = T_cw_arr.copy()

        # --- Pre-compute h_gas array if using integral BL method ---
        h_gas_bl = None
        if config.use_integral_bl:
            h_gas_bl = solve_boundary_layer(flow, cea, geom, T_hw_arr,
                                                C_bartz=config.C_bartz)

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

            # Per-station CEA (surface-layer composition under film cooling).
            # Falls back to global cea when no override supplied.
            cea_k = cea_per_station[k] if cea_override_active else cea

            # Wall-convective suppress: stations inside an intact liquid or
            # vaporising film (Ponomarenko phases 1 & 2) see zero wall q_conv.
            suppress_k = bool(phase_active and phase_code[k] in (1, 2))

            # Pre-computed h_gas from integral BL (None = use simplified Bartz)
            h_gas_k = float(h_gas_bl[k]) if h_gas_bl is not None else None

            mdot_per_ch_k = float(mdot_per_ch_arr[k])

            if config.wall_2d:
                # --- 2-D wall conduction path ---
                (T_hw_nwt, T_cw_nwt, _, q_c,
                 h_g, h_c, Re, Nu, Dh, v, rho, f_c,
                 h_enth) = _solve_wall_2d(
                    x, M, A, T_cool, P_cool, s, dx,
                    chan_geom, cea_k, geom, config, mdot_per_ch_k,
                    T_hw_nwt, T_cw_nwt, T_aw_eff=T_aw_k,
                    h_gas_override=h_gas_k,
                    C_bartz=config.C_bartz,
                    suppress=suppress_k)

                T_hw_arr[k] = relax * T_hw_nwt + (1.0 - relax) * T_hw_prev[k]
                T_cw_arr[k] = relax * T_cw_nwt + (1.0 - relax) * T_cw_prev[k]

                if suppress_k:
                    hf  = 0.0
                    q_g = 0.0
                else:
                    T_aw_v = T_aw_k if T_aw_k is not None else _T_aw(M, cea_k)
                    hf = h_g * (T_aw_v - T_hw_arr[k])
                    chan_w_k, _, _, chan_land_k = chan_geom.at(x)
                    q_g = hf * (chan_w_k + chan_land_k) * dx

                T_cool_new, P_cool_new = _advance_coolant(
                    x, q_c, T_cool, P_cool, dx, f_c, rho, v, Dh,
                    h_enth, chan_geom, config, mdot_per_ch_k)
            else:
                # --- 1-D wall conduction path (original) ---
                T_hw_nwt, T_cw_nwt = _newton_solve(
                    x, M, A, T_cool, P_cool, s, dx,
                    chan_geom, cea_k, geom, config, mdot_per_ch_k,
                    T_hw_nwt, T_cw_nwt,
                    T_aw_eff=T_aw_k,
                    suppress=suppress_k)

                T_hw_arr[k] = relax * T_hw_nwt + (1.0 - relax) * T_hw_prev[k]
                T_cw_arr[k] = relax * T_cw_nwt + (1.0 - relax) * T_cw_prev[k]

                q_g, hf, h_g = _gas_heat(x, M, A, T_hw_arr[k], dx, chan_geom,
                                          cea_k, geom, T_aw_eff=T_aw_k,
                                          h_gas_override=h_gas_k,
                                          C_bartz=config.C_bartz,
                                          suppress=suppress_k)
                (q_c, T_cool_new, P_cool_new,
                 h_c, Re, Nu, Dh, v, rho) = _coolant_heat(
                    x, T_cw_arr[k], T_cool, P_cool, s, dx,
                    chan_geom, geom, config, mdot_per_ch_k)

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

            # Update guess for next (upstream) station
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
    if (config.N_channels_throat is not None
            and config.N_channels_chamber is not None
            and config.N_channels_throat != config.N_channels_chamber):
        ncc_str = f'Ncc = {config.N_channels_throat}/{config.N_channels_chamber}'
    else:
        ncc_str = f'Ncc = {config.N_channels}'
    fig.suptitle(
        f'{config.fuel}/{config.oxidizer}  '
        f'Pc = {config.P_c/1e5:.0f} bar  '
        f'O/F = {config.OF}  '
        f'{ncc_str}  '
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
