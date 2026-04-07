"""
RocketThermalAnalysis.py
Generic regenerative cooling thermal analysis for liquid rocket engines.

Supported coolants : RP1, LH2, CH4  (via fuelType flag)
Supported oxidizers: LOX (combustion gas properties via Cantera)
Gas-side HTC       : Bartz equation with sigma correction
Coolant-side HTC   : Gnielinski (RP1, CH4) | Niino (LH2)
Wall conduction    : 1-D resistance with fin efficiency

Based on MixtureOptimization.py (SSME LH2/LOX, validated against Wang & Luong 1994
and Betti 2014).  Extend for any engine by filling in the ENGINE CONFIG section.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scipy
from scipy.interpolate import interp1d
from adjustText import adjust_text
from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP
import cantera as ct

# -----------------------------------------------------------------------
# REFPROP via CoolProp - coolant fluid objects
# -----------------------------------------------------------------------
REFPROP_PATH = '/home/jacob/Documents/REFPROP/'
REFPROP_LIB  = '/home/jacob/Documents/REFPROP-cmake/build/librefprop.so'
CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, REFPROP_PATH)
CP.set_config_string(CP.ALTERNATIVE_REFPROP_LIBRARY_PATH, REFPROP_LIB)

# Huber RP-1 surrogate  (MDEC + 5MC9 + C12 + C7CC6)
_RP1_FLUID_STR  = "MDEC.FLD&5MC9.FLD&C12.FLD&C7CC6.FLD"
_RP1_MOLE_FRACS = [0.354, 0.150, 0.183, 0.313]
rp1_state = CP.AbstractState("REFPROP", _RP1_FLUID_STR)
rp1_state.set_mole_fractions(_RP1_MOLE_FRACS)

lh2_state = CP.AbstractState("REFPROP", "Hydrogen")
ch4_state  = CP.AbstractState("REFPROP", "Methane")

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
Ru = 8.31446261815324  # Universal gas constant [J/mol·K]


# -----------------------------------------------------------------------
# Plotting helper
# -----------------------------------------------------------------------
def create_plot(x_axis, y_axis, x_label, y_label, title):
    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    texts = [
        plt.annotate(f"{x_axis[0]:.4f}, {y_axis[0]:.4f}",  xy=(x_axis[0],  y_axis[0]),  ha='center', bbox=dict(boxstyle="round", fc="w")),
        plt.annotate(f"{x_axis[-1]:.4f}, {y_axis[-1]:.4f}", xy=(x_axis[-1], y_axis[-1]), ha='center', bbox=dict(boxstyle="round", fc="w")),
    ]
    adjust_text(texts)


# -----------------------------------------------------------------------
# Engine geometry helpers
# -----------------------------------------------------------------------
def calc_exit_mach_num(area_ratio, gam, M_e):
    def equation(Me, gam, area_ratio):
        return 1/Me * ((1 + ((gam-1)/2)*Me**2) / (1 + (gam-1)/2))**((gam+1)/(2*(gam-1))) - area_ratio
    return scipy.fsolve(equation, M_e, args=(gam, area_ratio))

def calc_exit_pressure(P_c, gam, M_e):
    return P_c * (1 + ((gam-1)/2)*M_e**2)**(-gam/(gam-1))

def calc_thrust_coeff(P_c, P_e, gam, area_ratio):
    return np.sqrt((2*gam**2)/(gam-1) * (2/(gam+1))**((gam+1)/(gam-1)) * (1 - (P_e/P_c)**((gam-1)/gam))) + area_ratio*(P_e/P_c)

def calc_nozzle_throat_area(F_Vac, C_F, P_c):
    return F_Vac / (C_F * P_c)

def calc_nozzle_throat_temp(T_c, gam):
    return T_c * (1 + (gam-1)/2)**-1

def calc_nozzle_throat_pressure(P_c, gam):
    return P_c * (1 + (gam-1)/2)**(-gam/(gam-1))

def calc_chamber_volume(A_t, L_star_m):
    """L_star_m: characteristic length in metres."""
    return A_t * L_star_m

def calc_chamber_length(A_c, V_c):
    return V_c / A_c

def engine_geometry(gam, P_c, T_c, F_Vac, expRatio, L_star_m, contChamber, R_specific, showPlots=False):
    """
    Compute engine geometry given CEA outputs and design choices.

    Parameters
    ----------
    gam         : float  - specific heat ratio
    P_c         : float  - chamber pressure [Pa]
    T_c         : float  - chamber temperature [K]
    F_Vac       : float  - vacuum thrust [N]
    expRatio    : float  - nozzle area expansion ratio (Ae/At)
    L_star_m    : float  - characteristic length [m]
    contChamber : float  - contraction ratio (A_c/A_t)
    R_specific  : float  - specific gas constant of combustion products [J/kg·K]
    showPlots   : bool

    Returns
    -------
    A_c, A_t, A_e, L_c, mdot_chamber
    """
    area_ratio_array = np.arange(1.1, expRatio + 0.1, 0.1)
    M_e = 1.2
    A_t_arr, A_e_arr, C_F_arr, P_e_arr, m_dot_arr = [], [], [], [], []

    for ar in area_ratio_array:
        M_e = calc_exit_mach_num(ar, gam, M_e)
        P_e = calc_exit_pressure(P_c, gam, M_e)
        C_F = calc_thrust_coeff(P_c, P_e, gam, ar)
        A_t = calc_nozzle_throat_area(F_Vac, C_F, P_c)
        T_t = calc_nozzle_throat_temp(T_c, gam)
        P_t = calc_nozzle_throat_pressure(P_c, gam)
        rho_t = P_t / (R_specific * T_t)
        V_t   = np.sqrt(gam * R_specific * T_t)
        A_t_arr.append(A_t)
        A_e_arr.append(A_t * ar)
        C_F_arr.append(C_F)
        P_e_arr.append(P_e)
        m_dot_arr.append(rho_t * V_t * A_t)

    # Pick the slot closest to the desired expansion ratio
    idx = np.argmin(np.abs(area_ratio_array - expRatio))
    A_t = A_t_arr[idx]
    A_e = A_e_arr[idx]
    A_c = A_t * contChamber
    V_c = calc_chamber_volume(A_t, L_star_m)
    L_c = calc_chamber_length(A_c, V_c)
    mdot = m_dot_arr[idx]

    print(f'A_t={A_t:.6f} m²  A_c={A_c:.6f} m²  A_e={A_e:.6f} m²  L_c={L_c:.4f} m  mdot={mdot:.4f} kg/s')
    return A_c, A_t, A_e, L_c, mdot


# -----------------------------------------------------------------------
# Nozzle contour  (parameterised — no RS25 hardcodes)
# -----------------------------------------------------------------------
def calc_radius(x, A_c, A_t, A_e, L_c, geo_params=None):
    """
    Radius y(x) [m] along the engine axis (x=0 at injector).

    geo_params : dict with keys
        L_e     - axial length of straight cylindrical section [m]
        theta1  - convergence half-angle [deg]
        thetaD  - initial divergence half-angle [deg]
        alpha   - nozzle exit half-angle [deg]
        R1_mult - R1 = R1_mult * R_throat  (inner fillet of convergence)
        RU      - large convergence radius of curvature [m]
        RD      - divergence throat radius of curvature [m]

    If geo_params is None the function falls back to RS25 defaults so
    MixtureOptimization.py stays compatible.
    """
    # Derived geometry
    D_c = 2 * np.sqrt(A_c / np.pi)
    D_t = 2 * np.sqrt(A_t / np.pi)
    R_T = D_t / 2  # throat radius [m]
    R_e = np.sqrt(A_e / np.pi)

    if geo_params is not None:
        L_e    = geo_params['L_e']
        theta1 = geo_params['theta1']
        thetaD = geo_params['thetaD']
        alpha  = geo_params['alpha']
        R1     = geo_params['R1_mult'] * R_T
        RU     = geo_params['RU']
        RD     = geo_params['RD']
    else:
        # RS25 fallback (legacy, all converted to metres)
        L_e    = 5.339   * 0.0254
        theta1 = 25.4157
        thetaD = 37.0
        alpha  = 5.3738
        R1     = 1.73921 * R_T
        RU     = 5.1527  * 0.0254
        RD     = 2.019   * 0.0254
        print("DEFAULTED TO RS25 PARAMETERS")

    expansion_ratio = A_e / A_t
    R_c = D_c / 2

    m = (R_T + RU - R_c + R1 - (RU + R1) * np.cos(np.deg2rad(theta1))) / \
        (L_c - L_e - (RU + R1) * np.sin(np.deg2rad(theta1)))

    if x <= L_e:
        r = R_c
    elif x <= L_e + R1 * np.sin(np.deg2rad(theta1)):
        r = np.sqrt(max(R1**2 - (x - L_e)**2, 0.0)) + R_c - R1
    elif x <= L_c - RU * np.sin(np.deg2rad(theta1)):
        r = m * x + R_c - R1 + R1 * np.cos(np.deg2rad(theta1)) - m * (L_e + R1 * np.sin(np.deg2rad(theta1)))
    elif x <= L_c:
        r = -np.sqrt(max(RU**2 - (x - L_c)**2, 0.0)) + R_T + RU
    elif x <= L_c + RD * np.sin(np.deg2rad(thetaD)):
        r = -np.sqrt(max(RD**2 - (x - L_c)**2, 0.0)) + R_T + RD
    else:
        # Bezier bell curve
        N_x = RD * np.cos(np.deg2rad(thetaD) - np.pi/2)
        N_y = RD * np.sin(np.deg2rad(thetaD) - np.pi/2) + RD + R_T
        E_x = 0.8 * (R_T * (np.sqrt(expansion_ratio) - 1) + RU * (1/np.cos(np.deg2rad(alpha - 1)))) / np.tan(np.deg2rad(15))
        E_y = R_e
        denom = np.tan(np.deg2rad(thetaD)) - np.tan(np.deg2rad(alpha))
        Q_x = (E_y - np.tan(np.deg2rad(alpha))*E_x - N_y + np.tan(np.deg2rad(thetaD))*N_x) / denom
        Q_y = (np.tan(np.deg2rad(thetaD))*(D_c/2 - np.tan(np.deg2rad(alpha))*E_x) -
               np.tan(np.deg2rad(alpha))*(N_y - np.tan(np.deg2rad(thetaD))*N_x)) / denom
        a = N_x - 2*Q_x + E_x
        b = 2*(Q_x - N_x)
        c = N_x - x
        disc = max(b**2 - 4*a*c, 0.0)
        t_x = (-b + np.sqrt(disc)) / (2*a) if abs(a) > 1e-14 else -c/b
        r = (1 - t_x)**2 * N_y + 2*(t_x - t_x**2)*Q_y + t_x**2 * E_y
    return float(r)

# -----------------------------------------------------------------------
# 1-D flow ODEs  (RK4 solver)
# keydata = [A_c, A_t, A_e, L_c, gam, Cp, dF_dx, dQ_dx, R_specific, geo_params]
# -----------------------------------------------------------------------
def dN_dx(x, y, h, m, keydata):
    N, P, T = y[0], y[1], y[2]
    A_c, A_t, A_e, L_c = keydata[0], keydata[1], keydata[2], keydata[3]
    gam, Cp = keydata[4], keydata[5]
    dF = keydata[6][m]
    dQ = keydata[7][m]
    R_sp = keydata[8]
    geo  = keydata[9] if len(keydata) > 9 else None

    A    = np.pi * calc_radius(x,    A_c, A_t, A_e, L_c, geo)**2
    Ap   = np.pi * calc_radius(x+h,  A_c, A_t, A_e, L_c, geo)**2
    Am   = np.pi * calc_radius(x-h,  A_c, A_t, A_e, L_c, geo)**2
    dA   = (Ap - Am) / (2*h)

    return ((N/(1-N)) * ((1+gam*N)/(Cp*T)) * dQ +
            (N/(1-N)) * ((2+(gam-1)*N)/(R_sp*T)) * dF -
            (N/(1-N)) * ((2+(gam-1)*N)/A) * dA)


def dP_dx(x, y, h, m, keydata):
    N, P, T = y[0], y[1], y[2]
    A_c, A_t, A_e, L_c = keydata[0], keydata[1], keydata[2], keydata[3]
    gam, Cp = keydata[4], keydata[5]
    dF = keydata[6][m]
    dQ = keydata[7][m]
    R_sp = keydata[8]
    geo  = keydata[9] if len(keydata) > 9 else None

    A    = np.pi * calc_radius(x,    A_c, A_t, A_e, L_c, geo)**2
    Ap   = np.pi * calc_radius(x+h,  A_c, A_t, A_e, L_c, geo)**2
    Am   = np.pi * calc_radius(x-h,  A_c, A_t, A_e, L_c, geo)**2
    dA   = (Ap - Am) / (2*h)

    return (-(P/(1-N)) * ((gam*N)/(Cp*T)) * dQ -
             (P/(1-N)) * ((1+(gam-1)*N)/(R_sp*T)) * dF +
             (P/(1-N)) * ((gam*N)/A) * dA)


def dT_dx(x, y, h, m, keydata):
    N, P, T = y[0], y[1], y[2]
    A_c, A_t, A_e, L_c = keydata[0], keydata[1], keydata[2], keydata[3]
    gam, Cp = keydata[4], keydata[5]
    dF = keydata[6][m]
    dQ = keydata[7][m]
    R_sp = keydata[8]
    geo  = keydata[9] if len(keydata) > 9 else None

    A    = np.pi * calc_radius(x,    A_c, A_t, A_e, L_c, geo)**2
    Ap   = np.pi * calc_radius(x+h,  A_c, A_t, A_e, L_c, geo)**2
    Am   = np.pi * calc_radius(x-h,  A_c, A_t, A_e, L_c, geo)**2
    dA   = (Ap - Am) / (2*h)

    return ((T/(1-N)) * ((1-gam*N)/(Cp*T)) * dQ -
            (T/(1-N)) * ((gam-1)*N/(R_sp*T)) * dF +
            (T/(1-N)) * ((gam-1)*N/A) * dA)


def derivs(x, y, h, m, keydata):
    return [dN_dx(x, y, h, m, keydata),
            dP_dx(x, y, h, m, keydata),
            dT_dx(x, y, h, m, keydata)]


def rk4(x, y, n, h, m, keydata):
    ym = [0]*n
    ye = [0]*n
    k = [derivs(x, y, h, m, keydata)]
    for i in range(n): ym[i] = y[i] + k[0][i]*h/2
    k.append(derivs(x+h/2, ym, h, m, keydata))
    for i in range(n): ym[i] = y[i] + k[1][i]*h/2
    k.append(derivs(x+h/2, ym, h, m, keydata))
    for i in range(n): ye[i] = y[i] + k[2][i]*h
    k.append(derivs(x+h, ye, h, m, keydata))
    for i in range(n):
        y[i] = y[i] + h*(k[0][i] + 2*(k[1][i]+k[2][i]) + k[3][i])/6
    return x+h, y


def integrator(x, y, n, h, xend, m, keydata):
    while True:
        if xend - x < h:
            h = xend - x
        x, y = rk4(x, y, n, h, m, keydata)
        if x >= xend:
            break
    return x, y


def calc_flow_data(xi, xf, dx, M_c, P_c, T_c, keydata):
    n  = 3
    y  = [M_c**2, P_c, T_c]
    x  = xi
    m  = 0
    xp_m, yp_m = [], []
    while True:
        xend = min(x + dx, xf)
        x, y = integrator(x, y, n, dx, xend, m, keydata)
        m += 1
        xp_m.append(x)
        yp_m.append(y.copy())
        if x >= xf:
            break
    return dx, xp_m, yp_m

# -----------------------------------------------------------------------
# Heat transfer — gas side
# -----------------------------------------------------------------------
def calc_A_gas(dx, x, engine_info):
    return (2 * np.pi * engine_info.get_radius(x) * dx) / engine_info.Ncc

def calc_h_gas(x, y, dx, T_hw, T_star, engine_info):
    N, P, T = y[0], y[1], y[2]
    Dt    = 2 * np.sqrt(engine_info.A_t / np.pi)
    RU    = engine_info.RU
    RD    = engine_info.RD
    area  = engine_info.calc_area(x)
    A_t   = engine_info.A_t
    T_s   = engine_info.T_c
    gam   = engine_info.gam
    P_c   = engine_info.P_c
    C_star= engine_info.C_star

    eta, llambda, Cp, Pr, rho = engine_info.calc_transport_properties(T_s, P_c)

    sigma = 1.0 / ((0.5*(T_hw/T_s)*(1+(gam-1)/2*N) + 0.5)**0.68 * (1+(gam-1)/2*N)**0.12)
    h_gas = ((0.026/Dt**0.2) * ((eta**0.2 * Cp)/Pr**0.6) *
             (P_c/C_star)**0.8 * (Dt/(0.5*RU + 0.5*RD))**0.1 * (A_t/area)**0.9 * sigma)
    return h_gas

def calc_T_aw(x, y, T_hw, T_star, engine_info):
    N, P, T = y[0], y[1], y[2]
    T_s  = engine_info.T_c
    gam  = engine_info.gam
    eta, llambda, Cp, Pr, rho = engine_info.calc_transport_properties(T_star, P)
    return T_s * ((1 + Pr**0.33 * (gam-1)/2 * N) / (1 + (gam-1)/2 * N))

def calc_frictionFactor(Re, Dh, initial_guess, e):
    def g(f, e, Dh, Re):
        return 1/np.sqrt(f) + 2*np.log10(e/(3.7065*Dh) + 2.5226/(Re*np.sqrt(f)))
    return scipy.fsolve(g, initial_guess, args=(e, Dh, Re))[0]

def calc_q_gas(dx, x, y, T_hw, engine_info):
    N, P, T = y[0], y[1], y[2]
    chan_w, chan_h, chan_t, chan_land = engine_info.get_geo_interp(x, dx)
    gam = engine_info.gam
    v   = np.sqrt(N * gam * engine_info.R_specific * T)
    T_star = (1 + 0.032*N + 0.58*(T_hw/T - 1)) * T
    A_gas  = calc_A_gas(dx, x, engine_info)
    h_gas  = calc_h_gas(x, y, dx, T_hw, T_star, engine_info)
    T_aw   = calc_T_aw(x, y, T_hw, T_star, engine_info)
    R_hot  = 1.0 / (h_gas * (chan_w + chan_land) * dx)
    q_gas  = (T_aw - T_hw) / R_hot
    heatflux = h_gas * (T_aw - T_hw)

    eta, llambda, Cp, Pr, rho = engine_info.calc_transport_properties(T_star, P)
    r = engine_info.get_radius(x)
    Re_gas = rho * v * (2*r) / eta
    f_gas  = calc_frictionFactor(Re_gas, 2*r, 0.001, engine_info.e)
    dF     = f_gas * dx * v**2 / (2*(2*r))

    return q_gas, heatflux, h_gas, A_gas, R_hot, (chan_w+chan_land)*dx, dF

# -----------------------------------------------------------------------
# Heat transfer — coolant side  (multi-fuel)
# -----------------------------------------------------------------------
def _get_coolant_props(T_coolant, P_coolant, fuelType):
    """
    Return (rho, h, viscosity, conductivity, Cp) for the coolant.
    Uses CoolProp REFPROP backend.
    """
    try:
        if fuelType == "RP1":
            rp1_state.update(CP.PT_INPUTS, P_coolant, T_coolant)
            return (rp1_state.rhomass(), rp1_state.hmass(),
                    rp1_state.viscosity(), rp1_state.conductivity(), rp1_state.cpmass())
        elif fuelType == "LH2":
            lh2_state.update(CP.PT_INPUTS, P_coolant, T_coolant)
            return (lh2_state.rhomass(), lh2_state.hmass(),
                    lh2_state.viscosity(), lh2_state.conductivity(), lh2_state.cpmass())
        elif fuelType == "CH4":
            ch4_state.update(CP.PT_INPUTS, P_coolant, T_coolant)
            return (ch4_state.rhomass(), ch4_state.hmass(),
                    ch4_state.viscosity(), ch4_state.conductivity(), ch4_state.cpmass())
    except Exception as ex:
        raise RuntimeError(f"CoolProp failure ({fuelType}  T={T_coolant:.1f} K  P={P_coolant:.2e} Pa): {ex}")


def _get_T_from_enthalpy(h_next, P_coolant, fuelType):
    """Recover temperature from specific enthalpy [J/kg] and pressure [Pa]."""
    try:
        if fuelType == "RP1":
            rp1_state.update(CP.HmassP_INPUTS, h_next, P_coolant)
            return rp1_state.T()
        elif fuelType == "LH2":
            lh2_state.update(CP.HmassP_INPUTS, h_next, P_coolant)
            return lh2_state.T()
        elif fuelType == "CH4":
            ch4_state.update(CP.HmassP_INPUTS, h_next, P_coolant)
            return ch4_state.T()
    except Exception as ex:
        raise RuntimeError(f"Enthalpy lookup failure ({fuelType}): {ex}")


def calc_q_coolant(dx, x, y, s, T_cw, T_coolant, P_coolant, engine_info):
    """
    Calculate coolant-side heat transfer at station x.

    Returns
    -------
    q_coolant, T_coolant_new, P_coolant_new, dF_dx,
    h_coolant, C3, v, rho, chan_area, Re, Nu, Dh, conductivity
    """
    chan_w, chan_h, chan_t, chan_land = engine_info.get_geo_interp(x, dx)
    # Counter-current: next station is x-dx (coolant flows from nozzle exit toward injector)
    chan_w_next, chan_h_next, chan_t_next, chan_land_next = engine_info.get_geo_interp(x - dx, dx)

    chan_area      = chan_w      * chan_h
    chan_area_next = chan_w_next * chan_h_next
    P_wet  = 2 * (chan_w      + chan_h)
    P_next = 2 * (chan_w_next + chan_h_next)
    Dh      = 4 * chan_area      / P_wet
    Dh_next = 4 * chan_area_next / P_next

    rho, h, viscosity, conductivity, Cp = _get_coolant_props(T_coolant, P_coolant, engine_info.fuelType)

    v   = (engine_info.mdot_coolant / engine_info.Ncc) / (rho * chan_area)
    Re  = rho * v * Dh / viscosity
    Pr  = viscosity * Cp / conductivity
    f   = calc_frictionFactor(Re, Dh, 0.001, engine_info.e)

    # ---- Nusselt number ---
    if engine_info.fuelType in ("RP1", "CH4"):
        # Gnielinski (valid Re > 3000)
        Nu = ((f/8) * (Re - 1000) * Pr) / (1 + 12.7 * (f/8)**0.5 * (Pr**(2/3) - 1))
        C3 = 1.0
    else:
        # Niino (1982) for LH2
        xi = f / calc_frictionFactor(Re, Dh, 0.001, 0)
        eps_star = Re * (engine_info.e / Dh) * (f/8)**0.5
        B = 4.7 * eps_star**0.2 if eps_star >= 7 else 4.5 + 0.57 * eps_star**0.75
        C1 = ((1 + 1.5*Pr**(-1/6)*Re**(-1/8)*(Pr-1)) /
              (1 + 1.5*Pr**(-1/6)*Re**(-1/8)*(Pr*xi-1))) * xi
        # C2: entrance length correction (xi^0.1, NOT T_cw/T ratio)
        C2 = 1 + xi**0.1 * (s / Dh)**(-0.7)
        radius_curv, r_type = engine_info.get_r_value(x)
        if radius_curv != 0:
            concavity = 0.05 if r_type == 'Ru' else -0.05
            sign      = 1    if r_type == 'Ru' else -1
            C3 = (Re * ((0.25*Dh) / (sign*radius_curv))**2)**concavity
        else:
            C3 = 1.0
        Nu = ((f/8)*Re*Pr*(T_coolant/T_cw)**0.55 / (1 + (f/8)**0.5*(B - 8.48))) * C1 * C2 * C3

    h_coolant = Nu * conductivity / Dh

    # Fin efficiency
    perimiter = 2 * (chan_land + dx)
    fin_area  = chan_land * dx
    m_fin     = np.sqrt(h_coolant * perimiter) / (engine_info.k * fin_area)
    M_fin     = np.sqrt(h_coolant * perimiter * engine_info.k * fin_area) * (T_cw - T_coolant)
    L_fin     = chan_h + chan_land / 2

    q_fin  = M_fin * np.tanh(m_fin * L_fin)
    q_base = h_coolant * dx * chan_w * (T_cw - T_coolant)
    R_fin  = (T_cw - T_coolant) / q_fin  if q_fin  != 0 else 1e10
    R_base = (T_cw - T_coolant) / q_base if q_base != 0 else 1e10
    R_cold = 1.0 / (1.0/R_fin + 1.0/R_base)
    q_cool = (T_cw - T_coolant) / R_cold

    # Pressure losses (major + minor; no double-counting)
    major = f * rho * v**2 * dx / (2 * Dh)
    if chan_area_next > chan_area:
        K = ((Dh / Dh_next)**2 - 1)**2
    elif chan_area_next < chan_area:
        K = 0.5 - 0.167*(Dh_next/Dh) - 0.125*(Dh_next/Dh)**2 - 0.208*(Dh_next/Dh)**3
    else:
        K = 0.0
    minor = K * rho * v**2 / 2
    dP    = major + minor
    P_coolant_new = P_coolant - dP

    # Temperature update via enthalpy
    h_next      = h + q_cool / (engine_info.mdot_coolant / engine_info.Ncc)
    T_coolant_new = _get_T_from_enthalpy(h_next, P_coolant_new, engine_info.fuelType)

    dF_dx = f * dx * v**2 / (2 * Dh)

    return (q_cool, T_coolant_new, P_coolant_new, dF_dx,
            h_coolant, C3, v, rho, chan_area, Re, Nu, Dh, conductivity)


def calc_q_wall(dx, x, y, T_hw, T_cw, engine_info):
    chan_w, chan_h, chan_t, chan_land = engine_info.get_geo(x)
    R_wall = chan_t / (engine_info.k * dx * (chan_w + chan_land))
    return (T_hw - T_cw) / R_wall


# -----------------------------------------------------------------------
# Engine info container all inputs are specified 
# -----------------------------------------------------------------------
class EngineInfo:
    def __init__(self, gam, C_star, M_c, P_c, T_c, Cp, F_Vac, Ncc,
                 combustion_molecules, A_c, A_t, A_e, L_c,
                 x_j, chan_land, chan_w, chan_h, chan_t,
                 gas, mdot_coolant, e, k, mdot_chamber,
                 RD, RU, R1, theta1, thetaD, thetaE,
                 fuelType, R_specific, geo_params):

        self.gam   = gam
        self.C_star= C_star
        self.M_c   = M_c
        self.P_c   = P_c
        self.T_c   = T_c
        self.Cp    = Cp
        self.F_Vac = F_Vac
        self.Ncc   = Ncc
        self.combustion_molecules = combustion_molecules
        self.A_c   = A_c
        self.A_t   = A_t
        self.A_e   = A_e
        self.L_c   = L_c
        self.x_j       = x_j
        self.chan_land  = chan_land
        self.chan_w     = chan_w
        self.chan_h     = chan_h
        self.chan_t     = chan_t
        self.gas        = gas
        self.mdot_coolant  = mdot_coolant
        self.mdot_chamber  = mdot_chamber
        self.e   = e
        self.k   = k
        self.RD  = RD
        self.RU  = RU
        self.R1  = R1
        self.theta1  = theta1
        self.thetaD  = thetaD
        self.thetaE  = thetaE
        self.fuelType    = fuelType
        self.R_specific  = R_specific
        self.geo_params  = geo_params  # dict passed to calc_radius

    def get_geo(self, x):
        idx = int(np.argmin(np.abs(np.array(self.x_j) - x)))
        return self.chan_w[idx], self.chan_h[idx], self.chan_t[idx], self.chan_land[idx]

    def get_geo_interp(self, x, dx):
        land_f = interp1d(self.x_j, self.chan_land, kind='linear', bounds_error=False, fill_value="extrapolate")
        w_f    = interp1d(self.x_j, self.chan_w,    kind='linear', bounds_error=False, fill_value="extrapolate")
        h_f    = interp1d(self.x_j, self.chan_h,    kind='linear', bounds_error=False, fill_value="extrapolate")
        t_f    = interp1d(self.x_j, self.chan_t,    kind='linear', bounds_error=False, fill_value="extrapolate")
        return float(w_f(x)), float(h_f(x)), float(t_f(x)), float(land_f(x))

    def get_radius(self, x):
        return calc_radius(x, self.A_c, self.A_t, self.A_e, self.L_c, self.geo_params)

    def calc_area(self, x):
        return np.pi * self.get_radius(x)**2

    def calc_transport_properties(self, T, P):
        mole_fractions = {k: v[0] for k, v in self.combustion_molecules.items()}
        self.gas.TPX = T, P, mole_fractions
        viscosity    = self.gas.viscosity
        conductivity = self.gas.thermal_conductivity
        Cp           = self.gas.cp_mass
        density      = self.gas.density
        Pr           = Cp * viscosity / conductivity
        return viscosity, conductivity, Cp, Pr, density

    def get_r_value(self, x):
        """
        Radius of curvature for the Niino C3 correction.
        Uses engine-specific parameters (no RS25 hardcodes).
        """
        RU     = self.RU
        RD     = self.RD
        theta1 = self.theta1
        L_c    = self.L_c
        L_e    = self.geo_params['L_e'] if self.geo_params else 5.339*0.0254

        # Start of converging section
        L_conv_start = L_e

        if L_conv_start < x <= L_c:
            return RU, 'Ru'
        elif x > L_c:
            return RD, 'Rd'
        else:
            return 0, 'none'

    def displayChannelGeometry(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(self.x_j), np.array(self.chan_w)*1000,    label='Width',      color='orange')
        plt.plot(self.x_j,           np.array(self.chan_h)*1000,    label='Height',     color='green')
        plt.plot(self.x_j,           np.array(self.chan_t)*1000,    label='Thickness',  color='red')
        plt.plot(self.x_j,           np.array(self.chan_land)*1000, label='Land',       color='blue')
        plt.xlabel('Distance from Injector [m]')
        plt.ylabel('Dimension [mm]')
        plt.title('Cooling Channel Geometry')
        plt.legend(); plt.grid()


# -----------------------------------------------------------------------
# Newton-Raphson solver for T_hw, T_cw
# -----------------------------------------------------------------------
def newton_solve_temperatures(dx, x, y, s, T_coolant, P_coolant, engine_info,
                               T_hw_init, T_cw_init, tol=0.1, max_iter=50):
    def F(T_vec):
        T_hw, T_cw = T_vec
        q_gas, *_ = calc_q_gas(dx, x, y, T_hw, engine_info)
        result = calc_q_coolant(dx, x, y, s, T_cw, T_coolant, P_coolant, engine_info)
        q_cool = result[0]
        q_wall = calc_q_wall(dx, x, y, T_hw, T_cw, engine_info)
        return np.array([q_wall - q_gas, q_gas - q_cool])

    def jacobian(T_vec, h=1e-3):
        J = np.zeros((2, 2))
        for i in range(2):
            T1, T2 = np.array(T_vec), np.array(T_vec)
            T1[i] -= h; T2[i] += h
            J[:, i] = (F(T2) - F(T1)) / (2*h)
        return J

    T = np.array([T_hw_init, T_cw_init])
    for _ in range(max_iter):
        F_val = F(T)
        if np.linalg.norm(F_val) < tol:
            return T
        J = jacobian(T)
        try:
            delta = np.linalg.solve(J, -F_val)
        except np.linalg.LinAlgError:
            raise RuntimeError("Singular Jacobian — check initial guesses.")
        # Step damping
        if np.max(np.abs(delta)) > 50.0:
            delta *= 50.0 / np.max(np.abs(delta))
        T += delta
    raise RuntimeError("Newton-Raphson did not converge.")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():

    # ===================================================================
    # ENGINE CONFIG — fill in from CEA / design sheet
    # ===================================================================

    # --- Propellant combination ---
    fuelType     = "RP1"    # "RP1" | "LH2" | "CH4"
    oxidizerType = "LOX"    # informational label only

    # --- Combustion properties (from CEA at design O/F) ---
    gam    = 1.181          # Specific heat ratio at chamber [-]
    C_star = 1720.0         # Characteristic velocity [m/s]
    P_c    = 60.0e5         # Chamber pressure [Pa]  (60 bar)
    T_c    = 3450.0         # Chamber adiabatic flame temperature [K]
    Cp     = 2100.0         # Cp of combustion products at chamber [J/kg·K]
    F_Vac  = 5500.0         # Vacuum thrust target [N]

    # Mole fractions and molecular masses of combustion species (from CEA)
    # Format: 'Species': [mole_fraction, molecular_mass_kg_per_mol]
    combustion_molecules = {
        'H2O': [0.5700, 18.01528e-3],
        'CO2': [0.1600, 44.00950e-3],
        'CO':  [0.1900, 28.01000e-3],
        'H2':  [0.0500, 2.01588e-3 ],
        'OH':  [0.0200, 17.00734e-3],
        'O':   [0.0100, 15.99940e-3],
    }

    # Per-engine specific gas constant (computed from combustion molecules)
    mean_molar_mass = sum(v[0]*v[1] for v in combustion_molecules.values())  # kg/mol
    R_specific = Ru / mean_molar_mass   # J/(kg·K)
    print(f'Mean molar mass: {mean_molar_mass*1000:.2f} g/mol  R_specific: {R_specific:.2f} J/(kg·K)')

    # --- Solver initialisation ---
    # Injector Mach number (subsonic; tune until Mach profile is smooth through throat)
    M_c = 0.05      # Start low and adjust if solver oscillates

    # --- Geometry design parameters ---
    expRatio    = 5.0   # Nozzle area expansion ratio (Ae/At)
    L_star_m    = 1.143 # Characteristic chamber length [m]  (45 inch, typical RP1)
    contChamber = 6.0   # Contraction ratio (A_c / A_t)

    # Coolant channel material
    e           = 6.3e-6  # Surface roughness [m]  (SLM 3D-print)
    k           = 167.0   # Thermal conductivity [W/m·K]  (6061-T6 Al)
    meltingpoint= 855.0   # Melting point [K]              (6061-T6 Al)

    # Number of coolant channels
    Ncc = 36

    # --- Coolant inlet conditions (at nozzle exit, counter-current flow) ---
    T_coolant_init = 290.0    # RP1 inlet temperature [K]
    P_coolant_init = 80.0e5  # RP1 inlet pressure [Pa]  (80 bar)

    # Mass flow rate of coolant (RP1 side at O/F ~ 2.4)
    mdot_coolant = 0.57       # [kg/s]  — update from mdot_total / (1 + O/F)

    # --- Nozzle / contraction geometry (used in calc_radius and get_r_value) ---
    # All in SI (metres / degrees).  Adjust for your engine design.
    theta1  = 30.0            # Convergence half-angle [deg]
    thetaD  = 30.0            # Initial divergence half-angle [deg]  (bell nozzle)
    thetaE  = 12.0            # Exit half-angle [deg]
    # Throat radius of curvature radii — start with multiples of throat radius
    # (will be overwritten once A_t is known; approximate here for geo_params)
    # These are geometry parameters for the contour calculation
    R1_mult = 1.5             # R1 = R1_mult * R_throat  (inner contraction fillet)
    RU_mult = 1.5             # RU = RU_mult * R_throat  (outer contraction curve)
    RD_mult = 0.382           # RD = RD_mult * R_throat  (throat divergence fillet)

    # --- Solver initial guesses for wall temperatures ---
    T_hw_init = 700.0   # Hot wall temperature [K]
    T_cw_init = 400.0   # Cold wall temperature [K]

    # ===================================================================
    # END ENGINE CONFIG
    # ===================================================================

    showPlots = True
    gas = ct.Solution('gri30.yaml')

    # Compute engine geometry
    A_c, A_t, A_e, L_c, mdot_chamber = engine_geometry(
        gam, P_c, T_c, F_Vac, expRatio, L_star_m, contChamber, R_specific, showPlots=False)

    # Resolve geometry radii from throat area
    R_t  = np.sqrt(A_t / np.pi)
    R1   = R1_mult * R_t
    RU   = RU_mult * R_t
    RD   = RD_mult * R_t

    # Straight section length: total chamber minus the converging arc
    L_e  = L_c - RU * np.sin(np.deg2rad(theta1))

    geo_params = {
        'L_e'    : L_e,
        'theta1' : theta1,
        'thetaD' : thetaD,
        'alpha'  : thetaE,
        'R1_mult': R1_mult,
        'RU'     : RU,
        'RD'     : RD,
    }

    print(f'R_t={R_t*1000:.2f} mm  RU={RU*1000:.2f} mm  RD={RD*1000:.2f} mm  L_e={L_e*1000:.1f} mm  L_c={L_c*1000:.1f} mm')

    # -----------------------------------------------------------------------
    # Cooling channel geometry — define x stations and dimensions
    # Replace these arrays with your actual design values.
    # x_j : axial position from injector face [m]
    # chan_w, chan_h, chan_t, chan_land : channel dimensions [m]
    # -----------------------------------------------------------------------
    total_len  = L_c + np.sqrt(A_e/np.pi) / np.tan(np.deg2rad(15)) * 0.8  # approx nozzle length
    x_j = list(np.linspace(0, total_len, 15))  # TODO: replace with your actual x stations

    # Placeholder channel dimensions (constant cross-section for initial run)
    # TODO: replace with optimised channel geometry from your design
    chan_w    = [1.5e-3] * len(x_j)   # Channel width [m]
    chan_h    = [3.0e-3] * len(x_j)   # Channel height [m]
    chan_t    = [0.8e-3] * len(x_j)   # Wall thickness [m]
    chan_land = [1.0e-3] * len(x_j)   # Land width [m]

    # Build EngineInfo
    engine_info = EngineInfo(
        gam, C_star, M_c, P_c, T_c, Cp, F_Vac, Ncc,
        combustion_molecules,
        A_c, A_t, A_e, L_c,
        x_j, chan_land, chan_w, chan_h, chan_t,
        gas, mdot_coolant, e, k, mdot_chamber,
        RD, RU, R1, theta1, thetaD, thetaE,
        fuelType, R_specific, geo_params
    )

    engine_info.displayChannelGeometry()

    # -----------------------------------------------------------------------
    # 1-D flow solver
    # -----------------------------------------------------------------------
    xi = 0.0
    xf = x_j[-1]
    dx = (xf - xi) / 500
    array_length = int((xf - xi) / dx)

    dF_dx_arr = np.zeros(array_length)
    dQ_dx_arr = np.zeros(array_length)
    keydata = [engine_info.A_c, engine_info.A_t, engine_info.A_e, engine_info.L_c,
               engine_info.gam, engine_info.Cp, dF_dx_arr, dQ_dx_arr,
               engine_info.R_specific, engine_info.geo_params]

    # Initial isentropic solve
    dx, xp_m, yp_m = calc_flow_data(xi, xf, dx, engine_info.M_c, P_c, T_c, keydata)

    # -----------------------------------------------------------------------
    # Thermal analysis — outer iteration loop
    # -----------------------------------------------------------------------
    T_coolant_arr   = np.zeros(array_length)
    T_hw_arr        = np.zeros(array_length)
    T_cw_arr        = np.zeros(array_length)
    P_coolant_arr   = np.zeros(array_length)
    heatflux_arr    = np.zeros(array_length)
    h_gas_arr       = np.zeros(array_length)
    h_cool_arr      = np.zeros(array_length)
    Re_arr          = np.zeros(array_length)
    Nu_arr          = np.zeros(array_length)
    Dh_arr          = np.zeros(array_length)
    v_arr           = np.zeros(array_length)
    rho_arr         = np.zeros(array_length)
    q_gas_arr       = np.zeros(array_length)
    q_cool_arr      = np.zeros(array_length)
    melting_arr     = np.full(array_length, meltingpoint)

    T_hw_arr_prev   = np.zeros(array_length)
    T_cw_arr_prev   = np.zeros(array_length)

    T_cw_error = T_hw_error = 100.0
    relax = 0.8
    iteration = 0

    while T_cw_error > 1.0 or T_hw_error > 1.0:
        if iteration == 0:
            T_hw      = T_hw_init
            T_cw      = T_cw_init
            T_coolant = T_coolant_init
            P_coolant = P_coolant_init
        else:
            T_hw      = T_hw_arr[0]
            T_cw      = T_cw_arr[0]
            T_coolant = T_coolant_arr[0]
            P_coolant = P_coolant_arr[0]

        T_cw_arr_prev = T_cw_arr.copy()
        T_hw_arr_prev = T_hw_arr.copy()
        s = dx  # distance along coolant path (entrance length)

        for i, (x, y) in enumerate(zip(reversed(xp_m), reversed(yp_m))):
            T_hw, T_cw = newton_solve_temperatures(
                dx, x, y, s, T_coolant, P_coolant, engine_info,
                T_hw, T_cw, tol=0.1, max_iter=50)

            q_gas, heatflux, h_gas, A_gas, R_hot, asurf, dF = calc_q_gas(
                dx, x, y, T_hw, engine_info)
            (q_cool, T_coolant, P_coolant, dF_val,
             h_cool, C3, v, rho, chan_area, Re, Nu, Dh, cond) = calc_q_coolant(
                dx, x, y, s, T_cw, T_coolant, P_coolant, engine_info)

            # Update dQ/dF with relaxation
            dQ_new = (q_gas * engine_info.Ncc) / (mdot_chamber * dx)
            dQ_dx_arr[i] = relax * dQ_new  + (1 - relax) * dQ_dx_arr[i]
            dF_dx_arr[i] = relax * dF      + (1 - relax) * dF_dx_arr[i]
            s += dx

            T_coolant_arr[i]= T_coolant
            T_hw_arr[i]     = T_hw
            T_cw_arr[i]     = T_cw
            P_coolant_arr[i]= P_coolant
            heatflux_arr[i] = heatflux
            h_gas_arr[i]    = h_gas
            h_cool_arr[i]   = h_cool
            Re_arr[i]       = Re
            Nu_arr[i]       = Nu
            Dh_arr[i]       = Dh
            v_arr[i]        = v
            rho_arr[i]      = rho
            q_gas_arr[i]    = q_gas
            q_cool_arr[i]   = q_cool

        T_cw_error = np.mean(np.abs(T_cw_arr - T_cw_arr_prev))
        T_hw_error = np.mean(np.abs(T_hw_arr - T_hw_arr_prev))
        print(f'Iter {iteration+1}: ΔT_cw={T_cw_error:.3f} K  ΔT_hw={T_hw_error:.3f} K')

        keydata = [engine_info.A_c, engine_info.A_t, engine_info.A_e, engine_info.L_c,
                   engine_info.gam, engine_info.Cp, dF_dx_arr, dQ_dx_arr,
                   engine_info.R_specific, engine_info.geo_params]
        dx, xp_m, yp_m = calc_flow_data(xi, xf, dx, engine_info.M_c, P_c, T_c, keydata)
        iteration += 1

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    print('\n' + '='*55)
    print(f'Peak hot wall temperature : {max(T_hw_arr):.1f} K')
    print(f'Peak cold wall temperature: {max(T_cw_arr):.1f} K')
    print(f'Peak heat flux            : {max(heatflux_arr):.4E} W/m²')
    print(f'Peak gas-side HTC         : {max(h_gas_arr):.1f} W/m²K')
    dP_coolant = P_coolant_arr[0] - P_coolant_arr[-1]
    dT_coolant = T_coolant_arr[-1] - T_coolant_arr[0]
    print(f'Coolant ΔP                : {dP_coolant:.4E} Pa  ({dP_coolant/1e5:.2f} bar)')
    print(f'Coolant ΔT                : {dT_coolant:.2f} K')
    print('='*55)

    # Plots
    xp_throat = [x - engine_info.L_c for x in xp_m]
    T_cool_rev = list(reversed(T_coolant_arr))
    dT_rev     = [T - T_cool_rev[-1] for T in T_cool_rev]

    create_plot(xp_throat, dT_rev,
                "Distance from Throat [m] (– = chamber)", "ΔT coolant [K]",
                "Coolant Temperature Rise")
    create_plot(list(xp_m), list(reversed(heatflux_arr)),
                "Distance from Injector [m]", "Heat flux [W/m²]", "Heat flux")

    plt.figure()
    plt.plot(xp_m, list(reversed(T_cw_arr)),    label='T_cw [K]')
    plt.plot(xp_m, list(reversed(T_hw_arr)),    label='T_hw [K]')
    plt.plot(xp_m, list(reversed(melting_arr)), label='Melt point [K]', linestyle='--', color='red')
    plt.xlabel("Distance from Injector [m]"); plt.ylabel("Temperature [K]")
    plt.title("Wall Temperatures"); plt.legend(); plt.grid(True)

    create_plot(list(xp_m), [np.sqrt(y[0]) for y in yp_m],
                "Distance from Injector [m]", "Mach", "Mach Number")
    create_plot(list(xp_m), [y[1] for y in yp_m],
                "Distance from Injector [m]", "Pressure [Pa]", "Chamber Pressure")
    create_plot(list(xp_m), list(reversed(Re_arr)),
                "Distance from Injector [m]", "Reynolds Number", "Coolant Reynolds Number")
    create_plot(list(xp_m), list(reversed(h_gas_arr)),
                "Distance from Injector [m]", "h_gas [W/m²K]", "Gas-side HTC")

    plt.show()


if __name__ == "__main__":
    main()
