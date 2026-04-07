"""
geometry.py
Engine sizing and parametric nozzle contour.

Main entry point: size_engine(config, cea_result) → EngineGeometry
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

import scipy.optimize as sp

from config import EngineConfig
from cea_interface import CEAResult


# -----------------------------------------------------------------------
# Output dataclass
# -----------------------------------------------------------------------
@dataclass
class EngineGeometry:
    """All geometric quantities derived from CEA + design choices."""

    # Key areas [m²]
    A_t: float   # Throat
    A_c: float   # Chamber (injector face)
    A_e: float   # Exit

    # Axial lengths [m]
    L_c: float   # Chamber length (injector to throat)
    L_e: float   # Straight cylindrical section length

    # Radii [m]
    R_t: float   # Throat radius
    R_c: float   # Chamber radius
    R_e: float   # Exit radius

    # Throat curvature radii [m]
    R1: float    # Inner convergence fillet
    RU: float    # Outer convergence curve
    RD: float    # Divergence fillet

    # Nozzle contour angles [deg]
    theta1: float
    thetaD: float
    thetaE: float

    # Expansion ratio
    exp_ratio: float  # Ae/At

    # Mass flow rates [kg/s]
    mdot:      float
    mdot_fuel: float
    mdot_ox:   float

    # Performance at design point
    M_exit:  float   # Isentropic exit Mach
    P_exit:  float   # Exit static pressure [Pa]

    # Nozzle length [m] (throat to exit plane, 80% Rao bell)
    L_nozzle: float


# -----------------------------------------------------------------------
# Isentropic area-Mach relation  (supersonic root)
# -----------------------------------------------------------------------
def _area_mach(M, gam, AR):
    """Returns 0 when M satisfies A/A* = AR for given gamma."""
    return (1.0/M) * ((1 + (gam-1)/2 * M**2) / (1 + (gam-1)/2))**((gam+1)/(2*(gam-1))) - AR


def exit_mach(AR: float, gam: float) -> float:
    """Solve for supersonic exit Mach number from area ratio and gamma."""
    return sp.brentq(_area_mach, 1.001, 50.0, args=(gam, AR))


def exit_pressure(P_c: float, gam: float, M_e: float) -> float:
    return P_c * (1 + (gam-1)/2 * M_e**2) ** (-gam/(gam-1))


# -----------------------------------------------------------------------
# Engine sizing
# -----------------------------------------------------------------------
def size_engine(config: EngineConfig, cea: CEAResult) -> EngineGeometry:
    """
    Compute throat/chamber/exit areas, chamber length, and nozzle radii
    from CEA thermodynamic output and user design choices.

    Sizing route:
      mdot  = F_vac / Isp_vac_e          (thrust equation)
      A_t   = (mdot * C*) / Pc           (characteristic velocity definition)
      A_e   = A_t * er                   (expansion ratio)
      A_c   = A_t * cr                   (contraction ratio)
      L_c   = L_cylinder + L_cone        (L* volume balance)
    """
    gam = cea.gamma_c
    AR  = config.exp_ratio

    # --- Mass flow rates ---
    mdot      = config.F_vac / cea.Isp_vac_e
    mdot_fuel = mdot / (1.0 + config.OF)
    mdot_ox   = mdot_fuel * config.OF

    # --- Throat, exit, chamber areas ---
    A_t = (mdot * cea.C_star) / cea.P_c
    A_e = A_t * AR
    A_c = A_t * config.cont_ratio

    # --- Key radii ---
    R_t = np.sqrt(A_t / np.pi)
    R_c = np.sqrt(A_c / np.pi)
    R_e = np.sqrt(A_e / np.pi)

    # --- Curvature radii from throat radius ---
    R1 = config.R1_mult * R_t
    RU = config.RU_mult * R_t
    RD = config.RD_mult * R_t

    # --- Chamber length via L* volume balance ---
    # Convergent cone length (includes RU arc at throat)
    th1 = np.deg2rad(config.theta1)
    L_cone = ((R_t * (np.sqrt(config.cont_ratio) - 1)
               + RU * (1.0/np.cos(th1) - 1.0))
              / np.tan(th1))

    # Frustum volume of convergent cone
    V_cone = (np.pi / 3.0) * L_cone * (R_c**2 + R_t**2 + R_c*R_t)

    # Required chamber volume from L*
    V_c = A_t * config.L_star

    V_cylinder = V_c - V_cone
    if V_cylinder < 0:
        raise ValueError(
            f"L* = {config.L_star:.3f} m is too small — convergent cone volume "
            f"({V_cone*1e6:.1f} cm³) exceeds required chamber volume "
            f"({V_c*1e6:.1f} cm³). Increase L_star.")

    L_cylinder = V_cylinder / A_c
    L_c        = L_cylinder + L_cone   # injector face → throat

    # Straight section (used as convergence start in nozzle_radius)
    L_e = L_cylinder

    # --- Nozzle length (80% Rao bell) ---
    L_nozzle = 0.8 * (np.sqrt(AR) - 1.0) * R_t / np.tan(np.deg2rad(15.0))

    # --- Exit conditions (for flow solver reference) ---
    M_e = exit_mach(AR, gam)
    P_e = exit_pressure(cea.P_c, gam, M_e)

    geom = EngineGeometry(
        A_t=A_t, A_c=A_c, A_e=A_e,
        L_c=L_c, L_e=L_e,
        R_t=R_t, R_c=R_c, R_e=R_e,
        R1=R1, RU=RU, RD=RD,
        theta1=config.theta1, thetaD=config.thetaD, thetaE=config.thetaE,
        exp_ratio=AR,
        mdot=mdot, mdot_fuel=mdot_fuel, mdot_ox=mdot_ox,
        M_exit=M_e, P_exit=P_e,
        L_nozzle=L_nozzle,
    )

    _print_geometry(geom)
    return geom


def _print_geometry(g: EngineGeometry):
    print("\n--- Engine Geometry ---")
    print(f"  Throat    : D_t = {2*g.R_t*1000:.2f} mm   A_t = {g.A_t*1e4:.4f} cm²")
    print(f"  Chamber   : D_c = {2*g.R_c*1000:.2f} mm   A_c = {g.A_c*1e4:.4f} cm²")
    print(f"  Exit      : D_e = {2*g.R_e*1000:.2f} mm   A_e = {g.A_e*1e4:.4f} cm²")
    print(f"  Chamber L : L_c = {g.L_c*1000:.1f} mm  (cyl = {g.L_e*1000:.1f} mm)")
    print(f"  Nozzle L  : L_n = {g.L_nozzle*1000:.1f} mm   L_tot = {(g.L_c+g.L_nozzle)*1000:.2f} mm")
    print(f"  RU = {g.RU*1000:.2f} mm   RD = {g.RD*1000:.2f} mm   R1 = {g.R1*1000:.2f} mm")
    print(f"  mdot      : {g.mdot:.4f} kg/s  (fuel {g.mdot_fuel:.4f}  ox {g.mdot_ox:.4f})")
    print(f"  M_exit    : {g.M_exit:.4f}   P_exit = {g.P_exit/1000:.2f} kPa")


# -----------------------------------------------------------------------
# Parametric nozzle contour  r(x)
# -----------------------------------------------------------------------

def nozzle_radius(x: float, geom: EngineGeometry, dx: float) -> float:
    """
    Wall radius [m] at axial position x [m] (x = 0 at injector face).
    Uses the parabolic-Bezier (Rao) bell nozzle contour.
    """
    R_c   = geom.R_c
    R_t   = geom.R_t
    R_e   = geom.R_e
    L_c   = geom.L_c 
    L_e   = geom.L_nozzle
    R1    = geom.R1
    RU    = geom.RU
    RD    = geom.RD
    AR    = geom.exp_ratio
    theta1 = geom.theta1
    thetaD = geom.thetaD
    alpha  = geom.thetaE

    th1_r  = np.deg2rad(theta1)
    thD_r  = np.deg2rad(thetaD)
    alp_r  = np.deg2rad(alpha)
    
    # Shift x to center it at throat isntead of injector
    x = x - L_c 

    # Throat fillets
    r1 = RU * R_t # Converting to radius [m]
    ang1_start = -(90.0 + theta1) * np.pi/180.0     # Convergent radius angle
    xP4 = r1 * np.cos(ang1_start)                   # Convergent radius X coordinate
    yP4 = r1 * np.sin(ang1_start) + (r1 + R_t)      # Convergent radius Y coordinate

    r2 = RD * R_t # Converting to radius [m]
    ang2_end = (thetaD - 90) * np.pi/180            # Divergent radius angle
    Nx = r2 * np.cos(ang2_end)                      # Divergent radius X coordinate
    Ny = r2 * np.sin(ang2_end) + (r2 + R_t)         # Divergent radius Y coordinate

    Ex, Ey = L_e, R_e  

    # Converging big arc tangent to straight chamber and -theta1 line
    rc = R1 * R_t
    mc = -np.tan(th1_r)
    bc = yP4 - mc * xP4
    yc = R_c - rc
    s = np.sqrt(mc*mc + 1.0)
    xc_plus = (yc - bc + rc*s)/mc
    xc_minus = (yc - bc - rc*s)/mc
    xc = xc_minus if xc_minus < xc_plus else xc_plus # Upstream center x (more negative)

    # Tangency projection point P3 on -theta_c line
    S = mc*xc - yc + bc
    xP3 = xc - mc * S / (mc*mc + 1.0)
    yP3 = yc + S / (mc*mc + 1.0)

    # Straight chamber length remaining
    L_conv = -xc 
    L_straight = L_c - L_conv

    # Domain checks (use same geometric as plot)
    x_min = -L_c if L_straight > 1e-12 else xc
    x_max = L_e
    if x < x_min - dx or x > x_max + dx:
        print(f"{x:.4f}, {x_min:.4f}, {x_max:.4f}")
        raise ValueError("x is outside the chamber/nozzle axial domain")
    
    # Peicewise evaluation using smae equations

    # 1) Straight chamber: y = Rc, for x in [-Lc, xc]
    if L_straight > 1e-12 and x <= xc + 1e-12:
        return float(R_c)
    
    # 2) Convererging arc (circle centered at (xc, yc), radius rc, x in [xc, xP3]
    if x >= xc - 1e-12 and x <= xP3 + 1e-12:
        dx = x - xc
        if abs(dx) > rc + 1e-12:
            raise ValueError("x outside converging arc radius bounds")
        return float(yc + np.sqrt(max(0.0, rc*rc - dx*dx)))
    
    # 3) Short straight connector from P3 to P4 along -theta1
    if x >= xP3 - 1e-12 and x <= xP4 + 1e-12:
        return float(mc * (x - xP4) + yP4)
    
    # 4) Convergent throat fillet (circle centered at (0, r1 + R_t), x in [xP4, 0]
    if x >= xP4 - 1e-12 and x <= 0.0 + 1e-12:
        # Using circle: (x)^2 + (y - (r1 + R_t))^2 = r1^2, choose lower branch near throat
        return float((r1 + R_t)) - np.sqrt(max(0.0, r1*r1 - x*x))
    
    # 5) Divergent throat fillet (circle centered at (0, r2 + R_t), radius r2, x in [0, Nx]
    if x>=0.0 - 1e-12 and x <= Nx + 1e-12:
        return float((r2+R_t) - np.sqrt(max(0.0, r2*r2 - x*x)))
    
    # 6) Bell (quadratic Bezier) from (Nx, Ny) to (Ex, Ey) with slopes thD and alpha
    #   x(t) = a t^2 + b t + c, with 0 <= t <= 1 and x in [Nx. Ex]. Solve for t
    a = Nx - 2.0* ((Ny - np.tan(thD_r)*Nx) - (Ey - np.tan(alp_r) * Ex)) + Ex # Not used directly ; compute Q first

    # Compute Bezier control point Q from slop constraints
    m1, m2 = np.tan(thD_r), np.tan(alp_r)
    C1 = Ny - m1*Nx
    C2 = Ey - m2*Ex
    Qx = (C2-C1) / (m1 - m2)
    Qy = (m1*C2 - m2*C1) / (m1 - m2)

    # Quadratic coefficients for x(t)
    ax = Nx - 2.0*Qx + Ex
    bx = 2.0*(Qx - Nx)
    cx = Nx

    # Solve ax t^2 + bx t + (cx - x) = 0
    A = ax
    B = bx
    C = cx - x 
    t_candidates = []
    if abs(A) < 1e-14:
        # Degenerate to linear
        if abs(B) < 1e-14:
            t_candidates = [0.0]
        else: 
            t_candidates = [-C / B]
    else:
        disc = B*B - 4.0*A*C
        if disc < -1e-12:
            raise ValueError("No real solution for Bezier parameeter t at given x")
        disc = max(0.0 , disc)
        sqrt_disc = np.sqrt(disc)
        t1 = (-B + sqrt_disc) / (2.0 * A)
        t2 = (-B - sqrt_disc) / (2.0 * A)
        t_candidates = [t1, t2]

    # Pick a valid t in [0,1]
    t = None
    for ti in t_candidates:
        if ti >= -dx and ti <= 1.0 + dx:
            t = min(max(ti, 0.0), 1.0)
            break
    if t is None:
        print(t_candidates)
        raise ValueError("No valid Bezier parameter t in [0,1] for given x")
    
    # Compute y(t) for Bezier
    y = (1 - t)**2 * Ny + 2*(1 - t)*t * Qy + t**2 * Ey
    return float(y)


# def nozzle_radius(x: float, geom: EngineGeometry) -> float:
#     """
#     Wall radius [m] at axial position x [m] (x = 0 at injector face).
#     Uses the parabolic-Bezier (Rao) bell nozzle contour.
#     """
#     R_c   = geom.R_c
#     R_t   = geom.R_t
#     R_e   = geom.R_e
#     L_c   = geom.L_c
#     L_e   = geom.L_e
#     R1    = geom.R1
#     RU    = geom.RU
#     RD    = geom.RD
#     AR    = geom.exp_ratio
#     theta1 = geom.theta1
#     thetaD = geom.thetaD
#     alpha  = geom.thetaE

#     th1_r  = np.deg2rad(theta1)
#     thD_r  = np.deg2rad(thetaD)
#     alp_r  = np.deg2rad(alpha)

#     # Convergence linear slope
#     m = ((R_t + RU - R_c + R1 - (RU + R1) * np.cos(th1_r))
#          / (L_c - L_e - (RU + R1) * np.sin(th1_r)))

#     if x <= L_e:
#         # Straight cylindrical chamber
#         return R_c

#     elif x <= L_e + R1 * np.sin(th1_r):
#         # Inner convergence fillet (concave arc, radius R1)
#         return np.sqrt(max(R1**2 - (x - L_e)**2, 0.0)) + R_c - R1

#     elif x <= L_c - RU * np.sin(th1_r):
#         # Linear convergence
#         return (m * x + R_c - R1 + R1 * np.cos(th1_r)
#                 - m * (L_e + R1 * np.sin(th1_r)))

#     elif x <= L_c:
#         # Outer convergence curve (convex arc, radius RU)
#         return -np.sqrt(max(RU**2 - (x - L_c)**2, 0.0)) + R_t + RU

#     elif x <= L_c + RD * np.sin(thD_r):
#         # Divergence throat fillet (concave arc, radius RD)
#         return -np.sqrt(max(RD**2 - (x - L_c)**2, 0.0)) + R_t + RD

#     else:
#         # Rao bell: quadratic Bezier from N (end of RD arc) to E (exit)
#         N_x = L_c + RD * np.cos(thD_r - np.pi/2)
#         N_y = RD * np.sin(thD_r - np.pi/2) + RD + R_t
#         E_x = L_c + geom.L_nozzle
#         E_y = R_e

#         denom = np.tan(thD_r) - np.tan(alp_r)
#         Q_x = (E_y - np.tan(alp_r)*E_x - N_y + np.tan(thD_r)*N_x) / denom
#         Q_y = (np.tan(thD_r)*(R_e - np.tan(alp_r)*E_x)
#                - np.tan(alp_r)*(N_y - np.tan(thD_r)*N_x)) / denom

#         # Solve quadratic for Bezier parameter t at station x
#         a =  N_x - 2*Q_x + E_x
#         b =  2*(Q_x - N_x)
#         c =  N_x - x
#         disc = max(b**2 - 4*a*c, 0.0)
#         if abs(a) > 1e-14:
#             t = (-b + np.sqrt(disc)) / (2*a)
#         else:
#             t = -c / b
#         t = np.clip(t, 0.0, 1.0)
#         return (1 - t)**2 * N_y + 2*t*(1 - t)*Q_y + t**2 * E_y


def build_contour(geom: EngineGeometry, dx: float = 5e-4):
    """Return (x_array, r_array) for the full engine profile."""
    total_length = geom.L_c + geom.L_nozzle
    x_arr = np.arange(0.0, total_length, dx)
    r_arr = np.array([nozzle_radius(x, geom, dx) for x in x_arr])
    return x_arr, r_arr


def plot_contour(geom: EngineGeometry, dx: float = 5e-4):
    """Plot the engine inner-wall profile (upper half shown)."""
    x_arr, r_arr = build_contour(geom, dx)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_arr * 1000,  r_arr * 1000, 'b-', linewidth=1.5)
    ax.plot(x_arr * 1000, -r_arr * 1000, 'b-', linewidth=1.5)
    ax.fill_between(x_arr * 1000,  r_arr * 1000, -r_arr * 1000, alpha=0.15)

    ax.axvline(geom.L_c * 1000, color='r', linestyle='--',
               linewidth=0.8, label=f'Throat  x={geom.L_c*1000:.1f} mm')
    ax.axvline(geom.L_e * 1000, color='g', linestyle=':',
               linewidth=0.8, label=f'Conv. start  x={geom.L_e*1000:.1f} mm')

    ax.set_xlabel('Axial position [mm]')
    ax.set_ylabel('Radius [mm]')
    ax.set_title('Engine Inner-Wall Contour')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)
