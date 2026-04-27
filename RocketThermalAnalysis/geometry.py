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
    R_chamber:    float  # Big chamber-side converging arc
    R_throat_conv: float # Convergent-side throat fillet (small)
    R_throat_div:  float # Divergent-side throat fillet

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
    R_chamber     = config.R_chamber_mult     * R_t
    R_throat_conv = config.R_throat_conv_mult * R_t
    R_throat_div  = config.R_throat_div_mult  * R_t

    # --- Chamber length via L* volume balance ---
    # Convergent cone length (includes throat-conv fillet at throat)
    th1 = np.deg2rad(config.theta1)
    L_cone = ((R_t * (np.sqrt(config.cont_ratio) - 1)
               + R_throat_conv * (1.0/np.cos(th1) - 1.0))
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
        R_chamber=R_chamber,
        R_throat_conv=R_throat_conv,
        R_throat_div=R_throat_div,
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
    print(f"  R_chamber = {g.R_chamber*1000:.2f} mm   "
          f"R_throat_conv = {g.R_throat_conv*1000:.2f} mm   "
          f"R_throat_div = {g.R_throat_div*1000:.2f} mm")
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
    R_chamber     = geom.R_chamber
    R_throat_conv = geom.R_throat_conv
    R_throat_div  = geom.R_throat_div
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
    r1 = R_throat_conv  # convergent-side throat fillet radius [m]
    ang1_start = -(90.0 + theta1) * np.pi/180.0     # Convergent radius angle
    xP4 = r1 * np.cos(ang1_start)                   # Convergent radius X coordinate
    yP4 = r1 * np.sin(ang1_start) + (r1 + R_t)      # Convergent radius Y coordinate

    r2 = R_throat_div   # divergent-side throat fillet radius [m]
    ang2_end = (thetaD - 90) * np.pi/180            # Divergent radius angle
    Nx = r2 * np.cos(ang2_end)                      # Divergent radius X coordinate
    Ny = r2 * np.sin(ang2_end) + (r2 + R_t)         # Divergent radius Y coordinate

    Ex, Ey = L_e, R_e

    # Converging big arc tangent to straight chamber and -theta1 line
    rc = R_chamber
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


def _dxf_line(layer, x1, y1, x2, y2):
    return ('0\nLINE\n8\n' + layer +
            f'\n10\n{x1:.5f}\n20\n{y1:.5f}\n30\n0.0'
            f'\n11\n{x2:.5f}\n21\n{y2:.5f}\n31\n0.0')


def _dxf_arc(layer, cx, cy, r, start_deg, end_deg):
    # DXF arcs are always CCW from start_deg to end_deg
    return ('0\nARC\n8\n' + layer +
            f'\n10\n{cx:.5f}\n20\n{cy:.5f}\n30\n0.0'
            f'\n40\n{r:.5f}'
            f'\n50\n{start_deg:.5f}\n51\n{end_deg:.5f}')


def _dxf_point(layer, x, y):
    return ('0\nPOINT\n8\n' + layer +
            f'\n10\n{x:.5f}\n20\n{y:.5f}\n30\n0.0')


def _dxf_lwpolyline(layer, pts, closed=True):
    """Closed (or open) LWPOLYLINE from a list of (x, y) tuples."""
    flag = 1 if closed else 0
    head = ('0\nLWPOLYLINE\n8\n' + layer +
            f'\n90\n{len(pts)}\n70\n{flag}')
    body = ''.join(f'\n10\n{x:.5f}\n20\n{y:.5f}' for x, y in pts)
    return head + body


def _bell_control_points(geom: 'EngineGeometry'):
    """
    Return (Nx, Ny, Qx, Qy, Ex, Ey) for the bell quadratic Bezier in
    throat-centered MILLIMETER coordinates, plus a list of 3 control-point
    tuples for printing.
    """
    R_t = geom.R_t
    R_e = geom.R_e
    L_n = geom.L_nozzle
    thD = np.deg2rad(geom.thetaD)
    alp = np.deg2rad(geom.thetaE)
    r2  = geom.R_throat_div

    a2 = (geom.thetaD - 90.0) * np.pi/180.0
    Nx = r2 * np.cos(a2)
    Ny = r2 * np.sin(a2) + (r2 + R_t)

    Ex, Ey = L_n, R_e
    m1, m2 = np.tan(thD), np.tan(alp)
    C1 = Ny - m1 * Nx
    C2 = Ey - m2 * Ex
    Qx = (C2 - C1) / (m1 - m2)
    Qy = (m1*C2 - m2*C1) / (m1 - m2)

    M = 1000.0
    pts = [(Nx*M, Ny*M), (Qx*M, Qy*M), (Ex*M, Ey*M)]
    return None, None, pts


def _write_dxf(filename, entities):
    """entities = list of pre-formatted DXF entity strings."""
    dxf = (
        '0\nSECTION\n2\nHEADER\n'
        '9\n$INSUNITS\n70\n4\n'          # 4 = millimetres
        '9\n$MEASUREMENT\n70\n1\n'       # 1 = metric
        '0\nENDSEC\n'
        '0\nSECTION\n2\nENTITIES\n'
        + '\n'.join(entities) + '\n'
        '0\nENDSEC\n'
        '0\nEOF\n'
    )
    with open(filename, 'w') as f:
        f.write(dxf)


def _contour_primitives(geom: 'EngineGeometry', layer: str = 'CONTOUR',
                        radial_offset: float = 0.0):
    """
    Return (entities, key_points) for the inner-wall contour as a list of
    DXF primitive entities (LINE/ARC) plus a coarse LWPOLYLINE for the bell.

    Coordinates are throat-centered, in MILLIMETERS.
    `radial_offset` (m) shifts every radius outward by that amount — used to
    generate channel floor / centerline / ceiling curves on the same primitives.
    """
    R_c = geom.R_c
    R_t = geom.R_t
    R_e = geom.R_e
    L_c = geom.L_c
    L_n = geom.L_nozzle
    th1 = np.deg2rad(geom.theta1)
    thD = np.deg2rad(geom.thetaD)
    alp = np.deg2rad(geom.thetaE)

    r1 = geom.R_throat_conv          # convergent throat fillet [m]
    r2 = geom.R_throat_div           # divergent throat fillet [m]
    rc = geom.R_chamber              # big chamber arc [m]

    # Convergent fillet endpoint P4 (throat-centered, in m)
    a1 = -(90.0 + geom.theta1) * np.pi/180.0
    xP4 = r1 * np.cos(a1)
    yP4 = r1 * np.sin(a1) + (r1 + R_t)

    # Divergent fillet endpoint N (start of bell)
    a2 = (geom.thetaD - 90.0) * np.pi/180.0
    Nx = r2 * np.cos(a2)
    Ny = r2 * np.sin(a2) + (r2 + R_t)

    # Big chamber arc center & tangent point P3
    mc = -np.tan(th1)
    bc = yP4 - mc * xP4
    yc = R_c - rc
    s  = np.sqrt(mc*mc + 1.0)
    xc_plus  = (yc - bc + rc*s)/mc
    xc_minus = (yc - bc - rc*s)/mc
    xc = xc_minus if xc_minus < xc_plus else xc_plus
    S = mc*xc - yc + bc
    xP3 = xc - mc * S / (mc*mc + 1.0)
    yP3 = yc + S / (mc*mc + 1.0)

    # Bell exit (throat-centered)
    Ex, Ey = L_n, R_e

    # Bezier control point Q (from slope constraints at N and E)
    m1, m2 = np.tan(thD), np.tan(alp)
    C1 = Ny - m1 * Nx
    C2 = Ey - m2 * Ex
    Qx = (C2 - C1) / (m1 - m2)
    Qy = (m1*C2 - m2*C1) / (m1 - m2)

    # ---------- Apply radial offset to every radius ----------
    off = radial_offset
    R_c_o = R_c + off
    R_t_o = R_t + off
    R_e_o = R_e + off
    yP4_o = yP4 + off
    Ny_o  = Ny  + off
    yc_o  = yc  + off       # f2 center moves outward by `off`
    yP3_o = yP3 + off
    Qy_o  = Qy  + off
    Ey_o  = Ey  + off
    # f4 center: (0, r1+R_t) → (0, r1+R_t+off)
    f4_cy = (r1 + R_t) + off
    # f5 center: (0, r2+R_t) → (0, r2+R_t+off)
    f5_cy = (r2 + R_t) + off
    # NOTE: arc radii (rc, r1, r2) are unchanged; only centers and wall
    # radii shift.  This produces a parallel offset curve, which is exact
    # for straight lines and circular arcs.

    x_inj = -L_c            # injector face (throat-centered)

    # ---------- Build entities (mm, throat-centered) ----------
    M = 1000.0    # m → mm
    ents = []

    # f1: straight chamber  (x_inj, R_c) → (xc, R_c)
    ents.append(_dxf_line(layer, x_inj*M, R_c_o*M, xc*M, R_c_o*M))

    # f2: chamber arc  center (xc, yc), radius rc, from P3 to (xc, R_c) [CCW]
    a_top = 90.0   # (xc, R_c) is directly above center
    a_P3  = np.degrees(np.arctan2(yP3_o - yc_o, xP3 - xc))
    ents.append(_dxf_arc(layer, xc*M, yc_o*M, rc*M,
                         min(a_top, a_P3), max(a_top, a_P3)))

    # f3: linear taper  P3 → P4
    ents.append(_dxf_line(layer, xP3*M, yP3_o*M, xP4*M, yP4_o*M))

    # f4: throat conv fillet  center (0, r1+R_t), radius r1, from P4 to throat
    a_P4     = np.degrees(np.arctan2(yP4_o - f4_cy, xP4 - 0.0))
    a_throat = -90.0     # (0, R_t) is directly below center
    a_P4_n     = a_P4 + 360.0     if a_P4 < 0     else a_P4
    a_throat_n = a_throat + 360.0 if a_throat < 0 else a_throat
    ents.append(_dxf_arc(layer, 0.0, f4_cy*M, r1*M,
                         min(a_P4_n, a_throat_n), max(a_P4_n, a_throat_n)))

    # f5: throat div fillet  center (0, r2+R_t), radius r2, from throat to N
    a_N     = np.degrees(np.arctan2(Ny_o - f5_cy, Nx - 0.0))
    a_thr5  = -90.0
    a_N_n     = a_N + 360.0    if a_N < 0    else a_N
    a_thr5_n  = a_thr5 + 360.0 if a_thr5 < 0 else a_thr5
    ents.append(_dxf_arc(layer, 0.0, f5_cy*M, r2*M,
                         min(a_N_n, a_thr5_n), max(a_N_n, a_thr5_n)))

    # f6: Bezier bell  →  full control polygon as two LINE entities
    # OnShape drops standalone POINT entities on custom layers, so instead of
    # exporting Q as a POINT we draw the canonical Bezier "control polygon":
    # N → Q  and  Q → E.  Q becomes the explicit meeting vertex of the two
    # lines and is directly snappable in the sketch.
    #
    # In OnShape: Sketch → Spline → Control Point Spline (degree 2), then
    # click N → Q → E (snap to the line endpoints / intersection vertex).
    # The resulting curve is mathematically identical to our analytical bell.
    ents.append(_dxf_line('BELL_POLY', Nx*M, Ny_o*M, Qx*M, Qy_o*M))
    ents.append(_dxf_line('BELL_POLY', Qx*M, Qy_o*M, Ex*M, Ey_o*M))

    # Also add POINTs at the three control vertices on a CONTROL layer
    # (in case the OnShape importer is happy with POINTs on this particular
    # layer name — they're harmless extras if not).
    ents.append(_dxf_point('CONTROL', Nx*M, Ny_o*M))
    ents.append(_dxf_point('CONTROL', Qx*M, Qy_o*M))
    ents.append(_dxf_point('CONTROL', Ex*M, Ey_o*M))

    # ---------- Snap points at every transition ----------
    key_points = [
        ('inj',    x_inj, R_c_o),
        ('xc',     xc,    R_c_o),
        ('P3',     xP3,   yP3_o),
        ('P4',     xP4,   yP4_o),
        ('throat', 0.0,   R_t_o),
        ('N',      Nx,    Ny_o),
        ('exit',   Ex,    Ey_o),
    ]
    for _, x, y in key_points:
        ents.append(_dxf_point('POINTS', x*M, y*M))

    return ents, key_points


def export_dxf(geom: 'EngineGeometry', chan_geom, out_dir: str = 'exports'):
    """
    Export the engine geometry as DXF files for OnShape sketch import.

    Uses native DXF primitives (LINE / ARC / LWPOLYLINE) so each segment is
    a separately-dimensionable sketch entity, with POINT markers at every
    transition for snapping.  Coordinates are throat-centered (mm), so the
    throat lies at (0, R_t) and the X-axis IS the revolve axis.

    Files:
      engine_contour.dxf  — closed revolve profile: inner (hot-gas) wall
                            on layer CONTOUR_HOT and outer (coolant-side)
                            wall on layer CONTOUR_COOL, offset radially
                            by the wall thickness t_w, capped at both
                            ends so the result is a closed region ready
                            to revolve.
      channel_paths.dxf   — channel floor / center / ceiling contours on
                            three layers (offset from hot wall).
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Representative wall thickness and channel height (taper-averaged).
    t_w_arr = np.asarray(chan_geom.chan_t)
    h_arr   = np.asarray(chan_geom.chan_h)
    t_rep   = float(np.median(t_w_arr))
    h_rep   = float(np.median(h_arr))

    # ---------- 1) Closed wall contour (hot + cool + end caps) ----------
    L_c_mm = geom.L_c  * 1000.0   # throat-centered: injector at -L_c, exit at +L_n
    L_n_mm = geom.L_nozzle * 1000.0
    R_c_mm = geom.R_c * 1000.0
    R_e_mm = geom.R_e * 1000.0
    t_w_mm = t_rep    * 1000.0

    hot_ents,  _ = _contour_primitives(geom, layer='CONTOUR_HOT',  radial_offset=0.0)
    cool_ents, _ = _contour_primitives(geom, layer='CONTOUR_COOL', radial_offset=t_rep)

    # End caps (injector-face and nozzle-exit) joining hot ↔ cool walls
    cap_injector = _dxf_line('CONTOUR_CAP',
                             -L_c_mm, R_c_mm,
                             -L_c_mm, R_c_mm + t_w_mm)
    cap_exit     = _dxf_line('CONTOUR_CAP',
                              L_n_mm, R_e_mm,
                              L_n_mm, R_e_mm + t_w_mm)

    contour_ents = hot_ents + cool_ents + [cap_injector, cap_exit]
    contour_dxf = os.path.join(out_dir, 'engine_contour.dxf')
    _write_dxf(contour_dxf, contour_ents)

    # ---------- 1b) Channel-ceiling revolve profile ----------
    # Closed profile: hot wall + channel ceiling (offset by t_w + h_chan)
    # + end caps.  Revolve this to get a solid ring reaching out to the
    # top of the channels — use it as a revolve-cut reference for the
    # channel height.
    ceil_off_m = t_rep + h_rep
    ceil_off_mm = ceil_off_m * 1000.0
    ceil_hot_ents,  _ = _contour_primitives(geom, layer='CEIL_HOT',
                                             radial_offset=0.0)
    ceil_top_ents,  _ = _contour_primitives(geom, layer='CEIL_TOP',
                                             radial_offset=ceil_off_m)
    ceil_cap_inj = _dxf_line('CEIL_CAP',
                             -L_c_mm, R_c_mm,
                             -L_c_mm, R_c_mm + ceil_off_mm)
    ceil_cap_exit = _dxf_line('CEIL_CAP',
                               L_n_mm, R_e_mm,
                               L_n_mm, R_e_mm + ceil_off_mm)
    ceiling_ents = ceil_hot_ents + ceil_top_ents + [ceil_cap_inj, ceil_cap_exit]
    ceiling_dxf = os.path.join(out_dir, 'channel_ceiling.dxf')
    _write_dxf(ceiling_dxf, ceiling_ents)

    # ---------- 2) Channel paths ----------

    floor_off  = t_rep
    center_off = t_rep + 0.5 * h_rep
    ceil_off   = t_rep + h_rep
    # (t_rep / h_rep already computed above)

    floor_ents,  _ = _contour_primitives(geom, 'CH_FLOOR',   radial_offset=floor_off)
    center_ents, _ = _contour_primitives(geom, 'CH_CENTER',  radial_offset=center_off)
    ceil_ents,   _ = _contour_primitives(geom, 'CH_CEILING', radial_offset=ceil_off)

    paths_dxf = os.path.join(out_dir, 'channel_paths.dxf')
    _write_dxf(paths_dxf, floor_ents + center_ents + ceil_ents)

    # Print the 3 Bezier control points so the user can manually create an
    # exact 3-control-point spline in OnShape if the polyline isn't smooth enough.
    _, _, bell_ctrl = _bell_control_points(geom)
    print(f"\n--- DXF export (throat-centered, x=0 at throat) ---")
    print(f"  {contour_dxf}  (hot+cool walls, t_w={t_w_mm:.2f} mm, end-capped)")
    print(f"  {ceiling_dxf}  (hot wall + channel ceiling at t_w+h={ceil_off_mm:.2f} mm, end-capped)")
    print(f"  {paths_dxf}   (offset {floor_off*1000:.2f}/{center_off*1000:.2f}/"
          f"{ceil_off*1000:.2f} mm on 3 layers)")
    print(f"  Bell Bezier control points (mm, throat-centered) — for manual")
    print(f"  3-pt control-point spline in OnShape if you want the exact curve:")
    for label, (x, y) in zip(('N (start)', 'Q (ctrl)', 'E (exit) '), bell_ctrl):
        print(f"    {label}  ({x:9.4f}, {y:9.4f})")

    # ---------- 3) Top-down land plan ----------
    land_dxf = os.path.join(out_dir, 'land_plan.dxf')
    n_top = export_land_plan_dxf(geom, chan_geom, land_dxf)
    print(f"  {land_dxf}   (top-down land strip, {n_top} key vertices per edge)")


def export_land_plan_dxf(geom: 'EngineGeometry', chan_geom,
                         filename: str):
    """
    Write a top-down DXF showing ONE land as a closed strip along the
    engine axis, with a MINIMAL vertex count.

    Only one vertex per "slope change":
      - straight chamber (f1) and linear taper (f3): 2 vertices each
      - curved segments (chamber arc, throat fillets, bell): start +
        mid + end (3 vertices each), so each curve can be fit to an
        arc/spline and radius-dimensioned in OnShape
      - bifurcation transitions: extra vertex at each N-change location

    Coordinates:
      X = axial position, throat-centered, millimetres
      Y = circumferential land half-width, millimetres (±land/2)
    """
    x_j   = np.asarray(chan_geom.x_j)
    land  = np.asarray(chan_geom.chan_land)
    chw   = np.asarray(chan_geom.chan_w)
    n_ch  = np.asarray(getattr(chan_geom, 'n_chan', np.zeros_like(x_j)))
    L_c   = geom.L_c
    L_total = geom.L_c + geom.L_nozzle
    M     = 1000.0

    # Bifurcation extent: N_throat = min(n_ch), N_chamber = max(n_ch).
    # Fraction of "extra" land present at each station (0 at throat, 1 in
    # fully-bifurcated chamber, linear ramp through transitions).
    N_throat_i  = float(np.min(n_ch)) if n_ch.size else 1.0
    N_chamber_i = float(np.max(n_ch)) if n_ch.size else 1.0
    if N_chamber_i > N_throat_i + 1e-9:
        extra_frac = (n_ch - N_throat_i) / (N_chamber_i - N_throat_i)
    else:
        extra_frac = np.zeros_like(x_j)
    extra_land = land * extra_frac   # width of one full extra land [m]

    # --- Collect key x values (injector-face coordinates) ---
    segs = _segment_breakpoints(geom)
    curve_labels = {"f2: chamber arc R_chamber",
                    "f4: throat conv fillet",
                    "f5: throat div fillet",
                    "f6: Bezier bell"}

    key_x = set()
    key_x.add(0.0)
    key_x.add(L_total)
    for label, x0, x1, _ in segs:
        if x1 - x0 < 1e-9:
            continue
        key_x.add(x0)
        key_x.add(x1)
        if label in curve_labels:
            key_x.add(0.5 * (x0 + x1))

    # Bifurcation transitions: detect where n_chan actually changes
    n_chan = getattr(chan_geom, 'n_chan', None)
    if n_chan is not None:
        n_arr = np.asarray(n_chan)
        dn = np.abs(np.diff(n_arr))
        tol = 0.01 * max(1.0, float(np.max(np.abs(n_arr))))
        in_trans = False
        for i, d in enumerate(dn):
            if d > tol and not in_trans:
                key_x.add(float(x_j[i]))
                in_trans = True
            elif d <= tol and in_trans:
                key_x.add(float(x_j[i]))
                in_trans = False

    # Clamp to engine extent and sort
    key_x = sorted(x for x in key_x if 0.0 <= x <= L_total)

    # Interpolate per-station fields at each key x
    key_arr = np.asarray(key_x)
    main_hw_m  = 0.5 * np.interp(key_arr, x_j, land)        # main land half-width
    chw_m      =       np.interp(key_arr, x_j, chw)          # channel width
    ext_m      =       np.interp(key_arr, x_j, extra_land)   # full extra-land width
    ext_hw_m   = 0.5 * ext_m                                 # half of one extra land
    x_mm       = [(x - L_c) * M for x in key_x]
    main_hw_mm = [float(h * M) for h in main_hw_m]
    chw_mm     = [float(c * M) for c in chw_m]
    ext_hw_mm  = [float(e * M) for e in ext_hw_m]

    # Main land HALF-strip (mirror axis = DXF Y axis, which is the
    # central axis of the main land).  Left edge sits on the mirror
    # axis (x = 0); right edge is at +main_hw.  User mirrors this in
    # OnShape after dimensioning.
    outer_main = [(float(h),  float(x)) for h, x in zip(main_hw_mm, x_mm)]
    axis_main  = [(0.0,       float(x)) for x in reversed(x_mm)]
    main_poly  = outer_main + axis_main

    ents = [_dxf_lwpolyline('LAND_MAIN', main_poly, closed=True)]

    # Right extra half-land — only present in the bifurcated region.
    # Inner edge (closer to main): x = main_hw + chan_w
    # Outer edge (farther):         x = main_hw + chan_w + ext_hw
    inner_x = [mh + w for mh, w in zip(main_hw_mm, chw_mm)]
    outer_x = [ix + eh for ix, eh in zip(inner_x, ext_hw_mm)]

    tol_mm = 1e-4
    runs, current = [], []
    for i, eh in enumerate(ext_hw_mm):
        if eh > tol_mm:
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    for run in runs:
        i0, i1 = run[0], run[-1]
        outer_edge = [(outer_x[i], x_mm[i]) for i in run]
        inner_edge = [(inner_x[i], x_mm[i]) for i in run]
        # Pinch closed at each end along the inner (main-side) edge
        poly = ([(inner_x[i0], x_mm[i0])] + outer_edge
                + [(inner_x[i1], x_mm[i1])] + list(reversed(inner_edge)))
        ents.append(_dxf_lwpolyline('LAND_EXTRA', poly, closed=True))

    # Snap point at the engine start, end and throat (axial Y = 0 at throat)
    ents.append(_dxf_point('LAND_SNAP', 0.0, x_mm[0]))
    ents.append(_dxf_point('LAND_SNAP', 0.0, x_mm[-1]))
    ents.append(_dxf_point('LAND_SNAP', 0.0, 0.0))

    _write_dxf(filename, ents)
    return len(outer_main)


def export_csv(geom: 'EngineGeometry', chan_geom, out_dir: str = 'exports',
               key_stations=None, key_stations_N_throat=None):
    """
    Export the engine geometry as three CSV files for OnShape import.

    All x-coordinates are THROAT-CENTERED (x = 0 at the throat plane), so
    the OnShape sketch origin (0,0) lines up with the throat.

    Files written:
      engine_contour.csv     — inner wall profile (x, r) for revolve sketch
      channel_paths.csv      — channel floor / centerline / ceiling radii
                                (use as sweep paths or revolve sketches)
      channel_dimensions.csv — full bookkeeping at every axial station:
                                N, w, h, land, t_w, r_inner
    """
    import os
    import csv

    os.makedirs(out_dir, exist_ok=True)
    L_c = geom.L_c

    # ---------- 1) inner wall contour ----------
    x_arr, r_arr = build_contour(geom, dx=2e-4)   # 0.2 mm sampling for smoothness
    contour_path = os.path.join(out_dir, 'engine_contour.csv')
    with open(contour_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x_mm', 'r_mm'])
        for x, r in zip(x_arr, r_arr):
            w.writerow([f'{(x - L_c)*1000:.5f}', f'{r*1000:.5f}'])

    # ---------- 2) channel paths (floor, centerline, ceiling) ----------
    x_j = chan_geom.x_j
    c_t = chan_geom.chan_t
    c_h = chan_geom.chan_h
    paths_path = os.path.join(out_dir, 'channel_paths.csv')
    with open(paths_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x_mm', 'r_inner_mm', 'r_floor_mm',
                    'r_centerline_mm', 'r_ceiling_mm'])
        for i, x in enumerate(x_j):
            r_inner = nozzle_radius(x, geom, 1e-3)
            r_floor = r_inner + c_t[i]
            r_ceil  = r_floor + c_h[i]
            r_mid   = 0.5 * (r_floor + r_ceil)
            w.writerow([f'{(x - L_c)*1000:.5f}',
                        f'{r_inner*1000:.5f}',
                        f'{r_floor*1000:.5f}',
                        f'{r_mid*1000:.5f}',
                        f'{r_ceil*1000:.5f}'])

    # ---------- 3) channel dimensions at key stations only ----------
    # If key_stations is provided, use those; otherwise fall back to all stations
    c_w  = chan_geom.chan_w
    c_ld = chan_geom.chan_land
    n_ch = getattr(chan_geom, 'n_chan', None)
    dims_path = os.path.join(out_dir, 'channel_dimensions.csv')

    if key_stations is not None:
        # N_throat = number of primary sweep paths (75).  At stations where
        # channels bifurcate (N > N_throat), we report dimensions from the
        # perspective of one of the 75 slots: slot_pitch, and within that
        # slot either 1 channel or 2 sub-channels with a land between them.
        N_throat = key_stations_N_throat or 75
        with open(dims_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['station', 'x_throat_mm', 'r_inner_mm',
                        'slot_pitch_mm', 'ch_per_slot',
                        'ch_width_mm', 'ch_height_mm',
                        'land_mm', 'wall_t_mm',
                        'r_floor_mm', 'r_ceiling_mm'])
            for label, x_inj in key_stations:
                x_tc = (x_inj - L_c) * 1000.0
                r_inner = nozzle_radius(x_inj, geom, 1e-3)
                h_val  = float(np.interp(x_inj, x_j, c_h))
                w_val  = float(np.interp(x_inj, x_j, c_w))
                ld_val = float(np.interp(x_inj, x_j, c_ld))
                t_val  = float(np.interp(x_inj, x_j, c_t))
                n_val  = float(np.interp(x_inj, x_j, n_ch)) if n_ch is not None else float('nan')
                r_floor = r_inner + t_val
                r_ceil  = r_floor + h_val
                slot_pitch = 2.0 * np.pi * r_inner / N_throat
                ch_per_slot = max(1, round(n_val / N_throat))
                w.writerow([label, f'{x_tc:.3f}',
                            f'{r_inner*1000:.3f}',
                            f'{slot_pitch*1000:.3f}', f'{ch_per_slot}',
                            f'{w_val*1000:.3f}', f'{h_val*1000:.3f}',
                            f'{ld_val*1000:.3f}', f'{t_val*1000:.3f}',
                            f'{r_floor*1000:.3f}', f'{r_ceil*1000:.3f}'])
        n_rows = len(key_stations)
    else:
        with open(dims_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['x_mm', 'r_inner_mm', 'N_channels',
                        'width_mm', 'height_mm', 'land_mm', 'wall_t_mm'])
            for i, x in enumerate(x_j):
                r_inner = nozzle_radius(x, geom, 1e-3)
                N_local = float(n_ch[i]) if n_ch is not None else float('nan')
                w.writerow([f'{(x - L_c)*1000:.5f}',
                            f'{r_inner*1000:.5f}',
                            f'{N_local:.3f}',
                            f'{c_w[i]*1000:.5f}',
                            f'{c_h[i]*1000:.5f}',
                            f'{c_ld[i]*1000:.5f}',
                            f'{c_t[i]*1000:.5f}'])
        n_rows = len(x_j)

    print(f"\n--- CSV export (throat-centered, x=0 at throat) ---")
    print(f"  {contour_path}    ({len(x_arr)} points)")
    print(f"  {paths_path}     ({len(x_j)} points)")
    print(f"  {dims_path} ({n_rows} stations)")


def _segment_breakpoints(geom: EngineGeometry):
    """
    Compute the axial breakpoints of the 6 contour segments, in injector-face
    coordinates (x = 0 at injector face).  Returns a list of
    (label, x_start, x_end, color) tuples.
    """
    R_c   = geom.R_c
    R_t   = geom.R_t
    L_c   = geom.L_c
    th1_r = np.deg2rad(geom.theta1)

    r1 = geom.R_throat_conv
    r2 = geom.R_throat_div
    rc = geom.R_chamber

    # Throat-centric coordinates (x' = x - L_c)
    ang1_start = -(90.0 + geom.theta1) * np.pi/180.0
    xP4 = r1 * np.cos(ang1_start)
    yP4 = r1 * np.sin(ang1_start) + (r1 + R_t)

    ang2_end = (geom.thetaD - 90.0) * np.pi/180.0
    Nx = r2 * np.cos(ang2_end)

    mc = -np.tan(th1_r)
    bc = yP4 - mc * xP4
    yc = R_c - rc
    s  = np.sqrt(mc*mc + 1.0)
    xc_plus  = (yc - bc + rc*s)/mc
    xc_minus = (yc - bc - rc*s)/mc
    xc = xc_minus if xc_minus < xc_plus else xc_plus

    S = mc*xc - yc + bc
    xP3 = xc - mc * S / (mc*mc + 1.0)

    # Convert to injector-face coordinates and engine-end exit
    x_chamber_end  = L_c + xc          # straight chamber → big arc
    x_arc_end      = L_c + xP3         # big arc → linear taper
    x_taper_end    = L_c + xP4         # linear taper → throat-conv fillet
    x_throat       = L_c               # throat
    x_div_fil_end  = L_c + Nx          # throat-div fillet → bell
    x_exit         = L_c + geom.L_nozzle

    return [
        ("f1: straight chamber",       0.0,            x_chamber_end, "#1f77b4"),
        ("f2: chamber arc R_chamber",  x_chamber_end,  x_arc_end,     "#ff7f0e"),
        ("f3: linear taper θ1",        x_arc_end,      x_taper_end,   "#2ca02c"),
        ("f4: throat conv fillet",     x_taper_end,    x_throat,      "#d62728"),
        ("f5: throat div fillet",      x_throat,       x_div_fil_end, "#9467bd"),
        ("f6: Bezier bell",            x_div_fil_end,  x_exit,        "#8c564b"),
    ]


def plot_contour(geom: EngineGeometry, dx: float = 5e-4):
    """Plot the engine inner-wall profile with color-coded segments."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    segs = _segment_breakpoints(geom)
    for label, x0, x1, color in segs:
        if x1 - x0 < 1e-9:
            continue
        n = max(int(np.ceil((x1 - x0) / dx)) + 1, 2)
        xs = np.linspace(x0, x1, n)
        rs = np.array([nozzle_radius(x, geom, dx) for x in xs])
        ax.plot(xs * 1000,  rs * 1000, '-', color=color, linewidth=2.0, label=label)
        ax.plot(xs * 1000, -rs * 1000, '-', color=color, linewidth=2.0)
        ax.fill_between(xs * 1000, rs * 1000, -rs * 1000, color=color, alpha=0.10)

    ax.axvline(geom.L_c * 1000, color='k', linestyle='--', linewidth=0.8,
               label=f'Throat x={geom.L_c*1000:.1f} mm')

    # Annotate the curvature radii
    info = (f"R_chamber     = {geom.R_chamber*1000:6.2f} mm\n"
            f"R_throat_conv = {geom.R_throat_conv*1000:6.2f} mm\n"
            f"R_throat_div  = {geom.R_throat_div*1000:6.2f} mm\n"
            f"R_t           = {geom.R_t*1000:6.2f} mm")
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=8,
            family='monospace', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

    ax.set_xlabel('Axial position [mm]')
    ax.set_ylabel('Radius [mm]')
    ax.set_title('Engine Inner-Wall Contour (segments)')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
