"""
flow_solver.py
1-D flow solver along the nozzle axis.

Two solvers are provided:

(1) Isentropic (adiabatic first pass)
    Uses the analytical isentropic area-Mach relation at each axial station.
    Exact, no ODE singularity at the throat.  Entry point: solve_flow().

(2) RK4 ODE (coupled iterations with friction / heat addition)
    Wang & Luong (1994) ODEs for 1-D compressible flow.
    State vector y = [N, P, T] where N = M².

    dN/dx = (N/(1-N)) * [(1+γN)/(Cp·T)·dQ + (2+(γ-1)N)/(R·T)·dF
                         - (2+(γ-1)N)/A · dA/dx]

    dP/dx = -(P/(1-N)) * [(γN)/(Cp·T)·dQ + (1+(γ-1)N)/(R·T)·dF
                          - (γN)/A · dA/dx]

    dT/dx = (T/(1-N)) * [(1-γN)/(Cp·T)·dQ - (γ-1)N/(R·T)·dF
                         + (γ-1)N/A · dA/dx]

    Non-zero dF and dQ perturb the throat saddle-point so the flow passes
    through M = 1 naturally.  The correct Mc is found via scipy.brentq
    bisection (~30 evaluations vs thousands for the old brute-force scan).
    Entry point: integrate() / find_Mc().

Source-term conventions (same as RocketThermalAnalysis.py reference code)
--------------------------------------------------------------------------
dQ[m]  — specific heat addition at step m [J/(kg·m)],
          set as  (q_gas * N_channels) / (mdot_chamber * dx)
dF[m]  — friction source term at step m [m²/s²],
          set as  f * v²/(2·Dh) * dx   (gas-side velocity head loss)
"""

import numpy as np
import scipy.optimize as sp
from dataclasses import dataclass
from typing import Optional

from geometry import EngineGeometry, nozzle_radius, exit_mach as _exit_mach
from cea_interface import CEAResult
from config import EngineConfig


# -----------------------------------------------------------------------
# Output dataclass
# -----------------------------------------------------------------------
@dataclass
class FlowSolution:
    """Flow-state arrays along the nozzle axis (injector face → exit)."""

    x:  np.ndarray   # Axial position from injector face [m]
    M:  np.ndarray   # Mach number
    P:  np.ndarray   # Static pressure [Pa]
    T:  np.ndarray   # Static temperature [K]
    A:  np.ndarray   # Local cross-sectional area [m²]
    Mc: float        # Injection Mach number


# -----------------------------------------------------------------------
# Shared: injection Mach from area-Mach relation
# -----------------------------------------------------------------------
def isentropic_Mc(cont_ratio: float, gam: float) -> float:
    """
    Subsonic injection Mach from  A/A* = cont_ratio.

    Parameters
    ----------
    cont_ratio : float — A_chamber / A_throat
    gam        : float — specific heat ratio

    Returns
    -------
    Mc : float — subsonic injection Mach number
    """
    def _eq(M: float) -> float:
        return ((1.0 / M)
                * ((1.0 + (gam - 1.0) / 2.0 * M**2)
                   / ((gam + 1.0) / 2.0))**((gam + 1.0) / (2.0 * (gam - 1.0)))
                - cont_ratio)
    return sp.brentq(_eq, 1e-7, 1.0 - 1e-7)


def _mach_from_AR(AR: float, gam: float, supersonic: bool) -> float:
    """
    Mach number from area ratio A/A*.

    AR         : float — A(x) / A_throat  (must be >= 1)
    supersonic : bool  — True for diverging section (M > 1)
    """
    if AR <= 1.0 + 1e-9:
        return 1.0

    def _eq(M: float) -> float:
        return ((1.0 / M)
                * ((1.0 + (gam - 1.0) / 2.0 * M**2)
                   / ((gam + 1.0) / 2.0))**((gam + 1.0) / (2.0 * (gam - 1.0)))
                - AR)

    if supersonic:
        return sp.brentq(_eq, 1.001, 50.0)
    else:
        return sp.brentq(_eq, 1e-7, 0.9999)


# -----------------------------------------------------------------------
# (1) Isentropic solver — analytical, no ODE
# -----------------------------------------------------------------------
def solve_flow(geom: EngineGeometry,
               cea:  CEAResult,
               config: EngineConfig,
               xf: Optional[float] = None,) -> FlowSolution:
    """
    Adiabatic first-pass flow profile using the analytical isentropic
    area-Mach relation.

    No ODE integration is needed, so there is no throat singularity.
    This gives the exact isentropic Mach, pressure, and temperature at
    every axial station.

    Parameters
    ----------
    geom   : EngineGeometry from size_engine()
    cea    : CEAResult for γ and stagnation conditions
    config : EngineConfig (supplies dx)

    Returns
    -------
    FlowSolution with M ≡ 1.0 at the throat and correct M_exit.
    """
    gam = cea.gamma_c
    P0  = cea.P_c   # stagnation pressure
    T0  = cea.T_c   # stagnation temperature

    if xf == None: 
        xf  = geom.L_c + geom.L_nozzle
    
    x_arr = np.arange(0.0, xf, config.dx) # x_arr in [m]
    A_t = geom.A_t
    M_arr = np.empty(len(x_arr))
    A_arr = np.empty(len(x_arr))
    for i, x in enumerate(x_arr):
        r    = nozzle_radius(x, geom, config.dx)
        A    = np.pi * r**2
        AR   = A / A_t
        sup  = (x >= geom.L_c)
        M_arr[i] = _mach_from_AR(AR, gam, supersonic=sup)
        A_arr[i] = A
        # print(M_arr[i], A_arr[i])

    # Isentropic P and T from stagnation conditions
    fac   = 1.0 + (gam - 1.0) / 2.0 * M_arr**2
    P_arr = P0 * fac**(-gam / (gam - 1.0))
    T_arr = T0 / fac

    Mc = float(M_arr[0])

    M_exit_theory = _exit_mach(geom.exp_ratio, gam)
    i_throat = int(np.argmin(A_arr))

    print(f"\n--- Adiabatic Flow Solution (isentropic) ---")
    print(f"  Contraction ratio  = {geom.A_c / A_t:.3f}")
    print(f"  Mc (injector)      = {Mc:.5f}")
    print(f"  M @ throat         = {M_arr[i_throat]:.5f}  (theory 1.000)")
    print(f"  M @ exit           = {M_arr[-1]:.4f}  (isentropic {M_exit_theory:.4f})")
    print(f"  P @ throat         = {P_arr[i_throat]/1e5:.3f} bar")
    print(f"  T @ throat         = {T_arr[i_throat]:.1f} K")

    return FlowSolution(x=x_arr, M=M_arr, P=P_arr, T=T_arr, A=A_arr, Mc=Mc)


# -----------------------------------------------------------------------
# (2) RK4 ODE solver — for coupled iterations with friction / heat
# -----------------------------------------------------------------------
def _dAdx(x: float, geom: EngineGeometry, h: float, dx: float) -> float:
    """Numerical dA/dx at position x (central difference, step h)."""
    rp = nozzle_radius(x + h, geom, dx)
    rm = nozzle_radius(x - h, geom, dx)
    return (np.pi * rp**2 - np.pi * rm**2) / (2.0 * h)


def _derivs(x: float, y: list, dx: float,
            gam: float, Cp: float, R_sp: float,
            geom: EngineGeometry, h: float,
            dF: float, dQ: float) -> list:
    """
    Wang (1994) ODEs. Returns [dN/dx, dP/dx, dT/dx].
    y = [N, P, T],  N = M².

    The 1/(1-N) denominator is guarded against division-by-zero; the
    physical L'Hôpital limit is finite wherever dA/dx and dF/dQ are
    smooth.  With non-zero dF or dQ the throat is no longer a pure
    saddle-point and the flow passes through M = 1 naturally.
    """
    N, P, T = y[0], y[1], y[2]
    r   = nozzle_radius(x, geom, dx)
    A   = np.pi * r**2
    dA  = _dAdx(x, geom, h, dx)

    denom = 1.0 - N
    # if abs(denom) < 1e-10:
    #     denom = 1e-10 * (1.0 if denom >= 0.0 else -1.0)

    fq = 1.0 / (Cp  * T)    # [kg/J]
    ff = 1.0 / (R_sp * T)   # [s²/m²]

    dNdx = (N / denom) * (
        (1.0 +  gam * N)          * fq  * dQ
      + (2.0 + (gam - 1.0) * N)   * ff  * dF
      - (2.0 + (gam - 1.0) * N)  / A   * dA
    )

    dPdx = -(P / denom) * (
          gam * N                  * fq  * dQ
      + (1.0 + (gam - 1.0) * N)   * ff  * dF
      -  gam * N                  / A   * dA
    )

    dTdx = (T / denom) * (
        (1.0 -  gam * N)           * fq  * dQ
      - (gam - 1.0) * N            * ff  * dF
      + (gam - 1.0) * N           / A   * dA
    )

    return [dNdx, dPdx, dTdx]


def _rk4_step(x: float, y: list, h: float, dx: float,
              gam: float, Cp: float, R_sp: float,
              geom: EngineGeometry,
              dF: float, dQ: float):
    """Single RK4 step. Returns (x_new, y_new)."""
    n  = 3
    ym = [0.0] * n
    ye = [0.0] * n

    k0 = _derivs(x,         y, dx, gam, Cp, R_sp, geom, h, dF, dQ)
    for i in range(n):
        ym[i] = y[i] + k0[i] * h / 2.0
    k1 = _derivs(x + h/2.0, ym, dx, gam, Cp, R_sp, geom, h, dF, dQ)
    for i in range(n):
        ym[i] = y[i] + k1[i] * h / 2.0
    k2 = _derivs(x + h/2.0, ym, dx, gam, Cp, R_sp, geom, h, dF, dQ)
    for i in range(n):
        ye[i] = y[i] + k2[i] * h
    k3 = _derivs(x + h,     ye, dx, gam, Cp, R_sp, geom, h, dF, dQ)

    y_new = [
        y[i] + h * (k0[i] + 2.0 * k1[i] + 2.0 * k2[i] + k3[i]) / 6.0
        for i in range(n)
    ]
    return x + h, y_new


def integrate(Mc: float,
              P_c: float,
              T_c: float,
              geom: EngineGeometry,
              cea: CEAResult,
              dx: float,
              dF_arr: Optional[np.ndarray] = None,
              dQ_arr: Optional[np.ndarray] = None) -> FlowSolution:
    """
    RK4 integration of Wang ODEs from x = 0 to x = L_c + L_nozzle.

    Intended for the coupled thermal iterations where non-zero dF/dQ
    naturally push the flow through the throat singularity.

    Parameters
    ----------
    Mc      : injection Mach number (initial condition at x = 0)
    P_c     : chamber pressure [Pa]
    T_c     : chamber temperature [K]
    geom    : EngineGeometry from size_engine()
    cea     : CEAResult for γ, Cp, R_specific
    dx      : integration step [m]
    dF_arr  : gas-side friction source (one value per step); None → zeros
    dQ_arr  : heat addition array    (one value per step); None → zeros

    Returns
    -------
    FlowSolution
    """
    gam  = cea.gamma_c
    Cp   = cea.Cp_c
    R_sp = cea.R_specific

    xf     = geom.L_c + geom.L_nozzle
    nsteps = max(1, int(round(xf / dx)))

    if dF_arr is None:
        dF_arr = np.zeros(nsteps)
    if dQ_arr is None:
        dQ_arr = np.zeros(nsteps)

    y = [Mc**2, P_c, T_c]
    x = 0.0
    m = 0

    x_list, N_list, P_list, T_list, A_list = [], [], [], [], []

    while x < xf - dx * 0.5:
        step = min(dx, xf - x)
        dF_m = float(dF_arr[m]) if m < len(dF_arr) else 0.0
        dQ_m = float(dQ_arr[m]) if m < len(dQ_arr) else 0.0

        x, y = _rk4_step(x, y, step, dx, gam, Cp, R_sp, geom, dF_m, dQ_m)

        x_list.append(x)
        N_list.append(y[0])
        P_list.append(y[1])
        T_list.append(y[2])
        A_list.append(np.pi * nozzle_radius(x, geom, dx)**2)
        m += 1

    N_arr = np.array(N_list)
    return FlowSolution(
        x  = np.array(x_list),
        M  = np.sqrt(np.maximum(N_arr, 0.0)),
        P  = np.array(P_list),
        T  = np.array(T_list),
        A  = np.array(A_list),
        Mc = Mc,
    )


def find_Mc(geom: EngineGeometry,
            cea: CEAResult,
            dx: float,
            dF_arr: Optional[np.ndarray] = None,
            dQ_arr: Optional[np.ndarray] = None) -> float:
    """
    Find the injection Mach Mc such that the RK4-integrated exit Mach
    matches the isentropic exit Mach for the given expansion ratio.

    Bisection cost:  cost(Mc) = M_exit_integrated(Mc) − M_exit_isentropic

    Called from the outer thermal iteration loop once dF / dQ are
    populated.  Non-zero source terms perturb the throat saddle-point
    so that the cost function is monotonic and well-behaved.

    Parameters
    ----------
    geom    : EngineGeometry
    cea     : CEAResult
    dx      : integration step [m]
    dF_arr  : friction source array (or None)
    dQ_arr  : heat-addition array   (or None)

    Returns
    -------
    Mc : float
    """
    M_exit_target = _exit_mach(geom.exp_ratio, cea.gamma_c)

    def cost(Mc: float) -> float:
        sol = integrate(Mc, cea.P_c, cea.T_c, geom, cea, dx, dF_arr, dQ_arr)
        return float(sol.M[-1]) - M_exit_target

    Mc_iso = isentropic_Mc(geom.A_c / geom.A_t, cea.gamma_c)
    lo = max(1e-4, Mc_iso * 0.4)
    hi = min(0.6,  Mc_iso * 4.0)

    c_lo, c_hi = cost(lo), cost(hi)
    if c_lo * c_hi > 0.0:
        raise RuntimeError(
            f"find_Mc: same sign at lo={lo:.5f} (cost={c_lo:+.4f}) and "
            f"hi={hi:.5f} (cost={c_hi:+.4f}).  "
            f"Ensure dF/dQ arrays are non-trivial.")

    return float(sp.brentq(cost, lo, hi, xtol=1e-6, maxiter=60))
