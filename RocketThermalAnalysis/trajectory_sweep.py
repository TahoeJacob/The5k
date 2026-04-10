"""
trajectory_sweep.py

Standalone 1-DOF vertical-ascent trajectory sim for the 5 kN RP-1/LOX vehicle,
used to answer: "does swapping AlSi10Mg -> Inconel 718 / GRCop-42 kill the
Karman line shot, and how much extra propellant do we need to recover?"

Atmosphere + gravity model copied from Archive/The5K.py (NASA GRC piecewise
standard atmosphere, inverse-square gravity).  Unlike The5K.py this skips the
CEA/tank-sizing pipeline entirely -- Isp, thrust, propellant mass, and dry
mass are direct inputs.  The hot gas is not the part of this sim that's
uncertain; the engine + structure mass is.

Cases swept:
  1. Al baseline                -- matches user's old 110 km number
  2. Inconel 718, same prop     -- quantify the mass-penalty apogee hit
  3. Inconel 718 + stretch tank -- how much extra prop to recover Karman
  4. GRCop-42, thin walls       -- best-of-both-worlds scenario

Usage:  python trajectory_sweep.py
"""
import numpy as np
import matplotlib.pyplot as plt


# -------- environment ---------------------------------------------------

def gravity(h):
    """Altitude-corrected gravity (m/s^2)."""
    R_e = 6.371e6
    return 9.81 * (R_e / (R_e + h)) ** 2


def air_density(h):
    """NASA GRC piecewise standard atmosphere, kg/m^3.
    Source: https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    (same model The5K.py uses)."""
    if h < 11_000:
        T_C = 15.05 - 0.00649 * h
        p_kPa = 101.29 * ((T_C + 273.15) / 288.08) ** 5.256
    elif h < 25_000:
        T_C = -56.46
        p_kPa = 22.65 * np.exp(1.73 - 0.000157 * h)
    else:
        T_C = -131.21 + 0.00299 * h
        p_kPa = 2.488 * ((T_C + 273.15) / 216.6) ** -11.388
    return p_kPa / (0.2869 * (T_C + 273.15))


# -------- vehicle + sim --------------------------------------------------

class Vehicle:
    def __init__(self, name, thrust, Isp, m_prop, m_dry, diameter, Cd=0.25):
        self.name = name
        self.thrust = thrust          # N  (sea-level-ish, constant)
        self.Isp = Isp                # s  (effective, averaged SL->vac)
        self.m_prop = m_prop          # kg
        self.m_dry = m_dry            # kg
        self.diameter = diameter      # m
        self.Cd = Cd
        self.mdot = thrust / (Isp * 9.81)
        self.t_burn = m_prop / self.mdot
        self.m_wet = m_dry + m_prop
        self.A = np.pi * (diameter / 2) ** 2

    def dv_ideal(self):
        return self.Isp * 9.81 * np.log(self.m_wet / self.m_dry)


def simulate(v: Vehicle, dt=0.05):
    """Euler-integrate vertical 1-DOF ascent and drag coast.
    Returns dict with apogee, v_burnout, h_burnout, time/alt/vel arrays."""
    t, h, vel = 0.0, 0.0, 0.0
    m = v.m_wet
    T, M, H, V, A_log = [], [], [], [], []

    # Powered phase
    while t < v.t_burn:
        g = gravity(h)
        rho = air_density(h)
        drag = 0.5 * rho * v.Cd * v.A * vel * vel
        if vel < 0:
            drag = -drag
        a = (v.thrust - m * g - drag) / m
        vel += a * dt
        h += vel * dt
        m -= v.mdot * dt
        t += dt
        T.append(t); H.append(h); V.append(vel); M.append(m); A_log.append(a)

    h_bo, v_bo, t_bo = h, vel, t
    m_bo = m  # frozen dry mass

    # Coast phase (thrust = 0), until apogee
    while vel > 0:
        g = gravity(h)
        rho = air_density(h)
        drag = 0.5 * rho * v.Cd * v.A * vel * vel
        a = -(g + drag / m_bo)
        vel += a * dt
        h += vel * dt
        t += dt
        T.append(t); H.append(h); V.append(vel); M.append(m_bo); A_log.append(a)

    return {
        "name": v.name,
        "apogee": h,
        "t_total": t,
        "h_burnout": h_bo,
        "v_burnout": v_bo,
        "t_burnout": t_bo,
        "dv_ideal": v.dv_ideal(),
        "t": np.array(T),
        "h": np.array(H),
        "v": np.array(V),
        "a": np.array(A_log),
        "vehicle": v,
    }


# -------- cases ----------------------------------------------------------
# All cases: 5 kN thrust, RP-1/LOX at ~25 bar Pc, 10:1 expansion.
# Effective Isp = SL/vac average, taken as 270 s for baseline and bumped
# where noted.  Diameter 0.152 m matches The5K.py airframe.

DIAM = 0.152
CD   = 0.25

def cases():
    return [
        Vehicle(
            name="Al (AlSi10Mg) baseline",
            thrust=5000, Isp=270,
            m_prop=40, m_dry=30,
            diameter=DIAM, Cd=CD,
        ),
        Vehicle(
            name="IN718, same prop (+5 kg chamber)",
            thrust=5000, Isp=270,
            m_prop=40, m_dry=35,
            diameter=DIAM, Cd=CD,
        ),
        Vehicle(
            name="IN718 + stretch tank (+10 kg prop)",
            thrust=5000, Isp=270,
            m_prop=50, m_dry=36,  # +1 kg tank for the stretch
            diameter=DIAM, Cd=CD,
        ),
        Vehicle(
            name="IN718 + stretch tank (+12 kg prop)",
            thrust=5000, Isp=270,
            m_prop=52, m_dry=36.2,
            diameter=DIAM, Cd=CD,
        ),
        Vehicle(
            name="GRCop-42 thin-wall (+2.5 kg chamber)",
            thrust=5000, Isp=270,
            m_prop=40, m_dry=32.5,
            diameter=DIAM, Cd=CD,
        ),
        Vehicle(
            name="IN718 + 30 bar Pc (Isp 278)",
            thrust=5000, Isp=278,
            m_prop=40, m_dry=35,
            diameter=DIAM, Cd=CD,
        ),
    ]


def main():
    KARMAN = 100_000
    results = [simulate(v) for v in cases()]

    print()
    print(f"{'Case':<42} {'m_wet':>7} {'t_burn':>7} "
          f"{'dv':>7} {'v_bo':>7} {'h_bo':>7} {'apogee':>8} {'Karman':>7}")
    print(f"{'':<42} {'[kg]':>7} {'[s]':>7} "
          f"{'[m/s]':>7} {'[m/s]':>7} {'[km]':>7} {'[km]':>8} {'':>7}")
    print("-" * 97)
    for r in results:
        v = r["vehicle"]
        mark = "YES" if r["apogee"] >= KARMAN else "no"
        print(f"{r['name']:<42} "
              f"{v.m_wet:>7.1f} {v.t_burn:>7.1f} "
              f"{r['dv_ideal']:>7.0f} {r['v_burnout']:>7.0f} "
              f"{r['h_burnout']/1000:>7.1f} {r['apogee']/1000:>8.1f} "
              f"{mark:>7}")
    print()

    # Altitude vs time overlay
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in results:
        ax.plot(r["t"], r["h"] / 1000, label=r["name"])
    ax.axhline(KARMAN / 1000, color="k", ls="--", lw=1, label="Karman line (100 km)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("5 kN RP-1/LOX sounding rocket — material trade apogee")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig("exports/trajectory_sweep.png", dpi=120)
    print("Plot saved: exports/trajectory_sweep.png")


if __name__ == "__main__":
    import os
    os.makedirs("exports", exist_ok=True)
    main()
