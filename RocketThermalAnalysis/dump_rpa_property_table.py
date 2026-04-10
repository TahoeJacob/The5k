"""
dump_rpa_property_table.py
Generate a properties database block for 95/5 ethanol-water (by mass) from
REFPROP, in the *real* RPA `resources/properties.inp` format (Format Version 2)
as used by the bundled C2H5OH(L) entry:

    SpeciesName       n_p,n_T
      P  T  Cp  rho  mu  lambda  [Ts  Qs]   ← Ts/Qs only on first row of each P

  P       in MPa
  T       in K
  Cp      in kJ/(kg·K)               ← mass basis, NOT kJ/(mol·K) (manual p.48 is wrong)
  rho     in kg/m³
  mu      in µPa·s
  lambda  in mW/(m·K)                ← lives in properties.inp, NOT trans.inp
  Ts      saturation temperature [K] at this pressure (first row of P-block only)
  Qs      latent heat [kJ/kg] at this pressure (first row of P-block only)

For supercritical pressures we follow RPA's bundled-file convention: print the
critical-point temperature with Qs=0.0.  For (T > T_sat) cells inside a sub-
critical pressure block, REFPROP returns the gas-phase root — same approach
RPA's bundled C2H5OH(L) takes (line 110: 0.101325 MPa jumps from rho=754 at
333 K to rho=1.6 at 353 K, crossing T_sat≈351 K).

Why this exists:
RPA's bundled "Gurvich, calculated" 95/5 ethanol-water table appears to use
T-independent (or wrong-T-dependence) viscosities, inflating its h_cool
estimate's denominator and giving T_hw ≈ 1080 K vs our REFPROP-based 798 K
at the throat.  Replacing it with this REFPROP table tests the property
hypothesis directly.

Run:  python dump_rpa_property_table.py
Output: exports/ethanol95_usr_properties.inp  (paste into RPA usr_properties.inp)
"""
import os
import numpy as np
import CoolProp.CoolProp as CP

from coolant_props import (get_coolant_props, _state, _critical_pressure,
                           _ETHANOL_WATER_MIX)


FLUID    = "Ethanol95"
RPA_NAME = "C2H5OH(L),95%"   # 18-char field, left-justified

# Operating envelope: ethanol channel runs ~25–35 bar, T 290–470 K.
# Bracket the channel state with bundled-file-style pressures, plus extra
# resolution near 3 MPa where the channel actually lives.
P_GRID_MPA = [0.101325, 1.0, 3.0, 5.0, 7.0, 10.0]                 # MPa
# Temperature grid: REFPROP's transport correlations for the ethanol-water
# mixture have Tmin = 273.16 K (water triple point), and at 273 K + low P the
# state is inside the solid region.  Start at 280 K (well above triple point,
# liquid at all pressures) and step every 20 K up to 600 K.  17 points.
# This is narrower than RPA's bundled C2H5OH(L) (253–593 K) but pure ethanol
# stays liquid down to 159 K — the 95/5 mixture cannot.
T_GRID_K   = list(range(280, 601, 20))                            # 280..600 K


def saturation_at_p(P_pa: float):
    """(T_sat [K], latent heat Q_sat [J/kg]) at pressure P.
    For P >= P_crit, return (T_crit, 0.0) — matches the convention in
    RPA's bundled C2H5OH(L) entry for supercritical pressure rows."""
    st = _state(FLUID)
    Pc = _critical_pressure(FLUID)
    if P_pa >= Pc:
        try:
            Tc = float(st.T_critical())
        except Exception:
            Tc = float('nan')
        return Tc, 0.0
    try:
        st.update(CP.PQ_INPUTS, P_pa, 0.0)
        T_sat = float(st.T())
        h_l   = float(st.hmass())
        st.update(CP.PQ_INPUTS, P_pa, 1.0)
        h_v   = float(st.hmass())
        return T_sat, h_v - h_l
    except Exception:
        return float('nan'), float('nan')


def main():
    print(f"Generating RPA properties.inp block for {FLUID}")
    print(f"  Mass fractions: {_ETHANOL_WATER_MIX[FLUID]}  (ethanol, water)")
    Pc = _critical_pressure(FLUID)
    print(f"  P_crit (REFPROP) = {Pc/1e6:.3f} MPa")
    print(f"  P grid: {P_GRID_MPA} MPa")
    print(f"  T grid: {T_GRID_K[0]}–{T_GRID_K[-1]} K every 20 K "
          f"({len(T_GRID_K)} points)\n")

    n_p = len(P_GRID_MPA)
    n_T = len(T_GRID_K)

    # Build (P, T) blocks: pressure outer, temperature inner.
    # REFPROP's mixture EOS occasionally returns nonsense Cp near the
    # supercritical Widom line (e.g. 10 MPa, 540 K → Cp = −2.3×10⁵ kJ/kg-K).
    # We sanity-filter each block: any Cp outside [0.5, 50] kJ/kg-K is
    # replaced by a linear interpolation from the nearest valid neighbours.
    # rho/mu/lambda from the same call are kept (they look fine even when
    # Cp blows up — the glitch is in the energy derivative only).
    def _is_bad_cp(cp_kj):
        return (cp_kj is None) or (cp_kj < 0.5) or (cp_kj > 50.0)

    def _patch_bad_cps(rows, P_mpa):
        cps = [None if r is None else r[2] for r in rows]
        for i, cp in enumerate(cps):
            if not _is_bad_cp(cp):
                continue
            # Find nearest valid neighbours by index
            lo = next((j for j in range(i-1, -1, -1) if not _is_bad_cp(cps[j])), None)
            hi = next((j for j in range(i+1, len(cps)) if not _is_bad_cp(cps[j])), None)
            if lo is not None and hi is not None:
                f = (i - lo) / (hi - lo)
                new_cp = cps[lo] + f * (cps[hi] - cps[lo])
            elif lo is not None:
                new_cp = cps[lo]
            elif hi is not None:
                new_cp = cps[hi]
            else:
                new_cp = float('nan')
            old = rows[i]
            if old is None:
                continue
            rows[i] = (old[0], old[1], new_cp, old[3], old[4], old[5])
            print(f"  PATCH ({P_mpa} MPa, {old[1]:.0f} K): Cp {cp} → {new_cp:.4f} "
                  f"kJ/kg-K (interpolated from neighbours)")
        return rows

    blocks = []
    for P_mpa in P_GRID_MPA:
        P_pa = P_mpa * 1e6
        T_sat, Q_sat = saturation_at_p(P_pa)
        rows = []
        for T in T_GRID_K:
            try:
                cs = get_coolant_props(float(T), P_pa, FLUID)
                rows.append((
                    P_mpa, float(T),
                    cs.Cp / 1000.0,            # kJ/(kg·K)
                    cs.rho,                    # kg/m³
                    cs.viscosity * 1e6,        # µPa·s
                    cs.conductivity * 1000.0,  # mW/(m·K)
                ))
            except Exception as exc:
                print(f"  WARN  ({P_mpa} MPa, {T} K): {exc}")
                rows.append(None)
        rows = _patch_bad_cps(rows, P_mpa)
        blocks.append((P_mpa, T_sat, Q_sat / 1000.0, rows))

    # Write file in real RPA format (matches bundled C2H5OH(L) layout).
    out_dir = os.path.join(os.path.dirname(__file__), 'exports')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ethanol95_usr_properties.inp')
    sep = "!" + "*" * 118 + "\n"

    with open(out_path, 'w') as f:
        f.write("!\n")
        f.write("!                                                 PROPERTIES OF COMPONENTS\n")
        f.write("!Format Version 2\n")
        f.write(sep)
        f.write("!Pressure      Temperature    Cp             Density        "
                "Viscosity      Therm.Cond.    Temp.of Vapor. Heat of Vapor.\n")
        f.write("!MPa           K              kJ/(kg-K)      kg/m^3         "
                "µPa-s          mW/m-K         Ts(p), K       Qs(Ts), kJ/kg \n")
        f.write(sep)
        f.write("!REFPROP-derived 95/5 wt% ethanol-water (mass fractions 0.95, 0.05)\n")
        f.write("!Generated by dump_rpa_property_table.py from coolant_props.py\n")
        f.write(f"!P_crit = {Pc/1e6:.3f} MPa.  Supercritical rows print T_crit with Qs=0.0\n")
        f.write("!Subcritical rows above T_sat are gas-phase roots (matches bundled\n")
        f.write("!C2H5OH(L) which jumps from liquid to vapor across T_sat in-block).\n")
        f.write(sep)
        # Header line: 18-char left-justified species name, then "n_p,n_T"
        f.write(f"{RPA_NAME:<18}{n_p},{n_T}\n")
        for (P_mpa, T_sat, Q_sat_kj, rows) in blocks:
            for i, row in enumerate(rows):
                if row is None:
                    f.write("! *FAILED LOOKUP*\n")
                    continue
                P, T, Cp, rho, mu, lam = row
                line = (f"  {P:10.6f}     {T:6.2f}          "
                        f"{Cp:7.5f}      {rho:9.4f}      "
                        f"{mu:9.4f}      {lam:9.5f}")
                if i == 0:
                    # First row of pressure block carries Ts and Qs
                    line += f"      {T_sat:6.2f}         {Q_sat_kj:8.4f}"
                f.write(line + "\n")
        f.write(sep)

    print(f"Wrote {n_p}×{n_T} = {n_p*n_T} rows → {out_path}\n")
    print("Procedure:")
    print("  1. Locate RPA's resources/usr_properties.inp (typically alongside")
    print("     properties.inp in the RPA install directory).")
    print("  2. If usr_properties.inp doesn't exist yet, copy this file there")
    print("     directly (it has the full Format Version 2 header).")
    print("     If it does exist, append everything from the species-name line")
    print("     onward (omit the !Format Version 2 header on the second-and-")
    print("     subsequent block).")
    print(f"  3. Verify the species name matches RPA's exact label.  This file")
    print(f"     uses '{RPA_NAME}'.  If RPA uses a space ('C2H5OH(L), 95%') or")
    print(f"     different capitalisation, edit the species-name line.")
    print("  4. In RPA: Species Editor → Reload, then re-run the 5kN ethanol")
    print("     case.  Check Run → Show log → Warnings/Errors tabs for any")
    print("     parser complaints.\n")
    print("If T_hw drops from ~1080 K toward our ~798 K, the property hypothesis")
    print("is confirmed and the gap was just stale viscosity data on RPA's side.")
    print()

    # Throat-regime sanity print
    print("Sanity check — throat-regime rows (P near 3 MPa, T near 400 K):")
    print(f"  {'P[MPa]':>8} {'T[K]':>7} {'Cp[kJ/kg-K]':>12} "
          f"{'rho[kg/m³]':>11} {'mu[µPa·s]':>11} {'lam[mW/m-K]':>12}")
    for (_, _, _, rows) in blocks:
        for row in rows:
            if row is None:
                continue
            P, T, Cp, rho, mu, lam = row
            if abs(P - 3.0) < 1e-6 and 373 <= T <= 433:
                print(f"  {P:8.3f} {T:7.1f} {Cp:12.5f} {rho:11.4f} "
                      f"{mu:11.4f} {lam:12.5f}")


if __name__ == '__main__':
    main()
