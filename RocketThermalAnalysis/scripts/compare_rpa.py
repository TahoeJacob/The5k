"""
compare_rpa.py
Side-by-side comparison of our thermal profile vs
References/5kN_22%FilmResults.txt.

RPA is counter-current: its x=0 is injector face (coolant exits here,
hot), its x=353 mm is nozzle exit (coolant enters here, cold).  Our
x-axis uses the same injector-face convention, so x aligns directly.

Prints a table of h_gas, q_conv, T_hw, T_cool at matching axial stations.
"""
# --- run-from-anywhere shim (file lives in subfolder) ---
import os, sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
os.chdir(_PARENT)
# --------------------------------------------------------

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None
plt.savefig = lambda *a, **k: None

import io, sys

# Run main.run() and capture thermal solution
import heat_transfer as ht

_captured = {}
_orig_solve = ht.solve_thermal
def _wrap(*a, **k):
    t = _orig_solve(*a, **k)
    _captured['thermal'] = t
    _captured['flow']   = a[0]
    _captured['geom']   = a[1]
    _captured['cea']    = a[2]
    _captured['chan']   = a[3]
    _captured['config'] = a[4]
    return t
ht.solve_thermal = _wrap
# main.py does `from heat_transfer import solve_thermal` at import time, so
# patch the imported name there too BEFORE main.run() executes.
import main
main.solve_thermal = _wrap

_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    main.run()
except SystemExit:
    pass
sys.stdout = _stdout

t      = _captured['thermal']
geom   = _captured['geom']
cfg    = _captured['config']
cea    = _captured['cea']

# --- CEA property comparison vs RPA header ---
# RPA injector-face values from 5kN_22%FilmResults.txt header
rpa_props = {
    'T_c':       3241.50,    # K
    'mu_c':      9.837e-5,   # Pa·s
    'Cp_froz_c': 2141.7,     # J/kg/K
    'Pr_froz_c': 0.5943,     # -
    'Cp_eq_c':   4231.8,     # J/kg/K  (equilibrium / "effective")
    'Pr_eq_c':   0.3938,     # -
    'gamma_c':   1.1792,
    'C_star':    1779.81,    # m/s
}
print("=" * 72)
print("CEA chamber property comparison — ours (rocketcea) vs RPA header")
print("=" * 72)
def _dev(ours, rpa):
    return 100.0 * (ours - rpa) / rpa if rpa else 0.0
rows = [
    ('T_c      [K]',     cea.T_c,         rpa_props['T_c']),
    ('gamma_c  [-]',     cea.gamma_c,     rpa_props['gamma_c']),
    ('C*       [m/s]',   cea.C_star,      rpa_props['C_star']),
    ('mu_c     [Pa·s]',  cea.visc_c,      rpa_props['mu_c']),
    ('Cp_froz_c [J/kg/K]', cea.Cp_froz_c, rpa_props['Cp_froz_c']),
    ('Pr_froz_c [-]',    cea.Pr_froz_c,   rpa_props['Pr_froz_c']),
]
for label, ours, rpa in rows:
    print(f"  {label:<22}  ours={ours:12.4g}   RPA={rpa:12.4g}   Δ={_dev(ours,rpa):+6.2f}%")

# Bartz property group (Cp / Pr^0.6) × μ^0.2 — the sensitivity knob
def _bartz_prop_group(mu, Cp, Pr):
    return (mu**0.2) * Cp / (Pr**0.6)
our_grp   = _bartz_prop_group(cea.visc_c, cea.Cp_froz_c, cea.Pr_froz_c)
rpa_frz   = _bartz_prop_group(rpa_props['mu_c'], rpa_props['Cp_froz_c'], rpa_props['Pr_froz_c'])
rpa_eq    = _bartz_prop_group(rpa_props['mu_c'], rpa_props['Cp_eq_c'],   rpa_props['Pr_eq_c'])
print()
print(f"Bartz property group μ^0.2 · Cp / Pr^0.6:")
print(f"  ours (froz)   = {our_grp:10.3f}")
print(f"  RPA  (froz)   = {rpa_frz:10.3f}   (Δ vs ours: {_dev(our_grp, rpa_frz):+.1f}%)")
print(f"  RPA  (equil)  = {rpa_eq:10.3f}   (Δ vs ours: {_dev(our_grp, rpa_eq):+.1f}%)")
print("=" * 72)
print()


# --- Load RPA table ---
rpa_path = "/home/jacob/Documents/5kRegenCode/References/5kN_22%FilmResults.txt"
rpa = {'x': [], 'r': [], 'h_gas': [], 'q_conv': [], 'Twg': [], 'Tc': [], 'Pc': []}
with open(rpa_path) as f:
    for line in f:
        if 'regen channel' not in line:
            continue
        parts = line.split('\t')
        if len(parts) < 11:
            continue
        try:
            rpa['x'].append(float(parts[0]) * 1e-3)        # m
            rpa['r'].append(float(parts[1]) * 1e-3)
            rpa['h_gas'].append(float(parts[2]))            # kW/m²K (nan rows skipped later)
            rpa['q_conv'].append(float(parts[3]))           # kW/m²
            rpa['Twg'].append(float(parts[6]))              # K
            rpa['Tc'].append(float(parts[9]))               # K
            rpa['Pc'].append(float(parts[10]))              # MPa
        except ValueError:
            continue

for k in rpa:
    rpa[k] = np.array(rpa[k])

# RPA's throat station: argmin of r
i_rthr = int(np.argmin(rpa['r']))
x_rthr = rpa['x'][i_rthr]

# Our throat station
our_r = np.array([ (np.pi * cfg.dx)**0  # placeholder
                   for _ in t.x])
# Actually use geom.L_c as x_throat
x_othr = geom.L_c

# Build comparison stations (by fraction of engine length)
L_rpa = rpa['x'][-1]  # should match ~Lc + Le
L_our = t.x[-1]
print(f"RPA total L = {L_rpa*1000:.1f} mm,  ours = {L_our*1000:.1f} mm")
print(f"RPA throat at x = {x_rthr*1000:.1f} mm,  ours at x = {x_othr*1000:.1f} mm")
print()

# Choose comparison points: injector, 4 chamber stations, throat, 3 nozzle stations
frac_list = [0.0, 0.1, 0.2, 0.3, 0.5]   # chamber
our_throat_frac = x_othr / L_our
rpa_throat_frac = x_rthr / L_rpa
stations = []
for f in frac_list:
    stations.append(("Cham", f, f))
stations.append(("THR ", our_throat_frac, rpa_throat_frac))
# Nozzle stations: interp between throat_frac and 1.0
for f in [0.25, 0.5, 0.75, 1.0]:
    our_f = our_throat_frac + (1.0 - our_throat_frac) * f
    rpa_f = rpa_throat_frac + (1.0 - rpa_throat_frac) * f
    stations.append(("Nzl ", our_f, rpa_f))

# --- Table ---
def sample(arr_x, arr_y, frac):
    i = int(round(frac * (len(arr_x) - 1)))
    return arr_y[i]

def sample_at(arr_x, arr_y, x_target):
    return float(np.interp(x_target, arr_x, arr_y))

print(f"{'zone':<5} {'frac':>5} | {'x_our':>7} {'x_rpa':>7} | "
      f"{'h_our':>8} {'h_rpa':>8}  {'h_dev':>6} | "
      f"{'q_our':>8} {'q_rpa':>8}  {'q_dev':>6} | "
      f"{'Twg_our':>8} {'Twg_rpa':>8}  {'dTwg':>6} | "
      f"{'Tc_our':>7} {'Tc_rpa':>7}")
print("-" * 145)

for lbl, fo, fr in stations:
    xo = fo * L_our
    xr = fr * L_rpa
    h_our = sample_at(t.x, t.h_gas, xo) / 1000.0      # kW/m²K
    h_rpa = sample_at(rpa['x'], rpa['h_gas'], xr)     # already kW/m²K
    q_our = sample_at(t.x, t.heatflux, xo) / 1000.0   # kW/m²
    q_rpa = sample_at(rpa['x'], rpa['q_conv'], xr)    # already kW/m²
    Two   = sample_at(t.x, t.T_hw, xo)
    Twr   = sample_at(rpa['x'], rpa['Twg'], xr)
    Tco   = sample_at(t.x, t.T_coolant, xo)
    Tcr   = sample_at(rpa['x'], rpa['Tc'], xr)
    h_dev = 100*(h_our - h_rpa)/h_rpa if h_rpa > 0 else 0
    q_dev = 100*(q_our - q_rpa)/q_rpa if q_rpa > 0 else 0
    print(f"{lbl:<5} {fo:5.2f} | {xo*1000:7.1f} {xr*1000:7.1f} | "
          f"{h_our:8.3f} {h_rpa:8.3f}  {h_dev:+5.1f}% | "
          f"{q_our:8.1f} {q_rpa:8.1f}  {q_dev:+5.1f}% | "
          f"{Two:8.1f} {Twr:8.1f}  {Two-Twr:+5.1f} | "
          f"{Tco:7.1f} {Tcr:7.1f}")

# --- Integrated q_gas ---
# Our: heatflux [W/m²] × wetted area per station
# Wetted length per axial dx ≈ 2π·r(x)·dx
from geometry import nozzle_radius
dx = cfg.dx
r_our = np.array([nozzle_radius(float(x), geom, dx) for x in t.x])
Q_our = np.trapezoid(t.heatflux * 2 * np.pi * r_our, t.x) / 1e3  # kW

# RPA: integrate along its x
Q_rpa = np.trapezoid(rpa['q_conv'] * 1e3 * 2 * np.pi * rpa['r'], rpa['x']) / 1e3  # kW

print()
print(f"Integrated convective heat into coolant:")
print(f"  Ours = {Q_our:.1f} kW")
print(f"  RPA  = {Q_rpa:.1f} kW   (Δ = {100*(Q_our-Q_rpa)/Q_rpa:+.1f} %)")

# --- Make plots ---
fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
axes[0].plot(t.x*1000, t.h_gas/1000, label='Ours', lw=1.5)
axes[0].plot(rpa['x']*1000, rpa['h_gas'], label='RPA', lw=1.2)
axes[0].set_ylabel('h_gas [kW/m²K]'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(t.x*1000, t.heatflux/1e6, label='Ours', lw=1.5)
axes[1].plot(rpa['x']*1000, rpa['q_conv']/1e3, label='RPA', lw=1.2)
axes[1].set_ylabel('q_conv [MW/m²]'); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(t.x*1000, t.T_hw, label='Ours T_hw', lw=1.5)
axes[2].plot(rpa['x']*1000, rpa['Twg'], label='RPA Twg', lw=1.2)
axes[2].set_ylabel('T_hw [K]'); axes[2].legend(); axes[2].grid(alpha=0.3)

axes[3].plot(t.x*1000, t.T_coolant, label='Ours T_cool', lw=1.5)
axes[3].plot(rpa['x']*1000, rpa['Tc'], label='RPA T_c', lw=1.2)
axes[3].set_ylabel('T_coolant [K]'); axes[3].set_xlabel('x [mm, injector→exit]')
axes[3].legend(); axes[3].grid(alpha=0.3)

fig.suptitle('Our code vs RPA — 5kN RP-1/LOX, 22% film', fontsize=12)
plt.tight_layout()
import matplotlib.pyplot as _plt  # restore
_plt.savefig.__wrapped__ if hasattr(_plt.savefig, '__wrapped__') else None
# Force-save (override monkeypatch)
plt.savefig = matplotlib.pyplot.Figure.savefig.__get__(fig) if False else None
fig.savefig("exports/compare_rpa.png", dpi=120)
print("\nSaved plot: exports/compare_rpa.png")
