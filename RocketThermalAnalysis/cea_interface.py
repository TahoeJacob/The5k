"""
cea_interface.py
Wraps rocketcea to extract thermodynamic data and run O/F sweeps.

Entry point: get_cea_for_analysis(config)
  - No OF and no OF_sweep  → raises ValueError
  - OF_sweep only          → plots sweep, returns None  (caller should stop)
  - OF only or both        → (optionally plots sweep), returns CEAResult
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from rocketcea.cea_obj import CEA_Obj

from config import EngineConfig

# Unit conversions
_PA_TO_PSIA = 0.000145038
_PA_TO_BAR  = 1e-5
Ru          = 8.31446261815324  # J / (mol·K)


# -----------------------------------------------------------------------
# Output dataclass
# -----------------------------------------------------------------------
@dataclass
class CEAResult:
    """Thermodynamic outputs from CEA at a single O/F point."""

    OF: float

    # Chamber conditions
    P_c:   float   # [Pa]
    T_c:   float   # [K]
    rho_c: float   # [kg/m³]
    H_c:   float   # [J/kg]

    # Throat conditions
    P_t:   float   # [Pa]
    T_t:   float   # [K]
    rho_t: float   # [kg/m³]
    H_t:   float   # [J/kg]

    # Isentropic exponents
    gamma_c: float
    gamma_t: float

    # Equilibrium transport properties (chamber / throat)
    visc_c:     float   # [Pa·s]
    visc_t:     float   # [Pa·s]
    Cp_c:       float   # [J/(kg·K)] — equilibrium
    Cp_t:       float   # [J/(kg·K)]
    cond_c:     float   # [W/(m·K)]
    cond_t:     float   # [W/(m·K)]
    Pr_c:       float
    Pr_t:       float

    # Frozen transport properties (chamber / throat)
    Cp_froz_c:   float  # [J/(kg·K)]
    Cp_froz_t:   float
    cond_froz_c: float  # [W/(m·K)]
    cond_froz_t: float
    Pr_froz_c:   float
    Pr_froz_t:   float

    # Performance
    C_star:    float   # [m/s]
    CF:        float   # Vacuum thrust coefficient
    Isp_vac_e: float   # Vacuum Isp at nozzle exit [m/s]
    Isp_sl_e:  float   # Sea-level Isp at nozzle exit [m/s]

    # Gas constants
    molar_mass:      float   # [kg/mol]
    R_specific:      float   # [J/(kg·K)] = Ru / molar_mass

    # Combustion products — list of [species, mass_fraction] pairs
    # e.g. [['H2O', '0.5500'], ['CO2', '0.1200'], ...]
    mass_fractions: List = field(default_factory=list)


# -----------------------------------------------------------------------
# Internal: parse full CEA output text
# -----------------------------------------------------------------------
def _fix_sci(s: str) -> str:
    """Fix missing 'e' in CEA scientific notation e.g. '1.23-4' → '1.23e-4'."""
    return re.sub(r'([0-9])\-([0-9])', r'\1e-\2', s)


def _parse_cea_output(full_output: str, OF: float,
                      molar_mass_g_mol: float) -> CEAResult:
    """
    Parse the text block returned by cea.get_full_cea_output().
    Raises ValueError if any expected field is missing.
    """
    lines = full_output.splitlines()

    data: Dict = {}
    i = 0
    while i < len(lines):
        ln = lines[i]

        if ln.startswith(' P, BAR'):
            parts = ln.split()
            data['P_c'] = float(parts[2]) * 1e5
            data['P_t'] = float(parts[3]) * 1e5

        elif ln.startswith(' T, K'):
            parts = ln.split()
            data['T_c'] = float(parts[2])
            data['T_t'] = float(parts[3])

        elif ln.startswith(' RHO, KG/CU M'):
            parts = ln.split()
            data['rho_c'] = float(_fix_sci(parts[3]))
            data['rho_t'] = float(_fix_sci(parts[5]))

        elif ln.startswith(' H, KJ/KG'):
            parts = ln.split()
            data['H_c'] = float(parts[2]) * 1000.0
            data['H_t'] = float(parts[3]) * 1000.0

        elif ln.startswith(' GAMMAs'):
            parts = ln.split()
            data['gamma_c'] = float(parts[1])
            data['gamma_t'] = float(parts[2])

        elif ln.startswith(' VISC,MILLIPOISE'):
            parts = ln.split()
            data['visc_c'] = float(parts[1]) * 1e-4   # millipoise → Pa·s
            data['visc_t'] = float(parts[2]) * 1e-4

        elif ln.startswith('  WITH EQUILIBRIUM'):
            block = lines[i + 1: i + 6]
            data['Cp_c']    = float(block[1].split()[2]) * 1000.0
            data['Cp_t']    = float(block[1].split()[3]) * 1000.0
            data['cond_c']  = float(block[2].split()[1]) * 0.1    # mW/(cm·K) → W/(m·K)
            data['cond_t']  = float(block[2].split()[2]) * 0.1
            data['Pr_c']    = float(block[3].split()[2])
            data['Pr_t']    = float(block[3].split()[3])

        elif ln.startswith('  WITH FROZEN'):
            block = lines[i + 1: i + 6]
            data['Cp_froz_c']   = float(block[1].split()[2]) * 1000.0
            data['Cp_froz_t']   = float(block[1].split()[3]) * 1000.0
            data['cond_froz_c'] = float(block[2].split()[1]) * 0.1  # mW/(cm·K) → W/(m·K)
            data['cond_froz_t'] = float(block[2].split()[2]) * 0.1
            data['Pr_froz_c']   = float(block[3].split()[2])
            data['Pr_froz_t']   = float(block[3].split()[3])

        elif ln.startswith(' CSTAR'):
            data['C_star'] = float(ln.split()[2])

        elif ln.startswith(' CF'):
            data['CF'] = float(ln.split()[1])

        elif ln.startswith(' Ivac'):
            parts = ln.split()
            data['Isp_vac_e'] = float(parts[3])   # exit column

        elif ln.startswith(' Isp'):
            parts = ln.split()
            data['Isp_sl_e'] = float(parts[3])    # exit column

        elif ln.startswith(' MASS FRACTIONS'):
            fracs = []
            j = i + 1
            while j < len(lines):
                if lines[j].startswith('  * THERMODYNAMIC'):
                    break
                row = lines[j].split()
                if row:   # skip blank lines inside the block
                    fracs.append(row)
                j += 1
            data['mass_fractions'] = fracs

        i += 1

    required = ['P_c', 'T_c', 'rho_c', 'H_c', 'P_t', 'T_t', 'rho_t', 'H_t',
                'gamma_c', 'gamma_t', 'visc_c', 'visc_t',
                'Cp_c', 'Cp_t', 'cond_c', 'cond_t', 'Pr_c', 'Pr_t',
                'Cp_froz_c', 'Cp_froz_t', 'Pr_froz_c', 'Pr_froz_t',
                'C_star', 'CF', 'Isp_vac_e', 'Isp_sl_e', 'mass_fractions']
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"CEA parse failed — missing fields: {missing}")

    molar_mass_kg = molar_mass_g_mol / 1000.0
    return CEAResult(
        OF=OF,
        P_c=data['P_c'], T_c=data['T_c'], rho_c=data['rho_c'], H_c=data['H_c'],
        P_t=data['P_t'], T_t=data['T_t'], rho_t=data['rho_t'], H_t=data['H_t'],
        gamma_c=data['gamma_c'], gamma_t=data['gamma_t'],
        visc_c=data['visc_c'],   visc_t=data['visc_t'],
        Cp_c=data['Cp_c'],       Cp_t=data['Cp_t'],
        cond_c=data['cond_c'],   cond_t=data['cond_t'],
        Pr_c=data['Pr_c'],       Pr_t=data['Pr_t'],
        Cp_froz_c=data['Cp_froz_c'], Cp_froz_t=data['Cp_froz_t'],
        cond_froz_c=data.get('cond_froz_c', data['cond_c']),
        cond_froz_t=data.get('cond_froz_t', data['cond_t']),
        Pr_froz_c=data['Pr_froz_c'], Pr_froz_t=data['Pr_froz_t'],
        C_star=data['C_star'], CF=data['CF'],
        Isp_vac_e=data['Isp_vac_e'], Isp_sl_e=data['Isp_sl_e'],
        molar_mass=molar_mass_kg,
        R_specific=Ru / molar_mass_kg,
        mass_fractions=data['mass_fractions'],
    )


# -----------------------------------------------------------------------
# Internal: single CEA call
# -----------------------------------------------------------------------
def _run_single(cea: CEA_Obj, config: EngineConfig, OF: float) -> CEAResult:
    Pc_bar  = config.P_c * _PA_TO_BAR
    Pc_psia = config.P_c * _PA_TO_PSIA
    er      = config.exp_ratio if config.exp_ratio else 5.0

    molar_mass_g_mol, _ = cea.get_Chamber_MolWt_gamma(
        Pc=Pc_psia, MR=OF, eps=er)

    full = cea.get_full_cea_output(
        Pc=Pc_bar, MR=OF, eps=er,
        short_output=1, pc_units='bar',
        show_mass_frac=1, output='siunits')

    return _parse_cea_output(full, OF, molar_mass_g_mol)


# -----------------------------------------------------------------------
# Internal: O/F sweep plot
# -----------------------------------------------------------------------
def _plot_sweep(sweep_results: dict, config: EngineConfig,
                highlight_OF: Optional[float] = None):
    OF_vals    = sorted(sweep_results.keys())
    T_c_vals   = [sweep_results[o].T_c      for o in OF_vals]
    Cstar_vals = [sweep_results[o].C_star    for o in OF_vals]
    Isp_vals   = [sweep_results[o].Isp_vac_e for o in OF_vals]
    gamma_vals = [sweep_results[o].gamma_c   for o in OF_vals]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f'{config.fuel} / {config.oxidizer}  '
        f'Pc = {config.P_c/1e5:.1f} bar  '
        f'{"Frozen" if config.frozen else "Equilibrium"}',
        fontsize=13)

    plots = [
        (axes[0, 0], T_c_vals,   'T_c [K]',     'Chamber Temperature'),
        (axes[0, 1], Cstar_vals, 'C* [m/s]',    'Characteristic Velocity'),
        (axes[1, 0], Isp_vals,   'Isp_vac [s]', 'Vacuum Specific Impulse'),
        (axes[1, 1], gamma_vals, 'γ [-]',        'Specific Heat Ratio (chamber)'),
    ]
    for ax, y, ylabel, title in plots:
        ax.plot(OF_vals, y, 'b-o', markersize=3)
        if highlight_OF is not None:
            # Mark the selected design point
            closest = min(OF_vals, key=lambda o: abs(o - highlight_OF))
            idx = OF_vals.index(closest)
            ax.axvline(highlight_OF, color='r', linestyle='--', linewidth=1.0,
                       label=f'O/F = {highlight_OF}')
            ax.plot(closest, y[idx], 'ro', markersize=7)
            ax.legend(fontsize=8)
        ax.set_xlabel('O/F ratio')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show(block=False)


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------
def get_cea_for_analysis(config: EngineConfig) -> Optional[CEAResult]:
    """
    Main entry point.

    Returns
    -------
    CEAResult  if a single O/F design point is defined (with or without sweep)
    None       if only an O/F sweep was requested (caller should stop here)

    Raises
    ------
    ValueError if neither OF nor OF_sweep is provided, or if exp_ratio is
               missing when thermal analysis will proceed.
    """
    if config.OF is None and config.OF_sweep is None:
        raise ValueError(
            "No O/F ratio defined.  Set config.OF for a single design point, "
            "config.OF_sweep = (start, stop, step) for a sweep, or both.")

    cea = CEA_Obj(fuelName=config.fuel, oxName=config.oxidizer)

    # ---- Run sweep if requested ----
    sweep_results = None
    if config.OF_sweep is not None:
        start, stop, step = config.OF_sweep
        OF_range = np.arange(start, stop + step * 0.5, step)
        print(f"Running O/F sweep {start} → {stop} ({len(OF_range)} points) ...")
        sweep_results = {}
        for of in OF_range:
            try:
                sweep_results[round(of, 4)] = _run_single(cea, config, of)
            except Exception as e:
                print(f"  Warning: CEA failed at O/F={of:.4f}: {e}")

        _plot_sweep(sweep_results, config, highlight_OF=config.OF)
        print("Sweep complete. Close plot window to continue (or inspect results).")

    # ---- If no single O/F, stop here ----
    if config.OF is None:
        plt.show()
        return None

    # ---- Validate that geometry info is present for analysis ----
    if config.exp_ratio is None:
        raise ValueError(
            "config.exp_ratio (Ae/At) must be set to proceed with thermal analysis.")

    # ---- Single design-point run ----
    print(f"\nRunning CEA at O/F = {config.OF} ...")
    result = _run_single(cea, config, config.OF)

    print(f"  T_c    = {result.T_c:.1f} K")
    print(f"  P_c    = {result.P_c/1e5:.2f} bar")
    print(f"  C*     = {result.C_star:.1f} m/s")
    print(f"  Isp_v  = {result.Isp_vac_e:.1f} m/s  ({result.Isp_vac_e/9.81:.1f} s)")
    print(f"  γ_c    = {result.gamma_c:.4f}")
    print(f"  M_mol  = {result.molar_mass*1000:.2f} g/mol")
    print(f"  R_sp   = {result.R_specific:.2f} J/(kg·K)")

    if config.OF_sweep is not None:
        plt.show()  # Block until user closes sweep plots before continuing

    return result
