"""
coolant_props.py
CoolProp / REFPROP wrappers for liquid rocket engine coolants.

Supported coolant strings (config.coolant):
  "RP1"      — Huber 4-component RP-1 surrogate (REFPROP required)
  "Hydrogen" — LH2 (REFPROP preferred; falls back to CoolProp HEOS)
  "Methane"  — LCH4 (REFPROP preferred; falls back to CoolProp HEOS)
  <other>    — passed directly to CoolProp HEOS backend

Configuration:
  Set REFPROP_PATH / REFPROP_LIB below to match your installation.
  REFPROP is configured lazily on first use and cached.
"""

from dataclasses import dataclass
import CoolProp.CoolProp as CP


# -----------------------------------------------------------------------
# REFPROP installation paths — edit these
# -----------------------------------------------------------------------
REFPROP_PATH = '/home/jacob/Documents/REFPROP/'
REFPROP_LIB  = '/home/jacob/Documents/REFPROP-cmake/build/librefprop.so'

# Huber et al. (2010) RP-1 surrogate: MDEC + 5MC9 + n-C12 + MCH
_RP1_FLUID    = "MDEC.FLD&5MC9.FLD&C12.FLD&C7CC6.FLD"
_RP1_MOLEFRAC = [0.354, 0.150, 0.183, 0.313]


# -----------------------------------------------------------------------
# Internal: lazy REFPROP configuration + AbstractState cache
# -----------------------------------------------------------------------
_refprop_ok: bool = False
_state_cache: dict = {}


def _configure_refprop() -> bool:
    """Configure REFPROP backend once. Returns True on success."""
    global _refprop_ok
    if _refprop_ok:
        return True
    try:
        CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, REFPROP_PATH)
        CP.set_config_string(CP.ALTERNATIVE_REFPROP_LIBRARY_PATH, REFPROP_LIB)
        _refprop_ok = True
        return True
    except Exception as exc:
        print(f"[coolant_props] REFPROP config failed: {exc}  "
              f"(Hydrogen and Methane will use CoolProp HEOS; RP1 unavailable)")
        return False


def _state(fluid: str) -> CP.AbstractState:
    """Return a cached AbstractState for the given fluid identifier."""
    if fluid in _state_cache:
        return _state_cache[fluid]

    rp_ok = _configure_refprop()

    if fluid == "RP1":
        if not rp_ok:
            raise RuntimeError(
                "RP1 requires REFPROP. "
                "Check REFPROP_PATH / REFPROP_LIB in coolant_props.py.")
        st = CP.AbstractState("REFPROP", _RP1_FLUID)
        st.set_mole_fractions(_RP1_MOLEFRAC)

    elif fluid in ("Hydrogen", "Methane"):
        backend = "REFPROP" if rp_ok else "HEOS"
        st      = CP.AbstractState(backend, fluid)

    else:
        # Generic CoolProp HEOS backend (handles most pure fluids)
        st = CP.AbstractState("HEOS", fluid)

    _state_cache[fluid] = st
    return st


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------
@dataclass
class CoolantState:
    """Thermodynamic and transport properties at one (T, P) state point."""
    rho:          float   # Density [kg/m³]
    h:            float   # Specific enthalpy [J/kg]
    viscosity:    float   # Dynamic viscosity [Pa·s]
    conductivity: float   # Thermal conductivity [W/(m·K)]
    Cp:           float   # Isobaric specific heat [J/(kg·K)]


def get_coolant_props(T: float, P: float, fluid: str) -> CoolantState:
    """
    Coolant thermodynamic state at temperature T [K] and pressure P [Pa].

    Parameters
    ----------
    T      : temperature [K]
    P      : pressure [Pa]
    fluid  : coolant identifier, e.g. "RP1", "Hydrogen", "Methane"
    """
    st = _state(fluid)
    try:
        st.update(CP.PT_INPUTS, P, T)
        return CoolantState(
            rho          = st.rhomass(),
            h            = st.hmass(),
            viscosity    = st.viscosity(),
            conductivity = st.conductivity(),
            Cp           = st.cpmass(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"CoolProp ({fluid}  T={T:.1f} K  P={P/1e5:.2f} bar): {exc}") from exc


def get_T_from_enthalpy(h: float, P: float, fluid: str) -> float:
    """
    Temperature [K] from specific enthalpy h [J/kg] at pressure P [Pa].
    """
    st = _state(fluid)
    try:
        st.update(CP.HmassP_INPUTS, h, P)
        return float(st.T())
    except Exception as exc:
        raise RuntimeError(
            f"Enthalpy inversion ({fluid}  h={h:.3e} J/kg  P={P/1e5:.2f} bar): {exc}") from exc
