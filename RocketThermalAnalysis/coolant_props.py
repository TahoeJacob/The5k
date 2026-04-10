"""
coolant_props.py
CoolProp / REFPROP wrappers for liquid rocket engine coolants.

Supported coolant strings (config.coolant):
  "RP1"      — Huber 4-component RP-1 surrogate (REFPROP required)
  "Hydrogen" — LH2 (REFPROP preferred; falls back to CoolProp HEOS)
  "Methane"  — LCH4 (REFPROP preferred; falls back to CoolProp HEOS)
  "Ethanol"  — Ethanol (REFPROP preferred; falls back to CoolProp HEOS)
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

# Ethanol-water mixtures (matches the property tables RPA exposes:
# 95%, 90%, 70%, 50%, 40% ethanol by MASS).  REFPROP takes mass or mole
# fractions; we store mass fractions and convert in _state() so the names
# read naturally ("Ethanol95" = 95 wt% ethanol, 5 wt% water).
_ETHANOL_WATER_FLUID = "ETHANOL&WATER"
_ETHANOL_WATER_MIX = {
    "Ethanol95": (0.95, 0.05),
    "Ethanol90": (0.90, 0.10),
    "Ethanol70": (0.70, 0.30),
    "Ethanol50": (0.50, 0.50),
    "Ethanol40": (0.40, 0.60),
}


# -----------------------------------------------------------------------
# Internal: lazy REFPROP configuration + AbstractState cache
# -----------------------------------------------------------------------
_refprop_ok: bool = False
_state_cache: dict = {}
_pcrit_cache: dict = {}   # fluid str -> P_crit [Pa]

# One-shot warnings: each (fluid, regime) string is only printed the first time.
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[coolant_props] {msg}")


def _critical_pressure(fluid: str) -> float:
    """Critical pressure [Pa] for the given fluid (cached by fluid name)."""
    if fluid in _pcrit_cache:
        return _pcrit_cache[fluid]
    try:
        pc = float(_state(fluid).p_critical())
    except Exception:
        pc = float("inf")
    _pcrit_cache[fluid] = pc
    return pc


def _T_sat(st: CP.AbstractState, P: float) -> float:
    """Saturation temperature [K] at pressure P [Pa] using a fresh QP update."""
    st.update(CP.PQ_INPUTS, P, 0.0)
    return float(st.T())


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

    elif fluid in ("Hydrogen", "Methane", "Ethanol"):
        backend = "REFPROP" if rp_ok else "HEOS"
        st      = CP.AbstractState(backend, fluid)

    elif fluid in _ETHANOL_WATER_MIX:
        if not rp_ok:
            raise RuntimeError(
                f"{fluid} requires REFPROP. "
                f"Check REFPROP_PATH / REFPROP_LIB in coolant_props.py.")
        st = CP.AbstractState("REFPROP", _ETHANOL_WATER_FLUID)
        st.set_mass_fractions(list(_ETHANOL_WATER_MIX[fluid]))

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

    Two-phase handling
    ------------------
    Our regen-flow model is single-phase: the bulk fluid is always either
    fully liquid or fully gas, never a wet mixture.  When P < P_crit, REFPROP's
    PT_INPUTS call can return junk inside the saturation dome (or pick the
    wrong root near saturation).  We therefore:

      * P >= P_crit  → supercritical, pure PT_INPUTS is unambiguous.
      * P <  P_crit, T < T_sat(P)  → subcooled liquid, force liquid root via
                                     QT_INPUTS(Q=0) then update to T.
      * P <  P_crit, T > T_sat(P)  → superheated vapor, force gas root via
                                     QT_INPUTS(Q=1) then update to T.  Print a
                                     one-shot warning the first time this
                                     happens for the fluid (it almost always
                                     means the channel has flashed; the
                                     downstream Chen correlation should be
                                     handling that, but the bulk treatment
                                     here at least returns sane gas props).

    Parameters
    ----------
    T      : temperature [K]
    P      : pressure [Pa]
    fluid  : coolant identifier, e.g. "RP1", "Hydrogen", "Methane", "Ethanol"
    """
    st = _state(fluid)
    Pc = _critical_pressure(fluid)

    try:
        if P >= Pc:
            # Supercritical — PT_INPUTS is single-valued.
            st.update(CP.PT_INPUTS, P, T)
        else:
            T_sat = _T_sat(st, P)
            if T < T_sat - 1e-3:
                # Subcooled liquid: anchor at the liquid saturation root, then
                # walk in to (P, T).  Some fluids let PT_INPUTS get this wrong
                # near saturation; the explicit anchor avoids it.
                st.update(CP.PT_INPUTS, P, T)
            elif T > T_sat + 1e-3:
                # Superheated vapor: same idea, anchor at the gas root.
                _warn_once(
                    f"{fluid}_flash",
                    f"{fluid}: bulk T={T:.1f} K exceeds T_sat={T_sat:.1f} K "
                    f"at P={P/1e5:.2f} bar — forcing gas-phase root. "
                    f"This means the coolant has flashed; enable two-phase "
                    f"(Chen) HTC if you have not already.")
                # Force gas root then update to (P, T)
                st.update(CP.PQ_INPUTS, P, 1.0)
                st.update(CP.PT_INPUTS, P, T)
            else:
                # T essentially at saturation: pick the liquid root by default
                # (regen flow is normally subcooled even when grazing T_sat).
                st.update(CP.PQ_INPUTS, P, 0.0)

        # REFPROP has no transport equations for alcohol-water mixtures
        # (TRNPRP error 542).  Use mixture thermo (rho, h, cp) but compute
        # mu and k from the pure components at the same (P, T) and combine
        # by mass fraction — first-order but adequate for 95/5-type splits
        # where one component dominates.
        if fluid in _ETHANOL_WATER_MIX:
            rho = st.rhomass()
            h_  = st.hmass()
            cp  = st.cpmass()
            mu, k_ = _mix_transport_pure(fluid, T, P)
            return CoolantState(rho=rho, h=h_, viscosity=mu,
                                conductivity=k_, Cp=cp)

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


def _mix_transport_pure(fluid: str, T: float, P: float) -> tuple:
    """
    Compute viscosity and thermal conductivity for an ethanol-water mixture
    by mass-weighting the pure-component values from REFPROP.

    REFPROP raises TRNPRP error 542 for transport on mixtures containing
    alcohols, so we have to fall back to a mixing rule.  We use a simple
    mass-fraction average; for the high-ethanol blends we care about (≥70%
    ethanol) the result is dominated by ethanol's properties anyway and
    the residual error from a more sophisticated rule (Wilke, etc.) is
    small compared to the discrepancy we're chasing with RPA.
    """
    w_eth, w_h2o = _ETHANOL_WATER_MIX[fluid]
    st_eth = _state("Ethanol")
    st_h2o = _pure_water_state()

    def _props_at(st, T, P):
        # Same liquid/gas-root logic as get_coolant_props, condensed.
        Pc = float(st.p_critical()) if hasattr(st, "p_critical") else float("inf")
        if P >= Pc:
            st.update(CP.PT_INPUTS, P, T)
        else:
            st.update(CP.PQ_INPUTS, P, 0.0)
            T_sat = float(st.T())
            if T < T_sat - 1e-3:
                st.update(CP.PT_INPUTS, P, T)
            elif T > T_sat + 1e-3:
                st.update(CP.PQ_INPUTS, P, 1.0)
                st.update(CP.PT_INPUTS, P, T)
        return st.viscosity(), st.conductivity()

    mu_e, k_e = _props_at(st_eth, T, P)
    mu_w, k_w = _props_at(st_h2o, T, P)
    mu = w_eth * mu_e + w_h2o * mu_w
    k_ = w_eth * k_e + w_h2o * k_w
    return mu, k_


def _pure_water_state() -> CP.AbstractState:
    """Cached pure-water AbstractState (REFPROP if available, else HEOS)."""
    if "Water" in _state_cache:
        return _state_cache["Water"]
    rp_ok = _configure_refprop()
    backend = "REFPROP" if rp_ok else "HEOS"
    st = CP.AbstractState(backend, "Water")
    _state_cache["Water"] = st
    return st


def get_T_from_enthalpy(h: float, P: float, fluid: str) -> float:
    """
    Temperature [K] from specific enthalpy h [J/kg] at pressure P [Pa].

    Subcritical case: if h falls inside the two-phase dome (h_l_sat < h <
    h_v_sat), we have no single-phase temperature.  Our model treats this as
    "fully evaporated at the same enthalpy" by clamping the lookup to the
    gas-side saturation root and continuing — this is consistent with the
    gas-root fallback in get_coolant_props.  A one-shot warning is printed.
    """
    st = _state(fluid)
    Pc = _critical_pressure(fluid)
    try:
        if P < Pc:
            st.update(CP.PQ_INPUTS, P, 0.0)
            h_l = st.hmass()
            st.update(CP.PQ_INPUTS, P, 1.0)
            h_v = st.hmass()
            if h_l <= h <= h_v:
                _warn_once(
                    f"{st.fluid_names()[0] if hasattr(st,'fluid_names') else 'fluid'}_h_dome",
                    f"Coolant enthalpy h={h:.3e} J/kg lies inside the two-phase "
                    f"dome at P={P/1e5:.2f} bar (h_l={h_l:.3e}, h_v={h_v:.3e}). "
                    f"Returning T_sat; enable two-phase HTC for accuracy.")
                return float(st.T())
            if h > h_v:
                # Superheated: gas root then HmassP
                st.update(CP.PQ_INPUTS, P, 1.0)
        st.update(CP.HmassP_INPUTS, h, P)
        return float(st.T())
    except Exception as exc:
        raise RuntimeError(
            f"Enthalpy inversion ({fluid}  h={h:.3e} J/kg  P={P/1e5:.2f} bar): {exc}") from exc
