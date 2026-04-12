"""
cea_lookup.py

O/F → chamber property lookup table for Bartz evaluation on an
arbitrary surface-layer mixture (used by the film-cooling model to
capture RPA's boundary-layer composition effect on h_gas).

RPA evaluates Bartz on the evolving surface-layer mixture rather than
the core flow. This module provides a cached sweep of CEA at multiple
O/F values so we can look up (T_c, visc_c, Cp_froz_c, Pr_froz_c, ...)
at any OF_eff in the range without paying the CEA call cost per station.

Usage
-----
    lut = build_of_lut(config)
    props = lut.at(OF_eff=1.62)
    # props is a dict with T_c, visc_c, Cp_froz_c, Pr_froz_c, gamma_c, ...

Cache
-----
Pickled to ~/.cache/RocketThermalAnalysis/cea_lut_{fuel}_{ox}_{Pc_bar:.0f}bar.pkl
Cache key includes (fuel, oxidizer, P_c, OF_range, n_points, exp_ratio).
"""
import os
import pickle
from dataclasses import dataclass, replace
from typing import Dict

import numpy as np
from rocketcea.cea_obj import CEA_Obj

from config import EngineConfig
from cea_interface import _run_single, CEAResult


# Fields we interpolate across OF. Frozen AND equilibrium Cp/Pr are
# included so we can compare Bartz evaluated on either (RPA's behaviour
# with film cooling suggests it uses equilibrium "effective" properties
# on the surface layer, not frozen).
_LUT_FIELDS = (
    "T_c", "visc_c",
    "Cp_froz_c", "Pr_froz_c",   # frozen (what we use today)
    "Cp_c",      "Pr_c",         # equilibrium (what RPA may use in film)
    "gamma_c", "R_specific", "C_star",
)


@dataclass
class CEALookup:
    """
    Precomputed CEA property table vs O/F.

    All arrays share shape (n,) and are ordered by ascending OF_grid.
    """
    OF_grid:    np.ndarray
    T_c:        np.ndarray
    visc_c:     np.ndarray
    Cp_froz_c:  np.ndarray
    Pr_froz_c:  np.ndarray
    Cp_c:       np.ndarray    # equilibrium ("effective")
    Pr_c:       np.ndarray    # equilibrium ("effective")
    gamma_c:    np.ndarray
    R_specific: np.ndarray
    C_star:     np.ndarray

    # Metadata (for cache invalidation & debugging)
    fuel:     str = ""
    oxidizer: str = ""
    P_c:      float = 0.0
    exp_ratio: float = 0.0

    def at(self, OF: float) -> Dict[str, float]:
        """
        Linear-interpolated properties at arbitrary OF.

        Clamped to grid endpoints — callers should ensure OF stays inside
        [OF_grid[0], OF_grid[-1]].
        """
        of = float(np.clip(OF, self.OF_grid[0], self.OF_grid[-1]))
        return {
            field: float(np.interp(of, self.OF_grid, getattr(self, field)))
            for field in _LUT_FIELDS
        }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_dir() -> str:
    d = os.path.expanduser("~/.cache/RocketThermalAnalysis")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(fuel: str, ox: str, P_c: float,
                OF_min: float, OF_max: float, n: int,
                exp_ratio: float) -> str:
    safe_fuel = fuel.replace("/", "_")
    safe_ox   = ox.replace("/", "_")
    return os.path.join(
        _cache_dir(),
        f"cea_lut_{safe_fuel}_{safe_ox}_{P_c/1e5:.0f}bar"
        f"_OF{OF_min:.2f}-{OF_max:.2f}_n{n}_er{exp_ratio:.2f}.pkl",
    )


def _load_cached(path: str) -> "CEALookup | None":
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [cea_lookup] cache read failed ({e}); rebuilding")
        return None


def _save_cached(path: str, lut: CEALookup) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(lut, f)
    except Exception as e:
        print(f"  [cea_lookup] cache write failed: {e}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_of_lut(config: EngineConfig,
                 OF_min: float = 1.0,
                 OF_max: float = 3.0,
                 n: int = 25,
                 verbose: bool = True) -> CEALookup:
    """
    Build (or load from cache) a CEA property table vs OF.

    Parameters
    ----------
    config    : EngineConfig — uses fuel, oxidizer, P_c, exp_ratio
    OF_min/OF_max : sweep range. Default 1.0–3.0 covers fuel-rich surface
                    layer down to ~half of stoichiometric for RP-1/LOX.
    n         : grid points (25 is plenty for smooth interp).
    verbose   : print progress

    Returns
    -------
    CEALookup
    """
    path = _cache_path(config.fuel, config.oxidizer, config.P_c,
                       OF_min, OF_max, n, config.exp_ratio or 0.0)

    cached = _load_cached(path)
    if cached is not None:
        if verbose:
            print(f"  [cea_lookup] loaded cached LUT from {os.path.basename(path)}")
        return cached

    if verbose:
        print(f"  [cea_lookup] building LUT ({n} points, OF {OF_min}-{OF_max}) "
              f"— this takes ~{n*0.5:.0f} s")

    cea = CEA_Obj(fuelName=config.fuel, oxName=config.oxidizer)
    OF_grid = np.linspace(OF_min, OF_max, n)
    arrays = {field: np.zeros(n) for field in _LUT_FIELDS}

    for i, of in enumerate(OF_grid):
        # _run_single takes a config, so we temporarily swap OF
        cfg_i = replace(config, OF=float(of))
        try:
            result = _run_single(cea, cfg_i, float(of))
        except Exception as e:
            raise RuntimeError(
                f"CEA failed at OF={of:.3f} during LUT build: {e}")
        for field in _LUT_FIELDS:
            arrays[field][i] = getattr(result, field)
        if verbose:
            print(f"    OF={of:.3f}  T_c={result.T_c:7.1f}K  "
                  f"μ={result.visc_c*1e5:.3f}e-5  "
                  f"Cp_froz={result.Cp_froz_c:7.1f}  "
                  f"Pr_froz={result.Pr_froz_c:.4f}")

    lut = CEALookup(
        OF_grid=OF_grid,
        **arrays,
        fuel=config.fuel,
        oxidizer=config.oxidizer,
        P_c=config.P_c,
        exp_ratio=config.exp_ratio or 0.0,
    )

    _save_cached(path, lut)
    if verbose:
        print(f"  [cea_lookup] cached to {os.path.basename(path)}")
    return lut
