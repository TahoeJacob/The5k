"""
materials.py
Temperature-dependent mechanical properties for chamber-wall materials.

Each Material carries linear-interpolation tables for:
  E(T)        Young's modulus         [Pa]
  alpha(T)    CTE                     [1/K]
  sigma_y(T)  0.2% proof yield        [Pa]
  nu          Poisson's ratio (≈ const over service range)

Tables are digitized from published datasheets and are accurate to
engineering-first-pass resolution (~5%).

References
----------
CuCrZr:  Ellis D.L., NASA/TM-2005-213961 "GRCop-84 vs CuCrZr"
         NIST Monograph 177 (cryogenic → 300 K)
         CERN EDMS 1334498 (LHC collimator jaws, 300–800 K)
Inconel 718: Special Metals INCO Alloy 718 datasheet (rev. 2007)
GRCop-84: Ellis D.L., NASA/TM-2005-213566
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Material:
    name: str
    T_tab:       np.ndarray     # [K]
    E_tab:       np.ndarray     # [Pa]
    alpha_tab:   np.ndarray     # [1/K]
    sigma_y_tab: np.ndarray     # [Pa]
    nu:          float          # — (constant over service range)

    def E(self, T):
        return float(np.interp(T, self.T_tab, self.E_tab))

    def alpha(self, T):
        return float(np.interp(T, self.T_tab, self.alpha_tab))

    def sigma_y(self, T):
        return float(np.interp(T, self.T_tab, self.sigma_y_tab))

    def lame(self, T):
        """Lamé parameters (λ, μ) at temperature T."""
        E = self.E(T)
        nu = self.nu
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu  = E / (2.0 * (1.0 + nu))
        return lam, mu

    def beta(self, T):
        """Thermal stress coefficient β = E·α / (1 − 2ν) = (3λ+2μ)·α."""
        return self.E(T) * self.alpha(T) / (1.0 - 2.0 * self.nu)


# ---------------------------------------------------------------------------
# CuCrZr (Cu–0.8Cr–0.1Zr, precipitation hardened)
#   NIST + CERN EDMS 1334498 composite
# ---------------------------------------------------------------------------
_CuCrZr = Material(
    name    = "CuCrZr",
    T_tab   = np.array([293., 373., 473., 573., 673., 773., 873., 973.]),
    E_tab   = np.array([128., 125., 121., 116., 110., 103.,  95.,  86.]) * 1e9,
    alpha_tab = np.array([16.7, 17.0, 17.4, 17.9, 18.4, 18.9, 19.4, 19.9]) * 1e-6,
    sigma_y_tab = np.array([350., 330., 300., 260., 210., 150.,  90.,  50.]) * 1e6,
    nu      = 0.33,
)


# ---------------------------------------------------------------------------
# Inconel 718 (aged, AMS 5662)
#   Special Metals datasheet (rev. 2007)
# ---------------------------------------------------------------------------
_Inconel718 = Material(
    name    = "Inconel718",
    T_tab   = np.array([293., 473., 673., 873., 1073.]),
    E_tab   = np.array([205., 194., 181., 164., 146.]) * 1e9,
    alpha_tab = np.array([13.0, 13.8, 14.6, 15.4, 16.2]) * 1e-6,
    sigma_y_tab = np.array([1100., 1050., 1010.,  950.,  830.]) * 1e6,
    nu      = 0.294,
)


# ---------------------------------------------------------------------------
# GRCop-84 (Cu–8Cr–4Nb)
#   Ellis D.L., NASA/TM-2005-213566
# ---------------------------------------------------------------------------
_GRCop84 = Material(
    name    = "GRCop-84",
    T_tab   = np.array([293., 473., 673., 873., 1073.]),
    E_tab   = np.array([121., 115., 107.,  97.,  85.]) * 1e9,
    alpha_tab = np.array([16.5, 17.2, 17.9, 18.7, 19.5]) * 1e-6,
    sigma_y_tab = np.array([186., 176., 162., 138., 100.]) * 1e6,
    nu      = 0.33,
)


MATERIALS = {
    "CuCrZr":     _CuCrZr,
    "Inconel718": _Inconel718,
    "GRCop-84":   _GRCop84,
}


def get_material(name: str) -> Material:
    if name not in MATERIALS:
        raise KeyError(
            f"Unknown material '{name}'. Available: {sorted(MATERIALS)}")
    return MATERIALS[name]
