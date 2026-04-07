"""
config.py
User-defined engine design parameters.
Fill in this file for each new engine — no other file should need editing for a new design.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EngineConfig:
    # -----------------------------------------------------------------------
    # Propellants
    # -----------------------------------------------------------------------
    fuel: str       # rocketcea fuel name  e.g. "RP-1", "LH2", "CH4"
    oxidizer: str   # rocketcea oxidizer   e.g. "LOX"
    coolant: str    # CoolProp fluid name  e.g. "RP1", "Hydrogen", "Methane"

    # -----------------------------------------------------------------------
    # Performance requirements
    # -----------------------------------------------------------------------
    P_c: float      # Chamber pressure [Pa]
    F_vac: float    # Target vacuum thrust [N]

    # -----------------------------------------------------------------------
    # O/F ratio — at least one must be provided
    #   OF only        → run thermal analysis at that point
    #   OF_sweep only  → plot sweep data, then stop (no thermal analysis)
    #   Both           → plot sweep, then run thermal analysis at OF
    # -----------------------------------------------------------------------
    OF: Optional[float] = None
    OF_sweep: Optional[Tuple[float, float, float]] = None  # (start, stop, step)

    # -----------------------------------------------------------------------
    # CEA settings
    # -----------------------------------------------------------------------
    frozen: bool = False    # False = equilibrium (recommended); True = frozen

    # -----------------------------------------------------------------------
    # Engine geometry
    # -----------------------------------------------------------------------
    exp_ratio: Optional[float] = None  # Ae/At — required for thermal analysis
    cont_ratio: float = 6.0            # Ac/At contraction ratio
    L_star: float = 1.143              # Characteristic chamber length [m]

    # Nozzle contour angles [degrees]
    theta1: float = 30.0    # Convergence half-angle
    thetaD: float = 30.0    # Initial divergence half-angle (bell nozzle)
    thetaE: float = 12.0    # Nozzle exit half-angle
    # Throat radius of curvature multipliers (as fraction of throat radius)
    R1_mult: float = 1.5    # Inner convergence fillet
    RU_mult: float = 1.5    # Outer convergence curve
    RD_mult: float = 0.382  # Divergence throat fillet

    # -----------------------------------------------------------------------
    # Wall material
    # -----------------------------------------------------------------------
    wall_k: float = 167.0       # Thermal conductivity [W/m·K]   (6061-T6 Al)
    wall_roughness: float = 6.3e-6  # Surface roughness [m]      (SLM 3D-print)
    wall_melt_T: float = 855.0  # Melting point [K]              (6061-T6 Al)

    # -----------------------------------------------------------------------
    # Coolant circuit
    # -----------------------------------------------------------------------
    T_coolant_inlet: float = 290.0   # Coolant inlet temperature [K]
    P_coolant_inlet: float = 80.0e5  # Coolant inlet pressure [Pa]
    # If None, mdot_coolant is computed from mdot_total / (1 + OF) at the end
    mdot_coolant: Optional[float] = None

    # -----------------------------------------------------------------------
    # Channel count and solver settings
    # -----------------------------------------------------------------------
    N_channels: int = 36
    dx: float = 1e-3     # Axial integration step [m]
