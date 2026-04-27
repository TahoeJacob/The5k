"""
config.py
User-defined engine design parameters.
Fill in this file for each new engine — no other file should need editing for a new design.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

# A taper entry is (station, value_in_meters) where station is one of
# {"chamber", "throat", "exit"} or a float in [0, 1] (axial fraction:
# 0 = injector face, 1 = nozzle exit).
TaperEntry = Tuple[Union[str, float], float]


@dataclass
class ChannelDesign:
    """Per-station cooling-channel geometry, decoupled from EngineConfig.

    Width, height, and hot-wall thickness are specified as control-point
    tapers along the engine axis. The channel_builder linearly interpolates
    each onto the dx grid. Land width is derived: land = circumference/N - w.
    """
    n_throat:         int                  # channel count at the throat
    n_chamber:        int                  # channel count in the chamber (== n_throat → no bifurcation)
    height_taper:     List[TaperEntry]     # channel radial depth [m]
    width_taper:      List[TaperEntry]     # channel azimuthal width [m]
    wall_t_taper:     List[TaperEntry]     # hot-wall thickness [m]
    split_r_ratio:    float = 2.0          # split when local r/R_t exceeds this
    split_transition: float = 10e-3        # [m] Y-cusp ramp length
    dx:               float = 1e-3         # axial integration step [m]


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
    R_chamber_mult:    float = 1.5    # Big chamber-side converging arc
    R_throat_conv_mult: float = 1.5   # Convergent-side throat fillet (small)
    R_throat_div_mult:  float = 0.382 # Divergent-side throat fillet

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
    # Bifurcating channels: if N_channels_throat is set, channels at the throat
    # use N_channels_throat and split into N_channels_chamber wherever the local
    # wall radius exceeds channel_split_r_ratio · R_t.  Leave as None to use the
    # constant N_channels above for the entire engine.
    N_channels_throat:  Optional[int] = None
    N_channels_chamber: Optional[int] = None
    channel_split_r_ratio: float = 2.0   # split when r/R_t exceeds this
    channel_split_transition: float = 10e-3  # axial length [m] over which the
                                              # Y-cusp split is smoothed (real
                                              # cusps are not instantaneous).
                                              # Set to 0 for hard step.
    dx: float = 1e-3     # Axial integration step [m]
    # Tapered channel height [m].  If any of these are None, channel height
    # falls back to a constant 2 mm (set in main.py).  When all three are
    # provided, height is linearly interpolated: chamber → throat → exit.
    chan_h_throat:  Optional[float] = None
    chan_h_chamber: Optional[float] = None
    chan_h_exit:    Optional[float] = None

    wall_2d: bool = False  # True = 2-D wall conduction (Betti method);
                           # False = 1-D flat-plate + fin model

    # -----------------------------------------------------------------------
    # Thermal-stress post-processing  (generalized plane strain, free axial)
    # -----------------------------------------------------------------------
    wall_2d_stress: bool = False       # Run plane-strain thermo-elastic solve
                                       # on the same unit cell as wall_2d
    wall_material:  str  = "CuCrZr"    # Key into materials.MATERIALS table
    T_ref_stress:   float = 293.0      # Stress-free reference temperature [K]
    use_integral_bl: bool = False  # True = Bartz 1965 integral BL method;
                                   # False = simplified Bartz (Eq. 50)
    C_bartz: float = 0.026  # Bartz coefficient: 0.026 = thin BL (default),
                             # 0.023 = thick BL (Bartz 1965 Fig 10)
    h_cool_scale: float = 1.0  # Debug multiplier on coolant-side HTC.
                               # 1.0 = correlation as-is (default).
                               # NASA NTRS 20040076962 reports ±24-36% Nu
                               # uncertainty for RP-1, so values in
                               # 0.7–1.5 are within the experimental
                               # uncertainty band. Use to bracket model
                               # sensitivity, not as a permanent calibration.

    # -----------------------------------------------------------------------
    # Film cooling  (set film_fraction > 0 to enable)
    # -----------------------------------------------------------------------
    film_fraction:  float = 0.0     # Film flow as fraction of mdot_coolant
    film_inject_x:  float = 0.0     # Injection axial location [m] (0 = injector face)
    film_coolant:   str   = "RP1"   # CoolProp name of film fluid
    film_T_inlet:   float = 290.0   # Film injection temperature [K]
    film_model:     str   = "simon" # Film cooling model for gaseous phase:
                                    #   "vasiliev" = Vasiliev & Kudryavtsev (1993) exponential mixing
                                    #   "simon"    = Simon wall-jet (1986) + Spalding (1967) power-law decay
    film_Kt:        float = 0.0013  # Turbulent mixing intensity (Vasiliev & Kudryavtsev 1993)
                                    # Published range: (0.05-0.20)×10⁻² = 0.0005-0.002
    film_T_from_regen: bool = False # If True, film_T_inlet = regen coolant exit temperature
                                    # (iterates thermal solve until converged)
    film_L_pyrolysis: float = 350e3  # Endothermic pyrolysis enthalpy [J/kg] added to L_vap
                                     # for Spalding B-number transpiration correction.
                                     # RP-1 cracking: 300-600 kJ/kg (Edwards 2003, Ward 2004)
                                     # Set to 0 to use latent heat only (no pyrolysis credit).
    film_BL_thickness: float = 0.025 # RPA "Relative thickness of near-wall layer" —
                                     # fraction of total mass flow in the surface layer
                                     # used to compute OF_eff for h_gas property evaluation.
                                     # 0.0 disables surface-layer CEA lookup (bare Bartz).

    # -----------------------------------------------------------------------
    # Channel geometry (preferred)
    # -----------------------------------------------------------------------
    # If `channels` is set, it takes precedence and the legacy aggregate fields
    # (N_channels, N_channels_throat/chamber, chan_h_*, channel_split_*, dx)
    # are mirrored from it for any legacy reader.  If `channels` is None, a
    # ChannelDesign is synthesized from the legacy fields in __post_init__.
    channels: Optional[ChannelDesign] = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = self._channels_from_legacy()
        else:
            # Mirror the new design back into the legacy fields so any code
            # path still reading config.N_channels / config.chan_h_throat /
            # config.dx sees consistent values.
            cd = self.channels
            self.N_channels             = cd.n_throat
            self.N_channels_throat      = cd.n_throat
            self.N_channels_chamber     = cd.n_chamber
            self.channel_split_r_ratio  = cd.split_r_ratio
            self.channel_split_transition = cd.split_transition
            self.dx                     = cd.dx
            self.chan_h_chamber = _lookup_taper(cd.height_taper, "chamber")
            self.chan_h_throat  = _lookup_taper(cd.height_taper, "throat")
            self.chan_h_exit    = _lookup_taper(cd.height_taper, "exit")

    def _channels_from_legacy(self) -> ChannelDesign:
        """Build a ChannelDesign from the legacy aggregate fields so the
        new code path can be used unconditionally without breaking existing
        inline EngineConfig literals.  Width and wall-thickness default to
        the historical hardcoded values from main.py (1.0 mm each)."""
        n_throat  = self.N_channels_throat  or self.N_channels
        n_chamber = self.N_channels_chamber or n_throat
        h_throat  = self.chan_h_throat  if self.chan_h_throat  is not None else 2e-3
        h_chamber = self.chan_h_chamber if self.chan_h_chamber is not None else 2e-3
        h_exit    = self.chan_h_exit    if self.chan_h_exit    is not None else 2e-3
        return ChannelDesign(
            n_throat         = n_throat,
            n_chamber        = n_chamber,
            split_r_ratio    = self.channel_split_r_ratio,
            split_transition = self.channel_split_transition,
            dx               = self.dx,
            height_taper = [("chamber", h_chamber), ("throat", h_throat), ("exit", h_exit)],
            width_taper  = [("chamber", 1.0e-3),    ("throat", 1.0e-3),   ("exit", 1.0e-3)],
            wall_t_taper = [("chamber", 1.0e-3),    ("throat", 1.0e-3),   ("exit", 1.0e-3)],
        )


def _lookup_taper(taper: List[TaperEntry], station: str) -> Optional[float]:
    """Return the value for a named station ("chamber"/"throat"/"exit") in a
    taper table, or None if not present.  Used to mirror taper values back
    into the legacy `chan_h_chamber`/`_throat`/`_exit` fields."""
    for s, v in taper:
        if s == station:
            return float(v)
    return None
