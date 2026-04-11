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
    # Tapered channel width [m] — same convention as chan_h_*.  When all three
    # are provided, chan_w is PRESCRIBED directly and the land width falls out
    # of the circumferential packing:  land(x) = 2π·r(x)/N(x) − chan_w(x).
    # Leave as None to use the legacy land-profile path (land is prescribed,
    # width is derived).
    chan_w_throat:  Optional[float] = None
    chan_w_chamber: Optional[float] = None
    chan_w_exit:    Optional[float] = None

    wall_2d: bool = False  # True = 2-D wall conduction (Betti method);
                           # False = 1-D flat-plate + fin model
    use_integral_bl: bool = False  # True = Bartz 1965 integral BL method;
                                   # False = simplified Bartz (Eq. 50)
    C_bartz: float = 0.026  # Bartz coefficient: 0.026 = thin BL (default),
                             # 0.023 = thick BL (Bartz 1965 Fig 10)

    # -----------------------------------------------------------------------
    # Film cooling  (set film_fraction > 0 to enable)
    # -----------------------------------------------------------------------
    film_fraction:  float = 0.0     # Film flow as fraction of mdot_coolant
    film_inject_x:  float = 0.0     # Injection axial location [m] (0 = injector face)
    film_coolant:   str   = "RP1"   # CoolProp name of film fluid
    film_T_inlet:   float = 290.0   # Film injection temperature [K]
    film_Kt:        float = 0.0013  # Turbulent mixing intensity (Vasiliev & Kudryavtsev 1993)
                                    # Published range: (0.05-0.20)×10⁻² = 0.0005-0.002
    film_T_from_regen: bool = False # If True, film_T_inlet = regen coolant exit temperature
                                    # (iterates thermal solve until converged)
    # Film cooling gaseous-mixing model selector:
    #   "v_k"  — Vasiliev-Kudryavtsev (Ponomarenko 2012, used by RPA)
    #   "aedc" — Kutateladze/Stollery with virtual upstream edge
    #            (AEDC-TR-91-1 Eq. 3.5, self-contained, no free Kt)
    film_model:     str   = "v_k"
    # AEDC-only: free-stream turbulence rms fraction (e_t).  Typical rocket
    # chamber values 0.05–0.10.  Turbulence correction:  K_t = 1 + 10.2·e_t
    film_aedc_et:   float = 0.07
