# Thermal Model Report — No Film Cooling Baseline

**Question for reviewer:** Hot-wall temperature `T_hw` looks massively over-predicted for what should be a tractable no-film, copper-walled chamber. Please audit the model (formulation, boundary conditions, units) using the data and code excerpts below.

- Pipeline entry point: `RocketThermalAnalysis/main.py` → `run(plot=False)`
- Solver: 1-D axial march coupled to a 2-D wall conduction slice (`config.wall_2d = False`)
- Film cooling: `film_fraction = 0.0` (none)

## 1. Engine config

| Field | Value |
|---|---|
| Fuel / Oxidizer | RP-1 / LOX |
| Coolant | RP1 |
| Chamber pressure P_c | 20.00 bar |
| Vacuum thrust F_vac | 3074.0 N |
| O/F ratio | 2.00 |
| Expansion ratio Ae/At | 6.00 |
| Contraction ratio Ac/At | 8.00 |
| L* | 1270.0 mm |
| Nozzle θ1 / θD / θE | 30.0° / 30.0° / 12.0° |
| R_chamber, R_throat_conv, R_throat_div mults | 1.500, 1.500, 0.382 |
| CEA mode | equilibrium |
| Bartz coefficient C | 0.026 |
| Use integral BL | False |

## 2. Wall material & coolant boundary

| Field | Value |
|---|---|
| Wall material | CuCrZr |
| Wall conductivity k | 300.0 W/(m·K) |
| Wall roughness | 63.0 µm |
| Wall T limit (config.wall_melt_T) | 1073 K (800°C) |
| Coolant inlet T | 298.0 K |
| Coolant inlet P | 35.00 bar |
| Coolant mass flow (computed) | 0.3436 kg/s |
| Coolant flow direction | counter-current (enters at nozzle exit, exits at injector face) |
| 2-D wall solve | `wall_2d = False` |
| 2-D wall stress | `wall_2d_stress = False` |

## 3. CEA stagnation-state outputs

| Field | Value | Units |
|---|---|---|
| T_c (chamber stag temp) | 3254.1 | K |
| γ_c | 1.1614 | — |
| C* | 1783.6 | m/s |
| Cp_froz_c | 2140.0 | J/(kg·K) |
| Pr_froz_c | 0.5941 | — |
| µ_c (visc) | 9.8655e-05 | Pa·s |
| Isp_vac (exit) | 2982.1 | m/s |

## 4. Computed engine geometry

| Quantity | Value | Units |
|---|---|---|
| Throat dia D_t | 34.212 | mm |
| Chamber dia D_c | 96.766 | mm |
| Exit dia D_e | 83.802 | mm |
| Throat area A_t | 919.283 | mm² |
| Chamber area A_c | 7354.262 | mm² |
| Exit area A_e | 5515.697 | mm² |
| Chamber length L_c | 189.71 | mm |
| Nozzle length L_n | 74.03 | mm |
| Total length | 263.74 | mm |
| R_chamber arc | 25.66 | mm |
| R_throat_conv | 25.66 | mm |
| R_throat_div  | 6.53 | mm |
| Total mass flow | 1.0308 | kg/s |
| Fuel mass flow | 0.3436 | kg/s |
| Ox mass flow | 0.6872 | kg/s |
| Exit Mach | 2.839 | — |
| Exit static pressure | 54.35 | kPa |

## 5. Cooling channel layout

- N_throat = 53, N_chamber = 106, split radius = 2.0 × R_t
- Channel width (constant): 1.000 mm
- Wall thickness (constant): 1.000 mm
- Channel height taper: chamber 1.30 mm → throat 1.10 mm → exit 1.50 mm

Channel dimensions at key stations:

| Station | x [mm] | r_wall [mm] | N | w [mm] | h [mm] | land [mm] | t_wall [mm] |
|---|---|---|---|---|---|---|---|
| Injector face | 0.0 | 48.38 | 106.0 | 1.000 | 1.300 | 1.868 | 1.000 |
| Throat | 190.0 | 17.11 | 53.0 | 1.000 | 1.102 | 1.029 | 1.000 |
| Nozzle exit | 263.0 | 41.74 | 53.0 | 1.000 | 1.496 | 3.949 | 1.000 |
| T_hw peak | 189.0 | 17.12 | 53.0 | 1.000 | 1.101 | 1.029 | 1.000 |

## 6. Peak / extreme values

| Quantity | Value | Location |
|---|---|---|
| max T_hw | 1419.7 K (1147°C) | x = 189.0 mm |
| max T_cw | 1368.5 K | x = 190.0 mm |
| max heat flux | 15.41 MW/m² | x = 188.0 mm |
| max h_gas | 8560.1 W/(m²·K) | x = 189.0 mm |
| max T_coolant | 637.6 K | x = 0.0 mm |
| min P_coolant | 32.82 bar | x = 0.0 mm |
| ΔT_wall (T_hw-T_cw) max | 51.4 K | x = 188.0 mm |
| Wall T limit (config) | 1073 K | — |
| Margin (limit - max T_hw) | -346.7 K | — |

## 7. Axial-station table

Counter-current coolant: T_coolant *decreases* with x (it enters at exit and leaves at injector). Indices marked * are the additional rows added for injector / throat / exit / T_hw peak.

| x [mm] | r_wall [mm] | M | P_gas [kPa] | T_aw [K] | h_gas [kW/m²K] | q_w [MW/m²] | T_hw [K] | T_cw [K] | T_cool [K] | P_cool [bar] | h_cool [kW/m²K] | Re_cool [1e3] | v_cool [m/s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|    0.0* |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.57 |    840 |    828 |    638 | 32.82 |  10.29 |   47.4 |   5.1 |
|    9.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.57 |    839 |    827 |    629 | 32.86 |   9.92 |   43.4 |   5.0 |
|   18.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.57 |    838 |    826 |    621 | 32.90 |   9.58 |   39.8 |   4.9 |
|   27.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.57 |    837 |    825 |    612 | 32.94 |   9.28 |   36.7 |   4.8 |
|   36.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.58 |    836 |    824 |    604 | 32.98 |   8.99 |   33.9 |   4.7 |
|   45.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.58 |    835 |    823 |    595 | 33.01 |   8.72 |   31.3 |   4.7 |
|   54.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.58 |    833 |    821 |    586 | 33.05 |   8.47 |   29.0 |   4.6 |
|   63.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.58 |    832 |    820 |    577 | 33.09 |   8.23 |   26.9 |   4.6 |
|   72.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.59 |    830 |    818 |    568 | 33.13 |   8.01 |   25.0 |   4.5 |
|   81.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.59 |    829 |    816 |    559 | 33.16 |   7.79 |   23.2 |   4.5 |
|   90.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.59 |    827 |    815 |    549 | 33.20 |   7.58 |   21.6 |   4.5 |
|   99.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.60 |    826 |    814 |    539 | 33.24 |   7.38 |   20.1 |   4.4 |
|  108.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.60 |    824 |    812 |    530 | 33.28 |   7.18 |   18.7 |   4.4 |
|  117.0 |  48.38 | 0.075 |  1993.6 |   3254 |   1.48 |  3.60 |    823 |    811 |    520 | 33.32 |   6.99 |   17.3 |   4.4 |
|  126.0 |  48.03 | 0.076 |  1993.4 |   3254 |   1.50 |  3.65 |    825 |    813 |    510 | 33.36 |   6.80 |   16.1 |   4.3 |
|  135.0 |  44.72 | 0.087 |  1991.2 |   3254 |   1.70 |  4.08 |    854 |    841 |    499 | 33.40 |   6.55 |   14.9 |   4.3 |
|  144.0 |  39.53 | 0.112 |  1985.5 |   3254 |   2.10 |  4.92 |    913 |    896 |    488 | 33.44 |   6.25 |   13.7 |   4.3 |
|  153.0 |  34.33 | 0.149 |  1974.3 |   3253 |   2.69 |  6.17 |    958 |    937 |    476 | 33.49 |   7.44 |   16.5 |   5.6 |
|  162.0 |  29.14 | 0.210 |  1949.5 |   3252 |   3.58 |  8.07 |   1001 |    974 |    462 | 33.62 |   9.90 |   22.5 |   8.4 |
|  171.0 |  23.94 | 0.321 |  1884.3 |   3250 |   4.99 | 10.51 |   1142 |   1107 |    447 | 33.78 |   9.22 |   19.9 |   8.4 |
|  180.0 |  19.01 | 0.574 |  1656.2 |   3241 |   7.29 | 13.95 |   1328 |   1282 |    430 | 33.94 |   8.53 |   17.4 |   8.3 |
|  189.0* |  17.12 | 0.965 |  1187.4 |   3218 |   8.56 | 15.39 |   1420 |   1368 |    412 | 34.11 |   7.98 |   15.0 |   8.2 |
|  190.0* |  17.11 | 1.029 |  1109.1 |   3213 |   8.54 | 15.31 |   1420 |   1369 |    410 | 34.13 |   7.91 |   14.7 |   8.2 |
|  198.0 |  20.72 | 1.732 |   420.1 |   3153 |   5.94 | 11.29 |   1253 |   1215 |    395 | 34.27 |   7.33 |   12.5 |   7.7 |
|  207.0 |  24.95 | 2.082 |   231.2 |   3120 |   4.23 |  8.45 |   1123 |   1094 |    379 | 34.41 |   6.73 |   10.5 |   7.3 |
|  216.0 |  28.56 | 2.298 |   155.5 |   3099 |   3.31 |  6.80 |   1044 |   1021 |    365 | 34.53 |   6.15 |    8.8 |   6.9 |
|  225.0 |  31.71 | 2.454 |   115.7 |   3085 |   2.73 |  5.72 |    992 |    973 |    351 | 34.64 |   5.59 |    7.3 |   6.6 |
|  234.0 |  34.50 | 2.574 |    91.7 |   3073 |   2.34 |  4.95 |    959 |    943 |    337 | 34.74 |   5.06 |    6.1 |   6.2 |
|  243.0 |  37.00 | 2.671 |    75.8 |   3065 |   2.06 |  4.37 |    939 |    925 |    324 | 34.83 |   4.56 |    5.0 |   6.0 |
|  252.0 |  39.25 | 2.752 |    64.7 |   3057 |   1.84 |  3.91 |    929 |    916 |    312 | 34.92 |   4.09 |    4.0 |   5.7 |
|  262.0* |  41.53 | 2.827 |    55.7 |   3051 |   1.65 |  3.50 |    930 |    918 |    298 | 35.00 |   3.58 |    3.1 |   5.4 |

## 8. Model excerpts (verbatim from `heat_transfer.py`)

### `_bartz_h` — gas-side HTC

```python
def _bartz_h(M: float, A: float, T_hw: float,
             cea: CEAResult, geom: EngineGeometry,
             C: float = 0.026) -> float:
    """
    Bartz (1957) gas-side HTC [W/(m²·K)].

    Transport properties are evaluated at the chamber stagnation state
    (T_c, P_c) using FROZEN Cp and Pr per the Bartz reference-state
    convention.  Frozen (composition-fixed) properties are used because
    the boundary layer cannot equilibrate chemically on the residence-time
    scale — consistent with Cantera at fixed mole fractions (the method
    used in the reference MixtureOptimization.py code).  Using equilibrium
    Cp (which includes the ∂H/∂T contribution of reaction progress) would
    over-predict h_g by ~1.5–2× for hydrogen flames.

    σ correction per Sutton & Biblarz (2010) eq. 8-22:
      σ = 1 / [(½·(T_hw/T_c)·(1+(γ-1)/2·M²) + ½)^0.68 · (1+(γ-1)/2·M²)^0.12]

    Parameters
    ----------
    C : float
        Bartz leading coefficient.  C = 0.026 (thin BL, default) or
        C = 0.023 (thick BL, Bartz 1965 Fig 10).
    """
    gam   = cea.gamma_c
    D_t   = 2.0 * geom.R_t
    R_cur = 0.5 * (geom.R_throat_conv + geom.R_throat_div)  # mean throat curvature

    fac   = 1.0 + (gam - 1.0) / 2.0 * M**2
    sigma = 1.0 / ((0.5 * (T_hw / cea.T_c) * fac + 0.5)**0.68 * fac**0.12)

    return (
        (C / D_t**0.2)
        * (cea.visc_c**0.2 * cea.Cp_froz_c / cea.Pr_froz_c**0.6)
        * (cea.P_c / cea.C_star)**0.8
        * (D_t / R_cur)**0.1
        * (geom.A_t / A)**0.9
        * sigma
    )
```

### `_T_aw` — adiabatic wall (recovery) temperature

```python
def _T_aw(M: float, cea: CEAResult) -> float:
    """
    Adiabatic wall temperature [K].

    Recovery factor for turbulent flow:  r = Pr_froz^(1/3)
    T_aw = T_c · (1 + r·(γ-1)/2·M²) / (1 + (γ-1)/2·M²)

    Frozen Pr is used for the same reason as in _bartz_h — the recovery
    factor is a boundary-layer quantity evaluated at fixed composition.
    """
    gam = cea.gamma_c
    r   = cea.Pr_froz_c**(1.0 / 3.0)
    N   = M**2
    return cea.T_c * (1.0 + r * (gam - 1.0) / 2.0 * N) / (1.0 + (gam - 1.0) / 2.0 * N)
```

## 9. What I want the reviewer to look at

1. Is the `_bartz_h` formulation (Bartz 1957 with σ correction) correctly implemented? Are units consistent?
2. Recovery factor in `_T_aw` uses `r = Pr_froz^(1/3)` — appropriate for turbulent BL?
3. Coolant-side: is the Nu correlation a reasonable choice for RP-1 in narrow rectangular SLM channels at these Re?
4. Is the wall model self-consistent? With `wall_2d=True` the 2-D conduction should match a 1-D thin-wall estimate (`q = k(T_hw-T_cw)/t_wall`) at chamber stations where the fin effect is small.
5. Compare peak T_hw to physical intuition — for a copper (k=300 W/m·K, t=1 mm) wall at q≈30–60 MW/m² with subcooled RP-1 at ~300 K and h_cool ~10–30 kW/m²K, what range of T_hw is expected?

_Generated: Mon Apr 27 16:24:38 NZST 2026_
