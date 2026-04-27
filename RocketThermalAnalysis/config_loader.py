"""
config_loader.py
Parse a TOML engine-design file into an EngineConfig + ChannelDesign.

Schema overview (see configs/*.toml for full examples):

    [propellants]   fuel, oxidizer, coolant
    [performance]   P_c, F_vac, OF, [OF_sweep], frozen
    [geometry]      exp_ratio, cont_ratio, L_star,
                    theta1, thetaD, thetaE,
                    R_chamber_mult, R_throat_conv_mult, R_throat_div_mult
    [wall]          k, roughness, melt_T,
                    [material], [T_ref_stress]
    [coolant]       T_inlet, P_inlet, mdot ("auto" or float kg/s)
    [channels]      n_throat, n_chamber,
                    [split_r_ratio], [split_transition], [dx],
                    height_taper, width_taper, wall_t_taper
    [film]          fraction, [inject_x], [coolant], [T_inlet], [Kt],
                    [model], [T_from_regen], [L_pyrolysis], [BL_thickness]
    [solver]        wall_2d, [wall_2d_stress],
                    [use_integral_bl], [C_bartz]

Any field not specified falls back to the EngineConfig dataclass default.
"""
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List

from config import ChannelDesign, EngineConfig, TaperEntry


def _coerce_taper(rows: List[List[Any]]) -> List[TaperEntry]:
    """Convert TOML array-of-arrays [[ "throat", 1.1e-3 ], ...] to taper entries."""
    out: List[TaperEntry] = []
    for row in rows:
        if len(row) != 2:
            raise ValueError(f"Channel taper row must be [station, value]; got {row!r}")
        station, value = row
        out.append((station, float(value)))
    return out


def _build_channels(section: Dict[str, Any]) -> ChannelDesign:
    if "n_throat" not in section or "n_chamber" not in section:
        raise ValueError("[channels] requires n_throat and n_chamber")
    return ChannelDesign(
        n_throat         = int(section["n_throat"]),
        n_chamber        = int(section["n_chamber"]),
        split_r_ratio    = float(section.get("split_r_ratio", 2.0)),
        split_transition = float(section.get("split_transition", 10e-3)),
        dx               = float(section.get("dx", 1e-3)),
        height_taper     = _coerce_taper(section["height_taper"]),
        width_taper      = _coerce_taper(section["width_taper"]),
        wall_t_taper     = _coerce_taper(section["wall_t_taper"]),
    )


def _coolant_mdot(value: Any):
    """'auto' (or unset) → None (computed downstream); else a float."""
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return None
    return float(value)


def load_config(path: str | Path) -> EngineConfig:
    """Parse a TOML file into a fully-populated EngineConfig.

    Raises ValueError with section/key context on schema mismatch."""
    path = Path(path)
    with path.open("rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: invalid TOML — {exc}") from exc

    try:
        propellants = data["propellants"]
        perf        = data["performance"]
        geom        = data["geometry"]
        wall        = data["wall"]
        cool        = data["coolant"]
        channels    = _build_channels(data["channels"])
    except KeyError as exc:
        raise ValueError(f"{path}: missing required section/key {exc}") from exc

    film   = data.get("film", {})
    solver = data.get("solver", {})

    of_sweep = perf.get("OF_sweep")
    if of_sweep is not None:
        of_sweep = tuple(float(v) for v in of_sweep)

    cfg = EngineConfig(
        # Propellants
        fuel       = propellants["fuel"],
        oxidizer   = propellants["oxidizer"],
        coolant    = propellants["coolant"],

        # Performance
        P_c        = float(perf["P_c"]),
        F_vac      = float(perf["F_vac"]),
        OF         = perf.get("OF"),
        OF_sweep   = of_sweep,
        frozen     = bool(perf.get("frozen", False)),

        # Geometry
        exp_ratio          = float(geom["exp_ratio"]),
        cont_ratio         = float(geom.get("cont_ratio", 6.0)),
        L_star             = float(geom.get("L_star", 1.143)),
        theta1             = float(geom.get("theta1", 30.0)),
        thetaD             = float(geom.get("thetaD", 30.0)),
        thetaE             = float(geom.get("thetaE", 12.0)),
        R_chamber_mult     = float(geom.get("R_chamber_mult", 1.5)),
        R_throat_conv_mult = float(geom.get("R_throat_conv_mult", 1.5)),
        R_throat_div_mult  = float(geom.get("R_throat_div_mult", 0.382)),

        # Wall
        wall_k         = float(wall["k"]),
        wall_roughness = float(wall.get("roughness", 6.3e-6)),
        wall_melt_T    = float(wall.get("melt_T", 855.0)),
        wall_material  = wall.get("material", "CuCrZr"),
        T_ref_stress   = float(wall.get("T_ref_stress", 293.0)),

        # Coolant
        T_coolant_inlet = float(cool["T_inlet"]),
        P_coolant_inlet = float(cool["P_inlet"]),
        mdot_coolant    = _coolant_mdot(cool.get("mdot", "auto")),

        # Film cooling (all optional)
        film_fraction     = float(film.get("fraction", 0.0)),
        film_inject_x     = float(film.get("inject_x", 0.0)),
        film_coolant      = film.get("coolant", "RP1"),
        film_T_inlet      = float(film.get("T_inlet", 290.0)),
        film_model        = film.get("model", "simon"),
        film_Kt           = float(film.get("Kt", 0.0013)),
        film_T_from_regen = bool(film.get("T_from_regen", False)),
        film_L_pyrolysis  = float(film.get("L_pyrolysis", 350e3)),
        film_BL_thickness = float(film.get("BL_thickness", 0.025)),

        # Solver
        wall_2d         = bool(solver.get("wall_2d", False)),
        wall_2d_stress  = bool(solver.get("wall_2d_stress", False)),
        use_integral_bl = bool(solver.get("use_integral_bl", False)),
        C_bartz         = float(solver.get("C_bartz", 0.026)),

        # Channels (preferred path — overrides any legacy field)
        channels        = channels,
    )
    return cfg


if __name__ == "__main__":
    # Diagnostic CLI: `python config_loader.py configs/foo.toml` dumps the
    # parsed EngineConfig for visual inspection.
    if len(sys.argv) != 2:
        print("usage: python config_loader.py <path-to.toml>", file=sys.stderr)
        sys.exit(2)
    cfg = load_config(sys.argv[1])
    from dataclasses import fields
    for f in fields(cfg):
        print(f"{f.name:>22} = {getattr(cfg, f.name)!r}")
