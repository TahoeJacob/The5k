"""
channel_builder.py
Build a per-station ChannelGeometry from a ChannelDesign control-point spec.

Extracted from main.py so any caller (main, validation scripts, optimization
loops) can construct channels the same way without reimplementing the
bifurcation + Y-cusp transition + land-derivation logic.
"""
from typing import List, Optional, Tuple, Union

import numpy as np

from config import EngineConfig, TaperEntry
from geometry import EngineGeometry, nozzle_radius
from heat_transfer import ChannelGeometry


def _resolve_station(s: Union[str, float], L_c: float, L_total: float) -> float:
    """Map a station label to an absolute axial position [m].

    "chamber" → 0.0 (injector face)
    "throat"  → L_c
    "exit"    → L_total
    float in [0, 1] → fraction of L_total
    float > 1 (m) → treated as absolute axial position
    """
    if isinstance(s, str):
        key = s.lower()
        if key in ("chamber", "injector"):
            return 0.0
        if key == "throat":
            return L_c
        if key in ("exit", "nozzle_exit"):
            return L_total
        raise ValueError(f"Unknown channel-taper station label: {s!r}")
    val = float(s)
    if 0.0 <= val <= 1.0:
        return val * L_total
    return val  # absolute metres (legacy)


def _interp_taper(taper: List[TaperEntry], x_j: np.ndarray,
                  L_c: float, L_total: float) -> np.ndarray:
    """Linearly interpolate a (station, value) taper onto x_j."""
    if not taper:
        raise ValueError("Empty taper table")
    xs = np.array([_resolve_station(s, L_c, L_total) for s, _ in taper])
    ys = np.array([float(v) for _, v in taper])
    order = np.argsort(xs)
    return np.interp(x_j, xs[order], ys[order])


def _find_split_crossings(x_j: np.ndarray, r_arr: np.ndarray,
                          split_r: float, x_throat: float
                          ) -> Tuple[Optional[float], Optional[float]]:
    """Locate the upstream / downstream axial positions where r(x) = split_r.

    Mirrors the original logic in main.py: only the upstream crossing is
    populated; the nozzle side is intentionally left unsplit (simpler SLM
    geometry — no Y-merge in the diverging section)."""
    split_above = r_arr > split_r
    x_split_up = None
    x_split_down = None
    for i in range(1, len(x_j)):
        if split_above[i] != split_above[i - 1]:
            r0, r1 = r_arr[i - 1], r_arr[i]
            frac = (split_r - r0) / (r1 - r0) if r1 != r0 else 0.0
            x_cross = x_j[i - 1] + frac * (x_j[i] - x_j[i - 1])
            if x_cross < x_throat and x_split_up is None:
                x_split_up = x_cross
            # Nozzle side intentionally left unsplit
    return x_split_up, x_split_down


def _make_n_local(n_throat: int, n_chamber: int,
                  x_split_up: Optional[float], x_split_down: Optional[float],
                  split_transition: float):
    """Return n_local(x) — the (possibly non-integer) channel count at x.

    Equals n_throat between the two crossings, n_chamber outside, and ramps
    linearly across a band of width split_transition centred on each crossing
    to mimic a real Y-cusp split rather than an instantaneous jump."""
    half_trans = 0.5 * split_transition

    def n_local(x: float) -> float:
        if x_split_up is not None and x < x_split_up - half_trans:
            return float(n_chamber)
        if x_split_up is not None and x < x_split_up + half_trans:
            f = (x - (x_split_up - half_trans)) / max(2.0 * half_trans, 1e-12)
            return float(n_chamber + (n_throat - n_chamber) * f)
        if x_split_down is None or x < x_split_down - half_trans:
            return float(n_throat)
        if x < x_split_down + half_trans:
            f = (x - (x_split_down - half_trans)) / max(2.0 * half_trans, 1e-12)
            return float(n_throat + (n_chamber - n_throat) * f)
        return float(n_chamber)

    return n_local


def build_channel_geometry(config: EngineConfig, geom: EngineGeometry
                           ) -> Tuple[ChannelGeometry, dict]:
    """Construct per-station ChannelGeometry from config.channels.

    Returns
    -------
    chan_geom : ChannelGeometry
    info : dict
        Keys: N_throat, N_chamber, x_split_up, x_split_down, half_trans,
        x_throat, L_total — for downstream segment-summary printouts.
    """
    cd = config.channels
    x_throat = geom.L_c
    L_total  = geom.L_c + geom.L_nozzle

    x_j = np.arange(0.0, L_total, cd.dx)
    r_arr = np.array([nozzle_radius(x, geom, cd.dx) for x in x_j])

    # Bifurcation crossings + per-station channel count function
    split_r = cd.split_r_ratio * geom.R_t
    x_split_up, x_split_down = _find_split_crossings(x_j, r_arr, split_r, x_throat)
    n_local = _make_n_local(cd.n_throat, cd.n_chamber,
                            x_split_up, x_split_down, cd.split_transition)

    # Per-station tapers
    chan_h = _interp_taper(cd.height_taper, x_j, x_throat, L_total)
    chan_w = _interp_taper(cd.width_taper,  x_j, x_throat, L_total)
    chan_t = _interp_taper(cd.wall_t_taper, x_j, x_throat, L_total)

    # Channel count at each station + derived land width
    n_chan_float = np.array([n_local(x) for x in x_j])
    chan_land = 2.0 * np.pi * r_arr / n_chan_float - chan_w

    chan_geom = ChannelGeometry(x_j=x_j, chan_w=chan_w, chan_h=chan_h,
                                chan_t=chan_t, chan_land=chan_land,
                                n_chan=n_chan_float)

    info = dict(
        N_throat     = cd.n_throat,
        N_chamber    = cd.n_chamber,
        x_split_up   = x_split_up,
        x_split_down = x_split_down,
        half_trans   = 0.5 * cd.split_transition,
        x_throat     = x_throat,
        L_total      = L_total,
        split_r      = split_r,
        split_r_ratio = cd.split_r_ratio,
        split_transition = cd.split_transition,
    )
    return chan_geom, info


def report_channels(chan_geom: ChannelGeometry, info: dict,
                    geom: EngineGeometry) -> None:
    """Print the bifurcation summary + segment-dimension table that main.py
    used to emit inline.  Output is byte-identical to the original block."""
    N_throat  = info["N_throat"]
    N_chamber = info["N_chamber"]
    x_split_up   = info["x_split_up"]
    x_split_down = info["x_split_down"]
    half_trans   = info["half_trans"]
    x_throat     = info["x_throat"]
    L_total      = info["L_total"]

    chan_w = chan_geom.chan_w
    chan_h = chan_geom.chan_h
    chan_t = chan_geom.chan_t

    if N_chamber != N_throat:
        print(f"\n--- Bifurcating channels ---")
        print(f"  N_throat  = {N_throat}, N_chamber = {N_chamber}")
        print(f"  Split radius = {info['split_r']*1000:.2f} mm  "
              f"(= {info['split_r_ratio']:.2f} · R_t)")
        if x_split_up is not None:
            print(f"  Upstream split   x ≈ {x_split_up*1000:.1f} mm")
        if x_split_down is not None:
            print(f"  Downstream split x ≈ {x_split_down*1000:.1f} mm")
        print(f"  Y-cusp transition length: {info['split_transition']*1000:.1f} mm")
        print(f"  Channel width range:  {chan_w.min()*1000:.3f} – {chan_w.max()*1000:.3f} mm")
        print(f"  Channel height range: {chan_h.min()*1000:.3f} – {chan_h.max()*1000:.3f} mm")

    # Segment summary for RPA entry — dimensions at each transition point
    x_j = chan_geom.x_j
    chan_land = chan_geom.chan_land

    n_local = _make_n_local(N_throat, N_chamber, x_split_up, x_split_down,
                            info["split_transition"])

    def _dims_at(x: float):
        N = n_local(x)
        r = nozzle_radius(x, geom, chan_geom.x_j[1] - chan_geom.x_j[0]
                          if len(chan_geom.x_j) > 1 else 1e-3)
        h = float(np.interp(x, x_j, chan_h))
        ld = float(np.interp(x, x_j, chan_land))
        t  = float(np.interp(x, x_j, chan_t))
        w = (2.0 * np.pi * r / N) - ld
        return N, w, h, ld, t, r

    boundaries = [("Injector face", 0.0)]
    if x_split_up is not None:
        boundaries.append(("Upstream split START", x_split_up - half_trans))
        boundaries.append(("Upstream split END",   x_split_up + half_trans))
    boundaries.append(("Throat", x_throat))
    if x_split_down is not None:
        boundaries.append(("Downstream split START", x_split_down - half_trans))
        boundaries.append(("Downstream split END",   x_split_down + half_trans))
    boundaries.append(("Nozzle exit", L_total))
    boundaries = [(lbl, max(0.0, min(x, L_total))) for lbl, x in boundaries]

    print(f"\n--- Segment dimensions for RPA entry ---")
    print(f"  {'Station':<26} {'x[mm]':>7} {'r[mm]':>7} {'N':>6} "
          f"{'w[mm]':>7} {'h[mm]':>7} {'land[mm]':>9} {'t_w[mm]':>8}")
    print("  " + "-" * 82)
    for lbl, x in boundaries:
        N, w, h, ld, t, r = _dims_at(x)
        print(f"  {lbl:<26} {x*1000:7.1f} {r*1000:7.2f} {N:6.1f} "
              f"{w*1000:7.3f} {h*1000:7.3f} {ld*1000:9.3f} {t*1000:8.3f}")
    print()
    print(f"  Wall thickness (chan_t): {chan_t.min()*1000:.3f} mm (constant)")
    print(f"  Engine total length: {L_total*1000:.1f} mm  "
          f"(L_c={geom.L_c*1000:.1f} mm, L_nozzle={geom.L_nozzle*1000:.1f} mm)")
