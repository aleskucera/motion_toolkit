"""Visibility filter test: a wall background + a person that walks through it.

Geometry (sensor at the origin, looking down +x):
  - a flat "wall" of points at x = 8 m spanning a WIDE y/z patch (the static
    background); it must angularly cover the person, else the person's edge
    bearings have no background behind them and can't be judged,
  - a "person" = a dense vertical slab at x = 3 m over a narrower patch, so it
    occludes the wall along its bearings.

Two checks, mirroring the two things the filter must do:
  1. Drop the person from a new scan (they sit in front of the mapped wall).
  2. Carve a stale person-ghost from the map (the scan now sees the wall behind).

Run: python tests/dynamic/test_dynamic_filter.py   (CPU)
"""

from __future__ import annotations

import numpy as np
import warp as wp
from terrain_toolkit import DynamicFilterConfig
from terrain_toolkit import DynamicPointFilter


def _slab(x: float, half: float, n: int) -> np.ndarray:
    rng = np.random.default_rng(int(x * 100))
    y = rng.uniform(-half, half, n)
    z = rng.uniform(-half, half, n)
    return np.stack([np.full(n, x), y, z], axis=1).astype(np.float32)


def main() -> None:
    wp.init()
    cfg = DynamicFilterConfig(
        az_bins=120,
        el_bins=40,
        el_min_deg=-30.0,
        el_max_deg=30.0,
        margin_m=0.3,
        margin_rel=0.02,
        min_range_m=0.5,
    )
    filt = DynamicPointFilter(cfg, device=wp.get_device("cpu"))
    origin = np.zeros(3)

    # Wall must subtend a wider angle than the person: half/x for the wall
    # (2.5/8 = 0.31 rad) exceeds the person's (0.6/3 = 0.20 rad).
    wall = _slab(8.0, half=2.5, n=30000)
    person = _slab(3.0, half=0.6, n=3000)

    # --- Check 1: a moving person in the scan is dropped ------------------
    # map = just the static wall; scan = wall bearings + the intruding person.
    scan = np.vstack([wall, person])
    scan_keep, map_keep = filt.filter(wall, scan, origin)

    n_wall = len(wall)
    wall_kept = scan_keep[:n_wall].mean()
    person_kept = scan_keep[n_wall:].mean()
    assert wall_kept > 0.95, f"static wall wrongly dropped: kept {wall_kept:.2f}"
    assert person_kept < 0.05, f"person not dropped: kept {person_kept:.2f}"
    assert map_keep.all(), "map wall wrongly carved by a consistent scan"

    # --- Check 2: a stale ghost in the map is carved ----------------------
    # map = wall + leftover person-ghost; scan = wall only (person has left).
    map_with_ghost = np.vstack([wall, person])
    scan_keep2, map_keep2 = filt.filter(map_with_ghost, wall, origin)

    ghost_kept = map_keep2[n_wall:].mean()  # the person slab inside the map
    wall_in_map_kept = map_keep2[:n_wall].mean()
    assert ghost_kept < 0.05, f"ghost not carved: kept {ghost_kept:.2f}"
    assert wall_in_map_kept > 0.95, f"static wall wrongly carved: kept {wall_in_map_kept:.2f}"
    assert scan_keep2.all(), "clean wall scan wrongly dropped"

    # --- Check 3: empty map is a no-op ------------------------------------
    sk, mk = filt.filter(np.empty((0, 3), np.float32), scan, origin)
    assert sk.all() and len(mk) == 0

    print(
        "PASS: dynamic filter — "
        f"person dropped ({person_kept:.2f} kept), ghost carved ({ghost_kept:.2f} kept), "
        f"wall preserved ({wall_kept:.2f}/{wall_in_map_kept:.2f})"
    )


if __name__ == "__main__":
    main()
