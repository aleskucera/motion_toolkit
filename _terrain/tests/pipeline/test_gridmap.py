"""Frame-convention round-trip guard for GridMap (the perception->planning contract).

The one real sim-to-real risk in the seam is a transpose / handedness slip in the
point-cloud rasterizer: x<->y or row<->col swapped. This drops an ASYMMETRIC spike at a
known world (x, y) with x != y into an ASYMMETRIC grid (nx != ny), runs the real pipeline,
and asserts the spike lands at the expected [row, col] of `TerrainMap.as_gridmap()`. A
transpose would move it (or push it out of the non-square grid) and fail loudly.

Run: python tests/pipeline/test_gridmap.py   (CPU; no outlier filter, which is GPU-only)
"""

from __future__ import annotations

import numpy as np
import warp as wp
from terrain_toolkit import GridMap
from terrain_toolkit import TerrainPipeline

BOUNDS = (0.0, 4.0, 0.0, 2.0)  # asymmetric: nx=40, ny=20 -> a transpose changes the shape
RESOLUTION = 0.1
SPIKE_XY = (3.05, 0.55)  # cell-center of col=30, row=5 (x != y, far from the diagonal)
SPIKE_Z = 1.0


def _cloud() -> np.ndarray:
    rng = np.random.default_rng(0)
    # dense flat ground at z=0 over the whole extent
    gx = rng.uniform(BOUNDS[0], BOUNDS[1], 60_000)
    gy = rng.uniform(BOUNDS[2], BOUNDS[3], 60_000)
    ground = np.stack([gx, gy, np.zeros_like(gx)], axis=1)
    # a tall block (a few cells wide) centered on SPIKE_XY at z=SPIKE_Z
    sx = rng.uniform(SPIKE_XY[0] - 0.05, SPIKE_XY[0] + 0.05, 4_000)
    sy = rng.uniform(SPIKE_XY[1] - 0.05, SPIKE_XY[1] + 0.05, 4_000)
    spike = np.stack([sx, sy, np.full_like(sx, SPIKE_Z)], axis=1)
    return np.concatenate([ground, spike], axis=0).astype(np.float32)


def main() -> None:
    wp.init()
    pipe = TerrainPipeline(
        resolution=RESOLUTION,
        bounds=BOUNDS,
        primary="max",  # the spike's z wins its cell
        inpaint=True,
        smooth_sigma=0.0,  # don't smear the spike
        device="cpu",
    )
    gm = pipe.process(_cloud()).as_gridmap()
    assert isinstance(gm, GridMap)

    # geometry: bounds -> (origin = min corner, cell = resolution)
    assert gm.origin == (0.0, 0.0), gm.origin
    assert gm.cell == RESOLUTION, gm.cell

    elev = gm.elevation
    # dims order pins [ny, nx] (a transpose would give (40, 20))
    assert elev.shape == (20, 40), elev.shape

    # the spike must land at row=y, col=x -- expected cell center (col=30, row=5)
    r, c = np.unravel_index(int(np.argmax(elev)), elev.shape)
    exp_c = round((SPIKE_XY[0] - BOUNDS[0]) / RESOLUTION - 0.5)  # x -> col
    exp_r = round((SPIKE_XY[1] - BOUNDS[2]) / RESOLUTION - 0.5)  # y -> row
    assert abs(r - exp_r) <= 1, f"row(y) off: got {r}, expected ~{exp_r} (x<->y transpose?)"
    assert abs(c - exp_c) <= 1, f"col(x) off: got {c}, expected ~{exp_c} (x<->y transpose?)"
    assert elev[r, c] > 0.5 * SPIKE_Z, elev[r, c]

    print(
        f"PASS: spike at world {SPIKE_XY} -> cell [row={r}, col={c}] (expected ~[{exp_r}, {exp_c}])"
    )
    print(f"      grid {elev.shape} (ny, nx), origin {gm.origin}, cell {gm.cell}")


if __name__ == "__main__":
    main()
