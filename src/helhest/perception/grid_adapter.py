"""Consumer side of the perception->planning seam: adapt a `GridMap`
(the shared heightmap contract) onto the engine's `GridParams`.

The elevation/valid arrays pass through untouched (`gm.elevation` is fed straight to
`ForwardSimulator.set_terrain`, zero-copy if it is a wp.array).
"""

from __future__ import annotations

from ..engine import GridParams
from .gridmap import GridMap


def grid_params_from(gm: GridMap) -> GridParams:
    """`GridMap` -> engine `GridParams`. Both use the min-corner / cell-center convention
    (origin = min corner; cell i center at origin + (i+0.5)*cell), so origin and cell pass
    straight through; dims come from `elevation.shape == (ny, nx)`."""
    ny, nx = gm.elevation.shape
    return GridParams(nx, ny, gm.cell, gm.origin[0], gm.origin[1])
