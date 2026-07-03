"""Consumer side of the perception->planning seam: adapt a helhest.perception `GridMap`
(the shared heightmap contract) onto the engine's `GridParams`.

The import of `GridMap` is type-only -- the shim duck-types the instance at runtime, so
motion_toolkit keeps helhest.perception an OPTIONAL dependency (you only ever hold a GridMap
when helhest.perception produced one). The elevation/valid arrays pass through untouched
(`gm.elevation` is fed straight to `ForwardSimulator.set_terrain`, zero-copy if it is a wp.array).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..engine import GridParams

if TYPE_CHECKING:
    from helhest.perception import GridMap


def grid_params_from(gm: GridMap) -> GridParams:
    """`GridMap` -> engine `GridParams`. Both use the min-corner / cell-center convention
    (origin = min corner; cell i center at origin + (i+0.5)*cell), so origin and cell pass
    straight through; dims come from `elevation.shape == (ny, nx)`."""
    ny, nx = gm.elevation.shape
    return GridParams(nx, ny, gm.cell, gm.origin[0], gm.origin[1])
