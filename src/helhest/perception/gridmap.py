"""GridMap: the shared heightmap representation between the perception producer
(helhest.perception) and a planning consumer (helhest.planning).

Deliberately minimal -- just the elevation grid + its world placement + an optional
validity mask. The richer multi-layer `TerrainMap`/`TerrainMapGPU` stay helhest.perception's
own output; `GridMap` is the thin contract that crosses the package boundary, so the frame
convention is asserted in exactly one place (see test_gridmap.py for the round-trip guard).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class GridMap:
    """Heightmap on a regular grid.

    Layout: `elevation[ny, nx]`, row = y (outer), col = x (inner). The grid `origin` is the
    MIN corner; world coords are cell-CENTERED:
        x = origin[0] + (col + 0.5) * cell,  y = origin[1] + (row + 0.5) * cell.
    `elevation` is host (np.ndarray) or device (wp.array) -- a device array lets a downstream
    Warp consumer read it zero-copy. `valid` is an optional same-shape bool mask
    (False = unobserved / occluded; None = every cell valid).
    """

    elevation: np.ndarray | wp.array
    origin: tuple[float, float]  # world (x, y) of the min corner
    cell: float  # meters per cell
    valid: np.ndarray | wp.array | None = None
