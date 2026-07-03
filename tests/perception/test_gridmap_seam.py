"""Seam test: helhest.perception GridMap -> engine GridParams maps cleanly and preserves the
frame convention (dims order, min-corner origin, cell size).

Run: python tests/perception/test_gridmap_seam.py
"""

from __future__ import annotations

import numpy as np
from helhest.perception.grid_adapter import grid_params_from
from helhest.perception import GridMap


def main() -> None:
    elev = np.zeros((20, 40), np.float32)  # (ny, nx) -- deliberately asymmetric
    gm = GridMap(elevation=elev, origin=(1.0, -2.0), cell=0.1)
    gp = grid_params_from(gm)

    # dims: elevation (ny, nx) -> (cells_x=nx, cells_y=ny) -- not transposed
    assert (gp.cells_x, gp.cells_y) == (40, 20), (gp.cells_x, gp.cells_y)
    assert gp.cell_size == 0.1, gp.cell_size
    assert (gp.origin_x, gp.origin_y) == (1.0, -2.0), (gp.origin_x, gp.origin_y)

    # GridParams.bounds round-trips to the GridMap's world extent (min corner + dims*cell)
    xmin, xmax, ymin, ymax = gp.bounds
    assert (xmin, ymin) == gm.origin, (xmin, ymin)
    assert np.isclose(xmax, 1.0 + 40 * 0.1) and np.isclose(ymax, -2.0 + 20 * 0.1), (xmax, ymax)

    print("PASS: GridMap -> GridParams", (gp.cells_x, gp.cells_y), "bounds", gp.bounds)


if __name__ == "__main__":
    main()
