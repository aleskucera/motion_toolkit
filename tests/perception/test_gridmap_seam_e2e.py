"""End-to-end seam test: a motion_toolkit world -> point cloud -> helhest.perception pipeline ->
GridMap -> grid_params_from -> the motion_toolkit ENGINE settles on it.

Proves the whole chain: a helhest.perception-produced heightmap drives the planner's engine with
the frames aligned. The check that ties it down: the engine's settle on the helhest.perception-
sourced terrain matches its settle on the world's native terrain (same z/pitch/roll).

Run: python tests/perception/test_gridmap_seam_e2e.py   (CPU)
"""

from __future__ import annotations

import numpy as np
import warp as wp
from helhest import dynamics
from helhest import worlds as W
from helhest.engine import ForwardSimulator
from helhest.perception.grid_adapter import grid_params_from
from helhest.perception import TerrainPipeline

DEVICE = "cpu"


def cloud_from_heightmap(H: np.ndarray, x0: float, y0: float, cell: float) -> np.ndarray:
    """One point at each cell CENTER (min-corner convention: x = x0 + (col+0.5)*cell)."""
    ny, nx = H.shape
    rr, cc = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x = x0 + (cc + 0.5) * cell
    y = y0 + (rr + 0.5) * cell
    return np.stack([x.ravel(), y.ravel(), H.ravel()], axis=1).astype(np.float32)


def settle_z(grid, elevation: np.ndarray, pose) -> np.ndarray:
    """Settle a single robot at `pose` (zero control) on the given grid+elevation; return the
    derived (z, pitch, roll)."""
    sim = ForwardSimulator(dynamics.robot_params(), dynamics.execution_solver(), grid, 1, 1, DEVICE)
    sim.set_terrain(
        wp.array(np.ascontiguousarray(elevation, np.float32), dtype=wp.float32, device=DEVICE)
    )
    sim.set_uniform_friction(0.7)
    _, derived, _, _ = sim.rollout(np.zeros((1, 1, 3), np.float32), pose)
    return derived[0, 0].copy()  # (z, pitch, roll)


def main() -> None:
    wp.init()
    builder, start, _goal = W.WORLDS["pocket"]
    scene = builder()
    H0 = np.ascontiguousarray(scene.H, np.float32)

    # world -> point cloud -> helhest.perception pipeline -> GridMap
    pts = cloud_from_heightmap(H0, scene.x0, scene.y0, scene.cell)
    pipe = TerrainPipeline(
        resolution=scene.cell,
        bounds=(
            scene.x0,
            scene.x0 + scene.nx * scene.cell,
            scene.y0,
            scene.y0 + scene.ny * scene.cell,
        ),
        primary="max",
        inpaint=True,
        smooth_sigma=0.0,
        device=DEVICE,
    )
    gm = pipe.process(pts).as_gridmap()
    gp = grid_params_from(gm)

    # 1) the grid round-trips to the original world
    assert (gp.cells_x, gp.cells_y) == (scene.nx, scene.ny), (gp.cells_x, gp.cells_y)
    assert np.isclose(gp.cell_size, scene.cell) and np.isclose(gp.origin_x, scene.x0)
    assert np.isclose(gp.origin_y, scene.y0)

    # 2) the elevation survives the trip through helhest.perception (cell-center points, max primary)
    elev_err = float(np.abs(np.asarray(gm.elevation) - H0).max())
    assert elev_err < 1e-3, elev_err

    # 3) the engine settles IDENTICALLY on the seam-sourced terrain vs the world's native terrain
    native = settle_z(_native_grid(scene), H0, start)
    seam = settle_z(gp, np.asarray(gm.elevation), start)
    dz = float(np.abs(seam - native).max())
    assert dz < 1e-4, dz

    print(f"PASS: elevation round-trip err={elev_err:.2e}; engine settle match dz={dz:.2e}")
    print(f"      seam settle (z,pitch,roll)={tuple(round(float(v), 4) for v in seam)}")


def _native_grid(scene):
    from helhest.engine import GridParams

    return GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)


if __name__ == "__main__":
    main()
