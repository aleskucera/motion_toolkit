"""Device-native path: carve() and DeviceMapAccumulator vs numpy references.

Guards that the on-device carve + accumulate produce exactly what a plain-numpy
implementation of the same steps would, so the GPU loop is a faithful (faster)
version of the host loop.

Run: python tests/sim/test_accumulate.py   (CPU)
"""

from __future__ import annotations

import numpy as np
import warp as wp
from terrain_toolkit import DeviceMapAccumulator
from terrain_toolkit import DynamicFilterConfig
from terrain_toolkit import DynamicPointFilter

VOX = 0.15
RADIUS = 25.0
Z0, Z1 = -2.0, 6.0


def test_carve_matches_filter_and_device() -> None:
    dev = wp.get_device("cpu")
    f = DynamicPointFilter(
        DynamicFilterConfig(az_bins=180, el_bins=90, el_min_deg=-30, el_max_deg=30), device=dev
    )
    rng = np.random.default_rng(0)
    wall = np.stack(
        [np.full(3000, 8.0), rng.uniform(-2.5, 2.5, 3000), rng.uniform(-2.5, 2.5, 3000)], 1
    ).astype(np.float32)
    person = np.stack(
        [np.full(3000, 3.0), rng.uniform(-0.6, 0.6, 3000), rng.uniform(-0.6, 0.6, 3000)], 1
    ).astype(np.float32)
    mp = np.vstack([wall, person])
    scan = wall
    o = np.zeros(3)

    _, mk_filter = f.filter(mp, scan, o)  # host path's map_keep (bool)

    # Device-native carve must agree with filter()'s map_keep.
    mp_wp = wp.array(mp, dtype=wp.vec3, device=dev)
    scan_wp = wp.array(scan, dtype=wp.vec3, device=dev)
    mk_dev = f.carve(mp_wp, scan_wp, o)  # device in → device mask out
    assert isinstance(mk_dev, wp.array) and mk_dev.device == dev
    assert np.array_equal(
        mk_dev.numpy().astype(bool), mk_filter
    ), "device carve disagrees w/ filter"


def _voxel_cells(points: np.ndarray, center: tuple[float, float]) -> set[int]:
    cx, cy = center
    mn = np.array([cx - RADIUS, cy - RADIUS, Z0])
    dx = int(2 * RADIUS / VOX) + 1
    dz = int((Z1 - Z0) / VOX) + 1
    idx = ((points - mn) / VOX).astype(int)
    ok = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < dx)
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < dx)
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < dz)
    )
    idx = idx[ok]
    return set(((idx[:, 0] * dx + idx[:, 1]) * dz + idx[:, 2]).tolist())


def test_accumulator_matches_numpy() -> None:
    dev = wp.get_device("cpu")
    acc = DeviceMapAccumulator(VOX, RADIUS, z_bounds=(Z0, Z1), device=dev)
    rng = np.random.default_rng(1)
    mapp = rng.uniform([-10, -10, 0], [10, 10, 3], (2000, 3)).astype(np.float32)
    carve = (rng.random(2000) > 0.3).astype(np.int32)
    pts = rng.uniform([-30, -30, 0], [30, 30, 3], (3000, 3)).astype(np.float32)
    valid = (rng.random(3000) > 0.1).astype(np.int32)
    center = (1.0, -2.0)

    new = acc.step(
        wp.array(mapp, dtype=wp.vec3, device=dev),
        wp.array(carve, dtype=wp.int32, device=dev),
        wp.array(pts, dtype=wp.vec3, device=dev),
        wp.array(valid, dtype=wp.int32, device=dev),
        center,
    )
    dev_cells = _voxel_cells(new.numpy(), center)

    # Numpy reference: carve → crop to radius → voxel on the same fixed grid.
    survivors = np.vstack([mapp[carve.astype(bool)], pts[valid.astype(bool)]])
    inr = (survivors[:, 0] - center[0]) ** 2 + (survivors[:, 1] - center[1]) ** 2 <= RADIUS**2
    ref_cells = _voxel_cells(survivors[inr], center)

    assert dev_cells == ref_cells, "device accumulator occupies different voxels than numpy"


def main() -> None:
    wp.init()
    test_carve_matches_filter_and_device()
    test_accumulator_matches_numpy()
    print("PASS: device-native carve + accumulator match numpy (2 tests)")


if __name__ == "__main__":
    main()
