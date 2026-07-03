"""Reconstruct the free-space frontier from a real (organized) LiDAR scan.

The sim lidar emits a per-beam free-space frontier directly (`out_free`). On the
robot the driver gives an *organized* cloud instead — one cell per beam, with a
zero point where the beam returned nothing — so we rebuild the same frontier:
the surface hit where present, else the max-range point along that beam. Feeding
this to `DynamicPointFilter.carve` gives ray-carving that uses no-return beams as
free-space evidence (so it removes points with no background behind them).

Only works for organized sensors with a fixed beam grid (e.g. Ouster). A
non-repetitive sensor (Livox) has no per-beam miss information, so there's no
frontier to reconstruct — carving there degrades to the visibility filter.
"""

from __future__ import annotations

import numpy as np


def frontier_from_organized(
    points_xyz: np.ndarray,
    beam_dirs: np.ndarray,
    max_range: float,
    *,
    min_range: float = 1.0e-3,
) -> np.ndarray:
    """Per-beam free-space frontier (B, 3), in the sensor frame.

    `points_xyz` (B, 3) is the organized scan in the sensor frame, one row per
    beam, with a **miss** encoded as a ~zero point (range ≤ `min_range`).
    `beam_dirs` (B, 3) are the matching unit beam directions (same beam order).
    A hit keeps its measured point; a miss becomes `max_range · beam_dir`.
    """
    if points_xyz.shape != beam_dirs.shape or points_xyz.shape[1:] != (3,):
        raise ValueError(
            f"points_xyz and beam_dirs must be matching (B, 3); "
            f"got {points_xyz.shape} and {beam_dirs.shape}"
        )
    ranges = np.linalg.norm(points_xyz, axis=1)
    hit = ranges > min_range
    frontier = beam_dirs.astype(np.float32) * np.float32(max_range)
    frontier[hit] = points_xyz[hit]
    return frontier.astype(np.float32, copy=False)
