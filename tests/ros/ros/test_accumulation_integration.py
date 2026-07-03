"""Integration guard for the accumulator's robot-centric window math.

The accumulator builds the heightmap in a frame shifted by the FULL robot
translation (so the pipeline's z_max stays robot-relative), then republishes at
the true world position via a grid origin offset + a z_offset on elevation. This
test drops an asymmetric spike at a known WORLD (x, y, z), shifts the cloud by a
nonzero robot translation, runs the real pipeline on the shifted cloud, then
reconstructs world coordinates with the SAME formula `grid_to_cloud` uses and
asserts the spike reappears at the correct world (x, y, z).

`grid_to_cloud` itself isn't called (it imports sensor_msgs/rclpy, unavailable
off-robot); the reconstruction below mirrors its arithmetic exactly:
    x_world = x_min + (col + 0.5) * resolution      # x_min = robot_x - x_range
    y_world = y_min + (row + 0.5) * resolution      # y_min = robot_y - y_range
    z_world = elevation + z_offset                  # z_offset = robot_z

Run: python tests/ros/test_accumulation_integration.py   (CPU; no outlier filter)
"""

from __future__ import annotations

import numpy as np
import warp as wp
from terrain_toolkit import TerrainPipeline

X_RANGE = 4.0  # window half-extent in x -> nx = 80
Y_RANGE = 2.0  # window half-extent in y -> ny = 40 (asymmetric: catches a transpose)
RESOLUTION = 0.1
ROBOT_T = np.array([10.0, -5.0, 1.5])  # nonzero in all three axes
SPIKE_WORLD = (12.05, -4.45, 2.0)  # within the window; x-x0 != y-y0, off-diagonal


def _cloud_world() -> np.ndarray:
    rng = np.random.default_rng(0)
    x0, y0 = ROBOT_T[0] - X_RANGE, ROBOT_T[1] - Y_RANGE
    gx = rng.uniform(x0, x0 + 2 * X_RANGE, 80_000)
    gy = rng.uniform(y0, y0 + 2 * Y_RANGE, 80_000)
    ground = np.stack([gx, gy, np.full_like(gx, ROBOT_T[2])], axis=1)  # flat at robot height
    sx = rng.uniform(SPIKE_WORLD[0] - 0.05, SPIKE_WORLD[0] + 0.05, 4_000)
    sy = rng.uniform(SPIKE_WORLD[1] - 0.05, SPIKE_WORLD[1] + 0.05, 4_000)
    spike = np.stack([sx, sy, np.full_like(sx, SPIKE_WORLD[2])], axis=1)
    return np.concatenate([ground, spike], axis=0).astype(np.float64)


def main() -> None:
    wp.init()
    pipe = TerrainPipeline(
        resolution=RESOLUTION,
        bounds=(-X_RANGE, X_RANGE, -Y_RANGE, Y_RANGE),
        z_max=1.0,  # robot-relative: ground (z-robot=0) and spike (z-robot=0.5) both pass
        primary="max",
        inpaint=True,
        smooth_sigma=0.0,
        device="cpu",
    )

    # Shift by the FULL robot translation, exactly as the accumulator does.
    window_local = _cloud_world() - ROBOT_T
    tm = pipe.process(window_local)
    elev = tm.elevation
    assert elev is not None
    assert elev.shape == (40, 80), elev.shape  # (ny, nx); a transpose would be (80, 40)

    # Reconstruct the spike's world position (mirrors grid_to_cloud).
    x_min = ROBOT_T[0] - X_RANGE
    y_min = ROBOT_T[1] - Y_RANGE
    r, c = np.unravel_index(int(np.argmax(elev)), elev.shape)
    x_world = x_min + (c + 0.5) * RESOLUTION
    y_world = y_min + (r + 0.5) * RESOLUTION
    z_world = float(elev[r, c]) + ROBOT_T[2]  # z_offset = robot_z

    assert abs(x_world - SPIKE_WORLD[0]) <= RESOLUTION, (x_world, SPIKE_WORLD[0])
    assert abs(y_world - SPIKE_WORLD[1]) <= RESOLUTION, (y_world, SPIKE_WORLD[1])
    assert abs(z_world - SPIKE_WORLD[2]) <= 0.1, (z_world, SPIKE_WORLD[2])

    print(
        f"PASS: spike world {SPIKE_WORLD} -> reconstructed "
        f"({x_world:.2f}, {y_world:.2f}, {z_world:.2f})"
    )


if __name__ == "__main__":
    main()
