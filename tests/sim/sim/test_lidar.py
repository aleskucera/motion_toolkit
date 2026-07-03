"""Warp primitive-LiDAR kernel vs a numpy reference (noise/dropout off).

Casts the same beam grid against the same ground plane + box obstacles two ways
and checks the surviving hits agree: the valid masks match on virtually every
beam and the hit distances line up. Guards the kernel's intersection math (ground
plane + multi-box slab + yaw rotation) against regressions.

Run: python tests/sim/test_lidar.py   (CPU)
"""

from __future__ import annotations

import numpy as np
import warp as wp
from helhest.perception.sim import GroundSpec
from helhest.perception.sim import PrimitiveLidar

ORIGIN = np.array([0.3, -0.4, 0.5])
YAW = 0.6  # exercise the in-kernel beam rotation
GROUND = GroundSpec(z=0.0, x_range=(-12.0, 12.0), y_range=(-12.0, 12.0))
# A thin "wall" box and a "person" box.
BOXES_LO = np.array([[10.0, -6.0, 0.0], [4.75, -0.25, 0.0]])
BOXES_HI = np.array([[10.1, 6.0, 2.0], [5.25, 0.25, 1.8]])


def _beam_dirs() -> np.ndarray:
    az = np.linspace(np.deg2rad(-80.0), np.deg2rad(80.0), 200)
    el = np.linspace(np.deg2rad(-30.0), np.deg2rad(20.0), 60)
    az, el = np.meshgrid(az, el)
    az, el = az.ravel(), el.ravel()
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)


def _rotate_yaw(dirs: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return dirs @ R.T


def _ref_t(dirs: np.ndarray) -> np.ndarray:
    """Numpy nearest-hit distance per beam (inf on miss), dirs in world frame."""
    # Ground plane z = 0 over its xy window.
    dz = dirs[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_g = (GROUND.z - ORIGIN[2]) / dz
    t_g = np.where((np.abs(dz) > 1e-9) & (t_g > 1e-3), t_g, np.inf)
    p = ORIGIN + t_g[:, None] * dirs
    bad = (
        (p[:, 0] < GROUND.x_range[0])
        | (p[:, 0] > GROUND.x_range[1])
        | (p[:, 1] < GROUND.y_range[0])
        | (p[:, 1] > GROUND.y_range[1])
    )
    t = np.where(bad, np.inf, t_g)

    inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    for lo, hi in zip(BOXES_LO, BOXES_HI):
        t1 = (lo - ORIGIN) * inv
        t2 = (hi - ORIGIN) * inv
        t_near = np.max(np.minimum(t1, t2), axis=1)
        t_far = np.min(np.maximum(t1, t2), axis=1)
        hit = (t_far >= np.maximum(t_near, 0.0)) & (t_far > 1e-3)
        t_box = np.where(hit, np.where(t_near > 1e-3, t_near, t_far), np.inf)
        t = np.minimum(t, t_box)
    return t


def main() -> None:
    wp.init()
    local = _beam_dirs()

    lidar = PrimitiveLidar(
        local, ground=GROUND, noise_std=0.0, dropout=0.0, device=wp.get_device("cpu")
    )
    # Kernel writes a point per beam; read the valid mask directly for a per-beam
    # comparison (bypass scan()'s compaction so indices line up with the reference).
    lidar.scan(ORIGIN, YAW, BOXES_LO, BOXES_HI, seed=0)
    valid_wp = lidar._out_valid.numpy().astype(bool)
    t_wp = np.linalg.norm(lidar._out_pts.numpy() - ORIGIN, axis=1)  # unit beams → range

    t_ref = _ref_t(_rotate_yaw(local, YAW))
    valid_ref = np.isfinite(t_ref)

    agree = (valid_wp == valid_ref).mean()
    assert agree > 0.999, f"valid-mask agreement only {agree:.4f}"

    both = valid_wp & valid_ref
    max_dt = float(np.abs(t_wp[both] - t_ref[both]).max())
    assert max_dt < 1e-2, f"hit-distance mismatch up to {max_dt:.4f} m"

    # Max-range clamp: every returned point must lie within max_range (+ noise=0).
    near = PrimitiveLidar(
        local,
        ground=GROUND,
        noise_std=0.0,
        dropout=0.0,
        max_range=5.0,
        device=wp.get_device("cpu"),
    )
    pts = near.scan(ORIGIN, YAW, BOXES_LO, BOXES_HI, seed=0)
    assert len(pts) > 0
    assert float(np.linalg.norm(pts - ORIGIN, axis=1).max()) <= 5.0 + 1e-4

    # Min-range cull: nothing closer than min_range comes back (noise off).
    far = PrimitiveLidar(
        local,
        ground=GROUND,
        noise_std=0.0,
        dropout=0.0,
        min_range=6.0,
        device=wp.get_device("cpu"),
    )
    pts = far.scan(ORIGIN, YAW, BOXES_LO, BOXES_HI, seed=0)
    assert len(pts) > 0
    assert float(np.linalg.norm(pts - ORIGIN, axis=1).min()) >= 6.0 - 1e-4

    print(
        f"PASS: warp LiDAR — {int(valid_ref.sum())}/{len(local)} beams hit, "
        f"mask agreement {agree:.4f}, max |Δrange| {max_dt * 1e3:.2f} mm"
    )


if __name__ == "__main__":
    main()
