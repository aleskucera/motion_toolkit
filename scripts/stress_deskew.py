"""Quantify LiDAR motion skew and validate `deskew_scan`.

A spinning sensor (the OSDome) measures each azimuth at a different instant, so a
moving robot smears one sweep across its own motion. This drives a single sweep
while the robot translates + yaws, builds the physically-skewed cloud (cast each
azimuth wedge from the sensor pose at that wedge's time), then checks that
`pose_math.deskew_scan` puts the geometry back where an instantaneous scan
would. Metric: how far each pillar's centroid lands from the clean-scan centroid.

Run: python scripts/stress_deskew.py
"""

from __future__ import annotations

import numpy as np
import warp as wp
from terrain_toolkit.localization.pose_math import deskew_scan
from terrain_toolkit.localization.pose_math import invert_pose
from terrain_toolkit.localization.pose_math import transform_points_xyz
from terrain_toolkit.sim import GroundSpec
from terrain_toolkit.sim import PrimitiveLidar

SWEEP_PERIOD_S = 0.1  # OSDome at 10 Hz — the worst-case sweep window
SENSOR_HEIGHT_M = 0.5
PILLAR_RADIUS_M = 8.0
N_PILLARS = 24
N_WEDGES = 72  # azimuth slices per sweep (5° each) — finer = smoother skew sim


def _planar_pose(x: float, y: float, yaw: float, z: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = (x, y, z)
    return T


def _yaw_of(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def _sweep_pose(v: float, omega: float, alpha: float) -> np.ndarray:
    """Ground-truth base pose at sweep fraction `alpha` (0 = start, 1 = end)."""
    return _planar_pose(
        alpha * v * SWEEP_PERIOD_S, 0.0, alpha * omega * SWEEP_PERIOD_S, SENSOR_HEIGHT_M
    )


def _ring_directions() -> np.ndarray:
    els = np.deg2rad(np.linspace(-20.0, 18.0, 40))
    azs = np.linspace(-np.pi, np.pi, 1024, endpoint=False)
    el_grid, az_grid = np.meshgrid(els, azs, indexing="ij")
    d = np.stack(
        [np.cos(el_grid) * np.cos(az_grid), np.cos(el_grid) * np.sin(az_grid), np.sin(el_grid)],
        axis=-1,
    )
    return d.reshape(-1, 3).astype(np.float32)


def _pillar_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A ring of vertical pillars at every azimuth, so the whole sweep sees structure."""
    angles = np.linspace(0.0, 2.0 * np.pi, N_PILLARS, endpoint=False)
    centers = np.stack([PILLAR_RADIUS_M * np.cos(angles), PILLAR_RADIUS_M * np.sin(angles)], axis=1)
    lo = np.stack([centers[:, 0] - 0.15, centers[:, 1] - 0.15, np.zeros(N_PILLARS)], axis=1)
    hi = np.stack([centers[:, 0] + 0.15, centers[:, 1] + 0.15, np.full(N_PILLARS, 3.0)], axis=1)
    return lo.astype(np.float32), hi.astype(np.float32), centers


def _scan_world(lidar: PrimitiveLidar, pose: np.ndarray, blo, bhi, seed: int) -> np.ndarray:
    """Full scan from `pose`, returned per-beam in WORLD (index-aligned) with a valid mask."""
    origin = pose[:3, 3]
    pts, valid, _ = lidar.scan(origin, _yaw_of(pose), blo, bhi, seed=seed, return_device=True)
    return pts.numpy().copy(), valid.numpy().astype(bool)


def build_swept_scan(
    lidar: PrimitiveLidar,
    beam_alpha: np.ndarray,
    beam_wedge: np.ndarray,
    v: float,
    omega: float,
    blo,
    bhi,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Physically-skewed sweep in the sweep-END base frame, plus per-point alpha.

    Wedge k (its azimuth band) is cast from the sensor pose at its time alpha_k,
    then expressed in that instant's base frame — exactly what the sensor reports.
    Stacking the wedges and treating them as one cloud IS the skew.
    """
    ref_pose = _sweep_pose(v, omega, 1.0)
    pts_base: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    for k in range(N_WEDGES):
        alpha_k = (k + 0.5) / N_WEDGES
        pose_k = _sweep_pose(v, omega, alpha_k)
        world, valid = _scan_world(lidar, pose_k, blo, bhi, seed)
        sel = valid & (beam_wedge == k)
        hits = world[sel]
        # Express in the base frame at the instant of measurement (what the driver ships).
        pts_base.append(transform_points_xyz(invert_pose(pose_k), hits))
        alphas.append(np.full(hits.shape[0], alpha_k))
    return np.vstack(pts_base), np.concatenate(alphas)


def _pillar_centroids(
    cloud_base: np.ndarray, ref_pose: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Per-pillar xy centroid (in world) of a base-frame cloud; NaN if a pillar is unseen."""
    world = transform_points_xyz(ref_pose, cloud_base)
    body = world[(world[:, 2] > 0.3) & (world[:, 2] < 2.8)]  # pillar body, drop ground
    out = np.full((len(centers), 2), np.nan)
    for i, c in enumerate(centers):
        near = body[np.hypot(body[:, 0] - c[0], body[:, 1] - c[1]) < 1.5]
        if len(near) >= 10:
            out[i] = near[:, :2].mean(axis=0)
    return out


def _centroid_error(cloud_base, ref_pose, centers, baseline: np.ndarray) -> float:
    """Mean shift (m) of pillar centroids vs the clean-scan baseline."""
    cent = _pillar_centroids(cloud_base, ref_pose, centers)
    shift = np.linalg.norm(cent - baseline, axis=1)
    return float(np.nanmean(shift))


def main() -> None:
    wp.init()
    device = wp.get_device()
    print(f"device: {device}   sweep period: {SWEEP_PERIOD_S*1000:.0f} ms\n")

    blo, bhi, centers = _pillar_scene()
    directions = _ring_directions()
    beam_az = np.arctan2(directions[:, 1], directions[:, 0])
    beam_alpha = (beam_az + np.pi) / (2.0 * np.pi)  # azimuth ↔ sweep time
    beam_wedge = np.clip((beam_alpha * N_WEDGES).astype(int), 0, N_WEDGES - 1)

    lidar = PrimitiveLidar(
        directions,
        ground=GroundSpec(z=0.0, x_range=(-15.0, 15.0), y_range=(-15.0, 15.0)),
        noise_std=0.01,
        min_range=0.4,
        max_range=45.0,
        device=device,
    )

    # Realistic per-sweep odom error (0.1 s of drift): deskew uses odom, not truth.
    odom_rng = np.random.default_rng(3)

    levels = [
        ("static", 0.0, 0.0),
        ("walk", 1.0, 0.3),
        ("brisk", 2.0, 0.6),
        ("fast", 4.0, 1.2),
    ]

    header = (
        f"{'motion':>8} {'v (m/s)':>8} {'ω(°/s)':>8} │ {'sweep move':>10} │ "
        f"{'skew err':>9} {'deskew':>9} {'deskew+odom':>12}"
    )
    print(header)
    print("─" * len(header))
    for name, v, omega in levels:
        ref_pose = _sweep_pose(v, omega, 1.0)
        seed = 42

        clean_world, clean_valid = _scan_world(lidar, ref_pose, blo, bhi, seed)
        clean_base = transform_points_xyz(invert_pose(ref_pose), clean_world[clean_valid])
        baseline = _pillar_centroids(clean_base, ref_pose, centers)

        skewed, alphas = build_swept_scan(lidar, beam_alpha, beam_wedge, v, omega, blo, bhi, seed)

        # True sweep motion, and a noisy odom estimate of it.
        sweep_delta = invert_pose(_sweep_pose(v, omega, 0.0)) @ ref_pose
        noisy = sweep_delta.copy()
        noisy[:3, 3] += odom_rng.normal(0.0, 0.01, 3)  # ~1 cm/ sweep translation error
        d_ang = np.deg2rad(0.3)
        cz, sz = np.cos(d_ang), np.sin(d_ang)
        noisy[:3, :3] = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ noisy[:3, :3]

        desk = deskew_scan(skewed, alphas, sweep_delta)
        desk_odom = deskew_scan(skewed, alphas, noisy)

        e_skew = _centroid_error(skewed, ref_pose, centers, baseline)
        e_desk = _centroid_error(desk, ref_pose, centers, baseline)
        e_odom = _centroid_error(desk_odom, ref_pose, centers, baseline)
        sweep_move = np.hypot(v * SWEEP_PERIOD_S, PILLAR_RADIUS_M * omega * SWEEP_PERIOD_S)
        print(
            f"{name:>8} {v:>8.1f} {np.rad2deg(omega):>8.1f} │ {sweep_move*100:>8.1f}cm │ "
            f"{e_skew*100:>7.2f}cm {e_desk*100:>7.2f}cm {e_odom*100:>10.2f}cm"
        )

    print("\nskew err  = raw swept cloud vs clean instantaneous scan (per-pillar centroid shift)")
    print("deskew    = deskew_scan with the true sweep motion")
    print("deskew+odom = deskew_scan with a noisy (±1 cm, ±0.3°) odom estimate of the motion")


if __name__ == "__main__":
    main()
