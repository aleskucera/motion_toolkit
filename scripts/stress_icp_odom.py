"""Stress-test the accumulating mapper against noisy odometry.

Mirrors `terrain_accumulator_node._process` 1:1 (predict from the odom delta →
scan-to-submap ICP with the divergence gate → accumulate) but headless: it drives
a known ground-truth trajectory through a structured scene, simulates LiDAR scans,
and corrupts the odometry with per-step delta noise. Because the node integrates
odom *deltas*, only per-step noise matters — absolute odom drift is designed out.

The estimated world frame is bootstrapped to odom[0] = ground-truth[0], so the
estimated trajectory is directly comparable to ground truth. The map is just the
union of scans placed by the estimated poses, so trajectory error *is* map smear:
ATE (absolute trajectory error) is our map-quality metric.

Run: python scripts/stress_icp_odom.py
"""

from __future__ import annotations

import numpy as np
import warp as wp
from helhest.perception import DeviceMapAccumulator
from helhest.perception import IcpAligner
from helhest.perception import IcpConfig
from helhest.localization import Localizer
from helhest.localization import LocalizerConfig
from helhest.perception import transform_points
from helhest.localization.pose_math import invert_pose
from helhest.localization.pose_math import pose_correction_magnitude
from helhest.localization.pose_math import transform_points_xyz
from helhest.perception.sim import GroundSpec
from helhest.perception.sim import PrimitiveLidar

# Pipeline constants matching the node defaults.
ACC_VOXEL_M = 0.10
MAP_MAX_RADIUS_M = 50.0
ICP_SUBMAP_RADIUS_M = 15.0
GATE_MIN_INLIERS = 500
GATE_MAX_TRANS_M = 1.0
GATE_MAX_ROT_RAD = float(np.deg2rad(15.0))
GATE_MIN_SUBMAP = 2000

SENSOR_HEIGHT_M = 0.5


# ----------------------------------------------------------------------------
# SE(3) helpers for building ground truth and injecting noise
# ----------------------------------------------------------------------------


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=np.float64)


def _so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation from an axis-angle vector (Rodrigues)."""
    theta = float(np.linalg.norm(w))
    if theta < 1.0e-9:
        return np.eye(3) + _skew(w)
    k = w / theta
    K = _skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _planar_pose(x: float, y: float, yaw: float, z: float) -> np.ndarray:
    """SE(3) pose for a ground robot: yaw about +z at height z."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = (x, y, z)
    return T


def _noisy_delta(
    gt_delta: np.ndarray, rot_std_rad: float, trans_std_m: float, rng: np.random.Generator
) -> np.ndarray:
    """Ground-truth delta right-perturbed by a small SE(3) noise (all 6 DOF)."""
    perturb = np.eye(4)
    perturb[:3, :3] = _so3_exp(rng.normal(0.0, rot_std_rad, 3))
    perturb[:3, 3] = rng.normal(0.0, trans_std_m, 3)
    return gt_delta @ perturb


# ----------------------------------------------------------------------------
# Scene + sensor
# ----------------------------------------------------------------------------


def _ring_directions(
    channels: int, az_count: int, el_min_deg: float, el_max_deg: float
) -> np.ndarray:
    """Spinning-LiDAR beam directions (local frame): channels x azimuth, full 360°."""
    els = np.deg2rad(np.linspace(el_min_deg, el_max_deg, channels))
    azs = np.linspace(-np.pi, np.pi, az_count, endpoint=False)
    el_grid, az_grid = np.meshgrid(els, azs, indexing="ij")
    d = np.stack(
        [np.cos(el_grid) * np.cos(az_grid), np.cos(el_grid) * np.sin(az_grid), np.sin(el_grid)],
        axis=-1,
    )
    return d.reshape(-1, 3).astype(np.float32)


def _scene_boxes() -> tuple[np.ndarray, np.ndarray]:
    """Vertical structure (pillars + walls) so point-to-plane ICP constrains x/y/yaw.

    A bare ground plane under-constrains the horizontal DOF; these give the beams
    something to lock onto. Spread across the corridor the robot drives through.
    """
    boxes: list[tuple[float, float, float, float, float, float]] = []
    # Pillars scattered beside the path.
    for x in np.arange(-2.0, 18.0, 3.0):
        for y in (-4.5, 4.5):
            boxes.append((x - 0.25, y - 0.25, 0.0, x + 0.25, y + 0.25, 2.5))
    # Two long walls flanking the corridor.
    boxes.append((-3.0, -6.2, 0.0, 19.0, -5.8, 1.8))
    boxes.append((-3.0, 5.8, 0.0, 19.0, 6.2, 1.8))
    arr = np.array(boxes, dtype=np.float32)
    return arr[:, :3].copy(), arr[:, 3:].copy()


def _ground_truth_trajectory(n_frames: int) -> list[np.ndarray]:
    """A gentle S-curve drive: forward with a lateral wiggle, heading follows motion."""
    poses = []
    ts = np.linspace(0.0, 1.0, n_frames)
    xs = 15.0 * ts
    ys = 2.0 * np.sin(2.0 * np.pi * ts)
    for i, t in enumerate(ts):
        dx = 15.0
        dy = 2.0 * 2.0 * np.pi * np.cos(2.0 * np.pi * t)
        yaw = np.arctan2(dy, dx)
        poses.append(_planar_pose(float(xs[i]), float(ys[i]), float(yaw), SENSOR_HEIGHT_M))
    return poses


# ----------------------------------------------------------------------------
# The node loop (headless driver over the shared Localizer)
# ----------------------------------------------------------------------------


def run_pipeline(
    scans_base: list[np.ndarray],
    odom: list[np.ndarray],
    gt: list[np.ndarray],
    aligner: IcpAligner | None,
    device: wp.context.Device,
) -> dict[str, object]:
    """Drive the accumulating-mapper loop; return per-frame errors + gate stats.

    `aligner=None` runs pure odom dead-reckoning (ICP disabled). Uses the same
    Localizer the ROS node ships, so the harness exercises the shipped gate.
    """
    localizer = Localizer(
        aligner,
        LocalizerConfig(
            enable=aligner is not None,
            submap_radius_m=ICP_SUBMAP_RADIUS_M,
            min_submap_points=GATE_MIN_SUBMAP,
            min_inliers=GATE_MIN_INLIERS,
            max_correction_trans_m=GATE_MAX_TRANS_M,
            max_correction_rot_rad=GATE_MAX_ROT_RAD,
        ),
    )
    # Device-native map: the accumulator carves+adds+crops+voxel-thins on device and
    # keeps the map as a wp.array — no host round trip. z_bounds wide so the world-z
    # box never clips the scene (sparse hash → unused z range costs nothing).
    acc = DeviceMapAccumulator(
        ACC_VOXEL_M, MAP_MAX_RADIUS_M, z_bounds=(-100.0, 100.0), device=device
    )
    map_wp: wp.array | None = None
    pos_err: list[float] = []
    rot_err: list[float] = []
    n_reject = 0
    n_sparse = 0

    for k in range(len(scans_base)):
        n = len(scans_base[k])
        scan_base_wp = wp.array(scans_base[k], dtype=wp.vec3, device=device)  # single upload
        if not localizer.initialized:
            world_T_base = odom[k]  # bootstrap world ≡ odom (= gt[0])
            localizer.bootstrap(odom[k], world_T_base)
        else:
            world_T_base_pred, _ = localizer.predict(odom[k])
            outcome = localizer.update(scan_base_wp, world_T_base_pred, map_wp, odom[k])
            world_T_base = outcome.pose
            n_reject += outcome.status == "rejected"
            n_sparse += outcome.status == "sparse"

        world_pts = transform_points(scan_base_wp, n, world_T_base)
        valid = wp.full(n, 1, dtype=wp.int32, device=device)
        map_wp = acc.step(map_wp, None, world_pts, valid, (world_T_base[0, 3], world_T_base[1, 3]))

        rot, trans = pose_correction_magnitude(gt[k], world_T_base)
        pos_err.append(trans)
        rot_err.append(np.rad2deg(rot))

    pos = np.array(pos_err)
    rot = np.array(rot_err)
    return {
        "ate_pos": float(np.sqrt(np.mean(pos**2))),
        "final_pos": float(pos[-1]),
        "ate_rot": float(np.sqrt(np.mean(rot**2))),
        "final_rot": float(rot[-1]),
        "n_reject": n_reject,
        "n_sparse": n_sparse,
        "diverged_at": None,
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def _build_odom(
    gt: list[np.ndarray], rot_std_deg: float, trans_std_m: float, seed: int
) -> list[np.ndarray]:
    """Integrate a drifting odom stream whose per-step deltas are gt ⊕ noise."""
    rng = np.random.default_rng(seed)
    rot_std = np.deg2rad(rot_std_deg)
    odom = [gt[0]]  # absolute start matches gt so the world frame lines up
    for k in range(1, len(gt)):
        gt_delta = invert_pose(gt[k - 1]) @ gt[k]
        odom.append(odom[-1] @ _noisy_delta(gt_delta, rot_std, trans_std_m, rng))
    return odom


def _make_scans(
    lidar: PrimitiveLidar,
    gt: list[np.ndarray],
    boxes_lo: np.ndarray,
    boxes_hi: np.ndarray,
) -> list[np.ndarray]:
    """One scan per gt pose, returned in the BASE frame (as the node's TF stage yields)."""
    scans = []
    for k, pose in enumerate(gt):
        origin = pose[:3, 3]
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        world_hits = lidar.scan(origin, yaw, boxes_lo, boxes_hi, seed=1000 + k)
        scans.append(transform_points_xyz(invert_pose(pose), world_hits).astype(np.float32))
    return scans


def main() -> None:
    wp.init()
    device = wp.get_device()
    print(f"device: {device}\n")

    n_frames = 60
    gt = _ground_truth_trajectory(n_frames)
    boxes_lo, boxes_hi = _scene_boxes()
    directions = _ring_directions(channels=32, az_count=720, el_min_deg=-25.0, el_max_deg=12.0)
    lidar = PrimitiveLidar(
        directions,
        ground=GroundSpec(z=0.0, x_range=(-12.0, 26.0), y_range=(-16.0, 16.0)),
        noise_std=0.01,  # 1 cm range noise — realistic sensor floor
        min_range=0.4,
        max_range=45.0,
        device=device,
    )
    scans_base = _make_scans(lidar, gt, boxes_lo, boxes_hi)
    print(f"scene: {len(boxes_lo)} boxes, {n_frames} frames, ~{scans_base[0].shape[0]} pts/scan\n")

    icp_cfg = IcpConfig(
        max_iters=30,
        max_correspondence_dist_m=0.5,
        normal_radius_m=0.3,
        voxel_size_m=0.1,
        voxel_target=True,
    )
    aligner = IcpAligner(icp_cfg, device=device)

    # Per-step odom noise levels (std of the delta perturbation).
    levels = [
        ("clean", 0.0, 0.0),
        ("light", 0.5, 0.01),
        ("moderate", 1.5, 0.03),
        ("heavy", 3.0, 0.07),
        ("severe", 6.0, 0.15),
    ]

    header = (
        f"{'noise':>9} {'rot°/step':>9} {'trans/step':>11} │ "
        f"{'ATE pos (m)':>22} │ {'ATE rot (deg)':>22} │ {'ICP rej':>7}"
    )
    sub = f"{'':>9} {'':>9} {'':>11} │ {'odom-only':>10} {'ICP':>11} │ {'odom-only':>10} {'ICP':>11} │"
    print(header)
    print(sub)
    print("─" * len(header))

    def _fmt(x: float) -> str:
        return "diverged" if not np.isfinite(x) else f"{x:.3f}"

    for name, rot_std_deg, trans_std_m in levels:
        odom = _build_odom(gt, rot_std_deg, trans_std_m, seed=7)
        off = run_pipeline(scans_base, odom, gt, aligner=None, device=device)
        on = run_pipeline(scans_base, odom, gt, aligner=aligner, device=device)
        print(
            f"{name:>9} {rot_std_deg:>9.1f} {trans_std_m*100:>9.1f}cm │ "
            f"{_fmt(off['ate_pos']):>10} {_fmt(on['ate_pos']):>11} │ "
            f"{off['ate_rot']:>10.2f} {on['ate_rot']:>11.2f} │ "
            f"{on['n_reject']:>4d}/{n_frames-1:<2d}"
        )

    # Gross-glitch test: a single large odom jump — does the gate catch it?
    print("\ngross-glitch test (one bad odom step at frame 30, moderate baseline noise):")
    odom = _build_odom(gt, 1.5, 0.03, seed=7)
    glitch = np.eye(4)
    glitch[0, 3] = 1.6  # 1.6 m jump, beyond the 1.0 m gate
    for k in range(31, len(odom)):  # shift the rest of the stream by the glitch
        odom[k] = glitch @ odom[k]
    off = run_pipeline(scans_base, odom, gt, aligner=None, device=device)
    on = run_pipeline(scans_base, odom, gt, aligner=aligner, device=device)
    print(f"  odom-only : final drift {off['final_pos']:.2f} m  (glitch is never corrected)")
    print(
        f"  ICP       : final drift {on['final_pos']:.2f} m  "
        f"({on['n_reject']} rejections — gate caught the jump)"
    )


if __name__ == "__main__":
    main()
