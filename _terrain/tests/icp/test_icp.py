"""ICP sanity test: generate two overlapping clouds with a known pose offset,
add an odometry-like perturbation, and check that ICP recovers the true pose."""

import argparse

import numpy as np
import warp as wp
from terrain_toolkit import IcpAligner
from terrain_toolkit import IcpConfig


def make_scene_cloud(n: int, seed: int) -> np.ndarray:
    """Tilted plane + Gaussian bump + a vertical wall slab — mixed geometry."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n)
    y = rng.uniform(-5, 5, n)
    z = 0.1 * x + 0.05 * y + 1.5 * np.exp(-((x - 1) ** 2 + (y + 1) ** 2) / 1.5)
    z += rng.normal(0.0, 0.01, n)

    # Add a vertical slab to constrain rotation.
    n_wall = n // 5
    wx = rng.uniform(2.0, 2.2, n_wall)
    wy = rng.uniform(-3.0, 3.0, n_wall)
    wz = rng.uniform(0.0, 1.5, n_wall)
    wall = np.stack([wx, wy, wz], axis=1)

    pts = np.concatenate([np.stack([x, y, z], axis=1), wall], axis=0)
    return pts.astype(np.float32)


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pose_diff(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Return (rotation error [rad], translation error [m]) between two SE(3) poses."""
    R_err = T_a[:3, :3].T @ T_b[:3, :3]
    tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot = float(np.arccos(tr))
    trans = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    return rot, trans


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Ground-truth transform that maps source → target.
    R_gt = rpy_to_R(np.deg2rad(3.0), np.deg2rad(-2.0), np.deg2rad(5.0))
    t_gt = np.array([0.25, -0.15, 0.05])
    T_gt = pose(R_gt, t_gt)
    print(f"Ground truth: rpy=[3, -2, 5] deg, t={t_gt}")

    # Target: scene in world frame.
    target = make_scene_cloud(args.n, seed=args.seed)

    # Source: same scene observed from a shifted/rotated sensor. The transform
    # that maps source points into target frame is T_gt. So source = T_gt^-1 · target.
    R_inv = R_gt.T
    t_inv = -R_inv @ t_gt
    source = (R_inv @ target.T).T + t_inv
    source = source.astype(np.float32)
    # Add measurement noise to source.
    rng = np.random.default_rng(args.seed + 1)
    source += rng.normal(0.0, 0.005, source.shape).astype(np.float32)

    # Noisy odometry prior — off by ~2 deg and ~10 cm.
    R_noise = rpy_to_R(np.deg2rad(1.5), np.deg2rad(-1.0), np.deg2rad(3.0))
    t_noise = np.array([0.15, -0.08, 0.02])
    T_init = pose(R_noise, t_noise)
    rot_err, trans_err = pose_diff(T_init, T_gt)
    print(f"Initial guess error: rot={np.rad2deg(rot_err):.2f} deg  trans={trans_err:.3f} m")

    aligner = IcpAligner(IcpConfig(max_iters=args.max_iters), verbose=args.verbose)
    # Upload the clouds once; the aligner is device-native from here.
    source_wp = wp.array(source, dtype=wp.vec3, device=aligner.device)
    target_wp = wp.array(target, dtype=wp.vec3, device=aligner.device)
    result = aligner.align(source_wp, target_wp, init_pose=T_init)

    rot_err, trans_err = pose_diff(result.pose, T_gt)
    print(
        f"ICP result: iters={result.iterations}  inliers={result.num_inliers}  "
        f"cost={result.final_cost:.4f}  converged={result.converged}"
    )
    print(f"Final pose error:    rot={np.rad2deg(rot_err):.3f} deg  trans={trans_err:.4f} m")


if __name__ == "__main__":
    main()
