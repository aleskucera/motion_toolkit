"""Benchmark ICP on real Ouster lidar data.

Loads one scan, synthesizes a source cloud by applying a known SE(3) offset,
and times ICP across multiple runs.
"""

import argparse
import time

import numpy as np
import warp as wp
from terrain_toolkit import IcpAligner
from terrain_toolkit import IcpConfig


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


def pose_diff(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    R_err = A[:3, :3].T @ B[:3, :3]
    tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(tr)), float(np.linalg.norm(A[:3, 3] - B[:3, 3]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="ouster.npy")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Optional random subsample of both clouds (for scaling studies)",
    )
    p.add_argument(
        "--voxel",
        type=float,
        default=None,
        help="Voxel downsample size (m) applied to source inside ICP.",
    )
    p.add_argument(
        "--voxel-target", action="store_true", help="Also voxel-downsample the target cloud."
    )
    p.add_argument(
        "--fixed-bounds",
        action="store_true",
        help="Use fixed voxel bounds instead of per-call min/max.",
    )
    p.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip per-stage profiling (avoids per-stage wp.synchronize overhead).",
    )
    p.add_argument("--verbose-once", action="store_true")
    args = p.parse_args()

    pts = np.load(args.path).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected (N,3+) array, got {pts.shape}")
    pts = pts[:, :3]
    # Same axis convention as test_ouster.py.
    pts = np.stack([pts[:, 2], pts[:, 0], -pts[:, 1]], axis=1)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if args.subsample is not None and args.subsample < len(pts):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts), args.subsample, replace=False)
        pts = pts[idx]

    print(f"Loaded {len(pts)} points from {args.path}")

    # Synthesize source by applying a known transform + measurement noise.
    R_gt = rpy_to_R(np.deg2rad(2.0), np.deg2rad(-1.5), np.deg2rad(4.0))
    t_gt = np.array([0.20, -0.10, 0.03])
    T_gt = pose(R_gt, t_gt)

    target = pts
    R_inv = R_gt.T
    t_inv = -R_inv @ t_gt
    source = (R_inv @ target.T).T + t_inv
    rng = np.random.default_rng(1)
    source = (source + rng.normal(0.0, 0.005, source.shape)).astype(np.float32)

    # Noisy odometry prior.
    R_init = rpy_to_R(np.deg2rad(1.0), np.deg2rad(-0.8), np.deg2rad(2.5))
    t_init = np.array([0.12, -0.06, 0.01])
    T_init = pose(R_init, t_init)

    bounds = None
    if args.fixed_bounds:
        # Pad the target extent by 5 m so source points transformed from the
        # noisy prior still fall inside the fixed grid.
        pad = 5.0
        mn = target.min(axis=0) - pad
        mx = target.max(axis=0) + pad
        bounds = (
            float(mn[0]),
            float(mx[0]),
            float(mn[1]),
            float(mx[1]),
            float(mn[2]),
            float(mx[2]),
        )

    cfg = IcpConfig(
        max_iters=args.max_iters,
        max_correspondence_dist_m=0.5,
        normal_radius_m=0.3,
        voxel_size_m=args.voxel,
        voxel_target=args.voxel_target,
        voxel_bounds_m=bounds,
    )
    aligner = IcpAligner(cfg, verbose=args.verbose_once)

    # Warmup (kernel compile + JIT caches).
    for _ in range(args.warmup):
        aligner.align(source, target, init_pose=T_init)
    aligner.verbose = False

    # Timed runs.
    total_times = []
    iters_per_run = []
    rot_errs = []
    trans_errs = []
    profile_sum: dict[str, float] = {}
    for _ in range(args.runs):
        wp.synchronize()
        t0 = time.perf_counter()
        res = aligner.align(source, target, init_pose=T_init, profile=not args.no_profile)
        wp.synchronize()
        t1 = time.perf_counter()
        total_times.append((t1 - t0) * 1000.0)
        iters_per_run.append(res.iterations)
        re, te = pose_diff(res.pose, T_gt)
        rot_errs.append(np.rad2deg(re))
        trans_errs.append(te)
        for k, v in res.timings_ms.items():
            profile_sum[k] = profile_sum.get(k, 0.0) + v

    total_times = np.asarray(total_times)
    iters = np.asarray(iters_per_run)
    mean_iters = iters.mean()

    print()
    print(f"Points (target): {len(target):,}  (source): {len(source):,}")
    print(f"Runs: {args.runs} timed ({args.warmup} warmup) — max {args.max_iters} iters/run")
    print()
    print(
        f"Total time per align : mean={total_times.mean():.2f} ms  "
        f"median={np.median(total_times):.2f} ms  "
        f"min={total_times.min():.2f} ms  max={total_times.max():.2f} ms"
    )
    print(f"Iterations per align : mean={mean_iters:.1f}  min={iters.min()}  max={iters.max()}")
    print(f"Time per iteration   : ~{total_times.mean() / mean_iters:.2f} ms")
    print()
    print(
        f"Final pose error     : rot={np.mean(rot_errs):.4f} deg  "
        f"trans={np.mean(trans_errs) * 1000:.2f} mm"
    )

    print()
    print("Per-stage timing (mean ms per align):")
    order = [
        "voxel_downsample",
        "upload",
        "grid_build",
        "normals",
        "iterations",
    ]
    for key in order:
        v = profile_sum.get(key, 0.0) / args.runs
        pct = 100.0 * v / total_times.mean()
        label = f"{key} (GPU, all iters)" if key == "iterations" else f"{key} (GPU)"
        print(f"  {label:32s} {v:7.2f}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
