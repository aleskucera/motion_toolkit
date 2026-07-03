"""Pose algebra and cloud geometry for the accumulating terrain mapper.

Pure numpy, **no rclpy** — so the trajectory/accumulation core is unit-testable
without a ROS install. Poses are 4x4 homogeneous SE(3) matrices `T` such that a
point in the source frame maps to the target frame as `T @ [x, y, z, 1]`.
"""

from __future__ import annotations

import numpy as np


def invert_pose(T: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) pose, exploiting `R^-1 = R^T` (no general solve)."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two poses: the pose `a` followed by `b`, i.e. `a @ b`."""
    return a @ b


def odom_delta(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Frame-to-frame motion `prev_T_curr = inv(odom_T_prev) @ odom_T_curr`.

    Using the delta (not the absolute odom pose) makes the trajectory robust to
    odom's slowly-drifting global origin.
    """
    return invert_pose(prev) @ curr


def transform_points_xyz(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply pose `T` to an (N, 3) cloud, returning (N, 3) in `T`'s target frame."""
    if points.shape[0] == 0:
        return points.reshape(0, 3)
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def crop_box(
    points: np.ndarray,
    center: np.ndarray,
    radius: float | tuple[float, float],
) -> np.ndarray:
    """Keep points inside an axis-aligned xy box (inclusive) around `center`.

    `radius` is a single half-extent or `(half_x, half_y)`. Only x and y are
    tested; z is unbounded. Boundary points are kept.
    """
    if points.shape[0] == 0:
        return points.reshape(0, 3)
    half_x, half_y = (radius, radius) if np.isscalar(radius) else radius
    dx = np.abs(points[:, 0] - center[0])
    dy = np.abs(points[:, 1] - center[1])
    inside = (dx <= half_x) & (dy <= half_y)
    return points[inside]


def matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix (top-left 3x3 of a pose) → quaternion `(x, y, z, w)`.

    Shepperd's method: pick the largest diagonal branch for numerical stability.
    """
    m = R[:3, :3]
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle vector `omega` (‖omega‖ = angle).

    First-order fallback near zero; the theta≈π branch is skipped because the
    per-sweep rotations this is used for are always small.
    """
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1.0e-9:
        return 0.5 * axis  # sin(theta) ≈ theta → omega ≈ ½·vee(R − Rᵀ)
    return (theta / (2.0 * np.sin(theta))) * axis


def deskew_scan(points: np.ndarray, alphas: np.ndarray, sweep_delta: np.ndarray) -> np.ndarray:
    """Motion-compensate a swept cloud to the sweep-end pose (constant velocity).

    A spinning LiDAR measures each point at a slightly different instant, so a
    moving robot smears the sweep. `points` (N, 3, base frame) were measured at
    sweep fractions `alphas` (N,) in [0, 1] — 0 = sweep start, 1 = sweep end
    (the reference). `sweep_delta` is `base_start_T_base_end`, the base motion
    over the sweep (i.e. the odom frame-to-frame delta). Returns every point
    re-expressed in the sweep-end base frame.

    Constant-velocity model: interpolate rotation on the screw axis and shift
    translation linearly. Derivation — with `d = sweep_delta = [R_d | t_d]` and
    `R(α) = exp(α·log R_d)`, the point measured at α maps to the end frame by
    `p' = R_dᵀ · (R(α)·p + (α−1)·t_d)`. (α=1 → identity; α=0 → inv(d).)
    """
    if points.shape[0] == 0:
        return points.reshape(0, 3)
    R_delta = sweep_delta[:3, :3]
    t_delta = sweep_delta[:3, 3]
    omega = _so3_log(R_delta)
    theta = float(np.linalg.norm(omega))
    if theta < 1.0e-9:
        rotated = points  # R(α) ≈ I for every point
    else:
        axis = omega / theta
        angle = (alphas * theta)[:, None]
        # Rodrigues applied per point (shared axis, per-point angle α·θ).
        cross1 = np.cross(axis, points)
        cross2 = np.cross(axis, cross1)
        rotated = points + np.sin(angle) * cross1 + (1.0 - np.cos(angle)) * cross2
    shifted = rotated + (alphas[:, None] - 1.0) * t_delta
    return shifted @ R_delta  # right-multiply by R_delta ≡ apply R_deltaᵀ per row


def pose_correction_magnitude(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """How far pose `b` is from pose `a`, as `(rotation_rad, translation_m)`.

    Translation is the Euclidean distance between origins; rotation is the angle
    of the relative rotation `R_a^T R_b` (clamped before arccos for numerics).
    """
    trans = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    R_rel = a[:3, :3].T @ b[:3, :3]
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    rot = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return rot, trans
