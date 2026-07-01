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
