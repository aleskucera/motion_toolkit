"""Unit tests for the accumulator's pose / cloud math (`_mapping_math`).

Pure numpy — no rclpy, no Warp — so this runs without a ROS install. The module
under test lives in the (non-pip-installed) ROS package, so we add it to the path.

Run: python tests/ros/test_mapping_math.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROS_PKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ros", "terrain_toolkit_ros")
)
sys.path.insert(0, _ROS_PKG)

from terrain_toolkit_ros._mapping_math import crop_box  # noqa: E402
from terrain_toolkit_ros._mapping_math import invert_pose  # noqa: E402
from terrain_toolkit_ros._mapping_math import matrix_to_quaternion  # noqa: E402
from terrain_toolkit_ros._mapping_math import odom_delta  # noqa: E402
from terrain_toolkit_ros._mapping_math import pose_correction_magnitude  # noqa: E402
from terrain_toolkit_ros._mapping_math import transform_points_xyz  # noqa: E402


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _pose(roll: float, pitch: float, yaw: float, t: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(roll, pitch, yaw)
    T[:3, 3] = t
    return T


def _random_poses(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        rpy = rng.uniform(-np.pi, np.pi, 3)
        t = rng.uniform(-10.0, 10.0, 3)
        out.append(_pose(rpy[0], rpy[1], rpy[2], tuple(t)))
    return out


def test_invert_pose() -> None:
    for T in _random_poses(20):
        assert np.allclose(T @ invert_pose(T), np.eye(4), atol=1e-9)


def test_odom_delta_reconstructs_trajectory() -> None:
    # When world == odom (no ICP correction), composing the odom delta onto the
    # previous world pose must reproduce the ground-truth current pose.
    poses = _random_poses(10)
    for prev, curr in zip(poses[:-1], poses[1:]):
        world_prev = prev  # world == odom
        recon = world_prev @ odom_delta(prev, curr)
        assert np.allclose(recon, curr, atol=1e-9)


def test_transform_points_roundtrip() -> None:
    rng = np.random.default_rng(1)
    pts = rng.uniform(-5.0, 5.0, (1000, 3))
    T = _pose(0.3, -0.7, 1.2, (1.0, -2.0, 0.5))
    back = transform_points_xyz(invert_pose(T), transform_points_xyz(T, pts))
    assert np.allclose(back, pts, atol=1e-9)
    assert transform_points_xyz(T, np.empty((0, 3))).shape == (0, 3)


def test_crop_box() -> None:
    pts = np.array(
        [
            [0.0, 0.0, 9.0],  # center
            [1.0, 1.0, 0.0],  # on the boundary (inclusive)
            [1.01, 0.0, 0.0],  # just outside in x
            [0.0, -1.01, 0.0],  # just outside in y
            [0.5, -0.5, 50.0],  # inside, z ignored
        ]
    )
    center = np.zeros(3)
    inside = crop_box(pts, center, 1.0)
    assert inside.shape[0] == 3, inside
    # Rectangular box: wide in x, narrow in y.
    rect = crop_box(pts, center, (2.0, 0.4))
    assert rect.shape[0] == 2, rect  # center + the x-outlier (|x|<=2, |y|<=0.4)
    assert crop_box(np.empty((0, 3)), center, 1.0).shape == (0, 3)


def test_pose_correction_magnitude() -> None:
    a = _pose(0.0, 0.0, 0.0, (0.0, 0.0, 0.0))
    b = _pose(0.0, 0.0, np.deg2rad(30.0), (3.0, 4.0, 0.0))
    rot, trans = pose_correction_magnitude(a, b)
    assert abs(trans - 5.0) < 1e-9, trans
    assert abs(np.rad2deg(rot) - 30.0) < 1e-6, np.rad2deg(rot)
    rot0, trans0 = pose_correction_magnitude(b, b)
    assert abs(rot0) < 1e-9 and abs(trans0) < 1e-9


def test_matrix_to_quaternion() -> None:
    # Round-trip each branch of Shepperd's method against a known good path:
    # rebuild the rotation from the quaternion and compare.
    for T in _random_poses(20):
        x, y, z, w = matrix_to_quaternion(T)
        # quaternion (x,y,z,w) -> rotation matrix
        n = x * x + y * y + z * z + w * w
        s = 2.0 / n
        R = np.array(
            [
                [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
            ]
        )
        assert np.allclose(R, T[:3, :3], atol=1e-9), T[:3, :3]


def test_accumulation_reconstruction() -> None:
    # A static scene observed from several known base poses, accumulated with
    # perfect odom and identity ICP correction, must reproduce the scene.
    rng = np.random.default_rng(3)
    scene = rng.uniform(-3.0, 3.0, (2000, 3))  # world-frame points

    poses = [_pose(0.0, 0.0, 0.4 * k, (0.5 * k, -0.3 * k, 0.0)) for k in range(5)]
    accumulated = []
    for world_T_base in poses:
        # "Observe": project the world scene into this base frame.
        scan_base = transform_points_xyz(invert_pose(world_T_base), scene)
        # Accumulate back into the world frame using the (perfect) pose.
        accumulated.append(transform_points_xyz(world_T_base, scan_base))

    for chunk in accumulated:
        assert np.allclose(chunk, scene, atol=1e-9)


def main() -> None:
    test_invert_pose()
    test_odom_delta_reconstructs_trajectory()
    test_transform_points_roundtrip()
    test_crop_box()
    test_pose_correction_magnitude()
    test_matrix_to_quaternion()
    test_accumulation_reconstruction()
    print("PASS: _mapping_math (7 tests)")


if __name__ == "__main__":
    main()
