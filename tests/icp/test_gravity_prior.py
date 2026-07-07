"""The ICP gravity soft-prior levels a tilted pose (roll/pitch -> gravity, yaw free)."""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.perception.icp.kernels import accumulate_gravity_prior_kernel
from helhest.perception.icp.kernels import se3_update_kernel
from helhest.perception.icp.kernels import solve6x6_kernel


def _device() -> str:
    return "cuda" if wp.is_cuda_available() else "cpu"


def test_gravity_prior_levels_tilted_pose() -> None:
    """Iterating the prior alone drives R·up -> world +z from a combined roll+pitch+yaw start,
    while leaving the yaw (heading) essentially intact."""
    wp.init()
    with wp.ScopedDevice(_device()):
        roll, pitch, yaw = np.deg2rad([20.0, 15.0, 30.0])
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        pose_np = np.eye(4, dtype=np.float32)
        pose_np[:3, :3] = rz @ ry @ rx  # tilted + yawed

        pose = wp.array(pose_np.reshape(1, 4, 4), dtype=wp.mat44)
        up = wp.array(np.array([[0.0, 0.0, 1.0]], np.float32), dtype=wp.vec3)  # up-in-base = base z
        w = wp.array(np.array([1.0], np.float32), dtype=wp.float32)
        jtj = wp.zeros((6, 6), dtype=wp.float32)
        jtr = wp.zeros(6, dtype=wp.float32)
        delta = wp.zeros(6, dtype=wp.float32)
        dr = wp.zeros(1, dtype=wp.float32)
        dt = wp.zeros(1, dtype=wp.float32)

        for _ in range(40):
            jtj.zero_()
            jtr.zero_()
            wp.launch(
                accumulate_gravity_prior_kernel, dim=1, inputs=[pose, up, w], outputs=[jtj, jtr]
            )
            wp.launch(solve6x6_kernel, dim=1, inputs=[jtj, jtr, 1.0e-6], outputs=[delta])
            wp.launch(se3_update_kernel, dim=1, inputs=[delta], outputs=[pose, dr, dt])
        wp.synchronize()

        rf = pose.numpy().reshape(4, 4)[:3, :3]
        up_world = rf @ np.array([0.0, 0.0, 1.0])
        assert np.allclose(up_world, [0.0, 0.0, 1.0], atol=1.0e-3), up_world  # leveled

        x_world = rf @ np.array([1.0, 0.0, 0.0])
        assert abs(x_world[2]) < 1.0e-3, x_world  # body x stayed horizontal (no roll/pitch leak)
        heading = np.arctan2(x_world[1], x_world[0])
        assert abs(heading - yaw) < 0.1, np.rad2deg(heading)  # yaw preserved (prior leaves it free)


def test_gravity_prior_disabled_is_noop() -> None:
    """weight 0 adds nothing to the system, so a passed-but-disabled prior is inert."""
    wp.init()
    with wp.ScopedDevice(_device()):
        pose = wp.array(np.eye(4, dtype=np.float32).reshape(1, 4, 4), dtype=wp.mat44)
        up = wp.array(np.array([[0.1, 0.0, 0.99]], np.float32), dtype=wp.vec3)  # tilted, but off
        w = wp.array(np.array([0.0], np.float32), dtype=wp.float32)
        jtj = wp.zeros((6, 6), dtype=wp.float32)
        jtr = wp.zeros(6, dtype=wp.float32)
        wp.launch(accumulate_gravity_prior_kernel, dim=1, inputs=[pose, up, w], outputs=[jtj, jtr])
        wp.synchronize()
        assert np.count_nonzero(jtj.numpy()) == 0
        assert np.count_nonzero(jtr.numpy()) == 0
