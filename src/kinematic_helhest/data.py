"""Rosbag loading + sim/real alignment for the kinematic twin.

The remap/sign/align logic is copied (not imported) from
examples/helhest_junior/replay_real.py to keep this package free of Newton/Ostrich.
See that file and the project memory for why each reconciliation is needed.
"""

import pathlib

import h5py
import numpy as np

SYNCED_DIR = pathlib.Path.home().joinpath("rosbags_experiment", "synced")
DEFAULT_RUN = "run_2026_05_20-18_04_51.h5"

# HDF5 wheel order is [left, rear, right]; sim wants [left, right, rear].
DATA_TO_SIM = [0, 2, 1]
# Sign flip per sim wheel [left, right, rear] so recorded forward -> sim forward.
WHEEL_SIGN = np.array([-1.0, 1.0, -1.0], dtype=np.float32)

# Static box obstacle (real dimensions / measured placement), from replay_real.py.
BOX_HALF_EXTENTS = (0.37, 0.575, 0.06)  # X long, Y wide, Z tall
BOX_CENTER = (1.0 + BOX_HALF_EXTENTS[0], 0.0, BOX_HALF_EXTENTS[2])

# Total-station prism mount offset in the chassis frame (top-front of front box).
PRISM_OFFSET = np.array([0.11, 0.0, 0.10], dtype=np.float32)


def load_setpoints(h5_path, drive, dt, duration):
    """Resample recorded wheel commands onto the sim timestep grid.

    Returns (setpoints[T, 3] in sim order+sign [L,R,rear], real dict, run_id, t_grid).
    """
    with h5py.File(h5_path, "r") as f:
        src = "/joint_setpoint/velocity" if drive == "setpoint" else "/joint_states/velocity"
        t_src = "/joint_setpoint/t" if drive == "setpoint" else "/joint_states/t"
        sp_t = f[t_src][:]
        sp_v = f[src][:]  # [N, 3] in [left, rear, right]
        real = {
            "t": f["/pose_world/t"][:],
            "position": f["/pose_world/position"][:],
            "orientation": f["/pose_world/orientation"][:],
            "yaw": f["/pose_world/yaw"][:],
        }
        run_id = f.attrs["run_id"]

    T = int(round(duration / dt))
    t_grid = np.arange(T) * dt
    resampled = np.zeros((T, 3), dtype=np.float32)
    for c in range(3):
        resampled[:, c] = np.interp(t_grid, sp_t, sp_v[:, c])

    setpoints = resampled[:, DATA_TO_SIM] * WHEEL_SIGN
    return setpoints.astype(np.float32), real, run_id, t_grid


def align_real_to_sim(real, heading_dist=1.0):
    """Rigid 2D transform of the real (subt-frame) trajectory into the sim frame:
    start at origin, initial heading (direction of travel over first heading_dist
    metres) along +X. NaN holes preserved. Returns (pos[M,3], t[M])."""
    pos = real["position"].copy()
    t = real["t"]

    valid = ~np.isnan(pos[:, 0])
    first = np.argmax(valid)
    origin = pos[first].copy()
    rel = pos - origin

    vp = rel[valid]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(vp[:, :2], axis=0), axis=1))])
    j = int(np.argmax(cum >= heading_dist))
    if j == 0:
        j = len(vp) - 1
    heading = np.arctan2(vp[j, 1], vp[j, 0])

    theta = -heading
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty_like(pos)
    out[:, 0] = rel[:, 0] * c - rel[:, 1] * s
    out[:, 1] = rel[:, 0] * s + rel[:, 1] * c
    out[:, 2] = rel[:, 2]
    out[~valid] = np.nan
    return out, t


def _quat_rotate(q, v):
    """Rotate vec3 v by quaternion q=[x,y,z,w]."""
    u = q[:3]
    w = q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def prism_track(poses, offset=PRISM_OFFSET):
    """World position of the prism point for each chassis pose [T, 7]."""
    out = np.empty((poses.shape[0], 3), dtype=np.float32)
    for k in range(poses.shape[0]):
        out[k] = poses[k, :3] + _quat_rotate(poses[k, 3:7], offset)
    return out
