"""Export a simulator rollout to a Blender-friendly ``.npz``.

Two-step pipeline (this file is step 1):

    python -m kinematic_helhest.viz.blender_export [out.npz]
    blender [scene.blend] --background --python \
        src/kinematic_helhest/viz/blender_import.py -- --data out.npz [--robot robot.blend ...]

The ``.npz`` holds, per frame, the body pose (world position + orientation as BOTH a
quaternion and Euler ZYX), the integrated per-wheel spin angle, and a validity flag,
plus the terrain heightmap and the fixed robot geometry constants. It has no dependency
back on this package, so ``blender_import.py`` runs under Blender's bundled Python.

Frame 0 is the settled start pose (no motion yet); frame k+1 is the pose after applying
``wheel_omega[k]``. Wheel spin is the running integral ``cumsum(wheel_omega) * dt`` [rad],
ordered (left, right, rear) to match the engine's wheel order.
"""
from __future__ import annotations

import sys

import numpy as np

from .. import dynamics
from .. import heightmap
from ..driver import WarpDriver
from ..model import WHEEL_POS
from ..model import WHEEL_RADIUS
from .render import CHASSIS_BOXES
from .render import WHEEL_WIDTH


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Proper-rotation 3x3 -> unit quaternion (w, x, y, z), Blender's order."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        )
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        )
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        )
    return np.array([w, x, y, z], dtype=np.float64)


def collect_rollout(
    drv: WarpDriver, wheel_omega_seq: np.ndarray, dt: float
) -> dict[str, np.ndarray]:
    """Drive `drv` through wheel-speed sequence [T, 3] and record per-frame state [T+1, ...]."""
    seq = np.asarray(wheel_omega_seq, np.float64).reshape(-1, 3)
    n = len(seq) + 1  # +1 for the settled start frame
    pos = np.zeros((n, 3))
    euler = np.zeros((n, 3))  # (yaw, pitch, roll) [rad]
    quat = np.zeros((n, 4))  # (w, x, y, z)
    spin = np.zeros((n, 3))  # integrated wheel angle (left, right, rear) [rad]
    valid = np.zeros(n, dtype=bool)

    def record(f: int) -> None:
        st = drv.render_state()
        pos[f] = (st.x, st.y, st.place["z"])
        euler[f] = (st.yaw, st.place["pitch"], st.place["roll"])
        quat[f] = _mat_to_quat(np.asarray(st.place["R"]))
        valid[f] = st.valid

    record(0)
    for k, omega in enumerate(seq):
        drv.step(omega)
        spin[k + 1] = spin[k] + omega * dt
        record(k + 1)
    return {"pos": pos, "euler": euler, "quat": quat, "wheel_spin": spin, "valid": valid}


def write_npz(path: str, frames: dict[str, np.ndarray], hm: heightmap.Heightmap, dt: float) -> None:
    """Bundle per-frame state + terrain + robot geometry into one ``.npz`` for Blender."""
    np.savez_compressed(
        path,
        dt=np.float32(dt),
        pos=frames["pos"].astype(np.float32),
        quat=frames["quat"].astype(np.float32),
        euler=frames["euler"].astype(np.float32),
        wheel_spin=frames["wheel_spin"].astype(np.float32),
        valid=frames["valid"].astype(np.bool_),
        terrain_H=hm.H.astype(np.float32),
        terrain_x0=np.float32(hm.x0),
        terrain_y0=np.float32(hm.y0),
        terrain_cell=np.float32(hm.cell),
        wheel_pos=np.asarray(WHEEL_POS, np.float32),
        wheel_radius=np.float32(WHEEL_RADIUS),
        wheel_width=np.float32(WHEEL_WIDTH),
        chassis_boxes=np.asarray(CHASSIS_BOXES, np.float32),
    )


def demo(out_path: str) -> None:
    """A straight run up the ramp scene: wheels spin, the nose pitches up on the slope."""
    scene = heightmap.ramp_scene()
    # friction must match the terrain grid exactly (set_friction copies in place)
    mu = heightmap.Heightmap(np.full_like(scene.H, 0.8), (scene.x0, scene.y0), scene.cell)
    drv = WarpDriver(scene, mu, init_pose=(-1.0, 0.0, 0.0))

    n_steps = 90
    speed = 3.0  # [rad/s] equal wheel speeds -> v = R * speed = 1.05 m/s, straight ahead
    seq = np.tile([speed, speed, speed], (n_steps, 1)).astype(np.float32)

    frames = collect_rollout(drv, seq, dynamics.DT)
    write_npz(out_path, frames, scene, dynamics.DT)
    print(f"wrote {out_path}: {len(frames['pos'])} frames, terrain {scene.H.shape}")


if __name__ == "__main__":
    demo(sys.argv[1] if len(sys.argv) > 1 else "helhest_rollout.npz")
