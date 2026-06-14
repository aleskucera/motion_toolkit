from dataclasses import dataclass

import numpy as np
import warp as wp

# --- provenance: where the default mass/com come from (common.py). Not used at runtime. ---
_MASSES = np.array(
    [
        [-0.13, 0.0, 0.0, 78.8375],  # front box
        [-0.61, 0.0, 0.0, 10.8625],  # rear box
        [0.0, 0.365, 0.0, 5.5],  # left wheel
        [0.0, -0.365, 0.0, 5.5],  # right wheel
        [-0.75, 0.0, 0.0, 5.5],  # rear wheel
    ]
)
DEFAULT_MASS = float(_MASSES[:, 3].sum())  # 106.2 kg
DEFAULT_COM = (_MASSES[:, :3] * _MASSES[:, 3:4]).sum(0) / DEFAULT_MASS  # x≈-0.198


@wp.struct
class Robot:
    """Device-side robot constants, passed into kernels as one struct.

    Safe as a wp.struct: `wheel_pos`/`chassis_pts` are read-only (not
    differentiated), so the struct-autodiff limitation does not apply. Only the
    differentiated grids (height, friction) stay plain top-level kernel args.
    """

    wheel_pos: wp.array(dtype=wp.vec3)  # [3] left/right/rear
    chassis_pts: wp.array(dtype=wp.vec3)  # [Np] belly non-penetration samples
    n_chassis: wp.int32  # len(chassis_pts); struct-member .shape is unreliable on CUDA
    wheel_radius: wp.float32
    half_track: wp.float32
    com: wp.vec3
    mass: wp.float32
    gravity: wp.float32


@dataclass(frozen=True)
class RobotParams:  # host-side robot knobs — what you nudge
    wheel_radius: float = 0.35
    half_track: float = 0.365
    rear_offset: float = 0.75
    gravity: float = 9.81
    mass: float = DEFAULT_MASS
    com: tuple = (float(DEFAULT_COM[0]), 0.0, 0.0)  # full vec3, independent of mass
    chassis_nx: int = 3
    chassis_ny: int = 3

    def build(self, device="cuda") -> Robot:
        b, l = self.half_track, self.rear_offset
        wheel_pos = np.array([[0, b, 0], [0, -b, 0], [-l, 0, 0]], np.float32)
        cpts = self._chassis_pts()
        r = Robot()
        r.wheel_pos = wp.array(wheel_pos, dtype=wp.vec3, device=device)
        r.chassis_pts = wp.array(cpts, dtype=wp.vec3, device=device)
        r.n_chassis = int(cpts.shape[0])
        r.wheel_radius = self.wheel_radius
        r.half_track = self.half_track
        r.com = wp.vec3(*self.com)
        r.mass = self.mass
        r.gravity = self.gravity
        return r

    def _chassis_pts(self):
        boxes = [(-0.13, 0.0, 0.0, 0.24, 0.28, 0.10), (-0.61, 0.0, 0.0, 0.24, 0.12, 0.10)]
        pts = [
            [cx + sx * hx, cy + sy * hy, cz - hz]
            for cx, cy, cz, hx, hy, hz in boxes
            for sx in np.linspace(-1, 1, self.chassis_nx)
            for sy in np.linspace(-1, 1, self.chassis_ny)
        ]
        return np.array(pts, np.float32)


@wp.struct
class SolverC:
    """Device-side settle/integration numerics. All scalars -> safe as a struct.

    `k_turn` (the friction->alpha turning gain) rides along here as a non-diff
    scalar; promote it to a plain length-1 array only if d/dk is ever needed.
    """

    newton_iters: wp.int32
    max_step: wp.float32
    tilt_clamp: wp.float32
    dt: wp.float32
    k_turn: wp.float32


@dataclass
class SolverParams:  # settle/integration numerics — tuning, separate from the robot
    dt: float = 0.05
    newton_iters: int = 8
    max_step: float = 0.2
    tilt_clamp: float = 1.2
    k_turn: float = 2.0

    def build(self) -> SolverC:
        s = SolverC()
        s.newton_iters = self.newton_iters
        s.max_step = self.max_step
        s.tilt_clamp = self.tilt_clamp
        s.dt = self.dt
        s.k_turn = self.k_turn
        return s


class Solver:
    def __init__(
        self, robot: RobotParams, params: SolverParams = SolverParams(), device="cuda"
    ) -> None:
        self.robot = robot.build(device)  # device Robot struct
        self.solver = params.build()  # device SolverC struct
        self.p = params  # host params (dt etc. for host-side loops)
        self.device = device


class State:
    def __init__(self) -> None:
        self.q: wp.array | None = None  # (x, y, z, roll, pitch, yaw)

        self.contacts: wp.array | None = None  # contacts
        self.normals: wp.array | None = None  # contact normals
        self.loads: wp.array | None = None  # contact normal loads [3]

        self.chassis_clearance: wp.float32 | None = None  # Chassis clearance vs raw terrain
        self.valid: bool | None = None  # not hight-centered (chassis clears)
