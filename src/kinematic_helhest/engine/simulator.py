"""Preallocated forward-simulation context.

Build once from (robot params, solver params, grid spec, batch_size, n_steps); then `set_terrain`,
`set_uniform_friction`, and `rollout` reuse every device buffer — no allocation in the loop.
Replaces the old planning-side `BatchRollout`: the engine now owns the simulation
state (the rollout buffers + the terrain/envelope/friction grids), and the planner
just feeds controls in.

The grid (cells_x, cells_y, cell_size, origin) is FIXED at construction. In the robot-centered
rolling map the window dimensions never change — only the terrain *values* do — so
each cycle overwrites the owned buffers in place.

Forward-only for now (no requires_grad / tape); the differentiable calibration path
keeps its own buffers in tests/engine/gradients.py.
"""

import numpy as np
import warp as wp
from warp import Device

from .envelope import _contact_kernel
from .envelope import _gather_kernel
from .robot import RobotParams
from .step import rollout_kernel
from .step import SolverParams
from .terrain import GridParams


class Simulator:
    def __init__(
        self,
        robot_params: RobotParams,
        solver_params: SolverParams,
        grid_params: GridParams,
        batch_size: int,
        n_steps: int,
        device: Device | str | None = None,
    ):

        self.device = wp.get_device(device)

        self.batch_size = batch_size
        self.n_steps = n_steps

        self.robot = robot_params.build(device)  # device Robot struct
        self.solver = solver_params.build()  # device Solver struct
        self.grid = grid_params.build()  # Grid (fixed)
        self.wheel_radius = robot_params.wheel_radius
        self.env_radius = int(np.ceil(robot_params.wheel_radius / grid_params.cell_size))
        ny, nx = grid_params.cells_y, grid_params.cells_x

        with wp.ScopedDevice(self.device):
            # terrain: owned envelope + friction, plus the arg-max scratch; raw is borrowed.
            self.elevation = wp.zeros((ny, nx), dtype=wp.float32)
            self.envelope = wp.zeros((ny, nx), dtype=wp.float32)
            self.friction = wp.zeros((ny, nx), dtype=wp.float32)
            self._contact_iy = wp.zeros((ny, nx), dtype=wp.int32)
            self._contact_ix = wp.zeros((ny, nx), dtype=wp.int32)
            self._cap = wp.zeros((ny, nx), dtype=wp.float32)

            # rollout buffers + control inputs, allocated ONCE.
            self.controlled = wp.zeros((n_steps + 1, batch_size), dtype=wp.vec3f)
            self.derived = wp.zeros((n_steps + 1, batch_size), dtype=wp.vec3f)
            self.loads = wp.zeros((n_steps, batch_size), dtype=wp.vec3f)
            self.turning = wp.zeros((n_steps, batch_size), dtype=wp.vec2f)
            self.clearance = wp.zeros((n_steps, batch_size), dtype=wp.float32)
            self.residual = wp.zeros((n_steps, batch_size), dtype=wp.float32)
            self.wheel_omega = wp.zeros((n_steps, batch_size), dtype=wp.vec3f)
            self.start_pose = wp.zeros(batch_size, dtype=wp.vec3f)

    def set_terrain(self, elevation: wp.array):
        wp.copy(self.elevation, elevation)

        wp.launch(
            kernel=_contact_kernel,
            dim=self.elevation.shape,
            inputs=[
                self.elevation,
                self.grid.cell_size,
                self.wheel_radius,
                self.env_radius,
            ],
            outputs=[
                self._contact_iy,
                self._contact_ix,
                self._cap,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_gather_kernel,
            dim=self.elevation.shape,
            inputs=[
                self.elevation,
                self._contact_iy,
                self._contact_ix,
                self._cap,
            ],
            outputs=[self.envelope],
            device=self.device,
        )

    def set_uniform_friction(self, value: float):
        """Uniform friction: overwrite the owned friction grid in place."""
        self.friction.fill_(float(value))

    def set_friction(self, friction_hm: np.ndarray):
        """Per-cell friction from a numpy Heightmap matching the grid (copied in place)."""
        self.friction.assign(np.ascontiguousarray(friction_hm.H, np.float32))

    def rollout_launch(self):
        """Launch the whole rollout (init + T steps) in ONE fused kernel; NO host I/O.

        `self.wheel_omega` must already hold the controls and `self.start_pose` the init pose
        (e.g. filled on-device by the GPU MPPI sampler). Results stay on device in
        controlled/derived/loads/turning/clearance/residual -- the graph-capturable core.
        Forward-only fusion (~1.2x vs per-step launches); the differentiable path uses
        the per-step step_kernel instead (rollout_kernel's register carry isn't autodiffable).
        """
        wp.launch(
            rollout_kernel,
            self.batch_size,
            inputs=[
                self.n_steps,
                self.envelope,
                self.elevation,
                self.friction,
                self.grid,
                self.robot,
                self.solver,
                self.start_pose,
                self.wheel_omega,
            ],
            outputs=[
                self.controlled,
                self.derived,
                self.loads,
                self.turning,
                self.clearance,
                self.residual,
            ],
            device=self.device,
        )

    def rollout(self, wheel_omega, init_pose):
        """wheel_omega [T, B, 3], init_pose (x,y,yaw) shared by all rollouts. Returns
        controlled [T+1,B,3] (x,y,yaw), derived [T+1,B,3] (z,pitch,roll), clear/resid [T,B]."""
        self.wheel_omega.assign(np.ascontiguousarray(wheel_omega, np.float32))
        self.start_pose.assign(
            np.ascontiguousarray(
                np.tile(np.asarray(init_pose, np.float32), (self.batch_size, 1)), np.float32
            )
        )
        self.rollout_launch()
        return (
            self.controlled.numpy(),
            self.derived.numpy(),
            self.clearance.numpy(),
            self.residual.numpy(),
        )
