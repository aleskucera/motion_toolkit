"""Settle-based orientation-aware cost-to-go V(x, y, theta).

Feasibility comes from the robot's OWN settle, not a thresholded traversability map: for every pose
(x, y, theta) the robot is placed on the terrain and the engine's residual / clearance / tilt are
read. A pose is blocked iff residual > resid_tol OR clearance < clear_margin OR the body exceeds the
robot's stability ENVELOPE -- |roll| > max_roll, or pitch beyond the asymmetric climb/descend limits
(climbing is nose-up = negative pitch). So feasibility is direction-aware: a side-slope is fine to
CLIMB head-on (pitch, tolerated) but blocked to traverse sideways (roll, dangerous) -- exactly what
the orientation-aware lattice can exploit. The envelope + turn radius come from RobotParams, the
SAME robot the rollouts drive. Total tilt is still the graded arc cost (prefer flat). The static
(zero-control) settle is friction-independent, so compute() needs only the elevation and the goal.

Self-contained on purpose: this is the cost-to-go we keep; the traversability/2D variants in
costtogo.py are on their way out.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    import warp.context

    from ..engine import GridParams
    from ..engine import RobotParams
    from ..engine import SolverParams


@wp.kernel
def _clamp3d_kernel(
    v_in: wp.array3d(dtype=wp.float32), vcap: wp.float32, v_out: wp.array3d(dtype=wp.float32)
):
    """Copy V, replacing the solver's +inf (unreachable) with a large finite cap so the cost
    kernel's trilinear sampling never blends to inf."""
    r, c, t = wp.tid()
    val = v_in[r, c, t]
    if val > vcap:
        v_out[r, c, t] = vcap
    else:
        v_out[r, c, t] = val


class CostToGoLatticeSettle:
    """Orientation-aware cost-to-go with feasibility from the robot's settle (see module docstring).
    compute(elevation, goal) -> clamped V[ny, nx, n_theta]."""

    def __init__(
        self,
        grid_params: GridParams,
        robot_params: RobotParams,
        solver_params: SolverParams,
        device: wp.context.Device | str | None = None,
        n_theta: int = 24,
        step: float = 0.3,
        resid_tol: float = 1e-2,
        clear_margin: float = 0.05,
        tilt_weight: float = 2.0,
    ) -> None:
        # robot_params / solver_params are MANDATORY (like grid): the caller passes the same vehicle
        # it gave the planner, so the cost-to-go settles exactly the robot the rollouts drive -- no
        # silent fallback that could quietly disagree with the planner.
        try:
            from terrain_toolkit import LatticeValueSolver
        except ImportError as e:
            raise ImportError(
                "orientation-aware cost-to-go needs terrain_toolkit; install it, e.g. "
                "`uv pip install -e ../terrain_toolkit --no-deps`"
            ) from e
        from ..engine import Simulator
        from ..heightmap import Heightmap

        nx, ny, cell = grid_params.cells_x, grid_params.cells_y, grid_params.cell_size
        x0, y0 = grid_params.origin_x, grid_params.origin_y

        self.device = wp.get_device(device)  # resolve None -> default once, reuse everywhere
        self.resid_tol, self.clear_margin = resid_tol, clear_margin
        self.tilt_weight = tilt_weight
        # the robot's stability envelope (radians); climbing is nose-UP = negative pitch
        self.max_roll = np.radians(robot_params.max_roll_deg)
        self.max_pitch_up = np.radians(robot_params.max_pitch_up_deg)
        self.max_pitch_down = np.radians(robot_params.max_pitch_down_deg)
        self.roll_cost_weight = robot_params.roll_cost_weight    # graded cost: roll weighted more
        self.pitch_cost_weight = robot_params.pitch_cost_weight  # than pitch (prefer low-roll lines)
        self.bounds = (x0, x0 + nx * cell, y0, y0 + ny * cell)
        self._vcap = 1.5 * (nx + ny) * cell * (1.0 + tilt_weight)

        # world coords of every cell center -> the pose grid we settle (one heading bin at a time)
        cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
        self._X = (x0 + cols * cell).ravel().astype(np.float32)
        self._Y = (y0 + rows * cell).ravel().astype(np.float32)

        self.settle_sim = Simulator(robot_params, solver_params, grid_params, nx * ny, 1, self.device)
        # the static (zero-control) settle is friction-independent (verified bit-identical across mu),
        # so a dummy uniform friction is all the Simulator needs.
        self._mu = Heightmap(np.full((ny, nx), 0.8, np.float32), (x0, y0), cell)
        self.solver = LatticeValueSolver(cell, ny, nx, n_theta=n_theta,
                                         turn_radius=robot_params.min_turn_radius, step=step, device=self.device)
        self.V = wp.zeros((ny, nx, n_theta), dtype=wp.float32, device=self.device)

    def _settle_fields(self, elevation: np.ndarray) -> tuple[wp.array, wp.array]:
        """Settle every pose; return blocked[ny,nx,n_theta], tilt[ny,nx,n_theta] (rad) as wp.arrays."""
        ny, nx, n_theta = self.V.shape
        n_poses = nx * ny
        sim = self.settle_sim
        sim.set_terrain(wp.array(np.ascontiguousarray(elevation, np.float32), dtype=wp.float32, device=self.device))
        sim.set_friction(self._mu)
        blocked = np.zeros((ny, nx, n_theta), np.float32)
        tilt = np.zeros((ny, nx, n_theta), np.float32)
        for t in range(n_theta):
            heading = (t + 0.5) * 2.0 * np.pi / n_theta  # bin-center heading for this theta slice
            sim.start_pose.assign(np.stack([self._X, self._Y, np.full(n_poses, heading, np.float32)], 1))
            sim.omega.zero_()
            sim.rollout_launch()
            der = sim.derived.numpy()[0]  # (z, pitch, roll) settled at each pose
            res = sim.residual.numpy()[0]
            clr = sim.clearance.numpy()[0]
            pitch, roll = der[:, 1], der[:, 2]
            graded = self.roll_cost_weight * np.abs(roll) + self.pitch_cost_weight * np.abs(pitch)  # roll>pitch
            over_envelope = (np.abs(roll) > self.max_roll) | (pitch < -self.max_pitch_up) | (pitch > self.max_pitch_down)
            infeasible = (res > self.resid_tol) | (clr < self.clear_margin) | over_envelope
            blocked[:, :, t] = infeasible.reshape(ny, nx)
            tilt[:, :, t] = graded.reshape(ny, nx)
        return (wp.array(blocked, dtype=wp.float32, device=self.device),
                wp.array(tilt, dtype=wp.float32, device=self.device))

    def compute(self, elevation: np.ndarray, goal_xy: np.ndarray) -> wp.array:
        """elevation [ny, nx] + world goal -> clamped V[ny, nx, n_theta] (no friction needed)."""
        blocked, tilt = self._settle_fields(elevation)
        v = self.solver.compute_from_fields(blocked, tilt, goal_xy, self.bounds, tilt_weight=self.tilt_weight)
        wp.launch(_clamp3d_kernel, dim=self.V.shape, inputs=[v, self._vcap], outputs=[self.V], device=self.device)
        return self.V
