"""Settle-based orientation-aware cost-to-go V(x, y, theta).

Feasibility comes from the robot's OWN settle, not a thresholded traversability map: for every pose
(x, y, theta) the robot is placed on the terrain and the engine's residual / clearance / tilt are
read. A pose is blocked iff residual > resid_tol OR clearance < clear_margin OR the body exceeds the
robot's stability ENVELOPE -- |roll| > max_roll, or pitch beyond the asymmetric climb/descend limits
(climbing is nose-up = negative pitch). So feasibility is direction-aware: a side-slope is fine to
CLIMB head-on (pitch, tolerated) but blocked to traverse sideways (roll, dangerous) -- exactly what
the orientation-aware lattice can exploit. The envelope + turn radius come from RobotParams, the
SAME robot the rollouts drive. The static (zero-control) settle is friction-independent, so compute()
needs only the elevation and the goal.

Among FEASIBLE poses the arc cost prefers flatter ground via a graded penalty that splits into two
non-redundant pieces: the per-axis SHAPE (roll_cost_weight : pitch_cost_weight, the robot's relative
roll-vs-pitch susceptibility, from RobotParams) and a single STRENGTH gain (flatness_weight, a planner
knob: how much detour to trade for flatness). The lattice arc cost is
    arc_len * (1 + flatness_weight * mean(roll_cost_weight*|roll| + pitch_cost_weight*|pitch|)).
flatness_weight is the only global gain; the per-axis weights only set the shape (keep them a ratio,
e.g. 1.0 : 0.5, not a second gain).

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
        flatness_weight: float = 2.0,  # planner strength: how much detour to trade for flat ground
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
        self.resid_tol = resid_tol
        self.flatness_weight = flatness_weight  # the single global gain on the graded tilt cost
        # one robot object: build the dataclass once and read every robot property off it (the
        # built struct carries the planning fields: envelope / clearance / turn radius / cost shape).
        self.robot = robot_params.build(self.device)
        self.bounds = (x0, x0 + nx * cell, y0, y0 + ny * cell)
        self._vcap = 1.5 * (nx + ny) * cell * (1.0 + flatness_weight)

        # world coords of every cell center -> the pose grid we settle (one heading bin at a time)
        cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
        self._X = (x0 + cols * cell).ravel().astype(np.float32)
        self._Y = (y0 + rows * cell).ravel().astype(np.float32)

        self.settle_sim = Simulator(
            robot_params, solver_params, grid_params, nx * ny, 1, self.device
        )
        # the static (zero-control) settle is friction-independent (verified bit-identical across mu),
        # so a dummy uniform friction is all the Simulator needs.
        self._mu = Heightmap(np.full((ny, nx), 0.8, np.float32), (x0, y0), cell)
        self.solver = LatticeValueSolver(
            cell,
            ny,
            nx,
            n_theta=n_theta,
            turn_radius=self.robot.min_turn_radius,
            step=step,
            device=self.device,
        )
        self.V = wp.zeros((ny, nx, n_theta), dtype=wp.float32, device=self.device)

    def _settle_fields(self, elevation: np.ndarray) -> tuple[wp.array, wp.array]:
        """Settle every pose; return blocked[ny,nx,n_theta], tilt[ny,nx,n_theta] (rad) as wp.arrays."""
        ny, nx, n_theta = self.V.shape
        n_poses = nx * ny
        rob = self.robot  # read the robot's limits/weights off the one built struct
        max_roll = np.radians(rob.max_roll_deg)        # envelope -> radians (climb = nose-up = -pitch)
        max_pitch_up = np.radians(rob.max_pitch_up_deg)
        max_pitch_down = np.radians(rob.max_pitch_down_deg)
        sim = self.settle_sim
        sim.set_terrain(
            wp.array(
                np.ascontiguousarray(elevation, np.float32), dtype=wp.float32, device=self.device
            )
        )
        sim.set_friction(self._mu)
        blocked = np.zeros((ny, nx, n_theta), np.float32)
        tilt = np.zeros((ny, nx, n_theta), np.float32)
        for t in range(n_theta):
            heading = (t + 0.5) * 2.0 * np.pi / n_theta  # bin-center heading for this theta slice
            sim.start_pose.assign(
                np.stack([self._X, self._Y, np.full(n_poses, heading, np.float32)], 1)
            )
            sim.omega.zero_()
            sim.rollout_launch()
            der = sim.derived.numpy()[0]  # (z, pitch, roll) settled at each pose
            res = sim.residual.numpy()[0]
            clr = sim.clearance.numpy()[0]
            pitch, roll = der[:, 1], der[:, 2]
            graded = rob.roll_cost_weight * np.abs(roll) + rob.pitch_cost_weight * np.abs(pitch)  # roll>pitch
            over_envelope = (
                (np.abs(roll) > max_roll) | (pitch < -max_pitch_up) | (pitch > max_pitch_down)
            )
            infeasible = (res > self.resid_tol) | (clr < rob.clear_margin) | over_envelope
            blocked[:, :, t] = infeasible.reshape(ny, nx)
            tilt[:, :, t] = graded.reshape(ny, nx)
        return (
            wp.array(blocked, dtype=wp.float32, device=self.device),
            wp.array(tilt, dtype=wp.float32, device=self.device),
        )

    def compute(self, elevation: np.ndarray, goal_xy: np.ndarray) -> wp.array:
        """elevation [ny, nx] + world goal -> clamped V[ny, nx, n_theta] (no friction needed)."""
        blocked, graded_tilt = self._settle_fields(elevation)
        # the solver's generic per-cell cost gain is our flatness strength; the field already carries
        # the roll:pitch shape, so the two compose to the documented arc cost.
        v = self.solver.compute_from_fields(
            blocked, graded_tilt, goal_xy, self.bounds, tilt_weight=self.flatness_weight
        )
        wp.launch(
            _clamp3d_kernel,
            dim=self.V.shape,
            inputs=[v, self._vcap],
            outputs=[self.V],
            device=self.device,
        )
        return self.V
