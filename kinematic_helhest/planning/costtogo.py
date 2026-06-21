"""Cost-to-go field for MPPI (option E): obstacle-aware geodesic distance to the goal.

A horizon-limited MPPI that scores rollouts by straight-line distance stalls against
obstacles -- every rollout that shortens the Euclidean gap drives into the wall, and the
detour that actually pays off looks worse inside the horizon. The fix is to score by the
TRUE remaining path length instead: V(x, y) = least-cost distance to the goal routing
AROUND untraversable terrain. Sampling V (not ||x - goal||) in the cost makes the planner
detour.

Pipeline (all on device): terrain_toolkit's GeometricTraversabilityAnalyzer turns the
elevation into a [0, 1] traversability cost (slope + step + roughness), then its
GeodesicDistanceSolver (parallel min-relaxation == grid Dijkstra) solves the distance
field. Unreachable/obstacle cells come back +inf; we clamp them to a large finite value so
the planner's bilinear sampling stays finite (the rollouts' own graded clearance/residual
penalty still does the fine obstacle avoidance -- V only supplies the global routing).

terrain_toolkit is a lazy dependency: only this module imports it.
"""
import numpy as np
import warp as wp


@wp.kernel
def _clamp_kernel(
    v_in: wp.array2d(dtype=wp.float32), vcap: wp.float32, v_out: wp.array2d(dtype=wp.float32)
):
    """Copy V, replacing the solver's +inf (unreachable) with a large finite cap so the
    cost kernel's bilinear sampling never blends to inf."""
    r, c = wp.tid()
    val = v_in[r, c]
    if val > vcap:
        v_out[r, c] = vcap
    else:
        v_out[r, c] = val


class CostToGo:
    """Owns the terrain_toolkit analyzer + geodesic solver for a fixed grid size, and the
    clamped V buffer the MPPI cost samples. `compute(elevation, goal_xy)` updates V in place
    (a stable device buffer -> safe to hand to a CUDA-graph-captured cost kernel)."""

    def __init__(self, nx, ny, cell, x0, y0, device, *, obstacle_threshold=0.8, config=None):
        try:
            from terrain_toolkit import (
                GeodesicDistanceSolver,
                GeometricTraversabilityAnalyzer,
            )
        except ImportError as e:  # match navigate.py's helpful message
            raise ImportError(
                "cost-to-go (MPPI option E) needs terrain_toolkit; install it, e.g. "
                "`uv pip install -e ../terrain_toolkit --no-deps`"
            ) from e

        self.nx, self.ny = int(nx), int(ny)
        self.cell = float(cell)
        self.x0, self.y0 = float(x0), float(y0)
        self.bounds = (self.x0, self.x0 + self.nx * self.cell,
                       self.y0, self.y0 + self.ny * self.cell)
        self.obstacle_threshold = float(obstacle_threshold)
        self.device = device
        # unreachable penalty: larger than any reachable path on this grid (the L1 diameter
        # is an upper bound on the 8-connected geodesic), so unreached cells are always worst.
        self._vcap = float(1.5 * (self.nx + self.ny) * self.cell)

        self.analyzer = GeometricTraversabilityAnalyzer(self.cell, self.ny, self.nx, config, device=device)
        self.solver = GeodesicDistanceSolver(self.cell, self.ny, self.nx, device=device)
        self.V = wp.zeros((self.ny, self.nx), dtype=wp.float32, device=device)

    def compute(self, elevation, goal_xy):
        """elevation [ny, nx] (numpy or wp.array) + world goal -> V (wp.array, clamped meters)."""
        trav = self.analyzer.compute(elevation).total  # [ny, nx] in [0, 1]
        v = self.solver.compute(trav, goal_xy, self.bounds, obstacle_threshold=self.obstacle_threshold)
        wp.launch(_clamp_kernel, dim=(self.ny, self.nx),
                  inputs=[v, self._vcap], outputs=[self.V], device=self.device)
        return self.V


@wp.kernel
def _clamp3d_kernel(
    v_in: wp.array3d(dtype=wp.float32), vcap: wp.float32, v_out: wp.array3d(dtype=wp.float32)
):
    r, c, t = wp.tid()
    val = v_in[r, c, t]
    if val > vcap:
        v_out[r, c, t] = vcap
    else:
        v_out[r, c, t] = val


class CostToGoLattice:
    """Like CostToGo, but the ORIENTATION-AWARE cost-to-go V(x, y, theta) -- so the MPPI cost
    penalizes misaligned approaches a forward-only robot can't recover from (the pocket/ridge
    failure that position-only V causes). Runs the same terrain_toolkit traversability + the GPU
    LatticeValueSolver (forward-arc lattice value iteration), at the sim resolution with a larger
    arc step so its grid matches the sim grid. compute() updates the clamped V[ny,nx,n_theta] in
    place (stable buffer -> graph-safe)."""

    def __init__(self, nx, ny, cell, x0, y0, device, *, n_theta=16, turn_radius=0.6,
                 robot_radius=0.3, step=0.3, obstacle_threshold=0.8, trav_weight=0.0, config=None):
        try:
            from terrain_toolkit import (
                GeometricTraversabilityAnalyzer,
                LatticeValueSolver,
            )
        except ImportError as e:
            raise ImportError(
                "orientation-aware cost-to-go needs terrain_toolkit; install it, e.g. "
                "`uv pip install -e ../terrain_toolkit --no-deps`"
            ) from e
        self.nx, self.ny, self.cell = int(nx), int(ny), float(cell)
        self.x0, self.y0 = float(x0), float(y0)
        self.bounds = (self.x0, self.x0 + self.nx * self.cell, self.y0, self.y0 + self.ny * self.cell)
        self.n_theta = int(n_theta)
        self.obstacle_threshold = float(obstacle_threshold)
        self.device = device
        # graded arc cost inflates reachable V by up to (1 + trav_weight); grow the unreachable cap
        # to stay above any reachable path, else flat-but-long routes get clipped to "unreachable".
        self._vcap = float(1.5 * (self.nx + self.ny) * self.cell * (1.0 + trav_weight))
        self.analyzer = GeometricTraversabilityAnalyzer(self.cell, self.ny, self.nx, config, device=device)
        self.solver = LatticeValueSolver(self.cell, self.ny, self.nx, n_theta=self.n_theta,
                                         turn_radius=turn_radius, robot_radius=robot_radius,
                                         step=step, trav_weight=trav_weight, device=device)
        self.V = wp.zeros((self.ny, self.nx, self.n_theta), dtype=wp.float32, device=device)

    def compute(self, elevation, goal_xy):
        """elevation [ny, nx] + world goal -> V[ny, nx, n_theta] (wp.array, clamped meters)."""
        trav = self.analyzer.compute(elevation).total
        v = self.solver.compute(trav, goal_xy, self.bounds, obstacle_threshold=self.obstacle_threshold)
        wp.launch(_clamp3d_kernel, dim=(self.ny, self.nx, self.n_theta),
                  inputs=[v, self._vcap], outputs=[self.V], device=self.device)
        return self.V


class CostToGoLatticeSettle:
    """Like CostToGoLattice, but feasibility comes from the robot's own SETTLE instead of a
    thresholded traversability map. For every pose (x, y, theta) it places the robot and reads the
    engine's residual / clearance / tilt: a pose is blocked iff residual > resid_tol OR clearance <
    clear_margin OR tilt > tilt_max (the SAME validity the MPPI rollouts use -- no arbitrary obstacle
    threshold), and tilt is the graded arc cost (prefer flat). So walls block because the robot can't
    sit on their face, rough terrain is costly-but-passable, and the cost-to-go agrees with the
    rollouts by construction. compute(elevation, mu, goal) -> clamped V[ny, nx, n_theta]."""

    def __init__(self, nx, ny, cell, x0, y0, device, *, robot_params=None, solver_params=None,
                 n_theta=24, turn_radius=0.5, robot_radius=0.3, step=0.3,
                 resid_tol=1e-2, clear_margin=0.05, tilt_max_deg=40.0, tilt_weight=2.0):
        try:
            from terrain_toolkit import LatticeValueSolver
        except ImportError as e:
            raise ImportError(
                "orientation-aware cost-to-go needs terrain_toolkit; install it, e.g. "
                "`uv pip install -e ../terrain_toolkit --no-deps`"
            ) from e
        from .. import dynamics
        from ..engine import GridParams, Simulator

        self.nx, self.ny, self.cell = int(nx), int(ny), float(cell)
        self.x0, self.y0 = float(x0), float(y0)
        self.bounds = (self.x0, self.x0 + self.nx * self.cell, self.y0, self.y0 + self.ny * self.cell)
        self.n_theta = int(n_theta)
        self.device = device
        self.resid_tol, self.clear_margin = float(resid_tol), float(clear_margin)
        self.tilt_max = float(np.radians(tilt_max_deg))
        self.tilt_weight = float(tilt_weight)
        self._vcap = float(1.5 * (self.nx + self.ny) * self.cell * (1.0 + tilt_weight))
        # world coords of every cell center -> the pose grid we settle (heading added per bin)
        cols, rows = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        self._X = (self.x0 + cols * self.cell).ravel().astype(np.float32)
        self._Y = (self.y0 + rows * self.cell).ravel().astype(np.float32)
        rp = robot_params or dynamics.robot_params()
        sp = solver_params or dynamics.execution_solver()  # high-fidelity settle (the validated config)
        self.settle_sim = Simulator(rp, sp, GridParams(self.nx, self.ny, self.cell, self.x0, self.y0),
                                    self.nx * self.ny, 1, device)
        self.solver = LatticeValueSolver(self.cell, self.ny, self.nx, n_theta=self.n_theta,
                                         turn_radius=turn_radius, robot_radius=robot_radius,
                                         step=step, device=device)
        self.V = wp.zeros((self.ny, self.nx, self.n_theta), dtype=wp.float32, device=device)

    def _settle_fields(self, elevation, mu):
        """Settle every pose; return blocked[ny,nx,n_theta], tilt[ny,nx,n_theta] (rad) as wp.arrays."""
        sim, B = self.settle_sim, self.nx * self.ny
        sim.set_terrain(wp.array(np.ascontiguousarray(elevation, np.float32), dtype=wp.float32, device=self.device))
        sim.set_friction(mu)
        blocked = np.zeros((self.ny, self.nx, self.n_theta), np.float32)
        tilt = np.zeros((self.ny, self.nx, self.n_theta), np.float32)
        two_pi = 2.0 * np.pi
        for t in range(self.n_theta):
            th = (float(t) + 0.5) * two_pi / float(self.n_theta)  # bin-center heading
            sim.start_pose.assign(np.stack([self._X, self._Y, np.full(B, th, np.float32)], 1))
            sim.omega.zero_()
            sim.rollout_launch()
            der = sim.derived.numpy()[0]       # (z, pitch, roll) settled at each pose
            res = sim.residual.numpy()[0]
            clr = sim.clearance.numpy()[0]
            ti = np.arccos(np.clip(np.cos(der[:, 1]) * np.cos(der[:, 2]), -1.0, 1.0))
            blk = (res > self.resid_tol) | (clr < self.clear_margin) | (ti > self.tilt_max)
            blocked[:, :, t] = blk.reshape(self.ny, self.nx)
            tilt[:, :, t] = ti.reshape(self.ny, self.nx)
        return (wp.array(blocked, dtype=wp.float32, device=self.device),
                wp.array(tilt, dtype=wp.float32, device=self.device))

    def compute(self, elevation, mu, goal_xy):
        """elevation [ny, nx] + friction Heightmap mu + world goal -> clamped V[ny, nx, n_theta]."""
        blocked, tilt = self._settle_fields(elevation, mu)
        v = self.solver.compute_from_fields(blocked, tilt, goal_xy, self.bounds, tilt_weight=self.tilt_weight)
        wp.launch(_clamp3d_kernel, dim=(self.ny, self.nx, self.n_theta),
                  inputs=[v, self._vcap], outputs=[self.V], device=self.device)
        return self.V
