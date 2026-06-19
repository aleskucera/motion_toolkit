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
