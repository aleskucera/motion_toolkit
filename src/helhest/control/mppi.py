"""GPU-resident MPPI inner loop (sample -> rollout -> cost -> reweight) + CUDA graph.

The numpy MPPI loop is host-bound: per refine ~15 ms of numpy (cost, noise, target_wheel_omega)
vs ~2.7 ms of GPU work. These Warp kernels move the whole refine onto the device so
nothing but the executed pose ever comes back, and the refine is captured as a CUDA
graph and replayed. The cost kernel is validated by CONTRACT (analytic checks in
tests/control/test_mppi.py), not against a numpy twin.

Kernels (all suffixed _kernel):
  _sample_target_wheel_omega  spline-knot noise -> Ub -> writes the rollout's target_wheel_omega buffer (rear -> 0)
  _cost          per-rollout scalar cost J[B] (cost-to-go goal V^2 + graded-infeasible + effort/smooth)
  _minmax/_bisect_*/_count_below/_elite_u   CEM reweight (top-k elite mean) of U, on device
  _bump_seed/_reset_minmax     device-side RNG counter + reduction resets (graph-safe)
"""
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..engine.robot import Robot
from ..engine.simulator import ForwardSimulator
from ..engine.terrain import _locate
from ..engine.terrain import Grid
from ..profiling import StageProfiler


def _n_bisect(n_cand: int) -> int:
    """Device-side bisection steps for the CEM elite threshold tau. Each step halves the search
    interval, so n steps resolve tau to (jmax - jmin) / 2^n of the cost range. We want that finer
    than the spacing between candidate costs (~(jmax - jmin) / n_cand) so #{J <= tau} lands on
    target_k -> ceil(log2(n_cand)) bits + 5 (~32x margin). A FIXED count (not data-dependent) is what
    keeps the refine CUDA-graph-capturable -- a host sort/partition would need a readback + sync."""
    return int(np.ceil(np.log2(max(2, n_cand)))) + 5


@dataclass
class SamplingConfig:
    """MppiGpu config: how candidates are sampled + the elite selected each refine."""

    sigma: float = 0.5  # per-step Gaussian jitter on the wheel speeds
    sigma_knot: float = 1.0  # spline-knot noise -> smooth committed maneuvers
    n_knots: int = 4  # control knots spread over the horizon
    wmin: float = 0.0  # wheel-speed box [wmin, wmax]; wmin >= 0 -> no reverse
    wmax: float = 4.0
    wide_frac: float = 0.25  # fraction of candidates drawn from the WIDE global-search prior
    elite_frac: float = 0.02  # CEM top-k elite fraction


@wp.struct
class CostWeights:
    """Device-side per-rollout cost weights, passed into _cost_kernel. Built from CostParams (below).
    The robot's envelope + feasibility thresholds (roll/pitch limits, roll/pitch cost shape,
    clear_margin, resid_tol) are NOT here -- they come from the Robot struct, one shared source."""

    goal_terminal: float
    goal_running: float
    explore_fallback: float
    # the cost-to-go's unreachable cap (planner.cw.lattice_cap = ctg._vcap); V >= it means the goal
    # is unreachable in-window (the field is flat) -> arm the explore_fallback straight-line pull
    lattice_cap: float
    out_of_bounds: float
    effort: float
    smoothness: float
    infeasible: float


@dataclass(frozen=True)
class CostParams:  # host-side cost weights -- what you tune; build() -> the device CostWeights
    """The MPPI cost weights. The defaults are the lattice-routing tuning the demos run (routing +
    feasibility; the terminal dock handles reach+stop). A weight set to 0 disables its term in the
    kernel. The robot's envelope + feasibility thresholds are NOT here -- they live on the Robot
    struct (one shared source)."""

    goal_terminal: float = 3.0  # cost-to-go V^2 at the horizon end -> end the plan at the goal
    # cost-to-go V^2 averaged over the horizon -> make progress every step
    goal_running: float = 0.3
    # straight-line pull where V saturates (goal unreachable in-window)
    explore_fallback: float = 1.0
    out_of_bounds: float = 50.0  # soft wall just inside the world edge (V is clamped off-grid)
    effort: float = 2e-3  # penalize wheel-speed^2
    smoothness: float = 2e-3  # penalize wheel-speed CHANGES (jerk)
    # penalize clearance/residual/tip-over violations (does obstacle avoidance)
    infeasible: float = 1e5

    def build(self) -> CostWeights:
        cw = CostWeights()
        cw.goal_terminal = self.goal_terminal
        cw.goal_running = self.goal_running
        cw.explore_fallback = self.explore_fallback
        cw.lattice_cap = 1e9  # off until armed: planner.cw.lattice_cap = ctg._vcap
        cw.out_of_bounds = self.out_of_bounds
        cw.effort = self.effort
        cw.smoothness = self.smoothness
        cw.infeasible = self.infeasible
        return cw


@wp.func
def _bilinear(
    field: wp.array3d(dtype=float),
    yi: int,
    xi: int,
    fx: float,
    fy: float,
    t: int,
):
    """Bilinear read of the heading-t slice field[:, :, t] at the fractional cell (yi + fy, xi + fx)."""
    return (
        (1.0 - fx) * (1.0 - fy) * field[yi, xi, t]
        + fx * (1.0 - fy) * field[yi, xi + 1, t]
        + (1.0 - fx) * fy * field[yi + 1, xi, t]
        + fx * fy * field[yi + 1, xi + 1, t]
    )


@wp.func
def sample_lattice(
    field: wp.array3d(dtype=float),
    grid: Grid,
    n_theta: int,
    x: float,
    y: float,
    yaw: float,
):
    """Trilinear sample of the orientation-aware cost-to-go V(x, y, theta): bilinear in (x, y),
    linear in the (wrapped) heading. Misaligned poses read high/inf, so MPPI's rollouts prefer an
    approach the forward-only robot can actually complete."""
    c = _locate(grid, x, y)
    two_pi = 6.2831853
    dth = two_pi / float(n_theta)
    m = yaw - wp.floor(yaw / two_pi) * two_pi  # yaw mod 2pi in [0, 2pi)
    ft = m / dth
    ftf = wp.floor(ft)
    t0 = int(ftf) % n_theta
    t1 = (t0 + 1) % n_theta
    fth = ft - ftf
    fx = c[2]
    fy = c[3]
    xi = int(c[0])
    yi = int(c[1])
    va = _bilinear(field, yi, xi, fx, fy, t0)
    vb = _bilinear(field, yi, xi, fx, fy, t1)
    return (1.0 - fth) * va + fth * vb


@wp.func
def _knot_bracket(
    t: int,
    horizon: int,
    n_knots: int,
):
    """n_knots control knots evenly spaced over the horizon: the two bracketing step t and the
    interpolation fraction between them. Deterministic -> threads sharing a knot agree."""
    knot_spacing = float(horizon - 1) / float(n_knots - 1)
    knot_pos = float(t) / knot_spacing
    knot_lo = int(knot_pos)
    knot_hi = wp.min(knot_lo + 1, n_knots - 1)
    frac = knot_pos - float(knot_lo)
    return knot_lo, knot_hi, frac


@wp.kernel
def _sample_target_wheel_omega_kernel(
    U: wp.array2d(dtype=float),
    sigma: float,
    sigma_knot: float,
    wmin: float,
    wmax: float,
    n_wide: int,
    n_knots: int,
    seed: wp.array(dtype=int),
    target_wheel_omega: wp.array2d(dtype=wp.vec3),
):
    # one rollout per candidate: b = r. Candidate 0 keeps the nominal; the next n_wide draw from the
    # WIDE global-search prior; the rest jitter around the nominal (NARROW local refine).
    t, r = wp.tid()
    n_cand = target_wheel_omega.shape[1]
    b = r
    horizon = target_wheel_omega.shape[0]
    knot_lo, knot_hi, frac = _knot_bracket(t, horizon, n_knots)
    wheel_l = U[t, 0]
    wheel_r = U[t, 1]
    if b == 0:
        pass  # candidate 0 keeps the nominal (no noise)
    elif b < n_wide:
        # WIDE prior (global search): knots sampled UNIFORMLY over the full [wmin, wmax]
        # forward-arc box, independent of the nominal -> a broad variety of maneuvers
        # (the whole control space), so the elite can escape local minima.
        span = wmax - wmin
        left_lo = wmin + span * wp.randf(wp.rand_init(seed[0] + 1234, (b * n_knots + knot_lo) * 2))
        left_hi = wmin + span * wp.randf(wp.rand_init(seed[0] + 1234, (b * n_knots + knot_hi) * 2))
        right_lo = wmin + span * wp.randf(
            wp.rand_init(seed[0] + 1234, (b * n_knots + knot_lo) * 2 + 1)
        )
        right_hi = wmin + span * wp.randf(
            wp.rand_init(seed[0] + 1234, (b * n_knots + knot_hi) * 2 + 1)
        )
        wheel_l = (1.0 - frac) * left_lo + frac * left_hi
        wheel_r = (1.0 - frac) * right_lo + frac * right_hi
    else:
        # NARROW (local refine): SPLINE bias around the nominal + per-step jitter (option A).
        # Knots keyed on (b, knot) -> shared across t -> a smooth committed maneuver. Recompute
        # the bracketing knots on the fly (rand_init is deterministic) -- no shared storage.
        eps_l_lo = wp.randn(wp.rand_init(seed[0], (b * n_knots + knot_lo) * 2))
        eps_l_hi = wp.randn(wp.rand_init(seed[0], (b * n_knots + knot_hi) * 2))
        eps_r_lo = wp.randn(wp.rand_init(seed[0], (b * n_knots + knot_lo) * 2 + 1))
        eps_r_hi = wp.randn(wp.rand_init(seed[0], (b * n_knots + knot_hi) * 2 + 1))
        wheel_l += sigma_knot * ((1.0 - frac) * eps_l_lo + frac * eps_l_hi)
        wheel_r += sigma_knot * ((1.0 - frac) * eps_r_lo + frac * eps_r_hi)
        # light per-step jitter (distinct stream)
        jitter = wp.rand_init(seed[0] + 9176, t * n_cand + b)
        wheel_l += sigma * wp.randn(jitter)
        wheel_r += sigma * wp.randn(jitter)
    # clamp to the forward-arc box (wmin >= 0 -> no reverse)
    target_wheel_omega[t, r] = wp.vec3(wp.clamp(wheel_l, wmin, wmax), wp.clamp(wheel_r, wmin, wmax), 0.0)


@wp.kernel
def _cost_kernel(
    controlled: wp.array2d(dtype=wp.vec3),
    derived: wp.array2d(dtype=wp.vec3),
    clearance: wp.array2d(dtype=float),
    residual: wp.array2d(dtype=float),
    target_wheel_omega: wp.array2d(dtype=wp.vec3),  # Ub in components [0], [1]
    goal: wp.array(dtype=float),  # [2] world goal (device -> graph-safe, changes per replan)
    grid: Grid,  # geometry for sampling lattice_field at the rollout pose
    lattice_field: wp.array3d(
        dtype=float
    ),  # [ny, nx, n_theta] cost-to-go V(x,y,theta); the goal cost
    n_theta: int,
    cw: CostWeights,
    robot: Robot,  # envelope + feasibility thresholds (shared with the cost-to-go feasibility)
    horizon: int,
    Jout: wp.array(dtype=float),
):
    r = wp.tid()  # rollout
    run_sum = float(0.0)
    oob_sum = float(0.0)
    terminal_cost = float(0.0)
    edge = float(0.4)  # soft-wall margin inside the grid border
    x_lo = grid.origin_x + edge
    x_hi = grid.origin_x + float(grid.cells_x) * grid.cell_size - edge
    y_lo = grid.origin_y + edge
    y_hi = grid.origin_y + float(grid.cells_y) * grid.cell_size - edge
    for t in range(horizon + 1):
        pose = controlled[t, r]  # (x, y, yaw)
        if cw.out_of_bounds > 0.0:
            # soft wall at the world edge: depth past the margin (V is clamped off-grid, so the
            # goal term alone doesn't stop the robot driving off the map -- this does).
            oob_sum += wp.max(x_lo - pose[0], 0.0) + wp.max(pose[0] - x_hi, 0.0)
            oob_sum += wp.max(y_lo - pose[1], 0.0) + wp.max(pose[1] - y_hi, 0.0)
        # goal cost = the orientation-aware cost-to-go V(x,y,theta)^2 (routes around walls; misaligned
        # poses read high). Where V is SATURATED (>= cap, goal unreachable in-window so the field is
        # flat) fall back to a straight-line pull -> the robot EXPLORES toward the goal, not creeps.
        vl = sample_lattice(lattice_field, grid, n_theta, pose[0], pose[1], pose[2])
        if cw.explore_fallback > 0.0 and vl >= cw.lattice_cap * 0.9:
            dx = pose[0] - goal[0]
            dy = pose[1] - goal[1]
            goal_cost = cw.lattice_cap * cw.lattice_cap + cw.explore_fallback * (dx * dx + dy * dy)
        else:
            goal_cost = vl * vl
        run_sum += goal_cost
        terminal_cost = goal_cost  # last iter (t = horizon) sticks -> terminal goal cost
    effort_sum = float(0.0)
    smooth_sum = float(0.0)
    penalty_sum = float(0.0)
    prev_l = float(0.0)
    prev_r = float(0.0)
    for t in range(horizon):
        wheels = target_wheel_omega[t, r]  # (wL, wR)
        sp2 = wheels[0] * wheels[0] + wheels[1] * wheels[1]
        effort_sum += sp2
        if t > 0:
            dl = wheels[0] - prev_l
            dr = wheels[1] - prev_r
            smooth_sum += dl * dl + dr * dr
        prev_l = wheels[0]
        prev_r = wheels[1]
        # GRADED validity (option C): penalize HOW FAR past the margin/tol and HOW EARLY, not a
        # binary flag. De-saturates the cost (it still ranks when every sample violates), and
        # eating into the safety margin costs little while a real penetration costs a lot.
        clear_viol = wp.max(robot.clear_margin - clearance[t, r], 0.0)
        resid_viol = wp.max(residual[t, r] - robot.resid_tol, 0.0)
        # roll/pitch stability envelope (same limits as the cost-to-go feasibility): tipping is
        # invalid. climbing is nose-up = NEGATIVE pitch, so the climb limit is on -pitch.
        pitch = derived[t, r][1]
        roll = derived[t, r][2]
        roll_viol = wp.max(wp.abs(roll) - robot.max_roll, 0.0)
        climb_viol = wp.max(-pitch - robot.max_pitch_up, 0.0)
        descend_viol = wp.max(pitch - robot.max_pitch_down, 0.0)
        early = float(horizon - t) / float(horizon)  # earlier violations hurt more (imminent)
        penalty_sum += early * (clear_viol + resid_viol + roll_viol + climb_viol + descend_viol)
    # goal_running is a mean over the horizon; effort/smoothness are raw sums (so they scale with the
    # horizon) -- the weights are tuned to that, mind it if the horizon changes. (Reaching + stopping at
    # the goal, and the right approach heading, are the cost-to-go + dock controller's job -- no
    # heading/endgame term.)
    Jout[r] = (
        cw.goal_terminal * terminal_cost
        + cw.goal_running * (run_sum / float(horizon + 1))
        + cw.out_of_bounds * oob_sum
        + cw.effort * effort_sum
        + cw.smoothness * smooth_sum
        + penalty_sum * cw.infeasible
    )


# --- CEM reweight (option B): elite = top-k lowest-cost candidates; U = their mean. Rank-based,
# so the validity penalty can't blow up the weighting (invalid samples just don't make the
# elite). The top-k threshold tau is found by device-side BISECTION (a host partition would
# break the CUDA graph): bisect tau until #{J <= tau} ~= target_k. ---
@wp.kernel
def _reset_minmax_kernel(
    jmin: wp.array(dtype=float), jmax: wp.array(dtype=float), count: wp.array(dtype=float)
):
    jmin[0] = 1.0e30
    jmax[0] = -1.0e30
    count[0] = 0.0


@wp.kernel
def _minmax_kernel(
    J: wp.array(dtype=float), jmin: wp.array(dtype=float), jmax: wp.array(dtype=float)
):
    cost = J[wp.tid()]
    wp.atomic_min(jmin, 0, cost)
    wp.atomic_max(jmax, 0, cost)


@wp.kernel
def _bisect_init_kernel(
    jmin: wp.array(dtype=float),
    jmax: wp.array(dtype=float),
    tau_lo: wp.array(dtype=float),
    tau_hi: wp.array(dtype=float),
    tau: wp.array(dtype=float),
    count: wp.array(dtype=float),
):
    tau_lo[0] = jmin[0]
    tau_hi[0] = jmax[0]
    tau[0] = 0.5 * (jmin[0] + jmax[0])
    count[0] = 0.0


@wp.kernel
def _count_below_kernel(
    J: wp.array(dtype=float),
    tau: wp.array(dtype=float),
    count: wp.array(dtype=float),
):
    if J[wp.tid()] <= tau[0]:
        wp.atomic_add(count, 0, 1.0)


@wp.kernel
def _bisect_step_kernel(
    count: wp.array(dtype=float),
    target_k: float,
    tau_lo: wp.array(dtype=float),
    tau_hi: wp.array(dtype=float),
    tau: wp.array(dtype=float),
):
    if count[0] > target_k:
        tau_hi[0] = tau[0]  # too many below tau -> lower it
    else:
        tau_lo[0] = tau[0]  # too few -> raise it
    tau[0] = 0.5 * (tau_lo[0] + tau_hi[0])
    count[0] = 0.0  # reset for the next count pass


@wp.kernel
def _elite_u_kernel(
    J: wp.array(dtype=float),
    tau: wp.array(dtype=float),
    count: wp.array(dtype=float),
    target_wheel_omega: wp.array2d(dtype=wp.vec3),
    wmin: float,
    wmax: float,
    n_cand: int,
    U: wp.array2d(dtype=float),
):
    t, wheel = wp.tid()  # (timestep, wheel: 0=L, 1=R)
    elite_sum = float(0.0)
    for b in range(n_cand):
        if J[b] <= tau[0]:  # elite candidate
            elite_sum += target_wheel_omega[t, b][wheel]
    U[t, wheel] = wp.clamp(elite_sum / count[0], wmin, wmax)  # unweighted elite mean (forward arcs)


@wp.kernel
def _bump_seed_kernel(seed: wp.array(dtype=int)):
    seed[0] = seed[0] + 1


class MppiGpu:
    """GPU-resident MPPI: owns the nominal control `U` + scratch on device and runs the
    refine (sample -> rollout -> cost -> CEM reweight) entirely on the GPU. On CUDA
    the refine is captured once and replayed as a graph (the RNG counter is bumped
    in-graph, so each replay draws fresh noise); on CPU it runs eager. Wraps a ForwardSimulator.

    `goal` and `start_pose` are device arrays set per replan, so the captured graph picks
    up new values; the weights/sigma/wmax/elite_frac are baked at capture (fixed per planner)."""

    def __init__(
        self,
        sim: ForwardSimulator,
        cost: CostParams,  # cost weights (host) -> built into the device CostWeights struct
        sampling: SamplingConfig = SamplingConfig(),  # noise / wheel-speed box / elite fraction
        n_theta: int = 16,
        seed: int = 0,
        profile: bool = False,
    ):
        sampling = sampling or SamplingConfig()

        self.sim = sim
        self.device = sim.device

        self.n_rollouts, self.horizon = sim.batch_size, sim.n_steps

        # one rollout per candidate (plain MPPI)
        self.n_cand = self.n_rollouts
        self.n_bisect = _n_bisect(self.n_cand)  # CEM threshold bisection steps (scales with n_cand)
        self.sampling = sampling
        self.n_wide = int(sampling.wide_frac * self.n_cand)  # candidates drawn from the WIDE prior

        # CEM elite count (over candidates)
        self.target_k = float(int(sampling.elite_frac * self.n_cand))
        self.cw = cost.build()  # host CostParams -> device CostWeights struct (weights only)
        # the robot's envelope/shape + feasibility thresholds (max_roll/pitch, roll/pitch cost shape,
        # clear_margin, resid_tol) are read straight from sim.robot in the cost kernel -- not copied here.

        self.robot = sim.robot
        with wp.ScopedDevice(self.device):
            self.U = wp.zeros((self.horizon, 2), dtype=wp.float32)
            self.J = wp.zeros(self.n_rollouts, dtype=wp.float32)  # cost per candidate rollout
            self.jmin = wp.zeros(1, dtype=wp.float32)  # CEM bisection scalars
            self.jmax = wp.zeros(1, dtype=wp.float32)
            self.tau_lo = wp.zeros(1, dtype=wp.float32)
            self.tau_hi = wp.zeros(1, dtype=wp.float32)
            self.tau = wp.zeros(1, dtype=wp.float32)
            self.count = wp.zeros(1, dtype=wp.float32)
            self.seed = wp.array([int(seed)], dtype=wp.int32)
            self.goal = wp.zeros(2, dtype=wp.float32)
            ny, nx = sim.elevation.shape
            self.n_theta = int(n_theta)
            self.lattice_field = wp.zeros((ny, nx, n_theta), dtype=wp.float32)  # V(x,y,theta)

        # the grid the cost kernel samples the lattice field on: defaults to the sim grid, but a COARSER
        # grid can be set (set_lattice(V, grid)) so the routing field is solved at low resolution --
        # the rollouts do fine obstacle avoidance, so the global router needn't be sim-resolution.
        self.lattice_grid = sim.grid
        self._graph = None

        # opt-in per-stage profiling of the captured refine loop (CUDA-event timing; off = no overhead)
        self._prof = StageProfiler(self.device, ("sample", "rollout", "cost", "reweight"), profile)
        self._n_refine_done = 0

    def reset_timing(self):
        """Clear the accumulated per-stage refine timings (e.g. after a warmup replan)."""
        self._prof.reset()

    def timing_stats(self):
        """Per-stage refine timing over profiled replans (CUDA + profile=True), first refine excluded:
        {stage: {"mean_ms", "std_ms", "n"}} for sample / rollout / cost / reweight. Use the means (the
        event reads sync, so a profiling run is serialized -- its wall-clock isn't the real rate).
        """
        return self._prof.stats()

    def reset_nominal(self, value=1.5):
        self.U.fill_(float(value))

    def nominal(self):
        """The current nominal control U [T, 2], on host."""
        return self.U.numpy()

    def set_nominal(self, U_host):
        self.U.assign(np.ascontiguousarray(U_host, np.float32))

    def set_lattice(self, V, grid=None):
        """Copy the orientation-aware cost-to-go V[ny', nx', n_theta] into the stable buffer the cost
        kernel reads. `grid` is the Grid V was solved on (a coarse grid for a low-res routing field);
        defaults to the sim grid. Call before the first replan; on re-solve (moving goal) call again
        with the SAME shape -- it copies into the stable buffer the captured graph reads."""
        if tuple(V.shape) != tuple(self.lattice_field.shape):
            self.lattice_field = wp.zeros(V.shape, dtype=float, device=self.device)
        wp.copy(self.lattice_field, V)
        if grid is not None:
            self.lattice_grid = grid

    def _refine(self):
        """One MPPI iteration: sample -> rollout -> cost -> CEM reweight, all on device."""
        self._prof.mark(0)
        wp.launch(_bump_seed_kernel, 1, inputs=[self.seed], device=self.device)
        wp.launch(
            _sample_target_wheel_omega_kernel,
            (self.horizon, self.n_rollouts),
            inputs=[
                self.U,
                self.sampling.sigma,
                self.sampling.sigma_knot,
                self.sampling.wmin,
                self.sampling.wmax,
                self.n_wide,
                self.sampling.n_knots,
                self.seed,
            ],
            outputs=[self.sim.target_wheel_omega],
            device=self.device,
        )
        self._prof.mark(1)  # sample done
        self.sim.rollout_launch()
        self._prof.mark(2)  # rollout done
        wp.launch(
            _cost_kernel,
            self.n_rollouts,
            inputs=[
                self.sim.controlled,
                self.sim.derived,
                self.sim.clearance,
                self.sim.residual,
                self.sim.target_wheel_omega,
                self.goal,
                self.lattice_grid,
                self.lattice_field,
                self.n_theta,
                self.cw,
                self.robot,
                self.horizon,
            ],
            outputs=[self.J],
            device=self.device,
        )
        self._prof.mark(3)  # cost done
        self._cem_reweight()
        self._prof.mark(4)  # reweight (CEM) done

    def _cem_reweight(self):
        """Top-k elite mean -> U over candidates (cost = J): find the threshold tau by device-side
        bisection (#{J <= tau} ~= target_k), then average the elite candidates' controls."""

        wp.launch(
            _reset_minmax_kernel,
            1,
            inputs=[self.jmin, self.jmax, self.count],
            device=self.device,
        )
        wp.launch(
            _minmax_kernel,
            self.n_cand,
            inputs=[self.J, self.jmin, self.jmax],
            device=self.device,
        )
        wp.launch(
            _bisect_init_kernel,
            1,
            inputs=[
                self.jmin,
                self.jmax,
                self.tau_lo,
                self.tau_hi,
                self.tau,
                self.count,
            ],
            device=self.device,
        )
        for _ in range(self.n_bisect):
            wp.launch(
                _count_below_kernel,
                self.n_cand,
                inputs=[self.J, self.tau, self.count],
                device=self.device,
            )
            wp.launch(
                _bisect_step_kernel,
                1,
                inputs=[self.count, self.target_k, self.tau_lo, self.tau_hi, self.tau],
                device=self.device,
            )
        wp.launch(
            _count_below_kernel,
            self.n_cand,
            inputs=[self.J, self.tau, self.count],
            device=self.device,
        )  # final elite count

        wp.launch(
            _elite_u_kernel,
            (self.horizon, 2),
            inputs=[
                self.J,
                self.tau,
                self.count,
                self.sim.target_wheel_omega,
                self.sampling.wmin,
                self.sampling.wmax,
                self.n_cand,
                self.U,
            ],
            device=self.device,
        )

    def replan(self, state, goal_xy, n_refine):
        """Run n_refine GPU refines from `state` toward world `goal_xy`; updates U in place."""
        self.goal.assign(np.asarray(goal_xy[:2], np.float32))
        self.sim.start_pose.assign(
            np.ascontiguousarray(
                np.tile(np.asarray(state, np.float32), (self.n_rollouts, 1)), np.float32
            )
        )
        if self.device.is_cuda:
            if self._graph is None:
                with wp.ScopedCapture(device=self.device) as cap:
                    self._refine()
                self._graph = cap.graph
            for _ in range(n_refine):
                wp.capture_launch(self._graph)
                self._n_refine_done += 1
                if self._prof.enabled and self._n_refine_done > 1:  # skip the first (cold) refine
                    self._prof.accumulate()
        else:
            for _ in range(n_refine):
                self._refine()
        return self.U
