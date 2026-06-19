"""GPU-resident MPPI inner loop (sample -> rollout -> cost -> reweight) + CUDA graph.

The numpy MPPI loop is host-bound: per refine ~15 ms of numpy (cost, noise, omega)
vs ~2.7 ms of GPU work. These Warp kernels move the whole refine onto the device so
nothing but the executed pose ever comes back, and the refine is captured as a CUDA
graph and replayed. `mppi._cost` stays the numpy oracle for the parity test.

Kernels (all suffixed _kernel):
  _sample_omega  spline-knot noise -> Ub -> writes the rollout's omega buffer (rear -> 0)
  _cost          per-rollout scalar cost J[B] (goal/tilt/graded-invalid + eff/smooth)
  _minmax/_bisect_*/_count_below/_elite_u   CEM reweight (top-k elite mean) of U, on device
  _bump_seed/_reset_minmax     device-side RNG counter + reduction resets (graph-safe)
"""
import numpy as np
import warp as wp

from ..engine.terrain import Grid
from ..engine.terrain import _locate
from ..engine.terrain import sample_field

_N_BISECT = 14  # device-side bisection iterations for the CEM top-k threshold


@wp.func
def sample_lattice(field: wp.array3d(dtype=float), grid: Grid, n_theta: int,
                   x: float, y: float, yaw: float):
    """Trilinear sample of the orientation-aware cost-to-go V(x, y, theta): bilinear in (x, y),
    linear in the (wrapped) heading. Misaligned poses read high/inf, so MPPI's rollouts prefer an
    approach the forward-only robot can actually complete."""
    c = _locate(grid, x, y)
    two_pi = 6.2831853
    dth = two_pi / float(n_theta)
    m = yaw - wp.floor(yaw / two_pi) * two_pi  # yaw mod 2pi in [0, 2pi)
    ft = m / dth
    t0 = int(wp.floor(ft)) % n_theta
    t1 = (t0 + 1) % n_theta
    fth = ft - wp.floor(ft)
    fx = c.frac_x
    fy = c.frac_y
    xi = c.x_idx
    yi = c.y_idx
    va = ((1.0 - fx) * (1.0 - fy) * field[yi, xi, t0]
          + fx * (1.0 - fy) * field[yi, xi + 1, t0]
          + (1.0 - fx) * fy * field[yi + 1, xi, t0]
          + fx * fy * field[yi + 1, xi + 1, t0])
    vb = ((1.0 - fx) * (1.0 - fy) * field[yi, xi, t1]
          + fx * (1.0 - fy) * field[yi, xi + 1, t1]
          + (1.0 - fx) * fy * field[yi + 1, xi, t1]
          + fx * fy * field[yi + 1, xi + 1, t1])
    return (1.0 - fth) * va + fth * vb


@wp.func
def _knot_bracket(t: int, T: int, n_knots: int):
    """n_knots control knots evenly spaced over the horizon: the two bracketing step t and the
    interpolation fraction between them. Deterministic -> threads sharing a knot agree."""
    knot_spacing = float(T - 1) / float(n_knots - 1)
    knot_pos = float(t) / knot_spacing
    knot_lo = int(knot_pos)
    knot_hi = wp.min(knot_lo + 1, n_knots - 1)
    frac = knot_pos - float(knot_lo)
    return knot_lo, knot_hi, frac


@wp.kernel
def _sample_omega_kernel(
    U: wp.array2d(dtype=float),
    sigma: float,
    sigma_knot: float,
    wmin: float,
    wmax: float,
    n_wide: int,
    n_knots: int,
    n_scen: int,                     # K slip scenarios per candidate (1 = non-robust)
    slip: wp.array2d(dtype=float),   # [K, 2] wheel-speed retention; scenario 0 = (1, 1)
    seed: wp.array(dtype=int),
    omega: wp.array2d(dtype=wp.vec3),
):
    # rollout r = candidate b * K + scenario k: all K scenarios of a candidate share the SAME
    # sampled control (noise keyed on b), then scenario k scales the wheels by its slip. So the
    # cost sees how each candidate holds up across the disturbance. K=1 -> b=r, slip=(1,1) (today).
    t, r = wp.tid()
    n_cand = omega.shape[1] // n_scen
    b = r // n_scen
    k = r % n_scen
    T = omega.shape[0]
    knot_lo, knot_hi, frac = _knot_bracket(t, T, n_knots)
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
        right_lo = wmin + span * wp.randf(wp.rand_init(seed[0] + 1234, (b * n_knots + knot_lo) * 2 + 1))
        right_hi = wmin + span * wp.randf(wp.rand_init(seed[0] + 1234, (b * n_knots + knot_hi) * 2 + 1))
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
        jitter = wp.rand_init(seed[0] + 9176, t * n_cand + b)  # light per-step jitter (distinct stream)
        wheel_l += sigma * wp.randn(jitter)
        wheel_r += sigma * wp.randn(jitter)
    # apply scenario k's wheel slip, then clamp to the forward-arc box (wmin >= 0 -> no reverse)
    omega[t, r] = wp.vec3(wp.clamp(wheel_l * slip[k, 0], wmin, wmax),
                          wp.clamp(wheel_r * slip[k, 1], wmin, wmax), 0.0)


@wp.struct
class CostWeights:
    """Per-rollout cost weights + thresholds, baked once (fixed per planner)."""

    term: float
    run: float
    tilt: float
    head: float
    ctg: float  # >0 -> goal term is the cost-to-go field V(x,y), else Euclidean distance^2
    lattice: float  # >0 -> goal term is the orientation-aware cost-to-go V(x,y,theta)
    oob: float  # >0 -> penalize leaving the world (soft wall at the grid edge)
    term_v: float  # >0 -> penalize speed at the horizon end (plan should END stopped at the goal)
    eff: float
    smooth: float
    invalid: float
    tilt_free: float
    clear_margin: float
    resid_tol: float


@wp.kernel
def _cost_kernel(
    controlled: wp.array2d(dtype=wp.vec3),
    derived: wp.array2d(dtype=wp.vec3),
    clearance: wp.array2d(dtype=float),
    residual: wp.array2d(dtype=float),
    omega: wp.array2d(dtype=wp.vec3),  # Ub in components [0], [1]
    goal: wp.array(dtype=float),  # [2] world goal (device -> graph-safe, changes per replan)
    ctg_field: wp.array2d(dtype=float),  # [ny, nx] cost-to-go V (only read when cw.ctg > 0)
    grid: Grid,  # geometry for sampling ctg_field / lattice_field at the rollout pose
    lattice_field: wp.array3d(dtype=float),  # [ny, nx, n_theta] V(x,y,theta) (only read when cw.lattice > 0)
    n_theta: int,
    cw: CostWeights,
    T: int,
    Jout: wp.array(dtype=float),
):
    r = wp.tid()  # rollout
    run_sum = float(0.0)
    tilt_sum = float(0.0)
    heading_sum = float(0.0)
    oob_sum = float(0.0)
    terminal_cost = float(0.0)
    edge = float(0.4)  # soft-wall margin inside the grid border
    x_lo = grid.origin_x + edge
    x_hi = grid.origin_x + float(grid.cells_x) * grid.cell_size - edge
    y_lo = grid.origin_y + edge
    y_hi = grid.origin_y + float(grid.cells_y) * grid.cell_size - edge
    for t in range(T + 1):
        pose = controlled[t, r]  # (x, y, yaw)
        dx = pose[0] - goal[0]
        dy = pose[1] - goal[1]
        goal_d2 = dx * dx + dy * dy  # Euclidean; still drives the heading term below
        if cw.oob > 0.0:
            # soft wall at the world edge: depth past the margin (V is clamped off-grid, so the
            # goal term alone doesn't stop the robot driving off the map -- this does).
            oob_sum += wp.max(x_lo - pose[0], 0.0) + wp.max(pose[0] - x_hi, 0.0)
            oob_sum += wp.max(y_lo - pose[1], 0.0) + wp.max(pose[1] - y_hi, 0.0)
        # goal term: obstacle-aware cost-to-go V(x,y)^2 (routes around the wall) when enabled,
        # else straight-line distance^2. cw is uniform across rollouts -> warp-coherent branch.
        if cw.lattice > 0.0:
            vl = sample_lattice(lattice_field, grid, n_theta, pose[0], pose[1], pose[2])
            goal_cost = vl * vl  # orientation-aware: misaligned poses read high -> penalized
        elif cw.ctg > 0.0:
            v = sample_field(ctg_field, grid, pose[0], pose[1])
            goal_cost = v * v
        else:
            goal_cost = goal_d2
        run_sum += goal_cost
        terminal_cost = goal_cost  # last iter (t = T) sticks -> terminal goal cost
        # tilt + heading are optional; cw is uniform across rollouts, so these guards are
        # warp-coherent (no divergence) and skip the trig (and the derived read) when off.
        if cw.tilt > 0.0:
            pitch = derived[t, r][1]
            roll = derived[t, r][2]
            tilt_angle = wp.acos(wp.clamp(wp.cos(pitch) * wp.cos(roll), -1.0, 1.0))
            excess = wp.max(tilt_angle - cw.tilt_free, 0.0)
            tilt_sum += excess * excess
        if cw.head > 0.0:
            # heading: penalize facing AWAY from the desired direction (1 - cos angle, in [0, 2]).
            # Makes "turn the right way" cheaper than "stay", so a forward-only robot commits to a
            # U-turn / detour (it can't reverse or spin in place).
            if cw.ctg > 0.0:
                # follow the cost-to-go descent direction -grad V (the way AROUND the wall), by
                # central differences of V -- NOT the straight line to the goal, which points
                # across the wall and fights the routing.
                e = grid.cell_size
                gvx = sample_field(ctg_field, grid, pose[0] + e, pose[1]) - sample_field(ctg_field, grid, pose[0] - e, pose[1])
                gvy = sample_field(ctg_field, grid, pose[0], pose[1] + e) - sample_field(ctg_field, grid, pose[0], pose[1] - e)
                gnorm = wp.sqrt(gvx * gvx + gvy * gvy)
                if gnorm > 1e-6:  # grad V ~ 0 at the goal (flat minimum) -> no heading pressure
                    facing = -(wp.cos(pose[2]) * gvx + wp.sin(pose[2]) * gvy) / gnorm
                    heading_sum += 1.0 - facing
            else:
                goal_dist = wp.sqrt(goal_d2)
                if goal_dist > 1e-3:
                    facing = -(wp.cos(pose[2]) * dx + wp.sin(pose[2]) * dy) / goal_dist
                    heading_sum += 1.0 - facing
    effort_sum = float(0.0)
    smooth_sum = float(0.0)
    penalty_sum = float(0.0)
    term_speed = float(0.0)
    prev_l = float(0.0)
    prev_r = float(0.0)
    for t in range(T):
        wheels = omega[t, r]  # (wL, wR)
        sp2 = wheels[0] * wheels[0] + wheels[1] * wheels[1]
        effort_sum += sp2
        if t == T - 1:
            term_speed = sp2  # speed at the horizon end -> 0 means the plan stops (at the goal)
        if t > 0:
            dl = wheels[0] - prev_l
            dr = wheels[1] - prev_r
            smooth_sum += dl * dl + dr * dr
        prev_l = wheels[0]
        prev_r = wheels[1]
        # GRADED validity (option C): penalize HOW FAR past the margin/tol and HOW EARLY, not a
        # binary flag. De-saturates the cost (it still ranks when every sample violates), and
        # eating into the safety margin costs little while a real penetration costs a lot.
        clear_viol = wp.max(cw.clear_margin - clearance[t, r], 0.0)
        resid_viol = wp.max(residual[t, r] - cw.resid_tol, 0.0)
        early = float(T - t) / float(T)  # earlier violations hurt more (imminent)
        penalty_sum += early * (clear_viol + resid_viol)
    # run/tilt/heading are means over the horizon; effort/smooth are raw sums (so they scale with
    # T) -- the weights are tuned to that, mind it if T changes.
    Jout[r] = (
        cw.term * terminal_cost
        + cw.run * (run_sum / float(T + 1))
        + cw.tilt * (tilt_sum / float(T + 1))
        + cw.head * (heading_sum / float(T + 1))
        + cw.oob * oob_sum
        + cw.term_v * term_speed
        + cw.eff * effort_sum
        + cw.smooth * smooth_sum
        + penalty_sum * cw.invalid
    )


# --- Robust eval (option F as risk-aware planning): each candidate is rolled out under K slip
# scenarios; CVaR(J) = mean of its worst m_tail scenarios is the candidate's cost. A path that
# hugs an obstacle is cheap nominally but its slip fan high-centers -> bad CVaR, so clearance
# falls out of robustness (no margin set). K=1 -> J_cand = J (today's behaviour). ---
@wp.kernel
def _cvar_kernel(J: wp.array(dtype=float), n_scen: int, m_tail: int, J_cand: wp.array(dtype=float)):
    b = wp.tid()  # candidate
    base = b * n_scen
    acc = float(0.0)
    for i in range(n_scen):
        ci = J[base + i]
        worse = int(0)  # scenarios strictly worse (ties broken by index) -> a clean worst-m set
        for j in range(n_scen):
            cj = J[base + j]
            if cj > ci or (cj == ci and j < i):
                worse += 1
        if worse < m_tail:
            acc += ci
    J_cand[b] = acc / float(m_tail)


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
    J: wp.array(dtype=float), tau: wp.array(dtype=float), count: wp.array(dtype=float)
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
    J_cand: wp.array(dtype=float),
    tau: wp.array(dtype=float),
    count: wp.array(dtype=float),
    omega: wp.array2d(dtype=wp.vec3),
    n_scen: int,
    wmin: float,
    wmax: float,
    n_cand: int,
    U: wp.array2d(dtype=float),
):
    t, wheel = wp.tid()  # (timestep, wheel: 0=L, 1=R)
    elite_sum = float(0.0)
    for b in range(n_cand):
        if J_cand[b] <= tau[0]:  # elite candidate
            elite_sum += omega[t, b * n_scen][wheel]  # scenario 0 = the un-slipped control
    U[t, wheel] = wp.clamp(elite_sum / count[0], wmin, wmax)  # unweighted elite mean (forward arcs)


@wp.kernel
def _bump_seed_kernel(seed: wp.array(dtype=int)):
    seed[0] = seed[0] + 1


class MppiGpu:
    """GPU-resident MPPI: owns the nominal control `U` + scratch on device and runs the
    refine (sample -> rollout -> cost -> CEM reweight) entirely on the GPU. On CUDA
    the refine is captured once and replayed as a graph (the RNG counter is bumped
    in-graph, so each replay draws fresh noise); on CPU it runs eager. Wraps a Simulator.

    `goal` and `start_pose` are device arrays set per replan, so the captured graph picks
    up new values; the weights/sigma/wmax/elite_frac are baked at capture (fixed per planner)."""

    def __init__(
        self,
        sim,
        sigma,
        wmax,
        weights,
        clear_margin,
        resid_tol,
        seed=0,
        sigma_knot=0.0,
        n_knots=4,
        elite_frac=0.02,
        wmin=0.0,
        wide_frac=0.25,
        n_scenarios=1,
        cvar_beta=0.5,
        slip_lo=0.6,
        n_theta=16,
    ):
        self.sim = sim
        self.dev = sim.device
        self.B, self.T = sim.B, sim.T
        # robust eval (option F): the B rollouts are n_cand candidates x K slip scenarios.
        # K=1 is non-robust (one scenario, no slip) -> identical to plain MPPI.
        self.K = int(n_scenarios)
        if self.B % self.K != 0:
            raise ValueError(f"sim.B ({self.B}) must be divisible by n_scenarios ({self.K})")
        self.n_cand = self.B // self.K
        self.m_tail = max(1, int(round(float(cvar_beta) * self.K)))  # CVaR = mean of worst m_tail
        self.sigma, self.wmax, self.wmin = float(sigma), float(wmax), float(wmin)
        self.sigma_knot, self.n_knots = float(sigma_knot), int(n_knots)
        self.n_wide = int(float(wide_frac) * self.n_cand)  # candidates drawn from the WIDE prior
        self.target_k = float(int(float(elite_frac) * self.n_cand))  # CEM elite count (over candidates)
        w = weights
        cw = CostWeights()
        cw.term = float(w.get("term", 0.0))
        cw.run = float(w.get("run", 0.0))
        cw.tilt = float(w.get("tilt", 0.0))
        cw.head = float(w.get("head", 0.0))
        cw.ctg = float(w.get("ctg", 0.0))
        cw.lattice = float(w.get("lattice", 0.0))
        cw.oob = float(w.get("oob", 0.0))
        cw.term_v = float(w.get("term_v", 0.0))
        cw.eff = float(w.get("eff", 0.0))
        cw.smooth = float(w.get("smooth", 0.0))
        cw.invalid = float(w.get("invalid", 0.0))
        cw.tilt_free = float(w.get("tilt_free", 0.0))
        cw.clear_margin = float(clear_margin)
        cw.resid_tol = float(resid_tol)
        self.cw = cw
        d = self.dev
        slip = np.ones((self.K, 2), np.float32)  # scenario 0 = no slip; rest sample the disturbance
        if self.K > 1:
            rng = np.random.default_rng(int(seed) + 4242)
            slip[1:] = rng.uniform(float(slip_lo), 1.0, (self.K - 1, 2)).astype(np.float32)
        self.slip = wp.array(slip, dtype=float, device=d)
        self.U = wp.zeros((self.T, 2), dtype=float, device=d)
        self.J = wp.zeros(self.B, dtype=float, device=d)            # cost per scenario-rollout
        self.J_cand = wp.zeros(self.n_cand, dtype=float, device=d)  # CVaR cost per candidate
        self.jmin = wp.zeros(1, dtype=float, device=d)  # CEM bisection scalars
        self.jmax = wp.zeros(1, dtype=float, device=d)
        self.tau_lo = wp.zeros(1, dtype=float, device=d)
        self.tau_hi = wp.zeros(1, dtype=float, device=d)
        self.tau = wp.zeros(1, dtype=float, device=d)
        self.count = wp.zeros(1, dtype=float, device=d)
        self.seed = wp.array([int(seed)], dtype=int, device=d)
        self.goal = wp.zeros(2, dtype=float, device=d)
        # cost-to-go field V[ny, nx] (option E): a stable buffer the cost kernel samples;
        # zeros + cw.ctg == 0 means it's ignored. set_costtogo() copies V in (graph-safe).
        ny, nx = sim.elevation.shape
        self.ctg_field = wp.zeros((ny, nx), dtype=float, device=d)
        self.n_theta = int(n_theta)
        self.lattice_field = wp.zeros((ny, nx, self.n_theta), dtype=float, device=d)  # V(x,y,theta)
        self._graph = None

    def reset_nominal(self, value=1.5):
        self.U.fill_(float(value))

    def nominal(self):
        """The current nominal control U [T, 2], on host."""
        return self.U.numpy()

    def set_nominal(self, U_host):
        self.U.assign(np.ascontiguousarray(U_host, np.float32))

    def set_costtogo(self, V):
        """Copy the cost-to-go field V[ny, nx] into the stable buffer the cost kernel reads
        (so a captured graph picks up new contents). Call before the first replan to enable
        the cost-to-go goal term -- requires the planner to have been built with ctg weight > 0."""
        wp.copy(self.ctg_field, V)

    def set_lattice(self, V):
        """Copy the orientation-aware cost-to-go V[ny, nx, n_theta] into the stable buffer the cost
        kernel reads. Call before the first replan; requires the planner built with lattice weight > 0."""
        wp.copy(self.lattice_field, V)

    def _refine(self):
        """One MPPI iteration: sample -> rollout -> cost -> CEM reweight, all on device."""
        s, d = self.sim, self.dev
        wp.launch(_bump_seed_kernel, 1, inputs=[self.seed], device=d)
        wp.launch(
            _sample_omega_kernel,
            (self.T, self.B),
            inputs=[
                self.U,
                self.sigma,
                self.sigma_knot,
                self.wmin,
                self.wmax,
                self.n_wide,
                self.n_knots,
                self.K,
                self.slip,
                self.seed,
            ],
            outputs=[s.omega],
            device=d,
        )
        s.rollout_launch()
        wp.launch(
            _cost_kernel,
            self.B,
            inputs=[
                s.controlled,
                s.derived,
                s.clearance,
                s.residual,
                s.omega,
                self.goal,
                self.ctg_field,
                s.grid,
                self.lattice_field,
                self.n_theta,
                self.cw,
                self.T,
            ],
            outputs=[self.J],
            device=d,
        )
        # robust eval: reduce each candidate's K scenario costs to its CVaR (K=1 -> J_cand = J)
        wp.launch(_cvar_kernel, self.n_cand, inputs=[self.J, self.K, self.m_tail],
                  outputs=[self.J_cand], device=d)
        self._cem_reweight()

    def _cem_reweight(self):
        """Top-k elite mean -> U over CANDIDATES (cost = J_cand, the CVaR): find the threshold
        tau by device-side bisection (#{J_cand <= tau} ~= target_k), then average the elite
        candidates' un-slipped controls (scenario-0 column)."""
        d, omega = self.dev, self.sim.omega
        wp.launch(_reset_minmax_kernel, 1, inputs=[self.jmin, self.jmax, self.count], device=d)
        wp.launch(_minmax_kernel, self.n_cand, inputs=[self.J_cand, self.jmin, self.jmax], device=d)
        wp.launch(
            _bisect_init_kernel,
            1,
            inputs=[self.jmin, self.jmax, self.tau_lo, self.tau_hi, self.tau, self.count],
            device=d,
        )
        for _ in range(_N_BISECT):
            wp.launch(_count_below_kernel, self.n_cand, inputs=[self.J_cand, self.tau, self.count], device=d)
            wp.launch(
                _bisect_step_kernel,
                1,
                inputs=[self.count, self.target_k, self.tau_lo, self.tau_hi, self.tau],
                device=d,
            )
        wp.launch(
            _count_below_kernel, self.n_cand, inputs=[self.J_cand, self.tau, self.count], device=d
        )  # final elite count
        wp.launch(
            _elite_u_kernel,
            (self.T, 2),
            inputs=[self.J_cand, self.tau, self.count, omega, self.K,
                    self.wmin, self.wmax, self.n_cand, self.U],
            device=d,
        )

    def _seed_turn_if_facing_away(self, state, goal_xy):
        """Symmetry break for the behind-goal saddle: if the start faces >120 deg away from
        the goal, seed the nominal with a hard turn. A forward-only robot facing away can only
        reach by a U-turn, but the straight nominal is a local optimum the sampler won't leave
        (it just drives away); the turn seed kicks it into the U-turn basin. Fires only when
        facing away, so normal (facing-toward) planning keeps its warm-started nominal."""
        to_goal_x = float(goal_xy[0]) - float(state[0])
        to_goal_y = float(goal_xy[1]) - float(state[1])
        dist = np.hypot(to_goal_x, to_goal_y)
        if dist < 1e-3:
            return
        facing = (np.cos(state[2]) * to_goal_x + np.sin(state[2]) * to_goal_y) / dist  # heading . to-goal
        if facing < -0.5:
            self.U.assign(np.tile([self.wmin, self.wmax], (self.T, 1)).astype(np.float32))

    def replan(self, state, goal_xy, n_refine):
        """Run n_refine GPU refines from `state` toward world `goal_xy`; updates U in place."""
        self.goal.assign(np.asarray(goal_xy[:2], np.float32))
        self.sim.start_pose.assign(
            np.ascontiguousarray(np.tile(np.asarray(state, np.float32), (self.B, 1)), np.float32)
        )
        if self.cw.ctg == 0.0 and self.cw.lattice == 0.0:  # the Euclidean turn-seed misfires when a
            self._seed_turn_if_facing_away(state, goal_xy)  # cost-to-go field already supplies the dir
        if self.dev.is_cuda:
            if self._graph is None:
                with wp.ScopedCapture(device=self.dev) as cap:
                    self._refine()
                self._graph = cap.graph
            for _ in range(n_refine):
                wp.capture_launch(self._graph)
        else:
            for _ in range(n_refine):
                self._refine()
        return self.U
