"""GPU-resident MPPI inner loop (sample -> rollout -> cost -> reweight) + CUDA graph.

The numpy MPPI loop is host-bound: per refine ~15 ms of numpy (cost, noise, omega)
vs ~2.7 ms of GPU work. These Warp kernels move the whole refine onto the device so
nothing but the executed pose ever comes back, and the refine is captured as a CUDA
graph and replayed. `mppi._cost` stays the numpy oracle for the parity test.

Kernels (all suffixed _kernel):
  _sample_omega  noise -> Ub -> writes the rollout's omega buffer (rear unused -> 0)
  _cost          per-rollout scalar cost J[B] (goal/tilt/invalid + eff/smooth)
  _jmin/_softmax/_weighted_u   softmax reweight of the nominal U, on device
  _bump_seed/_reset_red        device-side RNG counter + reduction resets (graph-safe)
"""
import numpy as np
import warp as wp


@wp.kernel
def _sample_omega_kernel(U: wp.array2d(dtype=float), sigma: float, sigma_knot: float, wmax: float,
                  n_knots: int, seed: wp.array(dtype=int), omega: wp.array2d(dtype=wp.vec3)):
    t, b = wp.tid()
    B = omega.shape[1]
    T = omega.shape[0]
    el = U[t, 0]
    er = U[t, 1]
    if b > 0:  # b == 0 keeps the nominal (eps = 0)
        # SPLINE bias (option A): sample n_knots correlated knots per rollout and linearly
        # interpolate them over the horizon. Each knot is keyed on (b, knot) so it is shared
        # by all t in its span -> the rollout commits to a smooth, low-frequency maneuver. Unlike
        # a single constant bias (which can only arc), multiple knots express turn-then-straight,
        # and the control is smooth by construction. Recompute the two bracketing knots on the
        # fly (rand_init is deterministic) -- no shared knot storage.
        interval = float(T - 1) / float(n_knots - 1)
        pos = float(t) / interval
        j0 = int(pos)
        j1 = wp.min(j0 + 1, n_knots - 1)
        frac = pos - float(j0)
        be0 = wp.randn(wp.rand_init(seed[0], (b * n_knots + j0) * 2))
        be1 = wp.randn(wp.rand_init(seed[0], (b * n_knots + j1) * 2))
        br0 = wp.randn(wp.rand_init(seed[0], (b * n_knots + j0) * 2 + 1))
        br1 = wp.randn(wp.rand_init(seed[0], (b * n_knots + j1) * 2 + 1))
        el += sigma_knot * ((1.0 - frac) * be0 + frac * be1)
        er += sigma_knot * ((1.0 - frac) * br0 + frac * br1)
        # Light per-step jitter on top (distinct stream) for local variation/refinement.
        jit = wp.rand_init(seed[0] + 9176, t * B + b)
        el += sigma * wp.randn(jit)
        er += sigma * wp.randn(jit)
    omega[t, b] = wp.vec3(wp.clamp(el, -wmax, wmax), wp.clamp(er, -wmax, wmax), 0.0)


@wp.kernel
def _cost_kernel(controlled: wp.array2d(dtype=wp.vec3), derived: wp.array2d(dtype=wp.vec3),
          clearance: wp.array2d(dtype=float), residual: wp.array2d(dtype=float),
          omega: wp.array2d(dtype=wp.vec3),  # Ub in components [0], [1]
          goal: wp.array(dtype=float),  # [2] world goal (device -> graph-safe, changes per replan)
          clear_margin: float, resid_tol: float, tilt_free: float,
          w_term: float, w_run: float, w_tilt: float, w_eff: float, w_smooth: float,
          w_invalid: float, T: int, Jout: wp.array(dtype=float)):
    b = wp.tid()
    run_sum = float(0.0)
    tilt_sum = float(0.0)
    term = float(0.0)
    for t in range(T + 1):
        c = controlled[t, b]
        dx = c[0] - goal[0]
        dy = c[1] - goal[1]
        d2 = dx * dx + dy * dy
        run_sum += d2
        term = d2  # last iter (t = T) sticks
        dv = derived[t, b]
        ang = wp.acos(wp.clamp(wp.cos(dv[1]) * wp.cos(dv[2]), -1.0, 1.0))
        over = wp.max(ang - tilt_free, 0.0)
        tilt_sum += over * over
    eff = float(0.0)
    smooth = float(0.0)
    inv = float(0.0)
    prev_l = float(0.0)
    prev_r = float(0.0)
    for t in range(T):
        om = omega[t, b]
        eff += om[0] * om[0] + om[1] * om[1]
        if t > 0:
            dl = om[0] - prev_l
            dr = om[1] - prev_r
            smooth += dl * dl + dr * dr
        prev_l = om[0]
        prev_r = om[1]
        if clearance[t, b] < clear_margin or residual[t, b] > resid_tol:
            inv = 1.0
    Jout[b] = (w_term * term + w_run * (run_sum / float(T + 1)) + w_tilt * (tilt_sum / float(T + 1))
               + w_eff * eff + w_smooth * smooth + inv * w_invalid)


@wp.kernel
def _reset_red_kernel(jmin: wp.array(dtype=float), betasum: wp.array(dtype=float)):
    jmin[0] = 1.0e30
    betasum[0] = 0.0


@wp.kernel
def _jmin_kernel(J: wp.array(dtype=float), jmin: wp.array(dtype=float)):
    wp.atomic_min(jmin, 0, J[wp.tid()])


@wp.kernel
def _softmax_kernel(J: wp.array(dtype=float), jmin: wp.array(dtype=float), lam: float,
             beta: wp.array(dtype=float), betasum: wp.array(dtype=float)):
    b = wp.tid()
    bb = wp.exp(-(J[b] - jmin[0]) / lam)
    beta[b] = bb
    wp.atomic_add(betasum, 0, bb)


@wp.kernel
def _weighted_u_kernel(beta: wp.array(dtype=float), betasum: wp.array(dtype=float),
                omega: wp.array2d(dtype=wp.vec3), wmax: float, B: int, U: wp.array2d(dtype=float)):
    t, c = wp.tid()
    acc = float(0.0)
    for b in range(B):
        acc += beta[b] * omega[t, b][c]
    U[t, c] = wp.clamp(acc / betasum[0], -wmax, wmax)


@wp.kernel
def _bump_seed_kernel(seed: wp.array(dtype=int)):
    seed[0] = seed[0] + 1


class MppiGpu:
    """GPU-resident MPPI: owns the nominal control `U` + scratch on device and runs the
    refine (sample -> rollout -> cost -> softmax reweight) entirely on the GPU. On CUDA
    the refine is captured once and replayed as a graph (the RNG counter is bumped
    in-graph, so each replay draws fresh noise); on CPU it runs eager. Wraps a Simulator.

    `goal` and `start_pose` are device arrays set per replan, so the captured graph picks
    up new values; the weights/sigma/lam/wmax are baked at capture (fixed per planner)."""

    def __init__(self, sim, sigma, lam, wmax, weights, clear_margin, resid_tol, seed=0,
                 sigma_knot=0.0, n_knots=4):
        self.sim = sim
        self.dev = sim.device
        self.B, self.T = sim.B, sim.T
        self.sigma, self.lam, self.wmax = float(sigma), float(lam), float(wmax)
        self.sigma_knot, self.n_knots = float(sigma_knot), int(n_knots)
        self.clear_margin, self.resid_tol = float(clear_margin), float(resid_tol)
        w = weights
        self.w_term = float(w.get("term", 0.0))
        self.w_run = float(w.get("run", 0.0))
        self.w_tilt = float(w.get("tilt", 0.0))
        self.w_eff = float(w.get("eff", 0.0))
        self.w_smooth = float(w.get("smooth", 0.0))
        self.w_invalid = float(w.get("invalid", 0.0))
        self.tilt_free = float(w.get("tilt_free", 0.0))
        d = self.dev
        self.U = wp.zeros((self.T, 2), dtype=float, device=d)
        self.J = wp.zeros(self.B, dtype=float, device=d)
        self.beta = wp.zeros(self.B, dtype=float, device=d)
        self.jmin = wp.zeros(1, dtype=float, device=d)
        self.betasum = wp.zeros(1, dtype=float, device=d)
        self.seed = wp.array([int(seed)], dtype=int, device=d)
        self.goal = wp.zeros(2, dtype=float, device=d)
        self._graph = None

    def reset_nominal(self, value=1.5):
        self.U.fill_(float(value))

    def nominal(self):
        """The current nominal control U [T, 2], on host."""
        return self.U.numpy()

    def set_nominal(self, U_host):
        self.U.assign(np.ascontiguousarray(U_host, np.float32))

    def _refine(self):
        s, d = self.sim, self.dev
        wp.launch(_bump_seed_kernel, 1, inputs=[self.seed], device=d)
        wp.launch(_sample_omega_kernel, (self.T, self.B),
                  inputs=[self.U, self.sigma, self.sigma_knot, self.wmax, self.n_knots, self.seed],
                  outputs=[s.omega], device=d)
        s.rollout_launch()
        wp.launch(_cost_kernel, self.B, inputs=[s.controlled, s.derived, s.clearance, s.residual, s.omega,
                  self.goal, self.clear_margin, self.resid_tol, self.tilt_free, self.w_term, self.w_run,
                  self.w_tilt, self.w_eff, self.w_smooth, self.w_invalid, self.T],
                  outputs=[self.J], device=d)
        wp.launch(_reset_red_kernel, 1, inputs=[self.jmin, self.betasum], device=d)
        wp.launch(_jmin_kernel, self.B, inputs=[self.J, self.jmin], device=d)
        wp.launch(_softmax_kernel, self.B, inputs=[self.J, self.jmin, self.lam, self.beta, self.betasum], device=d)
        wp.launch(_weighted_u_kernel, (self.T, 2), inputs=[self.beta, self.betasum, s.omega, self.wmax, self.B, self.U],
                  device=d)

    def replan(self, state, goal_xy, n_refine):
        """Run n_refine GPU refines from `state` toward world `goal_xy`; updates U in place."""
        self.goal.assign(np.asarray(goal_xy[:2], np.float32))
        self.sim.start_pose.assign(np.ascontiguousarray(
            np.tile(np.asarray(state, np.float32), (self.B, 1)), np.float32))
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
