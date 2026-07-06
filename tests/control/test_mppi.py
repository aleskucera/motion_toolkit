"""GPU MPPI cost kernel + CEM reweight -- validated by CONTRACT, not by a numpy twin.

Differential-testing the cost against a hand-written numpy copy only proves the two AGREE, never
that either is CORRECT (a bug transcribed into both passes), and it taxes the hottest code with a
sync burden. So instead:
  * cost assembly : ANALYTIC -- fabricate a rollout with known poses/tilts/violations/controls and
                    check J against the cost computed BY HAND from the real Robot envelope. (The old
                    twin's constants had silently drifted from RobotParams and the test still passed
                    because demo_terrain never tilts into the gap -- exactly the false confidence
                    this replaces.)
  * sample_lattice: ANALYTIC -- a field whose value IS its column/heading index, so the trilinear
                    sample must return the fractional (column, heading) coordinate _locate defines.
  * fallback      : ANALYTIC -- a SATURATED field (V >= cap) must switch the goal term to the
                    straight-line pull cap^2 + explore_fallback*||pose-goal||^2.
  * cost terms    : ANALYTIC -- effort + smoothness + out-of-bounds, with the goal field ZEROED so
                    the tiny (2e-3) weights aren't drowned below float32 tolerance.
  * cvar          : the CVaR reduction (mean of a candidate's worst m_tail slip scenarios).
  * robust margin : BEHAVIORAL (deterministic) -- the CostToGo disturbance-tube keeps the routable
                    region, and thus the trajectory, >= the requested margin off obstacles.
  * reweight      : the GPU bisection top-k elite mean vs an EXACT numpy top-k (a different
                    algorithm for the same spec -- a real oracle, not a transcription).

Run:  python -m tests.control.test_mppi
"""

import numpy as np
import warp as wp
from helhest import friction
from helhest import heightmap as hmmod
from helhest.control import mppi as mg
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.engine.terrain import Grid
from tests._util import _to_target_wheel_omega

_W = dict(goal_terminal=3.0, goal_running=0.3, infeasible=1e5, effort=2e-3, smoothness=2e-3)
_WMAX = 4.0
_LAT_CONST = 5.0  # a non-saturated constant field -> V^2 is a known constant


def _build_sim(device, B, T):
    scene = hmmod.demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    sim = ForwardSimulator(
        RobotParams(),
        SolverParams(dt=0.05, k_turn=2.0, newton_iters=12),
        GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0),
        B,
        T,
        device,
    )
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_friction(mu)
    return sim


def _cw(explore_fallback=0.0, lattice_cap=1e9, out_of_bounds=0.0):
    cw = mg.CostWeights()
    cw.goal_terminal, cw.goal_running = _W["goal_terminal"], _W["goal_running"]
    cw.explore_fallback, cw.lattice_cap, cw.out_of_bounds = explore_fallback, lattice_cap, out_of_bounds
    cw.effort, cw.smoothness, cw.infeasible = _W["effort"], _W["smoothness"], _W["infeasible"]
    return cw


def _launch_cost(device, sim, poses, tilts, clear, resid, ctrl, field_val, goal, cw, T, B):
    """Fabricate a rollout (poses/tilts/violations/controls we CHOSE) and run the GPU cost kernel on
    it -> J[B]. Nothing is settled, so every input is known and J is hand-computable."""
    controlled = wp.array(np.ascontiguousarray(poses, np.float32), dtype=wp.vec3, device=device)
    derived = wp.array(np.ascontiguousarray(tilts, np.float32), dtype=wp.vec3, device=device)
    clearance = wp.array(np.ascontiguousarray(clear, np.float32), dtype=float, device=device)
    residual = wp.array(np.ascontiguousarray(resid, np.float32), dtype=float, device=device)
    twom = wp.array(np.ascontiguousarray(ctrl, np.float32), dtype=wp.vec3, device=device)
    cy, cx = sim.grid.cells_y, sim.grid.cells_x
    field = wp.full((cy, cx, 16), float(field_val), dtype=float, device=device)
    goal_d = wp.array(np.asarray(goal, np.float32), dtype=float, device=device)
    Jg = wp.zeros(B, dtype=float, device=device)
    wp.launch(
        mg._cost_kernel,
        B,
        inputs=[controlled, derived, clearance, residual, twom, goal_d, sim.grid, field, 16, cw, sim.robot, T],
        outputs=[Jg],
        device=device,
    )
    return Jg.numpy()


def selftest_cost_assembly(device="cuda"):
    """Every cost term at once, checked against the value computed BY HAND from the real Robot
    envelope: goal V^2 (terminal+running), effort, and the graded clearance/residual/roll/climb
    penalty with its (horizon-t)/horizon early weighting."""
    B, T = 1, 4
    sim = _build_sim(device, B, T)
    rp = RobotParams()  # host copy of the SAME envelope sim.robot was built from
    d_clear, d_resid, d_roll, d_climb = 0.02, 0.01, 0.02, 0.03  # chosen amounts PAST each limit

    poses = np.zeros((T + 1, B, 3), np.float32)  # anywhere (field is constant); yaw irrelevant
    poses[..., 0], poses[..., 1] = 2.0, 2.0
    tilts = np.zeros((T + 1, B, 3), np.float32)  # (z, pitch, roll)
    tilts[..., 1] = -(rp.max_pitch_up + d_climb)  # climbing = nose-UP = NEGATIVE pitch
    tilts[..., 2] = rp.max_roll + d_roll
    clear = np.full((T, B), rp.clear_margin - d_clear, np.float32)  # below margin -> clear_viol
    resid = np.full((T, B), rp.resid_tol + d_resid, np.float32)  # above tol -> resid_viol
    ctrl = np.full((T, B, 3), 1.0, np.float32)  # constant -> effort = T*2, smoothness = 0

    J = _launch_cost(device, sim, poses, tilts, clear, resid, ctrl, _LAT_CONST, [3.0, 1.0], _cw(), T, B)

    per_viol = d_clear + d_resid + d_roll + d_climb  # descend stays 0 (pitch is negative)
    sum_early = sum((T - t) / T for t in range(T))  # earlier violations weigh more
    exp = (
        (_W["goal_terminal"] + _W["goal_running"]) * _LAT_CONST**2  # goal: V^2, run mean == terminal
        + _W["effort"] * (T * 2.0)  # effort = sum wL^2+wR^2 = T*(1+1)
        + per_viol * sum_early * _W["infeasible"]
    )
    rel = abs(J[0] - exp) / abs(exp)
    print(f"  cost assembly: J={J[0]:.3f} expected={exp:.3f} rel={rel:.2e}")
    print(f"cost assembly  {'OK' if rel < 1e-4 else 'REVIEW'}")


def selftest_fallback(device="cuda"):
    """A SATURATED lattice (V >= cap) must drop V^2 and use the straight-line pull
    cap^2 + explore_fallback*||pose-goal||^2 -- the branch the constant-field test never reaches."""
    B, T = 1, 4
    sim = _build_sim(device, B, T)
    rp = RobotParams()
    cap, fb, V = 100.0, 1.0, 200.0  # V=200 >= 0.9*cap=90 -> saturated
    px, py, gx, gy = 2.0, 2.0, 5.0, 6.0

    poses = np.zeros((T + 1, B, 3), np.float32)
    poses[..., 0], poses[..., 1] = px, py
    tilts = np.zeros((T + 1, B, 3), np.float32)
    clear = np.full((T, B), rp.clear_margin + 1.0, np.float32)  # no violations anywhere
    resid = np.zeros((T, B), np.float32)
    ctrl = np.zeros((T, B, 3), np.float32)  # no effort/smoothness

    J = _launch_cost(
        device, sim, poses, tilts, clear, resid, ctrl, V, [gx, gy],
        _cw(explore_fallback=fb, lattice_cap=cap), T, B,
    )
    goal_cost = cap**2 + fb * ((px - gx) ** 2 + (py - gy) ** 2)
    exp = (_W["goal_terminal"] + _W["goal_running"]) * goal_cost
    rel = abs(J[0] - exp) / abs(exp)
    print(f"  fallback: J={J[0]:.3f} expected={exp:.3f} rel={rel:.2e}")
    print(f"fallback  {'OK' if rel < 1e-4 else 'REVIEW'}")


@wp.kernel
def _probe_sample(
    field: wp.array3d(dtype=float),
    grid: Grid,
    n_theta: int,
    xs: wp.array(dtype=float),
    ys: wp.array(dtype=float),
    yaws: wp.array(dtype=float),
    out: wp.array(dtype=float),
):
    i = wp.tid()
    out[i] = mg.sample_lattice(field, grid, n_theta, xs[i], ys[i], yaws[i])


def selftest_sample_lattice(device="cuda"):
    """Trilinear sample correctness. With a 1 m grid at origin 0, _locate maps world x to the
    fractional column (x - 0)/1 - 0.5. A field whose value IS its column index must sample back to
    that fraction (clamped in-bounds); a field whose value IS its heading index checks the theta
    interp + 2*pi wrap."""
    nx = ny = 10
    nt = 4
    grid = GridParams(nx, ny, 1.0, 0.0, 0.0).build()

    def _probe(field_np, xs, ys, yaws):
        field = wp.array(np.ascontiguousarray(field_np, np.float32), dtype=float, device=device)
        out = wp.zeros(len(xs), dtype=float, device=device)
        wp.launch(
            _probe_sample,
            len(xs),
            inputs=[
                field, grid, nt,
                wp.array(np.asarray(xs, np.float32), dtype=float, device=device),
                wp.array(np.asarray(ys, np.float32), dtype=float, device=device),
                wp.array(np.asarray(yaws, np.float32), dtype=float, device=device),
            ],
            outputs=[out],
            device=device,
        )
        return out.numpy()

    # field[r, c, t] = c -> sample returns the fractional column = clamp(x - 0.5, col in [0, nx-1])
    col = np.broadcast_to(np.arange(nx)[None, :, None], (ny, nx, nt))
    xs = [3.3, 0.5, -2.0, 100.0]  # in-cell, cell edge, off-grid low (clamp 0), off-grid high (clamp)
    exp_x = [2.8, 0.0, 0.0, 9.0]
    got_x = _probe(col, xs, [5.0] * 4, [0.0] * 4)

    # field[r, c, t] = t -> sample returns the interpolated heading index (wrapping at 2*pi)
    hd = np.broadcast_to(np.arange(nt)[None, None, :], (ny, nx, nt))
    two_pi = 2.0 * np.pi
    yaws = [0.0, two_pi / 8.0, two_pi * 7.0 / 8.0]  # bin 0; half of 0->1; half of 3->0 (wrap)
    exp_t = [0.0, 0.5, 1.5]
    got_t = _probe(hd, [5.0] * 3, [5.0] * 3, yaws)

    ex = float(np.abs(got_x - exp_x).max())
    et = float(np.abs(got_t - exp_t).max())
    print(f"  sample_lattice x: got={np.round(got_x, 4).tolist()} exp={exp_x} max|err|={ex:.2e}")
    print(f"  sample_lattice theta(+wrap): got={np.round(got_t, 4).tolist()} exp={exp_t} max|err|={et:.2e}")
    print(f"sample_lattice  {'OK' if max(ex, et) < 1e-4 else 'REVIEW'}")


def selftest_cost_terms(device="cuda"):
    """The three small terms -- effort (sum wL^2+wR^2), smoothness (sum of step-change^2), and the
    out-of-bounds soft wall -- checked TOGETHER with the goal field ZEROED. Isolating them matters:
    their weights are tiny (2e-3), so against the ~80+ goal term (let alone the ~1e5 penalty) they're
    below the float32 tolerance and effectively untested. With goal=0 and no violations, J is exactly
    effort + smoothness + out_of_bounds, so each is actually verified."""
    B, T = 1, 4
    sim = _build_sim(device, B, T)
    rp = RobotParams()
    g = sim.grid
    edge, d, oob_w = 0.4, 0.5, 1.0  # edge is hard-coded in the kernel; d = depth past the low-x wall
    x_lo = g.origin_x + edge
    y_mid = g.origin_y + 0.5 * g.cells_y * g.cell_size  # in-bounds in y -> only the x wall contributes

    poses = np.zeros((T + 1, B, 3), np.float32)
    poses[..., 0], poses[..., 1] = x_lo - d, y_mid
    tilts = np.zeros((T + 1, B, 3), np.float32)
    clear = np.full((T, B), rp.clear_margin + 1.0, np.float32)  # no violations -> no penalty
    resid = np.zeros((T, B), np.float32)
    wl = np.array([0.0, 1.0, 2.0, 3.0], np.float32)  # ramp -> constant step change
    wr = np.array([0.0, 0.5, 0.0, 0.5], np.float32)  # alternating -> varying step change
    ctrl = np.zeros((T, B, 3), np.float32)
    ctrl[:, 0, 0], ctrl[:, 0, 1] = wl, wr

    # field_val = 0 -> V = 0 everywhere -> the goal term drops out, leaving only the small terms
    J = _launch_cost(device, sim, poses, tilts, clear, resid, ctrl, 0.0, [3.0, 1.0],
                     _cw(out_of_bounds=oob_w), T, B)

    eff = float((wl**2 + wr**2).sum())
    smooth = float((np.diff(wl) ** 2 + np.diff(wr) ** 2).sum())
    oob = (T + 1) * d  # each of the T+1 poses is d past the wall
    exp = _W["effort"] * eff + _W["smoothness"] * smooth + oob_w * oob
    rel = abs(J[0] - exp) / abs(exp)
    print(f"  cost terms: J={J[0]:.4f} expected={exp:.4f} (eff={eff:.1f} smooth={smooth:.2f} oob={oob:.1f}) rel={rel:.2e}")
    print(f"cost terms  {'OK' if rel < 1e-4 else 'REVIEW'}")


def selftest_cvar(device="cuda"):
    """Robust eval (n_slip > 1): each candidate's cost is the CVaR = mean of its WORST m_tail slip
    scenarios (higher J = worse). Fabricated per-scenario J -> _cvar_kernel -> vs the numpy tail
    mean. Covers the whole robustness feature, which was previously untested (n_scen=1 skips it)."""
    n_cand, n_scen = 8, 5
    rng = np.random.default_rng(3)
    J = rng.uniform(0.0, 100.0, n_cand * n_scen).astype(np.float32)  # distinct -> no tie ambiguity
    blocks = J.reshape(n_cand, n_scen)
    ok = True
    for m_tail in (1, 2, n_scen):  # worst-only, worst-2, and the full mean (m_tail == n_scen)
        Jd = wp.array(J, dtype=float, device=device)
        Jc = wp.zeros(n_cand, dtype=float, device=device)
        wp.launch(mg._cvar_kernel, n_cand, inputs=[Jd, n_scen, m_tail], outputs=[Jc], device=device)
        exp = np.sort(blocks, axis=1)[:, -m_tail:].mean(1)  # mean of the m_tail largest (= worst)
        err = float(np.abs(Jc.numpy() - exp).max())
        ok = ok and err < 1e-4
        print(f"  CVaR n_scen={n_scen} m_tail={m_tail}: max|err|={err:.2e}")
    print(f"cvar  {'OK' if ok else 'REVIEW'}")


def selftest_robust_margin(device="cuda"):
    """BEHAVIORAL: the robust_margin tube must keep the PLAN a calibrated distance off obstacles.
    Erosion blocks every cell within the tube of an infeasible pose, so the routing V is reachable
    only OUTSIDE that band -- and the robot's trajectory, confined to reachable cells, therefore
    stays >= margin clear of the baseline obstacle boundary. Assert: the CLOSEST the plan lets the
    robot get grows by ~the margin (and monotonically). Deterministic -> a real pass/fail, unlike a
    stochastic MPPI-behavioral test."""
    from helhest.planning.costtogo import CostToGo

    nx, ny, cell, nth = 60, 60, 0.1, 16  # 6 x 6 m world
    xs = (np.arange(nx) + 0.5) * cell
    ys = (np.arange(ny) + 0.5) * cell
    XX, YY = np.meshgrid(xs, ys)
    a, b, c, d = 2.5, 3.5, 2.5, 3.5  # a 1 x 1 m block in the middle
    H = np.zeros((ny, nx), np.float32)
    H[(XX >= a) & (XX <= b) & (YY >= c) & (YY <= d)] = 2.0
    Hd = wp.array(np.ascontiguousarray(H), dtype=wp.float32, device=device)
    gp = GridParams(nx, ny, cell, 0.0, 0.0)
    goal = (5.5, 0.5)

    dx = np.maximum(np.maximum(a - XX, 0.0), XX - b)  # each cell's distance to the block AABB
    dy = np.maximum(np.maximum(c - YY, 0.0), YY - d)
    dcell = np.hypot(dx, dy)

    margins = [0.0, 0.2, 0.4]
    approach = []
    for m in margins:
        ctg = CostToGo(gp, RobotParams(), SolverParams(dt=0.1, k_turn=2.0, newton_iters=6, atol=1e-4),
                       n_theta=nth, robust_margin_m=m, device=device)
        V = ctg.compute(Hd, goal).numpy()
        reach = V.min(2) < ctg._vcap * 0.9  # cells the plan can route through
        approach.append(float(dcell[reach].min()))  # closest the plan lets the robot get to the block

    a0, a1, a2 = approach
    mono = a2 > a1 > a0
    keeps = (a1 >= a0 + 0.2 - cell) and (a2 >= a0 + 0.4 - cell)  # each margin adds >= itself (1 cell slack)
    print(f"  robust margin: closest-approach  m=0:{a0:.2f}  +0.2:{a1:.2f}  +0.4:{a2:.2f}  "
          f"(gained {a1 - a0:+.2f}, {a2 - a0:+.2f} m)")
    print(f"robust margin  {'OK' if (mono and keeps) else 'REVIEW'}")


def selftest_reweight_parity(device="cuda", B=2048, T=70, elite_frac=0.1):
    """GPU CEM (bisection top-k threshold -> elite mean) vs an EXACT numpy top-k mean. Different
    algorithm, same spec -- a genuine oracle, not a transcription."""
    rng = np.random.default_rng(1)
    Ub = np.clip(rng.normal(1.5, _WMAX, (B, T, 2)), -_WMAX, _WMAX).astype(np.float32)
    J = rng.uniform(0.0, 5.0e4, B).astype(np.float32)
    target_k = int(elite_frac * B)

    tau_np = np.partition(J, target_k)[target_k]
    U_np = np.clip(Ub[J <= tau_np].mean(0), -_WMAX, _WMAX).astype(np.float32)

    Jd = wp.array(J, dtype=float, device=device)
    target_wheel_omega = wp.array(_to_target_wheel_omega(Ub), dtype=wp.vec3, device=device)
    jmin = wp.zeros(1, dtype=float, device=device)
    jmax = wp.zeros(1, dtype=float, device=device)
    lo = wp.zeros(1, dtype=float, device=device)
    hi = wp.zeros(1, dtype=float, device=device)
    tau = wp.zeros(1, dtype=float, device=device)
    count = wp.zeros(1, dtype=float, device=device)
    Ud = wp.zeros((T, 2), dtype=float, device=device)
    wp.launch(mg._reset_minmax_kernel, 1, inputs=[jmin, jmax, count], device=device)
    wp.launch(mg._minmax_kernel, B, inputs=[Jd, jmin, jmax], device=device)
    wp.launch(mg._bisect_init_kernel, 1, inputs=[jmin, jmax, lo, hi, tau, count], device=device)
    for _ in range(mg._n_bisect(B)):  # n_scen=1 -> n_cand == B
        wp.launch(mg._count_below_kernel, B, inputs=[Jd, tau, count], device=device)
        wp.launch(
            mg._bisect_step_kernel, 1, inputs=[count, float(target_k), lo, hi, tau], device=device
        )
    wp.launch(mg._count_below_kernel, B, inputs=[Jd, tau, count], device=device)
    wp.launch(
        mg._elite_u_kernel,
        (T, 2),
        inputs=[Jd, tau, count, target_wheel_omega, 1, -_WMAX, _WMAX, B, Ud],
        device=device,
    )  # n_scen=1
    U_gpu = Ud.numpy()

    n_gpu = int((J <= float(tau.numpy()[0])).sum())
    err = np.abs(U_gpu - U_np).max()
    print(f"  CEM reweight B={B} T={T}: target_k={target_k} gpu_elite={n_gpu} max|dU|={err:.2e}")
    print(f"reweight parity  {'OK' if err < 5e-2 else 'REVIEW'}")


if __name__ == "__main__":
    wp.init()
    dev = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
    print(f"device: {dev}")
    selftest_cost_assembly(dev)
    selftest_fallback(dev)
    selftest_cost_terms(dev)
    selftest_sample_lattice(dev)
    selftest_cvar(dev)
    selftest_robust_margin(dev)
    selftest_reweight_parity(dev)
