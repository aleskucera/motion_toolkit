"""GPU MPPI inner loop (mppi_gpu) vs the numpy oracle (mppi._cost + numpy reweight).

The GPU RNG differs from numpy's, so trajectories can't be compared bit-for-bit.
Instead this checks the two host-replaceable pieces on *identical* inputs:
  * cost     : GPU `_cost` kernel J[B] vs numpy `_cost` on the same rollout + Ub
  * reweight : GPU jmin/softmax/weighted-U vs numpy softmax on the same J + Ub

Run:  python -m tests.planning.test_mppi_gpu
"""
import numpy as np
import warp as wp

from kinematic_helhest import friction
from kinematic_helhest import heightmap as hmmod
from kinematic_helhest.engine import GridParams
from kinematic_helhest.engine import RobotParams
from kinematic_helhest.engine import Simulator
from kinematic_helhest.engine import SolverParams
from kinematic_helhest.planning import mppi_gpu as mg
from kinematic_helhest.planning.mppi import _cost as cost_np
from kinematic_helhest.planning.mppi import _to_omega

_W = dict(term=3.0, run=0.3, invalid=1e5, eff=2e-3, smooth=2e-3,
          tilt=300.0, tilt_free=np.radians(12.0))
_CM, _RT, _WMAX = 0.05, 1e-2, 4.0


def _build_sim(device, B, T):
    scene = hmmod.demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    sim = Simulator(RobotParams(), SolverParams(dt=0.05, k_turn=2.0, newton_iters=12),
                    GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0), B, T, device)
    sim.set_terrain(wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device))
    sim.set_friction(mu)
    return sim


def selftest_cost_parity(device="cuda", B=2048, T=70):
    """Identical Ub -> GPU `_cost` J vs numpy `_cost` J. RNG-independent."""
    sim = _build_sim(device, B, T)
    rng = np.random.default_rng(0)
    start, goal = (0.0, 0.0, 0.0), np.array([3.0, 1.0])

    Ub = np.clip(rng.normal(1.5, _WMAX, (B, T, 2)), -_WMAX, _WMAX).astype(np.float32)  # arbitrary fan
    # same rollout feeds both: assign omega (= Ub) + start, launch, read back for the oracle.
    cc, dd, cl, rs = sim.rollout(_to_omega(Ub), start)
    J_np, _ = cost_np(cc, dd, cl, rs, Ub, goal, _CM, _RT, _W)

    goal_d = wp.array(goal.astype(np.float32), dtype=float, device=device)
    Jg = wp.zeros(B, dtype=float, device=device)
    cw = mg.CostWeights()
    cw.term, cw.run, cw.tilt = _W["term"], _W["run"], _W["tilt"]
    cw.eff, cw.smooth, cw.invalid = _W["eff"], _W["smooth"], _W["invalid"]
    cw.tilt_free, cw.clear_margin, cw.resid_tol = _W["tilt_free"], _CM, _RT
    wp.launch(mg._cost_kernel, B,
              inputs=[sim.controlled, sim.derived, sim.clearance, sim.residual, sim.omega, goal_d, cw, T],
              outputs=[Jg], device=device)
    J_gpu = Jg.numpy()

    rel = np.abs(J_gpu - J_np) / (np.abs(J_np) + 1e-6)
    print(f"  cost   B={B} T={T}: J~{J_np.mean():.0f}  max|rel|={rel.max():.2e}  "
          f"max|abs|={np.abs(J_gpu - J_np).max():.2e}")
    print(f"cost parity  {'OK' if rel.max() < 1e-2 else 'REVIEW'}")


def selftest_reweight_parity(device="cuda", B=2048, T=70, elite_frac=0.1):
    """Identical J + Ub -> GPU CEM (bisection top-k elite mean) U vs numpy exact-top-k mean."""
    rng = np.random.default_rng(1)
    Ub = np.clip(rng.normal(1.5, _WMAX, (B, T, 2)), -_WMAX, _WMAX).astype(np.float32)
    J = rng.uniform(0.0, 5.0e4, B).astype(np.float32)
    target_k = int(elite_frac * B)

    # numpy reference: exact top-k elite mean
    tau_np = np.partition(J, target_k)[target_k]
    U_np = np.clip(Ub[J <= tau_np].mean(0), -_WMAX, _WMAX).astype(np.float32)

    # GPU: bisection top-k threshold -> elite mean
    Jd = wp.array(J, dtype=float, device=device)
    omega = wp.array(_to_omega(Ub), dtype=wp.vec3, device=device)
    jmin = wp.zeros(1, dtype=float, device=device); jmax = wp.zeros(1, dtype=float, device=device)
    lo = wp.zeros(1, dtype=float, device=device); hi = wp.zeros(1, dtype=float, device=device)
    tau = wp.zeros(1, dtype=float, device=device); count = wp.zeros(1, dtype=float, device=device)
    Ud = wp.zeros((T, 2), dtype=float, device=device)
    wp.launch(mg._reset_minmax_kernel, 1, inputs=[jmin, jmax, count], device=device)
    wp.launch(mg._minmax_kernel, B, inputs=[Jd, jmin, jmax], device=device)
    wp.launch(mg._bisect_init_kernel, 1, inputs=[jmin, jmax, lo, hi, tau, count], device=device)
    for _ in range(mg._N_BISECT):
        wp.launch(mg._count_below_kernel, B, inputs=[Jd, tau, count], device=device)
        wp.launch(mg._bisect_step_kernel, 1, inputs=[count, float(target_k), lo, hi, tau], device=device)
    wp.launch(mg._count_below_kernel, B, inputs=[Jd, tau, count], device=device)
    wp.launch(mg._elite_u_kernel, (T, 2), inputs=[Jd, tau, count, omega, -_WMAX, _WMAX, B, Ud], device=device)
    U_gpu = Ud.numpy()

    n_gpu = int((J <= float(tau.numpy()[0])).sum())
    err = np.abs(U_gpu - U_np).max()
    print(f"  CEM reweight B={B} T={T}: target_k={target_k} gpu_elite={n_gpu} max|dU|={err:.2e}")
    print(f"reweight parity  {'OK' if err < 5e-2 else 'REVIEW'}")


if __name__ == "__main__":
    wp.init()
    dev = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
    print(f"device: {dev}")
    selftest_cost_parity(dev)
    selftest_reweight_parity(dev)
