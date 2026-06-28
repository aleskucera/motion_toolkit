"""Timing benchmark for the control/ stack: MPPI `replan` (one control tick).

`MppiGpu.replan` (control/mppi.py) is sample/rollout/cost/reweight over n_refine iterations,
CUDA-graph-captured. B is the total rollouts; K (slip samples) splits them into B//K candidates
for CVaR, so K trades candidates for robustness at ~fixed rollout work. It needs a cost-to-go
field to route toward the goal, so each planner is armed with one (built via _common.build_routing;
the solve itself is timed in benchmarks.planning).

Headline metric: ctrl_RTF = dt / replan -- how much faster than the real robot the controller plans
(dt is the control step, DT=0.1s); >1 means it keeps up. Also reports Hz. CUDA-only; skips without a GPU.

Run from the repo root:  python -m benchmarks.control [--world slalom]
"""

import argparse

import numpy as np
import warp as wp
from kinematic_helhest import dynamics
from kinematic_helhest import worlds as W
from kinematic_helhest.control.mppi import CostParams
from kinematic_helhest.control.mppi import MppiGpu
from kinematic_helhest.control.mppi import RobustConfig
from kinematic_helhest.engine import GridParams
from kinematic_helhest.engine import Simulator

from ._common import build_routing
from ._common import build_scene
from ._common import time_fn

DT = dynamics.DT  # control timestep [s]


def _planner(scene, mu, B, T, K, n_theta, V, grid, device):
    sim_grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    sim = Simulator(dynamics.robot_params(), dynamics.planning_solver(), sim_grid, B, T, device)
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_friction(mu)
    p = MppiGpu(sim, CostParams(), robust=RobustConfig(n_slip_samples=K), n_theta=n_theta)
    p.reset_nominal(1.5)
    p.set_lattice(V, grid.build())
    return p


def _header():
    print(
        f"    {'B':>6} {'K':>3} {'cand':>6} {'n_ref':>6} {'replan_ms':>10} {'Hz':>7} {'ctrl_RTF':>9}"
    )


def _row(B, K, n_ref, t):
    print(f"    {B:>6} {K:>3} {B//K:>6} {n_ref:>6} {t*1e3:>10.2f} {1.0/t:>7.0f} {DT/t:>8.1f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="slalom", choices=list(W.WORLDS))
    args = ap.parse_args()

    wp.init()
    if not wp.is_cuda_available():
        print("CUDA not available -- MPPI is GPU-only (CUDA graph capture). Skipping.")
        return
    device = "cuda"
    scene, mu, start, goal = build_scene(args.world)
    B0, T0, K0, ntheta0, nref0, reps = 4096, 70, 8, 24, 3, 10

    print(
        f"\n=== MPPI replan  device={device}  world={args.world}  grid={scene.ny}x{scene.nx}  "
        f"dt={DT:.2f}  T={T0}  n_theta={ntheta0} ==="
    )

    # one routing field (full-res) arms the planner for every sweep
    _, V, grid, _ = build_routing(scene, ntheta0, 1, goal, device)

    def bench(B, K, nref):
        p = _planner(scene, mu, B, T0, K, ntheta0, V, grid, device)
        return time_fn(lambda: p.replan(start, goal, nref), reps, device)

    print(f"  n_refine sweep (B={B0}, K={K0}):")
    _header()
    for nref in [1, 3, 5, 10]:
        _row(B0, K0, nref, bench(B0, K0, nref))

    print(f"  K (CVaR) sweep (B={B0}, n_refine={nref0}):")
    _header()
    for K in [1, 4, 8, 16]:
        _row(B0, K, nref0, bench(B0, K, nref0))

    print(f"  B (rollouts) sweep (K={K0}, n_refine={nref0}):")
    _header()
    for B in [1024, 4096, 8192]:
        _row(B, K0, nref0, bench(B, K0, nref0))


if __name__ == "__main__":
    main()
