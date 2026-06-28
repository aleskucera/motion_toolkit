"""Timing benchmark for the planning/ stack: the cost-to-go routing solve.

`CostToGo.compute` (planning/costtogo.py) builds the orientation-aware routing field V(x, y, theta)
via the internal LatticeValueSolver -- settle-based feasibility + value iteration. It runs per
window update (less often than the per-tick MPPI replan, which is benchmarked in control.py).

Cost scales with grid cells x n_theta, so this sweeps terrain coarsening (k -> ~k^3 fewer states)
and the heading bin count n_theta. CUDA-only (graph capture); skips cleanly without a GPU.

Run from the repo root:  python -m benchmarks.planning [--world slalom]
"""

import argparse

import warp as wp
from kinematic_helhest import worlds as W

from ._common import build_routing
from ._common import build_scene
from ._common import time_fn


def _header():
    print(f"    {'coarsen':>7} {'n_theta':>7} {'grid':>10} {'states':>9} {'solve_ms':>9}")


def _row(coarsen, n_theta, grid, t):
    states = grid.cells_x * grid.cells_y * n_theta
    print(
        f"    {coarsen:>7} {n_theta:>7} {f'{grid.cells_y}x{grid.cells_x}':>10} {states:>9} {t*1e3:>9.2f}"
    )


def _solve_ms(scene, n_theta, k, goal, device, reps):
    clat, _, grid, Hc = build_routing(scene, n_theta, k, goal, device)
    t = time_fn(lambda: clat.compute(Hc, (float(goal[0]), float(goal[1]))), reps, device)
    return grid, t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="slalom", choices=list(W.WORLDS))
    args = ap.parse_args()

    wp.init()
    if not wp.is_cuda_available():
        print("CUDA not available -- the cost-to-go solve is GPU-only (graph capture). Skipping.")
        return
    device = "cuda"
    scene, _, _, goal = build_scene(args.world)
    ntheta0, reps = 24, 15

    print(f"\n=== cost-to-go  device={device}  world={args.world}  grid={scene.ny}x{scene.nx} ===")

    print(f"  coarsen sweep (n_theta={ntheta0}):")
    _header()
    for k in [1, 2, 4]:
        grid, t = _solve_ms(scene, ntheta0, k, goal, device, reps)
        _row(k, ntheta0, grid, t)

    print("  n_theta sweep (coarsen=1):")
    _header()
    for nt in [8, 16, 24, 32]:
        grid, t = _solve_ms(scene, nt, 1, goal, device, reps)
        _row(1, nt, grid, t)


if __name__ == "__main__":
    main()
