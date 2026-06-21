"""Closed-loop evaluation on the REAL robot (WarpDriver) -- the canonical eval harness.

The offline mppi.plan() loop rolls the planner's OWN sim forward to "execute", so it never exhibits
the plan->real gap that CVaR and the terminal dock are designed for -- which makes it the wrong test
for them (CVaR even hurts there). This harness drives the actual WarpDriver with MPPI + cost-to-go
routing + the terminal dock, exactly as navigate_live does but headless, and reports per world:
reach / frames / closest approach / wall-contact frames. This is the loop that matches reality.

  python -m kinematic_helhest.eval --world pocket
  python -m kinematic_helhest.eval --stress [--K 8] [--dock-radius 1.5] [--feasibility settle]
"""
import argparse

import numpy as np
import warp as wp

from . import dynamics
from . import worlds as W
from .driver import WarpDriver
from .engine import GridParams
from .engine import Simulator
from .planning.mppi_gpu import MppiGpu
from .planning.terminal import dock_control

# the lattice routing weights (endgame terms here are the patches the terminal dock will replace)
_LATTICE_W = dict(term=3.0, run=0.3, head=0.0, invalid=1e5, eff=2e-3, smooth=2e-3, lattice=1.0,
                  oob=50.0, term_v=1.0, endgame=12.0, endgame_r2=2.25)


def evaluate(world, device="cuda", K=8, dock_radius=1.5, feasibility="traversability",
             n_theta=24, turn_radius=0.5, max_frames=1500, B=4096, T=70):
    builder, start, goal = W.WORLDS[world]
    scene = builder(); mu = W.matching_friction(scene); goal = np.asarray(goal, np.float64)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    plan_sim = Simulator(dynamics.robot_params(), dynamics.planning_solver(), grid, B, T, device)
    plan_sim.set_terrain(wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device))
    plan_sim.set_friction(mu)
    planner = MppiGpu(plan_sim, 0.5, 4.0, _LATTICE_W, 0.05, 1e-2, 0, sigma_knot=1.0, n_knots=4,
                      n_scenarios=K, n_theta=n_theta)
    planner.reset_nominal(1.5)
    if feasibility == "settle":
        from .planning.costtogo import CostToGoLatticeSettle
        clat = CostToGoLatticeSettle(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0, device,
                                     n_theta=n_theta, turn_radius=turn_radius)
        planner.set_lattice(clat.compute(np.ascontiguousarray(scene.H, np.float32), mu, goal))
    else:
        from .planning.costtogo import CostToGoLattice
        clat = CostToGoLattice(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0, device,
                               n_theta=n_theta, turn_radius=turn_radius)
        planner.set_lattice(clat.compute(np.ascontiguousarray(scene.H, np.float32), goal))
    drv = WarpDriver(scene, mu, init_pose=tuple(start), device=device)

    contacts, closest, reached, f = 0, 99.0, False, 0
    for f in range(max_frames):
        st = drv.render_state()
        state = np.array([st.x, st.y, st.yaw], np.float32)
        d = float(np.hypot(st.x - goal[0], st.y - goal[1]))
        closest = min(closest, d)
        if d < 0.3:
            reached = True
            break
        if dock_radius > 0.0 and d < dock_radius:
            cmd = dock_control(state, goal)          # terminal stage
        else:
            planner.replan(state, goal, 3)           # MPPI + cost-to-go routing
            u = planner.nominal()
            cmd = np.array([u[0, 0], u[0, 1], 0.5 * (u[0, 0] + u[0, 1])], np.float32)
        drv.step(cmd)
        if drv.clear < 0.05:
            contacts += 1
    return dict(reached=reached, frames=f + 1, closest=closest, contacts=contacts)


def stress(device="cuda", **kw):
    print(f"{'world':9}{'reach':7}{'frames':8}{'closest':9}{'contacts':9}")
    n_reached = 0
    for name in W.WORLDS:
        r = evaluate(name, device=device, **kw)
        n_reached += bool(r["reached"])
        print(f"{name:9}{str(r['reached']):7}{r['frames']:<8}{r['closest']:<9.2f}{r['contacts']:<9}")
    print(f"reached {n_reached}/{len(W.WORLDS)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default=None, choices=list(W.WORLDS))
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--K", type=int, default=8, help="CVaR robust scenarios (1 = off)")
    ap.add_argument("--dock-radius", type=float, default=1.5, help="terminal-dock handoff radius (0 = off)")
    ap.add_argument("--feasibility", default="traversability", choices=["traversability", "settle"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    wp.init()
    if args.world and not args.stress:
        print(evaluate(args.world, device=args.device, K=args.K, dock_radius=args.dock_radius,
                       feasibility=args.feasibility))
    else:
        stress(device=args.device, K=args.K, dock_radius=args.dock_radius, feasibility=args.feasibility)


if __name__ == "__main__":
    main()
