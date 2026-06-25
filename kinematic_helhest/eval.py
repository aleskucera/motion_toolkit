"""Closed-loop evaluation on the REAL robot (WarpDriver) -- the canonical eval harness.

The offline mppi.plan() loop rolls the planner's OWN sim forward to "execute", so it never exhibits
the plan->real gap that CVaR and the terminal dock are designed for -- which makes it the wrong test
for them (CVaR even hurts there). This harness drives the actual WarpDriver with MPPI + cost-to-go
routing + the terminal dock, exactly as navigate_live does but headless, and reports per world:
reach / frames / closest approach / wall-contact frames. This is the loop that matches reality.

  python -m kinematic_helhest.eval --world pocket
  python -m kinematic_helhest.eval --stress [--K 8] [--dock-radius 1.5]
"""

import argparse

import numpy as np
import warp as wp

from . import dynamics
from . import worlds as W
from .control.mppi import MppiGpu
from .control.mppi import RobustConfig
from .control.terminal import dock_control
from .driver import WarpDriver
from .engine import GridParams
from .engine import Simulator

# lattice routing weights: routing + feasibility only. The terminal dock handles reach+stop, so the
# Just the cost WEIGHTS. The robot's tip-over envelope + feasibility thresholds come from the Robot
# struct (sim.robot), read straight by the cost kernel -- not duplicated into this weight dict.
_LATTICE_W = dict(
    goal_terminal=3.0,
    goal_running=0.3,
    infeasible=1e5,
    effort=2e-3,
    smoothness=2e-3,
    out_of_bounds=50.0,
    explore_fallback=1.0,  # explore toward the goal where the routing field saturates
)


def evaluate(
    world,
    device="cuda",
    K=8,
    dock_radius=1.5,
    n_theta=24,
    lat_coarsen=1,
    max_frames=1500,
    B=4096,
    T=70,
):
    import time

    builder, start, goal = W.WORLDS[world]
    scene = builder()
    mu = W.matching_friction(scene)
    goal = np.asarray(goal, np.float64)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    plan_sim = Simulator(dynamics.robot_params(), dynamics.planning_solver(), grid, B, T, device)
    plan_sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    plan_sim.set_friction(mu)
    planner = MppiGpu(plan_sim, _LATTICE_W, robust=RobustConfig(n_scenarios=K), n_theta=n_theta)
    planner.reset_nominal(1.5)
    # routing field, optionally coarse (k>1): max-pool the terrain (keeps thin walls), solve low-res
    k = max(1, int(lat_coarsen))
    cny, cnx, ccell = scene.ny // k, scene.nx // k, scene.cell * k
    Hc = (
        scene.H[: cny * k, : cnx * k].reshape(cny, k, cnx, k).max(axis=(1, 3)) if k > 1 else scene.H
    )
    Hc = np.ascontiguousarray(Hc, np.float32)
    cgrid = GridParams(cnx, cny, ccell, scene.x0, scene.y0)
    from .planning.costtogo import CostToGo

    clat = CostToGo(
        cgrid, dynamics.robot_params(), dynamics.planning_solver(), n_theta=n_theta, device=device
    )
    Hc = wp.array(Hc, dtype=wp.float32, device=device)  # settle cost-to-go takes a device array
    t0 = time.perf_counter()
    V = clat.compute(Hc, goal)
    wp.synchronize()
    ctg_ms = (time.perf_counter() - t0) * 1000.0
    planner.set_lattice(V, cgrid.build())
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
            cmd = dock_control(state, goal)  # terminal stage
        else:
            planner.replan(state, goal, 3)  # MPPI + cost-to-go routing
            u = planner.nominal()
            cmd = np.array([u[0, 0], u[0, 1], 0.5 * (u[0, 0] + u[0, 1])], np.float32)
        drv.step(cmd)
        if drv.clear < 0.05:
            contacts += 1
    return dict(reached=reached, frames=f + 1, closest=closest, contacts=contacts, ctg_ms=ctg_ms)


def stress(device="cuda", **kw):
    print(f"{'world':9}{'reach':7}{'frames':8}{'closest':9}{'contacts':10}{'ctg_ms':8}")
    n_reached = 0
    for name in W.WORLDS:
        r = evaluate(name, device=device, **kw)
        n_reached += bool(r["reached"])
        print(
            f"{name:9}{str(r['reached']):7}{r['frames']:<8}{r['closest']:<9.2f}{r['contacts']:<10}{r['ctg_ms']:<8.0f}"
        )
    print(f"reached {n_reached}/{len(W.WORLDS)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default=None, choices=list(W.WORLDS))
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--K", type=int, default=8, help="CVaR robust scenarios (1 = off)")
    ap.add_argument(
        "--dock-radius", type=float, default=1.5, help="terminal-dock handoff radius (0 = off)"
    )
    ap.add_argument(
        "--lat-coarsen",
        type=int,
        default=1,
        help="solve the routing field at 1/k resolution (k>1 = faster, ~k^3)",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    wp.init()
    kw = dict(K=args.K, dock_radius=args.dock_radius, lat_coarsen=args.lat_coarsen)
    if args.world and not args.stress:
        print(evaluate(args.world, device=args.device, **kw))
    else:
        stress(device=args.device, **kw)


if __name__ == "__main__":
    main()
