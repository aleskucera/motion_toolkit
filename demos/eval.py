"""Closed-loop evaluation on the REAL robot (WarpDriver) -- the canonical eval harness.

The offline mppi.plan() loop rolls the planner's OWN sim forward to "execute", so it never exhibits
the plan->real gap that the terminal dock is designed for -- which makes it the wrong test for it.
This harness drives the actual WarpDriver with MPPI + cost-to-go routing + the terminal dock, exactly
as navigate_live does but headless, and reports per world: reach / frames / closest approach /
wall-contact frames. This is the loop that matches reality.

  python demos/eval.py --world pocket
  python demos/eval.py --stress [--dock-radius 1.5]
"""

import argparse

import numpy as np
import warp as wp
from helhest import dynamics
from helhest import worlds as W
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.control.terminal import dock_control
from helhest.driver import WarpDriver
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams


def evaluate(
    world,
    device="cuda",
    dock_radius=1.5,
    n_theta=24,
    lat_coarsen=1,
    max_frames=1500,
    B=4096,
    T=70,
    record=False,
    record_fan=False,
    fan_n=80,
    fan_pts=20,
    fan_every=2,
):
    import time

    record = record or record_fan  # the fan viz also needs the pose track

    builder, start, goal = W.WORLDS[world]
    scene = builder()
    mu = W.matching_friction(scene)
    goal = np.asarray(goal, np.float64)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    plan_sim = ForwardSimulator(
        dynamics.robot_params(), dynamics.planning_solver(), grid, B, T, device
    )
    plan_sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    plan_sim.set_friction(mu)
    planner = MppiGpu(
        plan_sim, CostParams(), n_theta=n_theta
    )
    planner.reset_nominal(1.5)
    # routing field, optionally coarse (k>1): max-pool the terrain (keeps thin walls), solve low-res
    k = max(1, int(lat_coarsen))
    cny, cnx, ccell = scene.ny // k, scene.nx // k, scene.cell * k
    Hc = (
        scene.H[: cny * k, : cnx * k].reshape(cny, k, cnx, k).max(axis=(1, 3)) if k > 1 else scene.H
    )
    Hc = np.ascontiguousarray(Hc, np.float32)
    cgrid = GridParams(cnx, cny, ccell, scene.x0, scene.y0)
    from helhest.planning.costtogo import CostToGo

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

    ctg = None
    if record_fan:  # the routing field is static here (full-scene plan, fixed goal): capture once
        ctg = dict(
            Vmin=V.numpy().min(axis=2).astype(np.float32),
            vcap=float(clat._vcap),
            cnx=cnx,
            cny=cny,
            ccell=ccell,
            cx0=scene.x0,
            cy0=scene.y0,
        )

    fans = []  # per-replan rollout snapshots for the Blender MPPI viz (record_fan=True)

    def _snap_fan():
        ctr = planner.sim.controlled.numpy()  # [T+1, B, 3] = (x, y, yaw), world coords
        zz = planner.sim.derived.numpy()[..., 0]  # [T+1, B] settled height
        J = planner.J.numpy()  # [n_cand] per-candidate cost
        ncand, t1 = planner.n_cand, ctr.shape[0]
        ci = np.linspace(0, ncand - 1, min(fan_n, ncand)).astype(int)  # subsample candidates
        r0 = ci  # each candidate is a rollout
        ti = np.linspace(0, t1 - 1, min(fan_pts, t1)).astype(int)  # decimate along the path
        paths = np.stack([ctr[ti][:, r0, 0], ctr[ti][:, r0, 1], zz[ti][:, r0]], -1).transpose(
            1, 0, 2
        )  # [n, P, 3]
        best = int(np.argmin(J))  # lowest-cost candidate -> highlighted plan
        nom = np.stack([ctr[ti][:, best, 0], ctr[ti][:, best, 1], zz[ti][:, best]], -1)  # [P, 3]
        fans.append(
            dict(
                frame=f,
                paths=paths.astype(np.float32),
                cost=J[ci].astype(np.float32),
                nominal=nom.astype(np.float32),
            )
        )

    contacts, closest, reached, f = 0, 99.0, False, 0
    poses, cmds = [], []  # trajectory recording (record=True): pose per frame, cmd per step
    for f in range(max_frames):
        st = drv.render_state()
        state = np.array([st.x, st.y, st.yaw], np.float32)
        if record:
            p = st.place
            poses.append((st.x, st.y, p["z"], st.yaw, p["pitch"], p["roll"], st.valid))
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
            if record_fan and f % fan_every == 0:
                _snap_fan()
        if record:
            cmds.append(tuple(float(c) for c in cmd))
        drv.step(cmd)
        if drv.clear < 0.05:
            contacts += 1
    result = dict(reached=reached, frames=f + 1, closest=closest, contacts=contacts, ctg_ms=ctg_ms)
    if record:
        if len(poses) == len(cmds):  # loop hit max_frames -> capture the final pose too
            st = drv.render_state()
            p = st.place
            poses.append((st.x, st.y, p["z"], st.yaw, p["pitch"], p["roll"], st.valid))
        result.update(poses=poses, cmds=cmds, scene=scene, dt=dynamics.DT)
    if record_fan:
        result.update(fans=fans, ctg=ctg)
    return result


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
    kw = dict(dock_radius=args.dock_radius, lat_coarsen=args.lat_coarsen)
    if args.world and not args.stress:
        print(evaluate(args.world, device=args.device, **kw))
    else:
        stress(device=args.device, **kw)


if __name__ == "__main__":
    main()
