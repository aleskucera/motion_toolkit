"""Closed-loop navigation on a map the robot BUILDS from lidar, planned over a FIXED robot window.

Like eval.py, but (a) the planner never sees the full world -- each cycle a synthetic lidar scan
(with occlusion) is accumulated into a world-frame multi-scan map -- and (b) the planner runs on a
FIXED-SIZE window cropped around the robot, not the whole map. So the planner cost is bounded no
matter how large the accumulated map gets. Unknown cells are inpainted to flat (optimistic). When the
goal lies outside the window the cost-to-go clamps it to the window edge -> a carrot the robot chases
as the window scrolls. The WarpDriver is reality (ground truth); its contacts reveal if optimism ever
drove the robot into an unseen wall.

  python -m kinematic_helhest.navigate_partial --world pocket --shot /tmp/partial_pocket.png
"""

import argparse
from collections import deque

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
from .eval import _LATTICE_W
from .perception.lidar import lidar_scan
from .perception.lidar import MultiScanMap
from .planning.costtogo import CostToGo


def _crop_window(mm, scene, rx, ry, ww, wh, cell):
    """Fixed (wh x ww) window centered on the robot, cropped from the world-frame map (out-of-world
    cells stay unknown). Returns (elev, known, wx0, wy0) -- wx0/wy0 is the window's world origin, so
    a world point maps to window-LOCAL coords as P - (wx0, wy0)."""
    c0 = int(round((rx - scene.x0) / cell)) - ww // 2
    r0 = int(round((ry - scene.y0) / cell)) - wh // 2
    elev = np.zeros((wh, ww), np.float32)
    known = np.zeros((wh, ww), bool)
    sr0, sr1 = max(0, r0), min(scene.ny, r0 + wh)  # overlap of window with the world array
    sc0, sc1 = max(0, c0), min(scene.nx, c0 + ww)
    if sr1 > sr0 and sc1 > sc0:
        dr, dc = sr0 - r0, sc0 - c0
        elev[dr : dr + (sr1 - sr0), dc : dc + (sc1 - sc0)] = mm.elev[sr0:sr1, sc0:sc1]
        known[dr : dr + (sr1 - sr0), dc : dc + (sc1 - sc0)] = mm.known[sr0:sr1, sc0:sc1]
    return elev, known, scene.x0 + c0 * cell, scene.y0 + r0 * cell


def _drift_scan(obs, known, x0, y0, cell, rx, ry, dx, dy, dyaw):
    """Apply an accumulated SE(2) drift -- rotation `dyaw` about the robot + translation (dx, dy) -- to
    a rasterized scan, then re-rasterize. The GLOBAL map then smears like a drifting pose estimate
    (rotation INCLUDED), while the live LOCAL scan is left untouched (the whole point of the split).
    """
    ri, ci = np.nonzero(known)
    if ri.size == 0:
        return obs, known
    px, py = x0 + ci * cell, y0 + ri * cell
    c, s = np.cos(dyaw), np.sin(dyaw)
    qx = rx + c * (px - rx) - s * (py - ry) + dx  # rotate about the robot, then translate
    qy = ry + s * (px - rx) + c * (py - ry) + dy
    qr = np.round((qy - y0) / cell).astype(int)
    qc = np.round((qx - x0) / cell).astype(int)
    ny, nx = known.shape
    ok = (qr >= 0) & (qr < ny) & (qc >= 0) & (qc < nx)
    o = np.zeros_like(obs)
    k = np.zeros_like(known)
    o[qr[ok], qc[ok]] = obs[ri[ok], ci[ok]]
    k[qr[ok], qc[ok]] = True
    return o, k


def navigate(
    world,
    device="cuda",
    K=8,
    dock_radius=1.5,
    n_theta=24,
    lat_coarsen=4,
    win_m=9.0,
    route_m=16.0,
    local_scans=0,
    drift=0.0,
    fov_deg=180.0,
    max_range=7.0,
    mount_height=0.4,
    max_frames=1500,
    B=4096,
    T=70,
):
    builder, start, goal = W.WORLDS[world]
    scene = builder()
    mu = W.matching_friction(scene)
    goal = np.asarray(goal, np.float64)
    cell = scene.cell
    ww = wh = int(round(win_m / cell))  # fixed square window, robot-centered

    drv = WarpDriver(scene, mu, init_pose=tuple(start), device=device)  # REALITY (ground truth)
    # the planner lives on the WINDOW grid (local origin 0); terrain is set per-cycle from the crop
    win_grid = GridParams(ww, wh, cell, 0.0, 0.0)
    plan_sim = Simulator(
        dynamics.robot_params(), dynamics.planning_solver(), win_grid, B, T, device
    )
    plan_sim.set_uniform_friction(0.8)
    planner = MppiGpu(plan_sim, _LATTICE_W, robust=RobustConfig(n_slip_samples=K), n_theta=n_theta)
    planner.reset_nominal(1.5)

    # DECOUPLED routing: the cost-to-go is solved on a LARGER (but still bounded, robot-centered)
    # window at COARSE resolution, so the router can see the way AROUND obstacles that don't fit in
    # the fine planning window -- without the per-frame cost of planning finely over that whole area.
    # Routing is topological (coarse suffices); fine obstacle avoidance stays in the MPPI rollouts.
    rww = rwh = int(round(max(route_m, win_m) / cell))
    kr = max(1, int(lat_coarsen))
    rcny, rcnx, rccell = rwh // kr, rww // kr, cell * kr
    route_grid = GridParams(rcnx, rcny, rccell, 0.0, 0.0)  # robot-centered, routing-LOCAL frame
    ctg = CostToGo(
        route_grid,
        dynamics.robot_params(),
        dynamics.planning_solver(),
        n_theta=n_theta,
        device=device,
    )
    # arm the saturation fallback (explore toward an out-of-window goal)
    planner.cw.lattice_cap = ctg._vcap
    # the routing field expressed in the PLANNING window's local frame: both windows snap to the same
    # fine grid and share the robot's center cell, so their origins differ by a CONSTANT cell offset
    # (frame-independent -> safe to bake into the captured replan graph; sampled poses are planning-local).
    sgrid = GridParams(
        rcnx, rcny, rccell, (ww // 2 - rww // 2) * cell, (wh // 2 - rwh // 2) * cell
    ).build()

    # the GLOBAL routing map the robot accumulates (drift-prone)
    mm = MultiScanMap(scene.ny, scene.nx)
    # SPLIT architecture (real-robot shape): global routing tolerates drift (coarse/topological);
    # local avoidance uses the FRESH sensor in the body frame (drift-free). local_scans=0 keeps the
    # old behaviour (fine window from the global map); >=1 builds the fine window from only the last
    # N scans (1 = a single live scan). `drift` injects synthetic SLAM drift into the GLOBAL map only.
    scan_buf = deque(maxlen=local_scans) if local_scans >= 1 else None
    rng = np.random.default_rng(0)
    # accumulated SE(2) global-map drift (m, m, rad), random walk
    drift_x = drift_y = drift_yaw = 0.0
    trail, snaps = [], []
    snap_every = max(1, max_frames // 60)
    win_world = ww * cell  # window side in meters (for the viz rectangle)
    contacts, closest, reached, f = 0, 99.0, False, 0

    for f in range(max_frames):
        st = drv.render_state()
        rx, ry, yaw = st.x, st.y, st.yaw
        trail.append((rx, ry))
        d = float(np.hypot(rx - goal[0], ry - goal[1]))
        closest = min(closest, d)
        if d < 0.3:
            reached = True
            break

        # PERCEPTION: one scan of reality (with occlusion)
        obs, known = lidar_scan(
            scene.H,
            scene.x0,
            scene.y0,
            cell,
            (rx, ry, yaw),
            fov_deg=fov_deg,
            max_range=max_range,
            mount_height=mount_height,
        )
        # GLOBAL routing map: accumulate, optionally smeared by a random-walk drift (features drift
        # relative to the robot/goal -- the coarse router must tolerate this).
        if drift > 0.0:
            drift_x += float(rng.normal(0.0, drift))
            drift_y += float(rng.normal(0.0, drift))
            drift_yaw += float(rng.normal(0.0, 0.1 * drift))  # coupled rotational drift (rad/step)
            gobs, gkn = _drift_scan(
                obs, known, scene.x0, scene.y0, cell, rx, ry, drift_x, drift_y, drift_yaw
            )
        else:
            gobs, gkn = obs, known
        mm.integrate(gobs, gkn)
        # LOCAL map for the fine MPPI window: the last `local_scans` live scans only (drift-free body
        # frame), or the global map when local_scans=0.
        if scan_buf is not None:
            scan_buf.append((obs, known))
            local = MultiScanMap(scene.ny, scene.nx)
            for o, kk in scan_buf:
                local.integrate(o, kk)
        else:
            local = mm

        # FINE planning window: the MPPI rollouts + feasibility run here (window-LOCAL coords)
        elev, kn, wx0, wy0 = _crop_window(local, scene, rx, ry, ww, wh, cell)
        elev = np.where(kn, elev, 0.0).astype(np.float32)  # unknown -> flat (optimistic)
        goal_l = (goal[0] - wx0, goal[1] - wy0)
        state_l = np.array([rx - wx0, ry - wy0, yaw], np.float32)  # robot ~at window center
        plan_sim.set_terrain(wp.array(np.ascontiguousarray(elev), dtype=wp.float32, device=device))

        # LARGER coarse routing window: cost-to-go solved robot-centered in its OWN local frame, then
        # sampled by the MPPI in the planning frame via sgrid's constant offset (graph-safe).
        relev, rkn, rwx0, rwy0 = _crop_window(mm, scene, rx, ry, rww, rwh, cell)
        relev = np.where(rkn, relev, 0.0).astype(np.float32)
        goal_r = (goal[0] - rwx0, goal[1] - rwy0)
        Hc = (
            relev[: rcny * kr, : rcnx * kr].reshape(rcny, kr, rcnx, kr).max(axis=(1, 3))
            if kr > 1
            else relev
        )
        V = ctg.compute(wp.array(np.ascontiguousarray(Hc), dtype=wp.float32, device=device), goal_r)
        planner.set_lattice(V, sgrid)

        if dock_radius > 0.0 and d < dock_radius:  # goal is in the window here -> dock to it
            cmd = dock_control(state_l, goal_l)
        else:
            planner.replan(state_l, goal_l, 3)
            u = planner.nominal()
            cmd = np.array([u[0, 0], u[0, 1], 0.5 * (u[0, 0] + u[0, 1])], np.float32)
        drv.step(cmd)  # execute on the REAL robot (wheel speeds are frame-independent)
        if drv.clear < 0.05:
            contacts += 1
        if f % snap_every == 0:
            snaps.append((f + 1, mm.known.copy(), list(trail), (rx, ry)))

    snaps.append((f + 1, mm.known.copy(), list(trail), (rx, ry)))
    return dict(
        reached=reached,
        frames=f + 1,
        closest=closest,
        contacts=contacts,
        known_frac=float(mm.known.mean()),
        scene=scene,
        goal=goal,
        snaps=snaps,
        win_world=win_world,
    )


def _viz(world, res, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    scene, goal, allsnaps, wwm = res["scene"], res["goal"], res["snaps"], res["win_world"]
    snaps = [allsnaps[i] for i in np.linspace(0, len(allsnaps) - 1, 4).astype(int)]
    ext = [scene.x0, scene.x0 + scene.nx * scene.cell, scene.y0, scene.y0 + scene.ny * scene.cell]
    wall = scene.H > 0.5
    cmap = ListedColormap(["#1a1a28", "#cfd3da", "#3a3a3a"])  # unknown / seen-free / seen-wall
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (fr, known, trail, rxy) in zip(axes.ravel(), snaps):
        cat = np.where(known, 1, 0)
        cat[known & wall] = 2
        ax.imshow(cat, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2, aspect="equal")
        tr = np.asarray(trail)
        ax.plot(tr[:, 0], tr[:, 1], "-", color="#ff7a1a", lw=2.0)
        ax.add_patch(
            Rectangle(
                (rxy[0] - wwm / 2, rxy[1] - wwm / 2),
                wwm,
                wwm,
                fill=False,
                ec="cyan",
                lw=1.5,
                zorder=6,
            )
        )  # the planner window
        ax.plot(rxy[0], rxy[1], "o", color="yellow", mec="k", ms=9, zorder=7)
        ax.plot(goal[0], goal[1], "*", color="red", ms=16, mec="k", zorder=7)
        ax.set_title(f"frame {fr}", fontsize=10)
    fig.suptitle(
        f"{world}: navigate on the lidar map, planned over a {wwm:.0f} m robot window  --  "
        f"reached={res['reached']}, frames={res['frames']}, contacts={res['contacts']}"
        f"   (cyan = the fixed planner window scrolling with the robot)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="pocket", choices=list(W.WORLDS))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--dock-radius", type=float, default=1.5)
    ap.add_argument("--lat-coarsen", type=int, default=4)
    ap.add_argument("--win-m", type=float, default=9.0, help="fine planning window side length (m)")
    ap.add_argument(
        "--route-m",
        type=float,
        default=16.0,
        help="coarse cost-to-go routing window side length (m); >= win-m",
    )
    ap.add_argument(
        "--local-scans",
        type=int,
        default=0,
        help="0 = fine window from the global map; >=1 = from the last N live scans (1 = single scan)",
    )
    ap.add_argument(
        "--drift",
        type=float,
        default=0.0,
        help="synthetic per-step SLAM drift std (m) injected into the GLOBAL routing map only",
    )
    ap.add_argument("--fov-deg", type=float, default=180.0)
    ap.add_argument("--max-range", type=float, default=7.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shot", default=None)
    args = ap.parse_args()
    wp.init()
    res = navigate(
        args.world,
        device=args.device,
        K=args.K,
        dock_radius=args.dock_radius,
        lat_coarsen=args.lat_coarsen,
        win_m=args.win_m,
        route_m=args.route_m,
        local_scans=args.local_scans,
        drift=args.drift,
        fov_deg=args.fov_deg,
        max_range=args.max_range,
    )
    print(
        f"{args.world}: reached={res['reached']} frames={res['frames']} "
        f"closest={res['closest']:.2f} contacts={res['contacts']} mapped={res['known_frac']*100:.0f}%"
    )
    if args.shot:
        _viz(args.world, res, args.shot)


if __name__ == "__main__":
    main()
