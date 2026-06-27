"""Simple synthetic lidar on a heightmap, for testing the multi-scan + unknown-cell pipeline.

A 2.5D HORIZON SWEEP: for each azimuth ray from the sensor, march outward tracking the running max
elevation ANGLE -- a cell is seen iff it pokes above the horizon set by nearer terrain, else it's in
the occlusion shadow (unknown). That occlusion is the whole point: it's what creates the unknown
regions (behind walls, beyond range) the planner must reason about. Ground truth is the world's
heightmap; output is (observed_elevation, known_mask) over the same grid. Accumulate scans across a
drive (MultiScanMap) to get the multi-scan map.

  python -m kinematic_helhest.perception.lidar --world pocket
"""

from __future__ import annotations

import numpy as np


def lidar_scan(
    H,
    x0,
    y0,
    cell,
    pose,
    *,
    mount_height=0.4,
    max_range=7.0,
    fov_deg=180.0,
    ang_res_deg=0.5,
    r_min=0.3,
    noise=0.0,
    rng=None,
):
    """One scan from world pose (x, y, yaw) over heightmap H[ny,nx] (grid origin x0,y0, size cell).

    Returns (obs_elev[ny,nx], known[ny,nx] bool): obs_elev = ground-truth height at the cells the
    sensor can see (others 0); known marks those cells. mount_height = sensor height above the local
    ground; cells beyond max_range / outside the fov / in occlusion shadow stay unknown."""
    ny, nx = H.shape
    px, py, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    sci = min(max(int(round((px - x0) / cell)), 0), nx - 1)
    sri = min(max(int(round((py - y0) / cell)), 0), ny - 1)
    sz = float(H[sri, sci]) + mount_height  # sensor height = local ground + mount

    half = np.radians(fov_deg) / 2.0
    ang = yaw + np.arange(-half, half, np.radians(ang_res_deg))
    ca, sa = np.cos(ang), np.sin(ang)
    max_ang = np.full(ang.size, -np.inf)  # running horizon (elevation angle) per ray
    known = np.zeros((ny, nx), bool)
    obs = np.zeros((ny, nx), np.float32)

    for r in np.arange(r_min, max_range, cell * 0.5):  # oversample radius so no cell is skipped
        ri = np.round((py + r * sa - y0) / cell).astype(int)
        ci = np.round((px + r * ca - x0) / cell).astype(int)
        inb = (ri >= 0) & (ri < ny) & (ci >= 0) & (ci < nx)
        if not inb.any():
            break  # every ray has left the (convex) window -> none re-enters
        rc, cc = np.clip(ri, 0, ny - 1), np.clip(ci, 0, nx - 1)
        a = np.arctan2(H[rc, cc] - sz, r)  # elevation angle to this cell's surface
        vis = inb & (a >= max_ang)  # pokes above the horizon set by nearer terrain -> seen
        known[rc[vis], cc[vis]] = True
        obs[rc[vis], cc[vis]] = H[rc[vis], cc[vis]]
        max_ang = np.where(inb, np.maximum(max_ang, a), max_ang)

    if noise > 0.0 and rng is not None:
        obs = obs + (rng.standard_normal(obs.shape).astype(np.float32) * noise) * known
    return obs, known


class MultiScanMap:
    """Accumulate scans into a persistent map (latest height wins, known grows monotonically)."""

    def __init__(self, ny, nx):
        self.known = np.zeros((ny, nx), bool)
        self.elev = np.zeros((ny, nx), np.float32)

    def integrate(self, obs, known):
        self.elev[known] = obs[known]
        self.known |= known


def _poses_along(traj, start, n):
    """Subsample a path [M,2] to n poses (x, y, yaw); yaw from the local tangent."""
    idx = np.linspace(0, len(traj) - 1, n).astype(int)
    out = []
    for j in idx:
        nj = min(j + 1, len(traj) - 1)
        dx, dy = traj[nj] - traj[j]
        yaw = np.arctan2(dy, dx) if (dx or dy) else float(start[2])
        out.append((float(traj[j, 0]), float(traj[j, 1]), float(yaw)))
    return out


def run(
    world="pocket", n_scans=16, fov_deg=180.0, max_range=7.0, lat_coarsen=6, device="cuda", out=None
):
    import warp as wp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from .. import dynamics
    from .. import worlds as W
    from ..engine import GridParams
    from ..planning.costtogo import CostToGo
    from ..viz.costfield import _trace_optimal

    wp.init()
    builder, start, goal = W.WORLDS[world]
    scene = builder()
    goalv = np.asarray(goal, np.float64)
    H = np.ascontiguousarray(scene.H, np.float32)  # ground truth the sensor reads

    # an optimal path to drive along (so the scan poses are a realistic route through the world)
    k = max(1, int(lat_coarsen))
    cny, cnx, ccell = scene.ny // k, scene.nx // k, scene.cell * k
    Hc = np.ascontiguousarray(
        scene.H[: cny * k, : cnx * k].reshape(cny, k, cnx, k).max(axis=(1, 3)), np.float32
    )
    cgrid = GridParams(cnx, cny, ccell, scene.x0, scene.y0)
    ctg = CostToGo(
        cgrid,
        dynamics.robot_params(),
        dynamics.planning_solver(),
        n_theta=24,
        step=max(0.3, 1.6 * ccell),
        device=device,
    )
    ctg.compute(wp.array(Hc, dtype=wp.float32, device=device), goalv)
    traj = _trace_optimal(ctg, start, 24, cnx, cny, scene.x0, scene.y0, ccell)
    poses = _poses_along(traj, start, n_scans)

    mm = MultiScanMap(scene.ny, scene.nx)
    snaps = {n_scans // 4, n_scans // 2, (3 * n_scans) // 4, n_scans}  # 4 progression frames
    frames = []
    for i, pose in enumerate(poses, 1):
        obs, known = lidar_scan(
            H, scene.x0, scene.y0, scene.cell, pose, fov_deg=fov_deg, max_range=max_range
        )
        mm.integrate(obs, known)
        if i in snaps:
            frames.append((i, mm.known.copy(), pose))

    ext = [scene.x0, scene.x0 + scene.nx * scene.cell, scene.y0, scene.y0 + scene.ny * scene.cell]
    wall = H > 0.5
    cmap = ListedColormap(
        ["#1a1a28", "#cfd3da", "#3a3a3a"]
    )  # 0 unknown / 1 known-free / 2 known-wall
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (i, kn, pose) in zip(axes.ravel(), frames):
        cat = np.where(kn, 1, 0)
        cat[kn & wall] = 2  # walls show only where actually observed (shadowed ones stay unknown)
        ax.imshow(cat, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2, aspect="equal")
        ax.plot(traj[:, 0], traj[:, 1], "-", color="#ff7a1a", lw=1.6, alpha=0.7)
        ax.plot(pose[0], pose[1], "o", color="yellow", mec="k", ms=9, zorder=5)
        ax.arrow(
            pose[0],
            pose[1],
            0.9 * np.cos(pose[2]),
            0.9 * np.sin(pose[2]),
            head_width=0.4,
            color="yellow",
            zorder=5,
            length_includes_head=True,
        )
        ax.plot(goal[0], goal[1], "*", color="red", ms=16, mec="k", zorder=5)
        ax.set_title(f"after scan {i}/{n_scans}", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle(
        f"{world}: synthetic-lidar multi-scan map filling in along the route  "
        f"(fov={fov_deg:g}°, range={max_range:g} m)  --  navy = unknown, light = seen-free, "
        f"grey = seen-wall",
        fontsize=12,
    )
    fig.tight_layout()
    out = out or f"/tmp/lidar_{world}.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    return out


def main():
    import argparse
    import kinematic_helhest.worlds as W

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="pocket", choices=list(W.WORLDS))
    ap.add_argument("--n-scans", type=int, default=16)
    ap.add_argument("--fov-deg", type=float, default=180.0)
    ap.add_argument("--max-range", type=float, default=7.0)
    ap.add_argument("--lat-coarsen", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(
        world=args.world,
        n_scans=args.n_scans,
        fov_deg=args.fov_deg,
        max_range=args.max_range,
        lat_coarsen=args.lat_coarsen,
        device=args.device,
        out=args.out,
    )


if __name__ == "__main__":
    main()
