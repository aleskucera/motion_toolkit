"""Visualize the settle-based orientation-aware cost-to-go as a heading FLOW FIELD over a world.

Solves V(x, y, theta) with CostToGo for a world's goal, then renders:
  * color    = min over heading of V(x, y, .)  -- the best-case cost-to-go (black = +inf unreachable)
  * arrows   = the argmin-heading direction at each cell -- "face this way for the cheapest forward-
               only path to the goal", i.e. the flow streaming toward the goal.

  python demos/costfield.py --world pocket
  python demos/costfield.py --world slalom --lat-coarsen 2 --stride 1
"""

import argparse

import numpy as np


def run(world="pocket", n_theta=24, stride=1, lat_coarsen=6, device="cuda", out=None):
    import warp as wp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kinematic_helhest import dynamics
    from kinematic_helhest import worlds as W
    from kinematic_helhest.engine import GridParams
    from kinematic_helhest.planning.costtogo import CostToGo
    from kinematic_helhest.planning.lattice_solver import trace_optimal

    wp.init()
    builder, start, goal = W.WORLDS[world]
    scene = builder()
    goal = np.asarray(goal, np.float64)

    # optionally coarsen the router like the planner does (max-pool keeps thin walls)
    k = max(1, int(lat_coarsen))
    cny, cnx, ccell = scene.ny // k, scene.nx // k, scene.cell * k
    Hc = (
        scene.H[: cny * k, : cnx * k].reshape(cny, k, cnx, k).max(axis=(1, 3)) if k > 1 else scene.H
    )
    Hc = np.ascontiguousarray(Hc, np.float32)
    cgrid = GridParams(cnx, cny, ccell, scene.x0, scene.y0)
    # scale the arc step with the coarse cell so the forward-arc primitives never shrink below a cell
    # (step < cell -> turns round to 0 cells -> degenerate "in-place" rotation, which a forward-only
    # robot can't do). ~1.6 cells/arc keeps the lattice meaningful at any coarsening.
    step = max(0.3, 1.6 * ccell)
    ctg = CostToGo(
        cgrid,
        dynamics.robot_params(),
        dynamics.planning_solver(),
        n_theta=n_theta,
        step=step,
        device=device,
    )
    V = ctg.compute(
        wp.array(Hc, dtype=wp.float32, device=device), goal
    ).numpy()  # [cny, cnx, n_theta]
    traj = trace_optimal(ctg, start, n_theta, cnx, cny, scene.x0, scene.y0, ccell)

    Vmin = V.min(axis=2)  # best-case cost-to-go per cell
    tbest = V.argmin(axis=2)  # best heading bin per cell
    heading = (tbest + 0.5) * 2.0 * np.pi / n_theta  # bin-center angle (matches the lattice)
    reachable = Vmin < ctg._vcap * 0.9  # below the unreachable clamp

    x0, y0 = scene.x0, scene.y0
    ext = [x0, x0 + cnx * ccell, y0, y0 + cny * ccell]
    Vshow = np.where(reachable, Vmin, np.nan)  # +inf -> distinct color
    vmax = float(np.percentile(Vmin[reachable], 98)) if reachable.any() else 1.0

    Xc, Yc = np.meshgrid(x0 + (np.arange(cnx) + 0.5) * ccell, y0 + (np.arange(cny) + 0.5) * ccell)
    U = np.where(reachable, np.cos(heading), np.nan)  # arrow dir = the best heading to face
    Vq = np.where(reachable, np.sin(heading), np.nan)
    s = (slice(None, None, stride), slice(None, None, stride))

    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    cmap = plt.cm.viridis.copy()
    # +inf (unreachable: actual wall OR an orientation/topology trap) -> black
    cmap.set_bad("black")
    im = ax.imshow(Vshow, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    # the ACTUAL walls at full resolution (a SUBSET of the black region) -- solid grey so you can tell
    # a wall-black from a trap-black (standable cells with no forward-only path to the goal).
    ext_fine = [x0, x0 + scene.nx * scene.cell, y0, y0 + scene.ny * scene.cell]
    wallmask = np.where(scene.H > 0.5, 1.0, np.nan)
    ax.imshow(
        wallmask,
        origin="lower",
        extent=ext_fine,
        cmap=ListedColormap(["#9a9a9a"]),
        vmin=0,
        vmax=1,
        zorder=4,
    )
    ax.quiver(
        Xc[s],
        Yc[s],
        U[s],
        Vq[s],
        color="white",
        pivot="mid",
        angles="xy",
        scale=30,
        width=0.004,
        headwidth=4,
        headlength=5,
        alpha=0.85,
        zorder=3,
    )
    if len(traj) > 1:  # the optimal forward-only trajectory the policy rolls out from the start
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            "-",
            color="#ff5a00",
            lw=3.0,
            zorder=6,
            solid_capstyle="round",
            label="optimal trajectory",
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.plot(start[0], start[1], "o", color="white", mec="k", ms=12, zorder=7)
    ax.plot(goal[0], goal[1], "*", color="red", ms=22, mec="k", zorder=7)
    fig.colorbar(im, ax=ax, shrink=0.85, label="min over heading  V(x, y)  [m]")
    ax.set_title(
        f"{world}: cost-to-go flow + optimal trajectory  (color = min_theta V, arrows = best heading)\n"
        f"n_theta={n_theta}, grid {cnx}x{cny}  --  black = unreachable at every heading, grey = actual wall"
    )
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (left, m)")
    fig.tight_layout()
    out = out or f"/tmp/costflow_{world}.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    import kinematic_helhest.worlds as W

    ap.add_argument("--world", default="pocket", choices=list(W.WORLDS))
    ap.add_argument("--n-theta", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1, help="draw an arrow every `stride` cells")
    ap.add_argument("--lat-coarsen", type=int, default=6, help="solve the router at 1/k resolution")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(
        world=args.world,
        n_theta=args.n_theta,
        stride=args.stride,
        lat_coarsen=args.lat_coarsen,
        device=args.device,
        out=args.out,
    )


if __name__ == "__main__":
    main()
