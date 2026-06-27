"""Adversarial test worlds for stress-testing the planner.

Each builder returns a Heightmap; WORLDS maps a name -> (builder, start (x,y,yaw), goal (x,y))
for the stress harness. They target different weaknesses:

  gap      a wall with one narrow gap -> threading clearance (and robustness under slip)
  slalom   alternating walls -> a forced S-weave, repeated tight clearance
  pillars  a field of pillars -> dense local avoidance
  pocket   a U-shaped cul-de-sac opening AWAY from the start -> GLOBAL routing (cost-to-go);
           a greedy Euclidean planner drives into the closed side and stalls
  ridge    a diagonal barrier with one notch -> direction-dependent crossing
  bumpy    rough terrain, some bumps tall enough to high-center -> tilt / settle feasibility

Render them:  python -m kinematic_helhest.worlds [--out /tmp/worlds.png]
"""

import argparse

import numpy as np

from .heightmap import _grid
from .heightmap import Heightmap

_WALL = 1.0  # impassable obstacle height (drive in -> infeasible settle)


def _box(H, XX, YY, cx, cy, hx, hy, h=_WALL):
    H[(np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)] = h


def gap_world(cell=0.06):
    xlim, ylim = (-2.0, 14.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    H[(np.abs(XX - 6.0) <= 0.2) & (np.abs(YY) >= 0.9)] = _WALL  # wall, 1.8 m gap at |y| < 0.9
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def slalom_world(cell=0.06):
    xlim, ylim = (-2.0, 19.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    H[(np.abs(XX - 4.0) <= 0.2) & (YY <= 1.2)] = _WALL  # open top (lane y > 1.2)
    H[(np.abs(XX - 9.0) <= 0.2) & (YY >= -1.2)] = _WALL  # open bottom
    # open CENTER -> exit aligned to goal
    H[(np.abs(XX - 14.0) <= 0.2) & (np.abs(YY) >= 1.2)] = _WALL
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def pillars_world(cell=0.06):
    xlim, ylim = (-2.0, 16.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    # 0.9 m pillars on a staggered ~3 m grid -> ~2 m clear corridors; last row flanks the center
    # so the robot exits aligned with the goal (a straight run-up, like gap)
    for cx, cy in [
        (4.0, -2.0),
        (4.0, 2.0),
        (7.0, 0.0),
        (7.0, -4.0),
        (7.0, 4.0),
        (10.0, -2.0),
        (10.0, 2.0),
        (13.0, -2.5),
        (13.0, 2.5),
    ]:
        _box(H, XX, YY, cx, cy, 0.45, 0.45)
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def pocket_world(cell=0.06):
    xlim, ylim = (-2.0, 16.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    H[(np.abs(XX - 7.0) <= 0.2) & (np.abs(YY) <= 2.5)] = _WALL  # closed side (faces start)
    H[(np.abs(YY - 2.5) <= 0.2) & (XX >= 7.0) & (XX <= 11.0)] = _WALL  # top
    H[(np.abs(YY + 2.5) <= 0.2) & (XX >= 7.0) & (XX <= 11.0)] = _WALL  # bottom; opening at x > 11
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def ridge_world(cell=0.06):
    xlim, ylim = (-2.0, 14.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    line = YY - 0.3 * (XX - 6.0)  # gentle diagonal ridge across the middle
    H[np.abs(line) <= 0.3] = _WALL
    H[(np.abs(line) <= 0.3) & (np.abs(XX - 6.0) <= 1.0)] = 0.0  # notch near x = 6
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def bumpy_world(cell=0.06, seed=0):
    xlim, ylim = (-2.0, 16.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    rng = np.random.default_rng(seed)
    # mounds of mixed height -- a few tall enough to be real obstacles to route AROUND, many gentle
    # -- with flat ground still left between them, so the flat path is harder to find but exists
    for _ in range(22):
        cx, cy = rng.uniform(1.5, 12.5), rng.uniform(-4.0, 4.0)
        amp, wid = rng.uniform(0.25, 0.8), rng.uniform(0.35, 0.7)
        H += amp * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * wid**2))
    return Heightmap(H, (xlim[0], ylim[0]), cell)


WORLDS = {
    "gap": (gap_world, (0.0, 0.0, 0.0), (11.0, 0.0)),
    "slalom": (slalom_world, (0.0, 0.0, 0.0), (17.0, 0.0)),
    "pillars": (pillars_world, (0.0, 0.0, 0.0), (15.0, 0.0)),
    "pocket": (pocket_world, (0.0, 0.0, 0.0), (9.0, 0.0)),
    "ridge": (ridge_world, (0.0, -4.0, 0.0), (9.0, 2.5)),
    "bumpy": (bumpy_world, (0.0, 0.0, 0.0), (14.0, 0.0)),
}


def matching_friction(hm, value=0.8):
    """Uniform-friction Heightmap matching a scene's grid exactly (avoids the dim mismatch)."""
    return Heightmap(np.full((hm.ny, hm.nx), value, np.float32), (hm.x0, hm.y0), hm.cell)


def _plot_all(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (name, (builder, start, goal)) in zip(axes.ravel(), WORLDS.items()):
        hm = builder()
        ext = [hm.x0, hm.x0 + hm.nx * hm.cell, hm.y0, hm.y0 + hm.ny * hm.cell]
        ax.imshow(hm.H, origin="lower", extent=ext, cmap="terrain", vmin=0.0, vmax=1.0)
        ax.plot(start[0], start[1], "o", color="white", mec="k", ms=9)
        ax.plot(goal[0], goal[1], "*", color="red", ms=16)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    fig.suptitle("Stress-test worlds (white = start, red star = goal, bright = obstacle)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/worlds.png")
    args = ap.parse_args()
    _plot_all(args.out)


if __name__ == "__main__":
    main()
