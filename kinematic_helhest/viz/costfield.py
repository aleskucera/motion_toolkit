"""Visualize the cost-to-go field: 2D (position-only) vs the orientation-aware lattice.

Renders three panels for a world: the 2D cost-to-go V(x,y) (heading-ignored), and two heading
slices of the lattice V(x,y,theta) -- the chosen heading and its opposite -- so you can see how
the SAME cell costs differently depending on which way the (forward-only) robot faces. Black is
+inf (no forward-only path from that pose), grey is wall.

  python -m kinematic_helhest.viz.costfield --world pocket
  python -m kinematic_helhest.viz.costfield --world pocket --turn-radius 1.2 --heading-deg 180
  python -m kinematic_helhest.viz.costfield --world bumpy --trav-weight 3   # flat-preferring routing
"""
import argparse

import numpy as np

from .. import worlds as W
from ..planning.costtogo import CostToGo
from ..planning.costtogo import CostToGoLattice


def run(world="pocket", turn_radius=1.2, heading_deg=180.0, trav_weight=0.0, n_theta=24,
        device="cuda", out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    builder, start, goal = W.WORLDS[world]
    hm = builder()
    goal = np.asarray(goal, np.float64)
    H = np.ascontiguousarray(hm.H, np.float32)
    ext = [hm.x0, hm.x0 + hm.nx * hm.cell, hm.y0, hm.y0 + hm.ny * hm.cell]
    vcap = 1.5 * (hm.nx + hm.ny) * hm.cell * (1.0 + trav_weight)

    V2 = CostToGo(hm.nx, hm.ny, hm.cell, hm.x0, hm.y0, device).compute(H, goal).numpy()
    V3 = CostToGoLattice(hm.nx, hm.ny, hm.cell, hm.x0, hm.y0, device, n_theta=n_theta,
                         turn_radius=turn_radius, trav_weight=trav_weight).compute(H, goal).numpy()
    dth = 360.0 / n_theta
    b0 = int(round((heading_deg % 360.0) / dth)) % n_theta
    b1 = int(round(((heading_deg + 180.0) % 360.0) / dth)) % n_theta

    vmax = float(np.percentile(V2[V2 < vcap * 0.9], 98))

    def prep(V):
        Vm = V.astype(float).copy()
        Vm[Vm >= vcap * 0.9] = np.nan  # +inf (unreachable) -> distinct color
        return Vm

    wall = np.ma.masked_where(H <= 0.5, H)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("black")
    panels = [(f"2D cost-to-go  V(x,y)\nPOSITION ONLY (heading ignored)", prep(V2)),
              (f"lattice  V(x,y, heading {heading_deg:.0f}°)", prep(V3[:, :, b0])),
              (f"lattice  V(x,y, heading {(heading_deg + 180) % 360:.0f}°)", prep(V3[:, :, b1]))]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    for a, (t, Vp) in zip(ax, panels):
        im = a.imshow(Vp, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
        a.imshow(wall, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1)
        a.plot(start[0], start[1], "o", color="white", mec="k", ms=10)
        a.plot(goal[0], goal[1], "*", color="red", ms=20)
        a.set_title(t, fontsize=11)
        fig.colorbar(im, ax=a, shrink=0.82, label="cost-to-go (m)")
    tw = f", trav_weight={trav_weight:g}" if trav_weight else ""
    fig.suptitle(f"{world}: position-only vs orientation-aware cost-to-go (min turn radius {turn_radius} m{tw})"
                 "  --  black = +inf (no forward-only path), grey = wall", fontsize=12)
    fig.tight_layout()
    out = out or f"/tmp/costfield_{world}.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="pocket", choices=list(W.WORLDS))
    ap.add_argument("--turn-radius", type=float, default=1.2,
                    help="min turn radius for the lattice (bigger -> orientation matters more)")
    ap.add_argument("--heading-deg", type=float, default=180.0, help="heading slice to show (and its opposite)")
    ap.add_argument("--trav-weight", type=float, default=0.0, help="graded flat-preferring arc cost (try 3 on bumpy)")
    ap.add_argument("--n-theta", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(world=args.world, turn_radius=args.turn_radius, heading_deg=args.heading_deg,
        trav_weight=args.trav_weight, n_theta=args.n_theta, device=args.device, out=args.out)


if __name__ == "__main__":
    main()
