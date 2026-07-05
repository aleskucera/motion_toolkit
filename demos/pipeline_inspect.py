"""Pipeline inspector — a per-frame dashboard of the full closed loop → scrubable GIF.
The rig for hunting parameters and validating each stage.

Layout (robot-centered, dot+arrow = robot; top 4 share the z colour-scale):
  TOP:    [ real world (ground truth) ] [ live lidar scan ] [ local single-scan map (MPPI) ] [ global rolling map (routing) ]
  BOTTOM: [ cost-to-go V + best-heading flow ]              [ MPPI rollout cloud (by cost) + nominal ]

Localization + the dynamic filter are read by COMPARING the real-world panel against the
global-map panel: offset/smear = drift; walker-smear-vs-clean = the ray-carve filter.

  python demos/pipeline_inspect.py --out /tmp/inspect.gif [--dynamic] [--world lane|narrow|slalom]
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pipeline_sim  # same demos/ dir when run as a script
import warp as wp
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from PIL import Image

_RED = ListedColormap(["#ff2020"])  # cells the dynamic ray-carve filter removed

ZMIN, ZMAX = -0.2, 2.0  # shared height colour-scale (ground → pillar top)
HCMAP = plt.cm.viridis
_NORM = plt.Normalize(ZMIN, ZMAX)
_GROUND = HCMAP(_NORM(0.0))  # colour of flat ground — used as the "continuous world" backdrop


def _settled_nominal(pl, xmin, ymin, rx, ry, yaw):
    """The committed nominal U rolled through the SETTLING simulator (same physics as reality) ->
    world-frame xy path. Unlike a kinematic/unicycle rollout, a settled robot stalls at walls, so this
    shows where the robot ACTUALLY goes -- it never climbs terrain the real robot couldn't."""
    U = pl.nominal()  # [T, 2] committed wheel speeds (wL, wR)
    tw = np.empty((U.shape[0], pl.sim.batch_size, 3), np.float32)  # rollout wants [T, B, 3]
    tw[:, :, 0] = U[:, 0:1]
    tw[:, :, 1] = U[:, 1:2]
    tw[:, :, 2] = 0.5 * (U[:, 0:1] + U[:, 1:2])  # rear wheel = mean (matches the driver command)
    controlled = pl.sim.rollout(tw, (rx, ry, yaw))[0]  # [T+1, B, 3] window-local; all B identical here
    return controlled[:, 0, :2] + np.array([xmin, ymin])


def _robot(ax, x, y, yaw, scale):
    """Robot = a dot + a heading arrow."""
    ax.plot(x, y, "o", color="magenta", ms=6, mec="k", zorder=6)
    ax.arrow(x, y, scale * math.cos(yaw), scale * math.sin(yaw), color="magenta",
             width=scale * 0.08, head_width=scale * 0.42, length_includes_head=True, zorder=6)


class Dashboard:
    def __init__(self, stride, view_m):
        self.stride, self.V = stride, view_m
        self.frames: list[Image.Image] = []
        self.fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        # top: perception (truth / raw scan / local map) ; bottom: planning (global map / cost-to-go / MPPI)
        (self.ax_world, self.ax_scan, self.ax_local) = axes[0]
        (self.ax_global, self.ax_ctg, self.ax_mppi) = axes[1]
        from helhest.perception import HeightMapBuilder

        self._HMB = HeightMapBuilder

    def __call__(self, s):
        if s["f"] % self.stride:
            return
        V = self.V
        ex, ey, eyaw = s["est"]
        cell, ww, wh = s["cell"], s["ww"], s["wh"]
        xmin, ymin = s["xmin"], s["ymin"]
        gx, gy = s["goal"]
        walker = s.get("walker")
        for a in (self.ax_world, self.ax_scan, self.ax_local, self.ax_global, self.ax_ctg, self.ax_mppi):
            a.clear()

        def big(ax):
            ax.set_xlim(ex - V, ex + V)
            ax.set_ylim(ey - V, ey + V)
            ax.set_aspect("equal")

        # --- world (ground truth): a continuous heightmap; ground fills everywhere so it never "ends"
        sc = s["scene"]
        aw = self.ax_world
        aw.set_facecolor(_GROUND)
        aw.imshow(sc.H, origin="lower", extent=[sc.x0, sc.x0 + sc.nx * sc.cell, sc.y0, sc.y0 + sc.ny * sc.cell],
                  cmap=HCMAP, vmin=ZMIN, vmax=ZMAX)
        if walker is not None:
            aw.add_patch(Rectangle((walker[0] - 0.35, walker[1] - 0.35), 0.7, 0.7, color=HCMAP(_NORM(1.8))))
        _robot(aw, ex, ey, eyaw, V * 0.12)
        big(aw)
        aw.set_title("Real world (ground truth)")

        # --- live lidar scan (this frame), coloured by height, same frame + scale
        asc = self.ax_scan
        asc.set_facecolor("#101014")
        sw = s["scan_world"].numpy()
        asc.scatter(sw[:, 0], sw[:, 1], c=sw[:, 2], s=2, cmap=HCMAP, vmin=ZMIN, vmax=ZMAX)
        _robot(asc, ex, ey, eyaw, V * 0.12)
        big(asc)
        asc.set_title("Live lidar scan (what it sees now)")

        # --- local single-scan map (inpaint + confidence) — the MPPI terrain (8 m window)
        al = self.ax_local
        al.set_facecolor(_GROUND)
        wext = [xmin, xmin + ww * cell, ymin, ymin + wh * cell]
        al.imshow(s["elev_local"], origin="lower", extent=wext, cmap=HCMAP, vmin=ZMIN, vmax=ZMAX)
        al.imshow(np.where(s["known_local"], np.nan, 1.0), origin="lower", extent=wext, cmap="Greys", alpha=0.5, vmin=0, vmax=1)
        _robot(al, ex, ey, eyaw, 0.8)
        al.set_xlim(wext[0], wext[1])
        al.set_ylim(wext[2], wext[3])
        al.set_aspect("equal")
        al.set_title("Local single-scan map → MPPI (grey = unknown)")

        # --- global rolling map (accumulator) → routing; rasterized big, robot-centered
        ag = self.ax_global
        ag.set_facecolor(_GROUND)
        gext = [ex - V, ex + V, ey - V, ey + V]
        if s["map_wp"] is not None and len(s["map_wp"]):
            dev = s["map_wp"].device
            gl = self._HMB(0.15, tuple(gext), device=dev).build(s["map_wp"])
            gcount = gl.count.numpy()
            ag.imshow(np.where(gcount > 0, gl.max.numpy(), np.nan), origin="lower", extent=gext, cmap=HCMAP, vmin=ZMIN, vmax=ZMAX)
            if s.get("map_raw") is not None and len(s["map_raw"]):  # what the ray-carve filter removed
                rc = self._HMB(0.15, tuple(gext), device=dev).build(s["map_raw"]).count.numpy()
                carved = (rc > 0) & (gcount == 0)
                ag.imshow(np.where(carved, 1.0, np.nan), origin="lower", extent=gext, cmap=_RED, alpha=0.95)
        if walker is not None:
            ag.add_patch(Rectangle((walker[0] - 0.35, walker[1] - 0.35), 0.7, 0.7, fill=False, ec="orange", lw=2))
        _robot(ag, ex, ey, eyaw, V * 0.12)
        big(ag)
        ag.set_title("Global map → routing" + ("  (red = filtered out)" if walker is not None else ""))

        # --- cost-to-go V + best-heading flow (routing window)
        Vf = s["V"].numpy()
        nt = Vf.shape[2]
        Vmin, tbest = Vf.min(axis=2), Vf.argmin(axis=2)
        heading = (tbest + 0.5) * 2.0 * np.pi / nt
        reach = Vmin < s["ctg"]._vcap * 0.9
        rc = s["rccell"]
        rxmin, rymin = s["rxmin"], s["rymin"]
        rext = [rxmin, rxmin + Vf.shape[1] * rc, rymin, rymin + Vf.shape[0] * rc]
        ac = self.ax_ctg
        ac.imshow(np.where(reach, Vmin, np.nan), origin="lower", extent=rext, cmap="magma")
        cxs = rxmin + (np.arange(Vf.shape[1]) + 0.5) * rc
        cys = rymin + (np.arange(Vf.shape[0]) + 0.5) * rc
        XX, YY = np.meshgrid(cxs, cys)
        st = max(1, Vf.shape[1] // 26)
        ac.quiver(XX[::st, ::st], YY[::st, ::st], np.where(reach, np.cos(heading), np.nan)[::st, ::st],
                  np.where(reach, np.sin(heading), np.nan)[::st, ::st], color="cyan", scale=32, width=0.003)
        ac.plot(np.clip(gx, rext[0], rext[1]), np.clip(gy, rext[2], rext[3]), "*", color="lime", ms=17, mec="k")
        _robot(ac, ex, ey, eyaw, 0.8)
        ac.set_aspect("equal")
        ac.set_title("Cost-to-go V (colour) + best-heading flow → goal")

        # --- MPPI rollout cloud (candidates by CVaR cost) + nominal, in the 8 m window
        pl = s["planner"]
        ctrl = pl.sim.controlled.numpy()  # [T+1, B, 3] window-local
        Jc = pl.J_cand.numpy()
        n_scen, n_cand = pl.n_slip, len(Jc)
        fin = Jc[np.isfinite(Jc)]
        lo, hi = (np.percentile(fin, [2, 92]) if len(fin) else (0.0, 1.0))
        norm = plt.Normalize(lo, max(hi, lo + 1e-6))
        am = self.ax_mppi
        am.set_facecolor(_GROUND)
        am.imshow(np.where(s["elev_local"] > 0.5, 1.0, np.nan), origin="lower", extent=wext, cmap="Greys", alpha=0.55, vmin=0, vmax=1)
        rx, ry = ex - xmin, ey - ymin
        for b in np.argsort(-np.nan_to_num(Jc, nan=lo))[:: max(1, n_cand // 240)]:
            p = ctrl[:, b * n_scen, :2] + np.array([xmin, ymin])  # window-local -> world
            col = plt.cm.viridis_r(norm(Jc[b])) if np.isfinite(Jc[b]) else (0.6, 0.6, 0.6, 0.15)
            am.plot(p[:, 0], p[:, 1], "-", color=col, lw=0.7, alpha=0.55)
        # nominal drawn through the SETTLING sim (not a unicycle) -> where the robot actually goes; the
        # host `ctrl` copy above is already taken, so this rollout may clobber pl.sim.controlled safely.
        path0 = _settled_nominal(pl, xmin, ymin, rx, ry, eyaw)
        am.plot(path0[:, 0], path0[:, 1], "-", color="#ff2d95", lw=2.6, label="nominal (settled)")
        _robot(am, ex, ey, eyaw, 0.8)
        am.set_xlim(wext[0], wext[1])
        am.set_ylim(wext[2], wext[3])
        am.set_aspect("equal")
        am.legend(loc="upper left", fontsize=8)
        am.set_title(f"MPPI cloud ({n_cand} candidates, bright=low cost) + nominal")

        self.fig.suptitle(f"frame {s['f']}   loc-err {s['err']:.2f} m   contacts {s['contacts']}", fontsize=15)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.frames.append(Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())).convert("RGB"))

    def save(self, out, fps):
        if not self.frames:
            print("no frames")
            return
        self.frames[0].save(out, save_all=True, append_images=self.frames[1:], duration=int(1000 / fps), loop=0)
        print(f"saved {out}  ({len(self.frames)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--world", choices=list(pipeline_sim._WORLDS), default="lane")
    ap.add_argument("--max-frames", type=int, default=340)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--view-m", type=float, default=12.0, help="half-extent of the big robot-centered panels (m)")
    ap.add_argument("--dynamic", action="store_true", help="add a moving obstacle (compare world vs global map)")
    ap.add_argument("--robust-margin-m", type=float, default=0.0, help="cost-to-go safety tube: lateral (m)")
    ap.add_argument("--robust-margin-deg", type=float, default=0.0, help="cost-to-go safety tube: heading (deg)")
    ap.add_argument("--exec-slip", type=float, default=0.0, help="reality wheel slip: driver keeps uniform[x,1] per wheel")
    ap.add_argument("--out", default="/tmp/pipeline_inspect.gif")
    args = ap.parse_args()
    wp.init()

    dash = Dashboard(args.stride, args.view_m)
    res = pipeline_sim.run_closed_loop(
        device=args.device, world=args.world, max_frames=args.max_frames, frame_hook=dash, dynamic=args.dynamic,
        robust_margin_m=args.robust_margin_m, robust_margin_deg=args.robust_margin_deg, exec_slip=args.exec_slip,
    )
    print(f"reached={res['reached']} frames={res['frames']} contacts={res['contacts']}")
    dash.save(args.out, args.fps)


if __name__ == "__main__":
    main()
