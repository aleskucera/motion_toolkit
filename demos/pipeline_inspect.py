"""Pipeline inspector: a per-frame DASHBOARD of the full closed loop, stitched to a
scrubable GIF — the rig for hunting good parameters and validating each stage.

Six panels (all real data):
  GLOBAL MAP     — accumulated device voxel map (top-down) + true/odom/ICP trails + window.
  PLANNER WINDOW — the local heightmap MPPI drives on (unknown=grey) + the MPPI nominal.
  COST-TO-GO     — routing field: colour = min-over-heading V, arrows = best heading (flow to goal).
  MPPI CLOUD     — a subsample of the B candidates coloured by CVaR cost (bright=low), nominal on top.
  ICP ERROR      — ICP vs odom localization error + inlier count over time (the localization tuning view).
  ICP SNAP       — this scan at the odom PREDICTION (red) vs after ICP (green) over the local map.

  python demos/pipeline_inspect.py --out /tmp/inspect.gif
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
from matplotlib.patches import Rectangle
from PIL import Image


def _rollout_nominal(U, x, y, yaw, R, halfb, dt):
    """Planar twist rollout of the MPPI nominal wheel-speed sequence U[T,2] -> xy path."""
    path = [(x, y)]
    for uL, uR in U:
        vx = R * (uL + uR) / 2.0
        wz = R * (uR - uL) / (2.0 * halfb)
        x += vx * math.cos(yaw) * dt
        y += vx * math.sin(yaw) * dt
        yaw += wz * dt
        path.append((x, y))
    return np.asarray(path)


def _xform(T, pts):
    """Apply a 4x4 pose to (N,3) points."""
    return (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]


class Dashboard:
    def __init__(self, R, halfb, dt, stride):
        self.R, self.halfb, self.dt, self.stride = R, halfb, dt, stride
        self.frames: list[Image.Image] = []
        self.hist = {"f": [], "icp": [], "odom": [], "inl": []}
        self.fig, self.axes = plt.subplots(2, 3, figsize=(19, 10))

    def __call__(self, s):
        # accumulate the localization history EVERY frame (render only on stride)
        tr = np.asarray(s["true_tr"])[-1]
        od = np.asarray(s["odom_tr"])[-1]
        self.hist["f"].append(s["f"])
        self.hist["icp"].append(s["err"])
        self.hist["odom"].append(float(np.hypot(od[0] - tr[0], od[1] - tr[1])))
        self.hist["inl"].append(s["outcome"].num_inliers if s["outcome"] else 0)
        if s["f"] % self.stride:
            return

        cell, ww, wh = s["cell"], s["ww"], s["wh"]
        xmin, ymin = s["xmin"], s["ymin"]
        ex, ey, eyaw = s["est"]
        gx, gy = s["goal"]
        ext = [0, ww * cell, 0, wh * cell]
        rx, ry = ex - xmin, ey - ymin
        ax = self.axes.ravel()
        for a in ax:
            a.clear()

        # --- 0: global accumulated map + true/odom/ICP trails ---
        if s["map_wp"] is not None and len(s["map_wp"]):
            p = s["map_wp"].numpy()
            ax[0].scatter(p[:, 0], p[:, 1], c=p[:, 2], s=1.3, cmap="viridis", vmin=-0.2, vmax=2.0)
        for key, col, lab in (("true_tr", "#2ca02c", "true"), ("odom_tr", "#d62728", "odom"), ("est_tr", "#1f77b4", "ICP")):
            a = np.asarray(s[key])
            ax[0].plot(a[:, 0], a[:, 1], "-", color=col, lw=1.6, label=lab)
        ax[0].plot(ex, ey, "o", color="k", ms=6)
        ax[0].plot(gx, gy, "*", color="red", ms=15, mec="k")
        ax[0].add_patch(Rectangle((xmin, ymin), ww * cell, wh * cell, fill=False, ec="cyan", lw=1.2))
        ax[0].set_aspect("equal")
        ax[0].legend(loc="upper left", fontsize=8)
        ax[0].set_title("Global map + true / odom / ICP trails")

        # --- 1: planner window + MPPI nominal ---
        ax[1].imshow(s["elev"], origin="lower", extent=ext, cmap="terrain", vmin=-0.2, vmax=2.0)
        ax[1].imshow(np.where(s["known"], np.nan, 1.0), origin="lower", extent=ext, cmap="Greys", alpha=0.35, vmin=0, vmax=1)
        path0 = _rollout_nominal(s["planner"].nominal(), rx, ry, eyaw, self.R, self.halfb, self.dt)
        ax[1].plot(path0[:, 0], path0[:, 1], "-", color="#ffd400", lw=2.5, label="MPPI nominal")
        ax[1].plot(rx, ry, "o", color="k", ms=7)
        ax[1].plot(np.clip(gx - xmin, 0, ww * cell), np.clip(gy - ymin, 0, wh * cell), "*", color="red", ms=15, mec="k")
        ax[1].set_aspect("equal")
        ax[1].legend(loc="upper left", fontsize=8)
        ax[1].set_title("Planner window (unknown=grey) + nominal")

        # --- 2: cost-to-go field + best-heading flow ---
        V = s["V"].numpy()
        nt = V.shape[2]
        Vmin, tbest = V.min(axis=2), V.argmin(axis=2)
        heading = (tbest + 0.5) * 2.0 * np.pi / nt
        reach = Vmin < s["ctg"]._vcap * 0.9
        rcell = s["rccell"]
        rext = [0, V.shape[1] * rcell, 0, V.shape[0] * rcell]
        ax[2].imshow(np.where(reach, Vmin, np.nan), origin="lower", extent=rext, cmap="magma")
        cxs = (np.arange(V.shape[1]) + 0.5) * rcell
        cys = (np.arange(V.shape[0]) + 0.5) * rcell
        XX, YY = np.meshgrid(cxs, cys)
        st = max(1, V.shape[1] // 22)
        ax[2].quiver(XX[::st, ::st], YY[::st, ::st], np.where(reach, np.cos(heading), np.nan)[::st, ::st],
                     np.where(reach, np.sin(heading), np.nan)[::st, ::st], color="cyan", scale=30, width=0.004)
        ax[2].plot(np.clip(gx - xmin, 0, rext[1]), np.clip(gy - ymin, 0, rext[3]), "*", color="lime", ms=15, mec="k")
        ax[2].set_aspect("equal")
        ax[2].set_title("Cost-to-go V (colour) + best-heading flow")

        # --- 3: MPPI rollout cloud (candidates coloured by CVaR cost) ---
        pl = s["planner"]
        ctrl = pl.sim.controlled.numpy()  # [T+1, B, 3] window-local
        Jc = pl.J_cand.numpy()
        n_scen, n_cand = pl.n_slip, len(Jc)
        fin = Jc[np.isfinite(Jc)]
        lo, hi = (np.percentile(fin, [2, 92]) if len(fin) else (0.0, 1.0))
        norm = plt.Normalize(lo, max(hi, lo + 1e-6))
        cmap = plt.cm.viridis_r
        ax[3].imshow(np.where(s["elev"] > 0.5, 1.0, np.nan), origin="lower", extent=ext, cmap="Greys", alpha=0.5, vmin=0, vmax=1)
        order = np.argsort(-np.nan_to_num(Jc, nan=lo))
        for b in order[:: max(1, n_cand // 220)]:
            pth = ctrl[:, b * n_scen, :2]
            col = cmap(norm(Jc[b])) if np.isfinite(Jc[b]) else (0.6, 0.6, 0.6, 0.15)
            ax[3].plot(pth[:, 0], pth[:, 1], "-", color=col, lw=0.7, alpha=0.55)
        ax[3].plot(path0[:, 0], path0[:, 1], "-", color="#ff2d95", lw=2.4, label="nominal")
        ax[3].plot(rx, ry, "o", color="k", ms=7)
        ax[3].set_xlim(0, ww * cell)
        ax[3].set_ylim(0, wh * cell)
        ax[3].set_aspect("equal")
        ax[3].legend(loc="upper left", fontsize=8)
        ax[3].set_title(f"MPPI cloud ({n_cand} candidates, bright=low cost)")

        # --- 4: ICP vs odom error + inliers over time ---
        h = self.hist
        ax[4].plot(h["f"], h["odom"], "-", color="#d62728", lw=1.4, label="odom-only err")
        ax[4].plot(h["f"], h["icp"], "-", color="#1f77b4", lw=1.8, label="ICP err")
        ax[4].set_xlabel("frame")
        ax[4].set_ylabel("localization error (m)")
        ax[4].legend(loc="upper left", fontsize=8)
        axr = ax[4].twinx()
        axr.plot(h["f"], h["inl"], "-", color="#999", lw=1.0, alpha=0.7)
        axr.set_ylabel("ICP inliers", color="#999")
        ax[4].set_title("ICP vs odom error  +  inliers (grey)")

        # --- 5: ICP snap — scan at odom prediction (red) vs after ICP (green) over local map ---
        ax[5].set_title("ICP: scan @ odom-pred (red) vs after ICP (green)")
        if s["pred"] is not None and s["map_wp"] is not None:
            base = s["scan_base"].numpy()
            base = base[:: max(1, len(base) // 4000)]
            mp = s["map_wp"].numpy()
            m = (np.abs(mp[:, 0] - ex) < 3.5) & (np.abs(mp[:, 1] - ey) < 3.5)
            ax[5].scatter(mp[m, 0], mp[m, 1], s=2, color="#bbb", label="map")
            pp = _xform(s["pred"], base)
            cp = _xform(s["T_wb"], base)
            ax[5].scatter(pp[:, 0], pp[:, 1], s=2, color="#d62728", alpha=0.5, label="scan @ pred")
            ax[5].scatter(cp[:, 0], cp[:, 1], s=2, color="#2ca02c", alpha=0.5, label="scan @ ICP")
            oc = s["outcome"]
            ax[5].set_title(f"ICP snap — inliers {oc.num_inliers}, Δ {oc.correction_trans_m*100:.1f} cm, {oc.status}")
            ax[5].legend(loc="upper left", fontsize=7, markerscale=3)
            ax[5].set_xlim(ex - 3.5, ex + 3.5)
            ax[5].set_ylim(ey - 3.5, ey + 3.5)
            ax[5].set_aspect("equal")

        self.fig.suptitle(f"frame {s['f']}   ICP-err {s['err']:.2f} m   contacts {s['contacts']}", fontsize=14)
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
    from helhest import dynamics

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--world", choices=list(pipeline_sim._WORLDS), default="lane")
    ap.add_argument("--max-frames", type=int, default=340)
    ap.add_argument("--stride", type=int, default=4, help="render every Nth frame")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="/tmp/pipeline_inspect.gif")
    args = ap.parse_args()
    wp.init()

    rp = dynamics.robot_params()
    dash = Dashboard(rp.wheel_radius, rp.half_track, dynamics.planning_solver().dt, args.stride)
    res = pipeline_sim.run_closed_loop(
        device=args.device, world=args.world, max_frames=args.max_frames, frame_hook=dash,
    )
    print(f"reached={res['reached']} frames={res['frames']} contacts={res['contacts']}")
    dash.save(args.out, args.fps)


if __name__ == "__main__":
    main()
