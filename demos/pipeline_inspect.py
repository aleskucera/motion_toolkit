"""Pipeline inspector: a per-frame DASHBOARD of the full closed loop, stitched to a
scrubable GIF — the rig for hunting good parameters and validating each stage.

Panels (step 1 — all real data, no new plumbing):
  * GLOBAL MAP    — the accumulated device voxel map (top-down, height-coloured) with the
                    true (green) vs ICP-estimated (blue) trails and the planner window.
  * PLANNER WINDOW— the single/local heightmap the MPPI drives on (unknown = flat/optimistic),
                    with the robot, the goal carrot, and the MPPI NOMINAL trajectory.
  * COST-TO-GO    — the routing field: colour = min-over-heading V, arrows = best heading
                    (the flow streaming toward the goal around obstacles).

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


class Dashboard:
    def __init__(self, R, halfb, dt, stride):
        self.R, self.halfb, self.dt, self.stride = R, halfb, dt, stride
        self.frames: list[Image.Image] = []
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5.2))

    def __call__(self, s):
        if s["f"] % self.stride:
            return
        cell, ww, wh = s["cell"], s["ww"], s["wh"]
        xmin, ymin = s["xmin"], s["ymin"]
        ex, ey, eyaw = s["est"]
        gx, gy = s["goal"]
        ax0, ax1, ax2 = self.axes
        for a in self.axes:
            a.clear()

        # --- Panel 0: global accumulated map ---
        if s["map_wp"] is not None and len(s["map_wp"]):
            p = s["map_wp"].numpy()
            ax0.scatter(p[:, 0], p[:, 1], c=p[:, 2], s=1.5, cmap="viridis", vmin=-0.2, vmax=2.0)
        tr, et = np.asarray(s["true_tr"]), np.asarray(s["est_tr"])
        ax0.plot(tr[:, 0], tr[:, 1], "-", color="#2ca02c", lw=2, label="true")
        ax0.plot(et[:, 0], et[:, 1], "-", color="#1f77b4", lw=1.4, label="ICP est")
        ax0.plot(ex, ey, "o", color="k", ms=6)
        ax0.plot(gx, gy, "*", color="red", ms=16, mec="k")
        ax0.add_patch(Rectangle((xmin, ymin), ww * cell, wh * cell, fill=False, ec="cyan", lw=1.3))
        ax0.set_aspect("equal")
        ax0.legend(loc="upper left", fontsize=8)
        ax0.set_title("Global accumulated map (device voxel map)")

        # --- Panel 1: local planner window + MPPI nominal ---
        ext = [0, ww * cell, 0, wh * cell]
        ax1.imshow(s["elev"], origin="lower", extent=ext, cmap="terrain", vmin=-0.2, vmax=2.0)
        unknown = np.where(s["known"], np.nan, 1.0)
        ax1.imshow(unknown, origin="lower", extent=ext, cmap="Greys", alpha=0.35, vmin=0, vmax=1)
        rx, ry = ex - xmin, ey - ymin
        U = s["planner"].nominal()
        path = _rollout_nominal(U, rx, ry, eyaw, self.R, self.halfb, self.dt)
        ax1.plot(path[:, 0], path[:, 1], "-", color="#ffd400", lw=2.5, label="MPPI nominal")
        ax1.plot(rx, ry, "o", color="k", ms=7)
        ax1.plot(np.clip(gx - xmin, 0, ww * cell), np.clip(gy - ymin, 0, wh * cell), "*", color="red", ms=16, mec="k")
        ax1.set_aspect("equal")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.set_title("Planner window (unknown=grey) + nominal")

        # --- Panel 2: cost-to-go field (min-heading V + best-heading flow) ---
        V = s["V"].numpy()  # [cy, cx, n_theta]
        nt = V.shape[2]
        Vmin, tbest = V.min(axis=2), V.argmin(axis=2)
        heading = (tbest + 0.5) * 2.0 * np.pi / nt
        reach = Vmin < s["ctg"]._vcap * 0.9
        rcell = s["rccell"]
        rext = [0, V.shape[1] * rcell, 0, V.shape[0] * rcell]
        im = ax2.imshow(np.where(reach, Vmin, np.nan), origin="lower", extent=rext, cmap="magma")
        cxs = (np.arange(V.shape[1]) + 0.5) * rcell
        cys = (np.arange(V.shape[0]) + 0.5) * rcell
        XX, YY = np.meshgrid(cxs, cys)
        st = max(1, V.shape[1] // 22)
        Uq = np.where(reach, np.cos(heading), np.nan)
        Vq = np.where(reach, np.sin(heading), np.nan)
        ax2.quiver(XX[::st, ::st], YY[::st, ::st], Uq[::st, ::st], Vq[::st, ::st],
                   color="cyan", scale=30, width=0.004)
        ax2.plot(np.clip(gx - xmin, 0, rext[1]), np.clip(gy - ymin, 0, rext[3]), "*", color="lime", ms=16, mec="k")
        ax2.set_aspect("equal")
        ax2.set_title("Cost-to-go V (colour) + best-heading flow")

        self.fig.suptitle(
            f"frame {s['f']}   loc-err {s['err']:.2f} m   contacts {s['contacts']}", fontsize=13
        )
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.frames.append(Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())).convert("RGB"))

    def save(self, out, fps):
        if not self.frames:
            print("no frames")
            return
        self.frames[0].save(
            out, save_all=True, append_images=self.frames[1:], duration=int(1000 / fps), loop=0
        )
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
