"""Visualize what occlusion masking does on a saved point cloud.

Runs the pipeline twice (occlusion off vs on) and renders elevation, the two
traversability maps, and the cells occlusion removed.

    python viz_occlusion.py livox.npy
    python viz_occlusion.py wall.npy --sensor-z 1.5 --range 8
    python viz_occlusion.py field.npy --wall      # inject a demo wall to see the effect

Capture a fresh cloud near a wall first if your data is open terrain:
    python capture_cloud.py /livox/lidar_both_filtered wall.npy
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from terrain_toolkit import TerrainPipeline, TraversabilityConfig, OcclusionConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud", help="(N,3) .npy point cloud in the grid frame")
    ap.add_argument("--range", type=float, default=8.0, help="grid half-extent (m)")
    ap.add_argument("--res", type=float, default=0.15, help="cell size (m)")
    ap.add_argument("--sensor-x", type=float, default=0.0)
    ap.add_argument("--sensor-y", type=float, default=0.0)
    ap.add_argument("--sensor-z", type=float, default=None,
                    help="lidar height; default = ground-near-origin + 1.5 m")
    ap.add_argument("--eps-deg", type=float, default=0.6, help="view-angle margin (deg)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--wall", action="store_true", help="inject a demo wall to demonstrate")
    ap.add_argument("--out", default="viz_occlusion.png")
    args = ap.parse_args()

    pts = np.load(args.cloud).astype(np.float32)
    gnd = float(np.median(pts[(np.abs(pts[:, 0]) < 2) & (np.abs(pts[:, 1]) < 2), 2]))
    sz = args.sensor_z if args.sensor_z is not None else gnd + 1.5
    R, X = args.res, args.range
    B = (-X, X, -X, X)

    if args.wall:
        bearing = np.arctan2(pts[:, 1], pts[:, 0])
        behind = (np.abs(bearing) <= np.arctan2(2.0, 4.0)) & (pts[:, 0] > 4.3)
        rng = np.random.default_rng(0)
        wx = rng.uniform(4.0, 4.3, 4000); wy = rng.uniform(-2, 2, 4000)
        wz = rng.uniform(gnd, gnd + 2.5, 4000)
        pts = np.vstack([pts[~behind], np.column_stack([wx, wy, wz])]).astype(np.float32)

    def run(occ):
        p = TerrainPipeline(
            resolution=R, bounds=B, primary="max", inpaint=True, smooth_sigma=0.0, z_max=4.0,
            traversability=TraversabilityConfig(),
            occlusion=OcclusionConfig(
                sensor_xy=(args.sensor_x, args.sensor_y), sensor_z=sz,
                angle_eps_rad=float(np.deg2rad(args.eps_deg)),
            ) if occ else None,
            device=args.device,
        )
        return p.process(pts)

    off, on = run(False), run(True)
    removed = np.isfinite(off.traversability) & ~np.isfinite(on.traversability)
    print(f"ground~{gnd:.2f}  sensor_z={sz:.2f}  cells removed by occlusion = {int(removed.sum())}")

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(off.elevation, origin="lower", extent=B, cmap="terrain"); ax[0].set_title("elevation")
    ax[1].imshow(off.traversability, origin="lower", extent=B, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax[1].set_title("traversability: occlusion OFF")
    ax[2].imshow(on.traversability, origin="lower", extent=B, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax[2].set_title("traversability: occlusion ON")
    ax[3].imshow(removed, origin="lower", extent=B, cmap="Reds", vmin=0, vmax=1)
    ax[3].set_title(f"cells removed ({int(removed.sum())})")
    for a in ax:
        a.plot(args.sensor_x, args.sensor_y, "b*", ms=12)
    plt.tight_layout(); plt.savefig(args.out, dpi=85)
    print("saved", args.out)


if __name__ == "__main__":
    main()
