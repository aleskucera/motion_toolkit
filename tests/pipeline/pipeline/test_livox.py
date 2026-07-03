import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helhest.perception import gaussian_smooth
from helhest.perception import HeightMapBuilder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="ouster.npy")
    p.add_argument("--resolution", type=float, default=0.2)
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--pad", type=float, default=0.5, help="extra margin around cloud bounds (m)")
    p.add_argument("--cloud-max-points", type=int, default=20_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pts = np.load(args.path).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected (N,3+) array, got {pts.shape}")
    pts = pts[:, :3]

    xmin, ymin, zmin = pts.min(axis=0) - args.pad
    xmax, ymax, zmax = pts.max(axis=0) + args.pad
    bounds = (float(xmin), float(xmax), float(ymin), float(ymax))
    print(
        f"points: {len(pts)}  bounds x=[{xmin:.2f},{xmax:.2f}] y=[{ymin:.2f},{ymax:.2f}] z=[{zmin:.2f},{zmax:.2f}]"
    )

    builder = HeightMapBuilder(args.resolution, bounds)
    layers = builder.build(pts).to_numpy()
    hm_max = layers["max"]
    hm_mean = layers["mean"]
    hm_smooth = gaussian_smooth(hm_max, sigma=args.smooth_sigma)
    print(f"heightmap shape: {hm_max.shape}  filled: {np.isfinite(hm_max).sum()}/{hm_max.size}")

    h, w = hm_max.shape
    x = xmin + (np.arange(w) + 0.5) * args.resolution
    y = ymin + (np.arange(h) + 0.5) * args.resolution

    if len(pts) > args.cloud_max_points:
        idx = np.random.default_rng(0).choice(len(pts), args.cloud_max_points, replace=False)
        sub = pts[idx]
    else:
        sub = pts

    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "scene"}] * 4],
        subplot_titles=("Raw cloud", "Max", "Mean", f"Max smoothed (σ={args.smooth_sigma})"),
    )
    fig.add_trace(
        go.Scatter3d(
            x=sub[:, 0],
            y=sub[:, 1],
            z=sub[:, 2],
            mode="markers",
            marker=dict(size=1.2, color=sub[:, 2], colorscale="Viridis"),
        ),
        1,
        1,
    )
    fig.add_trace(go.Surface(x=x, y=y, z=hm_max, colorscale="Viridis", showscale=False), 1, 2)
    fig.add_trace(go.Surface(x=x, y=y, z=hm_mean, colorscale="Viridis", showscale=False), 1, 3)
    fig.add_trace(go.Surface(x=x, y=y, z=hm_smooth, colorscale="Viridis", showscale=False), 1, 4)

    z_lo = float(min(np.nanmin(hm_max), pts[:, 2].min()))
    z_hi = float(max(np.nanmax(hm_max), pts[:, 2].max()))
    scene = dict(
        xaxis=dict(range=[xmin, xmax]),
        yaxis=dict(range=[ymin, ymax]),
        zaxis=dict(range=[z_lo, z_hi]),
        aspectmode="data",
    )
    fig.update_layout(
        title=f"Livox heightmap ({w}×{h} @ {args.resolution} m/cell)",
        height=650,
        scene=scene,
        scene2=scene,
        scene3=scene,
        scene4=scene,
    )
    out = "heightmap_livox.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
