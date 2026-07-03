import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from terrain_toolkit import FilterConfig
from terrain_toolkit import TerrainPipeline
from terrain_toolkit import TraversabilityConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="ouster.npy")
    p.add_argument("--resolution", type=float, default=0.15)
    p.add_argument("--smooth-sigma", type=float, default=0.8)
    p.add_argument("--pad", type=float, default=0.5, help="extra margin around cloud bounds (m)")
    p.add_argument("--cloud-max-points", type=int, default=20_000)
    p.add_argument("--coarse-iters", type=int, default=200)
    p.add_argument("--iters-per-level", type=int, default=50)
    p.add_argument(
        "--primary",
        choices=("max", "mean", "min"),
        default="max",
        help="Which reduction feeds inpaint → smooth → cost.",
    )
    p.add_argument("--z-max", type=float, default=None, help="Discard points above this height (m)")
    p.add_argument(
        "--filter",
        action="store_true",
        help="Enable support-ratio filter + obstacle inflation on traversability.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pts = np.load(args.path).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected (N,3+) array, got {pts.shape}")
    pts = pts[:, :3]
    # Ouster mounted with top forward: sensor Z → world X, sensor X → world Y, sensor Y → world -Z
    pts = np.stack([pts[:, 2], pts[:, 0], -pts[:, 1]], axis=1)

    xmin, ymin, zmin = pts.min(axis=0) - args.pad
    xmax, ymax, zmax = pts.max(axis=0) + args.pad
    bounds = (float(xmin), float(xmax), float(ymin), float(ymax))
    print(
        f"points: {len(pts)}  bounds x=[{xmin:.2f},{xmax:.2f}] "
        f"y=[{ymin:.2f},{ymax:.2f}] z=[{zmin:.2f},{zmax:.2f}]"
    )

    pipe = TerrainPipeline(
        resolution=args.resolution,
        bounds=bounds,
        primary=args.primary,
        inpaint=True,
        smooth_sigma=args.smooth_sigma,
        inpaint_coarse_iters=args.coarse_iters,
        inpaint_iters_per_level=args.iters_per_level,
        z_max=args.z_max,
        traversability=TraversabilityConfig(),
        filter=FilterConfig() if args.filter else None,
    )
    tm = pipe.process(pts)
    h, w = tm.max.shape
    filled = np.isfinite(tm.max).sum()
    print(
        f"heightmap shape: {tm.max.shape}  filled: {filled}/{tm.max.size} "
        f"({100 * filled / tm.max.size:.1f}%)"
    )

    x = xmin + (np.arange(w) + 0.5) * args.resolution
    y = ymin + (np.arange(h) + 0.5) * args.resolution

    if len(pts) > args.cloud_max_points:
        idx = np.random.default_rng(0).choice(len(pts), args.cloud_max_points, replace=False)
        sub = pts[idx]
    else:
        sub = pts

    specs = [
        [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
        [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
    ]
    titles = (
        "Raw cloud",
        f"Primary ({args.primary})",
        "Inpainted",
        f"Elevation (inpaint + σ={args.smooth_sigma} smooth)",
        "Slope cost",
        "Step-height cost",
        "Roughness cost",
        "Traversability" + (" (filtered)" if args.filter else ""),
    )
    fig = make_subplots(rows=2, cols=4, specs=specs, subplot_titles=titles, vertical_spacing=0.08)

    # --- Row 1: 3D elevation views ---
    primary_layer = getattr(tm, args.primary)
    fig.add_trace(
        go.Scatter3d(
            x=sub[:, 0],
            y=sub[:, 1],
            z=sub[:, 2],
            mode="markers",
            marker=dict(size=1.2, color=sub[:, 2], colorscale="Viridis"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Surface(x=x, y=y, z=primary_layer, colorscale="Viridis", showscale=False), row=1, col=2
    )
    from terrain_toolkit import multigrid_inpaint

    inpainted = multigrid_inpaint(
        primary_layer,
        coarse_iters=args.coarse_iters,
        iters_per_level=args.iters_per_level,
    )
    fig.add_trace(
        go.Surface(x=x, y=y, z=inpainted, colorscale="Viridis", showscale=False), row=1, col=3
    )
    fig.add_trace(
        go.Surface(x=x, y=y, z=tm.elevation, colorscale="Viridis", showscale=False), row=1, col=4
    )

    # --- Row 2: 3D surfaces colored by cost (z = elevation, color = cost) ---
    # Custom colorscale: black for NaN (sentinel = -0.1), then RdYlGn_r for [0, 1].
    import plotly.colors as pc

    NAN_SENTINEL = -0.1
    boundary = (0.0 - NAN_SENTINEL) / (1.0 - NAN_SENTINEL)  # where cost=0 maps to
    rdylgn_r = pc.get_colorscale("RdYlGn_r")
    cost_scale = [[0.0, "black"], [boundary, "black"]]
    for pos, color in rdylgn_r:
        cost_scale.append([boundary + pos * (1.0 - boundary), color])

    cost_layers = [
        ("Slope", tm.slope_cost),
        ("Step", tm.step_cost),
        ("Roughness", tm.roughness_cost),
        ("Traversability", tm.traversability),
    ]
    for col, (name, cost) in enumerate(cost_layers, start=1):
        cost_viz = np.where(np.isnan(cost), NAN_SENTINEL, cost)
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=tm.elevation,
                surfacecolor=cost_viz,
                colorscale=cost_scale,
                cmin=NAN_SENTINEL,
                cmax=1.0,
                showscale=(col == 4),
                colorbar=dict(title="cost", len=0.45, y=0.22) if col == 4 else None,
            ),
            row=2,
            col=col,
        )

    z_lo = float(min(np.nanmin(primary_layer), pts[:, 2].min()))
    z_hi = float(max(np.nanmax(primary_layer), pts[:, 2].max()))
    scene = dict(
        xaxis=dict(range=[xmin, xmax]),
        yaxis=dict(range=[ymin, ymax]),
        zaxis=dict(range=[z_lo, z_hi]),
        aspectmode="data",
    )
    scene_keys = {f"scene{i}" if i > 1 else "scene": scene for i in range(1, 9)}
    fig.update_layout(
        title=f"Ouster terrain pipeline ({w}×{h} @ {args.resolution} m/cell, primary={args.primary})",
        height=900,
        **scene_keys,
    )

    out = "heightmap_ouster.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")
    print(f"slope       cost=[{np.nanmin(tm.slope_cost):.3f}, {np.nanmax(tm.slope_cost):.3f}]")
    print(f"step        cost=[{np.nanmin(tm.step_cost):.3f}, {np.nanmax(tm.step_cost):.3f}]")
    print(
        f"roughness   cost=[{np.nanmin(tm.roughness_cost):.3f}, {np.nanmax(tm.roughness_cost):.3f}]"
    )
    print(
        f"traversable cost=[{np.nanmin(tm.traversability):.3f}, {np.nanmax(tm.traversability):.3f}]"
    )

    # Save traversability as a plain image for debugging (not affected by Plotly rendering)
    import matplotlib.pyplot as plt

    fig_img, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="black")
    im = ax.imshow(
        tm.traversability,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    fig_img.colorbar(im, ax=ax, label="cost")
    ax.set_title("Traversability cost" + (" (filtered)" if args.filter else ""))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    img_out = "traversability_ouster.png"
    fig_img.savefig(img_out, dpi=150, bbox_inches="tight")
    plt.close(fig_img)
    print(f"Wrote {img_out}")


if __name__ == "__main__":
    main()
