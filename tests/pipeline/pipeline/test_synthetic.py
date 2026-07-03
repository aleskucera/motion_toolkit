import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helhest.perception import TerrainPipeline
from helhest.perception import TraversabilityConfig

BOUNDS = (-5.0, 5.0, -5.0, 5.0)
RESOLUTION = 0.1


def make_synthetic_cloud(
    n: int = 200_000,
    seed: int = 0,
    noise_std: float = 0.01,
    outlier_frac: float = 0.0,
    outlier_std: float = 0.5,
    dropout_frac: float = 0.0,
) -> np.ndarray:
    """Generate a tilted plane + Gaussian bump point cloud with configurable noise.

    - noise_std: stdev of per-point Gaussian z noise (meters)
    - outlier_frac: fraction of points that get large extra z noise (spikes)
    - outlier_std: stdev of the outlier z perturbation (meters)
    - dropout_frac: fraction of points to remove (simulates sparse cloud)
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(BOUNDS[0], BOUNDS[1], n)
    y = rng.uniform(BOUNDS[2], BOUNDS[3], n)
    z = 0.1 * x + 0.05 * y + 1.5 * np.exp(-((x - 1.0) ** 2 + (y + 1.0) ** 2) / 1.5)
    z += rng.normal(0.0, noise_std, n)

    if outlier_frac > 0.0:
        mask = rng.random(n) < outlier_frac
        z[mask] += rng.normal(0.0, outlier_std, mask.sum())

    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    if dropout_frac > 0.0:
        keep = rng.random(n) >= dropout_frac
        pts = pts[keep]
    return pts


def grid_axes(hm_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = hm_shape
    xmin, _, ymin, _ = BOUNDS
    x = xmin + (np.arange(w) + 0.5) * RESOLUTION
    y = ymin + (np.arange(h) + 0.5) * RESOLUTION
    return x, y


def parse_args() -> argparse.Namespace:
    presets = {
        "clean": dict(noise_std=0.01, outlier_frac=0.0, dropout_frac=0.0),
        "noisy": dict(noise_std=0.05, outlier_frac=0.0, dropout_frac=0.0),
        "very_noisy": dict(noise_std=0.15, outlier_frac=0.02, outlier_std=0.5, dropout_frac=0.0),
        "sparse": dict(noise_std=0.05, outlier_frac=0.01, outlier_std=0.3, dropout_frac=0.9),
    }
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(presets), default="clean")
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=None)
    p.add_argument("--outlier-frac", type=float, default=None)
    p.add_argument("--outlier-std", type=float, default=None)
    p.add_argument("--dropout-frac", type=float, default=None)
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument(
        "--cloud-max-points",
        type=int,
        default=20_000,
        help="Downsample cap for the raw cloud scatter.",
    )
    args = p.parse_args()
    cfg = dict(presets[args.preset])
    for k in ("noise_std", "outlier_frac", "outlier_std", "dropout_frac"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    args.cfg = cfg
    return args


def main() -> None:
    args = parse_args()
    pts = make_synthetic_cloud(n=args.n, seed=args.seed, **args.cfg)
    print(f"preset={args.preset} cfg={args.cfg} points={len(pts)}")

    pipe = TerrainPipeline(
        resolution=RESOLUTION,
        bounds=BOUNDS,
        primary="max",
        inpaint=True,
        smooth_sigma=args.smooth_sigma,
        traversability=TraversabilityConfig(),
    )
    tm = pipe.process(pts)
    x, y = grid_axes(tm.max.shape)
    xmin, xmax, ymin, ymax = BOUNDS

    specs = [
        [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
        [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
    ]
    titles = (
        "Raw cloud",
        "Max",
        "Mean",
        f"Elevation (inpaint + σ={args.smooth_sigma} smooth)",
        "Slope cost",
        "Step-height cost",
        "Roughness cost",
        "Traversability (combined)",
    )
    fig = make_subplots(rows=2, cols=4, specs=specs, subplot_titles=titles, vertical_spacing=0.08)

    # --- Row 1: 3D surfaces + raw cloud ---
    if len(pts) > args.cloud_max_points:
        idx = np.random.default_rng(0).choice(len(pts), args.cloud_max_points, replace=False)
        sub = pts[idx]
    else:
        sub = pts
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
        go.Surface(x=x, y=y, z=tm.max, colorscale="Viridis", showscale=False), row=1, col=2
    )
    fig.add_trace(
        go.Surface(x=x, y=y, z=tm.mean, colorscale="Viridis", showscale=False), row=1, col=3
    )
    fig.add_trace(
        go.Surface(x=x, y=y, z=tm.elevation, colorscale="Viridis", showscale=False), row=1, col=4
    )

    # --- Row 2: 3D surfaces colored by cost (z = elevation, color = cost) ---
    cost_scale = "RdYlGn_r"  # green (0) = cheap, red (1) = expensive
    cost_layers = [
        ("Slope", tm.slope_cost),
        ("Step", tm.step_cost),
        ("Roughness", tm.roughness_cost),
        ("Traversability", tm.traversability),
    ]
    for col, (name, cost) in enumerate(cost_layers, start=1):
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=tm.elevation,
                surfacecolor=cost,
                colorscale=cost_scale,
                cmin=0.0,
                cmax=1.0,
                showscale=(col == 4),
                colorbar=dict(title="cost", len=0.45, y=0.22) if col == 4 else None,
            ),
            row=2,
            col=col,
        )

    # --- Layout ---
    zmin = float(
        np.nanmin([np.nanmin(tm.max), np.nanmin(tm.mean), np.nanmin(tm.elevation), pts[:, 2].min()])
    )
    zmax = float(
        np.nanmax([np.nanmax(tm.max), np.nanmax(tm.mean), np.nanmax(tm.elevation), pts[:, 2].max()])
    )
    scene = dict(
        xaxis=dict(range=[xmin, xmax]),
        yaxis=dict(range=[ymin, ymax]),
        zaxis=dict(range=[zmin, zmax]),
        aspectmode="cube",
    )
    scene_keys = {f"scene{i}" if i > 1 else "scene": scene for i in range(1, 9)}
    fig.update_layout(
        title=f"Terrain pipeline [{args.preset}] "
        f"({tm.max.shape[1]}×{tm.max.shape[0]} @ {RESOLUTION} m/cell)",
        height=900,
        **scene_keys,
    )

    out = "heightmap.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")
    print(f"max:  z=[{np.nanmin(tm.max):.3f}, {np.nanmax(tm.max):.3f}]")
    print(f"mean: z=[{np.nanmin(tm.mean):.3f}, {np.nanmax(tm.mean):.3f}]")
    print(f"slope       cost=[{tm.slope_cost.min():.3f}, {tm.slope_cost.max():.3f}]")
    print(f"step        cost=[{tm.step_cost.min():.3f}, {tm.step_cost.max():.3f}]")
    print(f"roughness   cost=[{tm.roughness_cost.min():.3f}, {tm.roughness_cost.max():.3f}]")
    print(f"traversable cost=[{tm.traversability.min():.3f}, {tm.traversability.max():.3f}]")


if __name__ == "__main__":
    main()
