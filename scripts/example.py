"""Minimal example: configure and run the full terrain pipeline on synthetic data."""

import numpy as np
from helhest.perception import FilterConfig
from helhest.perception import TerrainPipeline
from helhest.perception import TraversabilityConfig

# ---------------------------------------------------------------------------
# Configuration — every parameter explicit
# ---------------------------------------------------------------------------

# Grid
RESOLUTION = 0.15  # meters per cell
BOUNDS = (-5.0, 5.0, -5.0, 5.0)  # (xmin, xmax, ymin, ymax) in meters

# Point cloud filter
Z_MAX = 3.0  # discard points above this height (m)

# Primary reduction fed into inpaint → smooth → cost
PRIMARY = "max"  # "max" | "mean" | "min"

# Inpainting (multigrid diffusion to fill empty cells)
INPAINT = True
INPAINT_ITERS_PER_LEVEL = 50
INPAINT_COARSE_ITERS = 200

# Gaussian smoothing after inpainting (0.0 = disabled)
SMOOTH_SIGMA = 0.8  # meters

# Traversability cost thresholds & weights
TRAVERSABILITY = TraversabilityConfig(
    max_slope_deg=60.0,  # slope saturates to cost=1 at this angle
    max_step_height_m=0.55,  # step height saturates to cost=1
    max_roughness_m=0.2,  # roughness (local std-dev) saturates to cost=1
    step_window_radius_m=0.15,  # morphological window for step detection
    roughness_window_radius_m=0.3,  # window for roughness (local std-dev)
    slope_weight=0.2,  # weight in combined cost (normalized by sum)
    step_weight=0.2,
    roughness_weight=0.6,
)

# Support-ratio filter + obstacle inflation + temporal hysteresis
FILTER = FilterConfig(
    support_radius_m=0.5,  # neighborhood radius for support check
    support_ratio=0.5,  # min fraction of measured cells to keep
    inflation_sigma_m=0.3,  # Gaussian sigma for obstacle dilation
    obstacle_threshold=0.8,  # cost above this is an obstacle source
    obstacle_growth_threshold=2.0,  # reject frame if obstacle count jumps by this factor
    rejection_limit_frames=5,  # force-accept after this many consecutive rejections
    min_obstacle_baseline=10,  # skip hysteresis until this many obstacles seen
)

# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

pipe = TerrainPipeline(
    resolution=RESOLUTION,
    bounds=BOUNDS,
    z_max=Z_MAX,
    primary=PRIMARY,
    inpaint=INPAINT,
    inpaint_iters_per_level=INPAINT_ITERS_PER_LEVEL,
    inpaint_coarse_iters=INPAINT_COARSE_ITERS,
    smooth_sigma=SMOOTH_SIGMA,
    traversability=TRAVERSABILITY,
    filter=FILTER,
)

# ---------------------------------------------------------------------------
# Generate synthetic point cloud (tilted plane + Gaussian bump)
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
n = 100_000
x = rng.uniform(BOUNDS[0], BOUNDS[1], n)
y = rng.uniform(BOUNDS[2], BOUNDS[3], n)
z = 0.1 * x + 0.05 * y + 1.5 * np.exp(-((x - 1) ** 2 + (y + 1) ** 2) / 1.5)
z += rng.normal(0.0, 0.02, n)
points = np.stack([x, y, z], axis=1).astype(np.float32)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

tm = pipe.process(points)

print(f"Grid shape: {tm.elevation.shape} ({pipe.width}x{pipe.height} @ {RESOLUTION} m/cell)")
print(f"Elevation:     [{np.nanmin(tm.elevation):.3f}, {np.nanmax(tm.elevation):.3f}]")
print(f"Slope cost:    [{np.nanmin(tm.slope_cost):.3f}, {np.nanmax(tm.slope_cost):.3f}]")
print(f"Step cost:     [{np.nanmin(tm.step_cost):.3f}, {np.nanmax(tm.step_cost):.3f}]")
print(f"Roughness cost:[{np.nanmin(tm.roughness_cost):.3f}, {np.nanmax(tm.roughness_cost):.3f}]")
print(f"Traversability:[{np.nanmin(tm.traversability):.3f}, {np.nanmax(tm.traversability):.3f}]")
nan_pct = 100 * np.isnan(tm.traversability).sum() / tm.traversability.size
print(f"NaN cells:     {nan_pct:.1f}%")
