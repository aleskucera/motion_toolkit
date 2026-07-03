# Heightmap building blocks

Three pieces, usable either standalone or as the lower half of
`TerrainPipeline`:

1. `HeightMapBuilder` — rasterize `(N, 3)` points into grid layers.
2. `multigrid_inpaint` — fill NaN cells via Laplace diffusion.
3. `gaussian_smooth` — NaN-aware separable blur.

All three accept `np.ndarray` or `wp.array` and return the matching type.

## `HeightMapBuilder`

One kernel pass over the point cloud populates four grids in parallel:

- `max` — highest z per cell (useful for top-surface terrain)
- `mean` — average z per cell
- `min` — lowest z per cell
- `count` — int32, number of points that landed in the cell

Empty cells are `NaN` (or `0` for `count`).

```python
from terrain_toolkit import HeightMapBuilder

builder = HeightMapBuilder(
    resolution=0.1,                    # meters per cell
    bounds=(-5.0, 5.0, -5.0, 5.0),     # (xmin, xmax, ymin, ymax)
)
layers = builder.build(points)         # → HeightMapLayers
# layers.max, layers.mean, layers.min, layers.count
```

Internally the builder uses float atomics for `mean` accumulation and compare-
and-swap for `max`/`min`. Grid dimensions are derived from `bounds` /
`resolution`.

## `multigrid_inpaint`

Fills `NaN` cells via Laplace diffusion on a pyramid. The algorithm:

1. Build a pyramid by 2×2 NaN-aware downsampling until the smallest side
   reaches `min_size`.
2. Solve the coarsest level with `coarse_iters` Jacobi iterations.
3. Upsample and run `iters_per_level` refinement iterations at each finer
   level.

Each level's iteration loop is CUDA-graph-captured, so Python launch overhead
is amortized.

```python
from terrain_toolkit import multigrid_inpaint

filled = multigrid_inpaint(
    heightmap,
    iters_per_level=50,    # default
    coarse_iters=200,      # default
    min_size=8,            # don't pyramid below this dimension
)
```

### Tuning

The defaults (50 / 200) are conservative for small grids. For typical lidar
grids (100×100 to 300×300 cells) you can drop to `(20, 80)` without visible
quality change. The coarse level is usually ~6×10 — 200 iters there is
overkill.

If `inpaint` is disabled on `TerrainPipeline`, the primary layer still flows
through but NaN cells remain, and `traversability` cannot be computed (the
cost kernels assume a fully filled grid — the constructor enforces this).

## `gaussian_smooth`

Separable NaN-aware Gaussian blur. `sigma` is in **cells**, not meters:

```python
from terrain_toolkit import gaussian_smooth

blurred = gaussian_smooth(heightmap, sigma=1.0, truncate=3.0)
```

- `sigma <= 0` returns a fresh copy (no kernel launches).
- Kernel radius is `ceil(truncate * sigma)` cells.
- NaN cells contribute 0 weight and 0 value, so the output NaN-hole pattern
  shrinks by `radius` cells (a partially-inside window still produces a
  value).

## `diffuse_inpaint` (single-resolution variant)

Same idea as `multigrid_inpaint` but without the pyramid — just Jacobi
iterations at full resolution. Simpler, slower to converge. Exposed as
`terrain_toolkit.diffuse_inpaint`. Use the multigrid version unless you have a
reason not to.

## When to skip the pipeline and use these directly

- You want only a heightmap, no traversability cost.
- You're running several pipelines with different configs over the same input
  (build once, analyze many).
- You're prototyping a custom cost function and want to reuse the raster +
  inpaint stages.

```python
builder = HeightMapBuilder(resolution, bounds)
layers = builder.build(points)
elev = gaussian_smooth(multigrid_inpaint(layers.max), sigma=1.0)
# your custom cost logic on `elev` (still a wp.array)
```
