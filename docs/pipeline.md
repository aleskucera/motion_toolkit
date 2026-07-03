# Pipeline reference

`TerrainPipeline` chains the stages — outlier → rasterize → inpaint → smooth
→ cost → filter — into one call. It owns all GPU buffers and reuses them
frame-to-frame, so a single instance should be constructed once and called
every frame.

## Construction

```python
TerrainPipeline(
    resolution: float,
    bounds: tuple[float, float, float, float],
    *,
    primary: Literal["max", "mean", "min"] = "max",
    inpaint: bool = True,
    smooth_sigma: float = 0.0,
    inpaint_iters_per_level: int = 50,
    inpaint_coarse_iters: int = 200,
    z_max: float | None = None,
    outlier: OutlierFilterConfig | RadiusOutlierFilterConfig | None = None,
    traversability: TraversabilityConfig | None = None,
    filter: FilterConfig | None = None,
    layers: tuple[str, ...] | None = None,
)
```

### Parameters

| Name | Purpose |
|---|---|
| `resolution` | Grid resolution in meters per cell. |
| `bounds` | `(xmin, xmax, ymin, ymax)` in meters. |
| `primary` | Which raw reduction feeds the cost stack. `max` is the default for top-surface terrain. |
| `inpaint` | Fill NaN cells via multigrid diffusion before smoothing/cost. Required when `traversability` is enabled. |
| `smooth_sigma` | Gaussian smoothing sigma in *cells*. `0.0` disables the pass entirely. |
| `inpaint_iters_per_level`, `inpaint_coarse_iters` | Jacobi iteration counts. Defaults are conservative — on small grids you can drop them substantially. |
| `z_max` | CPU pre-filter: drop points above this height. Use `None` to skip. |
| `outlier` | `OutlierFilterConfig` → SOR, `RadiusOutlierFilterConfig` → ROR, `None` disables. See [outlier.md](outlier.md). |
| `traversability` | Enable the cost analyzer. `None` disables the entire cost stack. |
| `filter` | Enable `ObstacleInflator` → `TemporalGate` → `SupportRatioMask`. Only meaningful when `traversability` is set. |
| `layers` | Which layers to download at the end. `None` = all. Skipping unused layers saves ~0.05-0.1 ms each. |

## `TerrainPipeline.process(points) → TerrainMap`

Takes `(N, 3) float32` points. Fully synchronous — returns a populated
`TerrainMap` once the frame is done.

Internally:

1. **`z_max` filter** (if set) — CPU boolean mask on `points[:, 2]`.
2. **Upload** points to GPU as `wp.vec3`.
3. **Outlier** — SOR or ROR, or skip.
4. **Rasterize** → `max`, `mean`, `min`, `count` layers (single kernel).
5. **Inpaint** — multigrid diffusion fills NaN cells in the primary layer.
6. **Smooth** — NaN-aware separable Gaussian (skipped if `smooth_sigma == 0`).
7. **Analyze** — slope + signed step + roughness → combined total cost.
8. **Inflate** — Gaussian obstacle dilation.
9. **Temporal gate** — reject frame if obstacle count spikes; emit `rejected_frame()` if so.
10. **Support-ratio mask** — NaN low-support cells.
11. **Download** — one `wp.synchronize()`, then D2H copy of the selected layers.

Only step 11 touches the host, and it only copies layers you asked for.

## `TerrainMap`

All fields are `np.ndarray | None`. A field is `None` when:
- its stage was not configured (e.g. `traversability` is `None` if `filter=None`), OR
- it was excluded from the `layers` selection.

| Field | Stage that produces it | Shape | dtype |
|---|---|---|---|
| `max`, `mean`, `min` | rasterize | (H, W) | float32 |
| `count` | rasterize | (H, W) | int32 |
| `elevation` | primary → inpaint → smooth | (H, W) | float32 |
| `slope_cost`, `step_cost`, `roughness_cost` | analyzer | (H, W) | float32 |
| `traversability` | combined cost → inflate → gate → support mask | (H, W) | float32 |

`TerrainMap.as_dict()` returns the non-`None` subset as a flat
`{name: ndarray}` dict, convenient for passing to plotting code.

## Selective download

By default every populated stage's output is copied to CPU:

```python
pipe = TerrainPipeline(resolution=0.1, bounds=..., traversability=..., filter=...)
# ≈ 5.3 ms / frame, 9 D2H copies
```

If you only consume the final cost map:

```python
pipe = TerrainPipeline(..., layers=("traversability",))
# ≈ 4.8 ms / frame, 1 D2H copy
```

The compute is the same — selective download only skips the copies.
Cost layers (`slope_cost`, `step_cost`, `roughness_cost`, `traversability`) are
automatically dropped from the selection when `traversability` is `None`.

## Stateful fields

`TerrainPipeline` caches:
- Outlier filter buffers (per-point scratch + hashgrid, shared across frames).
- Heightmap builder grids (`max/mean/min/count` buffers).
- Analyzer buffers (`normals`, `slope`, `step`, `rough`, `total`, `dilated`, `eroded`).
- Inflator output buffer.
- Temporal gate's previous-obstacle-count and rejection counter (actual
  hysteresis state).
- Support-mask output buffer.

Reusing a single instance across frames avoids allocating ~2 MB of GPU buffers
every frame and is the intended usage.

## Usage patterns

### Stream a single-sensor loop

```python
pipe = TerrainPipeline(
    resolution=0.1,
    bounds=(xmin, xmax, ymin, ymax),
    outlier=RadiusOutlierFilterConfig(),
    traversability=TraversabilityConfig(),
    filter=FilterConfig(),
    layers=("traversability", "elevation"),
)
while True:
    frame = next_frame()
    tm = pipe.process(frame)
    publish(tm.traversability)
```

### Offline batch, all layers

```python
pipe = TerrainPipeline(resolution=0.05, bounds=big_bounds,
                      traversability=TraversabilityConfig(), filter=FilterConfig())
for frame in frames:
    tm = pipe.process(frame)
    save(tm.as_dict())
```

### Use stages directly without `TerrainPipeline`

The stage classes and free functions are all exported — construct them yourself
for custom pipelines. See [`heightmap.md`](heightmap.md) and
[`traversability.md`](traversability.md) for the building blocks.
