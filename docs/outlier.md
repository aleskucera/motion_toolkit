# Outlier filtering

Two GPU-native filters ship with the toolkit. They share the same
`wp.HashGrid`-backed k-NN infrastructure but make different quality / speed
tradeoffs.

## Which to pick

| | StatisticalOutlierFilter (SOR) | RadiusOutlierFilter (ROR) |
|---|---|---|
| **Rejection rule** | μ + k·σ of range-normalized mean neighbor distance | `count < min_neighbors` |
| **Adaptivity** | threshold adapts to frame density | fixed |
| **Early exit** | no (needs full mean) | yes (stops at `min_neighbors`) |
| **Typical cost** | ~17 ms @ 68k pts, r=0.5 | ~2.9 ms @ 68k pts, r=0.25 |
| **Default?** | — | **yes** |

ROR is the default — roughly 6× faster than SOR on real lidar data, with
comparable-or-better rejection quality because the early-exit scales the
per-point work to the neighborhood density rather than the radius. Use SOR if
you need a *statistical* threshold that adapts to each frame's density, but
expect to pay for it.

## Radius Outlier Removal (ROR)

Keep a point iff it has at least `min_neighbors` other points within
`search_radius_m`.

```python
from terrain_toolkit import RadiusOutlierFilter, RadiusOutlierFilterConfig

filt = RadiusOutlierFilter(RadiusOutlierFilterConfig(
    search_radius_m=0.25,   # default
    min_neighbors=10,       # default
))
clean = filt.apply(points)  # numpy or wp.array — matching type returned
```

Single fused kernel: counts neighbors in radius, early-exits at the threshold,
and atomically compacts survivors into the output buffer. No per-point
statistics, no global reduction, no μ/σ readback.

### Config

| Field | Default | Meaning |
|---|---|---|
| `search_radius_m` | `0.25` | Neighbor lookup radius. |
| `min_neighbors` | `10` | Keep point iff it has ≥ this many neighbors in radius. |

### Tuning

- **Lower `min_neighbors` = more permissive** (fewer rejections, more noise
  kept). Doesn't save much time because early-exit already caps the per-point
  work.
- **Smaller `search_radius_m` = stricter** (fewer neighbors fit in the
  sphere). Also doesn't save much time with ROR — early-exit dominates.
- Cost floor is ~2.8 ms @ 68k points on an RTX A500 at this grid density; the
  hashgrid build and kernel launch eat most of the remaining budget.

## Statistical Outlier Removal (SOR)

Computes per-point range-normalized mean distance to neighbors in radius, then
rejects points whose value exceeds μ + `std_multiplier`·σ of all valid points.
Range-normalization (dividing by `‖p − sensor_origin‖`) compensates for
lidar's linear point-spacing growth with range.

```python
from terrain_toolkit import StatisticalOutlierFilter, OutlierFilterConfig

filt = StatisticalOutlierFilter(OutlierFilterConfig(
    search_radius_m=0.25,
    min_neighbors=10,
    std_multiplier=1.0,
    sensor_origin=(0.0, 0.0, 0.0),
    range_eps_m=0.1,
))
clean = filt.apply(points)
```

Three kernels: per-point mean distance (with fused atomic reduction of
sum/sum²/count), threshold computation on CPU from a 16-byte readback, and
compaction.

### Config

| Field | Default | Meaning |
|---|---|---|
| `search_radius_m` | `0.25` | Neighbor lookup radius. |
| `min_neighbors` | `10` | Points with fewer neighbors are rejected outright. |
| `std_multiplier` | `1.0` | Threshold = μ + this × σ. Higher = more permissive. |
| `sensor_origin` | `(0, 0, 0)` | Sensor position in the same frame as the points. |
| `range_eps_m` | `0.1` | Floor on the range divisor; prevents blow-up near the sensor. |

## Reducing readback overhead

Both filters do *one* numpy round-trip on their first call to size the
internal `wp.HashGrid`. To skip that, pass `bounds` up front:

```python
filt = RadiusOutlierFilter(
    RadiusOutlierFilterConfig(),
    bounds=(-10.0, 5.0, -7.0, 1.5, -2.0, 5.0),  # (xmin, xmax, ymin, ymax, zmin, zmax)
)
```

Once the grid exists it is reused on every `apply()` — no per-frame CPU touch.

## Inputs / outputs

Both `apply` methods accept `np.ndarray` or `wp.array`, and return the
matching type. GPU-in → GPU-out is the fast path (no host copies either side);
the pipeline uses that path internally.

## Implementation notes

- The hashgrid is built on every frame (`grid.build(points, radius)`). Cost is
  linear in N.
- ROR's early-exit uses `break` inside the `wp.hash_grid_query` neighbor loop.
- SOR fuses the stats reduction into the mean-distance kernel — atomic
  sum/sum²/count are accumulated alongside the per-point output. One kernel
  launch fewer than the obvious two-pass implementation.
