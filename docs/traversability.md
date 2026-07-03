# Traversability

The cost stack turns a filled heightmap into a `(H, W) float32` traversability
map in `[0, 1]` (NaN for unknown). Higher = harder to cross; 1 = impassable.

## Stages

```
elevation  ──▶  GeometricTraversabilityAnalyzer  ──▶  ObstacleInflator  ──▶  TemporalGate  ──▶  SupportRatioMask  ──▶  traversability
                (slope + step + roughness → total)    (Gaussian dilation)     (hysteresis)       (NaN low-support)
```

All four live in `terrain_toolkit.traversability` and can be used directly.
`TerrainPipeline` chains them automatically.

## `GeometricTraversabilityAnalyzer`

Produces four cost layers in one pass:

| Layer | What it measures | Kernel |
|---|---|---|
| `slope_cost` | surface normal angle from vertical | Sobel gradients → `acos(n_z)` |
| `step_cost` | signed step height around each cell | morphological dilate/erode vs. elevation |
| `roughness_cost` | local std-dev of elevation | windowed mean + variance |
| `total` | weighted combination of the above | |

Each layer is normalized to `[0, 1]` by its saturation threshold and clamped.

### Config (`TraversabilityConfig`)

| Field | Default | Meaning |
|---|---|---|
| `max_slope_deg` | `60.0` | Slope cost saturates at this angle. |
| `max_step_height_m` | `0.55` | **Bump** (positive obstacle) cost saturates here. |
| `max_drop_height_m` | `0.3` | **Drop** (negative obstacle) cost saturates here. Set equal to `max_step_height_m` for unsigned behavior. |
| `max_roughness_m` | `0.2` | Roughness saturates at this local std-dev. |
| `step_window_radius_m` | `0.15` | Morphological window radius for step detection. |
| `roughness_window_radius_m` | `0.3` | Window for roughness local-σ. |
| `slope_weight`, `step_weight`, `roughness_weight` | `0.2 / 0.2 / 0.6` | Relative weights in the combined `total` cost (renormalized inside the kernel). |

### Signed step height

The step cost is `max(bump_cost, drop_cost)` where:

- `bump_cost = clamp((dilated − elev) / max_step_height_m)` — something taller
  nearby
- `drop_cost = clamp((elev − eroded) / max_drop_height_m)` — something lower
  nearby

Drops saturate sooner than bumps by default (0.3 m vs 0.55 m), so a cliff
edge or ditch costs more than a same-size curb or rock. This matters for
ground robots where falling off an edge is usually worse than climbing over
something of equal height.

To recover the old unsigned `dilated − eroded` behavior, set
`max_drop_height_m = max_step_height_m`.

### Roughness

Local std-dev of elevation in a square window. Flags lumpy or rubble-like
terrain that slope alone can't — a smooth 30° ramp and a cobbled 30° ramp have
similar slope but very different roughness.

### Combining

`total = (slope_weight·slope + step_weight·step + roughness_weight·rough) / (sum of weights)`

The default weights emphasize roughness (`0.6`) — on typical outdoor terrain
it's the most discriminating layer. Slope and step contribute equally (`0.2`
each).

## Filter chain (`FilterConfig`)

Three sequential post-processing stages, all configured from a single
`FilterConfig`:

### `ObstacleInflator`

Gaussian-weighted dilation of cells above `obstacle_threshold`. Each source's
influence decays with `exp(−d²/(2σ²))`; the kernel window extends to 3σ.
The output is `max(original, max_over_sources(source_cost × weight))` — costs
only go up, never down.

Use it to give the robot a safety buffer around high-cost cells without
over-smoothing low-cost terrain.

### `TemporalGate`

Per-frame stability check. If the obstacle count (cells above
`obstacle_threshold`) grows by more than `obstacle_growth_threshold` relative
to the previous accepted frame, reject the frame and emit
`SupportRatioMask.rejected_frame()` (a full-NaN map) instead. After
`rejection_limit_frames` consecutive rejections it force-accepts to prevent
permanent stuck state. The `min_obstacle_baseline` skips the check while
obstacle counts are too small to give a stable ratio.

This absorbs single-frame spikes (transient reflections, a person walking by)
without needing explicit object tracking.

### `SupportRatioMask`

Writes `NaN` into cells whose local neighborhood has fewer than
`support_ratio` measured cells. This is how the pipeline marks "this region's
cost came mostly from inpainting and shouldn't be trusted". Applied *after*
inflation, so inpainted regions still contribute to obstacle-neighborhood
detection — they just disappear from the final map.

### Config fields

| Field | Default | Stage | Meaning |
|---|---|---|---|
| `support_radius_m` | `0.5` | mask | Neighborhood radius for support check. |
| `support_ratio` | `0.5` | mask | Min fraction of measured cells in neighborhood to keep. |
| `inflation_sigma_m` | `0.3` | inflator | Gaussian σ for dilation weight. |
| `obstacle_threshold` | `0.8` | inflator + gate | Cells above this count as obstacles. |
| `obstacle_growth_threshold` | `2.0` | gate | Reject frame if obstacle count × this factor. |
| `rejection_limit_frames` | `5` | gate | Consecutive rejections before force-accept. |
| `min_obstacle_baseline` | `10` | gate | Skip hysteresis until this many obstacles seen. |

## Tuning notes

- **Start with `roughness_weight=0.6`**. It's the most informative layer on
  most outdoor terrain.
- **`max_slope_deg`** should match your robot's climbing ability. 60° is a
  generous default; legged/tracked robots typically want 30–40°, wheeled
  robots 20–25°.
- **`obstacle_threshold=0.8`** is conservative — only very red cells inflate.
  If your weights produce costs in the middle band more than at the
  saturation, lower the threshold.
- **`support_ratio`** trades map coverage for confidence. At `0.5` you keep
  roughly half of the field-of-view; lowering it extends the map into
  sparser regions.
