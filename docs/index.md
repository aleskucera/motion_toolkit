# Terrain Toolkit

GPU-accelerated point cloud → heightmap → traversability cost map using
[NVIDIA Warp](https://github.com/NVIDIA/warp).

Full pipeline on a 68k-point Ouster frame runs in **~5.5 ms / frame** on an
RTX A500 laptop GPU (≈180 FPS).

## What's in the box

| Stage | Module | Purpose |
|---|---|---|
| Outlier removal | `outlier/` | `StatisticalOutlierFilter` (SOR) or `RadiusOutlierFilter` (ROR), GPU-native k-NN via `wp.HashGrid` |
| Heightmap raster | `heightmap/` | `HeightMapBuilder` — max/mean/min/count layers in one kernel pass |
| Hole filling | `heightmap/` | `multigrid_inpaint` — NaN-aware Laplace diffusion on a pyramid |
| Smoothing | `heightmap/` | `gaussian_smooth` — NaN-aware separable blur |
| Geometric cost | `traversability/` | `GeometricTraversabilityAnalyzer` — slope + signed step + roughness |
| Post-process | `traversability/` | `ObstacleInflator`, `TemporalGate`, `SupportRatioMask` |
| Orchestration | `pipeline.py` | `TerrainPipeline` — points in, `TerrainMap` out |
| ICP | `icp/` | `IcpAligner` — GPU-native point-to-point ICP (standalone, not in the pipeline) |

## Install

```bash
uv sync                 # runtime
uv sync --group dev     # + matplotlib, plotly
```

Requirements: Python ≥ 3.12, NVIDIA GPU with CUDA support,
[uv](https://docs.astral.sh/uv/).

## Quick start

```python
import numpy as np
from terrain_toolkit import (
    TerrainPipeline, TraversabilityConfig, FilterConfig,
    RadiusOutlierFilterConfig,
)

pipe = TerrainPipeline(
    resolution=0.1,
    bounds=(-5, 5, -5, 5),
    outlier=RadiusOutlierFilterConfig(),       # fast, radius-based
    traversability=TraversabilityConfig(),
    filter=FilterConfig(),
    layers=("traversability",),                # only download what you need
)

tm = pipe.process(points)                      # points: (N, 3) float32
cost = tm.traversability                       # (H, W) float32, NaN = unknown
```

## Where to go next

<div class="grid cards" markdown>

-   :material-book-open-variant: **[Pipeline reference](pipeline.md)**

    `TerrainPipeline` and `TerrainMap` — constructor knobs, stage order,
    selective download, usage patterns.

-   :material-filter-variant: **[Outlier filtering](outlier.md)**

    SOR vs ROR, when to pick which, full config reference.

-   :material-chart-areaspline: **[Heightmap building blocks](heightmap.md)**

    `HeightMapBuilder`, `multigrid_inpaint`, `gaussian_smooth` used standalone.

-   :material-map-marker-path: **[Traversability](traversability.md)**

    Cost layers (slope, signed step, roughness), filter chain, tuning guide.

-   :material-speedometer: **[Performance](performance.md)**

    Per-stage profile, what dominates, optimization history, how to measure.

</div>
