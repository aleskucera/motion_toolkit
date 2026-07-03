# Terrain Toolkit

GPU-accelerated point cloud → heightmap → traversability cost map using
[NVIDIA Warp](https://github.com/NVIDIA/warp).

Full pipeline on a 68k-point Ouster frame runs in **~5.5 ms / frame** on an
RTX A500 laptop GPU (≈180 FPS).

**📖 Full documentation:** [aleskucera.github.io/terrain_toolkit](https://aleskucera.github.io/terrain_toolkit/)

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

## Requirements

- Python ≥ 3.12
- NVIDIA GPU with CUDA support
- [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv sync          # runtime
uv sync --group dev   # + matplotlib, plotly
```

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

See [`example.py`](example.py) for a fully-explicit configuration and
[`profile_pipeline.py`](profile_pipeline.py) for a per-stage timing harness.

## Deeper docs

- [**Pipeline reference**](docs/pipeline.md) — `TerrainPipeline`, `TerrainMap`,
  selective download, layer semantics
- [**Outlier filtering**](docs/outlier.md) — SOR vs ROR, config, which to pick
- [**Heightmap building blocks**](docs/heightmap.md) — builder, inpaint, smooth
- [**Traversability**](docs/traversability.md) — cost layers, filter chain,
  tuning guide
- [**Performance**](docs/performance.md) — current per-stage profile,
  optimization history, how to measure

## Test scripts

```bash
uv run python example.py                              # synthetic data
uv run python test_synthetic.py --preset noisy        # tilted plane + bump
uv run python test_ouster.py --path ouster.npy        # real lidar
uv run python profile_pipeline.py --path ouster.npy   # per-stage profile
```

## License

MIT — see [LICENSE](LICENSE).
