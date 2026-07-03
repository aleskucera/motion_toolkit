# Performance

## Current profile

Measured on an **RTX A500 (4 GiB, sm_86)** with a real Ouster frame
(`ouster.npy`, 68,712 points, 81×158 grid at 0.1 m/cell). Defaults throughout
(ROR outlier, full traversability stack, all layers downloaded).

| Stage | ms (median) | % of frame |
|---|---:|---:|
| upload | 0.22 | 3.9% |
| **outlier (ROR)** | **2.90** | **52.2%** |
| heightmap build | 0.14 | 2.5% |
| inpaint | 1.27 | 22.8% |
| smooth | 0.11 | 2.0% |
| analyzer | 0.09 | 1.7% |
| inflate | 0.11 | 2.0% |
| temporal gate | 0.08 | 1.4% |
| support mask | 0.05 | 0.9% |
| download | 0.58 | 10.4% |
| **total** | **5.55** | **100%** |

**≈ 180 FPS** on laptop-grade hardware.

Reproduce with `python profile_pipeline.py --path ouster.npy`. Each stage is
bracketed by `wp.synchronize()` so the per-stage GPU work is attributed
correctly.

## How to measure

```bash
# default (ROR outlier)
python profile_pipeline.py --path ouster.npy

# use SOR instead for comparison
python profile_pipeline.py --path ouster.npy --sor

# skip outlier filtering
python profile_pipeline.py --path ouster.npy --no-outlier

# tune iteration count / warmup
python profile_pipeline.py --path ouster.npy --frames 200 --warmup 20
```

The profiler reports the **median** of N frames after the warmup runs.

## Selective download

Restricting `layers=` lowers the download and the total:

| `layers` selector | total (ms) |
|---|---:|
| `None` (all 9 layers) | 5.34 |
| `("traversability", "elevation")` | 4.88 |
| `("traversability",)` | 4.82 |

The compute is identical; only D2H copies change. Useful if you're publishing
only the final cost map to a consumer.

## What dominates

After the current round of optimizations:

1. **Outlier (52%)** — hashgrid build + fused k-NN kernel. ROR's early-exit
   already caps the per-point work; remaining cost is the grid build and the
   single kernel launch itself. Nothing obvious left without changing
   algorithms (voxel downsample first, range-image filter, …).
2. **Inpaint (23%)** — dense Jacobi over a 5-level pyramid, ~800 kernel
   invocations per frame. Mostly wasted work on cells that aren't holes.
   Sparse iteration or a push-pull pyramid would replace this; not yet
   implemented.
3. **Download (10%)** — constant with selective download off. See above.

Everything else is < 5% and roughly at its per-launch floor — fusing these
small kernels would shave < 0.1 ms total.

## Optimization history (what's been done)

- **SOR → ROR for the default outlier path** (#6×)
  Early-exit at `min_neighbors` — 16.5 ms → 2.9 ms on this dataset.
- **Fused atomic reduction in SOR's mean-distance kernel**
  One kernel launch fewer; ~0.5 ms saved on the SOR path.
- **Lazy hashgrid + optional `bounds`**
  Skipped the per-frame `points.numpy()` readback that used to size the grid.
  Optional `bounds=...` skips even the first-call readback entirely.
- **Selective download**
  `TerrainMap` fields are now all optional; only selected ones are copied to CPU.
- **Signed step-height**
  No speed impact, but quality improvement: drops and bumps are now scored
  independently with their own saturation thresholds.

## Potential future wins

In rough order of ROI:

1. **Sparse inpaint** — iterate only hole cells, or switch to a push-pull
   (normalized-convolution) pyramid. Expected 0.5–1 ms on typical lidar data.
2. **Skip the `isfinite` readback in `_fixed_mask_from`** — currently
   downloads the heightmap, builds the mask in numpy, uploads it back. Should
   be a single kernel pass on-device.
3. **Return `wp.array` optionally** — skip the entire download (0.58 ms) if
   the consumer can accept GPU pointers directly.
4. **Overlap upload / compute / download with streams** — halves steady-state
   latency when streaming.
5. **Kernel fusion across analyzer stages** — slope + step + roughness all
   read overlapping 3×3 to 5×5 elevation windows; fusing saves launches and
   redundant loads. Marginal (~0.1 ms) at the current grid size.

## Hardware scaling

The profile above is the worst-case — an RTX A500 is a 2048-core laptop GPU
at ~3 TFLOPS. On a desktop 30/40-series card expect every stage to shrink
roughly proportionally to memory bandwidth / SM count. Kernel launch overhead
dominates at small grid sizes so the relative stage percentages may shift.
