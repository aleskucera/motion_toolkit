"""Per-stage profile of TerrainPipeline on real data.

Replays pipeline.process() with wp.synchronize() between stages so the GPU-side
work of each stage is attributed correctly. Reports median ms per stage and %
of total frame time.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp
from helhest.perception.heightmap import gaussian_smooth
from helhest.perception.heightmap import HeightMapBuilder
from helhest.perception.heightmap import multigrid_inpaint
from helhest.perception.outlier import OutlierFilterConfig
from helhest.perception.outlier import RadiusOutlierFilter
from helhest.perception.outlier import RadiusOutlierFilterConfig
from helhest.perception.outlier import StatisticalOutlierFilter
from helhest.perception.traversability import FilterConfig
from helhest.perception.traversability import GeometricTraversabilityAnalyzer
from helhest.perception.traversability import ObstacleInflator
from helhest.perception.traversability import SupportRatioMask
from helhest.perception.traversability import TemporalGate
from helhest.perception.traversability import TraversabilityConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="ouster.npy")
    p.add_argument("--resolution", type=float, default=0.1)
    p.add_argument("--smooth-sigma", type=float, default=1.0)
    p.add_argument("--z-max", type=float, default=3.0)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--no-outlier", action="store_true")
    p.add_argument(
        "--sor", action="store_true", help="use StatisticalOutlierFilter instead of default ROR"
    )
    return p.parse_args()


class Stopwatch:
    def __init__(self):
        self.laps: dict[str, list[float]] = {}
        self._t0 = 0.0

    def start(self):
        wp.synchronize()
        self._t0 = time.perf_counter()

    def lap(self, name: str):
        wp.synchronize()
        t = time.perf_counter()
        self.laps.setdefault(name, []).append(t - self._t0)
        self._t0 = t

    def report(self):
        medians = {n: float(np.median(v)) * 1000.0 for n, v in self.laps.items()}
        total = sum(medians.values())
        print(f"{'stage':<22} {'ms (median)':>12} {'%':>8}")
        print("-" * 44)
        for name, ms in medians.items():
            pct = ms / total * 100 if total > 0 else 0
            print(f"{name:<22} {ms:>12.3f} {pct:>7.1f}%")
        print("-" * 44)
        print(f"{'TOTAL':<22} {total:>12.3f} {'100.0':>7}%")


def main() -> None:
    args = parse_args()
    pts = np.load(args.path).astype(np.float32)[:, :3]
    pad = 0.5
    xmin, ymin = pts[:, 0].min() - pad, pts[:, 1].min() - pad
    xmax, ymax = pts[:, 0].max() + pad, pts[:, 1].max() + pad
    bounds = (float(xmin), float(xmax), float(ymin), float(ymax))
    print(f"input: {len(pts)} pts, bounds {bounds}, resolution {args.resolution}")

    # Build stages as the pipeline would.
    builder = HeightMapBuilder(args.resolution, bounds)
    H, W = builder.height, builder.width
    print(f"grid: {H}x{W}")

    if args.no_outlier:
        outlier = None
    elif args.sor:
        outlier = StatisticalOutlierFilter(OutlierFilterConfig())
    else:
        outlier = RadiusOutlierFilter(RadiusOutlierFilterConfig())
    analyzer = GeometricTraversabilityAnalyzer(args.resolution, H, W, TraversabilityConfig())
    fcfg = FilterConfig()
    inflator = ObstacleInflator(args.resolution, H, W, fcfg)
    gate = TemporalGate(fcfg)
    mask = SupportRatioMask(args.resolution, H, W, fcfg)

    def one_frame(sw: Stopwatch | None):
        # 1. Upload
        pts_wp = wp.array(np.ascontiguousarray(pts, dtype=np.float32), dtype=wp.vec3)
        if sw:
            sw.lap("upload")
        # 2. Outlier
        if outlier is not None:
            pts_wp = outlier.apply(pts_wp)
            if sw:
                sw.lap("outlier")
        # 3. Rasterize
        layers = builder.build(pts_wp)
        primary = layers.max
        if sw:
            sw.lap("heightmap_build")
        # 4. Multigrid inpaint
        elevation = multigrid_inpaint(primary)
        if sw:
            sw.lap("inpaint")
        # 5. Gaussian smooth
        elevation = gaussian_smooth(elevation, sigma=args.smooth_sigma)
        if sw:
            sw.lap("smooth")
        # 6. Traversability analyzer
        costs = analyzer.compute(elevation)
        if sw:
            sw.lap("analyzer")
        # 7. Obstacle inflate
        inflated = inflator.apply(costs.total)
        if sw:
            sw.lap("inflate")
        # 8. Temporal gate (does internal sync+readback)
        gate.is_stable(inflated)
        if sw:
            sw.lap("temporal_gate")
        # 9. Support mask
        total = mask.apply(primary, inflated)
        if sw:
            sw.lap("support_mask")
        # 10. Download
        out = {
            "max": layers.max.numpy().copy(),
            "mean": layers.mean.numpy().copy(),
            "min": layers.min.numpy().copy(),
            "count": layers.count.numpy().copy(),
            "elevation": elevation.numpy().copy(),
            "slope": costs.slope.numpy().copy(),
            "step": costs.step.numpy().copy(),
            "roughness": costs.roughness.numpy().copy(),
            "traversability": total.numpy().copy(),
        }
        if sw:
            sw.lap("download")
        return out

    # Warmup (JIT, first-frame allocations, etc.)
    for _ in range(args.warmup):
        one_frame(None)
    wp.synchronize()

    # Profile
    sw = Stopwatch()
    for _ in range(args.frames):
        sw.start()
        one_frame(sw)

    sw.report()


if __name__ == "__main__":
    main()
