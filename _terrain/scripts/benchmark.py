import time

import numpy as np
import warp as wp
from terrain_toolkit import HeightMapBuilder

from test_synthetic import make_synthetic_cloud

BOUNDS = (-5.0, 5.0, -5.0, 5.0)
RESOLUTION = 0.1
SIZES = [10_000, 100_000, 500_000, 1_000_000, 5_000_000]
RUNS = 10


def time_build(builder: HeightMapBuilder, pts: np.ndarray, runs: int) -> list[float]:
    # Warmup (kernel compile + first launch).
    builder.build(pts)
    wp.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        builder.build(pts)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    builder = HeightMapBuilder(RESOLUTION, BOUNDS)
    print(f"{'N':>10} {'median (ms)':>12} {'min (ms)':>10} {'M pts/s':>10}")
    for n in SIZES:
        pts = make_synthetic_cloud(n=n)
        ts = time_build(builder, pts, RUNS)
        med = float(np.median(ts)) * 1e3
        mn = min(ts) * 1e3
        mpts = n / (min(ts)) / 1e6
        print(f"{n:>10} {med:>12.3f} {mn:>10.3f} {mpts:>10.1f}")


if __name__ == "__main__":
    main()
