from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .kernels import finalize_kernel
from .kernels import rasterize_all_kernel


@dataclass
class HeightMapLayers:
    """Per-cell reductions from a single rasterization pass, on GPU.

    `max`, `mean`, `min` hold NaN for empty cells; `count` holds 0.

    NOTE: these arrays are owned by `HeightMapBuilder` and reused on the next
    `build()` call. Call `.to_numpy()` (or `array.numpy().copy()` per field)
    if you need to retain values beyond the current frame.
    """

    max: wp.array
    mean: wp.array
    min: wp.array
    count: wp.array

    def __getitem__(self, name: str) -> wp.array:
        return getattr(self, name)

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Download all four layers to numpy (copies — safe to retain)."""
        return {
            "max": self.max.numpy().copy(),
            "mean": self.mean.numpy().copy(),
            "min": self.min.numpy().copy(),
            "count": self.count.numpy().copy(),
        }


class HeightMapBuilder:
    """Rasterize 3D points into a 2D grid of max/mean/min/count layers.

    Preallocates the five GPU grid buffers once; `build()` only resets them
    in-place each call. Accepts points as either numpy (uploaded each call)
    or a pre-uploaded `wp.array` of `vec3` (used directly, no copy).
    """

    def __init__(
        self,
        resolution: float,
        bounds: tuple[float, float, float, float],
        *,
        device: wp.context.Device | None = None,
    ):
        xmin, xmax, ymin, ymax = bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Invalid bounds.")
        self.resolution = resolution
        self.bounds = bounds
        self.width = int(math.ceil((xmax - xmin) / resolution))
        self.height = int(math.ceil((ymax - ymin) / resolution))
        self.shape = (self.height, self.width)
        self.device = wp.get_device(device)

        with wp.ScopedDevice(self.device):
            self._max = wp.empty(self.shape, dtype=wp.float32)
            self._min = wp.empty(self.shape, dtype=wp.float32)
            self._sum = wp.empty(self.shape, dtype=wp.float32)
            self._count = wp.empty(self.shape, dtype=wp.int32)
            self._mean = wp.empty(self.shape, dtype=wp.float32)

        self._layers = HeightMapLayers(
            max=self._max,
            mean=self._mean,
            min=self._min,
            count=self._count,
        )

    def build(self, points: np.ndarray | wp.array) -> HeightMapLayers:
        """Scatter (N, 3) points into the grid. Input may be numpy or `wp.array`."""
        if isinstance(points, wp.array):
            pts_wp = points
            n = len(points)
        else:
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("points must have shape (N, 3).")
            n = len(points)
            pts_np = np.ascontiguousarray(points, dtype=np.float32)
            pts_wp = wp.array(pts_np, dtype=wp.vec3, device=self.device)

        self._max.fill_(-float("inf"))
        self._min.fill_(float("inf"))
        self._sum.zero_()
        self._count.zero_()

        xmin, _, ymin, _ = self.bounds
        with wp.ScopedDevice(self.device):
            wp.launch(
                rasterize_all_kernel,
                dim=n,
                inputs=[
                    pts_wp,
                    float(xmin),
                    float(ymin),
                    float(1.0 / self.resolution),
                    int(self.width),
                    int(self.height),
                    self._max,
                    self._min,
                    self._sum,
                    self._count,
                ],
            )
            wp.launch(
                finalize_kernel,
                dim=self.shape,
                inputs=[self._sum, self._count, self._max, self._min, self._mean],
            )

        return self._layers
