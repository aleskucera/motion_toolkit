from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .kernels import compact_inliers_kernel
from .kernels import mean_dist_in_radius_kernel
from .kernels import radius_outlier_filter_kernel


@dataclass
class OutlierFilterConfig:
    """Configuration for `StatisticalOutlierFilter`."""

    # Search radius (meters) for neighbor lookup. Each point's mean distance is
    # computed over all neighbors inside this radius.
    search_radius_m: float = 0.25
    # Points with fewer than this many neighbors inside `search_radius_m` are
    # rejected outright (too isolated to produce a reliable statistic).
    min_neighbors: int = 10
    # Reject points whose range-normalized mean neighbor distance exceeds
    # μ + std_multiplier·σ, where statistics are taken over all valid points.
    std_multiplier: float = 1.0
    # Sensor origin in the same frame as the points. The mean neighbor distance
    # is divided by ||p - origin|| before thresholding, which compensates for
    # the ≈ linear growth of lidar point spacing with range.
    sensor_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Floor on the range divisor to avoid blow-up for points at the sensor.
    range_eps_m: float = 0.1


def _hashgrid_dims_from_extent(extent: np.ndarray, radius: float) -> tuple[int, int, int]:
    extent = np.maximum(extent, radius)
    cells = np.ceil(extent / max(radius, 1.0e-6)).astype(int)
    cells = np.clip(cells, 8, 256)
    return int(cells[0]), int(cells[1]), int(cells[2])


def _hashgrid_dims_from_points(points_np: np.ndarray, radius: float) -> tuple[int, int, int]:
    mins = points_np.min(axis=0)
    maxs = points_np.max(axis=0)
    return _hashgrid_dims_from_extent(maxs - mins, radius)


def _hashgrid_dims_from_bounds(
    bounds: tuple[float, float, float, float, float, float],
    radius: float,
) -> tuple[int, int, int]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    extent = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float64)
    return _hashgrid_dims_from_extent(extent, radius)


class StatisticalOutlierFilter:
    """GPU-native k-NN distance-based statistical outlier removal.

    Per-point range-normalized mean distance to k nearest neighbors; reject
    points exceeding μ + std_multiplier·σ (statistics over all valid points).
    Points with fewer than k neighbors in `search_radius_m` are also rejected.

    All reductions and compaction happen on the GPU — only μ/σ scalars and the
    output count are read back (≤ 16 bytes per call). Accepts numpy or
    `wp.array` input; returns the matching type.
    """

    def __init__(
        self,
        config: OutlierFilterConfig | None = None,
        *,
        bounds: tuple[float, float, float, float, float, float] | None = None,
        device: wp.context.Device | None = None,
    ):
        self.config = config or OutlierFilterConfig()
        self.device = wp.get_device(device)
        self._grid: wp.HashGrid | None = None

        # If the caller knows the point-cloud extent ahead of time, precreate
        # the hashgrid so we never touch the CPU to size it. Otherwise the
        # first apply() does a one-time numpy readback to pick dims.
        if bounds is not None:
            dims = _hashgrid_dims_from_bounds(bounds, self.config.search_radius_m)
            with wp.ScopedDevice(self.device):
                self._grid = wp.HashGrid(*dims, device=self.device)

        # Per-point outputs, grown on demand.
        self._mean_dist: wp.array | None = None
        self._valid: wp.array | None = None
        self._out_pts: wp.array | None = None
        self._capacity: int = 0

        # Scalar reduction buffers (fixed size).
        with wp.ScopedDevice(self.device):
            self._sum = wp.zeros(1, dtype=wp.float32)
            self._sum_sq = wp.zeros(1, dtype=wp.float32)
            self._count = wp.zeros(1, dtype=wp.int32)
            self._out_counter = wp.zeros(1, dtype=wp.int32)

    def _ensure_grid(self, radius: float, pts_wp: wp.array) -> wp.HashGrid:
        if self._grid is None or self._grid.device != self.device:
            # One-time readback to size the grid when no bounds were supplied.
            dims = _hashgrid_dims_from_points(pts_wp.numpy(), radius)
            self._grid = wp.HashGrid(*dims, device=self.device)
        return self._grid

    def _ensure_buffers(self, n: int) -> None:
        if self._capacity >= n and self._mean_dist is not None:
            return
        with wp.ScopedDevice(self.device):
            self._mean_dist = wp.empty(n, dtype=wp.float32)
            self._valid = wp.empty(n, dtype=wp.int32)
            self._out_pts = wp.empty(n, dtype=wp.vec3)
        self._capacity = n

    def apply(self, points: np.ndarray | wp.array) -> np.ndarray | wp.array:
        """Return `points` with outliers removed. Input and output types match."""
        cfg = self.config
        return_numpy = isinstance(points, np.ndarray)

        if return_numpy:
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"points must be (N, 3); got {points.shape}")
            n = len(points)
            pts_np_f32 = np.ascontiguousarray(points, dtype=np.float32)
            pts_wp = wp.array(pts_np_f32, dtype=wp.vec3, device=self.device)
        else:
            n = len(points)
            pts_wp = points

        if n <= cfg.min_neighbors:
            return points

        with wp.ScopedDevice(self.device):
            grid = self._ensure_grid(cfg.search_radius_m, pts_wp)
            grid.build(points=pts_wp, radius=float(cfg.search_radius_m))

            self._ensure_buffers(n)
            origin = wp.vec3(
                float(cfg.sensor_origin[0]),
                float(cfg.sensor_origin[1]),
                float(cfg.sensor_origin[2]),
            )

            self._sum.zero_()
            self._sum_sq.zero_()
            self._count.zero_()
            wp.launch(
                mean_dist_in_radius_kernel,
                dim=n,
                inputs=[
                    grid.id,
                    pts_wp,
                    float(cfg.search_radius_m),
                    int(cfg.min_neighbors),
                    origin,
                    float(cfg.range_eps_m),
                ],
                outputs=[
                    self._mean_dist,
                    self._valid,
                    self._sum,
                    self._sum_sq,
                    self._count,
                ],
            )
            wp.synchronize()
            s = float(self._sum.numpy()[0])
            ssq = float(self._sum_sq.numpy()[0])
            count_valid = int(self._count.numpy()[0])

            if count_valid == 0:
                if return_numpy:
                    return points[:0]
                return wp.empty(0, dtype=wp.vec3, device=self.device)

            mu = s / count_valid
            var = max(0.0, ssq / count_valid - mu * mu)
            sigma = math.sqrt(var)
            threshold = mu + cfg.std_multiplier * sigma

            self._out_counter.zero_()
            wp.launch(
                compact_inliers_kernel,
                dim=n,
                inputs=[
                    pts_wp,
                    self._mean_dist,
                    self._valid,
                    float(threshold),
                ],
                outputs=[self._out_counter, self._out_pts],
            )
            wp.synchronize()
            n_out = int(self._out_counter.numpy()[0])

            if return_numpy:
                return self._out_pts.numpy()[:n_out].astype(points.dtype, copy=True)

            # GPU path: allocate a right-sized output and copy from the compact buffer.
            out = wp.empty(n_out, dtype=wp.vec3, device=self.device)
            if n_out > 0:
                wp.copy(out, self._out_pts, 0, 0, n_out)
            return out


@dataclass
class RadiusOutlierFilterConfig:
    """Configuration for `RadiusOutlierFilter`."""

    # Search radius (meters) for neighbor lookup.
    search_radius_m: float = 0.25
    # A point is kept iff at least this many other points lie within the radius.
    min_neighbors: int = 10


class RadiusOutlierFilter:
    """GPU-native Radius Outlier Removal (ROR).

    Keeps points with at least `min_neighbors` other points inside `search_radius_m`.
    Single fused kernel — counts neighbors, early-exits once the threshold is hit,
    and writes survivors straight into a compact output. No per-point distance
    statistics, no global μ/σ, no second launch. Accepts numpy or `wp.array` input
    and returns the matching type.
    """

    def __init__(
        self,
        config: RadiusOutlierFilterConfig | None = None,
        *,
        bounds: tuple[float, float, float, float, float, float] | None = None,
        device: wp.context.Device | None = None,
    ):
        self.config = config or RadiusOutlierFilterConfig()
        self.device = wp.get_device(device)
        self._grid: wp.HashGrid | None = None

        if bounds is not None:
            dims = _hashgrid_dims_from_bounds(bounds, self.config.search_radius_m)
            with wp.ScopedDevice(self.device):
                self._grid = wp.HashGrid(*dims, device=self.device)

        self._out_pts: wp.array | None = None
        self._capacity: int = 0
        with wp.ScopedDevice(self.device):
            self._out_counter = wp.zeros(1, dtype=wp.int32)

    def _ensure_grid(self, radius: float, pts_wp: wp.array) -> wp.HashGrid:
        if self._grid is None or self._grid.device != self.device:
            dims = _hashgrid_dims_from_points(pts_wp.numpy(), radius)
            self._grid = wp.HashGrid(*dims, device=self.device)
        return self._grid

    def _ensure_buffers(self, n: int) -> None:
        if self._capacity >= n and self._out_pts is not None:
            return
        with wp.ScopedDevice(self.device):
            self._out_pts = wp.empty(n, dtype=wp.vec3)
        self._capacity = n

    def apply(self, points: np.ndarray | wp.array) -> np.ndarray | wp.array:
        """Return `points` with outliers removed. Input and output types match."""
        cfg = self.config
        return_numpy = isinstance(points, np.ndarray)

        if return_numpy:
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"points must be (N, 3); got {points.shape}")
            n = len(points)
            pts_np_f32 = np.ascontiguousarray(points, dtype=np.float32)
            pts_wp = wp.array(pts_np_f32, dtype=wp.vec3, device=self.device)
        else:
            n = len(points)
            pts_wp = points

        if n <= cfg.min_neighbors:
            return points

        with wp.ScopedDevice(self.device):
            grid = self._ensure_grid(cfg.search_radius_m, pts_wp)
            grid.build(points=pts_wp, radius=float(cfg.search_radius_m))

            self._ensure_buffers(n)
            self._out_counter.zero_()
            wp.launch(
                radius_outlier_filter_kernel,
                dim=n,
                inputs=[
                    grid.id,
                    pts_wp,
                    float(cfg.search_radius_m),
                    int(cfg.min_neighbors),
                ],
                outputs=[self._out_counter, self._out_pts],
            )
            wp.synchronize()
            n_out = int(self._out_counter.numpy()[0])

            if return_numpy:
                return self._out_pts.numpy()[:n_out].astype(points.dtype, copy=True)
            out = wp.empty(n_out, dtype=wp.vec3, device=self.device)
            if n_out > 0:
                wp.copy(out, self._out_pts, 0, 0, n_out)
            return out
