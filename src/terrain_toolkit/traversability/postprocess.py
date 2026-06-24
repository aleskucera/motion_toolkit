from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..grid_utils import meters_to_cells
from .kernels import (
    count_obstacles_kernel,
    inflate_obstacles_kernel,
    occlusion_mask_kernel,
    support_ratio_mask_kernel,
)


@dataclass
class FilterConfig:
    """Configuration for the traversability post-processing stages.

    Single bag shared by `ObstacleInflator`, `TemporalGate`, and
    `SupportRatioMask` — each stage reads only the fields it needs.
    """

    # Support ratio: cells whose neighborhood has fewer than this fraction of
    # measured cells are rejected (set to NaN).
    support_radius_m: float = 0.5
    support_ratio: float = 0.5

    # Gaussian-weighted obstacle inflation. Only cells above `obstacle_threshold`
    # act as sources; their influence decays with exp(-d²/2σ²). The kernel
    # window extends to 3σ.
    inflation_sigma_m: float = 0.3
    obstacle_threshold: float = 0.8

    # Temporal hysteresis: reject frames where obstacle count grows by more than
    # `obstacle_growth_threshold` relative to the last accepted frame, up to
    # `rejection_limit_frames` consecutive rejections before force-accepting.
    obstacle_growth_threshold: float = 2.0
    rejection_limit_frames: int = 5
    min_obstacle_baseline: int = 10


@dataclass
class OcclusionConfig:
    """Configuration for occlusion (line-of-sight) cost masking.

    Cells hidden from the sensor by an obstacle and not actually measured are
    NaN-ed out, so inpainted free-space behind walls is not treated as
    traversable. The sensor xy/z are in the grid (gravity-aligned robot) frame.
    """

    sensor_xy: tuple[float, float] = (0.0, 0.0)  # sensor position in grid frame (m)
    sensor_z: float = 0.5                        # sensor height above the grid origin (m)
    angle_eps_rad: float = 0.01                  # horizon margin; guards flat-ground noise


def _as_gpu(arr: np.ndarray | wp.array, device: wp.context.Device) -> wp.array:
    if isinstance(arr, wp.array):
        return arr
    return wp.array(
        np.ascontiguousarray(arr, dtype=np.float32),
        dtype=wp.float32, device=device,
    )


class ObstacleInflator:
    """Gaussian-weighted obstacle dilation on the cost map.

    Preallocates a single owned output buffer — the returned `wp.array` is
    overwritten on the next `apply()` call. Accepts numpy or `wp.array` input;
    output is always `wp.array` (callers can `.numpy()` if needed).
    """

    def __init__(
        self,
        resolution: float,
        height: int,
        width: int,
        config: FilterConfig | None = None,
        *,
        device: wp.context.Device | None = None,
    ):
        self.resolution = resolution
        self.height = height
        self.width = width
        self.shape = (height, width)
        self.config = config or FilterConfig()
        self.device = device if device is not None else wp.get_device()

        sigma_cells = self.config.inflation_sigma_m / resolution
        self.radius_cells = int(math.ceil(3.0 * sigma_cells))
        self.inv_two_sigma_sq = (
            1.0 / (2.0 * sigma_cells * sigma_cells) if sigma_cells > 0 else 0.0
        )

        with wp.ScopedDevice(self.device):
            self._inflated = wp.zeros(self.shape, dtype=wp.float32)

    def apply(self, cost_map: np.ndarray | wp.array) -> wp.array:
        if cost_map.shape != self.shape:
            raise ValueError(f"cost_map shape {cost_map.shape} != inflator shape {self.shape}")
        cost_wp = _as_gpu(cost_map, self.device)
        wp.launch(
            inflate_obstacles_kernel,
            dim=self.shape,
            inputs=[
                cost_wp,
                self.height,
                self.width,
                self.radius_cells,
                float(self.inv_two_sigma_sq),
                float(self.config.obstacle_threshold),
            ],
            outputs=[self._inflated],
            device=self.device,
        )
        return self._inflated


class TemporalGate:
    """Frame-level accept/reject based on obstacle-count growth.

    Stateful: tracks the last accepted frame's obstacle count and the
    consecutive-rejection counter, so the same instance must be reused across
    frames. After `rejection_limit_frames` consecutive rejections, the next
    frame is force-accepted to establish a new baseline.
    """

    def __init__(
        self,
        config: FilterConfig | None = None,
        *,
        device: wp.context.Device | None = None,
    ):
        self.config = config or FilterConfig()
        self.device = device if device is not None else wp.get_device()

        self._last_obstacle_count = 0
        self._consecutive_rejections = 0

        with wp.ScopedDevice(self.device):
            self._num_obstacles = wp.zeros(1, dtype=wp.int32)

    def is_stable(self, cost_map: wp.array) -> bool:
        """Count obstacles in `cost_map` and apply the growth hysteresis."""
        self._num_obstacles.zero_()
        wp.launch(
            count_obstacles_kernel,
            dim=cost_map.shape,
            inputs=[cost_map, float(self.config.obstacle_threshold)],
            outputs=[self._num_obstacles],
            device=self.device,
        )
        wp.synchronize()
        current = int(self._num_obstacles.numpy()[0])

        cfg = self.config
        if self._last_obstacle_count < cfg.min_obstacle_baseline:
            self._last_obstacle_count = current
            self._consecutive_rejections = 0
            return True

        growth = current / self._last_obstacle_count
        if growth > cfg.obstacle_growth_threshold:
            self._consecutive_rejections += 1
            if self._consecutive_rejections <= cfg.rejection_limit_frames:
                return False
            # Force-accept: establish a new baseline.

        self._consecutive_rejections = 0
        self._last_obstacle_count = current
        return True


class SupportRatioMask:
    """NaN-out cost cells whose raw-elevation neighborhood lacks measurements.

    Preallocates a single owned output buffer — the returned `wp.array` is
    overwritten on the next `apply()`/`rejected_frame()` call.
    """

    def __init__(
        self,
        resolution: float,
        height: int,
        width: int,
        config: FilterConfig | None = None,
        *,
        device: wp.context.Device | None = None,
    ):
        self.resolution = resolution
        self.height = height
        self.width = width
        self.shape = (height, width)
        self.config = config or FilterConfig()
        self.device = device if device is not None else wp.get_device()

        self.support_radius_cells = meters_to_cells(
            self.config.support_radius_m, resolution,
        )

        with wp.ScopedDevice(self.device):
            self._filtered = wp.zeros(self.shape, dtype=wp.float32)

    def apply(
        self,
        raw_elevation: np.ndarray | wp.array,
        cost_map: np.ndarray | wp.array,
    ) -> wp.array:
        """`raw_elevation` is the pre-inpaint heightmap (NaN in unmeasured cells)."""
        if raw_elevation.shape != self.shape or cost_map.shape != self.shape:
            raise ValueError("raw_elevation and cost_map must match mask shape")

        elev_wp = _as_gpu(raw_elevation, self.device)
        cost_wp = _as_gpu(cost_map, self.device)
        wp.launch(
            support_ratio_mask_kernel,
            dim=self.shape,
            inputs=[
                elev_wp,
                cost_wp,
                self.height,
                self.width,
                self.support_radius_cells,
                float(self.config.support_ratio),
            ],
            outputs=[self._filtered],
            device=self.device,
        )
        return self._filtered

    def rejected_frame(self) -> wp.array:
        """Return an all-NaN buffer (the same owned buffer as `apply`)."""
        self._filtered.fill_(float("nan"))
        return self._filtered


class OcclusionMask:
    """NaN-out cost cells occluded from the sensor and not actually measured.

    Preallocates a single owned output buffer — the returned `wp.array` is
    overwritten on the next `apply()` call.
    """

    def __init__(
        self,
        resolution: float,
        bounds: tuple[float, float, float, float],
        height: int,
        width: int,
        config: OcclusionConfig | None = None,
        *,
        device: wp.context.Device | None = None,
    ):
        self.resolution = resolution
        self.bounds = bounds
        self.height = height
        self.width = width
        self.shape = (height, width)
        self.config = config or OcclusionConfig()
        self.device = device if device is not None else wp.get_device()

        with wp.ScopedDevice(self.device):
            self._filtered = wp.zeros(self.shape, dtype=wp.float32)

    def apply(
        self,
        raw_elevation: np.ndarray | wp.array,
        elevation: np.ndarray | wp.array,
        cost_map: np.ndarray | wp.array,
    ) -> wp.array:
        """`raw_elevation` is the pre-inpaint heightmap (NaN in unmeasured cells);
        `elevation` is the inpainted heightmap used for the line-of-sight march."""
        if (raw_elevation.shape != self.shape or elevation.shape != self.shape
                or cost_map.shape != self.shape):
            raise ValueError("raw_elevation, elevation and cost_map must match mask shape")

        raw_wp = _as_gpu(raw_elevation, self.device)
        elev_wp = _as_gpu(elevation, self.device)
        cost_wp = _as_gpu(cost_map, self.device)
        xmin, _, ymin, _ = self.bounds
        sx, sy = self.config.sensor_xy
        wp.launch(
            occlusion_mask_kernel,
            dim=self.shape,
            inputs=[
                elev_wp,
                raw_wp,
                cost_wp,
                float(xmin),
                float(ymin),
                float(self.resolution),
                float(sx),
                float(sy),
                float(self.config.sensor_z),
                float(self.config.angle_eps_rad),
            ],
            outputs=[self._filtered],
            device=self.device,
        )
        return self._filtered
