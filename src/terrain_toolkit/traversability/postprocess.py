from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .._warp_utils import as_gpu
from .kernels import count_obstacles_kernel
from .kernels import inflate_obstacles_kernel


@dataclass
class FilterConfig:
    """Configuration for the traversability post-processing stages.

    Single bag for the filter chain: `ObstacleInflator` and `TemporalGate` read
    it directly, and the pipeline forwards the support fields to a
    `confidence.SupportConfig` for the support-ratio mask.
    """

    # Support ratio (forwarded to confidence.SupportConfig): cells whose
    # neighborhood has fewer than this fraction of measured cells are set to NaN.
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
        self.device = wp.get_device(device)

        sigma_cells = self.config.inflation_sigma_m / resolution
        self.radius_cells = int(math.ceil(3.0 * sigma_cells))
        self.inv_two_sigma_sq = 1.0 / (2.0 * sigma_cells * sigma_cells) if sigma_cells > 0 else 0.0

        with wp.ScopedDevice(self.device):
            self._inflated = wp.zeros(self.shape, dtype=wp.float32)

    def apply(self, cost_map: np.ndarray | wp.array) -> wp.array:
        if cost_map.shape != self.shape:
            raise ValueError(f"cost_map shape {cost_map.shape} != inflator shape {self.shape}")
        cost_wp = as_gpu(cost_map, self.device)
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
        self.device = wp.get_device(device)

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
