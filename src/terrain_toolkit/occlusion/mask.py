"""Line-of-sight occlusion masking on the 2.5D heightmap.

A sensor-visibility stage, distinct from traversability geometry: inpainting
fills unmeasured cells with plausible elevations, but cells hidden behind an
obstacle were never observed. `OcclusionMask` NaN-outs the cost of exactly those
cells — unmeasured *and* in the sensor's line-of-sight shadow — so a planner
cannot route through inpainted free space behind walls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .._warp_utils import as_gpu
from .kernels import occlusion_mask_kernel


@dataclass
class OcclusionConfig:
    """Configuration for occlusion (line-of-sight) cost masking.

    Cells hidden from the sensor by an obstacle and not actually measured are
    NaN-ed out, so inpainted free-space behind walls is not treated as
    traversable. The sensor xy/z are in the grid (gravity-aligned robot) frame.
    """

    sensor_xy: tuple[float, float] = (0.0, 0.0)  # sensor position in grid frame (m)
    sensor_z: float = 0.5  # sensor height above the grid origin (m)
    angle_eps_rad: float = 0.01  # horizon margin; guards flat-ground noise


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
        self.device = wp.get_device(device)

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
        if (
            raw_elevation.shape != self.shape
            or elevation.shape != self.shape
            or cost_map.shape != self.shape
        ):
            raise ValueError("raw_elevation, elevation and cost_map must match mask shape")

        raw_wp = as_gpu(raw_elevation, self.device)
        elev_wp = as_gpu(elevation, self.device)
        cost_wp = as_gpu(cost_map, self.device)
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
