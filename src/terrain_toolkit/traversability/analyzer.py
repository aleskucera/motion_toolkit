from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..grid_utils import meters_to_cells
from .kernels import combine_costs_kernel
from .kernels import compute_roughness_kernel
from .kernels import compute_slope_sobel_kernel
from .kernels import compute_step_height_cost_kernel
from .kernels import morph_op_kernel


@dataclass
class TraversabilityConfig:
    """Configuration for `GeometricTraversabilityAnalyzer`."""

    # Normalization thresholds — cost saturates at 1 when these are reached.
    max_slope_deg: float = 60.0
    max_step_height_m: float = 0.55
    # Drops saturate sooner than bumps by default: a 0.3 m ledge already gives
    # cost 1, while a 0.3 m curb is ~0.55. Set equal to `max_step_height_m` to
    # recover the old unsigned behavior.
    max_drop_height_m: float = 0.3
    max_roughness_m: float = 0.2

    # Local window radii (in meters).
    step_window_radius_m: float = 0.15
    roughness_window_radius_m: float = 0.3

    # Weights for combining cost layers (normalized by their sum inside the kernel).
    slope_weight: float = 0.2
    step_weight: float = 0.2
    roughness_weight: float = 0.6


@dataclass
class TraversabilityCosts:
    """Outputs of `GeometricTraversabilityAnalyzer.compute`.

    Fields are `wp.array` buffers owned by the analyzer — they are overwritten on
    the next `compute()` call. Use `.to_numpy()` to retain values across frames.
    """

    slope: wp.array
    step: wp.array
    roughness: wp.array
    total: wp.array

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {
            "slope": self.slope.numpy().copy(),
            "step": self.step.numpy().copy(),
            "roughness": self.roughness.numpy().copy(),
            "total": self.total.numpy().copy(),
        }


class GeometricTraversabilityAnalyzer:
    """GPU pipeline that turns a filled heightmap into geometric cost layers.

    Preallocates Warp buffers so repeated `compute` calls on the same grid size
    reuse device memory. Accepts heightmap as numpy or `wp.array`.
    """

    def __init__(
        self,
        resolution: float,
        height: int,
        width: int,
        config: TraversabilityConfig | None = None,
        *,
        device: wp.context.Device | None = None,
        verbose: bool = False,
    ):
        self.resolution = resolution
        self.height = height
        self.width = width
        self.shape = (height, width)
        self.config = config or TraversabilityConfig()
        self.device = wp.get_device(device)
        self.verbose = verbose

        cfg = self.config
        self.max_slope_rad = math.radians(cfg.max_slope_deg)
        self.step_window_cells = meters_to_cells(cfg.step_window_radius_m, resolution)
        self.roughness_window_cells = meters_to_cells(cfg.roughness_window_radius_m, resolution)

        with wp.ScopedDevice(self.device):
            self._normals = wp.zeros(self.shape, dtype=wp.vec3)
            self._slope = wp.zeros(self.shape, dtype=wp.float32)
            self._step = wp.zeros(self.shape, dtype=wp.float32)
            self._rough = wp.zeros(self.shape, dtype=wp.float32)
            self._total = wp.zeros(self.shape, dtype=wp.float32)
            self._dilated = wp.zeros(self.shape, dtype=wp.float32)
            self._eroded = wp.zeros(self.shape, dtype=wp.float32)

        self._costs = TraversabilityCosts(
            slope=self._slope,
            step=self._step,
            roughness=self._rough,
            total=self._total,
        )

    def compute(self, heightmap: np.ndarray | wp.array) -> TraversabilityCosts:
        """Run slope + step + roughness + combined cost on a filled heightmap."""
        if heightmap.shape != self.shape:
            raise ValueError(f"heightmap shape {heightmap.shape} != analyzer shape {self.shape}")

        if isinstance(heightmap, wp.array):
            elev = heightmap
        else:
            elev = wp.array(
                np.ascontiguousarray(heightmap, dtype=np.float32),
                dtype=wp.float32,
                device=self.device,
            )

        cfg = self.config
        with wp.ScopedTimer("GeometricTraversability", active=self.verbose):
            wp.launch(
                compute_slope_sobel_kernel,
                dim=self.shape,
                inputs=[
                    elev,
                    float(self.resolution),
                    self.height,
                    self.width,
                    float(self.max_slope_rad),
                ],
                outputs=[self._normals, self._slope],
                device=self.device,
            )
            wp.launch(
                morph_op_kernel,
                dim=self.shape,
                inputs=[elev, self.height, self.width, self.step_window_cells, 1],
                outputs=[self._dilated],
                device=self.device,
            )
            wp.launch(
                morph_op_kernel,
                dim=self.shape,
                inputs=[elev, self.height, self.width, self.step_window_cells, 0],
                outputs=[self._eroded],
                device=self.device,
            )
            wp.launch(
                compute_step_height_cost_kernel,
                dim=self.shape,
                inputs=[
                    elev,
                    self._dilated,
                    self._eroded,
                    float(cfg.max_step_height_m),
                    float(cfg.max_drop_height_m),
                ],
                outputs=[self._step],
                device=self.device,
            )
            wp.launch(
                compute_roughness_kernel,
                dim=self.shape,
                inputs=[
                    elev,
                    self.height,
                    self.width,
                    self.roughness_window_cells,
                    float(cfg.max_roughness_m),
                ],
                outputs=[self._rough],
                device=self.device,
            )
            wp.launch(
                combine_costs_kernel,
                dim=self.shape,
                inputs=[
                    self._slope,
                    self._step,
                    self._rough,
                    float(cfg.slope_weight),
                    float(cfg.step_weight),
                    float(cfg.roughness_weight),
                ],
                outputs=[self._total],
                device=self.device,
            )

        return self._costs
