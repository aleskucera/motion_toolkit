"""Measurement-support confidence masking on the heightmap.

The inpainter fills unmeasured cells with plausible elevations, but a cell whose
neighborhood holds few real returns is a guess, not data. `SupportRatioMask`
NaN-outs the cost of cells whose local measured fraction falls below a threshold,
so downstream consumers don't trust thinly-supported inpaint. See also
`OcclusionMask` — the other half of "which cells not to trust".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .._warp_utils import as_gpu
from ..grid_utils import meters_to_cells
from .kernels import support_ratio_mask_kernel


@dataclass
class SupportConfig:
    """Configuration for measurement-support confidence masking.

    A cell is kept only if at least `support_ratio` of the cells within
    `support_radius_m` were actually measured (non-NaN in the raw heightmap).
    """

    support_radius_m: float = 0.5
    support_ratio: float = 0.5


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
        config: SupportConfig | None = None,
        *,
        device: wp.context.Device | None = None,
    ):
        self.resolution = resolution
        self.height = height
        self.width = width
        self.shape = (height, width)
        self.config = config or SupportConfig()
        self.device = wp.get_device(device)

        self.support_radius_cells = meters_to_cells(
            self.config.support_radius_m,
            resolution,
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

        elev_wp = as_gpu(raw_elevation, self.device)
        cost_wp = as_gpu(cost_map, self.device)
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
