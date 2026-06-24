from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import warp as wp

from .kernels import stamp_footprint_kernel

FootprintMode = Literal["overwrite", "fill"]


@dataclass
class FootprintConfig:
    """Configuration for forcing a flat ground patch under the robot.

    The patch is a fixed axis-aligned rectangle in the grid (robot-centered),
    described by its half-extents and optional center offset. The *height* of
    the patch is supplied per-frame as a plane (see `FlatGroundFootprint.apply`)
    because, although the robot footprint is flat in the robot body frame, it
    tilts with the robot's roll/pitch when expressed in the gravity-aligned grid
    frame. `ground_z` is only the level fallback used when no per-frame plane is
    given.
    """

    half_x: float  # footprint half-extent along grid x (m)
    half_y: float  # footprint half-extent along grid y (m)
    center: tuple[float, float] = (0.0, 0.0)  # footprint center in grid frame (m)
    ground_z: float = 0.0  # level-fallback height when no per-frame plane is passed
    mode: FootprintMode = "overwrite"

    def __post_init__(self) -> None:
        if self.half_x <= 0.0 or self.half_y <= 0.0:
            raise ValueError("footprint half_x and half_y must be positive")
        if self.mode not in ("overwrite", "fill"):
            raise ValueError(f"footprint mode must be 'overwrite' or 'fill'; got {self.mode!r}")


class FlatGroundFootprint:
    """Stamp a flat ground plane over a fixed rectangle under the robot.

    The grid-cell rectangle is computed once at construction (the footprint has a
    fixed transform relative to the robot, and the grid is robot-centered). Each
    `apply()` launches a single kernel over that block, in-place on the primary
    reduction. Intended to run before inpaint so the patch is treated as known.
    """

    def __init__(
        self,
        resolution: float,
        bounds: tuple[float, float, float, float],
        height: int,
        width: int,
        config: FootprintConfig,
        *,
        device: wp.context.Device | None = None,
    ):
        xmin, _, ymin, _ = bounds
        cx, cy = config.center
        # x maps to column (j), y maps to row (i), matching the rasterizer.
        j0 = int(math.floor((cx - config.half_x - xmin) / resolution))
        j1 = int(math.ceil((cx + config.half_x - xmin) / resolution))
        i0 = int(math.floor((cy - config.half_y - ymin) / resolution))
        i1 = int(math.ceil((cy + config.half_y - ymin) / resolution))
        self.i0 = max(0, i0)
        self.i1 = min(height, i1)
        self.j0 = max(0, j0)
        self.j1 = min(width, j1)

        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.resolution = float(resolution)
        self.config = config
        self.fill_only = 1 if config.mode == "fill" else 0
        self.device = device if device is not None else wp.get_device()

    @property
    def is_empty(self) -> bool:
        """True when the footprint rectangle falls entirely outside the grid."""
        return self.i1 <= self.i0 or self.j1 <= self.j0

    def apply(
        self,
        primary: wp.array,
        plane: tuple[float, float, float] | None = None,
    ) -> None:
        """Stamp `z = a*x + b*y + c` into the footprint cells of `primary`.

        `plane` is `(a, b, c)` in grid-frame coordinates. When `None`, falls back
        to a level plane at `config.ground_z` (a/b = 0).
        """
        if self.is_empty:
            return
        if plane is None:
            a, b, c = 0.0, 0.0, self.config.ground_z
        else:
            a, b, c = plane
        wp.launch(
            stamp_footprint_kernel,
            dim=(self.i1 - self.i0, self.j1 - self.j0),
            inputs=[
                primary,
                int(self.i0),
                int(self.j0),
                self.xmin,
                self.ymin,
                self.resolution,
                float(a),
                float(b),
                float(c),
                self.fill_only,
            ],
            device=self.device,
        )
