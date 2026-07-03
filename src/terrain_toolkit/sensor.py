"""LidarSensorConfig: the shared description of a LiDAR's geometry + range model.

One source of truth per sensor, consumed by both the simulator (`sim/`) and the
perception filters (`dynamic/`) so the field of view, resolution, and range specs
are never duplicated and can't drift out of sync. Built from a datasheet (see
`sim.ouster.osdome_sensor_config`) or, on the real robot, from the sensor's
metadata.

Convention: `az_fov_deg` / `el_fov_deg` are in the range-image frame the filters
bin in — azimuth about +z (`atan2(y, x)`), elevation from the horizontal
(`asin(z / r)`) — NOT the sensor's internal polar coordinates. So they describe
the sensor's coverage *as mounted*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LidarSensorConfig:
    """Intrinsic geometry + range model of a LiDAR.

    Required fields have no defaults on purpose: the vertical FOV and resolution
    are sensor-specific and a wrong guess silently drops returns, so the caller
    must state them.
    """

    # Vertical field of view (deg), in the (az about +z, el from horizontal) frame.
    el_fov_deg: tuple[float, float]
    # Vertical resolution (number of beams / channels).
    channels: int
    # Azimuth samples per frame (horizontal resolution).
    columns: int

    # Azimuth field of view (deg). Full spin by default; narrow it for a
    # forward-facing sensor to spend resolution only where it sees.
    az_fov_deg: tuple[float, float] = (-180.0, 180.0)
    # Range window (m). A return outside it is not produced.
    min_range_m: float = 0.0
    max_range_m: float = math.inf
    # Range precision, 1σ(r) = base + quad·r², capped at max (m).
    range_noise_base_m: float = 0.0
    range_noise_quad_m: float = 0.0
    range_noise_max_m: float = math.inf

    def __post_init__(self) -> None:
        if self.el_fov_deg[0] >= self.el_fov_deg[1]:
            raise ValueError(f"el_fov_deg must be (min, max) with min < max; got {self.el_fov_deg}")
        if self.az_fov_deg[0] >= self.az_fov_deg[1]:
            raise ValueError(f"az_fov_deg must be (min, max) with min < max; got {self.az_fov_deg}")
        if self.channels <= 0 or self.columns <= 0:
            raise ValueError(
                f"channels and columns must be positive; got {self.channels}, {self.columns}"
            )
        if self.min_range_m < 0.0 or self.max_range_m <= self.min_range_m:
            raise ValueError(
                f"need 0 <= min_range_m < max_range_m; got {self.min_range_m}, {self.max_range_m}"
            )
        if min(self.range_noise_base_m, self.range_noise_quad_m, self.range_noise_max_m) < 0.0:
            raise ValueError("range noise terms must be non-negative")
