from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..sensor import LidarSensorConfig
from .kernels import _DEPTH_SENTINEL
from .kernels import classify_kernel
from .kernels import classify_recency_kernel
from .kernels import classify_streak_kernel
from .kernels import render_depth_kernel


def _pose_to_warp(
    sensor_origin: np.ndarray, sensor_rotation: np.ndarray | None
) -> tuple[wp.vec3, wp.mat33]:
    """Marshal the sensor pose into the scalar types the kernels take.

    `sensor_rotation` is the world→sensor rotation (3x3); `None` → identity
    (bin in the world frame). `sensor_origin` is the sensor position (3,).
    """
    R = np.eye(3) if sensor_rotation is None else np.asarray(sensor_rotation, dtype=np.float64)
    rot = wp.mat33(R.flatten().tolist())
    origin = wp.vec3(*np.asarray(sensor_origin, dtype=np.float64).tolist())
    return origin, rot


def _upload_points(points: np.ndarray) -> wp.array:
    """Upload a host (N, 3) cloud to a device `vec3` array."""
    return wp.array(np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3)


@dataclass(frozen=True)
class DynamicFilterConfig:
    """Tuning for `DynamicPointFilter` (see that class for how the filter works).

    Most fields describe the range image and should match the sensor — prefer
    `from_sensor()`, which pulls the FOV, resolution, and min-range from a
    `LidarSensorConfig` so they can't silently disagree with it. Only `margin_m`
    / `margin_rel` are genuinely filter-specific.

    Frozen: the filter reads it once at construction to size its buffers and
    precompute the grid, so mutating it afterwards would silently desync those.
    Reconfigure by building a new config + filter.
    """

    # Range-image angular resolution. Azimuth spans the full 360°; elevation spans
    # [el_min_deg, el_max_deg]. Default roughly a 128-beam sensor.
    az_bins: int = 1024
    el_bins: int = 128
    # Elevation FOV (deg) the bins span. A point outside is ignored (kept) — too
    # narrow a band silently under-carves, so the default is the full hemisphere.
    el_min_deg: float = -90.0
    el_max_deg: float = 90.0
    # A point is carved if the other cloud's surface along its bearing is farther
    # than the point by more than `margin_m + range * margin_rel`. The relative
    # term absorbs angular quantization on slanted surfaces + registration error.
    margin_m: float = 0.3
    margin_rel: float = 0.02
    # Ignore returns closer than this (sensor self-hits / degenerate directions).
    min_range_m: float = 0.5

    @classmethod
    def from_sensor(
        cls,
        sensor: LidarSensorConfig,
        *,
        margin_m: float = 0.3,
        margin_rel: float = 0.02,
        az_bins: int | None = None,
        el_bins: int | None = None,
    ) -> DynamicFilterConfig:
        """Build from a `LidarSensorConfig`: FOV, min-range, and default range-image
        resolution come from the sensor; only the margins are supplied here.

        `az_bins`/`el_bins` default to the sensor's `columns`/`channels`; override
        for a coarser or finer range image than the sensor's native resolution.
        """
        return cls(
            az_bins=az_bins if az_bins is not None else sensor.columns,
            el_bins=el_bins if el_bins is not None else sensor.channels,
            el_min_deg=sensor.el_fov_deg[0],
            el_max_deg=sensor.el_fov_deg[1],
            margin_m=margin_m,
            margin_rel=margin_rel,
            min_range_m=sensor.min_range_m,
        )


class DynamicPointFilter:
    """Map-frame visibility / ray-carving filter for removing moving objects.

    Compares the accumulated map against a new scan through spherical range images
    rendered from the sensor origin: a map point is *carved* when the scan reached
    farther along its bearing (the beam passed through it → free space). Feed the
    scan's per-beam free-space frontier (surface hit, or the max-range point on a
    miss) and this works even with no background behind a point — e.g. the top of
    a person against open sky. It removes things that MOVE (a beam has to see
    through where something was); a motionless object is a static pillar and kept.

    Two entry points:
      * `carve(map, scan, origin)` → the `map_keep` mask, device-native (device
        `wp.array`s in and out). The per-frame mapping call, kept on the GPU.
      * `filter(map, scan, origin)` → `(scan_keep, map_keep)` numpy masks — the
        host path; also drops incoming scan points in front of known geometry.
    """

    def __init__(
        self,
        config: DynamicFilterConfig | None = None,
        *,
        device: wp.context.Device | None = None,
    ):
        self.config = config or DynamicFilterConfig()
        self.device = wp.get_device(device)
        cfg = self.config
        self._n_bins = cfg.az_bins * cfg.el_bins

        # Angular-grid args shared by the render/classify kernels, fixed by the config:
        # (az_bins, el_bins, az_min, az_span, el_min, el_max, min_range). Azimuth wraps the
        # full circle; el_min/el_max in radians. Precomputed once — the config never changes.
        self._grid = [
            int(cfg.az_bins),
            int(cfg.el_bins),
            float(-math.pi),
            float(2.0 * math.pi),
            float(math.radians(cfg.el_min_deg)),
            float(math.radians(cfg.el_max_deg)),
            float(cfg.min_range_m),
        ]

        with wp.ScopedDevice(self.device):
            self._map_depth = wp.empty(self._n_bins, dtype=wp.float32)
            self._scan_depth = wp.empty(self._n_bins, dtype=wp.float32)

    @classmethod
    def from_sensor(
        cls,
        sensor: LidarSensorConfig,
        *,
        margin_m: float = 0.3,
        margin_rel: float = 0.02,
        az_bins: int | None = None,
        el_bins: int | None = None,
        device: wp.context.Device | None = None,
    ) -> DynamicPointFilter:
        config = DynamicFilterConfig.from_sensor(
            sensor,
            margin_m=margin_m,
            margin_rel=margin_rel,
            az_bins=az_bins,
            el_bins=el_bins,
        )
        return cls(config, device=device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render(
        self,
        points_wp: wp.array,
        n: int,
        origin: wp.vec3,
        rot: wp.mat33,
        depth: wp.array,
    ) -> None:
        """Render points into `depth` as a nearest-range image (sentinel-reset first)."""
        depth.fill_(_DEPTH_SENTINEL)
        if n > 0:
            wp.launch(
                render_depth_kernel,
                dim=n,
                inputs=[points_wp, origin, rot, *self._grid],
                outputs=[depth],
            )

    def _classify(
        self,
        points_wp: wp.array,
        n: int,
        origin: wp.vec3,
        rot: wp.mat33,
        other_depth: wp.array,
    ) -> wp.array:
        """Per-point keep mask vs `other_depth` (0 = in front of it → remove)."""
        keep = wp.empty(n, dtype=wp.int32)
        wp.launch(
            classify_kernel,
            dim=n,
            inputs=[
                points_wp,
                origin,
                rot,
                *self._grid,
                float(self.config.margin_m),
                float(self.config.margin_rel),
                other_depth,
            ],
            outputs=[keep],
        )
        return keep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def carve(
        self,
        map_points: wp.array,
        scan_points: wp.array,
        sensor_origin: np.ndarray,
        *,
        sensor_rotation: np.ndarray | None = None,
    ) -> wp.array:
        """Carve-only, device-native: return the `map_keep` mask (int32, 0 = carve).

        The per-frame mapping call: given the map and the scan's free-space
        frontier as device `wp.array`s, return an on-device keep-mask marking map
        points the scan saw through (free space). Half the work of `filter()` — one
        range image, one classify — and the returned mask is a live device array,
        stream-ordered with the caller's downstream kernels (no host sync). For a
        host/numpy result, use `filter()`.
        """
        n_map = len(map_points)
        if n_map == 0:
            return wp.zeros(0, dtype=wp.int32)
        origin, rot = _pose_to_warp(sensor_origin, sensor_rotation)
        with wp.ScopedDevice(self.device):
            self._render(scan_points, len(scan_points), origin, rot, self._scan_depth)
            return self._classify(map_points, n_map, origin, rot, self._scan_depth)

    def carve_recency(
        self,
        map_points: wp.array,
        scan_points: wp.array,
        sensor_origin: np.ndarray,
        ages: wp.array,
        frame: int,
        max_unseen: int,
        *,
        sensor_rotation: np.ndarray | None = None,
    ) -> wp.array:
        """Carve + visibility-gated recency, device-native → the `map_keep` mask (0 = drop).

        Like `carve()`, but also drops points that ARE observable this frame (a beam reached
        their range) yet have not been reconfirmed for more than `max_unseen` frames — the
        stale dynamic residue the instantaneous carve leaves behind. Points the frontier never
        reached (blind rear, occluded) are kept, so history survives where the sensor can't
        currently look. `ages` (int32, len == map) is each point's last-seen frame, maintained
        by `DeviceMapAccumulator`. One frontier render, one classify — same cost as `carve()`.
        """
        n_map = len(map_points)
        if n_map == 0:
            return wp.zeros(0, dtype=wp.int32)
        origin, rot = _pose_to_warp(sensor_origin, sensor_rotation)
        with wp.ScopedDevice(self.device):
            self._render(scan_points, len(scan_points), origin, rot, self._scan_depth)
            keep = wp.empty(n_map, dtype=wp.int32)
            wp.launch(
                classify_recency_kernel,
                dim=n_map,
                inputs=[
                    map_points,
                    origin,
                    rot,
                    *self._grid,
                    float(self.config.margin_m),
                    float(self.config.margin_rel),
                    self._scan_depth,
                    ages,
                    int(frame),
                    int(max_unseen),
                ],
                outputs=[keep],
            )
            return keep

    def carve_streak(
        self,
        map_points: wp.array,
        scan_points: wp.array,
        sensor_origin: np.ndarray,
        streak_in: wp.array,
        persist: int,
        gap_persist: int = 0,
        *,
        sensor_rotation: np.ndarray | None = None,
    ) -> tuple[wp.array, wp.array]:
        """Consecutive-free carve, device-native → `(map_keep, streak_out)`.

        Like `carve()`, but a point is dropped only after the scan has seen PAST it for `persist`
        CONSECUTIVE frames — not on one frame's ambiguous evidence (a lone no-return from a
        grazing/dark/dropped beam). `streak_in` (int32, len == map) is each point's current
        seen-through streak (maintained by `DeviceMapAccumulator`; all-zero on the first frame);
        `streak_out` is the updated streak to thread back through the accumulator. `persist <= 1`
        reduces to the instantaneous `carve()`.

        `gap_persist` > 0 also ages out BETWEEN-BEAM points: a bearing with no beam whose neighbours
        ARE scanned (a stale fragment the discrete beams can never re-hit) is dropped after that many
        frames. Points in a fully unscanned region (outside the vertical FOV) are still held.
        """
        n_map = len(map_points)
        if n_map == 0:
            return wp.zeros(0, dtype=wp.int32), wp.zeros(0, dtype=wp.int32)
        origin, rot = _pose_to_warp(sensor_origin, sensor_rotation)
        with wp.ScopedDevice(self.device):
            self._render(scan_points, len(scan_points), origin, rot, self._scan_depth)
            keep = wp.empty(n_map, dtype=wp.int32)
            streak_out = wp.empty(n_map, dtype=wp.int32)
            wp.launch(
                classify_streak_kernel,
                dim=n_map,
                inputs=[
                    map_points,
                    origin,
                    rot,
                    *self._grid,
                    float(self.config.margin_m),
                    float(self.config.margin_rel),
                    self._scan_depth,
                    streak_in,
                    int(persist),
                    int(gap_persist),
                ],
                outputs=[keep, streak_out],
            )
            return keep, streak_out

    def filter(
        self,
        map_points: np.ndarray,
        scan_points: np.ndarray,
        sensor_origin: np.ndarray,
        *,
        sensor_rotation: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(scan_keep, map_keep)` boolean masks (numpy).

        `sensor_rotation` is the world→sensor rotation (3x3); it aligns the range
        image with the sensor's beams so the elevation band matches its FOV
        regardless of robot tilt. Identity if omitted. For carve-only device use,
        see `carve()`.
        """
        n_scan = len(scan_points)
        n_map = len(map_points)
        # Nothing to compare against → keep everything.
        if n_map == 0 or n_scan == 0:
            return np.ones(n_scan, dtype=bool), np.ones(n_map, dtype=bool)

        origin, rot = _pose_to_warp(sensor_origin, sensor_rotation)
        with wp.ScopedDevice(self.device):
            map_wp = _upload_points(map_points)
            scan_wp = _upload_points(scan_points)
            self._render(map_wp, n_map, origin, rot, self._map_depth)
            self._render(scan_wp, n_scan, origin, rot, self._scan_depth)
            scan_keep = self._classify(scan_wp, n_scan, origin, rot, self._map_depth)
            map_keep = self._classify(map_wp, n_map, origin, rot, self._scan_depth)
            wp.synchronize()
            return scan_keep.numpy().astype(bool), map_keep.numpy().astype(bool)
