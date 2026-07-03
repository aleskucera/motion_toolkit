"""GPU ray-cast LiDAR against analytic primitives (Warp).

A simulation utility (not part of the perception pipeline): one thread per beam
casts against a bounded ground plane plus a set of axis-aligned box obstacles,
keeping the nearest hit. The sensor has a movable pose (position + yaw), so it
can drive around; range noise and beam dropout are applied in-kernel. Used by
the demos to feed realistic scans into the terrain pipeline / dynamic filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..sensor import LidarSensorConfig

wp.init()

_MISS = wp.constant(1.0e30)


@wp.func
def _hit_bounded_plane(
    origin: wp.vec3,
    direction: wp.vec3,
    axis: int,
    plane_value: float,
    u_lo: float,
    u_hi: float,  # window bounds on the first in-plane axis
    v_lo: float,
    v_hi: float,  # window bounds on the second in-plane axis
) -> float:
    """Distance to an axis-aligned plane, valid only inside a 2D window (else miss)."""
    dir_along_axis = direction[axis]
    if wp.abs(dir_along_axis) < 1.0e-9:
        return _MISS
    dist = (plane_value - origin[axis]) / dir_along_axis
    if dist <= 1.0e-3:
        return _MISS
    point = origin + dist * direction
    # The two axes spanning the plane (the ones that are NOT `axis`).
    axis_u = (axis + 1) % 3
    axis_v = (axis + 2) % 3
    if point[axis_u] < u_lo or point[axis_u] > u_hi:
        return _MISS
    if point[axis_v] < v_lo or point[axis_v] > v_hi:
        return _MISS
    return dist


@wp.func
def _hit_aabb(origin: wp.vec3, direction: wp.vec3, box_lo: wp.vec3, box_hi: wp.vec3) -> float:
    """Slab-method nearest entry distance for a ray/AABB (miss → _MISS)."""
    t_near = float(-1.0e30)
    t_far = float(1.0e30)
    for axis in range(3):
        if wp.abs(direction[axis]) < 1.0e-12:
            # Ray parallel to this slab: miss if the origin is outside it.
            if origin[axis] < box_lo[axis] or origin[axis] > box_hi[axis]:
                return _MISS
        else:
            inv_dir = 1.0 / direction[axis]
            t_lo = (box_lo[axis] - origin[axis]) * inv_dir
            t_hi = (box_hi[axis] - origin[axis]) * inv_dir
            t_near = wp.max(t_near, wp.min(t_lo, t_hi))
            t_far = wp.min(t_far, wp.max(t_lo, t_hi))
    if t_far < wp.max(t_near, 0.0) or t_far <= 1.0e-3:
        return _MISS
    if t_near > 1.0e-3:
        return t_near
    return t_far


@wp.kernel
def raycast_kernel(
    origin: wp.vec3,
    yaw: wp.float32,
    dirs: wp.array(dtype=wp.vec3),  # beam directions in the sensor's local frame
    ground_z: wp.float32,
    gx_lo: wp.float32,
    gx_hi: wp.float32,
    gy_lo: wp.float32,
    gy_hi: wp.float32,
    boxes_lo: wp.array(dtype=wp.vec3),
    boxes_hi: wp.array(dtype=wp.vec3),
    n_boxes: wp.int32,
    noise_base: wp.float32,
    noise_quad: wp.float32,
    noise_max: wp.float32,
    dropout: wp.float32,
    min_range: wp.float32,
    max_range: wp.float32,
    seed: wp.int32,
    out_points: wp.array(dtype=wp.vec3),
    out_valid: wp.array(dtype=wp.int32),
    out_free: wp.array(dtype=wp.vec3),
):
    """Nearest primitive hit per beam, with dropout + range-dependent noise.

    Also writes `out_free`: the free-space frontier along each beam — the surface
    hit, or the max-range point if the beam sees nothing. Rendering these into a
    range image gives free-space evidence (a no-return beam proves empty space out
    to max range), which is what lets ray-carving remove points with no background
    behind them (e.g. the top of a person against open sky).
    """
    beam = wp.tid()
    # Rotate the local beam into the world by the sensor yaw (about +z).
    local_dir = dirs[beam]
    cos_yaw = wp.cos(yaw)
    sin_yaw = wp.sin(yaw)
    direction = wp.vec3(
        cos_yaw * local_dir[0] - sin_yaw * local_dir[1],
        sin_yaw * local_dir[0] + cos_yaw * local_dir[1],
        local_dir[2],
    )

    hit_dist = _MISS
    ground_dist = _hit_bounded_plane(origin, direction, 2, ground_z, gx_lo, gx_hi, gy_lo, gy_hi)
    if ground_dist < hit_dist:
        hit_dist = ground_dist
    for box in range(n_boxes):
        box_dist = _hit_aabb(origin, direction, boxes_lo[box], boxes_hi[box])
        if box_dist < hit_dist:
            hit_dist = box_dist

    # Free-space frontier: the surface, or max_range when the beam sees nothing.
    if hit_dist >= _MISS or hit_dist > max_range:
        out_free[beam] = origin + max_range * direction
    else:
        out_free[beam] = origin + hit_dist * direction

    # No hit, or the surface is outside the sensor's [min, max] range window.
    if hit_dist >= _MISS or hit_dist < min_range or hit_dist > max_range:
        out_valid[beam] = 0
        return

    # Draw dropout first, then noise, so the stream is deterministic per beam.
    rng = wp.rand_init(seed, beam)
    if wp.randf(rng) < dropout:
        out_valid[beam] = 0
        return
    # Range precision degrades with distance: sigma(r) = base + quad*r², capped.
    noise_sigma = noise_base + noise_quad * hit_dist * hit_dist
    if noise_sigma > noise_max:
        noise_sigma = noise_max
    hit_dist = hit_dist + noise_sigma * wp.randn(rng)  # noise is along the beam
    out_points[beam] = origin + hit_dist * direction
    out_valid[beam] = 1


@dataclass
class GroundSpec:
    z: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]


class PrimitiveLidar:
    """Ray-cast LiDAR: a movable sensor over a ground plane + box obstacles.

    Beam directions (in the sensor's local frame, looking down +x) and the
    ground are fixed at construction. Each `scan()` places the sensor at a pose
    (position + yaw), casts against the current set of box obstacles, and returns
    the surviving hit points as an (N, 3) numpy array. Range noise and dropout
    are applied on-device.
    """

    def __init__(
        self,
        directions: np.ndarray,
        *,
        ground: GroundSpec,
        noise_std: float = 0.0,
        range_noise_quad: float = 0.0,
        range_noise_max: float | None = None,
        dropout: float = 0.0,
        min_range: float = 0.0,
        max_range: float | None = None,
        device: wp.context.Device | None = None,
    ):
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(f"directions must be (B, 3); got {directions.shape}")
        self.device = wp.get_device(device)
        self.ground = ground
        # Range noise 1σ(r) = noise_std + range_noise_quad·r², capped at range_noise_max.
        self.noise_std = float(noise_std)
        self.range_noise_quad = float(range_noise_quad)
        self.range_noise_max = 1.0e30 if range_noise_max is None else float(range_noise_max)
        self.dropout = float(dropout)
        self.min_range = float(min_range)
        # None → effectively unlimited (the miss sentinel already caps real hits).
        self.max_range = 1.0e30 if max_range is None else float(max_range)

        n = len(directions)
        with wp.ScopedDevice(self.device):
            self._dirs = wp.array(np.ascontiguousarray(directions, dtype=np.float32), dtype=wp.vec3)
            self._out_pts = wp.empty(n, dtype=wp.vec3)
            self._out_valid = wp.empty(n, dtype=wp.int32)
            self._out_free = wp.empty(n, dtype=wp.vec3)
        self._n = n

    @classmethod
    def from_sensor(
        cls,
        sensor: LidarSensorConfig,
        directions: np.ndarray,
        *,
        ground: GroundSpec,
        dropout: float = 0.0,
        device: wp.context.Device | None = None,
    ) -> PrimitiveLidar:
        """Build a lidar whose range window and noise model come from `sensor`.

        `directions` (the beam layout) and `ground` (the sim scene) aren't sensor
        properties, so they're passed here; `dropout` isn't a datasheet figure.
        """
        return cls(
            directions,
            ground=ground,
            noise_std=sensor.range_noise_base_m,
            range_noise_quad=sensor.range_noise_quad_m,
            range_noise_max=sensor.range_noise_max_m,
            dropout=dropout,
            min_range=sensor.min_range_m,
            max_range=sensor.max_range_m,
            device=device,
        )

    def scan(
        self,
        origin: np.ndarray,
        yaw: float,
        boxes_lo: np.ndarray,
        boxes_hi: np.ndarray,
        seed: int,
        with_frontier: bool = False,
        return_device: bool = False,
    ) -> (
        np.ndarray  # default: kept hits (N, 3)
        | tuple[np.ndarray, np.ndarray]  # with_frontier: (hits, frontier)
        | tuple[wp.array, wp.array, wp.array]  # return_device: (points, valid, free)
    ):
        """Cast from `origin` with heading `yaw` against boxes `[boxes_lo, boxes_hi]`.

        `boxes_lo`/`boxes_hi` are (M, 3) AABB corners (M may be 0). Returns kept
        hit points (N, 3). With `with_frontier=True`, also returns the per-beam
        free-space frontier points (B, 3) (surface hit or max-range on a miss),
        for ray-carving free space.

        With `return_device=True`, skips the host copy and returns the live device
        buffers `(points, valid, free)` as `wp.array`s (all length B; `points`/`free`
        are indexed by `valid`) — valid only until the next `scan()`. This keeps the
        per-frame pipeline on-device (no host round trip).
        """
        boxes_lo = np.ascontiguousarray(boxes_lo, dtype=np.float32).reshape(-1, 3)
        boxes_hi = np.ascontiguousarray(boxes_hi, dtype=np.float32).reshape(-1, 3)
        n_boxes = len(boxes_lo)
        ground = self.ground
        with wp.ScopedDevice(self.device):
            # Warp arrays must be non-empty; pad to length 1 when there are no boxes.
            boxes_lo_wp = wp.array(
                boxes_lo if n_boxes else np.zeros((1, 3), np.float32), dtype=wp.vec3
            )
            boxes_hi_wp = wp.array(
                boxes_hi if n_boxes else np.zeros((1, 3), np.float32), dtype=wp.vec3
            )
            wp.launch(
                raycast_kernel,
                dim=self._n,
                inputs=[
                    wp.vec3(*np.asarray(origin, dtype=np.float64).tolist()),
                    float(yaw),
                    self._dirs,
                    float(ground.z),
                    float(ground.x_range[0]),
                    float(ground.x_range[1]),
                    float(ground.y_range[0]),
                    float(ground.y_range[1]),
                    boxes_lo_wp,
                    boxes_hi_wp,
                    int(n_boxes),
                    self.noise_std,
                    self.range_noise_quad,
                    self.range_noise_max,
                    self.dropout,
                    self.min_range,
                    self.max_range,
                    int(seed),
                ],
                outputs=[self._out_pts, self._out_valid, self._out_free],
            )
            if return_device:
                # Live device buffers, no host copy (valid until the next scan()).
                return self._out_pts, self._out_valid, self._out_free
            wp.synchronize()
            valid = self._out_valid.numpy().astype(bool)
            hits = self._out_pts.numpy()[valid]
            if with_frontier:
                return hits, self._out_free.numpy().copy()
            return hits
