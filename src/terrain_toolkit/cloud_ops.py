"""Device-resident point-cloud ops (Warp): transform and axis-aligned crop.

Keeps clouds on the GPU between stages — no host round trip. `transform_points`
applies a host 4x4 pose on device; `BoxCrop` crops to an xy box (with optional
recenter + z cutoff) and compacts, mirroring the host `pose_math` helpers but
without ever leaving the device. Only point counts cross back to the host.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.init()

_Z_UNBOUNDED = 1.0e30  # sentinel "no z cutoff" (float32-safe)


@wp.kernel
def _transform_kernel(
    src: wp.array(dtype=wp.vec3),
    n: wp.int32,
    m: wp.mat44,
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= n:
        return
    p = src[i]
    out[i] = wp.vec3(
        m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2] * p[2] + m[0, 3],
        m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2] * p[2] + m[1, 3],
        m[2, 0] * p[0] + m[2, 1] * p[1] + m[2, 2] * p[2] + m[2, 3],
    )


def transform_points(points: wp.array, n: int, pose: np.ndarray) -> wp.array:
    """Apply the host 4x4 `pose` to the first `n` device points; return a new device array."""
    m = wp.mat44(*[float(v) for v in np.asarray(pose, dtype=np.float32).reshape(-1)])
    out = wp.empty(n, dtype=wp.vec3, device=points.device)
    wp.launch(_transform_kernel, dim=n, inputs=[points, n, m], outputs=[out], device=points.device)
    return out


@wp.kernel
def _box_crop_kernel(
    points: wp.array(dtype=wp.vec3),
    n: wp.int32,
    cx: wp.float32,
    cy: wp.float32,
    half_x: wp.float32,
    half_y: wp.float32,
    sx: wp.float32,  # recenter shift subtracted from each kept point (0 = no shift)
    sy: wp.float32,
    sz: wp.float32,
    z_max: wp.float32,  # drop points whose (recentered) z exceeds this
    out: wp.array(dtype=wp.vec3),
    counter: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= n:
        return
    p = points[i]
    if wp.abs(p[0] - cx) > half_x or wp.abs(p[1] - cy) > half_y:  # xy box, inclusive
        return
    q = wp.vec3(p[0] - sx, p[1] - sy, p[2] - sz)
    if q[2] > z_max:
        return
    idx = wp.atomic_add(counter, 0, 1)  # append order is nondeterministic; callers don't rely on it
    out[idx] = q


class BoxCrop:
    """Crop a device cloud to an xy box (optional recenter + z cutoff), compacting on device.

    Preallocated output buffer sized for up to `max_points` input points, reused
    every call — the returned array is valid only until the next `crop()`.
    """

    def __init__(self, max_points: int, device: wp.context.Device | None = None) -> None:
        self.device = wp.get_device(device)
        self.max_points = int(max_points)
        with wp.ScopedDevice(self.device):
            self._out = wp.empty(self.max_points, dtype=wp.vec3)
            self._counter = wp.zeros(1, dtype=wp.int32)

    def crop(
        self,
        points: wp.array,
        n: int,
        center: tuple[float, float],
        half: float | tuple[float, float],
        *,
        recenter: np.ndarray | None = None,
        z_max: float | None = None,
    ) -> tuple[wp.array, int]:
        """Keep the first `n` points inside the xy box; return `(out_device, count)`.

        `center` is the box center `(cx, cy)`; `half` a single or `(half_x, half_y)`
        half-extent. `recenter` (a 3-vector, e.g. the robot translation) is
        subtracted from every kept point; `z_max` then drops points whose recentered
        z exceeds it. z is otherwise unbounded.
        """
        if n > self.max_points:
            raise ValueError(f"n={n} exceeds max_points={self.max_points}")
        if n == 0:
            return self._out, 0
        cx, cy = float(center[0]), float(center[1])
        half_x, half_y = (half, half) if np.isscalar(half) else (float(half[0]), float(half[1]))
        sx, sy, sz = (
            (0.0, 0.0, 0.0)
            if recenter is None
            else (float(recenter[0]), float(recenter[1]), float(recenter[2]))
        )
        zm = _Z_UNBOUNDED if z_max is None else float(z_max)
        with wp.ScopedDevice(self.device):
            self._counter.zero_()
            wp.launch(
                _box_crop_kernel,
                dim=n,
                inputs=[points, n, cx, cy, float(half_x), float(half_y), sx, sy, sz, zm],
                outputs=[self._out, self._counter],
            )
            wp.synchronize()
            n_out = int(self._counter.numpy()[0])
        return self._out, n_out
