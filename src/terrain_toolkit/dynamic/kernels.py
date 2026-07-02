from __future__ import annotations

import warp as wp

wp.init()

# Fill value for a range-image bin. `render_depth_kernel` only ever lowers a bin
# via atomic-min, so a bin no point mapped to keeps this exact value — that's how
# `classify_kernel` recognizes an unobserved bearing. `filter.py` fills with it too.
_DEPTH_SENTINEL = 1.0e18


@wp.func
def _bin_index(
    d: wp.vec3,
    az_bins: int,
    el_bins: int,
    az_min: wp.float32,
    az_span: wp.float32,
    el_min: wp.float32,
    el_max: wp.float32,
) -> int:
    """Angular bin of a direction vector, or -1 if outside the elevation band.

    `d` is the vector from the sensor origin to the point, already rotated into
    the sensor frame. Azimuth wraps the full circle; elevation is clamped to the
    sensor's vertical FOV so out-of-band points don't fold into edge rows.
    """
    r = wp.length(d)
    if r <= 0.0:
        return -1
    az = wp.atan2(d[1], d[0])  # [-pi, pi]
    el = wp.asin(wp.clamp(d[2] / r, -1.0, 1.0))
    if el < el_min or el > el_max:
        return -1
    col = int((az - az_min) / az_span * float(az_bins))
    row = int((el - el_min) / (el_max - el_min) * float(el_bins))
    if col < 0:
        col = 0
    if col >= az_bins:
        col = az_bins - 1
    if row < 0:
        row = 0
    if row >= el_bins:
        row = el_bins - 1
    return row * az_bins + col


@wp.kernel
def render_depth_kernel(
    points: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    rot: wp.mat33,  # sensor_R_world: rotate world directions into the sensor frame
    az_bins: int,
    el_bins: int,
    az_min: wp.float32,
    az_span: wp.float32,
    el_min: wp.float32,
    el_max: wp.float32,
    min_range: wp.float32,
    depth: wp.array(dtype=wp.float32),  # pre-filled with a large sentinel
):
    """Nearest-surface range per angular bin, as seen from `origin`.

    A spherical range image: each point contributes its range to its (az, el)
    bin via atomic-min, so `depth[bin]` ends up the closest surface along that
    bearing. Points closer than `min_range` (self-hits) are ignored.
    """
    i = wp.tid()
    d = rot @ (points[i] - origin)
    r = wp.length(d)
    if r < min_range:
        return
    idx = _bin_index(d, az_bins, el_bins, az_min, az_span, el_min, el_max)
    if idx < 0:
        return
    wp.atomic_min(depth, idx, r)


@wp.kernel
def classify_kernel(
    points: wp.array(dtype=wp.vec3),
    origin: wp.vec3,
    rot: wp.mat33,
    az_bins: int,
    el_bins: int,
    az_min: wp.float32,
    az_span: wp.float32,
    el_min: wp.float32,
    el_max: wp.float32,
    min_range: wp.float32,
    margin_m: wp.float32,
    margin_rel: wp.float32,
    other_depth: wp.array(dtype=wp.float32),
    keep: wp.array(dtype=wp.int32),
):
    """Flag points that sit in front of the *other* observation's surface.

    For each point, look up the nearest surface the other cloud saw along the
    same bearing (`other_depth[bin]`). If that surface is farther than this point
    by more than the margin, this point occupies space the other cloud saw
    through — it is inconsistent (a dynamic intruder, or a stale ghost) and gets
    `keep = 0`. Bins the other cloud never observed leave the point kept.

    The margin grows with range (`margin_m + r * margin_rel`) to absorb angular
    quantization on slanted surfaces plus pose/registration error.
    """
    i = wp.tid()
    keep[i] = 1
    d = rot @ (points[i] - origin)
    r = wp.length(d)
    if r < min_range:
        return
    idx = _bin_index(d, az_bins, el_bins, az_min, az_span, el_min, el_max)
    if idx < 0:
        return
    od = other_depth[idx]
    if od >= _DEPTH_SENTINEL:  # bin unobserved by the other cloud → no evidence
        return
    margin = margin_m + r * margin_rel
    if od > r + margin:
        keep[i] = 0
