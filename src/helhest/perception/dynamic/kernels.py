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


@wp.kernel
def classify_recency_kernel(
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
    ages: wp.array(dtype=wp.int32),
    frame: wp.int32,
    max_unseen: wp.int32,
    keep: wp.array(dtype=wp.int32),
):
    """Carve + visibility-gated recency in one pass → the map keep mask (0 = drop).

    A point is dropped when the frontier saw PAST it (dynamic, seen-through), OR when it is
    OBSERVABLE this frame (a beam reached at least its range) yet has gone unconfirmed for
    more than `max_unseen` frames (stale dynamic residue the instantaneous carve missed).

    Points the frontier did NOT reach are kept: an unobserved bearing (`od` == sentinel,
    e.g. a side-mounted sensor's blind rear) and an occluded point (`od` < range − margin,
    a nearer surface stopped the beam) are both things we cannot currently disprove, so
    accumulated history survives there until it leaves the map or odometry breaks.
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
    if od >= _DEPTH_SENTINEL:  # no beam along this bearing → not observable → keep
        return
    margin = margin_m + r * margin_rel
    if od > r + margin:  # frontier saw past → seen-through → carve now
        keep[i] = 0
        return
    if od >= r - margin:  # beam reached the point (observable); stale → forget
        if (frame - ages[i]) > max_unseen:
            keep[i] = 0
    # else od < r − margin: occluded (beam stopped short) → not observable → keep


@wp.kernel
def classify_streak_kernel(
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
    streak_in: wp.array(dtype=wp.int32),
    persist: wp.int32,
    keep: wp.array(dtype=wp.int32),
    streak_out: wp.array(dtype=wp.int32),
):
    """Consecutive-free carve → keep mask + updated seen-through streak (one per map point).

    A point is dropped only after the scan has seen PAST it for `persist` CONSECUTIVE frames,
    not on a single frame's evidence. Per frame: seen-through (`od > r + margin`) grows the
    streak; a beam that returns at ~the point's range confirms it and resets the streak to 0;
    a bearing the scan did not observe (sentinel) or an occluded point (`od < r − margin`)
    leaves the streak unchanged (no evidence either way). This tolerates the ambiguous single
    no-return — grazing/dark/dropped beam — that the instantaneous carve wrongly deleted.
    """
    i = wp.tid()
    keep[i] = 1
    s = streak_in[i]
    d = rot @ (points[i] - origin)
    r = wp.length(d)
    if r < min_range:
        streak_out[i] = s
        return
    idx = _bin_index(d, az_bins, el_bins, az_min, az_span, el_min, el_max)
    if idx < 0:
        streak_out[i] = s
        return
    od = other_depth[idx]
    if od >= _DEPTH_SENTINEL:  # bearing not observed → no evidence → hold
        streak_out[i] = s
        return
    margin = margin_m + r * margin_rel
    if od > r + margin:  # seen through → one free vote
        s = s + 1
        streak_out[i] = s
        if s >= persist:
            keep[i] = 0
        return
    if od >= r - margin:  # a beam returned at ~this range → confirmed occupied → reset
        streak_out[i] = 0
        return
    streak_out[i] = s  # od < r − margin: occluded → cannot judge → hold
