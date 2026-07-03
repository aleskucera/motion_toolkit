from __future__ import annotations

import warp as wp

wp.init()


@wp.kernel
def mean_dist_in_radius_kernel(
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    search_radius: wp.float32,
    min_neighbors: wp.int32,
    sensor_origin: wp.vec3,
    range_eps: wp.float32,
    mean_dist: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
    out_sum: wp.array(dtype=wp.float32),
    out_sum_sq: wp.array(dtype=wp.float32),
    out_count: wp.array(dtype=wp.int32),
):
    """Per-point mean neighbor distance within `search_radius`, range-normalized,
    with fused reduction into (sum, sum_sq, count) over the valid subset.

    Output is `mean(||q - p||) / max(||p - origin||, range_eps)` for all q inside
    the radius — scale-invariant against lidar's linearly-increasing point spacing
    with range. `valid[i] = 0` when fewer than `min_neighbors` lie inside the
    radius. Valid points additionally contribute to the global (sum, sum_sq,
    count) via atomics, saving an extra reduction pass.
    """
    i = wp.tid()
    p = points[i]

    s = float(0.0)
    count = int(0)
    r2 = search_radius * search_radius

    neighbors = wp.hash_grid_query(grid, p, search_radius)
    for index in neighbors:
        if index == i:
            continue
        q = points[index]
        diff = q - p
        d2 = wp.dot(diff, diff)
        if d2 > r2:
            continue
        s += wp.sqrt(d2)
        count += 1

    if count < min_neighbors:
        mean_dist[i] = float(0.0)
        valid[i] = 0
        return

    mean = s / float(count)
    to_sensor = p - sensor_origin
    r = wp.sqrt(wp.dot(to_sensor, to_sensor))
    if r < range_eps:
        r = range_eps
    v = mean / r
    mean_dist[i] = v
    valid[i] = 1

    wp.atomic_add(out_sum, 0, v)
    wp.atomic_add(out_sum_sq, 0, v * v)
    wp.atomic_add(out_count, 0, 1)


@wp.kernel
def radius_outlier_filter_kernel(
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    search_radius: wp.float32,
    min_neighbors: wp.int32,
    out_counter: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
):
    """Radius Outlier Removal: keep points with ≥ `min_neighbors` inside `search_radius`.

    Single-pass fused kernel — counts neighbors, early-exits once the threshold is
    hit, and writes survivors straight into a compact output via atomic. No sqrt,
    no global μ/σ reduction, no second launch.
    """
    i = wp.tid()
    p = points[i]
    r2 = search_radius * search_radius
    count = int(0)

    neighbors = wp.hash_grid_query(grid, p, search_radius)
    for index in neighbors:
        if index == i:
            continue
        q = points[index]
        diff = q - p
        if wp.dot(diff, diff) <= r2:
            count += 1
            if count >= min_neighbors:
                break

    if count >= min_neighbors:
        slot = wp.atomic_add(out_counter, 0, 1)
        out_points[slot] = p


@wp.kernel
def compact_inliers_kernel(
    points: wp.array(dtype=wp.vec3),
    mean_dist: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
    threshold: wp.float32,
    out_counter: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
):
    """Write surviving points (valid and mean_dist ≤ threshold) to a compact buffer."""
    i = wp.tid()
    if valid[i] == 1 and mean_dist[i] <= threshold:
        slot = wp.atomic_add(out_counter, 0, 1)
        out_points[slot] = points[i]
