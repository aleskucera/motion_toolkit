from __future__ import annotations

import warp as wp

wp.init()


@wp.kernel
def compute_slope_sobel_kernel(
    heightmap: wp.array(dtype=wp.float32, ndim=2),
    resolution: wp.float32,
    grid_height: wp.int32,
    grid_width: wp.int32,
    slope_norm_factor: wp.float32,
    # scratch (normal is needed for the slope angle; we don't expose it)
    normals: wp.array(dtype=wp.vec3, ndim=2),
    slope_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """Surface normal + slope cost via Sobel gradients, clamped at the borders."""
    r, c = wp.tid()
    h_tl = heightmap[wp.clamp(r - 1, 0, grid_height - 1), wp.clamp(c - 1, 0, grid_width - 1)]
    h_tm = heightmap[wp.clamp(r - 1, 0, grid_height - 1), c]
    h_tr = heightmap[wp.clamp(r - 1, 0, grid_height - 1), wp.clamp(c + 1, 0, grid_width - 1)]
    h_ml = heightmap[r, wp.clamp(c - 1, 0, grid_width - 1)]
    h_mr = heightmap[r, wp.clamp(c + 1, 0, grid_width - 1)]
    h_bl = heightmap[wp.clamp(r + 1, 0, grid_height - 1), wp.clamp(c - 1, 0, grid_width - 1)]
    h_bm = heightmap[wp.clamp(r + 1, 0, grid_height - 1), c]
    h_br = heightmap[wp.clamp(r + 1, 0, grid_height - 1), wp.clamp(c + 1, 0, grid_width - 1)]

    dzdx = (h_tr + 2.0 * h_mr + h_br - (h_tl + 2.0 * h_ml + h_bl)) / (8.0 * resolution)
    dzdy = (h_tl + 2.0 * h_tm + h_tr - (h_bl + 2.0 * h_bm + h_br)) / (8.0 * resolution)

    n = wp.normalize(wp.vec3(-dzdx, -dzdy, 1.0))
    normals[r, c] = n

    slope_angle = wp.acos(n[2])
    slope_cost[r, c] = wp.min(slope_angle / slope_norm_factor, 1.0)


@wp.kernel
def morph_op_kernel(
    src: wp.array(dtype=wp.float32, ndim=2),
    grid_height: wp.int32,
    grid_width: wp.int32,
    radius: wp.int32,
    op: wp.int32,  # 0 = erode (min), 1 = dilate (max)
    dst: wp.array(dtype=wp.float32, ndim=2),
):
    """Box-shaped morphological erosion or dilation with arbitrary radius."""
    r, c = wp.tid()
    val = src[r, c]
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr = wp.clamp(r + dr, 0, grid_height - 1)
            nc = wp.clamp(c + dc, 0, grid_width - 1)
            if op == 0:
                val = wp.min(val, src[nr, nc])
            else:
                val = wp.max(val, src[nr, nc])
    dst[r, c] = val


@wp.kernel
def compute_step_height_cost_kernel(
    elevation: wp.array(dtype=wp.float32, ndim=2),
    dilated_map: wp.array(dtype=wp.float32, ndim=2),
    eroded_map: wp.array(dtype=wp.float32, ndim=2),
    step_norm_factor: wp.float32,
    drop_norm_factor: wp.float32,
    step_height_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """Signed step-height cost = max(bump_cost, drop_cost).

    Splits `dilated - eroded` into a positive-obstacle part (`dilated - elev`,
    nearby cells higher — curbs, rocks) and a negative-obstacle part
    (`elev - eroded`, nearby cells lower — ledges, ditches), normalizes each by
    its own saturation threshold, and returns the max. Lets callers weight
    drops more aggressively than bumps, which matters for ground robots where
    a cliff edge is usually worse than a comparable curb.
    """
    r, c = wp.tid()
    z = elevation[r, c]
    up = dilated_map[r, c] - z
    down = z - eroded_map[r, c]
    bump_cost = wp.min(up / step_norm_factor, 1.0)
    drop_cost = wp.min(down / drop_norm_factor, 1.0)
    step_height_cost[r, c] = wp.max(bump_cost, drop_cost)


@wp.kernel
def compute_roughness_kernel(
    heightmap: wp.array(dtype=wp.float32, ndim=2),
    grid_height: wp.int32,
    grid_width: wp.int32,
    window_radius: wp.int32,
    roughness_norm_factor: wp.float32,
    roughness_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """Surface roughness = local std-dev of height in a window, normalized."""
    r, c = wp.tid()
    sum_h = float(0.0)
    sum_sq_h = float(0.0)
    count = float(0.0)
    for dr in range(-window_radius, window_radius + 1):
        for dc in range(-window_radius, window_radius + 1):
            nr = r + dr
            nc = c + dc
            if nr >= 0 and nr < grid_height and nc >= 0 and nc < grid_width:
                h = heightmap[nr, nc]
                sum_h += h
                sum_sq_h += h * h
                count += 1.0
    if count > 1.0:
        mean_h = sum_h / count
        variance = (sum_sq_h / count) - (mean_h * mean_h)
        if variance < 0.0:
            variance = 0.0
        std_dev = wp.sqrt(variance)
        roughness_cost[r, c] = wp.min(std_dev / roughness_norm_factor, 1.0)
    else:
        roughness_cost[r, c] = 0.0


@wp.kernel
def combine_costs_kernel(
    slope_cost: wp.array(dtype=wp.float32, ndim=2),
    step_height_cost: wp.array(dtype=wp.float32, ndim=2),
    surf_roughness_cost: wp.array(dtype=wp.float32, ndim=2),
    w_slope: wp.float32,
    w_step: wp.float32,
    w_roughness: wp.float32,
    total_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """Weighted average of the three geometric cost layers."""
    r, c = wp.tid()
    s = slope_cost[r, c]
    h = step_height_cost[r, c]
    rf = surf_roughness_cost[r, c]
    combined = w_slope * s + w_step * h + w_roughness * rf
    total_weight = w_slope + w_step + w_roughness
    if total_weight > 0.0:
        total_cost[r, c] = combined / total_weight
    else:
        total_cost[r, c] = 0.0


@wp.kernel
def inflate_obstacles_kernel(
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    map_height: wp.int32,
    map_width: wp.int32,
    inflation_radius: wp.int32,
    inv_two_sigma_sq: wp.float32,
    obstacle_threshold: wp.float32,
    inflated_cost_map: wp.array(dtype=wp.float32, ndim=2),
):
    """Gaussian-weighted obstacle dilation.

    Only cells above `obstacle_threshold` act as sources.  Each source's
    influence decays with squared distance via exp(-d²/(2σ²)).  The output
    is max(original_cost, max_over_sources(source_cost * weight)), so costs
    are only ever raised, never lowered.
    """
    r, c = wp.tid()
    own = cost_map[r, c]
    best = own
    for dr in range(-inflation_radius, inflation_radius + 1):
        for dc in range(-inflation_radius, inflation_radius + 1):
            nr = r + dr
            nc = c + dc
            if nr >= 0 and nr < map_height and nc >= 0 and nc < map_width:
                v = cost_map[nr, nc]
                if not wp.isnan(v) and v > obstacle_threshold:
                    dist_sq = float(dr * dr + dc * dc)
                    weight = wp.exp(-dist_sq * inv_two_sigma_sq)
                    candidate = v * weight
                    if candidate > best:
                        best = candidate
    inflated_cost_map[r, c] = best


@wp.kernel
def count_obstacles_kernel(
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    obstacle_threshold: wp.float32,
    num_obstacles: wp.array(dtype=wp.int32),
):
    """Atomically count cells whose cost exceeds the obstacle threshold."""
    r, c = wp.tid()
    v = cost_map[r, c]
    if not wp.isnan(v) and v > obstacle_threshold:
        wp.atomic_add(num_obstacles, 0, 1)


@wp.kernel
def support_ratio_mask_kernel(
    elevation_map: wp.array(dtype=wp.float32, ndim=2),
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    map_height: wp.int32,
    map_width: wp.int32,
    support_radius: wp.int32,
    support_ratio: wp.float32,
    filtered_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """Keep cost where the local neighborhood has enough measured cells; else NaN."""
    r, c = wp.tid()
    measured = int(0)
    total = int(0)
    for dr in range(-support_radius, support_radius + 1):
        for dc in range(-support_radius, support_radius + 1):
            nr = r + dr
            nc = c + dc
            if nr >= 0 and nr < map_height and nc >= 0 and nc < map_width:
                val = elevation_map[nr, nc]
                total += 1
                if not wp.isnan(val):
                    measured += 1
    ratio = float(0.0)
    if total > 0:
        ratio = float(measured) / float(total)
    if ratio >= support_ratio:
        filtered_cost[r, c] = cost_map[r, c]
    else:
        filtered_cost[r, c] = wp.float32(wp.nan)
