from __future__ import annotations

import warp as wp

wp.init()


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


@wp.kernel
def occlusion_mask_kernel(
    elevation_map: wp.array(dtype=wp.float32, ndim=2),
    raw_elevation: wp.array(dtype=wp.float32, ndim=2),
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    xmin: wp.float32,
    ymin: wp.float32,
    resolution: wp.float32,
    sensor_x: wp.float32,
    sensor_y: wp.float32,
    sensor_z: wp.float32,
    angle_eps: wp.float32,
    filtered_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """NaN-out cost where a cell is occluded from the sensor AND was not measured.

    Line-of-sight test: march from the sensor to the cell over the (inpainted)
    elevation, tracking the max view angle of the intervening cells. The cell is
    occluded if that horizon rises above the cell's own view angle by more than
    `angle_eps`. Only unmeasured cells (NaN in `raw_elevation`) are removed; real
    measurements always pass through — so this never deletes data the sensor
    actually saw, only inpainted free-space behind obstacles.
    """
    r, c = wp.tid()
    h = elevation_map[r, c]
    tx = xmin + (float(c) + 0.5) * resolution
    ty = ymin + (float(r) + 0.5) * resolution
    # sensor position in (fractional) grid coords: x -> column, y -> row.
    sj = (sensor_x - xmin) / resolution - 0.5
    si = (sensor_y - ymin) / resolution - 0.5
    dt = wp.sqrt((tx - sensor_x) * (tx - sensor_x) + (ty - sensor_y) * (ty - sensor_y))
    if dt < 1.0e-6 or wp.isnan(h):
        filtered_cost[r, c] = cost_map[r, c]
        return
    tang = wp.atan2(h - sensor_z, dt)
    di = float(r) - si
    dj = float(c) - sj
    steps = int(wp.max(wp.abs(di), wp.abs(dj)))
    maxang = float(-1.0e9)
    for k in range(1, steps):
        fk = float(k) / float(steps)
        cr = int(si + di * fk + 0.5)
        cc = int(sj + dj * fk + 0.5)
        if cr < 0 or cr >= elevation_map.shape[0] or cc < 0 or cc >= elevation_map.shape[1]:
            continue
        ch = elevation_map[cr, cc]
        if wp.isnan(ch):
            continue
        cx = xmin + (float(cc) + 0.5) * resolution
        cy = ymin + (float(cr) + 0.5) * resolution
        cd = wp.sqrt((cx - sensor_x) * (cx - sensor_x) + (cy - sensor_y) * (cy - sensor_y))
        ca = wp.atan2(ch - sensor_z, cd)
        maxang = wp.max(maxang, ca)
    if maxang > tang + angle_eps and wp.isnan(raw_elevation[r, c]):
        filtered_cost[r, c] = wp.float32(wp.nan)
    else:
        filtered_cost[r, c] = cost_map[r, c]
