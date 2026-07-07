# helhest_stack_ros

ROS 2 **Kilted** wrapper for the [`terrain_toolkit`](https://github.com/aleskucera/terrain_toolkit)
library. Subscribes to a LiDAR `PointCloud2`, transforms it into the robot
frame, runs the GPU terrain pipeline, and republishes the resulting grid as
a `PointCloud2` whose points carry one `FLOAT32` field per `TerrainMap`
layer (`x`, `y`, `z`, `max`, `mean`, `min`, `count`, `elevation`,
`slope_cost`, `step_cost`, `roughness_cost`, `traversability` — fields only
appear for layers the pipeline actually produces).

## Install

The wrapper does **not** vendor the core library. Install `terrain_toolkit`
into the same Python environment your ROS 2 workspace uses:

```bash
# from repo root
pip install -e .
```

Then build the ROS package inside a colcon workspace:

```bash
# assumes this repo is cloned into <ws>/src/terrain_toolkit
cd <ws>
colcon build --packages-select helhest_stack_ros --symlink-install
source install/setup.bash
```

> The ROS package sources live under `ros/helhest_stack_ros/`. Symlink or
> copy that directory into your colcon workspace's `src/`, or set the
> workspace root so colcon discovers it.

## Run

```bash
ros2 launch helhest_stack_ros single_scan_terrain_node.launch.py \
    lidar_topic:=/points \
    robot_frame_ga:=base_link \
    resolution:=0.15 \
    x_range:=12.0 y_range:=12.0
```

All pipeline parameters can be changed at runtime with `ros2 param set` /
`rqt_reconfigure` — the pipeline is rebuilt in-place.

## Accumulating mapper (`terrain_accumulator_node`)

`single_scan_terrain_node` maps a single scan. `terrain_accumulator_node` fuses the
LiDAR stream with robot odometry (`nav_msgs/Odometry`) into a persistent point
cloud and runs the same pipeline over a **robot-centric window** of it, so the
published `terrain_map` covers more than one scan (terrain already passed or
hidden behind obstacles is retained).

Each scan is registered **scan-to-submap** with point-to-plane ICP, using the
odometry frame-to-frame delta as the initial guess; the refined trajectory
corrects odom drift. The map is accumulated in `map_frame`, which is bootstrapped
to the odom frame on the first scan — so it must be **gravity-aligned**
(odom z-axis up). A `map→odom` correction transform is broadcast for RViz.

> Not a SLAM back-end: there is no loop closure, and ICP roll/pitch corrections
> can slowly tilt the world frame over very long runs (negligible for the
> bounded window).

```bash
ros2 launch helhest_stack_ros terrain_accumulator_node.launch.py \
    lidar_topic:=/points \
    odom_topic:=/odom \
    base_frame:=base_link \
    map_frame:=map \
    x_range:=12.0 y_range:=12.0
```

Key extra parameters (beyond the shared pipeline set): `odom_topic`,
`base_frame` (must match the odometry `child_frame_id`), `map_frame`,
`accumulation_voxel_m`, `map_max_radius_m`, `icp_enable`, `icp_submap_radius_m`,
`icp_voxel_size_m`, and the ICP divergence gate (`icp_min_inliers`,
`icp_max_corr_trans_m`, `icp_max_corr_rot_deg`, `icp_min_submap_points`). See
`launch/terrain_accumulator_node.launch.py` for the full list and defaults.

### Dynamic obstacle filter (`dynamic_enable`)

Moving objects (people walking around the robot) would otherwise smear into the
accumulated map. Set `dynamic_enable:=true` to drop them by **map-frame
visibility**: each scan is compared against the accumulated map through a
spherical range image rendered from the current sensor pose. Scan points sitting
*in front of* known static geometry are dropped (a person occluding the
background), and map points the scan now sees *through* are carved (the trail
that person left). It removes **moving** objects only — a person standing
motionless is geometrically a static pillar and is kept. It works for a
stationary robot (it degenerates to per-beam background subtraction) and for slow
motion (the tracked pose keeps the two range images aligned).

Tuning: `dynamic_el_min_deg`/`dynamic_el_max_deg` should match your sensor's
vertical FOV; `dynamic_az_bins`/`dynamic_el_bins` set the range-image resolution
(roughly the sensor's beam pattern); `dynamic_margin_m` + `dynamic_margin_rel`
trade false removals (too small) against missed dynamics (too large).

Bring-up checklist:

1. `pip install -e .` (core lib) then `colcon build --packages-select
   helhest_stack_ros --symlink-install`.
2. Play a bag with a `PointCloud2`, a `nav_msgs/Odometry`, and a static
   sensor→`base_frame` TF.
3. Launch as above; in RViz set the fixed frame to `map`.
4. Confirm `terrain_map` grows past a single scan, stays level, and tracks the
   robot; check that the `map→odom` TF appears. Watch the log for ICP
   divergence-fallback / sparse-submap messages.

## Parameters

| Group | Parameter | Default | Description |
|-------|-----------|---------|-------------|
| ROS | `lidar_topic` | `/lidar/points` | PointCloud2 input topic |
| ROS | `robot_frame_ga` | `base_link` | Gravity-aligned frame the heightmap is built in |
| ROS | `robot_frame` | `base_link` | Normal robot body frame (flat-footprint plane) |
| Grid | `resolution` | `0.15` | Cell size (m) |
| Grid | `x_range` / `y_range` | `12.0` / `12.0` | Half-extent of the ROI (m) |
| Pipeline | `z_max` | `1.0` | Discard points above this height |
| Pipeline | `primary` | `max` | Height reduction (`max`, `mean`, `min`) |
| Pipeline | `inpaint` | `true` | Multigrid inpaint of missing cells |
| Pipeline | `smooth_sigma` | `0.8` | Gaussian smoothing sigma (m) |
| Outlier | `outlier_enable` / `outlier_type` | `true` / `ror` | `ror` (radius) or `sor` (statistical) |
| Outlier | `outlier_search_radius_m` | `0.25` | Neighbor radius (m) |
| Outlier | `outlier_min_neighbors` | `10` | Keep points with at least this many neighbors |
| Trav. | `trav_enable` | `true` | Compute slope / step / roughness costs |
| Trav. | `trav_max_step_height_m` | `0.55` | Upward step saturating cost to 1 |
| Trav. | `trav_max_drop_height_m` | `0.3` | Downward drop saturating cost to 1 |
| Filter | `filter_enable` | `true` | Obstacle inflation + temporal gate |
| Occlusion | `occlusion_enable` | `false` | NaN-out cost in the line-of-sight shadow of obstacles |
| Occlusion | `occlusion_sensor_x/y/z` | `0/0/0.5` | Sensor position in the gravity-aligned grid frame (m) |
| Occlusion | `occlusion_angle_eps_deg` | `0.6` | View-angle margin guarding flat-ground noise (deg) |
| Footprint | `footprint_enable` | `false` | Force a flat ground patch under the robot |
| Footprint | `footprint_robot_height` | `0.4` | Vertical distance robot frame → ground (m) |
| Footprint | `footprint_half_x` / `footprint_half_y` | `0.5` / `0.5` | Footprint half-extents (m) |
| Footprint | `footprint_mode` | `overwrite` | `overwrite` real cells or `fill` gaps only |

See `launch/single_scan_terrain_node.launch.py` for the full list.
