# terrain_toolkit_ros

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
colcon build --packages-select terrain_toolkit_ros --symlink-install
source install/setup.bash
```

> The ROS package sources live under `ros/terrain_toolkit_ros/`. Symlink or
> copy that directory into your colcon workspace's `src/`, or set the
> workspace root so colcon discovers it.

## Run

```bash
ros2 launch terrain_toolkit_ros terrain_toolkit_node.launch.py \
    lidar_topic:=/points \
    robot_frame_ga:=base_link \
    resolution:=0.15 \
    x_range:=12.0 y_range:=12.0
```

All pipeline parameters can be changed at runtime with `ros2 param set` /
`rqt_reconfigure` — the pipeline is rebuilt in-place.

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

See `launch/terrain_toolkit_node.launch.py` for the full list.
