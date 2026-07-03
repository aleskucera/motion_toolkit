"""Launch file for the accumulating terrain mapper node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    args = [
        # ROS / sensors
        DeclareLaunchArgument(
            "lidar_topic", default_value="/lidar/points", description="PointCloud2 input topic"
        ),
        DeclareLaunchArgument(
            "odom_topic", default_value="/odom", description="nav_msgs/Odometry input topic"
        ),
        DeclareLaunchArgument(
            "base_frame",
            default_value="base_link",
            description="Robot body TF frame; must match the odometry child_frame_id",
        ),
        DeclareLaunchArgument(
            "map_frame", default_value="map", description="World frame the accumulated map lives in"
        ),
        DeclareLaunchArgument(
            "sync_slop_s", default_value="0.05", description="Cloud/odom time-sync tolerance (s)"
        ),
        DeclareLaunchArgument(
            "sync_queue", default_value="30", description="Cloud/odom sync queue size"
        ),
        # Accumulation / map
        DeclareLaunchArgument(
            "accumulation_voxel_m",
            default_value="0.10",
            description="Global cloud voxel-downsample size (m)",
        ),
        DeclareLaunchArgument(
            "map_max_radius_m",
            default_value="50.0",
            description="Drop accumulated points beyond this distance from the robot (m)",
        ),
        # ICP (scan-to-submap)
        DeclareLaunchArgument(
            "icp_enable",
            default_value="true",
            description="Refine odom with ICP (false = odom dead-reckoning)",
        ),
        DeclareLaunchArgument(
            "icp_submap_radius_m",
            default_value="15.0",
            description="Half-extent of the ICP target submap (m)",
        ),
        DeclareLaunchArgument(
            "icp_max_iters", default_value="30", description="Max ICP iterations"
        ),
        DeclareLaunchArgument(
            "icp_max_corr_dist_m",
            default_value="0.5",
            description="ICP max correspondence distance (m)",
        ),
        DeclareLaunchArgument(
            "icp_normal_radius_m",
            default_value="0.3",
            description="ICP target-normal estimation radius (m)",
        ),
        DeclareLaunchArgument(
            "icp_voxel_size_m",
            default_value="0.1",
            description="ICP source/target voxel size (0 = off) (m)",
        ),
        DeclareLaunchArgument(
            "icp_voxel_target",
            default_value="true",
            description="Also voxel-downsample the ICP target submap",
        ),
        DeclareLaunchArgument(
            "icp_min_inliers",
            default_value="500",
            description="Reject ICP below this inlier count",
        ),
        DeclareLaunchArgument(
            "icp_max_corr_trans_m",
            default_value="1.0",
            description="Reject ICP correcting the prediction more than this (m)",
        ),
        DeclareLaunchArgument(
            "icp_max_corr_rot_deg",
            default_value="15.0",
            description="Reject ICP correcting the prediction more than this (deg)",
        ),
        DeclareLaunchArgument(
            "icp_min_submap_points",
            default_value="2000",
            description="Skip ICP if the submap has fewer points",
        ),
        # Dynamic obstacle filter (map-frame visibility; removes moving people)
        DeclareLaunchArgument(
            "dynamic_enable",
            default_value="false",
            description="Drop moving objects by map-frame visibility before accumulation",
        ),
        DeclareLaunchArgument(
            "dynamic_az_bins", default_value="900", description="Range-image azimuth bins (360°)"
        ),
        DeclareLaunchArgument(
            "dynamic_el_bins", default_value="64", description="Range-image elevation bins (FOV)"
        ),
        DeclareLaunchArgument(
            "dynamic_el_min_deg",
            default_value="-25.0",
            description="Bottom of the sensor vertical FOV (deg)",
        ),
        DeclareLaunchArgument(
            "dynamic_el_max_deg",
            default_value="25.0",
            description="Top of the sensor vertical FOV (deg)",
        ),
        DeclareLaunchArgument(
            "dynamic_margin_m",
            default_value="0.3",
            description="Base depth margin for the visibility test (m)",
        ),
        DeclareLaunchArgument(
            "dynamic_margin_rel",
            default_value="0.02",
            description="Range-proportional depth margin (m per m)",
        ),
        DeclareLaunchArgument(
            "dynamic_min_range_m",
            default_value="0.5",
            description="Ignore returns closer than this (m)",
        ),
        # Viz
        DeclareLaunchArgument(
            "publish_map_tf",
            default_value="true",
            description="Broadcast the map→odom correction transform",
        ),
        # Grid (robot-centric window = pipeline bounds)
        DeclareLaunchArgument("resolution", default_value="0.15", description="Grid cell size (m)"),
        DeclareLaunchArgument(
            "x_range", default_value="12.0", description="Window half-extent in x (m)"
        ),
        DeclareLaunchArgument(
            "y_range", default_value="12.0", description="Window half-extent in y (m)"
        ),
        # Pipeline
        DeclareLaunchArgument(
            "z_max", default_value="1.0", description="Discard points above this height (m)"
        ),
        DeclareLaunchArgument(
            "primary", default_value="max", description="Height reduction: max | mean | min"
        ),
        DeclareLaunchArgument(
            "inpaint", default_value="true", description="Enable multigrid inpainting"
        ),
        DeclareLaunchArgument(
            "inpaint_coarse_iters", default_value="200", description="Inpaint coarse iterations"
        ),
        DeclareLaunchArgument(
            "inpaint_iters_per_level",
            default_value="50",
            description="Inpaint iterations per pyramid level",
        ),
        DeclareLaunchArgument(
            "smooth_sigma", default_value="0.8", description="Gaussian smoothing sigma (m)"
        ),
        # Outlier
        DeclareLaunchArgument(
            "outlier_enable", default_value="false", description="Enable outlier filtering"
        ),
        DeclareLaunchArgument(
            "outlier_type", default_value="ror", description="Outlier algorithm: ror | sor"
        ),
        DeclareLaunchArgument(
            "outlier_search_radius_m",
            default_value="0.25",
            description="Neighbor search radius (m)",
        ),
        DeclareLaunchArgument(
            "outlier_min_neighbors", default_value="10", description="Min neighbors within radius"
        ),
        DeclareLaunchArgument(
            "outlier_std_multiplier",
            default_value="1.0",
            description="SOR std multiplier (unused for ROR)",
        ),
        # Traversability
        DeclareLaunchArgument(
            "trav_enable", default_value="true", description="Compute traversability cost layers"
        ),
        DeclareLaunchArgument(
            "trav_max_slope_deg",
            default_value="60.0",
            description="Slope saturating cost to 1 (deg)",
        ),
        DeclareLaunchArgument(
            "trav_max_step_height_m",
            default_value="0.55",
            description="Upward step saturating cost to 1 (m)",
        ),
        DeclareLaunchArgument(
            "trav_max_drop_height_m",
            default_value="0.3",
            description="Downward drop saturating cost to 1 (m)",
        ),
        DeclareLaunchArgument(
            "trav_max_roughness_m",
            default_value="0.2",
            description="Roughness saturating cost to 1 (m)",
        ),
        DeclareLaunchArgument(
            "trav_step_window_radius_m",
            default_value="0.15",
            description="Morphological window radius for step detection (m)",
        ),
        DeclareLaunchArgument(
            "trav_roughness_window_radius_m",
            default_value="0.3",
            description="Window radius for roughness std-dev (m)",
        ),
        DeclareLaunchArgument(
            "trav_slope_weight", default_value="0.2", description="Slope weight in combined cost"
        ),
        DeclareLaunchArgument(
            "trav_step_weight", default_value="0.2", description="Step weight in combined cost"
        ),
        DeclareLaunchArgument(
            "trav_roughness_weight",
            default_value="0.6",
            description="Roughness weight in combined cost",
        ),
        # Temporal filter
        DeclareLaunchArgument(
            "filter_enable",
            default_value="true",
            description="Enable obstacle inflation + temporal gate",
        ),
        DeclareLaunchArgument(
            "filter_support_radius_m",
            default_value="0.5",
            description="Neighborhood radius for support check (m)",
        ),
        DeclareLaunchArgument(
            "filter_support_ratio",
            default_value="0.5",
            description="Min fraction of measured cells to keep",
        ),
        DeclareLaunchArgument(
            "filter_inflation_sigma_m",
            default_value="0.3",
            description="Gaussian sigma for obstacle dilation (m)",
        ),
        DeclareLaunchArgument(
            "filter_obstacle_threshold",
            default_value="0.8",
            description="Cost threshold for obstacle source",
        ),
        DeclareLaunchArgument(
            "filter_obstacle_growth_threshold",
            default_value="2.0",
            description="Reject frame if obstacle count grows by this factor",
        ),
        DeclareLaunchArgument(
            "filter_rejection_limit_frames",
            default_value="5",
            description="Force-accept after this many consecutive rejections",
        ),
        DeclareLaunchArgument(
            "filter_min_obstacle_baseline",
            default_value="10",
            description="Skip hysteresis until this many obstacles seen",
        ),
        # Occlusion (line-of-sight) masking
        DeclareLaunchArgument(
            "occlusion_enable",
            default_value="false",
            description="NaN-out cost in the line-of-sight shadow of obstacles",
        ),
        DeclareLaunchArgument(
            "occlusion_sensor_x",
            default_value="0.0",
            description="Sensor x in the gravity-aligned grid frame (m)",
        ),
        DeclareLaunchArgument(
            "occlusion_sensor_y",
            default_value="0.0",
            description="Sensor y in the gravity-aligned grid frame (m)",
        ),
        DeclareLaunchArgument(
            "occlusion_sensor_z",
            default_value="0.5",
            description="Sensor height above the grid origin (m)",
        ),
        DeclareLaunchArgument(
            "occlusion_angle_eps_deg",
            default_value="0.6",
            description="View-angle margin guarding flat-ground noise (deg)",
        ),
        # Flat ground footprint
        DeclareLaunchArgument(
            "footprint_enable",
            default_value="false",
            description="Force a flat ground patch under the robot",
        ),
        DeclareLaunchArgument(
            "footprint_robot_height",
            default_value="0.4",
            description="Vertical distance robot frame → ground (m)",
        ),
        DeclareLaunchArgument(
            "footprint_half_x", default_value="0.5", description="Footprint half-extent along x (m)"
        ),
        DeclareLaunchArgument(
            "footprint_half_y", default_value="0.5", description="Footprint half-extent along y (m)"
        ),
        DeclareLaunchArgument(
            "footprint_center_x",
            default_value="0.0",
            description="Footprint center offset along x (m)",
        ),
        DeclareLaunchArgument(
            "footprint_center_y",
            default_value="0.0",
            description="Footprint center offset along y (m)",
        ),
        DeclareLaunchArgument(
            "footprint_mode",
            default_value="overwrite",
            description="Footprint fill mode: overwrite | fill",
        ),
    ]

    lc = LaunchConfiguration

    node = Node(
        package="terrain_toolkit_ros",
        executable="terrain_accumulator_node",
        name="terrain_accumulator",
        output="screen",
        parameters=[
            {
                # ROS / sensors
                "lidar_topic": lc("lidar_topic"),
                "odom_topic": lc("odom_topic"),
                "base_frame": lc("base_frame"),
                "map_frame": lc("map_frame"),
                "sync_slop_s": lc("sync_slop_s"),
                "sync_queue": lc("sync_queue"),
                # Accumulation / map
                "accumulation_voxel_m": lc("accumulation_voxel_m"),
                "map_max_radius_m": lc("map_max_radius_m"),
                # ICP
                "icp_enable": lc("icp_enable"),
                "icp_submap_radius_m": lc("icp_submap_radius_m"),
                "icp_max_iters": lc("icp_max_iters"),
                "icp_max_corr_dist_m": lc("icp_max_corr_dist_m"),
                "icp_normal_radius_m": lc("icp_normal_radius_m"),
                "icp_voxel_size_m": lc("icp_voxel_size_m"),
                "icp_voxel_target": lc("icp_voxel_target"),
                "icp_min_inliers": lc("icp_min_inliers"),
                "icp_max_corr_trans_m": lc("icp_max_corr_trans_m"),
                "icp_max_corr_rot_deg": lc("icp_max_corr_rot_deg"),
                "icp_min_submap_points": lc("icp_min_submap_points"),
                # Dynamic obstacle filter
                "dynamic_enable": lc("dynamic_enable"),
                "dynamic_az_bins": lc("dynamic_az_bins"),
                "dynamic_el_bins": lc("dynamic_el_bins"),
                "dynamic_el_min_deg": lc("dynamic_el_min_deg"),
                "dynamic_el_max_deg": lc("dynamic_el_max_deg"),
                "dynamic_margin_m": lc("dynamic_margin_m"),
                "dynamic_margin_rel": lc("dynamic_margin_rel"),
                "dynamic_min_range_m": lc("dynamic_min_range_m"),
                # Viz
                "publish_map_tf": lc("publish_map_tf"),
                # Grid
                "resolution": lc("resolution"),
                "x_range": lc("x_range"),
                "y_range": lc("y_range"),
                # Pipeline
                "z_max": lc("z_max"),
                "primary": lc("primary"),
                "inpaint": lc("inpaint"),
                "inpaint_coarse_iters": lc("inpaint_coarse_iters"),
                "inpaint_iters_per_level": lc("inpaint_iters_per_level"),
                "smooth_sigma": lc("smooth_sigma"),
                # Outlier
                "outlier_enable": lc("outlier_enable"),
                "outlier_type": lc("outlier_type"),
                "outlier_search_radius_m": lc("outlier_search_radius_m"),
                "outlier_min_neighbors": lc("outlier_min_neighbors"),
                "outlier_std_multiplier": lc("outlier_std_multiplier"),
                # Traversability
                "trav_enable": lc("trav_enable"),
                "trav_max_slope_deg": lc("trav_max_slope_deg"),
                "trav_max_step_height_m": lc("trav_max_step_height_m"),
                "trav_max_drop_height_m": lc("trav_max_drop_height_m"),
                "trav_max_roughness_m": lc("trav_max_roughness_m"),
                "trav_step_window_radius_m": lc("trav_step_window_radius_m"),
                "trav_roughness_window_radius_m": lc("trav_roughness_window_radius_m"),
                "trav_slope_weight": lc("trav_slope_weight"),
                "trav_step_weight": lc("trav_step_weight"),
                "trav_roughness_weight": lc("trav_roughness_weight"),
                # Temporal filter
                "filter_enable": lc("filter_enable"),
                "filter_support_radius_m": lc("filter_support_radius_m"),
                "filter_support_ratio": lc("filter_support_ratio"),
                "filter_inflation_sigma_m": lc("filter_inflation_sigma_m"),
                "filter_obstacle_threshold": lc("filter_obstacle_threshold"),
                "filter_obstacle_growth_threshold": lc("filter_obstacle_growth_threshold"),
                "filter_rejection_limit_frames": lc("filter_rejection_limit_frames"),
                "filter_min_obstacle_baseline": lc("filter_min_obstacle_baseline"),
                # Occlusion masking
                "occlusion_enable": lc("occlusion_enable"),
                "occlusion_sensor_x": lc("occlusion_sensor_x"),
                "occlusion_sensor_y": lc("occlusion_sensor_y"),
                "occlusion_sensor_z": lc("occlusion_sensor_z"),
                "occlusion_angle_eps_deg": lc("occlusion_angle_eps_deg"),
                # Flat ground footprint
                "footprint_enable": lc("footprint_enable"),
                "footprint_robot_height": lc("footprint_robot_height"),
                "footprint_half_x": lc("footprint_half_x"),
                "footprint_half_y": lc("footprint_half_y"),
                "footprint_center_x": lc("footprint_center_x"),
                "footprint_center_y": lc("footprint_center_y"),
                "footprint_mode": lc("footprint_mode"),
            }
        ],
    )

    return LaunchDescription(args + [node])
