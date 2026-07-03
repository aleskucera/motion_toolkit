"""Launch terrain_toolkit_ros with Ouster-tuned defaults."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    args = [
        # ROS / sensor
        DeclareLaunchArgument(
            "lidar_topic", default_value="/ouster/points", description="PointCloud2 input topic"
        ),
        DeclareLaunchArgument(
            "map_frame", default_value="map", description="Map TF frame (unused)"
        ),
        DeclareLaunchArgument(
            "robot_frame_ga",
            default_value="base_link",
            description="Gravity-aligned robot TF frame the heightmap is built in "
            "(use a real gravity-aligned frame on non-flat terrain)",
        ),
        DeclareLaunchArgument(
            "robot_frame",
            default_value="base_link",
            description="Normal (un-leveled) robot body TF frame; used for the flat-footprint plane",
        ),
        DeclareLaunchArgument(
            "square_half_size", default_value="10.0", description="Half-side of square ROI (m)"
        ),
        # Grid
        DeclareLaunchArgument("resolution", default_value="0.15", description="Grid cell size (m)"),
        DeclareLaunchArgument(
            "x_range", default_value="5.0", description="Grid half-extent in x (m)"
        ),
        DeclareLaunchArgument(
            "y_range", default_value="5.0", description="Grid half-extent in y (m)"
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
            "inpaint_coarse_iters", default_value="50", description="Inpaint coarse iterations"
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
        executable="terrain_toolkit_node",
        name="helhest.perception",
        output="screen",
        parameters=[
            {
                # ROS / sensor
                "lidar_topic": lc("lidar_topic"),
                "map_frame": lc("map_frame"),
                "robot_frame_ga": lc("robot_frame_ga"),
                "robot_frame": lc("robot_frame"),
                "square_half_size": lc("square_half_size"),
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
