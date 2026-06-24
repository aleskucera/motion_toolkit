#!/usr/bin/env python3
"""
ROS 2 Kilted interface node for the terrain_toolkit library.

Subscribes to a LiDAR PointCloud2, transforms it into the robot frame, runs the
terrain_toolkit pipeline, and republishes the resulting grid as a PointCloud2
with one float32 PointField per TerrainMap layer.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    SetParametersResult,
)

import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

from terrain_toolkit import (
    FilterConfig,
    FootprintConfig,
    OcclusionConfig,
    OutlierFilterConfig,
    RadiusOutlierFilterConfig,
    TerrainMap,
    TerrainPipeline,
    TraversabilityConfig,
)


def _quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n == 0.0 else 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy,         0.0],
        [xy + wz,         1.0 - (xx + zz), yz - wx,         0.0],
        [xz - wy,         yz + wx,         1.0 - (xx + yy), 0.0],
        [0.0,             0.0,             0.0,             1.0],
    ], dtype=np.float64)


# Parameters that require rebuilding the TerrainPipeline when changed.
_PIPELINE_PARAMS = frozenset({
    "device",
    "resolution", "x_range", "y_range",
    "z_max", "primary", "inpaint", "inpaint_coarse_iters", "inpaint_iters_per_level", "smooth_sigma",
    "outlier_enable", "outlier_type",
    "outlier_search_radius_m", "outlier_min_neighbors", "outlier_std_multiplier",
    "trav_enable",
    "trav_max_slope_deg", "trav_max_step_height_m", "trav_max_drop_height_m", "trav_max_roughness_m",
    "trav_step_window_radius_m", "trav_roughness_window_radius_m",
    "trav_slope_weight", "trav_step_weight", "trav_roughness_weight",
    "filter_enable",
    "filter_support_radius_m", "filter_support_ratio", "filter_inflation_sigma_m",
    "filter_obstacle_threshold", "filter_obstacle_growth_threshold",
    "filter_rejection_limit_frames", "filter_min_obstacle_baseline",
    "occlusion_enable", "occlusion_sensor_x", "occlusion_sensor_y", "occlusion_sensor_z",
    "occlusion_angle_eps_deg",
    "footprint_enable", "footprint_robot_height",
    "footprint_half_x", "footprint_half_y",
    "footprint_center_x", "footprint_center_y", "footprint_mode",
})


class TerrainToolkitNode(Node):
    """Bridge a LiDAR PointCloud2 topic to the terrain_toolkit pipeline."""

    def __init__(self) -> None:
        super().__init__("terrain_toolkit")

        self._declare_parameters()
        p = self._read_parameters()

        self.lidar_topic: str = p["lidar_topic"]
        self.map_frame: str = p["map_frame"]
        # Gravity-aligned frame the heightmap is built in (cloud is transformed
        # here; slope/step are only meaningful on a level grid).
        self.robot_frame_ga: str = p["robot_frame_ga"]
        # Normal (un-leveled) robot body frame, used to derive the tilted ground
        # plane under the robot for the flat-footprint feature.
        self.robot_frame: str = p["robot_frame"]
        self.resolution: float = p["resolution"]
        self.x_range: float = p["x_range"]
        self.y_range: float = p["y_range"]
        self.square_half_size: float = p["square_half_size"]

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointCloud2, self.lidar_topic, self._cloud_callback, 10,
        )
        self.pub = self.create_publisher(PointCloud2, "terrain_map", 10)

        self._build_pipeline(p)
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self._log_config(p)

    # ------------------------------------------------------------------
    # Parameter declaration
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:

        def fp(desc: str, lo: float, hi: float) -> ParameterDescriptor:
            return ParameterDescriptor(
                description=desc,
                floating_point_range=[FloatingPointRange(from_value=lo, to_value=hi, step=0.0)],
            )

        def ip(desc: str, lo: int, hi: int) -> ParameterDescriptor:
            return ParameterDescriptor(
                description=desc,
                integer_range=[IntegerRange(from_value=lo, to_value=hi, step=1)],
            )

        def sp(desc: str) -> ParameterDescriptor:
            return ParameterDescriptor(description=desc)

        # ROS / sensor
        self.declare_parameter("lidar_topic", "/lidar/points", sp("PointCloud2 input topic"))
        self.declare_parameter("map_frame", "map", sp("Map TF frame (unused)"))
        self.declare_parameter(
            "robot_frame_ga", "base_link",
            sp("Gravity-aligned robot TF frame the heightmap is built in "
               "(use a real gravity-aligned frame on non-flat terrain)"),
        )
        self.declare_parameter(
            "robot_frame", "base_link",
            sp("Normal (un-leveled) robot body TF frame; used for the flat-footprint plane"),
        )
        self.declare_parameter("square_half_size", 10.0, fp("Half-side of square ROI (m)", 0.5, 200.0))
        self.declare_parameter(
            "device", "auto",
            sp("Warp compute device: 'auto', 'cpu', or 'cuda:N'. "
               "'auto' = CUDA if available else CPU. Outlier filtering requires CUDA."),
        )

        # Grid
        self.declare_parameter("resolution", 0.15, fp("Grid cell size (m)", 0.01, 5.0))
        self.declare_parameter("x_range", 12.0, fp("Grid half-extent in x (m)", 0.0, 50.0))
        self.declare_parameter("y_range", 12.0, fp("Grid half-extent in y (m)", 0.0, 50.0))

        # Pipeline
        self.declare_parameter("z_max", 1.0, fp("Discard points above this height (m)", -10.0, 50.0))
        self.declare_parameter("primary", "max", sp("Height reduction: 'max' | 'mean' | 'min'"))
        self.declare_parameter("inpaint", True, sp("Enable multigrid inpainting"))
        self.declare_parameter("inpaint_coarse_iters", 200, ip("Inpaint coarse iterations", 1, 10_000))
        self.declare_parameter("inpaint_iters_per_level", 50, ip("Inpaint iterations per pyramid level", 1, 5_000))
        self.declare_parameter("smooth_sigma", 0.8, fp("Gaussian smoothing sigma (m)", 0.0, 10.0))

        # Outlier filter
        self.declare_parameter("outlier_enable", True, sp("Enable outlier filtering before gridding"))
        self.declare_parameter("outlier_type", "ror", sp("Outlier algorithm: 'ror' (radius) | 'sor' (statistical)"))
        self.declare_parameter("outlier_search_radius_m", 0.25, fp("Neighbor search radius (m)", 0.01, 5.0))
        self.declare_parameter("outlier_min_neighbors", 10, ip("Min neighbors within radius to keep a point", 1, 1000))
        self.declare_parameter("outlier_std_multiplier", 1.0, fp("SOR std multiplier (ignored for ROR)", 0.0, 10.0))

        # Traversability
        self.declare_parameter("trav_enable", True, sp("Compute traversability cost layers"))
        self.declare_parameter("trav_max_slope_deg", 60.0, fp("Slope that saturates cost to 1 (deg)", 0.0, 90.0))
        self.declare_parameter("trav_max_step_height_m", 0.55, fp("Upward step height saturating cost to 1 (m)", 0.0, 5.0))
        self.declare_parameter("trav_max_drop_height_m", 0.3, fp("Downward drop height saturating cost to 1 (m)", 0.0, 5.0))
        self.declare_parameter("trav_max_roughness_m", 0.2, fp("Roughness saturating cost to 1 (m)", 0.0, 5.0))
        self.declare_parameter("trav_step_window_radius_m", 0.15, fp("Morphological window radius for step detection (m)", 0.01, 5.0))
        self.declare_parameter("trav_roughness_window_radius_m", 0.3, fp("Window radius for roughness std-dev (m)", 0.01, 5.0))
        self.declare_parameter("trav_slope_weight", 0.2, fp("Slope weight in combined cost", 0.0, 1.0))
        self.declare_parameter("trav_step_weight", 0.2, fp("Step weight in combined cost", 0.0, 1.0))
        self.declare_parameter("trav_roughness_weight", 0.6, fp("Roughness weight in combined cost", 0.0, 1.0))

        # Temporal filter
        self.declare_parameter("filter_enable", True, sp("Enable obstacle inflation + support-ratio + temporal gate"))
        self.declare_parameter("filter_support_radius_m", 0.5, fp("Neighborhood radius for support check (m)", 0.0, 10.0))
        self.declare_parameter("filter_support_ratio", 0.5, fp("Min fraction of measured cells to keep", 0.0, 1.0))
        self.declare_parameter("filter_inflation_sigma_m", 0.3, fp("Gaussian sigma for obstacle dilation (m)", 0.0, 10.0))
        self.declare_parameter("filter_obstacle_threshold", 0.8, fp("Cost above which a cell is an obstacle source", 0.0, 1.0))
        self.declare_parameter("filter_obstacle_growth_threshold", 2.0, fp("Reject frame if obstacle count grows by this factor", 1.0, 100.0))
        self.declare_parameter("filter_rejection_limit_frames", 5, ip("Force-accept after this many consecutive rejections", 1, 1000))
        self.declare_parameter("filter_min_obstacle_baseline", 10, ip("Skip hysteresis until this many obstacles seen", 0, 100_000))

        # Occlusion (line-of-sight) masking
        self.declare_parameter("occlusion_enable", False, sp("NaN-out cost in the line-of-sight shadow of obstacles"))
        self.declare_parameter("occlusion_sensor_x", 0.0, fp("Sensor x in the gravity-aligned grid frame (m)", -10.0, 10.0))
        self.declare_parameter("occlusion_sensor_y", 0.0, fp("Sensor y in the gravity-aligned grid frame (m)", -10.0, 10.0))
        self.declare_parameter("occlusion_sensor_z", 0.5, fp("Sensor height above the grid origin (m)", 0.0, 10.0))
        self.declare_parameter("occlusion_angle_eps_deg", 0.6, fp("View-angle margin guarding flat-ground noise (deg)", 0.0, 30.0))

        # Flat ground footprint
        self.declare_parameter("footprint_enable", False, sp("Force a flat ground patch under the robot"))
        self.declare_parameter("footprint_robot_height", 0.4, fp("Vertical distance robot frame → ground (m)", -5.0, 5.0))
        self.declare_parameter("footprint_half_x", 0.5, fp("Footprint half-extent along x (m)", 0.01, 10.0))
        self.declare_parameter("footprint_half_y", 0.5, fp("Footprint half-extent along y (m)", 0.01, 10.0))
        self.declare_parameter("footprint_center_x", 0.0, fp("Footprint center offset along x (m)", -10.0, 10.0))
        self.declare_parameter("footprint_center_y", 0.0, fp("Footprint center offset along y (m)", -10.0, 10.0))
        self.declare_parameter("footprint_mode", "overwrite", sp("Footprint fill mode: 'overwrite' | 'fill'"))

    def _read_parameters(self) -> dict:
        keys = [
            "lidar_topic", "map_frame", "robot_frame_ga", "robot_frame", "square_half_size", "device",
            "resolution", "x_range", "y_range",
            "z_max", "primary", "inpaint", "inpaint_coarse_iters", "inpaint_iters_per_level", "smooth_sigma",
            "outlier_enable", "outlier_type",
            "outlier_search_radius_m", "outlier_min_neighbors", "outlier_std_multiplier",
            "trav_enable",
            "trav_max_slope_deg", "trav_max_step_height_m", "trav_max_drop_height_m", "trav_max_roughness_m",
            "trav_step_window_radius_m", "trav_roughness_window_radius_m",
            "trav_slope_weight", "trav_step_weight", "trav_roughness_weight",
            "filter_enable",
            "filter_support_radius_m", "filter_support_ratio", "filter_inflation_sigma_m",
            "filter_obstacle_threshold", "filter_obstacle_growth_threshold",
            "filter_rejection_limit_frames", "filter_min_obstacle_baseline",
            "occlusion_enable", "occlusion_sensor_x", "occlusion_sensor_y", "occlusion_sensor_z",
            "occlusion_angle_eps_deg",
            "footprint_enable", "footprint_robot_height",
            "footprint_half_x", "footprint_half_y",
            "footprint_center_x", "footprint_center_y", "footprint_mode",
        ]
        return {k: self.get_parameter(k).value for k in keys}

    def _log_config(self, p: dict) -> None:
        groups: list[tuple[str, list[str]]] = [
            ("ROS / sensor",   ["lidar_topic", "map_frame", "robot_frame_ga", "robot_frame", "square_half_size", "device"]),
            ("Grid",           ["resolution", "x_range", "y_range"]),
            ("Pipeline",       ["z_max", "primary", "inpaint", "inpaint_coarse_iters",
                                "inpaint_iters_per_level", "smooth_sigma"]),
            ("Outlier",        ["outlier_enable", "outlier_type", "outlier_search_radius_m",
                                "outlier_min_neighbors", "outlier_std_multiplier"]),
            ("Traversability", ["trav_enable", "trav_max_slope_deg", "trav_max_step_height_m",
                                "trav_max_drop_height_m", "trav_max_roughness_m",
                                "trav_step_window_radius_m", "trav_roughness_window_radius_m",
                                "trav_slope_weight", "trav_step_weight", "trav_roughness_weight"]),
            ("Temporal filter",["filter_enable", "filter_support_radius_m", "filter_support_ratio",
                                "filter_inflation_sigma_m", "filter_obstacle_threshold",
                                "filter_obstacle_growth_threshold", "filter_rejection_limit_frames",
                                "filter_min_obstacle_baseline"]),
            ("Occlusion",      ["occlusion_enable", "occlusion_sensor_x", "occlusion_sensor_y",
                                "occlusion_sensor_z", "occlusion_angle_eps_deg"]),
            ("Footprint",      ["footprint_enable", "footprint_robot_height",
                                "footprint_half_x", "footprint_half_y",
                                "footprint_center_x", "footprint_center_y", "footprint_mode"]),
        ]
        lines = ["TerrainToolkitNode configuration:"]
        for title, keys in groups:
            lines.append(f"  [{title}]")
            width = max(len(k) for k in keys)
            for k in keys:
                lines.append(f"    {k:<{width}} = {p[k]!r}")
        self.get_logger().info("\n".join(lines))

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self, p: dict) -> None:
        outlier_cfg: OutlierFilterConfig | RadiusOutlierFilterConfig | None = None
        if p["outlier_enable"]:
            kind = p["outlier_type"].lower()
            if kind == "ror":
                outlier_cfg = RadiusOutlierFilterConfig(
                    search_radius_m=p["outlier_search_radius_m"],
                    min_neighbors=p["outlier_min_neighbors"],
                )
            elif kind == "sor":
                outlier_cfg = OutlierFilterConfig(
                    search_radius_m=p["outlier_search_radius_m"],
                    min_neighbors=p["outlier_min_neighbors"],
                    std_multiplier=p["outlier_std_multiplier"],
                )
            else:
                raise ValueError(f"outlier_type must be 'ror' or 'sor'; got {kind!r}")

        traversability_cfg: TraversabilityConfig | None = None
        if p["trav_enable"]:
            traversability_cfg = TraversabilityConfig(
                max_slope_deg=p["trav_max_slope_deg"],
                max_step_height_m=p["trav_max_step_height_m"],
                max_drop_height_m=p["trav_max_drop_height_m"],
                max_roughness_m=p["trav_max_roughness_m"],
                step_window_radius_m=p["trav_step_window_radius_m"],
                roughness_window_radius_m=p["trav_roughness_window_radius_m"],
                slope_weight=p["trav_slope_weight"],
                step_weight=p["trav_step_weight"],
                roughness_weight=p["trav_roughness_weight"],
            )

        filter_cfg: FilterConfig | None = None
        if p["filter_enable"] and traversability_cfg is not None:
            filter_cfg = FilterConfig(
                support_radius_m=p["filter_support_radius_m"],
                support_ratio=p["filter_support_ratio"],
                inflation_sigma_m=p["filter_inflation_sigma_m"],
                obstacle_threshold=p["filter_obstacle_threshold"],
                obstacle_growth_threshold=p["filter_obstacle_growth_threshold"],
                rejection_limit_frames=p["filter_rejection_limit_frames"],
                min_obstacle_baseline=p["filter_min_obstacle_baseline"],
            )

        occlusion_cfg: OcclusionConfig | None = None
        if p["occlusion_enable"] and traversability_cfg is not None:
            occlusion_cfg = OcclusionConfig(
                sensor_xy=(p["occlusion_sensor_x"], p["occlusion_sensor_y"]),
                sensor_z=p["occlusion_sensor_z"],
                angle_eps_rad=float(np.deg2rad(p["occlusion_angle_eps_deg"])),
            )

        footprint_cfg: FootprintConfig | None = None
        if p["footprint_enable"]:
            footprint_cfg = FootprintConfig(
                half_x=p["footprint_half_x"],
                half_y=p["footprint_half_y"],
                center=(p["footprint_center_x"], p["footprint_center_y"]),
                ground_z=-p["footprint_robot_height"],  # level fallback if no plane
                mode=p["footprint_mode"],
            )
        # Cached for the per-frame plane computation in the callback.
        self.footprint_enable: bool = bool(p["footprint_enable"])
        self.footprint_robot_height: float = float(p["footprint_robot_height"])

        # 'auto' picks CUDA if available, else CPU. Explicit "cpu" / "cuda:N"
        # passes straight to Warp.
        device_param = p["device"]
        if device_param == "auto":
            import warp as wp
            device_arg = "cuda:0" if wp.is_cuda_available() else "cpu"
        else:
            device_arg = device_param

        self.pipe = TerrainPipeline(
            resolution=p["resolution"],
            bounds=(-p["x_range"], p["x_range"], -p["y_range"], p["y_range"]),
            z_max=p["z_max"],
            primary=p["primary"],
            inpaint=p["inpaint"],
            inpaint_iters_per_level=p["inpaint_iters_per_level"],
            inpaint_coarse_iters=p["inpaint_coarse_iters"],
            smooth_sigma=p["smooth_sigma"],
            outlier=outlier_cfg,
            traversability=traversability_cfg,
            filter=filter_cfg,
            occlusion=occlusion_cfg,
            footprint=footprint_cfg,
            device=device_arg,
        )

    # ------------------------------------------------------------------
    # Dynamic reconfigure
    # ------------------------------------------------------------------

    def _on_parameters_changed(self, params) -> SetParametersResult:
        new_values = {param.name: param.value for param in params}
        merged = self._read_parameters()
        merged.update(new_values)

        for attr in ("map_frame", "robot_frame_ga", "robot_frame", "resolution", "x_range", "y_range", "square_half_size"):
            if attr in new_values:
                setattr(self, attr, new_values[attr])

        if "lidar_topic" in new_values and new_values["lidar_topic"] != self.lidar_topic:
            self.destroy_subscription(self.sub)
            self.lidar_topic = new_values["lidar_topic"]
            self.sub = self.create_subscription(
                PointCloud2, self.lidar_topic, self._cloud_callback, 10,
            )
            self.get_logger().info(f"Resubscribed to {self.lidar_topic}")

        if _PIPELINE_PARAMS & new_values.keys():
            try:
                self._build_pipeline(merged)
                self.get_logger().info("TerrainPipeline rebuilt with new parameters.")
            except Exception as exc:
                return SetParametersResult(successful=False, reason=str(exc))

        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def _cloud_callback(self, msg: PointCloud2) -> None:
        source_frame = msg.header.frame_id
        stamp = msg.header.stamp

        try:
            self.tf_buffer.lookup_transform(
                self.robot_frame_ga, source_frame, stamp,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup failed: {exc}")
            return

        points_xyz = self._transform_pointcloud_xyz(msg, self.robot_frame_ga, self.tf_buffer)
        if points_xyz is None or points_xyz.shape[0] == 0:
            self.get_logger().warn("Received empty / invalid point cloud — skipping.")
            return

        footprint_plane = self._footprint_plane(stamp)

        try:
            terrain_map: TerrainMap = self.pipe.process(
                points_xyz, footprint_plane=footprint_plane,
            )
        except Exception as exc:
            self.get_logger().error(f"terrain_toolkit error: {exc}")
            return

        out_cloud = self._grid_to_cloud(
            terrain_map=terrain_map,
            x_min=-self.x_range,
            y_min=-self.y_range,
            resolution=self.resolution,
            stamp=stamp,
        )
        if out_cloud is not None:
            self.pub.publish(out_cloud)

    # ------------------------------------------------------------------
    # Flat-footprint ground plane
    # ------------------------------------------------------------------

    def _footprint_plane(self, stamp) -> tuple[float, float, float] | None:
        """Ground plane `z = a*x + b*y + c` in the gravity-aligned grid frame.

        The robot footprint is flat (z = -robot_height) in the *normal* robot
        body frame. Expressed in the gravity-aligned grid frame it tilts with the
        robot's roll/pitch, so we look up robot_frame_ga ← robot_frame and project
        that plane. Returns None when the feature is off or TF is unavailable
        (the pipeline then falls back to its level default).
        """
        if not self.footprint_enable:
            return None

        try:
            tf = self.tf_buffer.lookup_transform(
                self.robot_frame_ga, self.robot_frame, stamp,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warn(f"footprint TF lookup failed: {exc}")
            return None

        r = tf.transform.rotation
        t = tf.transform.translation
        # Third column of R_{ga←robot}: the robot body z-axis in the grid frame.
        R = _quaternion_to_matrix(r.x, r.y, r.z, r.w)
        r3 = R[:3, 2]
        rz = float(r3[2])
        if abs(rz) < 1e-6:
            self.get_logger().warn("footprint plane near-vertical — skipping flat patch.")
            return None

        h = self.footprint_robot_height
        r3_dot_t = float(r3[0] * t.x + r3[1] * t.y + r3[2] * t.z)
        a = -float(r3[0]) / rz
        b = -float(r3[1]) / rz
        c = (-h + r3_dot_t) / rz
        return (a, b, c)

    # ------------------------------------------------------------------
    # PointCloud2 → numpy
    # ------------------------------------------------------------------

    def pointcloud2_to_xyz_array(self, msg: PointCloud2) -> np.ndarray:
        pc = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True,
                             reshape_organized_cloud=False)
        if isinstance(pc, np.ndarray) and pc.dtype.names is not None:
            xyz = np.stack([pc["x"], pc["y"], pc["z"]], axis=-1)
        else:
            xyz = np.array(list(pc), dtype=np.float32)
        return xyz.astype(np.float32)

    # ------------------------------------------------------------------
    # Grid → PointCloud2
    # ------------------------------------------------------------------

    def _grid_to_cloud(
        self,
        terrain_map: TerrainMap,
        x_min: float,
        y_min: float,
        resolution: float,
        stamp,
    ) -> PointCloud2 | None:
        if terrain_map.elevation is None:
            self.get_logger().warn("TerrainMap.elevation is None — skipping publish.")
            return None

        rows, cols = terrain_map.elevation.shape  # (ny, nx)

        row_idx = np.arange(rows, dtype=np.float32)
        col_idx = np.arange(cols, dtype=np.float32)
        row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing="ij")

        x_coords = (x_min + (col_grid + 0.5) * resolution).astype(np.float32)
        y_coords = (y_min + (row_grid + 0.5) * resolution).astype(np.float32)

        # as_dict() already skips layers that were not downloaded (None).
        layer_dict = terrain_map.as_dict()
        layer_names = sorted(layer_dict.keys())

        # Drop cells the SupportRatioMask flagged as too far from any real
        # measurement: those have NaN traversability (and NaN slope/step/roughness)
        # even though inpaint filled their elevation. Publishing them would make
        # the heightmap look complete in regions where we actually have no data.
        # When the filter chain is disabled (no traversability layer at all),
        # fall back to elevation finiteness — there's no support signal to use.
        valid = np.isfinite(terrain_map.elevation)
        if terrain_map.traversability is not None:
            valid &= np.isfinite(terrain_map.traversability)

        x_valid = x_coords[valid]
        y_valid = y_coords[valid]
        z_valid = terrain_map.elevation[valid].astype(np.float32)
        layers_valid = [layer_dict[k][valid].astype(np.float32) for k in layer_names]

        n_pts = x_valid.shape[0]
        point_data = np.column_stack([x_valid, y_valid, z_valid] + layers_valid)

        fields: list[PointField] = []
        offset = 0
        for name in ("x", "y", "z"):
            fields.append(PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1))
            offset += 4
        for name in layer_names:
            fields.append(PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1))
            offset += 4

        header = Header()
        header.stamp = stamp
        header.frame_id = self.robot_frame_ga

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = n_pts
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = offset
        cloud_msg.row_step = offset * n_pts
        cloud_msg.is_dense = False
        cloud_msg.data = point_data.astype(np.float32).tobytes()
        return cloud_msg

    # ------------------------------------------------------------------
    # TF transform
    # ------------------------------------------------------------------

    def _transform_pointcloud_xyz(
        self,
        cloud_msg: PointCloud2,
        target_frame: str,
        tf_buffer: tf2_ros.Buffer,
    ) -> np.ndarray | None:
        try:
            transform: TransformStamped = tf_buffer.lookup_transform(
                target_frame, cloud_msg.header.frame_id, cloud_msg.header.stamp,
            )
        except TransformException as exc:
            self.get_logger().error(f"TF lookup failed in transform: {exc}")
            return None

        points = self.pointcloud2_to_xyz_array(cloud_msg)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        t = transform.transform.translation
        r = transform.transform.rotation
        T = _quaternion_to_matrix(r.x, r.y, r.z, r.w)
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        return (T @ np.hstack((points, ones)).T).T[:, :3]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TerrainToolkitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
