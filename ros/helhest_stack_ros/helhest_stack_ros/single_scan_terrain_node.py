#!/usr/bin/env python3
"""
ROS 2 Kilted interface node for the helhest.perception library.

Subscribes to a LiDAR PointCloud2, transforms it into the robot frame, runs the
helhest.perception pipeline, and republishes the resulting grid as a PointCloud2
with one float32 PointField per TerrainMap layer.
"""

from __future__ import annotations

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import FloatingPointRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from helhest.perception import TerrainMap
from tf2_ros import TransformException

from ._pipeline_common import build_pipeline
from ._pipeline_common import declare_pipeline_parameters
from ._pipeline_common import grid_to_cloud
from ._pipeline_common import PIPELINE_PARAMS
from ._pipeline_common import pointcloud2_to_xyz_array
from ._pipeline_common import quaternion_to_matrix
from ._pipeline_common import read_pipeline_parameters


class SingleScanTerrainNode(Node):
    """Bridge a LiDAR PointCloud2 topic to the helhest.perception pipeline."""

    def __init__(self) -> None:
        super().__init__("single_scan_terrain")

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
            PointCloud2,
            self.lidar_topic,
            self._cloud_callback,
            10,
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

        def sp(desc: str) -> ParameterDescriptor:
            return ParameterDescriptor(description=desc)

        # ROS / sensor
        self.declare_parameter("lidar_topic", "/lidar/points", sp("PointCloud2 input topic"))
        self.declare_parameter("map_frame", "map", sp("Map TF frame (unused)"))
        self.declare_parameter(
            "robot_frame_ga",
            "base_link",
            sp(
                "Gravity-aligned robot TF frame the heightmap is built in "
                "(use a real gravity-aligned frame on non-flat terrain)"
            ),
        )
        self.declare_parameter(
            "robot_frame",
            "base_link",
            sp("Normal (un-leveled) robot body TF frame; used for the flat-footprint plane"),
        )
        self.declare_parameter(
            "square_half_size", 10.0, fp("Half-side of square ROI (m)", 0.5, 200.0)
        )

        # Compute device + grid + pipeline stage parameters (shared with the
        # accumulator node).
        declare_pipeline_parameters(self)

    def _read_parameters(self) -> dict:
        node_keys = [
            "lidar_topic",
            "map_frame",
            "robot_frame_ga",
            "robot_frame",
            "square_half_size",
        ]
        p = {k: self.get_parameter(k).value for k in node_keys}
        p.update(read_pipeline_parameters(self))
        return p

    def _log_config(self, p: dict) -> None:
        groups: list[tuple[str, list[str]]] = [
            (
                "ROS / sensor",
                [
                    "lidar_topic",
                    "map_frame",
                    "robot_frame_ga",
                    "robot_frame",
                    "square_half_size",
                    "device",
                ],
            ),
            ("Grid", ["resolution", "x_range", "y_range"]),
            (
                "Pipeline",
                [
                    "z_max",
                    "primary",
                    "inpaint",
                    "inpaint_coarse_iters",
                    "inpaint_iters_per_level",
                    "smooth_sigma",
                ],
            ),
            (
                "Outlier",
                [
                    "outlier_enable",
                    "outlier_type",
                    "outlier_search_radius_m",
                    "outlier_min_neighbors",
                    "outlier_std_multiplier",
                ],
            ),
            (
                "Traversability",
                [
                    "trav_enable",
                    "trav_max_slope_deg",
                    "trav_max_step_height_m",
                    "trav_max_drop_height_m",
                    "trav_max_roughness_m",
                    "trav_step_window_radius_m",
                    "trav_roughness_window_radius_m",
                    "trav_slope_weight",
                    "trav_step_weight",
                    "trav_roughness_weight",
                ],
            ),
            (
                "Temporal filter",
                [
                    "filter_enable",
                    "filter_support_radius_m",
                    "filter_support_ratio",
                    "filter_inflation_sigma_m",
                    "filter_obstacle_threshold",
                    "filter_obstacle_growth_threshold",
                    "filter_rejection_limit_frames",
                    "filter_min_obstacle_baseline",
                ],
            ),
            (
                "Occlusion",
                [
                    "occlusion_enable",
                    "occlusion_sensor_x",
                    "occlusion_sensor_y",
                    "occlusion_sensor_z",
                    "occlusion_angle_eps_deg",
                ],
            ),
            (
                "Footprint",
                [
                    "footprint_enable",
                    "footprint_robot_height",
                    "footprint_half_x",
                    "footprint_half_y",
                    "footprint_center_x",
                    "footprint_center_y",
                    "footprint_mode",
                ],
            ),
        ]
        lines = ["SingleScanTerrainNode configuration:"]
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
        self.pipe = build_pipeline(p)
        # Cached for the per-frame plane computation in the callback.
        self.footprint_enable: bool = bool(p["footprint_enable"])
        self.footprint_robot_height: float = float(p["footprint_robot_height"])

    # ------------------------------------------------------------------
    # Dynamic reconfigure
    # ------------------------------------------------------------------

    def _on_parameters_changed(self, params) -> SetParametersResult:
        new_values = {param.name: param.value for param in params}
        merged = self._read_parameters()
        merged.update(new_values)

        for attr in (
            "map_frame",
            "robot_frame_ga",
            "robot_frame",
            "resolution",
            "x_range",
            "y_range",
            "square_half_size",
        ):
            if attr in new_values:
                setattr(self, attr, new_values[attr])

        if "lidar_topic" in new_values and new_values["lidar_topic"] != self.lidar_topic:
            self.destroy_subscription(self.sub)
            self.lidar_topic = new_values["lidar_topic"]
            self.sub = self.create_subscription(
                PointCloud2,
                self.lidar_topic,
                self._cloud_callback,
                10,
            )
            self.get_logger().info(f"Resubscribed to {self.lidar_topic}")

        if PIPELINE_PARAMS & new_values.keys():
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
                self.robot_frame_ga,
                source_frame,
                stamp,
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
                points_xyz,
                footprint_plane=footprint_plane,
            )
        except Exception as exc:
            self.get_logger().error(f"helhest.perception error: {exc}")
            return

        out_cloud = grid_to_cloud(
            terrain_map=terrain_map,
            x_min=-self.x_range,
            y_min=-self.y_range,
            resolution=self.resolution,
            stamp=stamp,
            frame_id=self.robot_frame_ga,
            logger=self.get_logger(),
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
                self.robot_frame_ga,
                self.robot_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warn(f"footprint TF lookup failed: {exc}")
            return None

        r = tf.transform.rotation
        t = tf.transform.translation
        # Third column of R_{ga←robot}: the robot body z-axis in the grid frame.
        R = quaternion_to_matrix(r.x, r.y, r.z, r.w)
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
                target_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
            )
        except TransformException as exc:
            self.get_logger().error(f"TF lookup failed in transform: {exc}")
            return None

        points = pointcloud2_to_xyz_array(cloud_msg)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        t = transform.transform.translation
        r = transform.transform.rotation
        T = quaternion_to_matrix(r.x, r.y, r.z, r.w)
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        return (T @ np.hstack((points, ones)).T).T[:, :3]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SingleScanTerrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
