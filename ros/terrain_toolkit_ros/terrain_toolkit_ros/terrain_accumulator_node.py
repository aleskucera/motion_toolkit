#!/usr/bin/env python3
"""Accumulating terrain mapper for ROS 2 Kilted.

Fuses a LiDAR PointCloud2 stream with robot odometry (nav_msgs/Odometry) into a
persistent point cloud, then runs the terrain_toolkit pipeline over a
robot-centric window of it. Each scan is registered scan-to-submap with
point-to-plane ICP, using the odometry frame-to-frame delta as the initial
guess; the refined trajectory corrects odom drift. The published terrain_map
therefore covers more than a single scan (it retains terrain already passed or
hidden behind obstacles).

Frames: sensor (lidar) → base (base_frame, via static TF) → odom (Odometry
frame) → world ≡ map_frame (accumulation frame). The world frame is bootstrapped
to odom at the first scan, so it is gravity-aligned given a gravity-up odom
z-axis and the grid needs no per-scan leveling.

Limitation: scan-to-submap ICP can inject small roll/pitch corrections that,
accumulated over a very long run, slowly tilt the world frame off gravity. This
is negligible for the bounded robot-centric window but is why the map is not a
substitute for a full SLAM back-end.
"""

from __future__ import annotations

import numpy as np
import rclpy
import tf2_ros
import warp as wp
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import FloatingPointRange
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from terrain_toolkit import DynamicFilterConfig
from terrain_toolkit import DynamicPointFilter
from terrain_toolkit import IcpAligner
from terrain_toolkit import IcpConfig
from terrain_toolkit import TerrainMap
from terrain_toolkit import VoxelGrid
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformException

from ._mapping_math import crop_box
from ._mapping_math import deskew_scan
from ._mapping_math import invert_pose
from ._mapping_math import matrix_to_quaternion
from ._mapping_math import odom_delta
from ._mapping_math import pose_correction_magnitude
from ._mapping_math import transform_points_xyz
from ._pipeline_common import build_pipeline
from ._pipeline_common import declare_pipeline_parameters
from ._pipeline_common import grid_to_cloud
from ._pipeline_common import PIPELINE_PARAMS
from ._pipeline_common import pointcloud2_to_xyz_time_array
from ._pipeline_common import quaternion_to_matrix
from ._pipeline_common import read_pipeline_parameters

# ICP construction-time parameters: a change to any of these rebuilds the aligner.
# (The runtime gate params — min inliers, max correction — are read live.)
_ICP_BUILD_PARAMS = frozenset(
    {
        "icp_max_iters",
        "icp_max_corr_dist_m",
        "icp_normal_radius_m",
        "icp_voxel_size_m",
        "icp_voxel_target",
    }
)

# Dynamic-filter construction-time parameters: a change rebuilds the filter.
_DYNAMIC_BUILD_PARAMS = frozenset(
    {
        "dynamic_az_bins",
        "dynamic_el_bins",
        "dynamic_el_min_deg",
        "dynamic_el_max_deg",
        "dynamic_margin_m",
        "dynamic_margin_rel",
        "dynamic_min_range_m",
    }
)

# Node-level params that are cached on attributes and read each callback.
_NODE_PARAM_KEYS: tuple[str, ...] = (
    "base_frame",
    "map_frame",
    "accumulation_voxel_m",
    "map_max_radius_m",
    "icp_enable",
    "icp_submap_radius_m",
    "icp_min_inliers",
    "icp_max_corr_trans_m",
    "icp_max_corr_rot_deg",
    "icp_min_submap_points",
    "publish_map_tf",
    "dynamic_enable",
    "deskew_enable",
    "deskew_time_field",
)


class TerrainAccumulatorNode(Node):
    """Build a larger terrain map by fusing LiDAR scans with odometry via ICP."""

    def __init__(self) -> None:
        super().__init__("terrain_accumulator")

        self._declare_parameters()
        p = self._all_params()
        self._cache_node_params(p)

        self.lidar_topic: str = self.get_parameter("lidar_topic").value
        self.odom_topic: str = self.get_parameter("odom_topic").value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Persistent map state (all in the world / map_frame). Bootstrapped on
        # the first scan; see _process.
        self.global_cloud: np.ndarray | None = None
        self._map_voxel: VoxelGrid | None = None  # lazy; sizes to the map on first use
        self.odom_T_base_prev: np.ndarray | None = None
        self.world_T_base_prev: np.ndarray | None = None
        self._deskew_warned = False  # warn once if deskew is on but the cloud lacks times

        self._build_pipeline(p)
        self._build_aligner(p)
        self._build_dynamic_filter(p)

        sync_queue: int = self.get_parameter("sync_queue").value
        sync_slop: float = self.get_parameter("sync_slop_s").value
        self.cloud_sub = Subscriber(self, PointCloud2, self.lidar_topic)
        self.odom_sub = Subscriber(self, Odometry, self.odom_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.cloud_sub, self.odom_sub], queue_size=sync_queue, slop=sync_slop
        )
        self.sync.registerCallback(self._synced_callback)

        self.pub = self.create_publisher(PointCloud2, "terrain_map", 10)
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self.get_logger().info(
            f"TerrainAccumulatorNode: cloud={self.lidar_topic} odom={self.odom_topic} "
            f"map_frame={self.map_frame} base_frame={self.base_frame} "
            f"icp={'on' if self.icp_enable else 'off'} device={self.pipe.device}"
        )

    # ------------------------------------------------------------------
    # Parameters
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

        # ROS / sensors
        self.declare_parameter("lidar_topic", "/lidar/points", sp("PointCloud2 input topic"))
        self.declare_parameter("odom_topic", "/odom", sp("nav_msgs/Odometry input topic"))
        self.declare_parameter(
            "base_frame",
            "base_link",
            sp("Robot body TF frame; must match the odometry child_frame_id"),
        )
        self.declare_parameter("map_frame", "map", sp("World frame the accumulated map lives in"))
        self.declare_parameter(
            "sync_slop_s", 0.05, fp("Cloud/odom time-sync tolerance (s)", 0.0, 1.0)
        )
        self.declare_parameter("sync_queue", 30, ip("Cloud/odom sync queue size", 1, 1000))

        # Scan deskew (motion compensation over the sweep)
        self.declare_parameter(
            "deskew_enable",
            True,
            sp("Motion-compensate each sweep using the odom delta before ICP/accumulate"),
        )
        self.declare_parameter(
            "deskew_time_field",
            "t",
            sp("Per-point time field for deskew (Ouster organized clouds use 't')"),
        )

        # Accumulation / map
        self.declare_parameter(
            "accumulation_voxel_m", 0.10, fp("Global cloud voxel-downsample size (m)", 0.01, 5.0)
        )
        self.declare_parameter(
            "map_max_radius_m",
            50.0,
            fp("Drop accumulated points beyond this distance from the robot (m)", 1.0, 500.0),
        )

        # ICP (scan-to-submap)
        self.declare_parameter(
            "icp_enable", True, sp("Refine odom with ICP (false = odom dead-reckoning)")
        )
        self.declare_parameter(
            "icp_submap_radius_m", 15.0, fp("Half-extent of the ICP target submap (m)", 1.0, 200.0)
        )
        self.declare_parameter("icp_max_iters", 30, ip("Max ICP iterations", 1, 200))
        self.declare_parameter(
            "icp_max_corr_dist_m", 0.5, fp("ICP max correspondence distance (m)", 0.01, 10.0)
        )
        self.declare_parameter(
            "icp_normal_radius_m", 0.3, fp("ICP target-normal estimation radius (m)", 0.01, 5.0)
        )
        self.declare_parameter(
            "icp_voxel_size_m", 0.1, fp("ICP source/target voxel size (0 = off) (m)", 0.0, 5.0)
        )
        self.declare_parameter(
            "icp_voxel_target", True, sp("Also voxel-downsample the ICP target submap")
        )

        # ICP divergence gate (reject a bad alignment, fall back to odom prediction)
        self.declare_parameter(
            "icp_min_inliers", 500, ip("Reject ICP below this inlier count", 0, 10_000_000)
        )
        self.declare_parameter(
            "icp_max_corr_trans_m",
            1.0,
            fp("Reject ICP correcting the prediction more than this (m)", 0.0, 50.0),
        )
        self.declare_parameter(
            "icp_max_corr_rot_deg",
            15.0,
            fp("Reject ICP correcting the prediction more than this (deg)", 0.0, 180.0),
        )
        self.declare_parameter(
            "icp_min_submap_points", 2000, ip("Skip ICP if the submap has fewer points", 0, 10**7)
        )

        # Dynamic obstacle filter (map-frame visibility; removes moving people)
        self.declare_parameter(
            "dynamic_enable",
            False,
            sp("Drop moving objects by map-frame visibility before accumulation"),
        )
        self.declare_parameter(
            "dynamic_az_bins", 900, ip("Range-image azimuth bins (over 360°)", 16, 4096)
        )
        self.declare_parameter(
            "dynamic_el_bins", 64, ip("Range-image elevation bins (over the vertical FOV)", 4, 2048)
        )
        self.declare_parameter(
            "dynamic_el_min_deg", -25.0, fp("Bottom of the sensor vertical FOV (deg)", -90.0, 0.0)
        )
        self.declare_parameter(
            "dynamic_el_max_deg", 25.0, fp("Top of the sensor vertical FOV (deg)", 0.0, 90.0)
        )
        self.declare_parameter(
            "dynamic_margin_m", 0.3, fp("Base depth margin for the visibility test (m)", 0.0, 5.0)
        )
        self.declare_parameter(
            "dynamic_margin_rel",
            0.02,
            fp("Range-proportional depth margin (m per m)", 0.0, 1.0),
        )
        self.declare_parameter(
            "dynamic_min_range_m", 0.5, fp("Ignore returns closer than this (m)", 0.0, 10.0)
        )

        # Viz
        self.declare_parameter(
            "publish_map_tf", True, sp("Broadcast the map→odom correction transform")
        )

        # Compute device + grid + pipeline stage parameters (shared with the
        # single-frame node). x_range/y_range are the robot-centric window
        # half-extents (= the pipeline grid bounds).
        declare_pipeline_parameters(self)

    def _all_params(self) -> dict:
        """Current value of every parameter the rebuild/cache paths consume.

        Read via get_parameter (valid at construction and outside the set
        callback). During dynamic reconfigure the callback overlays the
        pending new values on top of this, since get_parameter still returns
        the *old* values until the set commits.
        """
        p = read_pipeline_parameters(self)  # device + grid + pipeline stages
        for k in _ICP_BUILD_PARAMS | _DYNAMIC_BUILD_PARAMS:
            p[k] = self.get_parameter(k).value
        for k in _NODE_PARAM_KEYS:
            p[k] = self.get_parameter(k).value
        return p

    def _cache_node_params(self, p: dict) -> None:
        self.base_frame: str = p["base_frame"]
        self.map_frame: str = p["map_frame"]
        self.accumulation_voxel_m: float = p["accumulation_voxel_m"]
        self.map_max_radius_m: float = p["map_max_radius_m"]
        self.icp_enable: bool = p["icp_enable"]
        self.icp_submap_radius_m: float = p["icp_submap_radius_m"]
        self.icp_min_inliers: int = p["icp_min_inliers"]
        self.icp_max_corr_trans_m: float = p["icp_max_corr_trans_m"]
        self.icp_max_corr_rot_rad: float = float(np.deg2rad(p["icp_max_corr_rot_deg"]))
        self.icp_min_submap_points: int = p["icp_min_submap_points"]
        self.publish_map_tf: bool = p["publish_map_tf"]
        self.deskew_enable: bool = p["deskew_enable"]
        self.deskew_time_field: str = p["deskew_time_field"]
        # Window half-extents double as the pipeline grid bounds.
        self.x_range: float = p["x_range"]
        self.y_range: float = p["y_range"]
        self.resolution: float = p["resolution"]

    # ------------------------------------------------------------------
    # Construction of heavy objects
    # ------------------------------------------------------------------

    def _build_pipeline(self, p: dict) -> None:
        self.pipe = build_pipeline(p)

    def _build_aligner(self, p: dict) -> None:
        cfg = IcpConfig(
            max_iters=p["icp_max_iters"],
            max_correspondence_dist_m=p["icp_max_corr_dist_m"],
            normal_radius_m=p["icp_normal_radius_m"],
            voxel_size_m=p["icp_voxel_size_m"] or None,
            voxel_target=p["icp_voxel_target"],
        )
        # Share the pipeline's resolved Warp device so both run on the same GPU/CPU.
        self.aligner = IcpAligner(cfg, device=self.pipe.device)

    def _build_dynamic_filter(self, p: dict) -> None:
        cfg = DynamicFilterConfig(
            az_bins=p["dynamic_az_bins"],
            el_bins=p["dynamic_el_bins"],
            el_min_deg=p["dynamic_el_min_deg"],
            el_max_deg=p["dynamic_el_max_deg"],
            margin_m=p["dynamic_margin_m"],
            margin_rel=p["dynamic_margin_rel"],
            min_range_m=p["dynamic_min_range_m"],
        )
        self.dynamic_filter = DynamicPointFilter(cfg, device=self.pipe.device)

    # ------------------------------------------------------------------
    # Dynamic reconfigure
    # ------------------------------------------------------------------

    def _on_parameters_changed(self, params) -> SetParametersResult:
        new_values = {param.name: param.value for param in params}
        # get_parameter still returns the OLD values inside this callback, so
        # overlay the pending values to get the effective config.
        merged = self._all_params()
        merged.update({k: v for k, v in new_values.items() if k in merged})

        try:
            if PIPELINE_PARAMS & new_values.keys():
                self._build_pipeline(merged)
                # The pipeline owns the resolved device; rebuild the aligner and
                # dynamic filter so they follow a device change.
                self._build_aligner(merged)
                self._build_dynamic_filter(merged)
            else:
                if _ICP_BUILD_PARAMS & new_values.keys():
                    self._build_aligner(merged)
                if _DYNAMIC_BUILD_PARAMS & new_values.keys():
                    self._build_dynamic_filter(merged)
        except Exception as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        self._cache_node_params(merged)
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def _synced_callback(self, cloud_msg: PointCloud2, odom_msg: Odometry) -> None:
        try:
            self._process(cloud_msg, odom_msg)
        except Exception as exc:
            self.get_logger().error(f"accumulator error: {exc}")

    def _process(self, cloud_msg: PointCloud2, odom_msg: Odometry) -> None:
        odom_T_base_curr = self._odom_to_matrix(odom_msg)

        scan = self._scan_in_base(cloud_msg)
        if scan is None or scan[0].shape[0] == 0:
            self.get_logger().warn("Empty / untransformable scan — skipping.")
            return
        scan_base, base_T_sensor, point_times = scan

        # First scan: bootstrap world ≡ odom, seed the map, publish, return.
        # (No odom delta yet, so this one sweep is left un-deskewed — negligible.)
        if self.odom_T_base_prev is None:
            world_T_base = odom_T_base_curr
            world_pts = transform_points_xyz(world_T_base, scan_base)
            self.global_cloud = self._downsample_map(world_pts)
            self._publish(world_T_base, cloud_msg.header.stamp)
            self._broadcast_map_tf(world_T_base, odom_msg)
            self.odom_T_base_prev = odom_T_base_curr
            self.world_T_base_prev = world_T_base
            return

        # Predict the new pose from the odom delta on top of the corrected pose.
        delta = odom_delta(self.odom_T_base_prev, odom_T_base_curr)
        world_T_base_pred = self.world_T_base_prev @ delta

        # Motion-compensate the sweep to its end instant before registering /
        # accumulating: the odom delta doubles as the constant-velocity sweep motion.
        if self.deskew_enable:
            scan_base = self._deskew(scan_base, point_times, delta)

        world_T_base = self._register(scan_base, world_T_base_pred)

        world_pts = transform_points_xyz(world_T_base, scan_base)

        # Drop moving objects (people) and carve their stale ghosts by map-frame
        # visibility, before they pollute the accumulated map.
        if self.dynamic_enable and self.global_cloud is not None:
            world_pts = self._filter_dynamic(world_pts, world_T_base, base_T_sensor)

        # Accumulate, bound to a radius around the robot, and downsample.
        merged = np.vstack((self.global_cloud, world_pts))
        merged = crop_box(merged, world_T_base[:3, 3], self.map_max_radius_m)
        self.global_cloud = self._downsample_map(merged)

        self._publish(world_T_base, cloud_msg.header.stamp)
        self._broadcast_map_tf(world_T_base, odom_msg)
        self.odom_T_base_prev = odom_T_base_curr
        self.world_T_base_prev = world_T_base

    # ------------------------------------------------------------------
    # Scan deskew (motion compensation over the sweep)
    # ------------------------------------------------------------------

    def _deskew(
        self, scan_base: np.ndarray, point_times: np.ndarray | None, delta: np.ndarray
    ) -> np.ndarray:
        """Motion-compensate the sweep to its end instant; pass through if unavailable.

        `delta` (base_prev→base_curr) is the sweep motion under the node's
        constant-velocity assumption (consecutive scans, cloud stamped at scan
        end — the Ouster default). Point times are normalized so the latest point
        maps to the reference (end) pose.
        """
        if point_times is None:
            if not self._deskew_warned:
                self.get_logger().warn(
                    f"deskew enabled but cloud has no '{self.deskew_time_field}' field "
                    "— skipping motion compensation."
                )
                self._deskew_warned = True
            return scan_base
        span = float(point_times.max() - point_times.min())
        if span <= 0.0:  # single-instant cloud (or a constant field) — nothing to correct
            return scan_base
        alphas = (point_times - point_times.min()) / span
        return deskew_scan(scan_base, alphas, delta).astype(np.float32)

    def _downsample_map(self, points: np.ndarray) -> np.ndarray:
        """Voxel-downsample the accumulated map via the device-native VoxelGrid.

        The map still lives on the host here, so this uploads/reads back at the
        boundary. Keeping the map on device (DeviceMapAccumulator) would remove
        the round trip entirely — see the accumulator device-path follow-up.
        """
        if len(points) == 0:
            return points
        if self._map_voxel is None or self._map_voxel.max_points < len(points):
            self._map_voxel = VoxelGrid(
                self.accumulation_voxel_m,
                max_points=max(len(points), 400_000),
                device=self.pipe.device,
            )
        pw = wp.array(np.ascontiguousarray(points, np.float32), dtype=wp.vec3, device=self.pipe.device)
        centroids, n = self._map_voxel.downsample(pw, len(points))
        return centroids.numpy()[:n].astype(points.dtype, copy=False)

    # ------------------------------------------------------------------
    # ICP registration (scan-to-submap, with divergence gate)
    # ------------------------------------------------------------------

    def _register(self, scan_base: np.ndarray, world_T_base_pred: np.ndarray) -> np.ndarray:
        if not self.icp_enable:
            return world_T_base_pred

        submap = crop_box(self.global_cloud, world_T_base_pred[:3, 3], self.icp_submap_radius_m)
        if submap.shape[0] < self.icp_min_submap_points:
            self.get_logger().debug(
                f"submap too sparse ({submap.shape[0]} pts) — using odom prediction."
            )
            return world_T_base_pred

        dev = self.aligner.device
        result = self.aligner.align(
            wp.array(scan_base, dtype=wp.vec3, device=dev),
            wp.array(submap, dtype=wp.vec3, device=dev),
            init_pose=world_T_base_pred,
        )

        rot, trans = pose_correction_magnitude(world_T_base_pred, result.pose)
        if (
            result.converged
            and result.num_inliers >= self.icp_min_inliers
            and trans <= self.icp_max_corr_trans_m
            and rot <= self.icp_max_corr_rot_rad
        ):
            return result.pose

        self.get_logger().warn(
            f"ICP rejected (converged={result.converged} inliers={result.num_inliers} "
            f"Δrot={np.rad2deg(rot):.1f}° Δtrans={trans:.2f}m) — using odom prediction."
        )
        return world_T_base_pred

    # ------------------------------------------------------------------
    # Dynamic obstacle removal (map-frame visibility)
    # ------------------------------------------------------------------

    def _filter_dynamic(
        self,
        world_pts: np.ndarray,
        world_T_base: np.ndarray,
        base_T_sensor: np.ndarray,
    ) -> np.ndarray:
        """Drop dynamic scan points and carve stale map ghosts, return kept scan.

        The visibility test bins directions in the sensor frame, so it needs the
        sensor origin and orientation in the map frame: world_T_sensor =
        world_T_base @ base_T_sensor.
        """
        world_T_sensor = world_T_base @ base_T_sensor
        sensor_origin = world_T_sensor[:3, 3]
        sensor_R_world = world_T_sensor[:3, :3].T  # world→sensor, to align bins to beams

        scan_keep, map_keep = self.dynamic_filter.filter(
            self.global_cloud,
            world_pts,
            sensor_origin,
            sensor_rotation=sensor_R_world,
        )
        self.global_cloud = self.global_cloud[map_keep]
        return world_pts[scan_keep]

    # ------------------------------------------------------------------
    # Publish + TF
    # ------------------------------------------------------------------

    def _publish(self, world_T_base: np.ndarray, stamp) -> None:
        robot_t = world_T_base[:3, 3]
        window = crop_box(self.global_cloud, robot_t, (self.x_range, self.y_range))
        if window.shape[0] == 0:
            return

        # Shift the window by the FULL robot translation so the grid is built in a
        # robot-relative frame: this keeps the pipeline's z_max threshold
        # robot-relative (matching the single-frame node). The world height is
        # restored on publish via z_offset / the grid origin.
        window_local = window - robot_t

        terrain_map: TerrainMap = self.pipe.process(window_local)
        cloud = grid_to_cloud(
            terrain_map=terrain_map,
            x_min=float(robot_t[0]) - self.x_range,
            y_min=float(robot_t[1]) - self.y_range,
            resolution=self.resolution,
            stamp=stamp,
            frame_id=self.map_frame,
            z_offset=float(robot_t[2]),
            logger=self.get_logger(),
        )
        if cloud is not None:
            self.pub.publish(cloud)

    def _broadcast_map_tf(self, world_T_base: np.ndarray, odom_msg: Odometry) -> None:
        if not self.publish_map_tf:
            return
        # SLAM-style correction: map→odom = world_T_base @ inv(odom_T_base).
        odom_T_base = self._odom_to_matrix(odom_msg)
        map_T_odom = world_T_base @ invert_pose(odom_T_base)
        qx, qy, qz, qw = matrix_to_quaternion(map_T_odom)

        tf = TransformStamped()
        tf.header.stamp = odom_msg.header.stamp
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = odom_msg.header.frame_id
        tf.transform.translation.x = float(map_T_odom[0, 3])
        tf.transform.translation.y = float(map_T_odom[1, 3])
        tf.transform.translation.z = float(map_T_odom[2, 3])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

    # ------------------------------------------------------------------
    # Message → numpy
    # ------------------------------------------------------------------

    def _odom_to_matrix(self, odom_msg: Odometry) -> np.ndarray:
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        T = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        T[0, 3] = p.x
        T[1, 3] = p.y
        T[2, 3] = p.z
        return T

    def _scan_in_base(
        self, cloud_msg: PointCloud2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Transform the cloud into base_frame using the (static) sensor→base TF.

        Returns `(scan_base, base_T_sensor, point_times)`: the points in
        base_frame, the 4x4 base←sensor transform (its translation is the sensor
        origin in base, needed by the dynamic filter's visibility test), and the
        per-point sweep times (aligned to `scan_base`; None if the cloud has no
        time field), used by deskew. The static extrinsic preserves point order,
        so the times stay aligned through the transform.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warn(f"sensor→base TF lookup failed: {exc}")
            return None

        t = transform.transform.translation
        r = transform.transform.rotation
        base_T_sensor = quaternion_to_matrix(r.x, r.y, r.z, r.w)
        base_T_sensor[0, 3] = t.x
        base_T_sensor[1, 3] = t.y
        base_T_sensor[2, 3] = t.z

        points, point_times = pointcloud2_to_xyz_time_array(cloud_msg, self.deskew_time_field)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32), base_T_sensor, None

        scan_base = transform_points_xyz(base_T_sensor, points.astype(np.float64)).astype(
            np.float32
        )
        return scan_base, base_T_sensor, point_times


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TerrainAccumulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
