#!/usr/bin/env python3
"""Dual elevation mapper for ROS 2 Kilted — the tuning front-end for the planner.

Publishes the two elevation maps the motion stack consumes, elevation-ONLY (no
traversability), mirroring the closed-loop sim's heightmap stage (demos/pipeline_sim):

  * `elevation_local`  — the SINGLE-SCAN map for MPPI: this scan rasterized in a
    robot-centered `win_m` window, trusted only where it has support and inpainted
    over small gaps; blind cells fall back to the accumulated map (memory).
  * `elevation_global` — the ACCUMULATED map for planning/routing: the rolling
    device map rasterized over a larger `route_m` robot-centered window.

Pose comes from odometry (`nav_msgs/Odometry`) refined by scan-to-submap 6-DOF
point-to-plane ICP. When `gravity_enable`, the IMU's gravity vector anchors the
ICP roll/pitch each scan (see IcpConfig.gravity_weight), so geometry-only tilt
cannot drift the map off level.

Frames: sensor -> base (base_frame, static TF) -> odom -> world == map_frame. The
world frame is bootstrapped to odom at the first scan.
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
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from helhest.perception import DeviceMapAccumulator
from helhest.perception import DynamicFilterConfig
from helhest.perception import DynamicPointFilter
from helhest.perception import FlatGroundFootprint
from helhest.perception import FootprintConfig
from helhest.perception import HeightMapBuilder
from helhest.perception import IcpAligner
from helhest.perception import IcpConfig
from helhest.perception import multigrid_inpaint
from helhest.perception import OutlierFilterConfig
from helhest.perception import StatisticalOutlierFilter
from helhest.perception import TerrainMap
from helhest.perception import transform_points
from helhest.perception.dynamic.frontier import frontier_from_organized
from helhest.localization import Localizer
from helhest.localization import LocalizerConfig
from helhest.localization import RegistrationOutcome
from helhest.localization.pose_math import deskew_scan
from helhest.localization.pose_math import invert_pose
from helhest.localization.pose_math import matrix_to_quaternion
from helhest.localization.pose_math import transform_points_xyz
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformException

from ._pipeline_common import grid_to_cloud
from ._pipeline_common import pointcloud2_to_xyz_time_array
from ._pipeline_common import quaternion_to_matrix

_EZ = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # world up

# Construction-time params: a change to any rebuilds the owning object.
_ICP_BUILD = frozenset(
    {
        "icp_max_iters",
        "icp_max_corr_dist_m",
        "icp_normal_radius_m",
        "icp_voxel_size_m",
        "icp_voxel_target",
        "gravity_enable",
        "gravity_weight",
        "device",
    }
)
_ACC_BUILD = frozenset(
    {"accumulation_voxel_m", "map_max_radius_m", "map_z_min_m", "map_z_max_m", "device"}
)
_DYN_BUILD = frozenset(
    {
        "dynamic_az_bins",
        "dynamic_el_bins",
        "dynamic_el_min_deg",
        "dynamic_el_max_deg",
        "dynamic_margin_m",
        "dynamic_margin_rel",
        "dynamic_min_range_m",
        "device",
    }
)
_OUTLIER_BUILD = frozenset(
    {"outlier_search_radius_m", "outlier_min_neighbors", "outlier_std_mult", "device"}
)


def _dilate_bool(mask: np.ndarray, k: int) -> np.ndarray:
    """Grow a boolean mask by k cells (4-neighbour) — the reach we trust the inpaint over."""
    out = mask.copy()
    for _ in range(max(0, k)):
        n = out.copy()
        n[1:, :] |= out[:-1, :]
        n[:-1, :] |= out[1:, :]
        n[:, 1:] |= out[:, :-1]
        n[:, :-1] |= out[:, 1:]
        out = n
    return out


class ElevationNode(Node):
    """Publish the single-scan (MPPI) and accumulated (planning) elevation maps."""

    def __init__(self) -> None:
        super().__init__("elevation")

        self._declare_parameters()
        self._cache_params()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.map_wp: wp.array | None = None  # accumulated device cloud (world frame)
        self.map_ages: wp.array | None = None  # per-map-point last-seen frame (recency pruning)
        self._frame: int = 0  # monotonic frame counter for recency stamps
        self._beam_dirs: np.ndarray | None = None  # per-beam unit dirs, built once for the frontier
        self.localizer: Localizer | None = None
        self._latest_imu: Imu | None = None
        self._deskew_warned = False
        self._imu_warned = False

        self.device = self._resolve_device(self.get_parameter("device").value)
        self._build_aligner()
        self._build_localizer()
        self._build_accumulator()
        self._build_dynamic_filter()
        self._build_outlier_filter()

        self.create_subscription(Imu, self.imu_topic, self._imu_callback, 20)
        # LiDAR is best-effort (SensorDataQoS); a reliable sub gets nothing from it.
        self.cloud_sub = Subscriber(
            self, PointCloud2, self.lidar_topic, qos_profile=qos_profile_sensor_data
        )
        self.odom_sub = Subscriber(self, Odometry, self.odom_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.cloud_sub, self.odom_sub],
            queue_size=self.get_parameter("sync_queue").value,
            slop=self.get_parameter("sync_slop_s").value,
        )
        self.sync.registerCallback(self._synced_callback)

        self.pub_local = self.create_publisher(PointCloud2, "elevation_local", 10)
        self.pub_global = self.create_publisher(PointCloud2, "elevation_global", 10)
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self.get_logger().info(
            f"ElevationNode: cloud={self.lidar_topic} odom={self.odom_topic} imu={self.imu_topic} "
            f"map_frame={self.map_frame} win_m={self.win_m} route_m={self.route_m} "
            f"gravity={'on' if self.gravity_enable else 'off'} device={self.device}"
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        d = self.declare_parameter
        # ROS / sensors
        d("lidar_topic", "/ouster/points")
        d("odom_topic", "/odom_2d")
        d("imu_topic", "/imu/data")
        d("base_frame", "base_link")
        d("map_frame", "map")
        d("sync_slop_s", 0.05)
        d("sync_queue", 30)
        d("device", "auto")
        # Scan deskew
        d("deskew_enable", True)
        d("deskew_time_field", "t")
        # Height crop on the input scan, in base_frame (robot-relative). Drops ceiling /
        # sub-floor noise before it reaches ICP and both maps. Bounds are metres in z.
        d("z_crop_enable", True)
        d("z_crop_min", -1.0)
        d("z_crop_max", 0.5)
        # Robot self-filter: drop the robot's own returns (wheels/body) — a base_frame
        # box. Measured from slow_translate: the side-mounted sensor sees the FRONT
        # wheels at x[0.2,0.5] y[-0.4,0.4]; box has a small margin.
        d("self_filter_enable", True)
        d("self_x_min", 0.15)
        d("self_x_max", 0.55)
        d("self_y_min", -0.45)
        d("self_y_max", 0.45)
        # Statistical outlier removal on the input scan (GPU, range-normalized k-NN):
        # drops sparse specks/noise before ICP and both maps. Range-normalized against
        # the sensor origin so it spares legitimately sparse distant ground; the
        # min_neighbors gate is an absolute count (6 is safe out to the routing window).
        d("outlier_enable", True)
        d("outlier_search_radius_m", 0.25)
        d("outlier_min_neighbors", 6)
        d("outlier_std_mult", 1.0)  # reject beyond mean + this*std of the neighbor distance
        # Heightmap (live-tunable)
        d("resolution", 0.15)
        d("win_m", 8.0)  # single-scan / MPPI window (robot-centered)
        d("route_m", 16.0)  # accumulated / planning window (robot-centered)
        d("local_support", 2)  # min points/cell to trust the single scan
        d("local_max_gap_m", 0.4)  # trust the inpaint this far from a real return
        d("inpaint_iters_per_level", 50)
        d("inpaint_coarse_iters", 200)
        # Robot footprint: force a flat ground patch under the robot in the LOCAL (MPPI)
        # map, so the blind spot the robot's own body/wheels carve out reads as level
        # ground instead of a hole. The patch tilts with roll/pitch (plane from
        # world_T_base). Robot-centered box; height = robot_frame → ground distance.
        d("footprint_enable", True)
        d("footprint_robot_height", 0.4)  # base_link -> ground (m); patch sits this far below
        d("footprint_half_x", 0.5)
        d("footprint_half_y", 0.5)
        d("footprint_center_x", 0.0)  # box center offset in base_frame (m)
        d("footprint_center_y", 0.0)
        d("footprint_mode", "overwrite")  # 'overwrite' | 'fill' (fill only stamps empty cells)
        # Accumulator
        d("accumulation_voxel_m", 0.10)
        d("map_max_radius_m", 50.0)
        d("map_z_min_m", -50.0)
        d("map_z_max_m", 50.0)
        # Dynamic-obstacle carving: remove accumulated points the current scan sees
        # through (moving things). Visibility ray-carve against the new scan.
        d("dynamic_enable", True)
        d("dynamic_az_bins", 1024)  # range-image resolution; match the sensor (Ouster 1024x128)
        d("dynamic_el_bins", 128)
        d("dynamic_el_min_deg", -90.0)  # full hemisphere (world-frame binning, robust to mount)
        d("dynamic_el_max_deg", 90.0)
        d("dynamic_margin_m", 0.3)  # carve only if the scan is farther by this + range*margin_rel
        d("dynamic_margin_rel", 0.02)
        d("dynamic_min_range_m", 0.5)
        # Ray-carve against the free-space FRONTIER (organized cloud: miss beams -> far point),
        # not just returns. Without it, a moving object with no background behind it is never
        # carved (see dynamic/frontier.py). Falls back to returns on a non-organized sensor.
        d("dynamic_frontier_enable", True)
        d("dynamic_frontier_max_range_m", 100.0)  # range a no-return beam is treated as free to
        # Recency pruning: forget accumulated cells that stay in view but go unconfirmed for
        # this many frames — clears the moving-object trail the geometric carve leaves behind.
        # In-view = within recency_view_range_m of the robot (360 deg sensor). Out-of-view
        # (occluded / off-camera) cells are preserved. At 10 Hz, 10 frames ~= 1 s.
        # Keep view_range tight: far static is sampled too sparsely to be re-hit every frame,
        # so a large range wrongly evicts it (measured: 8 m loses 6% static, 30 m loses 17%,
        # for the SAME dynamic cleanup — obstacles are near).
        d("dynamic_recency_enable", True)
        d("dynamic_max_unseen_frames", 10)
        d("dynamic_recency_view_range_m", 8.0)
        # ICP
        d("icp_enable", True)
        d("icp_submap_radius_m", 15.0)
        d("icp_max_iters", 30)
        d("icp_max_corr_dist_m", 0.5)
        d("icp_normal_radius_m", 0.3)
        d("icp_voxel_size_m", 0.1)
        d("icp_voxel_target", True)
        d("icp_min_inliers", 500)
        d("icp_max_corr_trans_m", 1.0)
        d("icp_max_corr_rot_deg", 15.0)
        d("icp_min_submap_points", 2000)
        # On a REJECTED registration the pose fell back to raw odom, so the old
        # accumulated map would smear against it — drop it and re-seed from this scan.
        d("reset_map_on_reject", True)
        # Gravity prior (IMU anchors ICP roll/pitch)
        d("gravity_enable", True)
        d("gravity_weight", 2000.0)
        d("gravity_use_accel", False)  # force accel gravity even if orientation is present
        # Viz
        d("publish_map_tf", True)

    def _cache_params(self) -> None:
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.lidar_topic: str = g("lidar_topic")
        self.odom_topic: str = g("odom_topic")
        self.imu_topic: str = g("imu_topic")
        self.base_frame: str = g("base_frame")
        self.map_frame: str = g("map_frame")
        self.deskew_enable: bool = g("deskew_enable")
        self.deskew_time_field: str = g("deskew_time_field")
        self.z_crop_enable: bool = g("z_crop_enable")
        self.z_crop_min: float = g("z_crop_min")
        self.z_crop_max: float = g("z_crop_max")
        self.self_filter_enable: bool = g("self_filter_enable")
        self.self_x_min: float = g("self_x_min")
        self.self_x_max: float = g("self_x_max")
        self.self_y_min: float = g("self_y_min")
        self.self_y_max: float = g("self_y_max")
        self.outlier_enable: bool = g("outlier_enable")
        self.resolution: float = g("resolution")
        self.win_m: float = g("win_m")
        self.route_m: float = g("route_m")
        self.local_support: int = g("local_support")
        self.local_max_gap_m: float = g("local_max_gap_m")
        self.inpaint_iters_per_level: int = g("inpaint_iters_per_level")
        self.inpaint_coarse_iters: int = g("inpaint_coarse_iters")
        self.footprint_enable: bool = g("footprint_enable")
        self.footprint_robot_height: float = g("footprint_robot_height")
        self.footprint_half_x: float = g("footprint_half_x")
        self.footprint_half_y: float = g("footprint_half_y")
        self.footprint_center_x: float = g("footprint_center_x")
        self.footprint_center_y: float = g("footprint_center_y")
        self.footprint_mode: str = g("footprint_mode")
        self.icp_enable: bool = g("icp_enable")
        self.icp_submap_radius_m: float = g("icp_submap_radius_m")
        self.icp_min_inliers: int = g("icp_min_inliers")
        self.icp_max_corr_trans_m: float = g("icp_max_corr_trans_m")
        self.icp_max_corr_rot_rad: float = float(np.deg2rad(g("icp_max_corr_rot_deg")))
        self.icp_min_submap_points: int = g("icp_min_submap_points")
        self.dynamic_enable: bool = g("dynamic_enable")
        self.dynamic_frontier_enable: bool = g("dynamic_frontier_enable")
        self.dynamic_frontier_max_range_m: float = g("dynamic_frontier_max_range_m")
        self.dynamic_recency_enable: bool = g("dynamic_recency_enable")
        self.dynamic_max_unseen_frames: int = g("dynamic_max_unseen_frames")
        self.dynamic_recency_view_range_m: float = g("dynamic_recency_view_range_m")
        self.gravity_enable: bool = g("gravity_enable")
        self.gravity_use_accel: bool = g("gravity_use_accel")
        self.reset_map_on_reject: bool = g("reset_map_on_reject")
        self.publish_map_tf: bool = g("publish_map_tf")

    @staticmethod
    def _resolve_device(name: str) -> wp.context.Device:
        if name == "auto":
            return wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")
        return wp.get_device(name)

    # ------------------------------------------------------------------
    # Heavy-object construction
    # ------------------------------------------------------------------

    def _build_aligner(self) -> None:
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        cfg = IcpConfig(
            max_iters=g("icp_max_iters"),
            max_correspondence_dist_m=g("icp_max_corr_dist_m"),
            normal_radius_m=g("icp_normal_radius_m"),
            voxel_size_m=g("icp_voxel_size_m") or None,
            voxel_target=g("icp_voxel_target"),
            gravity_weight=(g("gravity_weight") if self.gravity_enable else 0.0),
        )
        self.aligner = IcpAligner(cfg, device=self.device)
        if self.localizer is not None:
            self.localizer.aligner = self.aligner

    def _localizer_config(self) -> LocalizerConfig:
        return LocalizerConfig(
            enable=self.icp_enable,
            submap_radius_m=self.icp_submap_radius_m,
            min_submap_points=self.icp_min_submap_points,
            min_inliers=self.icp_min_inliers,
            max_correction_trans_m=self.icp_max_corr_trans_m,
            max_correction_rot_rad=self.icp_max_corr_rot_rad,
        )

    def _build_localizer(self) -> None:
        self.localizer = Localizer(self.aligner, self._localizer_config())

    def _build_accumulator(self) -> None:
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.acc = DeviceMapAccumulator(
            g("accumulation_voxel_m"),
            g("map_max_radius_m"),
            z_bounds=(g("map_z_min_m"), g("map_z_max_m")),
            device=self.device,
        )

    def _build_dynamic_filter(self) -> None:
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        cfg = DynamicFilterConfig(
            az_bins=g("dynamic_az_bins"),
            el_bins=g("dynamic_el_bins"),
            el_min_deg=g("dynamic_el_min_deg"),
            el_max_deg=g("dynamic_el_max_deg"),
            margin_m=g("dynamic_margin_m"),
            margin_rel=g("dynamic_margin_rel"),
            min_range_m=g("dynamic_min_range_m"),
        )
        self.dynamic_filter = DynamicPointFilter(cfg, device=self.device)

    def _build_outlier_filter(self) -> None:
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        cfg = OutlierFilterConfig(
            search_radius_m=g("outlier_search_radius_m"),
            min_neighbors=g("outlier_min_neighbors"),
            std_multiplier=g("outlier_std_mult"),
        )
        self.outlier_filter = StatisticalOutlierFilter(cfg, device=self.device)

    def _on_parameters_changed(self, params) -> SetParametersResult:
        names = {p.name for p in params}
        try:
            self._cache_params()
            if names & _ICP_BUILD:
                self.device = self._resolve_device(self.get_parameter("device").value)
                self._build_aligner()
            if names & _ACC_BUILD:
                self._build_accumulator()
            if names & _DYN_BUILD:
                self._build_dynamic_filter()
            if names & _OUTLIER_BUILD:
                self._build_outlier_filter()
            if "device" in names:  # device moved -> device-resident state is stale
                self.map_wp = None
                self.map_ages = None
                self._build_localizer()  # fresh pose state; re-bootstraps on the next scan
        except Exception as exc:  # a bad value must not kill the node
            return SetParametersResult(successful=False, reason=str(exc))
        self.localizer.config = self._localizer_config()
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _imu_callback(self, msg: Imu) -> None:
        self._latest_imu = msg

    def _synced_callback(self, cloud_msg: PointCloud2, odom_msg: Odometry) -> None:
        try:
            self._process(cloud_msg, odom_msg)
        except Exception as exc:
            self.get_logger().error(f"elevation error: {exc}")

    def _process(self, cloud_msg: PointCloud2, odom_msg: Odometry) -> None:
        self._frame += 1
        odom_T_base = self._odom_to_matrix(odom_msg)
        scan = self._scan_in_base(cloud_msg)
        if scan is None or scan[0].shape[0] == 0:
            self.get_logger().warn("Empty / untransformable scan — skipping.")
            return
        scan_base, point_times, base_T_sensor = scan
        scan_base, point_times = self._z_crop(scan_base, point_times)
        scan_base, point_times = self._self_filter(scan_base, point_times)
        if scan_base.shape[0] == 0:
            self.get_logger().warn("crop/self-filter removed all points — check bounds.")
            return
        gravity_up = self._gravity_up_base(cloud_msg.header.stamp)

        if not self.localizer.initialized:
            world_T_base = odom_T_base
            self.localizer.bootstrap(odom_T_base, world_T_base)
            scan_wp = wp.array(scan_base, dtype=wp.vec3, device=self.device)
            scan_wp = self._denoise(scan_wp, base_T_sensor)
        else:
            world_T_base_pred, sweep_delta = self.localizer.predict(odom_T_base)
            if self.deskew_enable:
                scan_base = self._deskew(scan_base, point_times, sweep_delta)
            scan_wp = wp.array(scan_base, dtype=wp.vec3, device=self.device)
            scan_wp = self._denoise(scan_wp, base_T_sensor)
            outcome = self.localizer.update(
                scan_wp, world_T_base_pred, self.map_wp, odom_T_base, gravity_up=gravity_up
            )
            self._log_registration(outcome)
            world_T_base = outcome.pose
            if self.reset_map_on_reject and outcome.status == "rejected":
                # Pose is now raw odom — start the global map over from this scan.
                self.map_wp = None
                self.map_ages = None
                self.get_logger().warn("ICP rejected -> resetting global map from this frame.")

        world_scan = transform_points(scan_wp, len(scan_wp), world_T_base)
        valid = wp.full(len(scan_wp), 1, dtype=wp.int32, device=self.device)
        # Dynamic-obstacle carving: drop accumulated points this scan saw THROUGH (moving
        # things). Carve the previous map by visibility against the fresh scan.
        carve = None
        if self.dynamic_enable and self.map_wp is not None and len(self.map_wp) > 0:
            world_T_sensor = world_T_base @ base_T_sensor
            sensor_origin = world_T_sensor[:3, 3].copy()
            # Carve against the free-space frontier (no-return beams = free space) so ghosts
            # with no background behind them are removed; returns-only if unavailable.
            carve_scan = self._frontier_world(cloud_msg, world_T_sensor) if self.dynamic_frontier_enable else None
            if carve_scan is None:
                carve_scan = world_scan
            carve = self.dynamic_filter.carve(self.map_wp, carve_scan, sensor_origin)
        center = (world_T_base[0, 3], world_T_base[1, 3])
        if self.dynamic_recency_enable:
            self.map_wp, self.map_ages = self.acc.step(
                self.map_wp, carve, world_scan, valid, center,
                map_ages=self.map_ages, frame=self._frame,
                max_unseen=self.dynamic_max_unseen_frames,
                view_range=self.dynamic_recency_view_range_m,
            )
        else:
            self.map_wp = self.acc.step(self.map_wp, carve, world_scan, valid, center)
            self.map_ages = None
        self._publish_maps(world_T_base, world_scan, cloud_msg.header.stamp)
        self._broadcast_map_tf(world_T_base, odom_msg)

    # ------------------------------------------------------------------
    # Dual elevation map (mirrors demos/pipeline_sim's heightmap stage)
    # ------------------------------------------------------------------

    def _footprint_plane_world(self, world_T_base: np.ndarray) -> tuple[float, float, float] | None:
        """Ground plane `z = a*x + b*y + c` (world/grid frame) under the robot.

        The footprint is flat at z = -robot_height in the base frame; expressed in the
        world grid it tilts with roll/pitch, so we project the base body z-axis (third
        column of world_R_base). Returns None if that axis is near-horizontal (rollover),
        where the level fallback would be meaningless in absolute world z.
        """
        r3 = world_T_base[:3, 2]  # base body z-axis in world
        rz = float(r3[2])
        if abs(rz) < 1e-6:
            return None
        t = world_T_base[:3, 3]
        h = self.footprint_robot_height
        r3_dot_t = float(r3[0] * t[0] + r3[1] * t[1] + r3[2] * t[2])
        a = -float(r3[0]) / rz
        b = -float(r3[1]) / rz
        c = (-h + r3_dot_t) / rz
        return (a, b, c)

    def _stamp_footprint(
        self,
        primary: wp.array,
        conf: np.ndarray,
        world_T_base: np.ndarray,
        cell: float,
        bounds: tuple[float, float, float, float],
    ) -> None:
        """Stamp the flat footprint patch into the local primary layer, in place.

        Mirrors the pipeline: write the ground plane into the `max` reduction (device)
        before it's read out, and force the patched cells to read as measured (`conf`)
        so they survive inpaint and show. No-op when disabled or the plane is degenerate.
        """
        if not self.footprint_enable:
            return
        plane = self._footprint_plane_world(world_T_base)
        if plane is None:
            return
        height, width = conf.shape
        ex, ey = float(world_T_base[0, 3]), float(world_T_base[1, 3])
        cfg = FootprintConfig(
            half_x=self.footprint_half_x,
            half_y=self.footprint_half_y,
            center=(ex + self.footprint_center_x, ey + self.footprint_center_y),
            ground_z=-self.footprint_robot_height,
            mode=self.footprint_mode,
        )
        fp = FlatGroundFootprint(cell, bounds, height, width, cfg, device=self.device)
        if fp.is_empty:
            return
        fp.apply(primary, plane)
        conf[fp.i0 : fp.i1, fp.j0 : fp.j1] = True

    def _publish_maps(self, world_T_base: np.ndarray, world_scan: wp.array, stamp) -> None:
        if self.map_wp is None or len(self.map_wp) == 0:
            return
        cell = self.resolution
        ex, ey = float(world_T_base[0, 3]), float(world_T_base[1, 3])

        with wp.ScopedDevice(self.device):
            # GLOBAL accumulated map over the routing window.
            rww = rwh = int(round(self.route_m / cell))
            rxmin, rymin = ex - 0.5 * rww * cell, ey - 0.5 * rwh * cell
            rgl = HeightMapBuilder(
                cell, (rxmin, rxmin + rww * cell, rymin, rymin + rwh * cell), device=self.device
            ).build(self.map_wp)
            rcount = rgl.count.numpy()
            rmax = rgl.max.numpy()
            rmeasured = rcount > 0
            relev_view = np.where(rmeasured, rmax, np.nan).astype(np.float32)
            relev_mem = np.where(rmeasured, rmax, 0.0).astype(np.float32)  # blind-cell fallback

            # LOCAL single-scan map over the (centered) MPPI window.
            lww = lwh = int(round(self.win_m / cell))
            ox, oy = (rww - lww) // 2, (rwh - lwh) // 2  # plan window == center of routing window
            lxmin, lymin = rxmin + ox * cell, rymin + oy * cell
            lbounds = (lxmin, lxmin + lww * cell, lymin, lymin + lwh * cell)
            ll = HeightMapBuilder(cell, lbounds, device=self.device).build(world_scan)
            conf = ll.count.numpy() >= self.local_support
            # Force the flat patch under the robot into the primary max (device) and mark
            # those cells measured — before max is read out and inpainted.
            self._stamp_footprint(ll.max, conf, world_T_base, cell, lbounds)
            hm = np.where(conf, ll.max.numpy(), np.nan).astype(np.float32)
            filled = np.nan_to_num(
                np.asarray(
                    multigrid_inpaint(
                        hm,
                        iters_per_level=self.inpaint_iters_per_level,
                        coarse_iters=self.inpaint_coarse_iters,
                    )
                ),
                nan=0.0,
            ).astype(np.float32)
            known = _dilate_bool(conf, int(round(self.local_max_gap_m / cell)))

        mem = relev_mem[oy : oy + lwh, ox : ox + lww]
        mem_known = rmeasured[oy : oy + lwh, ox : ox + lww]
        elev_local = np.where(known, filled, mem).astype(np.float32)
        # Only publish cells with real info (fresh scan or remembered); the rest are unknown.
        show = known | mem_known
        elev_local_view = np.where(show, elev_local, np.nan).astype(np.float32)

        self._publish_grid(self.pub_local, elev_local_view, lxmin, lymin, cell, stamp)
        self._publish_grid(self.pub_global, relev_view, rxmin, rymin, cell, stamp)

    def _publish_grid(self, pub, elev: np.ndarray, xmin: float, ymin: float, cell: float, stamp):
        ny, nx = elev.shape
        tm = TerrainMap(resolution=cell, bounds=(xmin, xmin + nx * cell, ymin, ymin + ny * cell))
        tm.elevation = elev
        cloud = grid_to_cloud(
            terrain_map=tm,
            x_min=xmin,
            y_min=ymin,
            resolution=cell,
            stamp=stamp,
            frame_id=self.map_frame,
            logger=self.get_logger(),
        )
        if cloud is not None:
            pub.publish(cloud)

    # ------------------------------------------------------------------
    # IMU gravity vector -> up-in-base
    # ------------------------------------------------------------------

    def _gravity_up_base(self, stamp) -> np.ndarray | None:
        """Up-direction in base_frame from the latest IMU: orientation if valid, else accel.

        Returns None when gravity is disabled or no usable IMU/TF is available (the ICP
        then just runs geometry-only). Uses the IMU frame's static TF into base_frame, so
        it works whether the IMU is `imu` (== base) or `os_imu`.
        """
        if not self.gravity_enable or self._latest_imu is None:
            return None
        imu = self._latest_imu
        q = imu.orientation
        have_orientation = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) > 0.5
        if have_orientation and not self.gravity_use_accel:
            # world_R_imu -> up expressed in the imu frame = R^T · ẑ (third row of R).
            r_imu = quaternion_to_matrix(q.x, q.y, q.z, q.w)[:3, :3]
            up_imu = r_imu.T @ _EZ
        else:
            a = np.array(
                [imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z]
            )
            n = float(np.linalg.norm(a))
            if n < 1e-6:  # no accel either -> give up gracefully
                if not self._imu_warned:
                    self.get_logger().warn("IMU has no orientation and no accel — gravity off.")
                    self._imu_warned = True
                return None
            up_imu = a / n  # accelerometer measures -g -> points up when static
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, imu.header.frame_id, stamp)
        except TransformException:
            try:  # fall back to the latest available IMU->base transform
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, imu.header.frame_id, rclpy.time.Time()
                )
            except TransformException as exc:
                self.get_logger().warn(f"IMU->base TF failed: {exc}")
                return None
        r = tf.transform.rotation
        base_R_imu = quaternion_to_matrix(r.x, r.y, r.z, r.w)[:3, :3]
        up_base = base_R_imu @ up_imu
        n = float(np.linalg.norm(up_base))
        return (up_base / n).astype(np.float64) if n > 1e-9 else None

    # ------------------------------------------------------------------
    # Scan / odom / deskew / TF (shared shape with terrain_accumulator_node)
    # ------------------------------------------------------------------

    def _odom_to_matrix(self, odom_msg: Odometry) -> np.ndarray:
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        T = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        T[0, 3], T[1, 3], T[2, 3] = p.x, p.y, p.z
        return T

    def _scan_in_base(
        self, cloud_msg: PointCloud2
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warn(f"sensor->base TF lookup failed: {exc}")
            return None
        t = transform.transform.translation
        r = transform.transform.rotation
        base_T_sensor = quaternion_to_matrix(r.x, r.y, r.z, r.w)
        base_T_sensor[0, 3], base_T_sensor[1, 3], base_T_sensor[2, 3] = t.x, t.y, t.z
        points, point_times = pointcloud2_to_xyz_time_array(cloud_msg, self.deskew_time_field)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32), None, base_T_sensor
        scan_base = transform_points_xyz(base_T_sensor, points.astype(np.float64))
        return scan_base.astype(np.float32), point_times, base_T_sensor

    def _z_crop(
        self, scan_base: np.ndarray, point_times: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Keep only base_frame points with z in [z_crop_min, z_crop_max]; times stay aligned."""
        if not self.z_crop_enable:
            return scan_base, point_times
        z = scan_base[:, 2]
        keep = (z >= self.z_crop_min) & (z <= self.z_crop_max)
        if point_times is not None:
            point_times = point_times[keep]
        return scan_base[keep], point_times

    def _self_filter(
        self, scan_base: np.ndarray, point_times: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Drop the robot's own returns (wheels/body): base_frame points inside the
        footprint box [self_x_min,max] x [self_y_min,max]. Times stay aligned."""
        if not self.self_filter_enable:
            return scan_base, point_times
        x, y = scan_base[:, 0], scan_base[:, 1]
        inside = (
            (x >= self.self_x_min)
            & (x <= self.self_x_max)
            & (y >= self.self_y_min)
            & (y <= self.self_y_max)
        )
        keep = ~inside
        if point_times is not None:
            point_times = point_times[keep]
        return scan_base[keep], point_times

    def _denoise(self, scan_wp: wp.array, base_T_sensor: np.ndarray) -> wp.array:
        """GPU-native statistical outlier removal on the base-frame scan (device in/out).

        Range-normalized k-NN (see StatisticalOutlierFilter): strips sparse specks
        without punishing legitimately sparse distant ground. The sensor origin (the
        static mount) is set per call so the range normalization is in the right frame.
        """
        if not self.outlier_enable or len(scan_wp) == 0:
            return scan_wp
        self.outlier_filter.config.sensor_origin = (
            float(base_T_sensor[0, 3]),
            float(base_T_sensor[1, 3]),
            float(base_T_sensor[2, 3]),
        )
        return self.outlier_filter.apply(scan_wp)

    def _ensure_beam_dirs(self, height: int, width: int, xyz: np.ndarray) -> np.ndarray:
        """Per-beam unit directions (sensor frame) for the organized cloud, built once.

        The beam geometry is fixed, so reconstruct azimuth-per-column and altitude-per-row
        from one frame's hits (median), interpolate beams that never returned, and cache.
        Miss beams then get a valid direction for the free-space frontier.
        """
        n = height * width
        if self._beam_dirs is not None and len(self._beam_dirs) == n:
            return self._beam_dirs
        r = np.linalg.norm(xyz, axis=1)
        hit = np.isfinite(r) & (r > 1e-3)
        d = np.zeros((n, 3))
        d[hit] = xyz[hit] / r[hit, None]
        g = d.reshape(height, width, 3)
        hg = hit.reshape(height, width)
        az = np.arctan2(g[..., 1], g[..., 0])
        alt = np.arctan2(g[..., 2], np.hypot(g[..., 0], g[..., 1]))
        az_col = np.array(
            [np.median(az[:, c][hg[:, c]]) if hg[:, c].any() else np.nan for c in range(width)]
        )
        alt_row = np.array(
            [np.median(alt[i, :][hg[i, :]]) if hg[i, :].any() else np.nan for i in range(height)]
        )
        # Interpolate never-returned beams; unwrap azimuth first so the +/-pi seam doesn't
        # corrupt the fill. Rows (altitude) are monotone, no wrap.
        cv = ~np.isnan(az_col)
        az_col = np.interp(np.arange(width), np.where(cv)[0], np.unwrap(az_col[cv]))
        rv = ~np.isnan(alt_row)
        alt_row = np.interp(np.arange(height), np.where(rv)[0], alt_row[rv])
        AZ, ALT = np.meshgrid(az_col, alt_row)
        ca = np.cos(ALT)
        beam = np.stack([ca * np.cos(AZ), ca * np.sin(AZ), np.sin(ALT)], axis=-1)
        self._beam_dirs = beam.reshape(n, 3).astype(np.float32)
        return self._beam_dirs

    def _frontier_world(self, cloud_msg: PointCloud2, world_T_sensor: np.ndarray) -> wp.array | None:
        """Free-space frontier as a device cloud in the world frame, for ray-carving.

        Hits keep their measured point; no-return beams become a far point along the beam
        (dynamic/frontier.py). Returns None when the cloud isn't organized (no per-beam miss
        info, e.g. Livox) so the caller falls back to carving against returns only.
        """
        height, width = cloud_msg.height, cloud_msg.width
        if height <= 1:
            return None
        xyz = (
            read_points_numpy(cloud_msg, field_names=("x", "y", "z"), reshape_organized_cloud=True)
            .reshape(height * width, 3)
            .astype(np.float64)
        )
        beam = self._ensure_beam_dirs(height, width, xyz)
        frontier = frontier_from_organized(
            xyz.astype(np.float32), beam, self.dynamic_frontier_max_range_m
        )
        fr_wp = wp.array(np.ascontiguousarray(frontier), dtype=wp.vec3, device=self.device)
        return transform_points(fr_wp, len(fr_wp), world_T_sensor)

    def _deskew(
        self, scan_base: np.ndarray, point_times: np.ndarray | None, delta: np.ndarray
    ) -> np.ndarray:
        if point_times is None:
            if not self._deskew_warned:
                self.get_logger().warn(
                    f"deskew on but cloud has no '{self.deskew_time_field}' field — skipping."
                )
                self._deskew_warned = True
            return scan_base
        span = float(point_times.max() - point_times.min())
        if span <= 0.0:
            return scan_base
        alphas = (point_times - point_times.min()) / span
        return deskew_scan(scan_base, alphas, delta).astype(np.float32)

    def _log_registration(self, outcome: RegistrationOutcome) -> None:
        if outcome.status == "sparse":
            self.get_logger().debug(
                f"submap too sparse ({outcome.submap_points} pts) — using odom prediction."
            )
        elif outcome.status == "rejected":
            self.get_logger().warn(
                f"ICP rejected (inliers={outcome.num_inliers} "
                f"Δrot={np.rad2deg(outcome.correction_rot_rad):.1f}° "
                f"Δtrans={outcome.correction_trans_m:.2f}m) — using odom prediction."
            )

    def _broadcast_map_tf(self, world_T_base: np.ndarray, odom_msg: Odometry) -> None:
        if not self.publish_map_tf:
            return
        map_T_odom = world_T_base @ invert_pose(self._odom_to_matrix(odom_msg))
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ElevationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
