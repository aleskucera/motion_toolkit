#!/usr/bin/env python3
"""Dual elevation mapper for ROS 2 Kilted — the tuning front-end for the planner.

Publishes the two elevation maps the motion stack consumes, elevation-ONLY (no
traversability), mirroring the closed-loop sim's heightmap stage (demos/pipeline_sim):

  * `elevation_local`  — the SINGLE-SCAN map for MPPI: this scan rasterized in a
    robot-centered `win_m` window, trusted only where it has support and inpainted
    over small gaps; blind cells fall back to the accumulated map (memory).
  * `elevation_global` — the ACCUMULATED map for planning/routing: the rolling
    device map rasterized over a larger `route_m` robot-centered window.
  * `accumulated_map` — the raw accumulated device cloud itself (up to the
    `map_max_radius_m` crop), for visualization. Unlike `elevation_global` (a
    small route-window heightmap driven by the planner), this shows the full
    mapped extent, so it's the topic to watch to confirm the map is persisting.

Pose comes from odometry (`nav_msgs/Odometry`) refined by scan-to-submap 6-DOF
point-to-plane ICP. When `gravity_enable`, the IMU's gravity vector anchors the
ICP roll/pitch each scan (see IcpConfig.gravity_weight), so geometry-only tilt
cannot drift the map off level.

Frames: sensor -> base (base_frame, static TF) -> odom -> world == map_frame. The
world frame is bootstrapped to odom at the first scan.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
import tf2_ros
import warp as wp
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs_py.point_cloud2 import read_points_numpy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
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
from helhest import dynamics
from helhest.control.command import condition_command
from helhest.control.command import JOINT_NAMES
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.control.mppi import SamplingConfig
from helhest.control.terminal import dock_control
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo
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

_IMU_BUFFER_LEN = 500  # ~5 s of 100 Hz IMU — enough to bracket any cloud stamp
_IMU_MAX_EXTRAP_S = 0.05  # fall back to odom if no IMU sample within this of the cloud stamp


def _rodrigues(omega: np.ndarray) -> np.ndarray:
    """Rotation matrix for the axis-angle vector `omega` (‖omega‖ = angle in rad)."""
    theta = float(np.linalg.norm(omega))
    if theta < 1.0e-9:
        return np.eye(3)
    k = omega / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)

# Construction-time params: a change to any rebuilds the owning object.
_ICP_BUILD = frozenset(
    {
        "icp_max_iters",
        "icp_max_corr_dist_m",
        "icp_trim_residual_m",
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
# Planner is sized to the windows + rollout shape; a change to any rebuilds it (and its
# CUDA graphs), so keep it off the per-frame path.
_PLAN_BUILD = frozenset(
    {
        "win_m",
        "route_m",
        "resolution",
        "plan_batch",
        "plan_horizon",
        "plan_n_theta",
        "plan_lat_coarsen",
        "plan_friction",
        "plan_robust_margin_m",
        "plan_robust_margin_deg",
        "plan_nominal_reset",
        "plan_goal_running",
        "plan_effort",
        "plan_wmax",
        "device",
    }
)


@dataclass(frozen=True)
class _MapFrame:
    """One frame's built maps + window geometry, shared by publishing and planning."""

    elev_local: np.ndarray  # (wh, ww) filled, NaN-free — planner terrain
    elev_local_view: np.ndarray  # (wh, ww) NaN in unknown cells — for RViz
    relev_view: np.ndarray  # (rwh, rww) NaN in unknown cells — for RViz
    relev_mem: np.ndarray  # (rwh, rww) blind cells = 0 — cost-to-go routing terrain
    cell: float
    ex: float
    ey: float
    lxmin: float
    lymin: float
    rxmin: float
    rymin: float


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
        self.map_streak: wp.array | None = None  # per-map-point seen-through streak (persist carve)
        self._frame: int = 0  # monotonic frame counter for recency stamps
        # Latest map->odom correction, cached from the last processed cloud and re-broadcast
        # at the full odom rate (see _odom_tf_callback) so base_link stays dense for TF lookups.
        self._map_T_odom: np.ndarray | None = None
        self._beam_dirs: np.ndarray | None = None  # per-beam unit dirs, built once for the frontier
        self.goal_xy: tuple[float, float] | None = None  # planning goal in map frame
        self._prev_cmd = np.zeros(3, np.float32)  # last published /cmd_joints [L, rear, R] (slew ref)
        self.planner: MppiGpu | None = None
        self.plan_sim: ForwardSimulator | None = None
        self.ctg: CostToGo | None = None
        self.sgrid = None  # routing lattice grid (built)
        self._plan_kr: int = 1
        self._plan_dims: tuple[int, int, int, int, int, int] | None = None
        self.localizer: Localizer | None = None
        self._latest_imu: Imu | None = None
        # (t_sec, quaternion xyzw, angular_velocity xyz) history so the deskew and the
        # rotation prior can read the gyro rate at the *cloud* stamp (not whatever arrived
        # last). The quaternion is buffered for gravity/debug only — the prior uses the gyro.
        self._imu_buffer: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=_IMU_BUFFER_LEN)
        # Running gyro-integrated world_R_base + the stamp it is integrated to (rotation prior).
        self._gyro_R_base: np.ndarray | None = None
        self._gyro_t: float | None = None
        self._base_R_gyro: np.ndarray | None = None  # cached static base<-imu rotation for the gyro
        self._consecutive_rejects = 0  # for reset-on-sustained-divergence
        self._prof: dict[str, float] = {}  # per-stage cumulative seconds (profile_stages)
        self._prof_n = 0
        self._prof_t = 0.0
        self._deskew_warned = False
        self._imu_warned = False

        self.device = self._resolve_device(self.get_parameter("device").value)
        self._build_aligner()
        self._build_localizer()
        self._build_accumulator()
        self._build_dynamic_filter()
        self._build_outlier_filter()
        if self.plan_enable:
            self._build_planner()

        # Sensor QoS (best-effort): a best-effort sub receives from BOTH a reliable publisher
        # (`/imu/data`) and a best-effort one (`/ouster/imu`); a reliable sub gets nothing from
        # the latter.
        self.create_subscription(Imu, self.imu_topic, self._imu_callback, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.get_parameter("goal_topic").value, self._goal_callback, 10)
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
        # Broadcast the pose TF on EVERY odom message (not just synced/processed frames), so
        # map->base_link is dense enough for RViz to look it up at any cloud stamp — e.g. with
        # base_link as the fixed frame. Same subscriber, extra callback: no second subscription.
        self.odom_sub.registerCallback(self._odom_tf_callback)

        self.pub_local = self.create_publisher(PointCloud2, "elevation_local", 10)
        self.pub_global = self.create_publisher(PointCloud2, "elevation_global", 10)
        self.pub_accum = self.create_publisher(PointCloud2, "accumulated_map", 1)
        self.pub_path = self.create_publisher(Path, "planned_path", 10)
        self.pub_path_marker = self.create_publisher(Marker, "planned_path_marker", 10)
        self.pub_frame = self.create_publisher(Marker, "frame_marker", 1)
        self.pub_cmd = self.create_publisher(JointState, self.get_parameter("cmd_topic").value, 10)
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
        # Horizontal range crop on the input scan: drop returns past this xy-distance from the
        # robot. Far Ouster returns are sparse grazing-angle ground — noise that only pollutes
        # ICP and the map. Cropped per SCAN so it never enters any stage. 0 disables.
        d("scan_max_range_m", 15.0)
        # Robot self-filter: drop the robot's own returns (wheels/body) — a base_frame
        # box. Measured from rotate (robot self-returns stay fixed in base while the scene
        # rotates): the sensor sees the front body/wheels as a bar at x[0.15,0.5] y[-0.5,0.5];
        # box has margin so the rim doesn't leak and trace rings in the rotating map.
        d("self_filter_enable", True)
        d("self_x_min", 0.10)
        d("self_x_max", 0.60)
        d("self_y_min", -0.55)
        d("self_y_max", 0.55)
        # Statistical outlier removal on the input scan (GPU, range-normalized k-NN):
        # drops sparse specks/noise before ICP and both maps. Range-normalized against
        # the sensor origin so it spares legitimately sparse distant ground; the
        # min_neighbors gate is an absolute count (6 is safe out to the routing window).
        d("outlier_enable", True)
        d("outlier_search_radius_m", 0.25)
        d("outlier_min_neighbors", 6)
        d("outlier_std_mult", 1.0)  # reject beyond mean + this*std of the neighbor distance
        # Heightmap (live-tunable). resolution/win_m validated on real bags: the finer 0.08 m cell
        # + 12 m fine window let the MPPI actually see berms across its plan (footprint violations
        # 36%->8% vs the old 0.15/8) -- see the Tier-B planner analysis.
        d("resolution", 0.08)
        d("win_m", 12.0)  # single-scan / MPPI window (robot-centered)
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
        # RADIUS (half-extent) of the robot-centered accumulated map: 15 m reaches 15 m in
        # every direction (15 ahead, 15 behind), matching the scan_max_range_m crop so the
        # trailing history stays as tight as the per-scan reach.
        d("map_max_radius_m", 15.0)
        d("map_z_min_m", -50.0)
        d("map_z_max_m", 50.0)
        # Dynamic-obstacle carving: remove accumulated points the current scan sees
        # through (moving things). Visibility ray-carve against the new scan.
        d("dynamic_enable", True)
        # Consecutive-free carve: only drop a point seen-through for this many frames IN A ROW,
        # so one grazing/dark/dropped-beam no-return can't delete static geometry. <=1 = the old
        # instantaneous carve. Threaded per-cell through the accumulator (survives re-voxelizing).
        # 25, not 8: when the robot drives PARALLEL to a wall the beams graze along it and read as
        # seen-through, but only INTERMITTENTLY (the 360° scan re-hits it as the pose changes), so a
        # high consecutive threshold keeps it — at 8 both corridor walls eroded ~43% while driving,
        # at 25 only ~24%. A moving obstacle's vacated spot is seen-through CONSECUTIVELY (open
        # ground, never re-hit), so its trail still carves — measured unchanged 8->25. Cost: a
        # vacated spot lingers ~2.5 s (25 frames @ 10 Hz) before clearing. Makes the frontier safe.
        d("carve_persist_frames", 25)
        # Age out a BETWEEN-BEAM speck: a map point on a bearing no beam reached, but whose
        # NEIGHBOURS were scanned, is dropped after this many frames (0 disables). Gated to NEAR +
        # IN-FRONT space (the two params below): ungated, the coarse-elevation gap test erased 37%
        # of the map (72% of structure past 8 m) — a distant real point lands in an empty el-bin
        # (128 bins / 180° = 1.4°/bin vs the ~0.35° beam pitch) and reads as a gap — plus the
        # wheel-occlusion shadows off to the sides. Near+front confines it to the path specks.
        d("carve_gap_frames", 8)
        d("carve_gap_max_range_m", 2.5)  # only gap-carve within this range (0 = no range gate)
        # Only gap-carve within this half-cone (deg) of the robot heading; excludes the wheel
        # shadows (~55-87° off heading) and the rear. 0 = no forward gate (carve all around).
        d("carve_gap_fwd_deg", 45.0)
        d("dynamic_az_bins", 1024)  # range-image resolution; match the sensor (Ouster 1024x128)
        d("dynamic_el_bins", 128)
        d("dynamic_el_min_deg", -90.0)  # full hemisphere (world-frame binning, robust to mount)
        d("dynamic_el_max_deg", 90.0)
        d("dynamic_margin_m", 0.3)  # carve only if the scan is farther by this + range*margin_rel
        d("dynamic_margin_rel", 0.05)  # range-proportional slack; absorbs angular-bin quantization
        #                                on slanted/radial walls that else reads as seen-through
        d("dynamic_min_range_m", 0.5)
        # Ray-carve against the free-space FRONTIER (organized cloud: miss beams -> far point),
        # not just returns. ON: needed to carve a moving person's TRAIL on open ground — where
        # the beam past a vacated spot hits nothing solid (a no-return), which only the frontier
        # treats as free. A lone no-return is ambiguous and used to over-carve static, but the
        # consecutive-free counter (carve_persist_frames above) now makes it safe: a static
        # surface that briefly no-returns is re-confirmed within `persist` frames and survives.
        d("dynamic_frontier_enable", True)
        d("dynamic_frontier_max_range_m", 100.0)  # range a no-return beam is treated as free to
        # Recency pruning: forget a cell that is OBSERVABLE this frame (a beam reached its
        # range) yet has gone unconfirmed for this many frames — the moving-object trail the
        # instantaneous carve leaves behind. Visibility-gated: cells the sensor cannot see
        # now (blind rear, occluded) are kept, so mapped history survives behind the robot
        # until it leaves the map radius or odometry breaks. At 10 Hz, 10 frames ~= 1 s.
        # OFF by default: this time-based age-out also erases legit STATIC structure seen at
        # grazing/sparse angles. The visibility ray-carve (dynamic_enable) still removes moving
        # obstacles the moment a beam passes through them; recency only cleaned up the residual
        # trail. Enable it if you need that trail removed and can accept eroding static cells.
        d("dynamic_recency_enable", False)
        d("dynamic_max_unseen_frames", 10)
        # ICP
        d("icp_enable", True)
        d("icp_submap_radius_m", 15.0)
        d("icp_max_iters", 30)
        d("icp_max_corr_dist_m", 0.5)
        d("icp_trim_residual_m", 0.0)  # reject correspondences past this p2plane residual; 0=off
        d("icp_normal_radius_m", 0.3)
        d("icp_voxel_size_m", 0.1)
        d("icp_voxel_target", True)
        d("icp_min_inliers", 500)
        d("icp_max_corr_trans_m", 1.0)
        # Correction caps are loose divergence rails; the RMS fit below is the real quality
        # gate. A fast in-place rotation legitimately needs a >15° per-frame correction when
        # the odom/IMU prior lags, so 25° admits those (good-fit) while rms rejects aliased fits.
        d("icp_max_corr_rot_deg", 25.0)
        d("icp_min_submap_points", 2000)
        # Point-to-plane RMS fit (m) above which a registration is rejected — the fitness
        # signal that lets the rot cap relax safely. Good rotate fits ~0.03-0.055, aliased/
        # diverged ones >=0.086, so 0.08 separates them. 0 = off (library default).
        d("icp_max_rms_residual_m", 0.08)
        # Yaw multi-start: run this many ICPs from headings spread over icp_yaw_search_deg about
        # the prediction and keep the best fit — escapes the wrong rotational basin under fast
        # skid-steer yaw. 1 = single ICP (off). GPU-parallel-friendly; costs ~N ICP launches.
        d("icp_yaw_restarts", 1)
        d("icp_yaw_search_deg", 30.0)
        # On a REJECTED registration the pose fell back to raw odom, so the old
        # accumulated map would smear against it — drop it and re-seed from this scan.
        d("reset_map_on_reject", True)
        d("reset_after_rejects", 5)  # wipe only after this many CONSECUTIVE rejects (sustained loss)
        d("debug_frames", False)  # INFO-log each frame's registration metrics (debugging)
        d("profile_stages", False)  # GPU-synced per-stage timing, logged every 30 frames (debugging)
        # Gravity prior (IMU anchors ICP roll/pitch)
        d("gravity_enable", True)
        d("gravity_weight", 2000.0)
        d("gravity_use_accel", False)  # force accel gravity even if orientation is present
        # Motion prior: take rotation from the IMU orientation (slip-immune), keeping only
        # translation from wheel odom — wheel odom yaw is wrong under skid (in-place rotation).
        d("imu_rotation_prior", True)
        # Reject single-sample gyro glitches before they reach the deskew / integrated rotation
        # prior: this robot's /imu/data spikes to >1000 deg/s for one sample (real motion peaks
        # ~300), and one such sample injects tens of degrees of phantom yaw. 0 disables.
        d("max_gyro_rate_dps", 600.0)
        # Viz
        d("publish_map_tf", True)
        # Close odom->base_link ourselves. helhest_llc publishes the /odom_2d message but
        # broadcasts no TF, so without this base_link is disconnected from map. Default on
        # here; set false if the odom source ever starts publishing it, or TF double-publishes.
        d("publish_odom_tf", True)
        d("publish_accumulated", True)  # republish the raw accumulated cloud on accumulated_map
        # MPPI planning (visualization only — no motor commands). Consumes the maps this node
        # already builds: elevation_local as the rollout terrain, elevation_global for the
        # cost-to-go routing field. Goal comes from RViz "2D Nav Goal" on goal_topic. Publishes
        # the intended path (nav_msgs/Path + a thick LINE_STRIP marker).
        d("plan_enable", True)
        d("goal_topic", "/goal_pose")
        d("plan_batch", 4096)  # MPPI rollouts B
        d("plan_horizon", 70)  # rollout steps T (planning_solver dt = 0.1 s)
        d("plan_n_theta", 24)  # cost-to-go heading bins
        d("plan_lat_coarsen", 4)  # routing/cost-to-go grid coarsening vs the map cell
        d("plan_n_refine", 3)  # MPPI refine iterations per frame
        d("plan_friction", 0.8)  # uniform rollout friction
        d("plan_robust_margin_m", 0.3)  # cost-to-go safety tube: lateral (m) ~ robot half-width;
        # keeps the routed center a footprint-width off berms (validated in the Tier-C closed loop:
        # 0 belly contacts). Tighten in narrow spaces -- it erodes the feasible set both sides.
        d("plan_robust_margin_deg", 0.0)  # cost-to-go safety tube: heading (deg)
        d("plan_nominal_reset", 1.5)  # nominal wheel speed the planner seeds from
        # MPPI speed knobs (rebuild the planner on change): the robot drives slow because the cost
        # balance prefers it. Raise goal_running (reward progress) and/or lower effort (penalty on
        # wheel-speed^2) to drive faster. plan_max_omega is only the output SAFETY clamp, not speed.
        d("plan_goal_running", 0.3)  # cost-to-go V^2 per step -> higher = faster (more progress pull)
        d("plan_effort", 2e-3)  # penalize wheel-speed^2 -> lower = faster (less speed penalty)
        # HARD speed ceiling: the MPPI wheel-speed sampling box [0, plan_wmax] rad/s. The planner
        # NEVER commands above this regardless of the cost -- raising goal_running does nothing once
        # it saturates at plan_wmax. This is the real top-speed knob. Keep <= the motor safe max
        # (plan_max_omega, the output clamp). ~1.4 m/s at 4.0; ~2.8 m/s at 8.0 (r=0.35).
        d("plan_wmax", 8.0)  # max per-wheel omega the planner may command [rad/s]
        # ACTUATION (drive the robot). plan_actuate publishes wheel commands to a real robot; set
        # it false to run planning as visualization only. All motor-safety conditioning (the
        # left-wheel sign flip, rear-follower, magnitude clamp, slew limit) is in control/command.py.
        d("plan_actuate", True)  # publish /cmd_joints wheel commands
        d("cmd_topic", "/cmd_joints")  # JointState wheel-velocity command topic (to the LLC)
        d("plan_max_omega", 8.0)  # hard cap on |wheel velocity| [rad/s] -- set to the motor safe max
        d("plan_max_slew", 50.0)  # hard cap on |d(cmd)/dt| per wheel [rad/s^2]
        d("plan_dock_radius", 1.5)  # hand off MPPI routing -> terminal dock within this range (m)
        d("plan_reach_radius", 0.3)  # goal reached -> command a (ramped) stop within this range (m)
        d("plan_path_width", 0.08)  # intended-path line marker width (m)

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
        self.scan_max_range_m: float = g("scan_max_range_m")
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
        self.icp_max_rms_residual_m: float = g("icp_max_rms_residual_m")
        self.icp_yaw_restarts: int = g("icp_yaw_restarts")
        self.icp_yaw_search_deg: float = g("icp_yaw_search_deg")
        self.dynamic_enable: bool = g("dynamic_enable")
        self.carve_persist_frames: int = g("carve_persist_frames")
        self.carve_gap_frames: int = g("carve_gap_frames")
        self.carve_gap_max_range_m: float = g("carve_gap_max_range_m")
        self.carve_gap_fwd_rad: float = np.deg2rad(g("carve_gap_fwd_deg"))
        self.dynamic_frontier_enable: bool = g("dynamic_frontier_enable")
        self.dynamic_frontier_max_range_m: float = g("dynamic_frontier_max_range_m")
        self.dynamic_recency_enable: bool = g("dynamic_recency_enable")
        self.dynamic_max_unseen_frames: int = g("dynamic_max_unseen_frames")
        self.gravity_enable: bool = g("gravity_enable")
        self.gravity_use_accel: bool = g("gravity_use_accel")
        self.imu_rotation_prior: bool = g("imu_rotation_prior")
        _max_gyro_dps: float = g("max_gyro_rate_dps")
        # squared rad/s gate, or inf when disabled (0) — compared against |omega|^2 per sample
        self._max_gyro_rate_sq: float = (
            np.deg2rad(_max_gyro_dps) ** 2 if _max_gyro_dps > 0.0 else np.inf
        )
        self.reset_map_on_reject: bool = g("reset_map_on_reject")
        self.reset_after_rejects: int = g("reset_after_rejects")
        self.debug_frames: bool = g("debug_frames")
        self.profile_stages: bool = g("profile_stages")
        self.publish_map_tf: bool = g("publish_map_tf")
        self.publish_odom_tf: bool = g("publish_odom_tf")
        self.publish_accumulated: bool = g("publish_accumulated")
        self.plan_enable: bool = g("plan_enable")
        self.plan_batch: int = g("plan_batch")
        self.plan_horizon: int = g("plan_horizon")
        self.plan_n_theta: int = g("plan_n_theta")
        self.plan_lat_coarsen: int = g("plan_lat_coarsen")
        self.plan_n_refine: int = g("plan_n_refine")
        self.plan_friction: float = g("plan_friction")
        self.plan_robust_margin_m: float = g("plan_robust_margin_m")
        self.plan_robust_margin_deg: float = g("plan_robust_margin_deg")
        self.plan_nominal_reset: float = g("plan_nominal_reset")
        self.plan_goal_running: float = g("plan_goal_running")
        self.plan_effort: float = g("plan_effort")
        self.plan_wmax: float = g("plan_wmax")
        self.plan_actuate: bool = g("plan_actuate")
        self.plan_max_omega: float = g("plan_max_omega")
        self.plan_max_slew: float = g("plan_max_slew")
        self.plan_dock_radius: float = g("plan_dock_radius")
        self.plan_reach_radius: float = g("plan_reach_radius")
        self.plan_path_width: float = g("plan_path_width")

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
            trim_residual_m=g("icp_trim_residual_m"),
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
            max_rms_residual_m=self.icp_max_rms_residual_m,
            yaw_restarts=self.icp_yaw_restarts,
            yaw_search_deg=self.icp_yaw_search_deg,
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

    def _build_planner(self) -> None:
        """Build the MPPI rollout simulator, planner, and cost-to-go, sized to the current
        windows/resolution. Expensive (rollout buffers + CUDA graphs) — only on structural
        param change, never per frame."""
        cell = self.resolution
        ww = wh = int(round(self.win_m / cell))
        rww = rwh = int(round(self.route_m / cell))
        kr = max(1, int(self.plan_lat_coarsen))
        rcny, rcnx, rccell = rwh // kr, rww // kr, cell * kr
        win_grid = GridParams(ww, wh, cell, 0.0, 0.0)
        self.plan_sim = ForwardSimulator(
            dynamics.robot_params(),
            dynamics.planning_solver(),
            win_grid,
            int(self.plan_batch),
            int(self.plan_horizon),
            self.device,
        )
        self.plan_sim.set_uniform_friction(self.plan_friction)
        self.planner = MppiGpu(
            self.plan_sim,
            CostParams(goal_running=self.plan_goal_running, effort=self.plan_effort),
            sampling=SamplingConfig(wmax=self.plan_wmax),
            n_theta=int(self.plan_n_theta),
        )
        self.planner.reset_nominal(self.plan_nominal_reset)
        self.ctg = CostToGo(
            GridParams(rcnx, rcny, rccell, 0.0, 0.0),
            dynamics.robot_params(),
            dynamics.planning_solver(),
            n_theta=int(self.plan_n_theta),
            robust_margin_m=self.plan_robust_margin_m,
            robust_margin_deg=self.plan_robust_margin_deg,
            device=self.device,
        )
        self.planner.cw.lattice_cap = self.ctg._vcap
        # Routing field expressed in the PLANNING window's frame: both windows are robot-centered,
        # so their origins differ by a constant cell offset.
        self.sgrid = GridParams(
            rcnx, rcny, rccell, (ww // 2 - rww // 2) * cell, (wh // 2 - rwh // 2) * cell
        ).build()
        self._plan_kr = kr
        self._plan_dims = (ww, wh, rww, rwh, rcnx, rcny)

    def _goal_callback(self, msg: PoseStamped) -> None:
        """RViz 2D Nav Goal. Assumes the pose is already in map_frame (RViz publishes in its
        fixed frame — set it to the map frame); warns otherwise and uses it as-is."""
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.get_logger().warn(
                f"goal frame '{msg.header.frame_id}' != map_frame '{self.map_frame}'; "
                "set the RViz Fixed Frame to the map frame."
            )
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"goal set: ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f})")

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
            if self.plan_enable and (names & _PLAN_BUILD or self.planner is None):
                self._build_planner()  # (re)build on structural change or first enable
            if "device" in names:  # device moved -> device-resident state is stale
                self.map_wp = None
                self.map_ages = None
                self.map_streak = None
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
        q = msg.orientation
        if q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w > 0.5:  # buffer valid fused orientations
            w = msg.angular_velocity
            # Drop a single-sample gyro glitch: the deskew and the integrated rotation prior both
            # read this buffer, and one 8000 deg/s spike sample injects ~80 deg of phantom yaw. The
            # integrator then bridges the missing sample with its good neighbours.
            if w.x * w.x + w.y * w.y + w.z * w.z > self._max_gyro_rate_sq:
                return
            base_R_imu = self._gyro_base_rotation(msg.header.frame_id)
            if base_R_imu is None:  # IMU->base TF not ready yet — skip until it is
                return
            w_base = base_R_imu @ np.array([w.x, w.y, w.z])
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._imu_buffer.append((t, np.array([q.x, q.y, q.z, q.w]), w_base))

    def _synced_callback(self, cloud_msg: PointCloud2, odom_msg: Odometry) -> None:
        try:
            self._process(cloud_msg, odom_msg)
        except Exception as exc:
            self.get_logger().error(f"elevation error: {exc}")

    def _ck(self, label: str) -> None:
        """Profiling checkpoint: sync the GPU (Warp is async) and accrue time since the last _ck."""
        if not self.profile_stages:
            return
        wp.synchronize()
        now = time.perf_counter()
        self._prof[label] = self._prof.get(label, 0.0) + (now - self._prof_t)
        self._prof_t = now

    def _process(self, cloud_msg: PointCloud2, odom_msg: Odometry) -> None:
        self._frame += 1
        if self.profile_stages:
            wp.synchronize()
            self._prof_t = time.perf_counter()
        odom_T_base = self._odom_to_matrix(odom_msg)
        scan = self._scan_in_base(cloud_msg)
        if scan is None or scan[0].shape[0] == 0:
            self.get_logger().warn("Empty / untransformable scan — skipping.")
            return
        scan_base, point_times, base_T_sensor = scan
        scan_base, point_times = self._z_crop(scan_base, point_times)
        scan_base, point_times = self._self_filter(scan_base, point_times)
        scan_base, point_times = self._range_crop(scan_base, point_times)
        if scan_base.shape[0] == 0:
            self.get_logger().warn("crop/self-filter removed all points — check bounds.")
            return
        gravity_up = self._gravity_up_base(cloud_msg.header.stamp)
        imu_R_base = self._gyro_orientation_base(cloud_msg.header.stamp)
        self._ck("preproc+prior")

        if not self.localizer.initialized:
            world_T_base = odom_T_base
            self.localizer.bootstrap(odom_T_base, world_T_base, imu_R_base)
            scan_wp = wp.array(scan_base, dtype=wp.vec3, device=self.device)
            scan_wp = self._denoise(scan_wp, base_T_sensor)
        else:
            world_T_base_pred, sweep_delta = self.localizer.predict(odom_T_base, imu_R_base)
            if self.deskew_enable:
                scan_base = self._deskew(scan_base, point_times, sweep_delta, cloud_msg.header.stamp)
            scan_wp = wp.array(scan_base, dtype=wp.vec3, device=self.device)
            scan_wp = self._denoise(scan_wp, base_T_sensor)
            self._ck("deskew+denoise")
            outcome = self.localizer.update(
                scan_wp,
                world_T_base_pred,
                self.map_wp,
                odom_T_base,
                imu_R_base_curr=imu_R_base,
                gravity_up=gravity_up,
            )
            self._ck("icp")
            self._log_registration(outcome)
            if self.debug_frames:
                self.get_logger().info(
                    f"F{self._frame} {outcome.status} "
                    f"rot={np.rad2deg(outcome.correction_rot_rad):.1f} "
                    f"trans={outcome.correction_trans_m:.2f} rms={outcome.rms_residual_m:.3f} "
                    f"inl={outcome.num_inliers} sub={outcome.submap_points} scan={len(scan_base)}"
                )
            world_T_base = outcome.pose
            # A single reject just uses the fallback pose (the map keeps accumulating). Only
            # SUSTAINED divergence — tracking genuinely lost — wipes the map and re-seeds from
            # this scan, so one bad frame no longer starves the ICP submap into a reset spiral.
            if outcome.status == "rejected":
                self._consecutive_rejects += 1
            elif outcome.status == "ok":
                self._consecutive_rejects = 0
            if self.reset_map_on_reject and self._consecutive_rejects >= self.reset_after_rejects:
                self.map_wp = None
                self.map_ages = None
                self.map_streak = None
                self._consecutive_rejects = 0
                self.get_logger().warn(
                    f"{self.reset_after_rejects} consecutive ICP rejects -> resetting global map."
                )

        world_scan = transform_points(scan_wp, len(scan_wp), world_T_base)
        valid = wp.full(len(scan_wp), 1, dtype=wp.int32, device=self.device)
        # Dynamic-obstacle carving: drop accumulated points this scan saw THROUGH (moving
        # things). Carve the previous map by visibility against the fresh scan.
        carve = None
        streak_out = None
        persist = self.carve_persist_frames
        streak_mode = self.dynamic_enable and persist > 1
        if self.dynamic_enable and self.map_wp is not None and len(self.map_wp) > 0:
            world_T_sensor = world_T_base @ base_T_sensor
            sensor_origin = world_T_sensor[:3, 3].copy()
            # Carve against the free-space frontier (no-return beams = free space) so ghosts
            # with no background behind them are removed; returns-only if unavailable.
            carve_scan = self._frontier_world(cloud_msg, world_T_sensor) if self.dynamic_frontier_enable else None
            if carve_scan is None:
                carve_scan = world_scan
            if streak_mode:
                # Consecutive-free carve: only drop a point the scan saw PAST for `persist`
                # frames in a row, so a single ambiguous no-return can't delete static geometry.
                n_map = len(self.map_wp)
                streak_in = (
                    self.map_streak
                    if self.map_streak is not None and len(self.map_streak) == n_map
                    else wp.zeros(n_map, dtype=wp.int32, device=self.device)
                )
                # Robot heading in world (base +x azimuth) — confines the gap age-out to the cone
                # in front of the robot, so it can't erode the wheel shadows off to the sides.
                fwd_az = float(np.arctan2(world_T_base[1, 0], world_T_base[0, 0]))
                carve, streak_out = self.dynamic_filter.carve_streak(
                    self.map_wp, carve_scan, sensor_origin, streak_in, persist,
                    self.carve_gap_frames, self.carve_gap_max_range_m, fwd_az,
                    self.carve_gap_fwd_rad,
                )
            elif self.dynamic_recency_enable and self.map_ages is not None:
                # Carve + visibility-gated recency: also forget cells that are OBSERVABLE now
                # but went unconfirmed for max_unseen frames. Cells the sensor can't currently
                # see (blind rear, occluded) are kept, so history survives behind the robot.
                carve = self.dynamic_filter.carve_recency(
                    self.map_wp,
                    carve_scan,
                    sensor_origin,
                    self.map_ages,
                    self._frame,
                    self.dynamic_max_unseen_frames,
                )
            else:
                carve = self.dynamic_filter.carve(self.map_wp, carve_scan, sensor_origin)
        if self.debug_frames and carve is not None:  # host readback — debugging only
            nmap = len(self.map_wp)
            ncarved = nmap - int(carve.numpy().sum())
            self.get_logger().info(f"F{self._frame} carved={ncarved}/{nmap} map points")
        self._ck("worldscan+carve")
        center = (world_T_base[0, 3], world_T_base[1, 3])
        if streak_mode:
            # Seed streaks at 0 on frames with no prior map (bootstrap / just reset).
            streak_arg = streak_out if streak_out is not None else wp.zeros(0, dtype=wp.int32, device=self.device)
            self.map_wp, self.map_streak = self.acc.step(
                self.map_wp, carve, world_scan, valid, center, map_streak=streak_arg,
            )
            self.map_ages = None
        elif self.dynamic_recency_enable:
            self.map_wp, self.map_ages = self.acc.step(
                self.map_wp, carve, world_scan, valid, center,
                map_ages=self.map_ages, frame=self._frame,
            )
            self.map_streak = None
        else:
            self.map_wp = self.acc.step(self.map_wp, carve, world_scan, valid, center)
            self.map_ages = None
            self.map_streak = None
        self._ck("accumulate")
        mf = self._build_maps(world_T_base, world_scan)
        self._ck("build_maps")
        if mf is not None:
            self._publish_maps(mf, cloud_msg.header.stamp)
            self._ck("publish_maps")
            if self.plan_enable and self.goal_xy is not None and self.planner is not None:
                self._plan(mf, world_T_base, cloud_msg.header.stamp)
        self._cache_map_correction(world_T_base, odom_msg)
        self._ck("cache_tf")
        if self.profile_stages:
            self._prof_n += 1
            if self._prof_n % 30 == 0:
                parts = " ".join(f"{k}={1000 * v / self._prof_n:.1f}"
                                 for k, v in sorted(self._prof.items(), key=lambda kv: -kv[1]))
                total = 1000 * sum(self._prof.values()) / self._prof_n
                self.get_logger().info(f"PROFILE avg ms/frame (n={self._prof_n}) total={total:.1f} | {parts}")

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

    def _build_maps(self, world_T_base: np.ndarray, world_scan: wp.array) -> _MapFrame | None:
        """Build the local (single-scan/MPPI) and global (routing) elevation maps for this frame."""
        if self.map_wp is None or len(self.map_wp) == 0:
            return None
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
        # Cells with real info (fresh scan or remembered); the rest are unknown -> NaN for RViz.
        show = known | mem_known
        elev_local_view = np.where(show, elev_local, np.nan).astype(np.float32)
        return _MapFrame(
            elev_local=elev_local,
            elev_local_view=elev_local_view,
            relev_view=relev_view,
            relev_mem=relev_mem,
            cell=cell,
            ex=ex,
            ey=ey,
            lxmin=lxmin,
            lymin=lymin,
            rxmin=rxmin,
            rymin=rymin,
        )

    def _publish_maps(self, mf: _MapFrame, stamp) -> None:
        self._publish_grid(self.pub_local, mf.elev_local_view, mf.lxmin, mf.lymin, mf.cell, stamp)
        self._publish_grid(self.pub_global, mf.relev_view, mf.rxmin, mf.rymin, mf.cell, stamp)
        if self.publish_accumulated:
            self._publish_accumulated(stamp)
        self._publish_frame_marker(mf, stamp)

    def _publish_frame_marker(self, mf: _MapFrame, stamp) -> None:
        """A floating text label with the current processed-frame index (over the robot), so
        the exact frame is readable in RViz — e.g. to pin down when tracking goes off."""
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.map_frame
        m.ns = "frame"
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(mf.ex)
        m.pose.position.y = float(mf.ey)
        m.pose.position.z = 2.5
        m.pose.orientation.w = 1.0
        m.scale.z = 0.6  # text height (m)
        m.color = ColorRGBA(r=1.0, g=1.0, b=0.2, a=1.0)
        m.text = f"#{self._frame}"
        self.pub_frame.publish(m)

    def _publish_accumulated(self, stamp) -> None:
        """Republish the raw accumulated device cloud (`self.map_wp`) as a PointCloud2.

        The one unavoidable host round-trip: the map lives on-device, ROS needs it on
        the host. Packed straight to bytes (no per-point Python loop), in map_frame.
        """
        if self.map_wp is None or len(self.map_wp) == 0:
            return
        pts = np.ascontiguousarray(self.map_wp.numpy(), dtype=np.float32)  # (N, 3)
        n = pts.shape[0]
        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = self.map_frame
        cloud.height = 1
        cloud.width = n
        cloud.fields = [
            PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, name in enumerate(("x", "y", "z"))
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * n
        cloud.is_dense = True
        cloud.data = pts.tobytes()
        self.pub_accum.publish(cloud)

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
    # MPPI planning (visualization only)
    # ------------------------------------------------------------------

    def _plan(self, mf: _MapFrame, world_T_base: np.ndarray, stamp) -> None:
        """Run MPPI toward the goal on this frame's maps; publish the intended path.

        Terrain = the single-scan elevation_local; the routing cost-to-go is solved on the
        accumulated elevation_global. Visualization only — no motor commands are emitted.
        """
        gx, gy = self.goal_xy
        ez = float(world_T_base[2, 3])
        eyaw = float(np.arctan2(world_T_base[1, 0], world_T_base[0, 0]))
        _, _, _, _, rcnx, rcny = self._plan_dims
        kr = self._plan_kr
        state_l = np.array([mf.ex - mf.lxmin, mf.ey - mf.lymin, eyaw], np.float32)
        goal_l = (gx - mf.lxmin, gy - mf.lymin)
        goal_r = (gx - mf.rxmin, gy - mf.rymin)
        with wp.ScopedDevice(self.device):
            self.plan_sim.set_terrain(
                wp.array(np.ascontiguousarray(mf.elev_local), dtype=wp.float32, device=self.device)
            )
            self._ck("plan:set_terrain")
            relev = mf.relev_mem  # (rwh, rww), blind cells = 0
            if kr > 1:
                Hc = relev[: rcny * kr, : rcnx * kr].reshape(rcny, kr, rcnx, kr).max(axis=(1, 3))
            else:
                Hc = relev
            V = self.ctg.compute(
                wp.array(np.ascontiguousarray(Hc), dtype=wp.float32, device=self.device), goal_r
            )
            self._ck("plan:ctg")
            self.planner.set_lattice(V, self.sgrid)
            self.planner.replan(state_l, goal_l, int(self.plan_n_refine))
            self._ck("plan:replan")
        # candidate 0 is the committed nominal rollout; window-local -> map coords.
        ctrl = self.planner.sim.controlled.numpy()  # [T+1, B, 3] = (x, y, yaw)
        self._ck("plan:readback")
        origin = np.array([mf.lxmin, mf.lymin], np.float32)
        self._publish_path(ctrl[:, 0, :2] + origin, ez, stamp)
        self._ck("plan:pub_path")

        # --- ACTUATION: turn the plan into a conditioned /cmd_joints command (default OFF) ---
        if not self.plan_actuate:
            return
        d = float(np.hypot(gx - mf.ex, gy - mf.ey))  # robot -> goal distance
        if d < self.plan_reach_radius:
            wl, wr = 0.0, 0.0  # reached -> stop (the slew limiter ramps the command down)
        elif d < self.plan_dock_radius:
            u = dock_control(state_l, goal_l, wmax=self.plan_max_omega)  # terminal dock
            wl, wr = float(u[0]), float(u[1])
        else:
            u0 = self.planner.nominal()[0]  # first committed step (wL, wR), model convention
            wl, wr = float(u0[0]), float(u0[1])
        # sign flip + rear-follower + magnitude clamp + slew limit, all in control/command.py
        cmd = condition_command(
            wl, wr, self._prev_cmd,
            max_omega=self.plan_max_omega, max_slew=self.plan_max_slew, dt=dynamics.DT,
        )
        self._prev_cmd = cmd
        self._publish_cmd(cmd)

    def _publish_cmd(self, cmd: np.ndarray) -> None:
        """Publish the conditioned [left, rear, right] wheel velocities to /cmd_joints.

        Stamped with the current clock (not the sensor stamp) so an LLC deadman sees a fresh
        command. VELOCITY ONLY: position/effort are left empty. Filling them with inf breaks
        serialization across the micro-ROS/XRCE bridge, so the LLC never receives the command
        (found live on the robot 2026-07-10)."""
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(JOINT_NAMES)
        m.velocity = [float(v) for v in cmd]
        self.pub_cmd.publish(m)

    def _publish_path(self, xy: np.ndarray, z: float, stamp) -> None:
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.map_frame
        for x, y in xy:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = z
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)
        # Same path as a thick LINE_STRIP marker (nav_msgs/Path renders as 1px GL lines).
        m = Marker()
        m.header = path.header
        m.ns = "planned_path"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = float(self.plan_path_width)
        m.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)  # magenta: reads over the green height map
        m.pose.orientation.w = 1.0
        m.points = [Point(x=float(x), y=float(y), z=z) for x, y in xy]
        self.pub_path_marker.publish(m)

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

    def _gyro_orientation_base(self, stamp) -> np.ndarray | None:
        """world_R_base (3x3) from INTEGRATING the base-frame gyro up to the cloud `stamp`.

        The motion prior's rotation source. We do NOT use the fused orientation quaternion:
        on this robot /imu/data's AHRS reports yaw with the wrong sign and attenuated (an
        ENU/NED handedness bug, with no magnetometer to anchor yaw), which drove the
        localization yaw the WRONG way and rotated the accumulated map ~20° over a spin.
        The gyro angular_velocity is correct and slip-immune — it matches wheel odom, which
        the fused orientation contradicts — so we integrate it instead.

        Only the frame-to-frame delta is consumed by predict(), so the arbitrary integration
        origin and any slow roll/pitch drift cancel: ICP's gravity prior re-anchors roll/pitch
        each frame, and yaw has no other source anyway. Advanced once per cloud (from the last
        cloud stamp to this one, piecewise over the buffered samples), so it stays correct even
        when a frame is rejected — the delta still spans the true inter-cloud rotation.

        The buffered gyro is already rotated into base_frame (see `_gyro_base_rotation`); returns
        None when the prior is disabled so predict() falls back to the pure odom delta.
        """
        if not self.imu_rotation_prior:
            return None
        t = stamp.sec + stamp.nanosec * 1e-9
        if self._gyro_R_base is None:  # seed at identity — only deltas matter downstream
            self._gyro_R_base = np.eye(3)
            self._gyro_t = t
            return self._gyro_R_base.copy()
        # Integrate omega across each buffered sample in (t_prev, t]; tail to the exact stamp.
        R = self._gyro_R_base
        tk = self._gyro_t
        for ts, _q, w in self._imu_buffer:
            if ts <= tk:
                continue
            if ts > t:
                break
            R = R @ _rodrigues(w * (ts - tk))  # body-frame rate -> right-multiply
            tk = ts
        if t > tk:
            w = self._imu_omega_at(t)
            if w is not None:
                R = R @ _rodrigues(w * (t - tk))
        self._gyro_R_base = R
        self._gyro_t = t
        return R.copy()

    def _gyro_base_rotation(self, frame_id: str) -> np.ndarray | None:
        """Cached base_R_imu (rotation only) from the static IMU mount TF, or None if not ready yet.

        The gyro angular_velocity arrives in the IMU frame — identity for `/imu/data` (imu == base)
        but a real rotation for `/ouster/imu` (os_imu). Rotating it into base_frame keeps the yaw
        axis correct for any IMU source. The mount is static, so we look it up once and cache it.
        """
        if self._base_R_gyro is not None:
            return self._base_R_gyro
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame_id, rclpy.time.Time())
        except TransformException:
            return None
        r = tf.transform.rotation
        self._base_R_gyro = quaternion_to_matrix(r.x, r.y, r.z, r.w)[:3, :3]
        return self._base_R_gyro

    def _imu_omega_at(self, t: float) -> np.ndarray | None:
        """IMU angular_velocity (rad/s, base frame) linearly interpolated to time `t`, or None.

        The gyro rate is smooth, so this is far less timing-sensitive than the orientation —
        it's what the deskew needs: the rotation rate during the sweep, no absolute-window
        integration and no dependence on the cloud header's start/end convention.
        """
        buf = self._imu_buffer
        if not buf:
            return None
        if t <= buf[0][0]:
            return buf[0][2] if buf[0][0] - t <= _IMU_MAX_EXTRAP_S else None
        if t >= buf[-1][0]:
            return buf[-1][2] if t - buf[-1][0] <= _IMU_MAX_EXTRAP_S else None
        for i in range(len(buf) - 1, 0, -1):
            t0, w0 = buf[i - 1][0], buf[i - 1][2]
            t1, w1 = buf[i][0], buf[i][2]
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return (1.0 - a) * w0 + a * w1
        return None

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

    def _range_crop(
        self, scan_base: np.ndarray, point_times: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Drop returns past scan_max_range_m (horizontal xy distance from the robot).

        Far Ouster returns are sparse grazing-angle ground — noise that only pollutes ICP
        and both maps. Cropping per scan keeps it out of every downstream stage. 0 disables;
        times stay aligned.
        """
        if self.scan_max_range_m <= 0.0:
            return scan_base, point_times
        r2 = scan_base[:, 0] ** 2 + scan_base[:, 1] ** 2
        keep = r2 <= self.scan_max_range_m * self.scan_max_range_m
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
        self, scan_base: np.ndarray, point_times: np.ndarray | None, delta: np.ndarray, stamp
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
        # Sweep rotation from the GYRO RATE: omega * sweep_duration. Uses only the per-point `t`
        # (for both the fractions and the duration) and the instantaneous angular velocity — the
        # rate is smooth, so this needs no absolute-window integration and doesn't depend on
        # whether the cloud stamp marks the sweep start or end. Keeps delta's odom translation;
        # falls back to delta's rotation if the gyro is unavailable.
        omega = self._imu_omega_at(stamp.sec + stamp.nanosec * 1e-9)
        if omega is not None:
            delta = delta.copy()
            delta[:3, :3] = _rodrigues(omega * (span * 1e-9))  # base_start_R_base_end
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
                f"Δtrans={outcome.correction_trans_m:.2f}m "
                f"rms={outcome.rms_residual_m:.3f}m converged={outcome.converged}) "
                "— using odom prediction."
            )

    def _make_tf(self, mat: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
        """Marshal a 4x4 parent_T_child pose into a stamped TF message."""
        qx, qy, qz, qw = matrix_to_quaternion(mat)
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = float(mat[0, 3])
        tf.transform.translation.y = float(mat[1, 3])
        tf.transform.translation.z = float(mat[2, 3])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        return tf

    def _cache_map_correction(self, world_T_base: np.ndarray, odom_msg: Odometry) -> None:
        """Cache the map->odom correction from a processed cloud; _odom_tf_callback broadcasts it."""
        self._map_T_odom = world_T_base @ invert_pose(self._odom_to_matrix(odom_msg))

    def _odom_tf_callback(self, odom_msg: Odometry) -> None:
        """Broadcast the pose TF at the full odom rate so base_link is dense and fresh.

        map->odom re-uses the last processed cloud's correction (it changes slowly between
        clouds); odom->base_link is this message's raw pose. Both are re-stamped at the odom
        time, so a lookup at any cloud stamp finds a bracketing sample instead of extrapolating.
        """
        stamp = odom_msg.header.stamp
        if self.publish_map_tf and self._map_T_odom is not None:
            self.tf_broadcaster.sendTransform(
                self._make_tf(self._map_T_odom, self.map_frame, odom_msg.header.frame_id, stamp)
            )
        # Close odom->base_link when the odom source broadcasts no TF (see publish_odom_tf).
        if self.publish_odom_tf:
            odom_T_base = self._odom_to_matrix(odom_msg)
            self.tf_broadcaster.sendTransform(
                self._make_tf(odom_T_base, odom_msg.header.frame_id, self.base_frame, stamp)
            )


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
