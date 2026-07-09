# helhest_stack ROS

The `elevation_node` (localization + accumulated mapping + MPPI planning viz) and the
`elevation-demo` tmuxinator. Run via the apptainer container + `dev-shell.sh` (see the
tmuxinator header). This file records **deployment gotchas** that are easy to lose hours to.

## KNOWN ISSUE: large LiDAR clouds silently dropped by DDS

**RMW note (read first):** this applies only when the RMW is **Fast DDS** (`rmw_fastrtps_cpp`)
— e.g. bag replay on a dev box. The **live robot runs Zenoh** (`rmw_zenoh_cpp`), whose
reliable-TCP transport has no such failure mode: measured same-host, a bare subscriber
received **201/201** `/ouster/points` at 10 Hz, zero loss. So `fastdds_shm.xml` below is
**inert under Zenoh** — don't chase it on the robot; check `RMW_IMPLEMENTATION` first.

**Symptom:** the node processes only a fraction of `/ouster/points` (e.g. ~40%); the
accumulated map is sparse/streaky and localization sees big per-frame rotations (ICP
rejects). Slowing the bag rate does **not** help. A bare do-nothing subscriber also only
receives a fraction — so it is **not** compute, ICP, or the node; it is the **transport**.

**Cause:** an Ouster 1024×128 cloud is **~6 MB**. With Fast DDS (the RMW in that case) over
best-effort UDP, each cloud is fragmented into thousands of packets; if the OS socket
receive buffer can't hold a whole cloud, reassembly fails and the **entire message is
dropped** — per message, regardless of playback rate. Defaults are far too small:
`net.core.rmem_max` is typically 4 MB (< one cloud) and Fast DDS's default shared-memory
segment is ~512 KB, so it silently falls back to the broken UDP path.

**Fix (same machine — lidar driver + node + rviz on one host):** use the shared-memory
transport profile `ros/fastdds_shm.xml` (64 MB SHM segment; SHM has no fragmentation and
ignores `rmem_max`). Point **every** participant at it:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE="$REPO/ros/fastdds_shm.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$REPO/ros/fastdds_shm.xml"
```

The `elevation-demo` tmuxinator already sets this in every pane. For any other launcher
(launch files, systemd, a robot bringup script) you must set it too, or clouds drop.

**Fix (multi-host — lidar and node on different machines over the network):** SHM is
same-host only. Instead raise the kernel socket buffer above one cloud (needs root):

```bash
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.rmem_default=8388608
# persist:
echo -e "net.core.rmem_max=134217728\nnet.core.rmem_default=8388608" \
  | sudo tee /etc/sysctl.d/60-ros2-pointcloud.conf
```

**Verify the fix:** a bare subscriber should receive ~all clouds. Measured on `rotate`
(325 clouds): before → 136/325 processed (58% dropped, ICP rejects); after SHM →
318/325 (2%, 0 rejects). A do-nothing subscriber went 109/325 → 325/325.

## KNOWN ISSUE: node runs but publishes nothing — cloud stamped in sensor time

**Symptom:** the node starts (Warp init + the `ElevationNode:` banner) but never publishes
`elevation_local`/`elevation_global`; RViz stays empty. `/odom_2d` and `/imu/data` are live.

**Cause:** the cloud↔odom `ApproximateTimeSynchronizer` never matches a pair because the
clocks differ. With the Ouster driver's `timestamp_mode` **unset (empty)** it falls back to
`TIME_FROM_INTERNAL_OSC` — the sensor's internal oscillator (uptime), e.g. a header stamp of
`7616 s` — while `/odom_2d` and `/imu/data` are in system time (`~1.78e9 s`). Off by ~1.78
billion seconds, so no slop can bridge it and `_process()` never fires.

**Fix:** launch the Ouster driver with `timestamp_mode:=TIME_FROM_ROS_TIME` so clouds carry
the host ROS clock (same as odom/imu). It configures the sensor at startup, so `ros2 param
set` won't take — **relaunch the driver**. Don't use `TIME_FROM_PTP_1588`: odom/imu here are
plain system time, not PTP-disciplined, so PTP clouds still wouldn't match (the `ptp4l` on
`eth_ouster` is not aligning the sensor to system UTC — the `7616 s` readback proves it).

**Verify:** `ros2 topic echo --once --field header.stamp /ouster/points` should read ~`1.78e9`,
matching `/odom_2d`.

## Other defaults worth knowing

- **Rotation prior = integrated gyro**, not the fused `/imu/data` orientation (its yaw is
  wrong-sign on this hardware — AHRS ENU/NED bug). See `elevation_node._gyro_orientation_base`.
- **Accumulator voxel grid is world-snapped** so the map does not erode under translation
  (`DeviceMapAccumulator._min_corner`).
- **Map maintenance:** consecutive-free visibility carve of dynamic obstacles ON — a point is
  dropped only after the scan sees PAST it for `carve_persist_frames` (8) frames in a row, so a
  single ambiguous no-return can't delete static geometry, while a spot a moving person vacated
  (no return for 8 straight frames) IS carved. Frontier no-return path ON (needed to carve a
  trail on open ground, where the vacated spot has no solid background); persist=8 is what makes
  it safe. Net: a moving person keeps their current pose but leaves no trail. Between-beam-gap
  age-out ON (`carve_gap_frames`, 8): a map point on a bearing no beam ever re-hits, but whose
  neighbours ARE scanned, is a stale fragment the discrete Ouster beams can't confirm or carve
  the normal way (it reads as "unobserved → keep" forever) — it is dropped after 8 frames. This
  clears the sparse elevated specks that used to sit in the robot's path; a point in a fully
  unscanned region (no scanned neighbour, outside the vertical FOV) is still held. Also: recency
  age-out OFF (erased static), reset-on-tracking-loss ON. `NO_FORGET=1` disables carve + recency
  + reset; `RECENCY=1` re-enables the age-out.
- **Node broadcasts `odom→base_link`** (`publish_odom_tf`, default on) at the full odom rate,
  because `helhest_llc` publishes the `/odom_2d` message but no TF — without it `base_link`
  is disconnected from `map` and RViz can't place/follow the robot. Set false only if the odom
  source starts broadcasting it. RViz: world-up view = Fixed Frame `map` + view Target Frame
  `base_link`; robot-up view = Fixed Frame `base_link` (needs the dense odom-rate TF above).
