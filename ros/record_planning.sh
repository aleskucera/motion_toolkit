#!/bin/bash
# Record a CLOSED-LOOP PLANNING bag: everything needed to debug why the planner
# (elevation_node MPPI) behaves unexpectedly. Captures three groups in ONE bag so the
# decision is reproducible AND observable:
#   1. INPUTS      -> replay through elevation_node to rebuild the exact map it saw
#   2. GOAL+OUTPUT -> what the planner actually decided (path/fan/cmd) for that state
#   3. EXECUTION   -> what the LLC did with the command (did /cmd_joints get executed?)
#
# Usage:  ./record_planning.sh <scenario>            (drive it, then Ctrl-C to stop)
#         ./record_planning.sh                        (list the standard scenarios)
#         WITH_MAPS=1 ./record_planning.sh <scenario> (also record the produced maps --
#                                                       big & reproducible from inputs; only
#                                                       for eyeballing without a replay)
# Bags land in ~/bags/<scenario> (data, NOT versioned -- only this script lives in the repo).
# Add --compression-mode file --compression-format zstd to the ros2 command below for
# smaller (slower) bags (/ouster/points is ~6 MB/frame).
set -e

# Standard planner-debug scenarios: name -> what to drive while recording.
declare -A SCENARIOS=(
  [goal_straight]="set a goal straight ahead on open ground — baseline: does the path track to it"
  [goal_around]="set a goal past an obstacle/berm — the path must route AROUND, not through"
  [wrong_turn]="reproduce the misbehavior: drive until the planner picks a bad path, keep recording"
  [dock]="approach a close goal (< plan_dock_radius) — MPPI->terminal-dock handoff + stop"
  [oscillate]="a goal where the command visibly hunts/jitters — slew/cost-balance tuning"
)
ORDER=(goal_straight goal_around wrong_turn dock oscillate)

list_scenarios() {
  echo "scenarios:"
  for k in "${ORDER[@]}"; do printf "  %-16s %s\n" "$k" "${SCENARIOS[$k]}"; done
}

if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" || "$1" == "list" ]]; then
  echo "usage: record_planning.sh <scenario>   (drive it, then Ctrl-C to stop)"
  list_scenarios
  exit 0
fi

NAME="$1"
if [[ -n "${SCENARIOS[$NAME]:-}" ]]; then
  echo "scenario '$NAME': ${SCENARIOS[$NAME]}"
else
  echo "note: '$NAME' is not a standard scenario (recording anyway)." >&2
  list_scenarios >&2
fi

# 1. INPUTS -- replay these through elevation_node to reproduce the map + pose it planned on.
TOPICS=(
  /ouster/points          # lidar (biggest topic; ~6 MB/frame) -- map + localization
  /odom_2d                # wheel odometry the node consumes
  /imu/data               # fused IMU (gravity prior + rotation prior)
  /ouster/imu             # sensor IMU (alternate gyro source; frames differ)
  /tf                     # dynamic frames (map->odom->base_link)
  /tf_static              # sensor->base extrinsics
  # 2. GOAL + PLANNER OUTPUTS -- what the planner was asked for and what it decided.
  /goal_pose              # the goal that triggers planning
  /planned_path           # nav_msgs/Path -- the committed path
  /planned_path_marker    # thick LINE_STRIP of the same path
  /mppi_fan               # the sampled rollout fan (why it chose that path)
  /frame_marker           # per-frame counter (sync replay to the run)
  # 3. COMMAND + EXECUTION -- did the decision reach and move the wheels?
  /cmd_joints             # the conditioned wheel-velocity command the node published
  /joint_setpoint         # per-wheel target the LLC actually drives to
  /joint_states           # actual wheel speeds (was the command executed?)
  /motors_enable          # motor-enable request
  /motors_enabled         # motor-enable state (are the motors even live?)
)

# Optional: the produced maps. Reproducible by replaying the INPUTS above, and large, so
# OFF by default -- set WITH_MAPS=1 only to eyeball them without a replay.
if [[ -n "${WITH_MAPS:-}" ]]; then
  TOPICS+=(/accumulated_map /elevation_global /elevation_local)
fi

# ros2 bag record refuses to write into an existing dir. Re-recording a scenario reuses the
# canonical name (~/bags/<name>) so replay tooling finds it -- overwrite on confirm.
DEST="$HOME/bags/$NAME"
if [[ -e "$DEST" ]]; then
  read -r -p "~/bags/$NAME already exists -- overwrite? [y/N] " ans || ans=""
  [[ "$ans" == [yY]* ]] || { echo "aborted (delete it or record under a different name)."; exit 1; }
  rm -rf "$DEST"
fi

source ~/.rosrc >/dev/null 2>&1
source ~/workspaces/helhest_ws/install/setup.bash >/dev/null 2>&1
mkdir -p ~/bags

echo
echo ">> Set the goal (RViz 2D Nav Goal), drive/observe the maneuver, Ctrl-C to stop."
echo ">> recording -> ~/bags/$NAME${WITH_MAPS:+  (+maps)}"
echo
exec ros2 bag record -o "$DEST" "${TOPICS[@]}"
