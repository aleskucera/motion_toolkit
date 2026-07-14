#!/bin/bash
# Record a MANUAL-DRIVE characterization bag (open-loop, human on the remote).
# Pairs commanded wheel omegas -> actual wheel speeds -> gyro/odom so we can fit the
# real turn gain, stopping distance, and localization sanity. See bag_recording_brief.md.
#
# Usage:  ./record_drive.sh <maneuver_surface>   (drive it, then Ctrl-C to stop)
#         ./record_drive.sh                       (list the standard maneuvers)
#
# Name each bag <maneuver>_<detail>_<surface>, e.g. arc_diff0.5_slow_concrete,
# stop_from_fast_grass, turn_in_place_left_gravel. Surface matters (friction).
# Bags land in ~/bags/<name>.  Add --compression-mode file --compression-format zstd
# to the ros2 command below for smaller (slower) bags.
set -e

# The command<->response set. Commanded targets AND actual states must be in the SAME
# bag or the maneuver is un-analyzable. Edit here if a topic name differs on the robot.
TOPICS=(
  # --- commanded wheel omegas (the control input) ---
  /joint_setpoint         # per-wheel target the LLC drives to (100 Hz) -- the command
  /cmd_joints             # incoming per-wheel request (populates while driving)
  /cmd_vel                # high-level remote command, if published
  # --- actual wheel speeds (the response) ---
  /joint_states
  # --- localization / odometry output ---
  /odom_2d
  # --- gyro (both sources; frames differ) ---
  /imu/data
  /ouster/imu
  # --- lidar (for localization + map sanity replay) ---
  /ouster/points
  # --- frames + motor state ---
  /tf
  /tf_static
  /motors_enable
  /motors_enabled
)

# Standard maneuvers: name -> what to drive while recording (from the brief).
declare -A MANEUVERS=(
  [arc]="fixed-differential arc: hold a CONSTANT L/R wheel-speed diff, sweep a steady circle several sec (repeat gentle->sharp, ~2 speeds) -- the key turn-gain bag"
  [straight]="straight line at a constant speed, several sec -- forward omega -> ground speed + wheel slip"
  [stop_from_fast]="reach a representative speed on a straight, then command STOP -- stopping distance = dock overshoot budget"
  [turn_in_place]="steady in-place spin, left then right -- the skid regime the model is least faithful to"
  [traverse]="one natural line near real terrain/obstacles -- replay to eyeball planner fan/path vs. the human line"
)
ORDER=(arc straight stop_from_fast turn_in_place traverse)

list_maneuvers() {
  echo "standard maneuvers (name your bag <maneuver>_<detail>_<surface>):"
  for k in "${ORDER[@]}"; do printf "  %-16s %s\n" "$k" "${MANEUVERS[$k]}"; done
}

if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" || "$1" == "list" ]]; then
  echo "usage: record_drive.sh <maneuver_surface>   (drive it, then Ctrl-C to stop)"
  list_maneuvers
  exit 0
fi

NAME="$1"
BASE="${NAME%%_*}"   # match the leading maneuver token for the hint
if [[ -n "${MANEUVERS[$BASE]:-}" ]]; then
  echo "maneuver '$BASE': ${MANEUVERS[$BASE]}"
else
  echo "note: '$NAME' does not start with a standard maneuver (recording anyway)." >&2
  list_maneuvers >&2
fi

# ros2 bag record refuses to write into an existing dir; overwrite on confirm so a
# re-run reuses the canonical ~/bags/<name> (replay tooling finds it by name).
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
echo ">> Hold STILL ~2-3 s for a clean zero baseline, then drive the maneuver,"
echo ">> then hold still ~2-3 s at the end. Ctrl-C to stop."
echo ">> recording -> ~/bags/$NAME"
echo
exec ros2 bag record -o "$DEST" "${TOPICS[@]}"
