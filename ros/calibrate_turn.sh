#!/bin/bash
# Calibrate the skid-steer turn gain K_TURN from a short turn-heavy drive.
#
#   ./calibrate_turn.sh record <name>   # drive turns, Ctrl-C to stop -> ~/bags/<name>
#   ./calibrate_turn.sh fit <bag> [--mu 0.8]   # fit K_TURN from a recorded bag (no robot/GPU)
#   ./calibrate_turn.sh <name>          # record then immediately fit
#
# The fit uses the IMU gyro as yaw ground truth (validated ~4% vs the tuned ICP), so it needs
# nothing but the bag -- no node, GPU, or container. It auto-detects the wheel convention and the
# yaw gyro axis. Drive a TURN-HEAVY maneuver (arcs + spins, both directions), ~20-40 s.
# Result feeds dynamics.K_TURN (or the node's terrain param). Record ONE bag per surface.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv/bin/python"
# gyro-based fit -> no lidar needed, so these bags stay small.
TOPICS=(/joint_states /imu/data /ouster/imu /odom_2d)

_setup_ros() {
  source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/kilted/setup.bash 2>/dev/null
  export FASTRTPS_DEFAULT_PROFILES_FILE="$REPO/ros/fastdds_shm.xml"
  export FASTDDS_DEFAULT_PROFILES_FILE="$REPO/ros/fastdds_shm.xml"
}

_record() {
  local out="$HOME/bags/$1"
  _setup_ros
  echo ">>> recording -> $out"
  echo ">>> DRIVE TURNS NOW (arcs + spins, both directions), ~20-40 s. Ctrl-C to stop."
  ros2 bag record -o "$out" "${TOPICS[@]}" || true   # Ctrl-C finalizes the bag
}

_fit() {
  [ -x "$VENV" ] || { echo "no repo .venv python at $VENV (run: uv venv && uv pip install -e .)"; exit 1; }
  "$VENV" -c "import rosbags" 2>/dev/null || {
    echo ">>> installing rosbags into the repo .venv ..."
    uv pip install --python "$VENV" rosbags >/dev/null 2>&1 || "$VENV" -m pip install rosbags >/dev/null 2>&1 \
      || { echo "could not install rosbags -- run: uv pip install --python $VENV rosbags"; exit 1; }
  }
  "$VENV" "$REPO/ros/calibrate_turn_fit.py" "$@"
}

case "${1:-}" in
  record) [ -n "$2" ] || { echo "usage: $0 record <name>"; exit 1; }; _record "$2" ;;
  fit)    [ -n "$2" ] || { echo "usage: $0 fit <bag>"; exit 1; }; _fit "${@:2}" ;;
  "")     echo "usage: $0 record <name> | fit <bag> | <name> (record+fit)"; exit 1 ;;
  *)      _record "$1"; echo; _fit "$HOME/bags/$1" ;;
esac
