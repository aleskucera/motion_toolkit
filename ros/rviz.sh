#!/usr/bin/env bash
# Launch rviz2 with the elevation config, no matter the current directory.
# Usage:  ros/rviz.sh                 # opens the repo's elevation.rviz
#         ros/rviz.sh --extra args    # any extra args pass straight through to rviz2
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO/ros/helhest_stack_ros/rviz/elevation.rviz"

exec rviz2 -d "$CONFIG" "$@"
