#!/bin/bash
# Record elevation_node's INPUT topics for offline parameter tuning.
# Usage:  ./record_bag.sh <scenario>     (do the maneuver, then Ctrl-C to stop)
#         ./record_bag.sh                (list the standard scenarios)
# Bags land in ~/bags/<scenario>.  Add --compression-mode file --compression-format zstd
# to the ros2 command below if you want smaller (slower) bags.
set -e

# Standard tuning scenarios: name -> maneuver to perform while recording.
declare -A SCENARIOS=(
  [rotate]="in-place spin (~1 rev) — stresses the gyro rotation prior + yaw ICP (map-twist)"
  [rotate_fast]="fast in-place spin — expected to break ICP (rotational aliasing stress)"
  [slow_translate]="slow straight drive (~10 m) — accumulator lattice-snap / translation erosion"
  [slow_translate_long]="long slow straight drive (>1 min) — sustained drift / map growth"
  [dynamic]="people/objects moving through a static scene — dynamic visibility-carve tuning"
)
# Display order (assoc arrays are unordered).
ORDER=(rotate rotate_fast slow_translate slow_translate_long dynamic)

list_scenarios() {
  echo "scenarios:"
  for k in "${ORDER[@]}"; do printf "  %-22s %s\n" "$k" "${SCENARIOS[$k]}"; done
}

if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" || "$1" == "list" ]]; then
  echo "usage: record_bag.sh <scenario>   (do the maneuver, then Ctrl-C to stop)"
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

# ros2 bag record refuses to write into an existing dir. Re-recording a scenario
# must reuse the canonical name (~/bags/<name>) so replay_bag.sh <name> still finds
# it, so overwrite on confirm rather than force a rename.
DEST="$HOME/bags/$NAME"
if [[ -e "$DEST" ]]; then
  read -r -p "~/bags/$NAME already exists — overwrite? [y/N] " ans || ans=""
  [[ "$ans" == [yY]* ]] || { echo "aborted (delete it or record under a different name)."; exit 1; }
  rm -rf "$DEST"
fi

source ~/.rosrc >/dev/null 2>&1
source ~/workspaces/helhest_ws/install/setup.bash >/dev/null 2>&1
mkdir -p ~/bags
echo "recording -> ~/bags/$NAME   (Ctrl-C to stop)"
exec ros2 bag record -o ~/bags/"$NAME" \
  /ouster/points /odom_2d /imu/data /ouster/imu /tf /tf_static
