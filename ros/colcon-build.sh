#!/usr/bin/env bash
# Build the ROS package(s) with colcon running UNDER the repo .venv python.
#
# WHY THIS EXISTS: colcon runs `setup.py install` with `sys.executable` -- the interpreter
# running the colcon process (colcon_core/task/python/build.py: `_PYTHON_CMD = [sys.executable, ...]`).
# The system `colcon` (/usr/bin/colcon) runs under /usr/bin/python3, so setuptools stamps
# `#!/usr/bin/python3` into every console-script wrapper. That interpreter has no `warp`, so
# `ros2 run helhest_stack_ros elevation_node` then dies with:
#     ModuleNotFoundError: No module named 'warp'
# Running colcon under <repo>/.venv/bin/python instead makes the generated shebang point at the
# venv (which has warp/numpy). The .venv is --system-site-packages, so colcon itself imports fine.
#
# Usage:  ros/colcon-build.sh                         # builds helhest_stack_ros (default)
#         ros/colcon-build.sh --packages-select foo   # any colcon build args pass straight through
# NB: no `-u` -- ROS setup.bash references unbound vars (AMENT_TRACE_SETUP_FILES) and would abort.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENVPY="$REPO/.venv/bin/python"
WS="$(cd "$REPO/../.." && pwd)"  # <ws>/src/helhest_stack -> <ws>

if [[ ! -x "$VENVPY" ]]; then
  echo "[colcon-build] no venv python at $VENVPY" >&2
  echo "[colcon-build] create it: cd $REPO && uv venv --system-site-packages && uv pip install -e ." >&2
  exit 1
fi

# ROS base only -- NOT the install overlay (building with the overlay sourced is discouraged).
for _d in kilted jazzy; do
  if [[ -f "/opt/ros/$_d/setup.bash" ]]; then
    # shellcheck disable=SC1090
    source "/opt/ros/$_d/setup.bash"
    break
  fi
done

# Default to the one package we care about; any explicit args replace the default.
if [[ $# -eq 0 ]]; then
  set -- --packages-select helhest_stack_ros
fi

cd "$WS"
exec "$VENVPY" "$(command -v colcon)" build "$@"
