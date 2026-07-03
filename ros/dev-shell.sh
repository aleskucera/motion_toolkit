#!/usr/bin/env bash
# Source this file (do NOT execute it) to get a shell set up for building and
# running the terrain_toolkit_ros package:
#
#   source <repo>/ros/dev-shell.sh
#
# It activates the uv-created .venv at the repo root, sources ROS 2 Kilted,
# and — if the repo sits inside a colcon workspace (<ws>/src/terrain_toolkit) —
# sources the workspace install overlay as well.

_dev_shell_path="${BASH_SOURCE[0]:-$0}"
_dev_shell_dir="$(cd "$(dirname "$_dev_shell_path")" && pwd)"
_dev_shell_repo="$(cd "$_dev_shell_dir/.." && pwd)"

if [[ -f "$_dev_shell_repo/.venv/bin/activate" ]]; then
    # Activate the venv ONLY on the host. Inside the Apptainer container, the
    # venv's python is a symlink to the host's interpreter and doesn't see the
    # container's system site-packages (yaml, etc.) that ROS needs — activating
    # it would break rclpy. PYTHONPATH injection (below) is enough either way,
    # since ament_python node scripts use the /usr/bin/python3 shebang anyway.
    if [[ -z "$SINGULARITY_CONTAINER" ]]; then
        # shellcheck disable=SC1091
        source "$_dev_shell_repo/.venv/bin/activate"
    fi
    # Expose what the venv holds via PYTHONPATH. Two entries are needed:
    #   - the venv's site-packages: makes regular installs (warp, numpy, …) visible
    #   - <repo>/src: exposes terrain_toolkit itself. The venv installs it as an
    #     editable (.pth) entry, but .pth files are only processed inside "site"
    #     directories — PYTHONPATH entries are added to sys.path directly and
    #     their .pth files are ignored. Adding the source dir bypasses that.
    _dev_shell_site="$(ls -d "$_dev_shell_repo"/.venv/lib/python*/site-packages 2>/dev/null | head -1)"
    if [[ -n "$_dev_shell_site" ]]; then
        export PYTHONPATH="$_dev_shell_repo/src:$_dev_shell_site${PYTHONPATH:+:$PYTHONPATH}"
    fi
    unset _dev_shell_site
else
    echo "[dev-shell] no .venv at $_dev_shell_repo/.venv" >&2
    echo "[dev-shell] create one with: cd $_dev_shell_repo && uv venv --system-site-packages && uv pip install -e ." >&2
fi

# Pick the first available ROS 2 distro. Prefer kilted (the host target) but
# fall back to jazzy so this script also works inside the Apptainer container,
# which is jazzy-based.
_dev_shell_ros=""
for _distro in kilted jazzy; do
    if [[ -f "/opt/ros/$_distro/setup.bash" ]]; then
        _dev_shell_ros="/opt/ros/$_distro/setup.bash"
        break
    fi
done
if [[ -n "$_dev_shell_ros" ]]; then
    # shellcheck disable=SC1091
    source "$_dev_shell_ros"
else
    echo "[dev-shell] no /opt/ros/{kilted,jazzy}/setup.bash found — skipping ROS source" >&2
fi
unset _dev_shell_ros _distro

# If the repo is <ws>/src/terrain_toolkit, source <ws>/install/setup.bash when present.
_dev_shell_ws="$(cd "$_dev_shell_repo/../.." 2>/dev/null && pwd)"
if [[ -n "$_dev_shell_ws" && -f "$_dev_shell_ws/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "$_dev_shell_ws/install/setup.bash"
    echo "[dev-shell] sourced workspace overlay at $_dev_shell_ws/install"
fi

unset _dev_shell_path _dev_shell_dir _dev_shell_repo _dev_shell_ws
