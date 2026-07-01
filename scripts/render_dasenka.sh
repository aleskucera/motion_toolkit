#!/usr/bin/env bash
# Render a motion_toolkit rollout on the dasenka multi-GPU box.
#
# Copies the (self-contained) blender_import.py + the rollout .npz to dasenka,
# renders headless on a pinned GPU, and copies the result back. The .npz already
# carries the terrain + robot geometry, so nothing needs installing on dasenka
# beyond Blender. GPU pinning (VK_DEVICE_INDEX + --gpu-backend vulkan) mirrors
# ostrich/experiments/2_dt_stability/render_dasenka.sh so EEVEE Next (Vulkan in
# Blender 5.x) doesn't fight CUDA workloads on GPU 0.
#
# Usage:
#   ./scripts/render_dasenka.sh <rollout.npz> [gpu_index] [-- extra blender_import args]
#   OUT=~/clip.mp4 RES=1920x1080 ./scripts/render_dasenka.sh rollout.npz 2
#
# Defaults: gpu_index=1, OUT=./dasenka_render.mp4, RES=1280x720, playback 30 fps
#           (pass `-- --fps N` to change it), HOST=dasenka, REMOTE_DIR=/local/kuceral4/mt_render.
#
# Extra args are forwarded to blender_import.py verbatim (e.g. --robot ...); any
# files they reference must already exist on dasenka.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPORT_PY="$HERE/../src/kinematic_helhest/viz/blender_import.py"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <rollout.npz> [gpu_index] [-- extra blender_import args]" >&2
    exit 1
fi
NPZ="$1"; shift
GPU_INDEX="${1:-1}"
if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then shift; fi
[[ "${1:-}" == "--" ]] && shift
EXTRA_ARGS=("$@")

[[ -f "$NPZ" ]]       || { echo "error: npz not found: $NPZ" >&2; exit 1; }
[[ -f "$IMPORT_PY" ]] || { echo "error: blender_import.py not found: $IMPORT_PY" >&2; exit 1; }

HOST="${HOST:-dasenka}"
REMOTE_DIR="${REMOTE_DIR:-/local/kuceral4/mt_render}"
OUT="${OUT:-./dasenka_render.mp4}"
RES="${RES:-1280x720}"

# dasenka's ssh config sets a RemoteCommand (auto-cd); disable it for one-off commands.
SSH=(ssh -o RemoteCommand=none -o RequestTTY=no)
SCP=(scp -o RemoteCommand=none)

npz_base="$(basename "$NPZ")"
remote_mp4="$REMOTE_DIR/render.mp4"

# The ffmpeg encode rate must match blender_import's --fps (default 30) so timing is right.
render_fps=30
for i in "${!EXTRA_ARGS[@]}"; do
    [[ "${EXTRA_ARGS[$i]}" == "--fps" ]] && render_fps="${EXTRA_ARGS[$((i + 1))]:-30}"
done

echo "[1/4] staging on $HOST:$REMOTE_DIR (GPU $GPU_INDEX)"
# clear stale outputs so a prior render.mp4 can't be mistaken for this run's result
"${SSH[@]}" "$HOST" "mkdir -p '$REMOTE_DIR' && rm -f '$REMOTE_DIR'/render.mp4 '$REMOTE_DIR'/render_*.png"
"${SCP[@]}" "$IMPORT_PY" "$NPZ" "$HOST:$REMOTE_DIR/"

# If extra args reference a local --robot .blend, ship it too and repoint at the remote copy.
for i in "${!EXTRA_ARGS[@]}"; do
    if [[ "${EXTRA_ARGS[$i]}" == "--robot" ]]; then
        j=$((i + 1))
        local_robot="${EXTRA_ARGS[$j]:-}"
        if [[ -n "$local_robot" && -f "$local_robot" ]]; then
            echo "      shipping model $(basename "$local_robot")"
            "${SCP[@]}" "$local_robot" "$HOST:$REMOTE_DIR/"
            EXTRA_ARGS[$j]="$REMOTE_DIR/$(basename "$local_robot")"
        fi
    fi
done

echo "[2/4] rendering on $HOST (Vulkan / EEVEE Next, GPU $GPU_INDEX)"
"${SSH[@]}" "$HOST" "cd '$REMOTE_DIR' && \
    VK_DEVICE_INDEX='$GPU_INDEX' CUDA_VISIBLE_DEVICES='$GPU_INDEX' \
    blender --gpu-backend vulkan -b --python blender_import.py -- \
    --data '$npz_base' --render '$remote_mp4' --res '$RES' ${EXTRA_ARGS[*]:-}"

echo "[3/4] encoding on $HOST"
# Blender may lack FFMPEG (writes render_*.png); if so, encode with dasenka's ffmpeg so we
# ship back one MP4 instead of 91 PNGs.
"${SSH[@]}" "$HOST" "cd '$REMOTE_DIR' && \
    if [ ! -f render.mp4 ] && ls render_*.png >/dev/null 2>&1; then \
        ffmpeg -y -framerate '$render_fps' -pattern_type glob -i 'render_*.png' \
            -c:v libx264 -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' render.mp4; \
    fi"

echo "[4/4] fetching result -> $OUT"
mkdir -p "$(dirname "$OUT")"
if "${SSH[@]}" "$HOST" "test -f '$remote_mp4'"; then
    "${SCP[@]}" "$HOST:$remote_mp4" "$OUT"
    echo "wrote $OUT"
else  # no ffmpeg anywhere on dasenka -> fall back to pulling the PNG frames
    "${SCP[@]}" "$HOST:$REMOTE_DIR/render_*.png" "$(dirname "$OUT")/"
    echo "no ffmpeg on $HOST; copied PNG frames next to $OUT"
fi