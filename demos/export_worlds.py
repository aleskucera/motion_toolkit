"""Export planner-driven rollouts through the stress worlds to Blender ``.npz`` files.

Runs the closed-loop nav harness (``demos/eval.evaluate``) on each world -- MPPI + cost-to-go
routing + terminal dock, the same loop as the real driver -- and writes one ``.npz`` per world for
the Blender pipeline (``viz/blender_import.py``). CUDA required (the planner runs on the GPU).

    python demos/export_worlds.py --all --out /tmp/worlds
    python demos/export_worlds.py --world pocket --out /tmp/worlds

Then render one, e.g.:  ./scripts/render_dasenka.sh /tmp/worlds/pocket.npz 1
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import warp as wp
from eval import evaluate  # sibling demo (demos/ is on sys.path when run as a script)
from kinematic_helhest import worlds as W
from kinematic_helhest.model import euler_zyx
from kinematic_helhest.viz.blender_export import _mat_to_quat
from kinematic_helhest.viz.blender_export import write_npz


def build_frames(poses: list, cmds: list, dt: float) -> dict[str, np.ndarray]:
    """Recorded (pose per frame, cmd per step) -> the per-frame arrays write_npz wants."""
    n = len(poses)
    pos = np.zeros((n, 3))
    euler = np.zeros((n, 3))  # (yaw, pitch, roll)
    quat = np.zeros((n, 4))
    spin = np.zeros((n, 3))  # integrated wheel angle (left, right, rear)
    valid = np.zeros(n, dtype=bool)
    for i, (x, y, z, yaw, pitch, roll, v) in enumerate(poses):
        pos[i] = (x, y, z)
        euler[i] = (yaw, pitch, roll)
        quat[i] = _mat_to_quat(euler_zyx(yaw, pitch, roll))
        valid[i] = v
    for k, cmd in enumerate(cmds):
        spin[k + 1] = spin[k] + np.asarray(cmd) * dt
    return {"pos": pos, "euler": euler, "quat": quat, "wheel_spin": spin, "valid": valid}


def export_world(name: str, out_dir: str, device: str, **kw) -> None:
    r = evaluate(name, device=device, record=True, **kw)
    frames = build_frames(r["poses"], r["cmds"], r["dt"])
    path = os.path.join(out_dir, f"{name}.npz")
    write_npz(path, frames, r["scene"], r["dt"])
    print(f"{name:9s} reached={str(r['reached']):5s} frames={len(frames['pos']):4d} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default=None, choices=list(W.WORLDS))
    ap.add_argument("--all", action="store_true", help="export every stress world")
    ap.add_argument("--out", default="/tmp/worlds", help="output directory for the .npz files")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--K", type=int, default=8, help="CVaR robust scenarios (1 = off)")
    ap.add_argument("--dock-radius", type=float, default=1.5)
    ap.add_argument("--lat-coarsen", type=int, default=1)
    args = ap.parse_args()

    if not args.world and not args.all:
        ap.error("pass --world <name> or --all")
    wp.init()
    os.makedirs(args.out, exist_ok=True)
    kw = dict(K=args.K, dock_radius=args.dock_radius, lat_coarsen=args.lat_coarsen)
    names = list(W.WORLDS) if args.all else [args.world]
    for name in names:
        export_world(name, args.out, args.device, **kw)


if __name__ == "__main__":
    main()
