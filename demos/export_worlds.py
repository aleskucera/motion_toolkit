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
from helhest import worlds as W
from helhest.model import euler_zyx
from helhest.viz.blender_export import _mat_to_quat
from helhest.viz.blender_export import write_npz


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


def build_fan_extra(fans: list, ctg: dict) -> dict[str, np.ndarray]:
    """Recorded fan snapshots + static routing field -> colored arrays for blender_import.

    Fan rollouts are colored by cost (green = low -> red = high). The green/red band is normalized
    ONCE over the pooled costs of every snapshot, so a given cost maps to the SAME color in every
    frame -- a consistently-bad candidate stays red across the animation (a per-snapshot rescale
    would recolor it frame-to-frame). The ground cost-to-go uses viridis (cheap -> far), dark where
    unreachable.
    """
    from matplotlib import cm

    fan_frame = np.array([s["frame"] for s in fans], np.int32)
    fan_xyz = np.stack([s["paths"] for s in fans])  # [F, N, P, 3]
    nominal_xyz = np.stack([s["nominal"] for s in fans])  # [F, P, 3]
    fan_rgb = np.zeros(fan_xyz.shape[:2] + (3,), np.float32)  # [F, N, 3]
    all_cost = np.concatenate([s["cost"] for s in fans])
    lo, hi = np.percentile(all_cost, 25), np.percentile(all_cost, 75)  # one band, shared by all frames
    for i, s in enumerate(fans):
        c = s["cost"]
        norm = np.clip((c - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.full_like(c, 0.5)
        fan_rgb[i] = cm.RdYlGn_r(norm)[:, :3]  # green = low cost (good) -> red = high (bad)

    V = ctg["Vmin"]
    reach = V < ctg["vcap"] * 0.9
    vmax = float(np.percentile(V[reach], 95)) if reach.any() else 1.0
    ctg_rgb = cm.viridis(np.clip(V / max(vmax, 1e-6), 0.0, 1.0))[:, :, :3].astype(np.float32)
    ctg_rgb *= 0.4  # dim the ground so it reads as a background behind the glowing fan
    ctg_rgb[~reach] = (0.04, 0.04, 0.06)

    return dict(
        fan_frame=fan_frame,
        fan_xyz=fan_xyz.astype(np.float32),
        fan_rgb=fan_rgb,
        nominal_xyz=nominal_xyz.astype(np.float32),
        ctg_rgb=ctg_rgb,
    )


def export_world(name: str, out_dir: str, device: str, fan: bool, **kw) -> None:
    r = evaluate(name, device=device, record=True, record_fan=fan, **kw)
    frames = build_frames(r["poses"], r["cmds"], r["dt"])
    extra = build_fan_extra(r["fans"], r["ctg"]) if fan and r["fans"] else None
    path = os.path.join(out_dir, f"{name}.npz")
    write_npz(path, frames, r["scene"], r["dt"], extra=extra)
    tag = f" fan={len(r['fans'])}" if fan else ""
    print(f"{name:9s} reached={str(r['reached']):5s} frames={len(frames['pos']):4d}{tag} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default=None, choices=list(W.WORLDS))
    ap.add_argument("--all", action="store_true", help="export every stress world")
    ap.add_argument("--out", default="/tmp/worlds", help="output directory for the .npz files")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fan", action="store_true", help="also record the MPPI fan + cost-to-go")
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
        export_world(name, args.out, args.device, args.fan, **kw)


if __name__ == "__main__":
    main()
