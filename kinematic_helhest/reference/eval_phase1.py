"""Phase 1 verification: flat-ground kinematic rollout vs the recorded run.

Drives the kinematic twin with run 18_04_51's recorded wheel-velocity setpoints
(ideal no-slip: alpha=1, x_ICR=0) over a flat heightmap and compares the
chassis/prism trajectory to the total-station ground truth. Validates the data
pipeline (order/sign/units) and the twist model against real cruise speed.

Run:  python -m kinematic_helhest.eval_phase1 [--run PATH] [--plot OUT.png]
"""
import argparse
import pathlib

import numpy as np

from .. import data
from .. import heightmap
from . import rollout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=str(data.SYNCED_DIR / data.DEFAULT_RUN))
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--x-icr", type=float, default=0.0)
    ap.add_argument("--plot", default=None, help="save a top-down sim-vs-real PNG")
    args = ap.parse_args()

    setpoints, real, run_id, _ = data.load_setpoints(
        pathlib.Path(args.run), "setpoint", args.dt, args.duration
    )
    real_aligned, real_t = data.align_real_to_sim(real)
    print(f"Run {run_id}: {setpoints.shape[0]} steps @ dt={args.dt}s "
          f"(alpha={args.alpha}, x_ICR={args.x_icr})")

    out = rollout.rollout_terrain(setpoints, args.dt, heightmap.flat(),
                                  alpha=args.alpha, x_icr=args.x_icr)
    pose2 = out["pose2"]

    # Sim prism track (start-relative), mirroring replay_real's comparison.
    sim_prism = data.prism_track(out["pose7"])
    sim_rel = sim_prism - sim_prism[0]

    net_sim = np.linalg.norm(pose2[-1, :2] - pose2[0, :2])
    rv = ~np.isnan(real_aligned[:, 0])
    net_real = np.linalg.norm(real_aligned[rv][-1, :2] - real_aligned[rv][0, :2])
    yaw_sim = np.rad2deg(pose2[-1, 2] - pose2[0, 2])

    print(f"\nNet XY displacement : sim {net_sim:.3f} m | real {net_real:.3f} m")
    print(f"Net yaw change (sim): {yaw_sim:+.1f} deg  (robot drives ~straight)")

    deco = rollout.cruise_decomposition(pose2, setpoints, args.dt)
    if deco:
        print("\n--- flat-ground cruise (x < 0.9 m) ---")
        print(f"  commanded wheel speed : {deco['commanded_wheel_speed']:.3f} rad/s")
        print(f"  no-slip ground (w*R)  : {deco['noslip_ground_speed']:.3f} m/s")
        print(f"  realized ground speed : {deco['ground_speed']:.3f} m/s")
        rspd = np.linalg.norm(np.diff(real_aligned[rv][:, :2], axis=0), axis=1) / np.diff(
            real_t[rv]
        )
        rt = real_t[rv][:-1]
        rwin = (rt > 0.3) & (rt < 1.7)
        if rwin.sum() > 3:
            print(f"  [real ground speed    : {np.nanmedian(rspd[rwin]):.3f} m/s]")

    # Pass/fail: kinematic cruise ~0.40 m/s and near-straight heading.
    ok = deco and abs(deco["ground_speed"] - 0.40) < 0.08 and abs(yaw_sim) < 25
    print(f"\nPhase 1 check: {'PASS' if ok else 'REVIEW'}")

    if args.plot:
        _plot(sim_rel, real_aligned, args.plot)


def _plot(sim_rel, real_aligned, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sim_rel[:, 0], sim_rel[:, 1], "-", color="tab:blue", label="sim (kinematic)")
    ax.plot(real_aligned[:, 0], real_aligned[:, 1], "-", color="tab:red", label="real (TS)")
    cx, cy, _ = data.BOX_CENTER
    hx, hy, _ = data.BOX_HALF_EXTENTS
    ax.add_patch(plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color="gray", alpha=0.3))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Phase 1: flat-ground kinematic vs real (start-aligned)")
    ax.axis("equal"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
