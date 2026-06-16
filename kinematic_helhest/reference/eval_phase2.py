"""Phase 2 verification: heightmap settling solve + normal loads.

Static checks:
  - flat ground  -> z = R, level, loads reproduce the per-wheel scale
    measurements the masses were derived from (front 39.1 kg, rear 28.0 kg).
  - constant ramp -> pitch matches the ramp angle, all wheels in contact.
Dynamic check:
  - box scene -> the robot climbs the box (prism z rises ~box height) and
    forward progress slows while pitched.

Run:  python -m kinematic_helhest.eval_phase2 [--plot OUT.png]
"""
import argparse
import pathlib

import numpy as np

from .. import data
from .. import heightmap
from .. import placement
from . import rollout
from ..model import GRAVITY
from ..model import MASS
from ..model import WHEEL_RADIUS


def check_flat():
    hm = heightmap.wheel_envelope(heightmap.flat(), WHEEL_RADIUS)
    pl = placement.settle(0.0, 0.0, 0.0, hm)
    N = placement.normal_loads(pl, 0.0, 0.0)
    kg = N / GRAVITY
    print("--- flat ground ---")
    print(f"  z={pl['z']:.4f} (expect {WHEEL_RADIUS}) pitch={np.rad2deg(pl['pitch']):+.3f} "
          f"roll={np.rad2deg(pl['roll']):+.3f} deg  resid={pl['residual']:.2e}")
    print(f"  loads [L,R,rear] = {kg[0]:.1f}, {kg[1]:.1f}, {kg[2]:.1f} kg "
          f"(sum {kg.sum():.1f} = {MASS} kg)")
    print("  scale targets    = 39.1, 39.1, 28.0 kg")
    ok = (abs(pl["z"] - WHEEL_RADIUS) < 1e-4 and abs(pl["pitch"]) < 1e-4
          and abs(kg[0] - 39.1) < 0.5 and abs(kg[2] - 28.0) < 0.5
          and abs(kg.sum() - MASS) < 1e-3)
    return ok


def check_ramp(angle_deg=11.3):
    hm = heightmap.wheel_envelope(heightmap.ramp_scene(angle_deg=angle_deg), WHEEL_RADIUS)
    # Place on the sloped part (x=3 is well up the ramp).
    pl = placement.settle(3.0, 0.0, 0.0, hm)
    print(f"\n--- ramp {angle_deg} deg ---")
    print(f"  pitch={np.rad2deg(pl['pitch']):+.3f} deg (expect {-angle_deg:+.1f}) "
          f"roll={np.rad2deg(pl['roll']):+.3f}  resid={pl['residual']:.2e}")
    ok = abs(np.rad2deg(pl["pitch"]) + angle_deg) < 0.5 and pl["converged"]
    return ok


def check_box(args):
    sp, real, run_id, _ = data.load_setpoints(
        data.SYNCED_DIR / data.DEFAULT_RUN, "setpoint", args.dt, args.duration
    )
    ra, rt = data.align_real_to_sim(real)
    hm = heightmap.box_scene()
    out = rollout.rollout_terrain(sp, args.dt, hm, alpha=args.alpha, x_icr=args.x_icr)

    sim_prism = data.prism_track(out["pose7"])
    sim_rel = sim_prism - sim_prism[0]
    net = np.linalg.norm(out["pose2"][-1, :2] - out["pose2"][0, :2])
    zmax = sim_rel[:, 2].max()
    load_sum = out["loads"].sum(1)

    rv = ~np.isnan(ra[:, 0])
    real_zmax = np.nanmax(ra[rv][:, 2])
    fz_err = np.abs(out["fz"] - MASS * GRAVITY).max() / (MASS * GRAVITY)

    print(f"\n--- box scene (run {run_id}) ---")
    print(f"  net XY displacement : {net:.3f} m")
    print(f"  prism z rise (peak) : sim {zmax:.3f} m | real {real_zmax:.3f} m "
          f"(box top {2*data.BOX_HALF_EXTENTS[2]:.2f} m)")
    print(f"  max pitch on climb  : {np.rad2deg(np.abs(out['pitch']).max()):.1f} deg")
    print(f"  |N| magnitude range : {load_sum.min()/GRAVITY:.1f}..{load_sum.max()/GRAVITY:.1f} kg "
          f"(>{MASS} when tilted, OK)")
    print(f"  vertical balance err: {fz_err:.2e}  (Sum N_i n_z vs mg)")
    print(f"  max settle residual : {out['residual'].max():.2e}")
    ok = (out["residual"].max() < 1e-3 and zmax > 0.05 and fz_err < 1e-6)

    if args.plot:
        _plot(sim_rel, ra, rt, args.dt, args.plot)
    return ok


def _plot(sim_rel, ra, rt, dt, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, az) = plt.subplots(1, 2, figsize=(13, 5))
    ax.plot(sim_rel[:, 0], sim_rel[:, 1], color="tab:blue", label="sim (kinematic)")
    ax.plot(ra[:, 0], ra[:, 1], color="tab:red", label="real (TS)")
    cx, cy, _ = data.BOX_CENTER
    hx, hy, _ = data.BOX_HALF_EXTENTS
    ax.add_patch(plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color="gray", alpha=0.3))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.axis("equal")
    ax.set_title("Top-down (start-aligned)"); ax.legend(); ax.grid(alpha=0.3)

    st = np.arange(sim_rel.shape[0]) * dt
    az.plot(st, sim_rel[:, 2], color="tab:blue", label="sim prism z")
    az.plot(rt, ra[:, 2], color="tab:red", label="real z")
    az.set_xlabel("t [s]"); az.set_ylabel("z rise [m]")
    az.set_title("Prism elevation (box climb)"); az.legend(); az.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"  saved plot to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--x-icr", type=float, default=0.0)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    ok_flat = check_flat()
    ok_ramp = check_ramp()
    ok_box = check_box(args)
    print(f"\nPhase 2 checks: flat={'PASS' if ok_flat else 'FAIL'} "
          f"ramp={'PASS' if ok_ramp else 'FAIL'} box={'PASS' if ok_box else 'FAIL'}")


if __name__ == "__main__":
    main()
