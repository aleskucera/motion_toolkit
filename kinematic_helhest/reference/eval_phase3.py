"""Phase 3 verification: chassis non-penetration / high-center detection.

Scope: wheels stay grounded (assumption 1); chassis non-penetration is a
feasibility signal (clearance + penetration depth) for planning, not a
pose-modifier. The pose-resolving wheel-lift active-set is deferred to the
FB-NCP Warp port (Phase 5), where complementarity makes it natural.

Checks:
  - box scene rollout -> belly clears the box throughout (no false high-center).
  - static block under the belly -> clearance goes negative, high-center fires
    with the right penetration depth.

Run:  python -m kinematic_helhest.eval_phase3
"""
import numpy as np

from .. import data
from .. import heightmap
from .. import placement
from . import rollout
from ..model import WHEEL_RADIUS


def check_box_benign():
    sp, _, run_id, _ = data.load_setpoints(
        data.SYNCED_DIR / data.DEFAULT_RUN, "setpoint", 0.05, 5.0
    )
    hm = heightmap.box_scene()
    out = rollout.rollout_terrain(sp, 0.05, hm)
    cmin = out["chassis_clear"].min()
    n_hc = int(out["high_center"].sum())
    print(f"--- box scene (run {run_id}) ---")
    print(f"  min chassis clearance : {cmin:.3f} m  (belly clears box: expect > 0)")
    print(f"  high-center steps     : {n_hc} / {len(out['high_center'])}")
    return cmin > 0.0 and n_hc == 0


def check_static_high_center(block_h=0.30):
    """Robot level on flat ground (z=R); a tall block sits under the belly."""
    hm = heightmap.flat()
    # Raise a small patch under the front-box belly center (x≈-0.13, y≈0).
    XX = hm.x0 + np.arange(hm.nx) * hm.cell
    YY = hm.y0 + np.arange(hm.ny) * hm.cell
    gx, gy = np.meshgrid(XX, YY)
    patch = (np.abs(gx - (-0.13)) <= 0.08) & (np.abs(gy - 0.0) <= 0.08)
    hm.H[patch] = block_h

    R = placement.euler_zyx(0.0, 0.0, 0.0)  # level
    z = WHEEL_RADIUS                          # wheels on flat ground beside block
    cc, world = placement.chassis_clearance(R, 0.0, 0.0, z, hm)
    cmin = cc.min()
    # Belly bottom sits at z - 0.10 = 0.25; block top 0.30 -> penetrate 0.05.
    expect = (z - 0.10) - block_h
    print(f"\n--- static high-center (block {block_h} m under belly) ---")
    print(f"  belly bottom z={z - 0.10:.3f} m, block top={block_h:.3f} m")
    print(f"  min chassis clearance : {cmin:+.3f} m (expect {expect:+.3f})")
    print(f"  high-center detected  : {cmin < 0.0}")
    return cmin < 0.0 and abs(cmin - expect) < 1e-6


def main():
    ok_box = check_box_benign()
    ok_hc = check_static_high_center()
    print(f"\nPhase 3 checks: box_benign={'PASS' if ok_box else 'FAIL'} "
          f"high_center={'PASS' if ok_hc else 'FAIL'}")


if __name__ == "__main__":
    main()
