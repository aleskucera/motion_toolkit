"""Phase 4 verification: friction field -> turning-param map.

Map unit checks (on flat-ground loads N = [39.1, 39.1, 28.0]*g):
  - uniform mu      -> alpha = 1 + k*mu, x_ICR = CoM_x
  - slippery rear   -> alpha smaller, x_ICR pulled toward the front axle (-> 0)
  - grippy rear     -> alpha larger,  x_ICR pulled toward the rear (-> more neg)
End-to-end check:
  - low-friction rear strip yaws MORE than uniform friction (the mu_rear finding).

Run:  python -m kinematic_helhest.eval_phase4
"""
import numpy as np

from .. import data
from .. import friction
from .. import heightmap
from .. import placement
from . import rollout
from . import turning
from ..model import COM
from ..model import GRAVITY


def _flat_loads():
    hm = heightmap.wheel_envelope(heightmap.flat(), 0.35)
    pl = placement.settle(0.0, 0.0, 0.0, hm)
    return placement.normal_loads(pl, 0.0, 0.0)  # [3] Newtons


def check_map(k=2.0):
    N = _flat_loads()
    mu0 = 0.8
    a_u, x_u = turning.turning_params([mu0, mu0, mu0], N, k)
    a_slip, x_slip = turning.turning_params([0.8, 0.8, 0.3], N, k)   # slippery rear
    a_grip, x_grip = turning.turning_params([0.8, 0.8, 1.5], N, k)   # grippy rear

    print("--- map unit checks (k=%.1f) ---" % k)
    print(f"  uniform mu=0.8 : alpha={a_u:.3f} (expect {1+k*mu0:.3f})  "
          f"x_ICR={x_u:+.3f} (expect CoM_x {COM[0]:+.3f})")
    print(f"  slippery rear  : alpha={a_slip:.3f}  x_ICR={x_slip:+.3f}")
    print(f"  grippy rear    : alpha={a_grip:.3f}  x_ICR={x_grip:+.3f}")

    ok_uniform = abs(a_u - (1 + k * mu0)) < 1e-9 and abs(x_u - COM[0]) < 1e-9
    # Slippery rear: less total grip -> smaller alpha (more yaw); x_ICR toward 0.
    ok_slip = a_slip < a_u and x_slip > x_u
    # Grippy rear: more grip -> larger alpha; x_ICR more negative (toward rear).
    ok_grip = a_grip > a_u and x_grip < x_u
    return ok_uniform and ok_slip and ok_grip


def check_end_to_end(k=2.0):
    sp, _, run_id, _ = data.load_setpoints(
        data.SYNCED_DIR / data.DEFAULT_RUN, "setpoint", 0.05, 5.0
    )
    hm = heightmap.flat()
    mu_uniform = friction.uniform(0.8)
    mu_strip = friction.with_strip(mu_uniform, low=0.3)  # slippery rear track

    out_u = rollout.rollout_terrain(sp, 0.05, hm, mu_field=mu_uniform, k=k)
    out_s = rollout.rollout_terrain(sp, 0.05, hm, mu_field=mu_strip, k=k)
    yaw_u = np.rad2deg(out_u["pose2"][-1, 2] - out_u["pose2"][0, 2])
    yaw_s = np.rad2deg(out_s["pose2"][-1, 2] - out_s["pose2"][0, 2])

    print(f"\n--- end-to-end (run {run_id}) ---")
    print(f"  uniform mu=0.8     : net yaw {yaw_u:+.1f} deg  "
          f"(alpha~{out_u['alpha'].mean():.2f}, x_ICR~{out_u['x_icr'].mean():+.3f})")
    print(f"  slippery rear strip: net yaw {yaw_s:+.1f} deg  "
          f"(alpha~{out_s['alpha'].mean():.2f}, x_ICR~{out_s['x_icr'].mean():+.3f})")
    print(f"  -> slippery rear turns {'MORE' if abs(yaw_s) > abs(yaw_u) else 'LESS'} "
          "(expect MORE)")
    return abs(yaw_s) > abs(yaw_u)


def main():
    ok_map = check_map()
    ok_e2e = check_end_to_end()
    print(f"\nPhase 4 checks: map={'PASS' if ok_map else 'FAIL'} "
          f"end_to_end={'PASS' if ok_e2e else 'FAIL'}")


if __name__ == "__main__":
    main()
