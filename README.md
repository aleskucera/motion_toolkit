# Helhest Junior — kinematic differentiable simulator

A fast, differentiable, **purely kinematic** twin of the Helhest Junior for
planning. No Newton/Ostrich, no dynamics. The robot is a rigid tripod:

- **Controlled DOF** `(x, y, yaw)` — driven by the 3 wheels via no-slip
  differential-drive kinematics, with friction-dependent turning captured by two
  ICR parameters `(alpha, x_ICR)` derived from a per-cell friction field.
- **Derived DOF** `(z, roll, pitch)` — from a quasi-static settling solve
  against a heightmap (FB-NCP; wheels bilateral, chassis unilateral).

Differentiable w.r.t. the heightmap `h`, the friction field `mu`, and the
turning coefficient `k` (implicit gradients through the settling solve).
Batched over many rollouts for sampling-based planning.

## Install

```bash
uv sync                       # core (numpy + warp-lang)
uv sync --extra viz --extra data   # + viewers (glfw/PyOpenGL/matplotlib) + rosbag loader (h5py)
```
or plain pip: `pip install -e ".[viz,data]"`.

## Run

```bash
# interactive driver (I/J/K/L drive, mouse orbit, scroll zoom, Q quit)
python -m kinematic_helhest.drive_warp            # Warp engine (--device cuda for GPU)
python -m kinematic_helhest.follow                # drive + follow a live MPPI plan
python -m kinematic_helhest.mppi --animate        # sampling planner -> GIF

# verify the Warp engine against the numpy oracle (CPU)
python -m kinematic_helhest.warp_engine.kinematics
python -m kinematic_helhest.warp_engine.implicit
```

The package is `kinematic_helhest`; `warp_engine/` is the runtime engine,
`reference/` is the numpy finite-difference oracle (verification only).

## Phases

This package is built in phases, each independently verifiable:

| Phase | Content | Verify | Status |
|-------|---------|--------|--------|
| 0 | scaffold, heightmap rasterizer + bilinear sampler, rosbag loader | height under wheel matches scene; run loads | ✅ |
| 1 | flat-ground forward twist, scalar `(alpha, x_ICR)` | reproduces ~0.40 m/s cruise on run 18_04_51 | ✅ |
| 2 | heightmap placement (settle), wheels bilateral, normal loads `N_i`, sphere-wheel envelope | flat→level; ramp→pitched; loads=scale meas.; box climbs | ✅ |
| 3 | chassis non-penetration = post-check only → high-center ⇒ reject trajectory (`valid` flag). Settle stays wheels-only (no hard constraint) | benign→valid; tall block→high-center w/ depth | ✅ |
| 4 | per-cell `mu` field + moment-centroid turning map | uniform→`1+k·mu`/CoM_x; slippery rear turns more; signs correct | ✅ |
| 5 | implicit gradients (`d/dh`, `d/dmu`, `d/dk`), BPTT | finite-diff check < 1e-2 | ⬜ |
| 6 | calibration vs rosbags | RMSE ≤ full-physics bar; cross-run | ⬜ |
| 7 | speed benchmark | orders faster than Ostrich replay | ⬜ |
| 8 | planning demo (MPPI / gradient) | reaches goal, avoids high-center | ⬜ |

Geometry/masses are pulled from `examples/helhest_junior/common.py`
(`HelhestJuniorConfig`); the wheel order/sign remap from
`examples/helhest_junior/replay_real.py` is reproduced in `data.py`.

## Known limitations

- **Spherical wheel is laterally too fat (near walls).** `heightmap.wheel_envelope`
  inflates the terrain by an *isotropic* disk of radius R, so the wheel is modeled
  as a sphere. R is only correct in the rolling plane (body X–Z); across the axle
  (body Y) the real wheel is thin (half-width 0.05 m, ~7× narrower). Effect: the
  robot "feels" walls up to R≈0.35 m to its side and acts ~1.4 m wide instead of
  ~0.83 m, so it cannot hug walls or thread narrow gaps. Fine for open/gentle
  terrain (current scenes). **Fix when wall-navigation is needed:** anisotropic
  thin-cylinder wheel contact (cap radius R only in the rolling direction, thin
  across the axle, evaluated per-step in the body frame since it's yaw-dependent),
  and/or treat walls as obstacles via a separate 2D thin-footprint clearance check
  (heightmap = traversable ground only). Both slot in at `wheel_envelope` + a new
  in-plane clearance without disturbing the rest.
- **High-center is detection-only, by design** (Phase 3): belly non-penetration is
  not *enforced* (wheels stay grounded — assumption 1). A high-centering trajectory
  is **rejected** via the `valid` flag, not resolved by lifting a wheel. This keeps
  the settle a clean wheels-only 3×3 equality solve (no chassis complementarity),
  which in turn keeps the Phase-5 implicit gradient simple.
