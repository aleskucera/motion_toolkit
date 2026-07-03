# Helhest Junior — kinematic differentiable simulator

A fast, differentiable, **purely kinematic** twin of the Helhest Junior for
planning and calibration. No Newton/Ostrich, no dynamics. The robot is a rigid tripod:

- **Controlled DOF** `(x, y, yaw)` — driven by the 3 wheels via no-slip
  differential-drive kinematics, with friction-dependent turning captured by two
  ICR parameters `(alpha, x_ICR)` derived from a per-cell friction field.
- **Derived DOF** `(z, pitch, roll)` — from a quasi-static settle against a
  heightmap: an analytic 3×3 Newton solve (wheels grounded). Chassis
  non-penetration is a post-check (high-centering rejects a trajectory, it isn't
  resolved by lifting a wheel).

Differentiable w.r.t. the **raw heightmap** `h` and the **friction field** `mu`,
via a hand-written implicit (IFT) adjoint through the settle. The wheel-envelope
dilation is recorded on the tape (arg-max contact off-tape, gather on-tape), so
gradients flow all the way to the *raw* elevation. Controls are **not**
differentiated. Batched over many rollouts for sampling-based planning and for
per-episode calibration.

## Architecture

Two simulators share a `BaseSimulator` (device structs, grid, buffer allocation):

- **`ForwardSimulator`** — the planner's workhorse: one fused `rollout_kernel` over
  `B` rollouts × `T` steps (graph-capturable, no host I/O, no gradients). CPU/CUDA.
- **`DifferentiableSimulator`** — calibration: a taped per-step rollout with
  per-rollout `[B, ny, nx]` terrain, so each episode calibrates its own grid.
  Gradients land in `elevation.grad` / `friction.grad`. A scalar-loss path
  (`rollout_taped(loss_fn) → backward()`) and a VJP path for framework bridges
  (`rollout_taped(loss_fn=None) → backward_from_cotangents(dL/dcontrolled, dL/dderived)`).
  CUDA-only (the dilation contact uses a shared-memory tiled arg-max).

Package layout (`src/kinematic_helhest/`):

| dir / module | role |
|---|---|
| `engine/` | Warp runtime — `simulator.py` (the 3 sims), `step.py` (settle + IFT adjoint, step kernels), `envelope.py` (wheel dilation), `terrain.py`, `robot.py`, `rotations.py`, `linalg.py` |
| `reference/` | numpy finite-difference oracle (verification only) |
| `control/` | `mppi.py` (GPU MPPI), `terminal.py` (terminal dock) |
| `planning/` | `costtogo.py` (orientation-aware routing), `lattice_solver.py` |
| `perception/` | `gridmap.py`, `lidar.py`, `rasterize.py` |
| `driver.py` | `WarpDriver` — the single driven robot (B=1, T=1) |
| `worlds.py`, `dynamics.py` | stress scenes + canonical robot/solver params |

Robot geometry/mass come from `dynamics.robot_params()` (`engine/robot.py`
`RobotParams`); `data.py` loads rosbags for calibration/eval.

## Install

```bash
uv sync                              # core (numpy + warp-lang)
uv sync --extra viz --extra data     # + viewers (glfw/PyOpenGL/matplotlib) + rosbag loader (h5py)
```
or plain pip: `pip install -e ".[viz,data]"`.

## Run

```bash
# closed-loop eval on the real driver (MPPI + cost-to-go routing + terminal dock)
python demos/eval.py --world pocket           # one world
python demos/eval.py --stress                 # all stress worlds
python demos/navigate_partial.py              # plan on a lidar-built map, fixed robot window

# timing benchmarks
python -m benchmarks.forward                  # ForwardSimulator fused rollout (CPU+CUDA)
python -m benchmarks.differentiable           # DifferentiableSimulator grad-step (CUDA)
python -m benchmarks.planning                 # cost-to-go solve (CUDA)
python -m benchmarks.control                  # MPPI replan (CUDA)

# verify the engine
python -m tests.engine.gpu_check              # forward / adjoint / VJP parity + throughput (CUDA)
python -m tests.engine.gradients              # implicit-gradient finite-diff oracle (CPU)
```

The package is `kinematic_helhest`; `engine/` is the runtime, `reference/` is the
numpy finite-difference oracle (verification only).

## Blender visualization

Export a rollout to a self-contained `.npz`, then animate it in Blender — the
heightmap becomes a mesh and the robot's 6-DOF pose + per-wheel spin are keyframed.

```bash
# 1. run a rollout and write the .npz (a straight climb up the ramp scene)
python -m kinematic_helhest.viz.blender_export rollout.npz

# 2a. build the scene (box+cylinder proxy robot) and open it in Blender
blender --python src/kinematic_helhest/viz/blender_import.py -- --data rollout.npz

# 2b. headless render straight to MP4 (PNG sequence if Blender lacks FFMPEG)
blender --background --python src/kinematic_helhest/viz/blender_import.py -- \
    --data rollout.npz --render out.mp4

# 2c. drive your own rigged model (a .blend with separate wheel objects)
blender --python src/kinematic_helhest/viz/blender_import.py -- --data rollout.npz \
    --robot robot.blend \
    --wheel-left WheelL --wheel-right WheelR --wheel-rear WheelRear
```

Frame convention is the sim's: X-forward, Y-left, Z-up, meters/radians; wheels spin
about body Y (`--wheel-axis`). A rigged model is appended whole (hierarchy + materials)
and auto-aligned: the animation root is placed at the front-axle hub center (mean of the
left/right wheels), matching the sim's pose origin, so no manual offset/scale is needed.
The 10 Hz sim is interpolated to `--fps` (default 30) for smooth playback.

### Remote GPU render (dasenka)

`scripts/render_dasenka.sh` ships the script + `.npz` (+ a local `--robot` model) to a
remote box, renders headless on a pinned GPU (`VK_DEVICE_INDEX` + Vulkan/EEVEE), encodes
to MP4 there, and copies it back:

```bash
OUT=~/clip.mp4 RES=1920x1080 ./scripts/render_dasenka.sh rollout.npz <gpu> -- \
    --robot robot.blend --wheel-left WheelL --wheel-right WheelR --wheel-rear WheelRear
```

## Phases

Built in phases, each independently verifiable:

| Phase | Content | Verify | Status |
|-------|---------|--------|--------|
| 0 | scaffold, heightmap rasterizer + bilinear sampler, rosbag loader | height under wheel matches scene; run loads | ✅ |
| 1 | flat-ground forward twist, scalar `(alpha, x_ICR)` | reproduces ~0.40 m/s cruise on run 18_04_51 | ✅ |
| 2 | heightmap settle, wheels grounded, normal loads `N_i`, sphere-wheel envelope | flat→level; ramp→pitched; loads=scaled meas.; box climbs | ✅ |
| 3 | chassis non-penetration = post-check → high-center ⇒ reject trajectory (`valid`) | benign→valid; tall block→high-center w/ depth | ✅ |
| 4 | per-cell `mu` field + grip-weighted ICR turning map | uniform→`1+k·mu`/CoM_x; slippery rear turns more; signs correct | ✅ |
| 5 | implicit gradients (`d/dh`, `d/dmu`), BPTT + VJP boundary | finite-diff check ~2e-5; VJP == scalar-loss backward | ✅ |
| 6 | calibration vs rosbags | RMSE ≤ full-physics bar; cross-run | ⬜ (gradient path ready: `DifferentiableSimulator`) |
| 7 | speed benchmarks | forward throughput; calibration grad-step cost | ✅ (`benchmarks/`) |
| 8 | planning demo (MPPI + cost-to-go) | reaches goal, avoids high-center | ✅ (`demos/eval.py`) |

## Known limitations

- **Spherical wheel is laterally too fat (near walls).** `engine/envelope.py`
  inflates the terrain by an *isotropic* disk of radius R, so the wheel is modeled
  as a sphere. R is only correct in the rolling plane (body X–Z); across the axle
  (body Y) the real wheel is thin (half-width 0.05 m, ~7× narrower). Effect: the
  robot "feels" walls up to R≈0.35 m to its side and acts ~1.4 m wide instead of
  ~0.83 m, so it cannot hug walls or thread narrow gaps. Fine for open/gentle
  terrain (current scenes). **Fix when wall-navigation is needed:** anisotropic
  thin-cylinder wheel contact (cap radius R only in the rolling direction, thin
  across the axle, evaluated per-step in the body frame since it's yaw-dependent),
  and/or treat walls as obstacles via a separate 2D thin-footprint clearance check
  (heightmap = traversable ground only). Both slot in at the dilation + a new
  in-plane clearance without disturbing the rest.
- **High-center is detection-only, by design** (Phase 3): belly non-penetration is
  not *enforced* (wheels stay grounded). A high-centering trajectory is **rejected**
  via the `valid` flag, not resolved by lifting a wheel. This keeps the settle a
  clean wheels-only 3×3 equality solve (no chassis complementarity), which in turn
  keeps the implicit (IFT) gradient simple.

## Perception stack (`terrain_toolkit`)

The on-robot perception front-end lives in the same repo (package `terrain_toolkit`):
GPU point cloud → heightmap → traversability cost map, plus GPU-native ICP
(`icp/`) and a device-resident rolling map/localization accumulator
(`mapping/DeviceMapAccumulator`). It feeds the planner across the
perception→planning seam via a shared `GridMap` (`terrain_toolkit.gridmap` →
`kinematic_helhest.perception.gridmap`).

| Stage | Module | Purpose |
|---|---|---|
| Heightmap raster | `heightmap/` | max/mean/min/count layers + NaN-aware inpaint & smooth |
| Geometric cost | `traversability/` | slope + signed step + roughness, obstacle inflation, temporal gate |
| Trust masks | `confidence/` | occlusion (line-of-sight) + support-ratio masking |
| Orchestration | `pipeline.py` | `TerrainPipeline` — points in, `TerrainMap` out |
| Localization | `icp/`, `mapping/` | GPU-native ICP + rolling accumulated map |

See [`docs/`](docs/) for the full perception reference (built with mkdocs).

## License

MIT — see [LICENSE](LICENSE).
