"""Timing benchmark for the kinematic rollout: forward (fused) vs forward (taped) vs backward.

Three paths, same dynamics:
  - fused   : `Simulator.rollout_launch` -- one fused `rollout_kernel`, the graph-capturable
              production path (register carry, not autodiffable).
  - taped   : `DifferentiableSimulator.rollout_taped` -- per-step `step_kernel` recorded on a
              `wp.Tape` plus the loss. The forward cost you pay to enable backprop.
  - backward: `DifferentiableSimulator.backward` -- adjoint replay over the recorded tape.

Reports ms/rollout, the backward/fused ratio (the number that matters for calibration cost),
and the fused forward real-time factor (simulated seconds per wall-second = B*T*dt / wall).
Sweeps batch B and horizon T independently, on CPU and (if present) CUDA.

Wall-clock is independent of `dt` (it only scales the integrated velocities), so the timings hold
for any `dt`; only the real-time factor moves with it -- set it with `--dt` (default 0.1).

Run from the repo root:  python -m benchmarks.rollout [--dt 0.1]
"""

import argparse
import time

import numpy as np
import warp as wp
from kinematic_helhest import friction
from kinematic_helhest import heightmap as hmmod
from kinematic_helhest.control.reference import _to_wheel_omega
from kinematic_helhest.engine import DifferentiableSimulator
from kinematic_helhest.engine import GridParams
from kinematic_helhest.engine import RobotParams
from kinematic_helhest.engine import Simulator
from kinematic_helhest.engine import SolverParams

NEWTON_ITERS = 12


def _scene():
    scene = hmmod.demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    return scene, mu


def _build(cls, scene, mu, B, T, device, dt):
    """A built simulator of `cls` with terrain/friction set and controls/start pose loaded."""
    sim = cls(
        RobotParams(),
        SolverParams(dt=dt, k_turn=2.0, newton_iters=NEWTON_ITERS),
        GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0),
        B,
        T,
        device,
    )
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_friction(mu)
    sim.wheel_omega.assign(
        np.ascontiguousarray(_to_wheel_omega(np.full((B, T, 2), 2.0, np.float32)), np.float32)
    )
    sim.start_pose.assign(np.tile(np.asarray((0.0, 0.0, 0.0), np.float32), (B, 1)))
    return sim


def _time(fn, reps, device):
    """Mean wall-clock of `fn` over `reps`, with a warmup and device syncs around the loop so
    async CUDA launches are actually accounted for (the gpu_check.time_rollout pattern)."""
    fn()  # warmup: triggers codegen + first launch
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    wp.synchronize_device(device)
    return (time.perf_counter() - t0) / reps


def _bench(scene, mu, B, T, device, dt, reps):
    fused = _build(Simulator, scene, mu, B, T, device, dt)
    diff = _build(DifferentiableSimulator, scene, mu, B, T, device, dt)
    diff.rollout_taped()  # record once so backward has a tape to replay during its warmup
    t_fused = _time(fused.rollout_launch, reps, device)
    t_taped = _time(diff.rollout_taped, reps, device)
    t_bwd = _time(diff.backward, reps, device)
    return t_fused, t_taped, t_bwd


def _header():
    print(
        f"    {'B':>6} {'T':>4} {'fused_ms':>9} {'taped_ms':>9} {'bwd_ms':>9} "
        f"{'bwd/fwd':>8} {'fused_RTF':>10}"
    )


def _row(B, T, tf, tt, tb, dt):
    rtf = B * T * dt / tf  # simulated seconds per wall-second (throughput)
    print(
        f"    {B:>6} {T:>4} {tf*1e3:>9.2f} {tt*1e3:>9.2f} {tb*1e3:>9.2f} "
        f"{tb/tf:>7.1f}x {rtf:>9.1e}x"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.1, help="sim timestep [s] for the RTF column")
    args = parser.parse_args()
    dt = args.dt

    wp.init()
    scene, mu = _scene()
    devices = ["cpu"] + (["cuda"] if wp.is_cuda_available() else [])
    for device in devices:
        # CPU is single-host-threaded per launch; keep B/T and reps modest so it finishes quickly.
        if device == "cpu":
            batch_sweep, fixed_T = [64, 256, 512], 40
            horizon_sweep, fixed_B = [20, 40, 80], 256
            reps = 5
        else:
            batch_sweep, fixed_T = [512, 2048, 8192], 40
            horizon_sweep, fixed_B = [20, 40, 80, 160], 2048
            reps = 20

        print(
            f"\n=== device={device}  dt={dt:.2f}  grid={scene.ny}x{scene.nx}  "
            f"newton_iters={NEWTON_ITERS}  reps={reps} ==="
        )
        rows = []  # (B, T, tf, tt, tb) across both sweeps, for the summary
        print(f"  batch sweep (T={fixed_T}):")
        _header()
        for B in batch_sweep:
            t = _bench(scene, mu, B, fixed_T, device, dt, reps)
            rows.append((B, fixed_T, *t))
            _row(B, fixed_T, *t, dt)
        print(f"  horizon sweep (B={fixed_B}):")
        _header()
        for T in horizon_sweep:
            t = _bench(scene, mu, fixed_B, T, device, dt, reps)
            rows.append((fixed_B, T, *t))
            _row(fixed_B, T, *t, dt)

        B, T, tf, tt, tb = max(rows, key=lambda r: r[0] * r[1] * dt / r[2])  # peak RTF row
        print(
            f"  peak: {B*T*dt/tf:.1e}x real-time @ B={B},T={T};  "
            f"grad step (taped+bwd) = {(tt+tb)/tf:.1f}x fused"
        )


if __name__ == "__main__":
    main()
