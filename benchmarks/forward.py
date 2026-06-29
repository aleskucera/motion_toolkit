"""Timing benchmark for ForwardSimulator: the fused MPPI rollout (no gradients).

One fused `rollout_kernel` over B rollouts x T steps. Reports ms/rollout, the real-time factor
(simulated seconds per wall-second = B*T*dt / wall), and throughput (M wheel-steps/s). Sweeps batch
B and horizon T at planner scale, on CPU and (if present) CUDA.

Wall-clock is independent of `dt` (it only scales the integrated velocities), so the timings hold
for any `dt`; only the real-time factor moves with it -- set it with `--dt` (default 0.1).

Run from the repo root:  python -m benchmarks.forward [--dt 0.1]
"""

import argparse

import numpy as np
import warp as wp
from kinematic_helhest import dynamics
from kinematic_helhest import friction
from kinematic_helhest import heightmap as hmmod
from kinematic_helhest.control.reference import _to_wheel_omega
from kinematic_helhest.engine import ForwardSimulator
from kinematic_helhest.engine import GridParams

from ._common import time_fn


def _build(scene, mu, B, T, device, dt):
    sim = ForwardSimulator(
        dynamics.robot_params(),
        dynamics.planning_solver(dt),
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


def _header():
    print(f"    {'B':>6} {'T':>4} {'ms':>9} {'RTF':>10} {'Mstep/s':>9}")


def _row(scene, mu, B, T, device, dt, reps):
    sim = _build(scene, mu, B, T, device, dt)
    t = time_fn(sim.rollout_launch, reps, device)
    print(f"    {B:>6} {T:>4} {t*1e3:>9.2f} {B*T*dt/t:>9.1e}x {B*T/t/1e6:>9.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dt", type=float, default=0.1, help="sim timestep [s] for the RTF column")
    dt = ap.parse_args().dt

    wp.init()
    scene = hmmod.demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    devices = ["cpu"] + (["cuda"] if wp.is_cuda_available() else [])
    for device in devices:
        if device == "cpu":
            batch_sweep, fixed_T, horizon_sweep, fixed_B, reps = (
                [64, 256, 512],
                40,
                [20, 40, 80],
                256,
                5,
            )
        else:
            batch_sweep, fixed_T = [512, 2048, 8192], 40
            horizon_sweep, fixed_B, reps = [20, 40, 80, 160], 2048, 20
        print(
            f"\n=== ForwardSimulator  device={device}  dt={dt:.2f}  grid={scene.ny}x{scene.nx}  reps={reps} ==="
        )
        print(f"  batch sweep (T={fixed_T}):")
        _header()
        for B in batch_sweep:
            _row(scene, mu, B, fixed_T, device, dt, reps)
        print(f"  horizon sweep (B={fixed_B}):")
        _header()
        for T in horizon_sweep:
            _row(scene, mu, fixed_B, T, device, dt, reps)


if __name__ == "__main__":
    main()
