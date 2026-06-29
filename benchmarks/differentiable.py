"""Timing benchmark for DifferentiableSimulator: the cost of one calibration gradient step.

A gradient step is `rollout_taped` (off-tape arg-max CONTACT + on-tape GATHER + the T-step taped
rollout) followed by `backward`. Reports each stage plus the grad-step total and steps/second.
Sweeps batch B at CALIBRATION scale (B = number of episodes, 10s-100s -- NOT the thousands used for
planning; per-rollout terrain is B x the grid memory) and horizon T. CUDA-only (the tiled contact
needs GPU shared memory).

Run from the repo root:  python -m benchmarks.differentiable
"""

import numpy as np
import warp as wp
from kinematic_helhest import dynamics
from kinematic_helhest import friction
from kinematic_helhest import heightmap as hmmod
from kinematic_helhest.control.reference import _to_wheel_omega
from kinematic_helhest.engine import DifferentiableSimulator
from kinematic_helhest.engine import GridParams

from ._common import time_fn


def _build(scene, mu, B, T, device):
    sim = DifferentiableSimulator(
        dynamics.robot_params(),
        dynamics.execution_solver(),  # deep settle (12 Newton iters): the gradient wants a converged root
        GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0),
        B,
        T,
        device,
    )
    sim.set_terrain(
        wp.array(
            np.ascontiguousarray(np.broadcast_to(scene.H, (B, *scene.H.shape)), np.float32),
            dtype=wp.float32,
            device=device,
        )
    )
    sim.set_friction(
        wp.array(
            np.ascontiguousarray(np.broadcast_to(mu.H, (B, *mu.H.shape)), np.float32),
            dtype=wp.float32,
            device=device,
        )
    )
    sim.wheel_omega.assign(
        np.ascontiguousarray(_to_wheel_omega(np.full((B, T, 2), 2.0, np.float32)), np.float32)
    )
    sim.start_pose.assign(np.tile(np.asarray((0.0, 0.0, 0.0), np.float32), (B, 1)))
    return sim


def _header():
    print(
        f"    {'B':>5} {'T':>4} {'contact_ms':>11} {'taped_ms':>9} {'bwd_ms':>8} "
        f"{'step_ms':>8} {'steps/s':>8}"
    )


def _row(scene, mu, B, T, device, reps):
    sim = _build(scene, mu, B, T, device)
    sim.rollout_taped()  # warm up: triggers codegen + gives backward a tape
    t_contact = time_fn(sim._contact, reps, device)
    t_taped = time_fn(sim.rollout_taped, reps, device)
    t_bwd = time_fn(sim.backward, reps, device)
    step = t_taped + t_bwd
    print(
        f"    {B:>5} {T:>4} {t_contact*1e3:>10.2f}m {t_taped*1e3:>8.2f}m {t_bwd*1e3:>7.2f}m "
        f"{step*1e3:>7.2f}m {1.0/step:>8.0f}"
    )


def main():
    wp.init()
    if not wp.is_cuda_available():
        print("CUDA not available -- DifferentiableSimulator is GPU-only. Skipping.")
        return
    device = "cuda"
    scene = hmmod.demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    reps = 20
    print(
        f"\n=== DifferentiableSimulator  device={device}  grid={scene.ny}x{scene.nx}  reps={reps} ==="
    )
    print("  batch sweep (T=40):")
    _header()
    for B in [16, 64, 256]:
        _row(scene, mu, B, 40, device, reps)
    print("  horizon sweep (B=64):")
    _header()
    for T in [20, 40, 80]:
        _row(scene, mu, 64, T, device, reps)


if __name__ == "__main__":
    main()
