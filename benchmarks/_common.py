"""Shared helpers for the planning/control benchmarks: scene build, terrain coarsening, timing,
and the cost-to-go routing field (which the control benchmark needs to arm the MPPI planner)."""

import time

import numpy as np
import warp as wp
from kinematic_helhest import dynamics
from kinematic_helhest import worlds as W
from kinematic_helhest.engine import GridParams
from kinematic_helhest.planning.costtogo import CostToGo


def build_scene(world):
    builder, start, goal = W.WORLDS[world]
    scene = builder()
    mu = W.matching_friction(scene)
    return scene, mu, np.asarray(start, np.float32), np.asarray(goal, np.float64)


def coarsen(scene, k):
    """Routing terrain at 1/k resolution (max-pool keeps thin walls), plus its GridParams."""
    if k <= 1:
        return np.ascontiguousarray(scene.H, np.float32), GridParams(
            scene.nx, scene.ny, scene.cell, scene.x0, scene.y0
        )
    cny, cnx, ccell = scene.ny // k, scene.nx // k, scene.cell * k
    Hc = scene.H[: cny * k, : cnx * k].reshape(cny, k, cnx, k).max(axis=(1, 3))
    return np.ascontiguousarray(Hc, np.float32), GridParams(cnx, cny, ccell, scene.x0, scene.y0)


def time_fn(fn, reps, device):
    """Mean wall-clock with a warmup (captures the CUDA graph) and syncs around the timed loop."""
    fn()
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    wp.synchronize_device(device)
    return (time.perf_counter() - t0) / reps


def build_routing(scene, n_theta, k, goal, device):
    """A CostToGo solver + its solved field V (to arm the planner, or to time the solve itself).
    Returns (solver, V, grid, terrain) -- terrain is the device array `compute` was called on."""
    Hc_np, cgrid = coarsen(scene, k)
    clat = CostToGo(
        cgrid, dynamics.robot_params(), dynamics.planning_solver(), n_theta=n_theta, device=device
    )
    Hc = wp.array(Hc_np, dtype=wp.float32, device=device)
    V = clat.compute(Hc, (float(goal[0]), float(goal[1])))
    return clat, V, cgrid, Hc
