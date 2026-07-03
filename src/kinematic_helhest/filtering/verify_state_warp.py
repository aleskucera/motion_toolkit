"""
Warp-engine-based state verification (filtering) via the Helhest kinematic model.

The caller that is trying to call verify_state has to do the following:

    from kinematic_helhest.engine import ForwardSimulator
    from kinematic_helhest.engine import GridParams
    from kinematic_helhest.engine import RobotParams
    from kinematic_helhest.engine import SolverParams

    sim = ForwardSimulator(RobotParams(), SolverParams(), grid_params, batch_size=1, n_steps=1)
    sim.set_terrain(elevation_warp_array)  # wp.array [ny, nx] float32
    sim.set_friction(friction_hm)          # or sim.set_uniform_friction(mu)

    # current_pose : wp.array(dtype=wp.vec3, shape=(1,))  -- (x, y, yaw)
    # omega        : wp.array2d(dtype=wp.vec3, shape=(1,1)) -- wheel speeds [L, R, rear]
    # estimate     : wp.array(dtype=wp.vec3, shape=(1,))  -- (x_hat, y_hat, psi_hat)
    score = verify_state(sim, current_pose, omega, estimate)
"""

from __future__ import annotations

import warp as wp

from kinematic_helhest.engine import ForwardSimulator

XY_THRES = 0.3
ANG_THRES = wp.pi / 8.0


def get_l2_error(controlled: wp.array, estimate: wp.array) -> float:
    # controlled is [T+1, B] vec3; .list() flattens row-major, so index 1 = (t=1, b=0)
    pred = controlled.list()[1]  # wp.vec3 — predicted pose after one step, batch 0
    est = estimate.list()[0]     # wp.vec3
    diff = pred - est
    dpsi = diff[2] - wp.floor((diff[2] + wp.pi) / (2.0 * wp.pi)) * (2.0 * wp.pi)
    return float(wp.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + dpsi * dpsi))


def get_max_confidence(controlled: wp.array, estimate: wp.array) -> float:
    pred = controlled.list()[1]  # wp.vec3
    est = estimate.list()[0]     # wp.vec3
    dxy = wp.length(wp.vec2(pred[0] - est[0], pred[1] - est[1]))
    dpsi_raw = pred[2] - est[2]
    dpsi = wp.abs(dpsi_raw - wp.floor((dpsi_raw + wp.pi) / (2.0 * wp.pi)) * (2.0 * wp.pi))
    # 0.1 ** (err / threshold) = 1.0 at zero error, 0.1 at threshold
    return float(wp.max(wp.pow(0.1, dxy / XY_THRES), wp.pow(0.1, dpsi / ANG_THRES)))


def verify_state(
    sim: ForwardSimulator,
    current_pose: wp.array,
    omega: wp.array,
    estimate: wp.array,
    verif_type: str, # l2 or confidence
) -> float:
    """L2 distance between the Warp-predicted planar pose and `estimate`.

    Advances `current_pose` one timestep via the ForwardSimulator (terrain must
    already be loaded via sim.set_terrain), then returns
    ||(x_pred - x_hat, y_pred - y_hat, wrap(psi_pred - psi_hat))||_2.

    sim          : ForwardSimulator built with batch_size=1, n_steps=1; terrain pre-loaded
    current_pose : wp.array(dtype=wp.vec3, shape=(1,))   -- (x, y, yaw) [m, m, rad]
    omega        : wp.array2d(dtype=wp.vec3, shape=(1,1)) -- wheel angular velocities [L, R, rear] (rad/s)
    estimate     : wp.array(dtype=wp.vec3, shape=(1,))   -- (x_hat, y_hat, psi_hat)
    """
    assert sim.batch_size == 1 and sim.n_steps == 1, (
        f"sim must have batch_size=1, n_steps=1; got {sim.batch_size=}, {sim.n_steps=}"
    )

    # Copy to sim object:
    wp.copy(sim.start_pose, current_pose)
    wp.copy(sim.wheel_omega, omega)
    
    # Step:
    sim.rollout_launch()
    helpers = {"l2": get_l2_error, "confidence": get_max_confidence}
    return helpers[verif_type](sim.controlled, estimate)


#------------------------------------------
# Test of the verify_state function
#------------------------------------------


if __name__ == "__main__":
    from kinematic_helhest.engine import ForwardSimulator
    from kinematic_helhest.engine import GridParams
    from kinematic_helhest.engine import RobotParams
    from kinematic_helhest.engine import SolverParams

    DEVICE = "cuda:0"

    # 10 m x 10 m flat terrain, 0.1 m resolution, centred on the origin
    CELL = 0.1
    NX = NY = 100
    grid_params = GridParams(NX, NY, CELL, -5.0, -5.0)
    elevation = wp.zeros((NY, NX), dtype=wp.float32, device=DEVICE)

    robot = RobotParams()
    solver = SolverParams()
    sim = ForwardSimulator(robot, solver, grid_params, batch_size=1, n_steps=1, device=DEVICE)
    sim.set_terrain(elevation)
    sim.set_uniform_friction(0.8)

    # straight drive at 1 m/s; all three wheels spin at v / wheel_radius
    v = 1.0
    w = v / robot.wheel_radius  # ≈ 2.857 rad/s

    # shapes must match sim.start_pose (1,) and sim.wheel_omega (1, 1)
    current_pose = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=DEVICE)
    omega = wp.array([[wp.vec3(w, w, w)]], dtype=wp.vec3, device=DEVICE)

    # after dt seconds of straight driving on flat ground: x ≈ v * dt, y = 0, yaw = 0
    x_pred = v * solver.dt
    estimate_near = wp.array([wp.vec3(x_pred, 0.1, 0.1)], dtype=wp.vec3, device=DEVICE)
    estimate_far = wp.array([wp.vec3(0.5, 0.2, 0.3)], dtype=wp.vec3, device=DEVICE)

    mode = "confidence"
    near = verify_state(sim, current_pose, omega, estimate_near, mode)
    far = verify_state(sim, current_pose, omega, estimate_far, mode)
    print(f"[{mode}].  estimate ≈ predicted: {near:.4f}   wrong estimate: {far:.4f}")
