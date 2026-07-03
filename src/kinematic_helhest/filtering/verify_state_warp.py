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

XY_THRES = wp.float32(0.3)
ANG_THRES = wp.float32(wp.pi / 8.0)


@wp.func
def _wrap_angle(diff: wp.float32) -> wp.float32:
    """Wrap angle difference to (-pi, pi)."""
    return diff - wp.floor((diff + wp.pi) / (2.0 * wp.pi)) * (2.0 * wp.pi)


@wp.func
def calculate_l2_error(
    controlled: wp.array2d(dtype=wp.vec3),
    estimate: wp.array(dtype=wp.vec3),
) -> wp.float32:
    pred = controlled[1, 0]
    est = estimate[0]
    dx = pred[0] - est[0]
    dy = pred[1] - est[1]
    dpsi = _wrap_angle(pred[2] - est[2])
    return wp.sqrt(dx * dx + dy * dy + dpsi * dpsi)


@wp.func
def calculate_confidence(
    controlled: wp.array2d(dtype=wp.vec3),
    estimate: wp.array(dtype=wp.vec3),
) -> wp.float32:
    pred = controlled[1, 0]
    est = estimate[0]
    dxy = wp.length(wp.vec2(pred[0] - est[0], pred[1] - est[1]))
    dpsi = wp.abs(_wrap_angle(pred[2] - est[2]))
    # exponential decay: 1.0 at zero error, 0.1 at threshold
    dxy_conf = wp.exp(wp.log(wp.float32(0.1)) / XY_THRES * dxy)
    dpsi_conf = wp.exp(wp.log(wp.float32(0.1)) / ANG_THRES * dpsi)
    return wp.max(dxy_conf, dpsi_conf)


@wp.kernel
def _l2_kernel(
    controlled: wp.array2d(dtype=wp.vec3),
    estimate: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=wp.float32),
):
    result[0] = calculate_l2_error(controlled, estimate)


@wp.kernel
def _confidence_kernel(
    controlled: wp.array2d(dtype=wp.vec3),
    estimate: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=wp.float32),
):
    result[0] = calculate_confidence(controlled, estimate)


def verify_state(
    sim: ForwardSimulator,
    current_pose: wp.array,
    omega: wp.array,
    estimate: wp.array,
    verif_type: str, # l2 or confidence
) -> wp.array:
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
    
    kernels = {"l2": _l2_kernel, "confidence": _confidence_kernel}
    assert verif_type in kernels, f"verif_type must be 'l2' or 'confidence', got {verif_type!r}"

    result = wp.zeros(1, dtype=wp.float32, device=sim.device)
    wp.launch(kernels[verif_type], dim=1, inputs=[sim.controlled, estimate], outputs=[result], device=sim.device)
    return result


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
    estimate_near = wp.array([wp.vec3(x_pred, 0.0, 0.0)], dtype=wp.vec3, device=DEVICE)
    estimate_far = wp.array([wp.vec3(0.5, 0.2, 0.3)], dtype=wp.vec3, device=DEVICE)

    for mode in ("l2", "confidence"):
        near = verify_state(sim, current_pose, omega, estimate_near, mode)
        far = verify_state(sim, current_pose, omega, estimate_far, mode)
        print(f"[{mode:10}] estimate ≈ predicted: {near.numpy()[0]:.4f}   wrong estimate: {far.numpy()[0]:.4f}")
