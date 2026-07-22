"""6D state-transition Jacobian for the Helhest kinematic model.

State:  q = [x, y, ψ, ẋᵂ, ẏᵂ, ψ̇]   (world-frame pose + velocity)
Input:  u = [ω_L, ω_R, ω_rear]       (wheel angular speeds, rad/s)

The Jacobian F = ∂q_next/∂q is computed via central differences, using a
ForwardSimulator batch to evaluate all 6 perturbed rollouts in one GPU launch.

Structural observation: the ForwardSimulator recomputes velocity internally
from the 3D pose q[0:3] and the input u — it never reads the stored velocity
q[3:6].  Perturbing q[3:6] therefore has no effect on q_next, so columns 3–5
of F are analytically zero.  Only the 3 position dims require simulation
(6 rollouts = 3 dims × 2 for central differences), all run in one batch.

See docs/state_model_proper.md for the full 6D model and the analytical
flat-ground Jacobian against which this numerical result can be verified.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import ForwardSimulator

from helhest.dynamics import DT as _DT
from helhest.model import HALF_TRACK
from helhest.model import WHEEL_RADIUS


def _euler_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Z-Y-X rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll), shape (3, 3)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _q6d_from_batch(
    ctrl: np.ndarray,
    deriv: np.ndarray,
    turn: np.ndarray,
    b: int,
    u: np.ndarray,
    omega_z: float | None = None,
) -> np.ndarray:
    """Extract one 6D next-state vector from a batch simulator readback.

    ctrl    : controlled.numpy()  shape [T+1, B, 3]  — (x, y, yaw) at each step
    deriv   : derived.numpy()     shape [T+1, B, 3]  — (z, pitch, roll)
    turn    : turning.numpy()     shape [T,   B, 2]  — (alpha, x_icr) this step
    b       : batch index
    u       : wheel-speed command [omega_L, omega_R, omega_rear]
    omega_z : measured base-frame gyro yaw rate [rad/s].  When provided, it
              replaces the wheel-differential yaw rate (slip-immune).  The xy
              translation is taken from the simulator rollout.
    """
    x_n, y_n, psi_sim = ctrl[1, b]
    _z_n, theta_n, phi_n = deriv[1, b]
    alpha, x_icr = turn[0, b]

    # Body-frame forward speed: unaffected by slip, so always from wheel average.
    vx_b = WHEEL_RADIUS * (u[0] + u[1]) / 2.0

    if omega_z is not None:
        # Gyro-driven yaw: immune to wheel slip.  Override the heading that the
        # simulator integrated internally with psi_start + omega_z * DT.
        yaw_rate = omega_z
        psi_n = float(ctrl[0, b][2]) + omega_z * _DT
    else:
        yaw_rate = WHEEL_RADIUS * (u[1] - u[0]) / (2.0 * HALF_TRACK * alpha)
        psi_n = psi_sim

    vy_b = -x_icr * yaw_rate

    # Rotate body-frame twist into world frame using the full settled orientation.
    vw = _euler_zyx(psi_n, theta_n, phi_n) @ np.array([vx_b, vy_b, 0.0])

    return np.array([x_n, y_n, psi_n, vw[0], vw[1], yaw_rate], dtype=np.float64)


def jacobian_F_6d(
    q0: np.ndarray,
    u0: np.ndarray,
    sim: ForwardSimulator,
    eps: float = 1e-4,
    omega_z: float | None = None,
) -> np.ndarray:
    """Discrete 6×6 state-transition Jacobian F = ∂q_next/∂q at (q0, u0).

    Uses central differences on q[0:3] only (q[3:6] columns are analytically
    zero — see module docstring).  All 6 perturbed rollouts run in a single
    batched GPU launch; sim must be pre-built with batch_size=6, n_steps=1 and
    terrain already loaded.
    q0      : [6]  current state [x, y, ψ, ẋᵂ, ẏᵂ, ψ̇]
    u0      : [3]  wheel speeds  [ω_L, ω_R, ω_rear]  (rad/s)
    omega_z : measured gyro yaw rate [rad/s]; see _q6d_from_batch.
    eps=1e-4 is chosen for float32 simulator output: smaller eps causes
    cancellation error; the O(eps²) central-difference truncation error is
    negligible at this scale.
    """
    assert sim.batch_size == 6 and sim.n_steps == 1, (
        f"sim must have batch_size=6, n_steps=1; got {sim.batch_size=}, {sim.n_steps=}"
    )

    # Build 6 start poses: pairs (q[j]+eps, q[j]-eps) for j in {0, 1, 2}.
    # batch layout: [x+, x-, y+, y-, ψ+, ψ-]
    poses = np.tile(q0[:3].astype(np.float32), (6, 1))  # [6, 3]
    for j in range(3):
        poses[2 * j, j] += eps
        poses[2 * j + 1, j] -= eps

    u_f32 = u0.astype(np.float32)
    sim.start_pose.assign(poses)
    # target_wheel_omega shape [T, B] = [1, 6]; same command for all rollouts.
    sim.target_wheel_omega.assign(np.tile(u_f32, (1, 6, 1)))
    # Steady-state: lagged omega == target (no motor lag for this step).
    sim.init_current_wheel_omega.assign(np.tile(u_f32, (6, 1)))

    sim.rollout_launch()

    ctrl = sim.controlled.numpy()  # [2, 6, 3]
    deriv = sim.derived.numpy()    # [2, 6, 3]
    turn = sim.turning.numpy()     # [1, 6, 2]

    F = np.zeros((6, 6))
    for j in range(3):
        b_plus = 2 * j
        b_minus = 2 * j + 1
        q_plus = _q6d_from_batch(ctrl, deriv, turn, b_plus, u0, omega_z)
        q_minus = _q6d_from_batch(ctrl, deriv, turn, b_minus, u0, omega_z)
        F[:, j] = (q_plus - q_minus) / (2.0 * eps)
    # Columns 3–5 remain zero: stored velocity q[3:6] is never read by the
    # simulator, so perturbing it has no effect on q_next.

    return F


def predict_q6d(
    q0: np.ndarray,
    u0: np.ndarray,
    sim: ForwardSimulator,
    omega_z: float | None = None,
) -> np.ndarray:
    """Nonlinear 6-D forward prediction q_next = f(q0, u0) at one timestep.

    The companion to jacobian_F_6d: it evaluates the model itself (the un-perturbed
    rollout) rather than its derivative, sharing the same twist/rotation extraction
    (_q6d_from_batch) so f and ∂f/∂q stay consistent. Only q0[0:3] seeds the sim; the
    stored velocity q0[3:6] is never read (see module docstring).
    q0      : [6]  current state [x, y, ψ, ẋᵂ, ẏᵂ, ψ̇]
    u0      : [3]  wheel speeds  [ω_L, ω_R, ω_rear]  (rad/s)
    omega_z : measured gyro yaw rate [rad/s]; see _q6d_from_batch.
    sim must be pre-built with batch_size=1, n_steps=1 and terrain already loaded.
    """
    assert sim.batch_size == 1 and sim.n_steps == 1, (
        f"sim must have batch_size=1, n_steps=1; got {sim.batch_size=}, {sim.n_steps=}"
    )

    u_f32 = u0.astype(np.float32)
    sim.start_pose.assign(q0[:3].astype(np.float32).reshape(1, 3))
    # target_wheel_omega shape [T, B] = [1, 1].
    sim.target_wheel_omega.assign(u_f32.reshape(1, 1, 3))
    # Steady-state: lagged omega == target (no motor lag for this step).
    sim.init_current_wheel_omega.assign(u_f32.reshape(1, 3))

    sim.rollout_launch()

    ctrl = sim.controlled.numpy()  # [2, 1, 3]
    deriv = sim.derived.numpy()    # [2, 1, 3]
    turn = sim.turning.numpy()     # [1, 1, 2]
    return _q6d_from_batch(ctrl, deriv, turn, 0, u0, omega_z)


#------------------------------------------
# Tests of jacobian_F_6d functions
#------------------------------------------


if __name__ == "__main__":
    from helhest.engine import GridParams
    from helhest.engine import RobotParams
    from helhest.engine import SolverParams

    DEVICE = "cuda:0" 

    # 20 m × 20 m flat terrain, 0.1 m resolution, centred on the origin.
    CELL = 0.1
    NX = NY = 200
    grid_params = GridParams(NX, NY, CELL, -10.0, -10.0)
    elevation = wp.zeros((NY, NX), dtype=wp.float32, device=DEVICE)

    robot = RobotParams()
    solver = SolverParams()
    # batch_size=6: one batch covers all 6 central-difference rollouts.
    sim = ForwardSimulator(robot, solver, grid_params, batch_size=6, n_steps=1, device=DEVICE)
    sim.set_terrain(elevation)
    sim.set_uniform_friction(0.8)

    np.set_printoptions(precision=4, suppress=True)
    dt = solver.dt  # 0.1 s

    # -----------------------------------------------------------------------
    # Test 1: straight drive at 1 m/s, ψ = 0.
    #
    # Flat-ground analytical Jacobian (columns 3-5 = 0):
    #   F[1,2] = vx_B * cos(ψ) * dt  (heading → y_next coupling)
    #   F[4,2] = vx_B * cos(ψ_next)  (heading → vy_W_next)
    # All others are identity diagonal for rows/cols 0-2 or zero.
    # -----------------------------------------------------------------------
    v = 1.0
    w = v / robot.wheel_radius  # equal L/R wheel speed → zero yaw rate
    u0 = np.array([w, w, w], dtype=np.float64)

    # Velocity states consistent with ψ=0 kinematics (α irrelevant for wz=0).
    q0 = np.array([0.0, 0.0, 0.0, v, 0.0, 0.0])

    F = jacobian_F_6d(q0, u0, sim)

    print("=== Test 1: straight drive, ψ=0, v=1 m/s ===")
    print("q0 =", q0)
    print("u0 =", u0)
    print("F (numerical) =")
    print(F)

    # Analytical: ψ=0, wz=0 → ψ_next=0; vx_B=v, vy_B=0.
    psi, psi_next = 0.0, 0.0
    vx_b = v
    F_ref = np.zeros((6, 6))
    F_ref[0, 0] = F_ref[1, 1] = F_ref[2, 2] = 1.0
    F_ref[0, 2] = -vx_b * np.sin(psi) * dt         # = 0
    F_ref[1, 2] = vx_b * np.cos(psi) * dt          # = v * dt = 0.1
    F_ref[3, 2] = -vx_b * np.sin(psi_next)         # = 0
    F_ref[4, 2] = vx_b * np.cos(psi_next)          # = v = 1.0
    print("F (analytical reference) =")
    print(F_ref)
    print(f"max |F_numerical - F_analytical| = {np.abs(F - F_ref).max():.2e}")

    # -----------------------------------------------------------------------
    # Test 2: turning, ψ = π/4, differential wheel speeds.
    #
    # α is terrain/friction dependent (~2.6 with mu=0.8, k=2).  The velocity
    # states in q1 are set from the analytical flat-ground twist at α=1 just
    # to give a plausible initial state; the Jacobian itself uses the simulator's
    # actual α.  The analytical reference below accounts for ψ_next exactly.
    # -----------------------------------------------------------------------
    psi = np.pi / 4.0
    omega_L, omega_R = 2.0, 3.5   # rad/s
    u1 = np.array([omega_L, omega_R, (omega_L + omega_R) / 2.0], dtype=np.float64)

    # Body-frame twist at α=1 (nominal; sim will use its own α internally).
    vx_b = WHEEL_RADIUS * (omega_L + omega_R) / 2.0
    yaw_rate_nom = WHEEL_RADIUS * (omega_R - omega_L) / (2.0 * HALF_TRACK)
    vw = _euler_zyx(psi, 0.0, 0.0) @ np.array([vx_b, 0.0, 0.0])
    q1 = np.array([1.0, 0.5, psi, vw[0], vw[1], yaw_rate_nom])

    F1 = jacobian_F_6d(q1, u1, sim)

    print()
    print("=== Test 2: turning, ψ=π/4, ω_L=2 ω_R=3.5 rad/s ===")
    print("q1 =", q1)
    print("u1 =", u1)
    print("F (numerical) =")
    print(F1)

    # Approximate analytical reference using α=1 (simulator uses α>1 with mu=0.8,
    # so a small residual discrepancy in the turning-sensitive entries is expected).
    psi_next_nom = psi + yaw_rate_nom * dt
    vx_W_next = vx_b * np.cos(psi_next_nom)
    vy_W_next = vx_b * np.sin(psi_next_nom)

    F1_ref = np.zeros((6, 6))
    F1_ref[0, 0] = F1_ref[1, 1] = F1_ref[2, 2] = 1.0
    F1_ref[0, 2] = -vx_b * np.sin(psi) * dt        # ∂x_next/∂ψ
    F1_ref[1, 2] = vx_b * np.cos(psi) * dt         # ∂y_next/∂ψ
    F1_ref[3, 2] = -vy_W_next                       # ∂vx_W_next/∂ψ
    F1_ref[4, 2] = vx_W_next                        # ∂vy_W_next/∂ψ
    print("F (analytical reference, α=1 approx) =")
    print(F1_ref)
    print(f"max |F_numerical - F_analytical| = {np.abs(F1 - F1_ref).max():.2e}")
    print("(residual discrepancy in test 2 is expected: sim uses α>1 with mu=0.8)")

    # -----------------------------------------------------------------------
    # Test 3: non-flat ground — linear ramp sloping uphill in the +x direction.
    #
    # A slope of 15° produces meaningful pitch (θ ≈ 15°) when the robot is
    # aligned with the slope and meaningful roll when it crosses it.  No
    # analytical reference: the goal is to confirm the code path that uses the
    # full R(ψ, θ, φ) rotation and terrain-derived α, x_ICR executes without
    # error and returns a plausible (non-degenerate) Jacobian.
    # -----------------------------------------------------------------------
    slope_deg = 15.0
    slope = np.tan(np.deg2rad(slope_deg))  # dz/dx

    # Build elevation: z = slope * (x - x_origin), zero at the grid origin.
    x_coords = (np.arange(NX) * CELL) + (-NX // 2 * CELL)  # [NX] x-coordinates
    elev_np = (np.ones((NY, NX)) * x_coords * slope).astype(np.float32)
    elevation_ramp = wp.array(elev_np, dtype=wp.float32, device=DEVICE)

    sim.set_terrain(elevation_ramp)
    sim.set_uniform_friction(0.8)

    # Place the robot near the grid centre, heading up-slope (ψ=0).
    # Velocity states are left as zeros — the sim ignores q[3:6] for integration.
    q2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Straight drive at 0.5 m/s up the slope.
    v_ramp = 0.5
    w_ramp = v_ramp / robot.wheel_radius
    u2 = np.array([w_ramp, w_ramp, w_ramp], dtype=np.float64)

    F2 = jacobian_F_6d(q2, u2, sim)

    print()
    print(f"=== Test 3: {slope_deg}° uphill ramp, ψ=0, v={v_ramp} m/s ===")
    print("q2 =", q2)
    print("u2 =", u2)
    print("F (numerical) =")
    print(F2)
    print("Columns 3-5 zero:", np.allclose(F2[:, 3:], 0))
    print(f"F[2,2] (ψ self) = {F2[2,2]:.4f}  (expect ≈ 1.0)")
    print("Note: F[3,2] and F[4,2] are now driven by the full R(ψ,θ,φ) rotation — θ≠0 on the ramp.")
