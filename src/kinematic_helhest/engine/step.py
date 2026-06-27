"""Device kinematics: the quasi-static settle + the monolithic forward step.

One Warp thread = one rollout. The 3x3 Newton settle runs in registers (numerical
Jacobian, fixed iters). Mirrors the numpy `placement`/`state` reference so that
stays the finite-diff oracle. Orientation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

Arg convention (Option 2): the differentiated grids (height, friction) are plain
`wp.array` kernel args; everything else rides in three structs --
`Grid` (terrain.py), `Robot` (robot.py), `Solver` (this module).

State across timesteps is two vec3s (avoids the length-6 spatial_vector type):
controlled = (x, y, yaw) the wheel-driven DOF; derived = (z, pitch, roll) the terrain-settled DOF.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .linalg import solve3
from .robot import Robot
from .rotations import drot_x
from .rotations import drot_y
from .rotations import drot_z
from .rotations import euler_zyx
from .rotations import rot_x
from .rotations import rot_y
from .rotations import rot_z
from .terrain import _locate
from .terrain import Grid
from .terrain import sample_field
from .terrain import sample_height_grad
from .terrain import sample_normal

# Warp 1.13/1.14 ptxas MISCOMPILES this module's large combined `step` kernel at
# -O3 on CUDA: a register spill produces an invalid __local__ read (illegal
# memory access at runtime). Verified via compute-sanitizer + an -O level sweep
# (-O3 crashes; -O2/-O1/-O0 are correct). -O2 is correct and ~as fast as -O3, so
# pin this module to it. CPU is unaffected (defaults to -O2).
wp.set_module_options({"optimization_level": 2})


# --- settle/integration numerics: host params + the device-side `Solver` struct ---
@wp.struct
class Solver:
    """Device-side settle/integration numerics — the built form of SolverParams.
    All scalars -> safe as a struct.

    `k_turn` (the friction->alpha turning gain) rides along here as a non-diff
    scalar; promote it to a plain length-1 array only if d/dk is ever needed.
    """

    newton_iters: wp.int32  # max Newton iterations (cap)
    atol: wp.float32  # stop early once |residual| < atol
    max_step: wp.vec3  # per-iter Newton cap (z[m], pitch[rad], roll[rad])
    tilt_clamp: wp.float32
    dt: wp.float32
    k_turn: wp.float32


@dataclass
class SolverParams:  # settle/integration numerics — tuning, separate from the robot
    dt: float = 0.1  # integration / control timestep [s]
    newton_iters: int = 12  # settle Newton cap; DEEP by default (the IFT adjoint needs a
    # converged root); forward-only planning caps at 6 (warm-started settle needs ~2).
    atol: float = 1e-6  # settle early-exit tol; TIGHT by default because the IFT settle
    # adjoint assumes residual~=0 at the root. Forward-only planning loosens it to ~1e-4
    # (0.1mm, 100x under the resid_tol=1e-2 validity gate) to save ~1 Newton iter/settle.
    max_step: tuple = (0.1, 0.2, 0.2)  # per-iter Newton cap (z[m], pitch[rad], roll[rad])
    tilt_clamp: float = 1.05  # clamp |pitch|, |roll| to ~60 deg
    k_turn: float = 2.0

    def build(self) -> Solver:
        s = Solver()
        s.newton_iters = self.newton_iters
        s.atol = self.atol
        s.max_step = wp.vec3(self.max_step[0], self.max_step[1], self.max_step[2])
        s.tilt_clamp = self.tilt_clamp
        s.dt = self.dt
        s.k_turn = self.k_turn
        return s


@wp.func
def clearances(
    envelope: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    x: float,
    y: float,
    yaw: float,
    z: float,
    pitch: float,
    roll: float,
):
    """Signed wheel clearances c_i = wc_z - H_env(wc_xy) - r_wheel for the 3 wheels (wc = wheel center)."""
    R = euler_zyx(yaw, pitch, roll)
    p = wp.vec3(x, y, z)

    c = wp.vec3()
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_center = p + R * robot.wheel_pos[st_i]
        height = sample_field(envelope, grid, wheel_center[0], wheel_center[1])
        c[st_i] = wheel_center[2] - height - robot.wheel_radius

    return c


@wp.func
def settle(
    envelope: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    controlled: wp.vec3,
    derived_init: wp.vec3,
):
    """Solve (z, pitch, roll) with an ANALYTIC 3x3 Newton Jacobian.

    `controlled` = (x, y, yaw) the fixed planar pose; solve `derived` = (z, pitch, roll).
    c_i = wc_iz - envelope(wc_ixy) - r_wheel, wc_i = (x,y,z) + Rz Ry Rx wheel_i.
    dc_i/dz = 1; dc_i/dpitch and dc_i/droll come from dR/dpitch, dR/droll applied
    to wheel_i, combined with the terrain gradient (gx,gy) at the wheel center. One
    euler + one value+grad sample per wheel per iter (vs 4 evals numerically).
    """
    x = controlled[0]
    y = controlled[1]
    yaw = controlled[2]
    Rz = rot_z(yaw)
    derived = derived_init
    for _ in range(solver.newton_iters):
        Ry = rot_y(derived[1])
        Rx = rot_x(derived[2])
        Rot = Rz * Ry * Rx
        dRp = Rz * drot_y(derived[1]) * Rx  # d(Rot)/dpitch
        dRr = Rz * Ry * drot_x(derived[2])  # d(Rot)/droll
        p = wp.vec3(x, y, derived[0])

        res = wp.vec3()  # residual: per-wheel clearance c_i
        J = wp.mat33()  # Jacobian: row i = dc_i/d(z, pitch, roll)
        for i in range(wp.static(3)):
            st_i = wp.static(i)

            wheel_pos = robot.wheel_pos[st_i]
            wheel_center = p + Rot * wheel_pos

            s = sample_height_grad(envelope, grid, wheel_center[0], wheel_center[1])
            height, gx, gy = s[0], s[1], s[2]  # surface height + slope under the wheel center

            res[st_i] = wheel_center[2] - height - robot.wheel_radius

            # z lifts the wheel center (dc/dz = 1); pitch/roll swing it (dRp/dRr * wheel)
            # across the terrain slope (gx, gy).
            dp = dRp * wheel_pos
            dr = dRr * wheel_pos
            J[st_i, 0] = 1.0
            J[st_i, 1] = dp[2] - gx * dp[0] - gy * dp[1]
            J[st_i, 2] = dr[2] - gx * dr[0] - gy * dr[1]

        if wp.dot(res, res) < solver.atol * solver.atol:
            break  # converged: the current derived is the root (skip the rest)
        delta = solve3(J, res)  # Newton step: J @ delta = res
        # damped Newton step on every DOF...
        for i in range(wp.static(3)):
            st_i = wp.static(i)
            derived[st_i] = derived[st_i] - wp.clamp(
                delta[st_i], -solver.max_step[st_i], solver.max_step[st_i]
            )
        # ...then clamp the tilt angles; z (height) is left free
        derived[1] = wp.clamp(derived[1], -solver.tilt_clamp, solver.tilt_clamp)
        derived[2] = wp.clamp(derived[2], -solver.tilt_clamp, solver.tilt_clamp)
    return derived


@wp.func
def _scatter_h(
    adj_envelope: wp.array2d(dtype=wp.float32), grid: Grid, x: float, y: float, coef: float
):
    """Accumulate coef * (bilinear weights of (x,y)) into the envelope adjoint array.

    This is d(sample_field)/dH at (x,y): the same 4-node stencil sample_field
    reads (via the shared `_locate`), scattered with atomics (many output cells may
    hit the same node).
    """
    c = _locate(grid, x, y)
    wp.atomic_add(adj_envelope, c.y_idx, c.x_idx, coef * (1.0 - c.frac_x) * (1.0 - c.frac_y))
    wp.atomic_add(adj_envelope, c.y_idx, c.x_idx + 1, coef * c.frac_x * (1.0 - c.frac_y))
    wp.atomic_add(adj_envelope, c.y_idx + 1, c.x_idx, coef * (1.0 - c.frac_x) * c.frac_y)
    wp.atomic_add(adj_envelope, c.y_idx + 1, c.x_idx + 1, coef * c.frac_x * c.frac_y)


@wp.func_grad(settle)
def adj_settle(
    envelope: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    controlled: wp.vec3,
    derived_init: wp.vec3,
    adj_ret: wp.vec3,
):
    """Implicit (IFT) adjoint of the settle. adj_ret = cotangent on the settled `derived`.

    With residual c(derived, controlled, envelope) = 0 and J = dc/d(derived) at the root:
        lambda = J^-T adj_ret
        adj_theta = -(dc/dtheta)^T lambda   for theta in {controlled, envelope}
    Everything analytic (same J as the forward + closed-form pose derivatives from
    the rotation derivatives and the terrain gradient). d/d(derived_init) = 0 (root
    independent of warm start).
    """
    x = controlled[0]
    y = controlled[1]
    yaw = controlled[2]
    # recompute the converged derived
    derived = settle(envelope, grid, robot, solver, controlled, derived_init)
    Rz = rot_z(yaw)
    Ry = rot_y(derived[1])
    Rx = rot_x(derived[2])
    Rot = Rz * Ry * Rx
    dRp = Rz * drot_y(derived[1]) * Rx
    dRr = Rz * Ry * drot_x(derived[2])
    dRyaw = drot_z(yaw) * Ry * Rx
    p = wp.vec3(x, y, derived[0])

    # build J (same formula as the forward) and stash the terrain slopes for phase 2
    J = wp.mat33()
    gx = wp.vec3()
    gy = wp.vec3()
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + Rot * wheel_pos
        s = sample_height_grad(envelope, grid, wheel_center[0], wheel_center[1])
        gx[st_i] = s[1]
        gy[st_i] = s[2]
        dp = dRp * wheel_pos
        dr = dRr * wheel_pos
        J[st_i, 0] = 1.0
        J[st_i, 1] = dp[2] - s[1] * dp[0] - s[2] * dp[1]
        J[st_i, 2] = dr[2] - s[1] * dr[0] - s[2] * dr[1]
    lam = solve3(wp.transpose(J), adj_ret)

    # adj_controlled = -(dc/dcontrolled)^T lambda  (dc_i/dx = -gx_i, dc_i/dy = -gy_i, yaw via dRyaw);
    # adj_envelope: scatter lambda_i into wheel-center i's bilinear stencil.
    adj_pose = wp.vec3()
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + Rot * wheel_pos  # cheap recompute (no re-sample)
        dy = dRyaw * wheel_pos
        cw = dy[2] - gx[st_i] * dy[0] - gy[st_i] * dy[1]
        adj_pose[0] = adj_pose[0] + gx[st_i] * lam[st_i]
        adj_pose[1] = adj_pose[1] + gy[st_i] * lam[st_i]
        adj_pose[2] = adj_pose[2] - cw * lam[st_i]
        _scatter_h(wp.adjoint[envelope], grid, wheel_center[0], wheel_center[1], lam[st_i])
    wp.adjoint[controlled] += adj_pose


@wp.func
def normal_loads(
    envelope: wp.array2d(dtype=wp.float32), grid: Grid, robot: Robot, R: wp.mat33, p: wp.vec3
):
    """Quasi-static contact normal loads N_i from gravity (3x3 force/torque solve).

    The body pose is (R, p): R is the orientation (body->world, from euler_zyx of
    yaw/pitch/roll) and p is the body-origin position (x, y, z). Body-frame points
    map to world as q_world = p + R * q_body, so the CoM and each wheel center are
    placed at this pose; the contacts then sit on `envelope` (the wheel-envelope
    grid). `robot` carries mass/gravity/com/wheel_pos/wheel_radius.

    Row 0: vertical force balance Sum N_i n_iz = m g.
    Rows 1-2: horizontal torque balance about the CoM. Returns N = vec3(N0,N1,N2).
    """
    com_world = p + R * robot.com

    A = wp.mat33()  # row 0: n_iz (vertical force); rows 1-2: (r_i x n_i)_xy (torque about CoM)
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + R * wheel_pos
        n = sample_normal(envelope, grid, wheel_center[0], wheel_center[1])
        ct = wheel_center - robot.wheel_radius * n  # contact point
        r = ct - com_world  # moment arm about the CoM
        m = wp.cross(r, n)
        A[0, st_i] = n[2]
        A[1, st_i] = m[0]
        A[2, st_i] = m[1]

    b = wp.vec3(robot.mass * robot.gravity, 0.0, 0.0)
    return solve3(A, b)


@wp.func
def chassis_clearance(
    elevation: wp.array2d(dtype=wp.float32), grid: Grid, robot: Robot, R: wp.mat33, p: wp.vec3
):
    """Min signed clearance of the chassis bottom-face points above RAW terrain.

    Negative == high-centered (belly penetrates). `elevation` is the raw heightmap.
    """
    cmin = float(1.0e9)
    for i in range(robot.n_chassis):
        w = p + R * robot.chassis_pts[i]
        c = w[2] - sample_field(elevation, grid, w[0], w[1])
        cmin = wp.min(cmin, c)
    return cmin


# ----------------------------------------------------------------------------
# forward step + rollout
# ----------------------------------------------------------------------------
@wp.kernel
def init_state_kernel(
    envelope: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    start_pose: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw)
    controlled: wp.array2d(dtype=wp.vec3),  # [T+1, B] (x, y, yaw)      -> writes row 0
    derived: wp.array2d(dtype=wp.vec3),  # [T+1, B] (z, pitch, roll) -> writes row 0
):
    """Seed row 0 of a rollout: settle the start pose onto the terrain.

    Each thread is one rollout (tid = batch index). The controlled DOF (x, y, yaw)
    come from `start_pose`; the derived DOF (z, pitch, roll) are solved by `settle`,
    warm-started with z = envelope(x, y) + wheel_radius and zero tilt. Writes
    controlled[0]/derived[0]; `step` advances from there.
    """
    tid = wp.tid()
    pc = start_pose[tid]
    z0 = sample_field(envelope, grid, pc[0], pc[1]) + robot.wheel_radius
    settled = settle(envelope, grid, robot, solver, pc, wp.vec3(z0, 0.0, 0.0))
    controlled[0, tid] = pc
    derived[0, tid] = settled


@wp.kernel
def step_kernel(
    envelope: wp.array2d(dtype=wp.float32),
    elevation: wp.array2d(dtype=wp.float32),
    friction: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    wheel_omega: wp.array(dtype=wp.vec3),  # [B] (wL, wR, w_rear) this step
    controlled: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw) current state
    derived: wp.array(dtype=wp.vec3),  # [B] (z, pitch, roll) current state
    controlled_next: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw) settled NEW state -> written
    derived_next: wp.array(dtype=wp.vec3),  # [B] (z, pitch, roll) NEW state -> written
    loads_out: wp.array(dtype=wp.vec3),  # [B] N_i of the NEW state
    turn_out: wp.array(dtype=wp.vec2),  # [B] (alpha, x_icr) used this step
    clear_out: wp.array(dtype=float),  # [B] belly clearance of the NEW state
    resid_out: wp.array(dtype=float),  # [B] settle residual (max|c|) of the NEW state
):
    tid = wp.tid()
    pc = controlled[tid]
    tc = derived[tid]
    x = pc[0]
    y = pc[1]
    yaw = pc[2]
    R = euler_zyx(yaw, tc[1], tc[2])
    p = wp.vec3(x, y, tc[0])

    # --- turning params from the CURRENT pose: the grip-weighted ICR + turn resistance ---
    loads = normal_loads(envelope, grid, robot, R, p)  # per-wheel normal load N_i
    total_grip = float(0.0)  # Sum_i grip_i
    grip_x = float(0.0)  # Sum_i grip_i * wheel_x  (x_icr = grip_x / total_grip)
    for i in range(wp.static(3)):
        st_i = wp.static(i)
        wheel_pos = robot.wheel_pos[st_i]
        wheel_center = p + R * wheel_pos
        n = sample_normal(envelope, grid, wheel_center[0], wheel_center[1])
        ct = wheel_center - robot.wheel_radius * n  # contact point
        grip = sample_field(friction, grid, ct[0], ct[1]) * loads[st_i]  # grip_i = mu_i * N_i
        total_grip += grip
        grip_x += grip * wheel_pos[0]
    x_icr = grip_x / total_grip  # grip-weighted ICR offset
    alpha = 1.0 + solver.k_turn * total_grip / (robot.gravity * robot.mass)  # turn resistance

    # --- predict: twist through the CURRENT orientation, Euler integrate ---
    om = wheel_omega[tid]
    vx = robot.wheel_radius * (om[0] + om[1]) / 2.0
    wz = robot.wheel_radius * (om[1] - om[0]) / (2.0 * robot.half_track * alpha)
    vy = -x_icr * wz
    vw = R * wp.vec3(vx, vy, 0.0)
    xn = x + vw[0] * solver.dt
    yn = y + vw[1] * solver.dt
    yawn = yaw + wz * solver.dt

    # --- project: settle the new pose (warm-started from current derived) ---
    pose_next = wp.vec3(xn, yn, yawn)
    settled = settle(envelope, grid, robot, solver, pose_next, tc)
    controlled_next[tid] = pose_next
    derived_next[tid] = settled

    Rn = euler_zyx(yawn, settled[1], settled[2])
    pn = wp.vec3(xn, yn, settled[0])
    loads_out[tid] = normal_loads(envelope, grid, robot, Rn, pn)
    turn_out[tid] = wp.vec2(alpha, x_icr)
    clear_out[tid] = chassis_clearance(elevation, grid, robot, Rn, pn)
    cres = clearances(envelope, grid, robot, xn, yn, yawn, settled[0], settled[1], settled[2])
    resid_out[tid] = wp.max(wp.max(wp.abs(cres[0]), wp.abs(cres[1])), wp.abs(cres[2]))


@wp.kernel
def rollout_kernel(
    n_steps: int,
    envelope: wp.array2d(dtype=wp.float32),
    elevation: wp.array2d(dtype=wp.float32),
    friction: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    start_pose: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw)
    wheel_omega: wp.array2d(dtype=wp.vec3),  # [T, B] (wL, wR, w_rear)
    controlled: wp.array2d(dtype=wp.vec3),  # [T+1, B] (x, y, yaw)
    derived: wp.array2d(dtype=wp.vec3),  # [T+1, B] (z, pitch, roll)
    loads_out: wp.array2d(dtype=wp.vec3),  # [T, B]
    turn_out: wp.array2d(dtype=wp.vec2),  # [T, B]
    clear_out: wp.array2d(dtype=float),  # [T, B]
    resid_out: wp.array2d(dtype=float),  # [T, B]
):
    """FORWARD-ONLY whole-rollout fusion: one thread per rollout walks all n_steps steps,
    carrying the state (pc, tc) in registers instead of round-tripping it through
    global memory between per-step launches (~1.2x faster than init_state_kernel +
    n_steps*step_kernel). This is the hot planning path; the differentiable/calibration
    path keeps the per-step step_kernel (the register carry is NOT auto-diffable --
    backprop needs the intermediate states this kernel overwrites).

    MUST stay bit-identical to init_state_kernel + n_steps*step_kernel (guarded by
    tests/engine/step.selftest_rollout_kernel). Edit the physics in both.
    """
    b = wp.tid()
    # init_state: settle the start pose -> row 0
    pc = start_pose[b]
    z0 = sample_field(envelope, grid, pc[0], pc[1]) + robot.wheel_radius
    tc = settle(envelope, grid, robot, solver, pc, wp.vec3(z0, 0.0, 0.0))
    controlled[0, b] = pc
    derived[0, b] = tc

    for t in range(n_steps):
        x = pc[0]
        y = pc[1]
        yaw = pc[2]
        R = euler_zyx(yaw, tc[1], tc[2])
        p = wp.vec3(x, y, tc[0])

        loads = normal_loads(envelope, grid, robot, R, p)  # per-wheel normal load N_i
        total_grip = float(0.0)  # Sum_i grip_i
        grip_x = float(0.0)  # Sum_i grip_i * wheel_x  (x_icr = grip_x / total_grip)
        for i in range(wp.static(3)):
            st_i = wp.static(i)
            wheel_pos = robot.wheel_pos[st_i]
            wheel_center = p + R * wheel_pos
            n = sample_normal(envelope, grid, wheel_center[0], wheel_center[1])
            ct = wheel_center - robot.wheel_radius * n  # contact point
            grip = sample_field(friction, grid, ct[0], ct[1]) * loads[st_i]  # grip_i = mu_i * N_i
            total_grip += grip
            grip_x += grip * wheel_pos[0]
        x_icr = grip_x / total_grip  # grip-weighted ICR offset
        alpha = 1.0 + solver.k_turn * total_grip / (robot.gravity * robot.mass)  # turn resistance

        om = wheel_omega[t, b]
        vx = robot.wheel_radius * (om[0] + om[1]) / 2.0
        wz = robot.wheel_radius * (om[1] - om[0]) / (2.0 * robot.half_track * alpha)
        vy = -x_icr * wz
        vw = R * wp.vec3(vx, vy, 0.0)
        xn = x + vw[0] * solver.dt
        yn = y + vw[1] * solver.dt
        yawn = yaw + wz * solver.dt

        pose_next = wp.vec3(xn, yn, yawn)
        settled = settle(envelope, grid, robot, solver, pose_next, tc)
        controlled[t + 1, b] = pose_next
        derived[t + 1, b] = settled

        Rn = euler_zyx(yawn, settled[1], settled[2])
        pn = wp.vec3(xn, yn, settled[0])
        loads_out[t, b] = normal_loads(envelope, grid, robot, Rn, pn)
        turn_out[t, b] = wp.vec2(alpha, x_icr)
        clear_out[t, b] = chassis_clearance(elevation, grid, robot, Rn, pn)
        cres = clearances(envelope, grid, robot, xn, yn, yawn, settled[0], settled[1], settled[2])
        resid_out[t, b] = wp.max(wp.max(wp.abs(cres[0]), wp.abs(cres[1])), wp.abs(cres[2]))

        pc = pose_next  # carry state in registers (no global round-trip)
        tc = settled
