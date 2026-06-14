"""Device kinematics: the quasi-static settle + the monolithic forward step.

One Warp thread = one rollout. The 3x3 Newton settle runs in registers (numerical
Jacobian, fixed iters). Mirrors the numpy `placement`/`state` reference so that
stays the finite-diff oracle. Orientation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

Arg convention (Option 2): the differentiated grids (height, friction) are plain
`wp.array` kernel args; everything else rides in three structs --
`GridMeta` (terrain.py), `Robot`, `SolverC` (solver.py).

State across timesteps is two vec3s (avoids the length-6 spatial_vector type):
planar = (x, y, yaw) controlled DOF; tilt = (z, pitch, roll) derived DOF.
"""
import numpy as np
import warp as wp

from .solver import Robot
from .solver import SolverC
from .terrain import GridMeta
from .terrain import sample_height
from .terrain import sample_normal

# Warp 1.13/1.14 ptxas MISCOMPILES this module's large combined `step` kernel at
# -O3 on CUDA: a register spill produces an invalid __local__ read (illegal
# memory access at runtime). Verified via compute-sanitizer + an -O level sweep
# (-O3 crashes; -O2/-O1/-O0 are correct). -O2 is correct and ~as fast as -O3, so
# pin this module to it. CPU is unaffected (defaults to -O2).
wp.set_module_options({"optimization_level": 2})


@wp.func
def euler_zyx(yaw: float, pitch: float, roll: float):
    cz = wp.cos(yaw)
    sz = wp.sin(yaw)
    cy = wp.cos(pitch)
    sy = wp.sin(pitch)
    cx = wp.cos(roll)
    sx = wp.sin(roll)
    Rz = wp.mat33(cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0)
    Ry = wp.mat33(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy)
    Rx = wp.mat33(1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx)
    return Rz * Ry * Rx


@wp.func
def solve3(A: wp.mat33, b: wp.vec3):
    """Solve A x = b (3x3) via cofactors.

    Used instead of wp.inverse: Warp 1.13 miscompiles wp.inverse(mat33) on CUDA
    when two inverse call sites share a kernel (e.g. settle + normal_loads),
    causing illegal memory access. An explicit solve has a single code path.
    """
    a = A[0, 0]; b1 = A[0, 1]; c1 = A[0, 2]
    d = A[1, 0]; e = A[1, 1]; f = A[1, 2]
    g = A[2, 0]; h = A[2, 1]; i = A[2, 2]
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    det = a * c00 + b1 * c01 + c1 * c02
    inv = 1.0 / det
    c10 = -(b1 * i - c1 * h); c11 = a * i - c1 * g; c12 = -(a * h - b1 * g)
    c20 = b1 * f - c1 * e; c21 = -(a * f - c1 * d); c22 = a * e - b1 * d
    return wp.vec3((c00 * b[0] + c10 * b[1] + c20 * b[2]) * inv,
                   (c01 * b[0] + c11 * b[1] + c21 * b[2]) * inv,
                   (c02 * b[0] + c12 * b[1] + c22 * b[2]) * inv)


@wp.func
def clearances(H: wp.array2d(dtype=wp.float32), g: GridMeta, robot: Robot,
               x: float, y: float, yaw: float, z: float, pitch: float, roll: float):
    """Signed wheel clearances c_i = hub_z - H_env(hub_xy) - R for the 3 wheels."""
    R = euler_zyx(yaw, pitch, roll)
    p = wp.vec3(x, y, z)
    h0 = p + R * robot.wheel_pos[0]
    h1 = p + R * robot.wheel_pos[1]
    h2 = p + R * robot.wheel_pos[2]
    c0 = h0[2] - sample_height(H, g, h0[0], h0[1]) - robot.wheel_radius
    c1 = h1[2] - sample_height(H, g, h1[0], h1[1]) - robot.wheel_radius
    c2 = h2[2] - sample_height(H, g, h2[0], h2[1]) - robot.wheel_radius
    return wp.vec3(c0, c1, c2)


@wp.func
def settle(H: wp.array2d(dtype=wp.float32), g: GridMeta, robot: Robot, sp: SolverC,
           x: float, y: float, yaw: float, u_init: wp.vec3):
    """Solve (z, pitch, roll). Returns the solved u = vec3(z, pitch, roll)."""
    eps = 1.0e-6
    u = u_init
    for _ in range(sp.newton_iters):
        c = clearances(H, g, robot, x, y, yaw, u[0], u[1], u[2])
        # numerical 3x3 Jacobian dc/d(z,pitch,roll), columns = perturbations
        jz = (clearances(H, g, robot, x, y, yaw, u[0] + eps, u[1], u[2]) - c) / eps
        jp = (clearances(H, g, robot, x, y, yaw, u[0], u[1] + eps, u[2]) - c) / eps
        jr = (clearances(H, g, robot, x, y, yaw, u[0], u[1], u[2] + eps) - c) / eps
        J = wp.mat33(jz[0], jp[0], jr[0],
                     jz[1], jp[1], jr[1],
                     jz[2], jp[2], jr[2])
        step = solve3(J, c)
        sz = wp.clamp(step[0], -sp.max_step, sp.max_step)
        sp_ = wp.clamp(step[1], -sp.max_step, sp.max_step)
        sr = wp.clamp(step[2], -sp.max_step, sp.max_step)
        u = wp.vec3(u[0] - sz,
                    wp.clamp(u[1] - sp_, -sp.tilt_clamp, sp.tilt_clamp),
                    wp.clamp(u[2] - sr, -sp.tilt_clamp, sp.tilt_clamp))
    return u


@wp.func
def _scatter_h(H_adj: wp.array2d(dtype=wp.float32), g: GridMeta,
               x: float, y: float, coef: float):
    """Accumulate coef * (bilinear weights of (x,y)) into the H adjoint array.

    This is d(sample_height)/dH at (x,y): the same 4-node stencil sample_height
    reads, scattered with atomics (many output cells may hit the same node).
    """
    fx = (x - g.x0) / g.cell
    fy = (y - g.y0) / g.cell
    ix = wp.clamp(int(wp.floor(fx)), 0, g.nx - 2)
    iy = wp.clamp(int(wp.floor(fy)), 0, g.ny - 2)
    tx = wp.clamp(fx - float(ix), 0.0, 1.0)
    ty = wp.clamp(fy - float(iy), 0.0, 1.0)
    wp.atomic_add(H_adj, iy, ix, coef * (1.0 - tx) * (1.0 - ty))
    wp.atomic_add(H_adj, iy, ix + 1, coef * tx * (1.0 - ty))
    wp.atomic_add(H_adj, iy + 1, ix, coef * (1.0 - tx) * ty)
    wp.atomic_add(H_adj, iy + 1, ix + 1, coef * tx * ty)


@wp.func_grad(settle)
def adj_settle(H: wp.array2d(dtype=wp.float32), g: GridMeta, robot: Robot, sp: SolverC,
               x: float, y: float, yaw: float, u_init: wp.vec3, adj_ret: wp.vec3):
    """Implicit (IFT) adjoint of the settle. adj_ret = cotangent on u*.

    With residual c(u*, x, y, yaw, H) = 0 and J = dc/du at u*:
        lambda = J^-T adj_ret
        adj_theta = -(dc/dtheta)^T lambda   for theta in {x, y, yaw, H}
    J and the pose derivatives are numerical (matching the forward); the H term is
    the analytic bilinear stencil. d/du_init = 0 (root independent of warm start).
    """
    e = 1.0e-4
    u = settle(H, g, robot, sp, x, y, yaw, u_init)  # recompute converged u*
    c = clearances(H, g, robot, x, y, yaw, u[0], u[1], u[2])
    jz = (clearances(H, g, robot, x, y, yaw, u[0] + e, u[1], u[2]) - c) / e
    jp = (clearances(H, g, robot, x, y, yaw, u[0], u[1] + e, u[2]) - c) / e
    jr = (clearances(H, g, robot, x, y, yaw, u[0], u[1], u[2] + e) - c) / e
    J = wp.mat33(jz[0], jp[0], jr[0],
                 jz[1], jp[1], jr[1],
                 jz[2], jp[2], jr[2])
    lam = solve3(wp.transpose(J), adj_ret)

    # pose adjoints: -(dc/dtheta) . lambda  (numerical dc/dtheta)
    cx = (clearances(H, g, robot, x + e, y, yaw, u[0], u[1], u[2]) - c) / e
    cy = (clearances(H, g, robot, x, y + e, yaw, u[0], u[1], u[2]) - c) / e
    cw = (clearances(H, g, robot, x, y, yaw + e, u[0], u[1], u[2]) - c) / e
    wp.adjoint[x] += -wp.dot(cx, lam)
    wp.adjoint[y] += -wp.dot(cy, lam)
    wp.adjoint[yaw] += -wp.dot(cw, lam)

    # H adjoint: adj_H[node] += lambda_i * (stencil of hub_i)  (per wheel)
    R = euler_zyx(yaw, u[1], u[2])
    p = wp.vec3(x, y, u[0])
    h0 = p + R * robot.wheel_pos[0]
    h1 = p + R * robot.wheel_pos[1]
    h2 = p + R * robot.wheel_pos[2]
    _scatter_h(wp.adjoint[H], g, h0[0], h0[1], lam[0])
    _scatter_h(wp.adjoint[H], g, h1[0], h1[1], lam[1])
    _scatter_h(wp.adjoint[H], g, h2[0], h2[1], lam[2])


@wp.func
def normal_loads(H: wp.array2d(dtype=wp.float32), g: GridMeta, robot: Robot,
                 R: wp.mat33, p: wp.vec3):
    """Quasi-static contact normal loads N_i from gravity (3x3 force/torque solve).

    Row 0: vertical force balance Sum N_i n_iz = m g.
    Rows 1-2: horizontal torque balance about the CoM. Returns N = vec3(N0,N1,N2).
    `H` is the wheel-envelope grid (same surface the contacts sit on).
    """
    com_world = p + R * robot.com
    hub0 = p + R * robot.wheel_pos[0]
    hub1 = p + R * robot.wheel_pos[1]
    hub2 = p + R * robot.wheel_pos[2]
    n0 = sample_normal(H, g, hub0[0], hub0[1])
    n1 = sample_normal(H, g, hub1[0], hub1[1])
    n2 = sample_normal(H, g, hub2[0], hub2[1])
    r0 = (hub0 - robot.wheel_radius * n0) - com_world
    r1 = (hub1 - robot.wheel_radius * n1) - com_world
    r2 = (hub2 - robot.wheel_radius * n2) - com_world
    m0 = wp.cross(r0, n0)
    m1 = wp.cross(r1, n1)
    m2 = wp.cross(r2, n2)
    A = wp.mat33(n0[2], n1[2], n2[2],
                 m0[0], m1[0], m2[0],
                 m0[1], m1[1], m2[1])
    b = wp.vec3(robot.mass * robot.gravity, 0.0, 0.0)
    return solve3(A, b)


@wp.func
def chassis_clearance(H: wp.array2d(dtype=wp.float32), g: GridMeta, robot: Robot,
                      R: wp.mat33, p: wp.vec3):
    """Min signed clearance of the chassis bottom-face points above RAW terrain.

    Negative == high-centered (belly penetrates). `H` is the raw heightmap.
    """
    cmin = float(1.0e9)
    for i in range(robot.n_chassis):
        w = p + R * robot.chassis_pts[i]
        c = w[2] - sample_height(H, g, w[0], w[1])
        cmin = wp.min(cmin, c)
    return cmin


# ----------------------------------------------------------------------------
# forward step + rollout
# ----------------------------------------------------------------------------
@wp.kernel
def init_state(H: wp.array2d(dtype=wp.float32), g: GridMeta, robot: Robot, sp: SolverC,
               pose: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw)
               planar: wp.array2d(dtype=wp.vec3),  # [T+1, B] -> writes row 0
               tilt: wp.array2d(dtype=wp.vec3)):  # [T+1, B] -> writes row 0
    tid = wp.tid()
    pc = pose[tid]
    z0 = sample_height(H, g, pc[0], pc[1]) + robot.wheel_radius
    u = settle(H, g, robot, sp, pc[0], pc[1], pc[2], wp.vec3(z0, 0.0, 0.0))
    planar[0, tid] = pc
    tilt[0, tid] = u  # (z, pitch, roll)


@wp.kernel
def step(t: int,
         Henv: wp.array2d(dtype=wp.float32), Hraw: wp.array2d(dtype=wp.float32), g: GridMeta,
         Hmu: wp.array2d(dtype=wp.float32), gmu: GridMeta,
         robot: Robot, sp: SolverC,
         omega: wp.array2d(dtype=wp.vec3),  # [T, B] (wL, wR, w_rear)
         planar: wp.array2d(dtype=wp.vec3),  # [T+1, B] (x, y, yaw)
         tilt: wp.array2d(dtype=wp.vec3),  # [T+1, B] (z, pitch, roll)
         loads_out: wp.array2d(dtype=wp.vec3),  # [T, B] N_i of the NEW state
         turn_out: wp.array2d(dtype=wp.vec2),  # [T, B] (alpha, x_icr) used this step
         clear_out: wp.array2d(dtype=float)):  # [T, B] belly clearance of the NEW state
    tid = wp.tid()
    pc = planar[t, tid]
    tc = tilt[t, tid]
    x = pc[0]
    y = pc[1]
    yaw = pc[2]
    R = euler_zyx(yaw, tc[1], tc[2])
    p = wp.vec3(x, y, tc[0])

    # --- turning params from the CURRENT pose (loads + friction at contacts) ---
    N = normal_loads(Henv, g, robot, R, p)
    w0v = robot.wheel_pos[0]
    w1v = robot.wheel_pos[1]
    w2v = robot.wheel_pos[2]
    h0 = p + R * w0v
    h1 = p + R * w1v
    h2 = p + R * w2v
    ct0 = h0 - robot.wheel_radius * sample_normal(Henv, g, h0[0], h0[1])
    ct1 = h1 - robot.wheel_radius * sample_normal(Henv, g, h1[0], h1[1])
    ct2 = h2 - robot.wheel_radius * sample_normal(Henv, g, h2[0], h2[1])
    mw0 = sample_height(Hmu, gmu, ct0[0], ct0[1]) * N[0]
    mw1 = sample_height(Hmu, gmu, ct1[0], ct1[1]) * N[1]
    mw2 = sample_height(Hmu, gmu, ct2[0], ct2[1]) * N[2]
    sw = mw0 + mw1 + mw2
    x_icr = (mw0 * w0v[0] + mw1 * w1v[0] + mw2 * w2v[0]) / sw
    alpha = 1.0 + sp.k_turn * sw / (robot.gravity * robot.mass)

    # --- predict: twist through the CURRENT orientation, Euler integrate ---
    om = omega[t, tid]
    vx = robot.wheel_radius * (om[0] + om[1]) / 2.0
    wz = robot.wheel_radius * (om[1] - om[0]) / (2.0 * robot.half_track * alpha)
    vy = -x_icr * wz
    vw = R * wp.vec3(vx, vy, 0.0)
    xn = x + vw[0] * sp.dt
    yn = y + vw[1] * sp.dt
    yawn = yaw + wz * sp.dt

    # --- project: settle the new pose (warm-started from current tilt) ---
    u = settle(Henv, g, robot, sp, xn, yn, yawn, tc)
    planar[t + 1, tid] = wp.vec3(xn, yn, yawn)
    tilt[t + 1, tid] = u

    Rn = euler_zyx(yawn, u[1], u[2])
    pn = wp.vec3(xn, yn, u[0])
    loads_out[t, tid] = normal_loads(Henv, g, robot, Rn, pn)
    turn_out[t, tid] = wp.vec2(alpha, x_icr)
    clear_out[t, tid] = chassis_clearance(Hraw, g, robot, Rn, pn)


def rollout_device(scene, mu_field, setpoints, init_pose, params,
                   robot_params=None, device="cpu"):
    """Single-rollout (B=1) device rollout. Returns numpy logs to match the oracle."""
    from .. import heightmap as hmmod
    from .solver import RobotParams
    from .terrain import to_terrain

    robot_params = robot_params or RobotParams()
    robot = robot_params.build(device)
    sp = params.build()
    Rw = robot_params.wheel_radius

    te = to_terrain(hmmod.wheel_envelope(scene, Rw), device)
    tr = to_terrain(scene, device)
    tm = to_terrain(mu_field, device)
    setpoints = np.asarray(setpoints, np.float32)
    T = setpoints.shape[0]

    omega = wp.array(setpoints.reshape(T, 1, 3), dtype=wp.vec3, device=device)
    pose0 = wp.array(np.asarray([init_pose], np.float32), dtype=wp.vec3, device=device)
    planar = wp.zeros((T + 1, 1), dtype=wp.vec3, device=device)
    tilt = wp.zeros((T + 1, 1), dtype=wp.vec3, device=device)
    loads = wp.zeros((T, 1), dtype=wp.vec3, device=device)
    turn = wp.zeros((T, 1), dtype=wp.vec2, device=device)
    clear = wp.zeros((T, 1), dtype=float, device=device)

    wp.launch(init_state, 1, inputs=[te.H, te.g, robot, sp, pose0],
              outputs=[planar, tilt], device=device)
    for t in range(T):
        wp.launch(step, 1,
                  inputs=[t, te.H, tr.H, te.g, tm.H, tm.g, robot, sp, omega],
                  outputs=[planar, tilt, loads, turn, clear], device=device)
    return {
        "planar": planar.numpy()[:, 0, :], "tilt": tilt.numpy()[:, 0, :],
        "loads": loads.numpy()[:, 0, :], "turn": turn.numpy()[:, 0, :],
        "clear": clear.numpy()[:, 0],
    }


# ----------------------------------------------------------------------------
# self-tests vs the numpy oracle
# ----------------------------------------------------------------------------
@wp.kernel
def _settle_probe(H: wp.array2d(dtype=wp.float32), g: GridMeta, robot: Robot, sp: SolverC,
                  pose: wp.array(dtype=wp.vec3),
                  u_out: wp.array(dtype=wp.vec3), contacts: wp.array(dtype=wp.vec3),
                  normals: wp.array(dtype=wp.vec3), residual: wp.array(dtype=float)):
    tid = wp.tid()
    x = pose[tid][0]
    y = pose[tid][1]
    yaw = pose[tid][2]
    z0 = sample_height(H, g, x, y) + robot.wheel_radius
    u = settle(H, g, robot, sp, x, y, yaw, wp.vec3(z0, 0.0, 0.0))
    u_out[tid] = u
    R = euler_zyx(yaw, u[1], u[2])
    p = wp.vec3(x, y, u[0])
    c = clearances(H, g, robot, x, y, yaw, u[0], u[1], u[2])
    residual[tid] = wp.max(wp.max(wp.abs(c[0]), wp.abs(c[1])), wp.abs(c[2]))
    for i in range(3):
        hub = p + R * robot.wheel_pos[i]
        n = sample_normal(H, g, hub[0], hub[1])
        normals[3 * tid + i] = n
        contacts[3 * tid + i] = hub - robot.wheel_radius * n


@wp.kernel
def _loads_probe(Henv: wp.array2d(dtype=wp.float32), Hraw: wp.array2d(dtype=wp.float32),
                 g: GridMeta, robot: Robot, sp: SolverC, pose: wp.array(dtype=wp.vec3),
                 loads: wp.array(dtype=wp.vec3), clearance: wp.array(dtype=float)):
    tid = wp.tid()
    x = pose[tid][0]
    y = pose[tid][1]
    yaw = pose[tid][2]
    z0 = sample_height(Henv, g, x, y) + robot.wheel_radius
    u = settle(Henv, g, robot, sp, x, y, yaw, wp.vec3(z0, 0.0, 0.0))
    R = euler_zyx(yaw, u[1], u[2])
    p = wp.vec3(x, y, u[0])
    loads[tid] = normal_loads(Henv, g, robot, R, p)
    clearance[tid] = chassis_clearance(Hraw, g, robot, R, p)


def _build_test(device="cpu", iters=12):
    from .solver import RobotParams, SolverParams
    robot = RobotParams().build(device)
    sp = SolverParams(newton_iters=iters, max_step=0.2, tilt_clamp=1.2).build()
    return robot, sp


def _selftest_settle():
    from .. import heightmap as hmmod
    from .. import placement
    from .terrain import to_terrain

    wp.init()
    robot, sp = _build_test()
    cases = [("flat", hmmod.flat(), [(0.0, 0.0, 0.0), (1.0, 0.5, 0.7)]),
             ("ramp", hmmod.ramp_scene(), [(2.0, 0.0, 0.0), (3.0, 0.5, 0.3)]),
             ("box", hmmod.box_scene(), [(-1.0, 0.0, 0.0), (0.5, 0.0, 0.0)])]
    worst = 0.0
    for name, scene, poses in cases:
        env = hmmod.wheel_envelope(scene, 0.35)
        te = to_terrain(env, "cpu")
        B = len(poses)
        pose = wp.array(np.asarray(poses, np.float32), dtype=wp.vec3, device="cpu")
        u_out = wp.zeros(B, dtype=wp.vec3, device="cpu")
        contacts = wp.zeros(B * 3, dtype=wp.vec3, device="cpu")
        normals = wp.zeros(B * 3, dtype=wp.vec3, device="cpu")
        resid = wp.zeros(B, dtype=float, device="cpu")
        wp.launch(_settle_probe, B, inputs=[te.H, te.g, robot, sp, pose],
                  outputs=[u_out, contacts, normals, resid], device="cpu")
        uo = u_out.numpy()
        co = contacts.numpy().reshape(B, 3, 3)
        for i, (x, y, yaw) in enumerate(poses):
            ref = placement.settle(x, y, yaw, env)
            du = max(abs(uo[i, 0] - ref["z"]), abs(uo[i, 1] - ref["pitch"]), abs(uo[i, 2] - ref["roll"]))
            dc = np.abs(co[i] - ref["contacts"]).max()
            worst = max(worst, du, dc)
            print(f"  {name:4s} ({x:+.1f},{y:+.1f},{yaw:+.1f})  d(z,p,r)={du:.2e}  dcontacts={dc:.2e}")
    print(f"settle device-vs-oracle worst={worst:.2e}  {'OK' if worst < 1e-4 else 'REVIEW'}")


def _selftest_loads():
    from .. import heightmap as hmmod
    from .. import placement
    from .terrain import to_terrain

    wp.init()
    robot, sp = _build_test()
    cases = [("flat", hmmod.flat(), [(0.0, 0.0, 0.0), (1.0, 0.5, 0.7)]),
             ("ramp", hmmod.ramp_scene(), [(2.0, 0.0, 0.0), (3.0, 0.5, 0.3)]),
             ("box", hmmod.box_scene(), [(-1.0, 0.0, 0.0), (0.9, 0.0, 0.0)])]
    worst = 0.0
    for name, scene, poses in cases:
        env, raw = hmmod.wheel_envelope(scene, 0.35), scene
        te, tr = to_terrain(env, "cpu"), to_terrain(raw, "cpu")
        B = len(poses)
        pose = wp.array(np.asarray(poses, np.float32), dtype=wp.vec3, device="cpu")
        loads = wp.zeros(B, dtype=wp.vec3, device="cpu")
        clear = wp.zeros(B, dtype=float, device="cpu")
        wp.launch(_loads_probe, B, inputs=[te.H, tr.H, te.g, robot, sp, pose],
                  outputs=[loads, clear], device="cpu")
        lo, cl = loads.numpy(), clear.numpy()
        for i, (x, y, yaw) in enumerate(poses):
            place = placement.settle(x, y, yaw, env)
            N_ref = placement.normal_loads(place, x, y)
            cc_ref = placement.chassis_clearance(place["R"], x, y, place["z"], raw)[0].min()
            dN = np.abs(lo[i] - N_ref).max()
            dC = abs(cl[i] - cc_ref)
            worst = max(worst, dN, dC)
            print(f"  {name:4s} ({x:+.1f},{y:+.1f},{yaw:+.1f})  dN={dN:.2e}  dclear={dC:.2e}")
    print(f"loads/clearance device-vs-oracle worst={worst:.2e}  {'OK' if worst < 1e-3 else 'REVIEW'}")


def _selftest_step():
    from .. import friction
    from .. import heightmap as hmmod
    from .. import rollout as rollout_np
    from .solver import SolverParams

    wp.init()
    k, dt = 2.0, 0.05
    params = SolverParams(newton_iters=12, dt=dt, k_turn=k)
    mu = friction.uniform(0.8)
    worst = 0.0
    cases = [("flat-turn", hmmod.flat(), (0.0, 0.0, 0.0), np.tile([1.0, 2.0, 1.5], (40, 1))),
             ("box-climb", hmmod.box_scene(), (-1.0, 0.0, 0.0), np.tile([2.0, 2.0, 2.0], (60, 1)))]
    for name, scene, init_pose, sp in cases:
        out = rollout_device(scene, mu, sp, init_pose, params)
        ref = rollout_np.rollout_terrain(sp, dt, scene, init_pose=init_pose, mu_field=mu, k=k)
        d_xy = np.abs(out["planar"][1:, :2] - ref["pose2"][:, :2]).max()
        d_yaw = np.abs(out["planar"][1:, 2] - ref["pose2"][:, 2]).max()
        d_N = np.abs(out["loads"] - ref["loads"]).max()
        d_a = np.abs(out["turn"][:, 0] - ref["alpha"]).max()
        d_x = np.abs(out["turn"][:, 1] - ref["x_icr"]).max()
        d_c = np.abs(out["clear"] - ref["chassis_clear"]).max()
        worst = max(worst, d_xy, d_yaw, d_N * 1e-3, d_c)
        print(f"  {name:9s} dXY={d_xy:.2e} dyaw={d_yaw:.2e} dN={d_N:.2e} "
              f"dalpha={d_a:.2e} dxicr={d_x:.2e} dclear={d_c:.2e}")
    print(f"step device-vs-oracle  {'OK' if worst < 5e-3 else 'REVIEW'}")


if __name__ == "__main__":
    _selftest_settle()
    _selftest_loads()
    _selftest_step()
