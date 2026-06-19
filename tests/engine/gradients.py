"""Option-B implicit gradient for the settle (single settle, d/dHenv).

The settle u* solves c(u*, Henv) = 0 (3 wheel clearances). The forward Newton
runs DETACHED (not on the tape). For the backward we use the implicit function
theorem: with J = dc/du at u*,

    du*/dHenv = -J^{-1} dc/dHenv
    => adj_Henv = -(dc/dHenv)^T lambda,   where  J^T lambda = adj_u

We hand-solve the 3x3 transpose system for lambda, then run the *residual* kernel
c(u*, Henv) on the tape with cotangent -lambda; Warp autodiff turns that into the
bilinear-stencil scatter into Henv.grad. No Newton, no max on the tape.
"""
import numpy as np
import warp as wp

from kinematic_helhest.engine import clearances
from kinematic_helhest.engine import settle
from kinematic_helhest.engine import Robot
from kinematic_helhest.engine import Solver
from kinematic_helhest.engine import Grid
from kinematic_helhest.engine import sample_field
from kinematic_helhest.engine.envelope import _contact_kernel
from kinematic_helhest.engine.envelope import _gather_kernel


def wheel_envelope(elevation, cell_size, wheel_radius, device="cpu"):
    """Verification-only: allocate scratch + run the two engine envelope passes
    (raw elevation -> dilated). Carries elevation.requires_grad so the backward tape
    routes d(loss)/d(raw elevation) to the contact cell."""
    ny, nx = elevation.shape
    env_radius = int(np.ceil(wheel_radius / cell_size))
    contact_iy = wp.zeros((ny, nx), dtype=wp.int32, device=device)
    contact_ix = wp.zeros((ny, nx), dtype=wp.int32, device=device)
    contact_cap = wp.zeros((ny, nx), dtype=wp.float32, device=device)
    envelope = wp.zeros((ny, nx), dtype=wp.float32, device=device,
                        requires_grad=elevation.requires_grad)
    wp.launch(_contact_kernel, dim=elevation.shape,
              inputs=[elevation, float(cell_size), float(wheel_radius), env_radius],
              outputs=[contact_iy, contact_ix, contact_cap], device=device)
    wp.launch(_gather_kernel, dim=elevation.shape,
              inputs=[elevation, contact_iy, contact_ix, contact_cap],
              outputs=[envelope], device=device)
    return envelope


@wp.kernel
def _settle_only(Henv: wp.array2d(dtype=wp.float32), g: Grid, robot: Robot, sp: Solver,
                 pose: wp.array(dtype=wp.vec3), u_out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    x = pose[tid][0]
    y = pose[tid][1]
    yaw = pose[tid][2]
    z0 = sample_field(Henv, g, x, y) + robot.wheel_radius
    u_out[tid] = settle(Henv, g, robot, sp, wp.vec3(x, y, yaw), wp.vec3(z0, 0.0, 0.0))


@wp.kernel
def _settle_jac(Henv: wp.array2d(dtype=wp.float32), g: Grid, robot: Robot,
                pose: wp.array(dtype=wp.vec3), u_star: wp.array(dtype=wp.vec3),
                eps: float, Jout: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    x = pose[tid][0]
    y = pose[tid][1]
    yaw = pose[tid][2]
    u = u_star[tid]
    c = clearances(Henv, g, robot, x, y, yaw, u[0], u[1], u[2])
    jz = (clearances(Henv, g, robot, x, y, yaw, u[0] + eps, u[1], u[2]) - c) / eps
    jp = (clearances(Henv, g, robot, x, y, yaw, u[0], u[1] + eps, u[2]) - c) / eps
    jr = (clearances(Henv, g, robot, x, y, yaw, u[0], u[1], u[2] + eps) - c) / eps
    Jout[tid] = wp.mat33(jz[0], jp[0], jr[0],
                         jz[1], jp[1], jr[1],
                         jz[2], jp[2], jr[2])


@wp.kernel
def _solve_jt(Jin: wp.array(dtype=wp.mat33), adj_u: wp.array(dtype=wp.vec3),
              minus_lam: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    lam = wp.inverse(wp.transpose(Jin[tid])) * adj_u[tid]
    minus_lam[tid] = -lam


@wp.kernel
def _residual(Henv: wp.array2d(dtype=wp.float32), g: Grid, robot: Robot,
              pose: wp.array(dtype=wp.vec3), u_star: wp.array(dtype=wp.vec3),
              c: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    x = pose[tid][0]
    y = pose[tid][1]
    yaw = pose[tid][2]
    u = u_star[tid]
    c[tid] = clearances(Henv, g, robot, x, y, yaw, u[0], u[1], u[2])


def dsettle_dHenv(env_hm, poses, adj_u, params, jac_eps=1e-4, device="cpu"):
    """Implicit grad d(sum_p adj_u_p . u*_p)/dHenv. Returns (grad_Henv, u_star)."""
    from kinematic_helhest.engine import GridParams, RobotParams

    elev = wp.array(np.ascontiguousarray(env_hm.H, np.float32), dtype=wp.float32,
                    device=device, requires_grad=True)
    g = GridParams(env_hm.nx, env_hm.ny, env_hm.cell, env_hm.x0, env_hm.y0).build()
    robot = RobotParams().build(device)
    sp = params.build()
    B = len(poses)
    pose = wp.array(np.asarray(poses, np.float32), dtype=wp.vec3, device=device)

    # forward settle (detached) + Jacobian + transpose solve
    u_star = wp.zeros(B, dtype=wp.vec3, device=device)
    wp.launch(_settle_only, B, inputs=[elev, g, robot, sp, pose, u_star], device=device)
    J = wp.zeros(B, dtype=wp.mat33, device=device)
    wp.launch(_settle_jac, B, inputs=[elev, g, robot, pose, u_star, float(jac_eps), J],
              device=device)
    adj = wp.array(np.asarray(adj_u, np.float32), dtype=wp.vec3, device=device)
    minus_lam = wp.zeros(B, dtype=wp.vec3, device=device)
    wp.launch(_solve_jt, B, inputs=[J, adj, minus_lam], device=device)

    # residual VJP on the tape with cotangent -lambda -> scatters into Henv.grad
    c = wp.zeros(B, dtype=wp.vec3, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(_residual, B, inputs=[elev, g, robot, pose, u_star], outputs=[c],
                  device=device)
    tape.backward(grads={c: minus_lam})
    return elev.grad.numpy(), u_star.numpy()


def _selftest():
    """Implicit d/dHenv vs finite differences (numpy settle oracle), on the
    nonzero (contact) cells only."""
    from kinematic_helhest import heightmap as hmmod
    from kinematic_helhest.reference import placement
    from kinematic_helhest.engine import SolverParams

    wp.init()
    params = SolverParams(newton_iters=12)
    cases = [("ramp", hmmod.ramp_scene(), [(2.0, 0.0, 0.0), (3.0, 0.3, 0.2)]),
             ("box", hmmod.box_scene(), [(0.9, 0.0, 0.0)])]
    adj_template = np.array([0.3, 1.0, 0.5], np.float32)  # weights on (z, pitch, roll)
    worst = 0.0
    for name, scene, poses in cases:
        env = hmmod.wheel_envelope(scene, 0.35)
        adj_u = np.tile(adj_template, (len(poses), 1))
        g_imp, _ = dsettle_dHenv(env, poses, adj_u, params)

        # FD only on cells the implicit grad marks nonzero (the contact stencils)
        cells = list(zip(*np.where(np.abs(g_imp) > 1e-6)))
        eps = 1e-3
        err = 0.0
        for (i, j) in cells:
            gp = _fd_loss(env, poses, adj_u, i, j, +eps)
            gm = _fd_loss(env, poses, adj_u, i, j, -eps)
            g_fd = (gp - gm) / (2 * eps)
            err = max(err, abs(g_imp[i, j] - g_fd))
        worst = max(worst, err)
        print(f"  {name:4s}  {len(cells)} contact cells  max|g_imp-g_fd|={err:.2e}  "
              f"||g||={np.abs(g_imp).max():.3f}")
    print(f"implicit settle d/dHenv vs FD  worst={worst:.2e}  "
          f"{'OK' if worst < 5e-2 else 'REVIEW'}")


def _fd_loss(env, poses, adj_u, i, j, delta):
    from kinematic_helhest import heightmap as hmmod
    from kinematic_helhest.reference import placement

    Hp = env.H.copy()
    Hp[i, j] += delta
    hm = hmmod.Heightmap(Hp, (env.x0, env.y0), env.cell)
    total = 0.0
    for p, (x, y, yaw) in enumerate(poses):
        s = placement.settle(x, y, yaw, hm)
        u = np.array([s["z"], s["pitch"], s["roll"]])
        total += float(adj_u[p] @ u)
    return total


@wp.kernel
def _row_loss(controlled: wp.array2d(dtype=wp.vec3), derived: wp.array2d(dtype=wp.vec3),
              wpv: wp.vec3, wtv: wp.vec3, row: int, loss: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(loss, 0, wp.dot(wpv, controlled[row, tid]) + wp.dot(wtv, derived[row, tid]))


def _gmeta(hm):
    from kinematic_helhest.engine import Grid
    g = Grid()
    g.origin_x, g.origin_y, g.cell_size = float(hm.x0), float(hm.y0), float(hm.cell)
    g.cells_x, g.cells_y = int(hm.nx), int(hm.ny)
    return g


def _fwd(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, grad=False):
    """Forward init + T steps + loss on the FINAL state (B=1). If grad, taped
    backward -> (loss, gHenv, gHmu). T inferred from omega_np."""
    from kinematic_helhest.engine import init_state_kernel, step_kernel

    dev = "cpu"
    T = omega_np.shape[0]
    Henv = wp.array(envH, dtype=wp.float32, device=dev, requires_grad=grad)
    Hraw = wp.array(rawH, dtype=wp.float32, device=dev)
    Hmu = wp.array(muH, dtype=wp.float32, device=dev, requires_grad=grad)
    omega = wp.array(omega_np, dtype=wp.vec3, device=dev)
    pose0 = wp.array(np.asarray([init_pose], np.float32), dtype=wp.vec3, device=dev)
    controlled = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev, requires_grad=grad)
    derived = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev, requires_grad=grad)
    loads = wp.zeros((T, 1), dtype=wp.vec3, device=dev)
    turn = wp.zeros((T, 1), dtype=wp.vec2, device=dev)
    clear = wp.zeros((T, 1), dtype=float, device=dev)
    resid = wp.zeros((T, 1), dtype=float, device=dev)
    loss = wp.zeros(1, dtype=float, device=dev, requires_grad=grad)

    def launches():
        wp.launch(init_state_kernel, 1, inputs=[Henv, g, robot, sp, pose0],
                  outputs=[controlled, derived], device=dev)
        for t in range(T):
            wp.launch(step_kernel, 1, inputs=[Henv, Hraw, Hmu, g, robot, sp, omega[t], controlled[t], derived[t]],
                      outputs=[controlled[t + 1], derived[t + 1], loads[t], turn[t], clear[t], resid[t]], device=dev)
        wp.launch(_row_loss, 1, inputs=[controlled, derived, wpv, wtv, T], outputs=[loss], device=dev)

    if not grad:
        launches()
        return float(loss.numpy()[0])
    tape = wp.Tape()
    with tape:
        launches()
    tape.backward(loss=loss)
    return float(loss.numpy()[0]), Henv.grad.numpy(), Hmu.grad.numpy()


def _selftest_step_grad():
    """One full step on the tape: d(loss)/dHenv and d(loss)/dHmu vs finite diff."""
    from kinematic_helhest import friction
    from kinematic_helhest import heightmap as hmmod
    from kinematic_helhest.engine import RobotParams, SolverParams

    wp.init()
    scene = hmmod.flat()
    env = hmmod.wheel_envelope(scene, 0.35)
    mu = friction.uniform(0.8)
    robot = RobotParams().build("cpu")
    sp = SolverParams(newton_iters=12, dt=0.05, k_turn=2.0).build()
    g = _gmeta(env)
    omega_np = np.array([[[1.0, 2.0, 1.5]]], np.float32)  # [T=1,B=1,3]: a turn
    wpv, wtv = wp.vec3(0.5, 0.3, 1.0), wp.vec3(0.2, 1.0, 0.5)
    init_pose = (0.0, 0.0, 0.0)

    envH = np.ascontiguousarray(env.H, np.float32)
    rawH = np.ascontiguousarray(scene.H, np.float32)
    muH = np.ascontiguousarray(mu.H, np.float32)

    _, gHenv, gHmu = _fwd(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, grad=True)

    eps = 1e-3
    err_e = _fd_grid(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv,
                     gHenv, "env", eps)
    err_m = _fd_grid(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv,
                     gHmu, "mu", eps)
    worst = max(err_e, err_m)
    print(f"  dHenv: {np.count_nonzero(np.abs(gHenv) > 1e-5)} cells ||g||={np.abs(gHenv).max():.3f}  "
          f"max|err|={err_e:.2e}")
    print(f"  dHmu : {np.count_nonzero(np.abs(gHmu) > 1e-5)} cells ||g||={np.abs(gHmu).max():.3f}  "
          f"max|err|={err_m:.2e}")
    print(f"step grad d/dHenv,d/dHmu vs FD  worst={worst:.2e}  {'OK' if worst < 5e-2 else 'REVIEW'}")


def _fd_grid(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, g_an, which, eps):
    cells = list(zip(*np.where(np.abs(g_an) > 1e-5)))
    err = 0.0
    for (i, j) in cells:
        def loss_at(delta):
            e, r, m = envH.copy(), rawH.copy(), muH.copy()
            (e if which == "env" else m)[i, j] += delta
            return _fwd(e, r, m, g, robot, sp, omega_np, init_pose, wpv, wtv)
        g_fd = (loss_at(+eps) - loss_at(-eps)) / (2 * eps)
        err = max(err, abs(g_an[i, j] - g_fd))
    return err


def _fwd_h(rawH, muH, g, Rwheel, robot, sp, omega_np, init_pose, wpv, wtv, grad=False):
    """Like _fwd but the leaf is the RAW heightmap: Henv = wheel_envelope(rawH) is
    computed on the tape, so backward yields d(loss)/d(raw h)."""
    from kinematic_helhest.engine import init_state_kernel, step_kernel

    dev = "cpu"
    T = omega_np.shape[0]
    Hraw = wp.array(rawH, dtype=wp.float32, device=dev, requires_grad=grad)
    Hmu = wp.array(muH, dtype=wp.float32, device=dev, requires_grad=grad)
    omega = wp.array(omega_np, dtype=wp.vec3, device=dev)
    pose0 = wp.array(np.asarray([init_pose], np.float32), dtype=wp.vec3, device=dev)
    controlled = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev, requires_grad=grad)
    derived = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev, requires_grad=grad)
    loads = wp.zeros((T, 1), dtype=wp.vec3, device=dev)
    turn = wp.zeros((T, 1), dtype=wp.vec2, device=dev)
    clear = wp.zeros((T, 1), dtype=float, device=dev)
    resid = wp.zeros((T, 1), dtype=float, device=dev)
    loss = wp.zeros(1, dtype=float, device=dev, requires_grad=grad)

    def launches():
        Henv = wheel_envelope(Hraw, g.cell_size, Rwheel, dev)  # raw h -> envelope, on the tape
        wp.launch(init_state_kernel, 1, inputs=[Henv, g, robot, sp, pose0],
                  outputs=[controlled, derived], device=dev)
        for t in range(T):
            wp.launch(step_kernel, 1, inputs=[Henv, Hraw, Hmu, g, robot, sp, omega[t], controlled[t], derived[t]],
                      outputs=[controlled[t + 1], derived[t + 1], loads[t], turn[t], clear[t], resid[t]], device=dev)
        wp.launch(_row_loss, 1, inputs=[controlled, derived, wpv, wtv, T], outputs=[loss], device=dev)

    if not grad:
        launches()
        return float(loss.numpy()[0])
    tape = wp.Tape()
    with tape:
        launches()
    tape.backward(loss=loss)
    return float(loss.numpy()[0]), Hraw.grad.numpy(), Hmu.grad.numpy()


def _selftest_dh():
    """End-to-end d(loss)/d(raw h) through the envelope dilation + rollout vs FD."""
    from kinematic_helhest import friction
    from kinematic_helhest import heightmap as hmmod
    from kinematic_helhest.engine import RobotParams, SolverParams

    wp.init()
    T, R = 8, 0.35
    scene = hmmod.ramp_scene()  # sloped: envelope arg-max routes uphill (non-trivial)
    mu = friction.uniform(0.8, xlim=(scene.x0, scene.x0 + (scene.nx - 1) * scene.cell))
    robot = RobotParams().build("cpu")
    sp = SolverParams(newton_iters=12, dt=0.05, k_turn=2.0).build()
    g = _gmeta(scene)
    omega_np = np.tile([1.0, 2.0, 1.5], (T, 1, 1)).astype(np.float32)
    wpv, wtv = wp.vec3(0.5, 0.3, 1.0), wp.vec3(0.2, 1.0, 0.5)
    init_pose = (1.5, 0.0, 0.0)  # on the slope

    rawH = np.ascontiguousarray(scene.H, np.float32)
    muH = np.ascontiguousarray(mu.H, np.float32)

    _, gH, gMu = _fwd_h(rawH, muH, g, R, robot, sp, omega_np, init_pose, wpv, wtv, grad=True)

    eps = 1e-3
    fwd = lambda rh, mh: _fwd_h(rh, mh, g, R, robot, sp, omega_np, init_pose, wpv, wtv)
    err_h = _fd_cells(rawH, muH, gH, "raw", eps, fwd)
    err_m = _fd_cells(rawH, muH, gMu, "mu", eps, fwd)
    worst = max(err_h, err_m)
    print(f"  d/d(raw h): {np.count_nonzero(np.abs(gH) > 1e-5)} cells "
          f"||g||={np.abs(gH).max():.3f}  max|err|={err_h:.2e}")
    print(f"  d/dHmu    : {np.count_nonzero(np.abs(gMu) > 1e-5)} cells "
          f"||g||={np.abs(gMu).max():.3f}  max|err|={err_m:.2e}")
    print(f"end-to-end d/d(raw h) vs FD  worst={worst:.2e}  {'OK' if worst < 5e-2 else 'REVIEW'}")


def _fd_cells(rawH, muH, g_an, which, eps, fwd):
    cells = list(zip(*np.where(np.abs(g_an) > 1e-5)))
    err = 0.0
    for (i, j) in cells:
        rp, mp = rawH.copy(), muH.copy()
        rm, mm = rawH.copy(), muH.copy()
        (rp if which == "raw" else mp)[i, j] += eps
        (rm if which == "raw" else mm)[i, j] -= eps
        g_fd = (fwd(rp, mp) - fwd(rm, mm)) / (2 * eps)
        err = max(err, abs(g_an[i, j] - g_fd))
    return err


def _fwd_batch(envH, rawH, muH, g, robot, sp, omega_np, poses, wpv, wtv, grad=False):
    """Batched (B>1) forward init + T steps + summed loss over all rollouts.
    omega_np: [T, B, 3]; poses: [B, 3]. Grads accumulate into shared Henv/Hmu."""
    from kinematic_helhest.engine import init_state_kernel, step_kernel

    dev = "cpu"
    T, B = omega_np.shape[0], omega_np.shape[1]
    Henv = wp.array(envH, dtype=wp.float32, device=dev, requires_grad=grad)
    Hraw = wp.array(rawH, dtype=wp.float32, device=dev)
    Hmu = wp.array(muH, dtype=wp.float32, device=dev, requires_grad=grad)
    omega = wp.array(omega_np, dtype=wp.vec3, device=dev)
    pose0 = wp.array(np.asarray(poses, np.float32), dtype=wp.vec3, device=dev)
    controlled = wp.zeros((T + 1, B), dtype=wp.vec3, device=dev, requires_grad=grad)
    derived = wp.zeros((T + 1, B), dtype=wp.vec3, device=dev, requires_grad=grad)
    loads = wp.zeros((T, B), dtype=wp.vec3, device=dev)
    turn = wp.zeros((T, B), dtype=wp.vec2, device=dev)
    clear = wp.zeros((T, B), dtype=float, device=dev)
    resid = wp.zeros((T, B), dtype=float, device=dev)
    loss = wp.zeros(1, dtype=float, device=dev, requires_grad=grad)

    def launches():
        wp.launch(init_state_kernel, B, inputs=[Henv, g, robot, sp, pose0],
                  outputs=[controlled, derived], device=dev)
        for t in range(T):
            wp.launch(step_kernel, B, inputs=[Henv, Hraw, Hmu, g, robot, sp, omega[t], controlled[t], derived[t]],
                      outputs=[controlled[t + 1], derived[t + 1], loads[t], turn[t], clear[t], resid[t]], device=dev)
        wp.launch(_row_loss, B, inputs=[controlled, derived, wpv, wtv, T], outputs=[loss], device=dev)

    if not grad:
        launches()
        return float(loss.numpy()[0])
    tape = wp.Tape()
    with tape:
        launches()
    tape.backward(loss=loss)
    return float(loss.numpy()[0]), Henv.grad.numpy(), Hmu.grad.numpy()


def _selftest_batch():
    """Batched B rollouts == sum of B solo rollouts (forward loss + grads)."""
    from kinematic_helhest import friction
    from kinematic_helhest import heightmap as hmmod
    from kinematic_helhest.engine import RobotParams, SolverParams

    wp.init()
    T, B = 5, 4
    scene = hmmod.flat()
    env = hmmod.wheel_envelope(scene, 0.35)
    mu = friction.uniform(0.8)
    robot = RobotParams().build("cpu")
    sp = SolverParams(newton_iters=12, dt=0.05, k_turn=2.0).build()
    g = _gmeta(env)
    wpv, wtv = wp.vec3(0.5, 0.3, 1.0), wp.vec3(0.2, 1.0, 0.5)

    # B distinct rollouts: different turn rates and start poses
    omega = np.stack([np.tile([1.0, 1.0 + 0.4 * b, 1.5], (T, 1)) for b in range(B)], axis=1)
    omega = omega.astype(np.float32)  # [T, B, 3]
    poses = np.array([[0.0, 0.3 * b, 0.1 * b] for b in range(B)], np.float32)

    envH = np.ascontiguousarray(env.H, np.float32)
    rawH = np.ascontiguousarray(scene.H, np.float32)
    muH = np.ascontiguousarray(mu.H, np.float32)

    lb, gHb, gMb = _fwd_batch(envH, rawH, muH, g, robot, sp, omega, poses, wpv, wtv, grad=True)

    ls, gHs, gMs = 0.0, np.zeros_like(gHb), np.zeros_like(gMb)
    for b in range(B):
        lo, gh, gm = _fwd_batch(envH, rawH, muH, g, robot, sp,
                                omega[:, b:b + 1], poses[b:b + 1], wpv, wtv, grad=True)
        ls += lo
        gHs += gh
        gMs += gm

    d_loss = abs(lb - ls)
    d_H = np.abs(gHb - gHs).max()
    d_M = np.abs(gMb - gMs).max()
    worst = max(d_loss, d_H, d_M)
    print(f"  B={B} T={T}  |loss_batch-sum_solo|={d_loss:.2e}  "
          f"dgHenv={d_H:.2e}  dgHmu={d_M:.2e}")
    print(f"batch == sum-of-solo  worst={worst:.2e}  {'OK' if worst < 1e-3 else 'REVIEW'}")


def _selftest_bptt():
    """BPTT over a T-step rollout: d(loss on final state)/dHenv,dHmu vs finite diff."""
    from kinematic_helhest import friction
    from kinematic_helhest import heightmap as hmmod
    from kinematic_helhest.engine import RobotParams, SolverParams

    wp.init()
    T = 8
    scene = hmmod.flat()
    env = hmmod.wheel_envelope(scene, 0.35)
    mu = friction.uniform(0.8)
    robot = RobotParams().build("cpu")
    sp = SolverParams(newton_iters=12, dt=0.05, k_turn=2.0).build()
    g = _gmeta(env)
    omega_np = np.tile([1.0, 2.0, 1.5], (T, 1, 1)).astype(np.float32)  # [T,1,3]: a turn
    wpv, wtv = wp.vec3(0.5, 0.3, 1.0), wp.vec3(0.2, 1.0, 0.5)
    init_pose = (0.0, 0.0, 0.0)

    envH = np.ascontiguousarray(env.H, np.float32)
    rawH = np.ascontiguousarray(scene.H, np.float32)
    muH = np.ascontiguousarray(mu.H, np.float32)

    _, gHenv, gHmu = _fwd(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, grad=True)

    eps = 1e-3
    err_e = _fd_grid(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, gHenv, "env", eps)
    err_m = _fd_grid(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, gHmu, "mu", eps)
    worst = max(err_e, err_m)
    print(f"  T={T}  dHenv: {np.count_nonzero(np.abs(gHenv) > 1e-5)} cells "
          f"||g||={np.abs(gHenv).max():.3f}  max|err|={err_e:.2e}")
    print(f"  T={T}  dHmu : {np.count_nonzero(np.abs(gHmu) > 1e-5)} cells "
          f"||g||={np.abs(gHmu).max():.3f}  max|err|={err_m:.2e}")
    print(f"BPTT d/dHenv,d/dHmu vs FD  worst={worst:.2e}  {'OK' if worst < 5e-2 else 'REVIEW'}")


if __name__ == "__main__":
    _selftest()
    _selftest_step_grad()
    _selftest_bptt()
    _selftest_dh()
    _selftest_batch()
