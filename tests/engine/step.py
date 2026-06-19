"""Engine forward physics (settle, loads, full step) vs the numpy oracle (reference/).

Run:  python -m tests.engine.step
"""
import numpy as np
import warp as wp

from kinematic_helhest import friction
from kinematic_helhest import heightmap as hmmod
from kinematic_helhest.engine import clearances
from kinematic_helhest.engine import Grid
from kinematic_helhest.engine import GridParams
from kinematic_helhest.engine import init_state_kernel
from kinematic_helhest.engine import Robot
from kinematic_helhest.engine import RobotParams
from kinematic_helhest.engine import sample_height
from kinematic_helhest.engine import sample_normal
from kinematic_helhest.engine import settle
from kinematic_helhest.engine import Solver
from kinematic_helhest.engine import SolverParams
from kinematic_helhest.engine import step_kernel
from kinematic_helhest.engine.step import chassis_clearance
from kinematic_helhest.engine.rotations import euler_zyx
from kinematic_helhest.engine.step import normal_loads
from kinematic_helhest.engine.step import rollout_kernel
from kinematic_helhest.reference import placement
from kinematic_helhest.reference import rollout as rollout_np


def _upload(hm, device, requires_grad=False):
    """Verification-only: numpy Heightmap -> (device elevation array, Grid)."""
    elev = wp.array(np.ascontiguousarray(hm.H, np.float32), dtype=wp.float32,
                    device=device, requires_grad=requires_grad)
    return elev, GridParams(hm.nx, hm.ny, hm.cell, hm.x0, hm.y0).build()


def rollout_device(scene, mu_field, setpoints, init_pose, params,
                   robot_params=None, device="cpu", resid_tol=1e-2, clear_margin=0.0):
    """Single-rollout (B=1) device rollout. Returns numpy logs to match the oracle."""
    robot_params = robot_params or RobotParams()
    robot = robot_params.build(device)
    sp = params.build()
    Rw = robot_params.wheel_radius

    te, g = _upload(hmmod.wheel_envelope(scene, Rw), device)
    tr, _ = _upload(scene, device)
    tm, _ = _upload(mu_field, device)
    setpoints = np.asarray(setpoints, np.float32)
    T = setpoints.shape[0]

    omega = wp.array(setpoints.reshape(T, 1, 3), dtype=wp.vec3, device=device)
    pose0 = wp.array(np.asarray([init_pose], np.float32), dtype=wp.vec3, device=device)
    controlled = wp.zeros((T + 1, 1), dtype=wp.vec3, device=device)
    derived = wp.zeros((T + 1, 1), dtype=wp.vec3, device=device)
    loads = wp.zeros((T, 1), dtype=wp.vec3, device=device)
    turn = wp.zeros((T, 1), dtype=wp.vec2, device=device)
    clear = wp.zeros((T, 1), dtype=float, device=device)
    resid = wp.zeros((T, 1), dtype=float, device=device)

    wp.launch(init_state_kernel, 1, inputs=[te, g, robot, sp, pose0],
              outputs=[controlled, derived], device=device)
    for t in range(T):
        wp.launch(step_kernel, 1,
                  inputs=[te, tr, tm, g, robot, sp, omega[t], controlled[t], derived[t]],
                  outputs=[controlled[t + 1], derived[t + 1], loads[t], turn[t], clear[t], resid[t]],
                  device=device)
    clear_np, resid_np = clear.numpy()[:, 0], resid.numpy()[:, 0]
    bad = (clear_np < clear_margin) | (resid_np > resid_tol)
    return {
        "controlled": controlled.numpy()[:, 0, :], "derived": derived.numpy()[:, 0, :],
        "loads": loads.numpy()[:, 0, :], "turn": turn.numpy()[:, 0, :],
        "clear": clear_np, "residual": resid_np,
        "valid": not bool(bad.any()),
        "first_invalid": int(np.argmax(bad)) if bad.any() else -1,
    }


@wp.kernel
def _settle_probe(H: wp.array2d(dtype=wp.float32), g: Grid, robot: Robot, sp: Solver,
                  pose: wp.array(dtype=wp.vec3),
                  u_out: wp.array(dtype=wp.vec3), contacts: wp.array(dtype=wp.vec3),
                  normals: wp.array(dtype=wp.vec3), residual: wp.array(dtype=float)):
    tid = wp.tid()
    x = pose[tid][0]
    y = pose[tid][1]
    yaw = pose[tid][2]
    z0 = sample_height(H, g, x, y) + robot.wheel_radius
    u = settle(H, g, robot, sp, wp.vec3(x, y, yaw), wp.vec3(z0, 0.0, 0.0))
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
                 g: Grid, robot: Robot, sp: Solver, pose: wp.array(dtype=wp.vec3),
                 loads: wp.array(dtype=wp.vec3), clearance: wp.array(dtype=float)):
    tid = wp.tid()
    x = pose[tid][0]
    y = pose[tid][1]
    yaw = pose[tid][2]
    z0 = sample_height(Henv, g, x, y) + robot.wheel_radius
    u = settle(Henv, g, robot, sp, wp.vec3(x, y, yaw), wp.vec3(z0, 0.0, 0.0))
    R = euler_zyx(yaw, u[1], u[2])
    p = wp.vec3(x, y, u[0])
    loads[tid] = normal_loads(Henv, g, robot, R, p)
    clearance[tid] = chassis_clearance(Hraw, g, robot, R, p)


def _build_test(device="cpu", iters=12):
    robot = RobotParams().build(device)
    sp = SolverParams(newton_iters=iters, tilt_clamp=1.2).build()
    return robot, sp


def selftest_settle():
    wp.init()
    robot, sp = _build_test()
    cases = [("flat", hmmod.flat(), [(0.0, 0.0, 0.0), (1.0, 0.5, 0.7)]),
             ("ramp", hmmod.ramp_scene(), [(2.0, 0.0, 0.0), (3.0, 0.5, 0.3)]),
             ("box", hmmod.box_scene(), [(-1.0, 0.0, 0.0), (0.5, 0.0, 0.0)])]
    worst = 0.0
    for name, scene, poses in cases:
        env = hmmod.wheel_envelope(scene, 0.35)
        te, g = _upload(env, "cpu")
        B = len(poses)
        pose = wp.array(np.asarray(poses, np.float32), dtype=wp.vec3, device="cpu")
        u_out = wp.zeros(B, dtype=wp.vec3, device="cpu")
        contacts = wp.zeros(B * 3, dtype=wp.vec3, device="cpu")
        normals = wp.zeros(B * 3, dtype=wp.vec3, device="cpu")
        resid = wp.zeros(B, dtype=float, device="cpu")
        wp.launch(_settle_probe, B, inputs=[te, g, robot, sp, pose],
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


def selftest_loads():
    wp.init()
    robot, sp = _build_test()
    cases = [("flat", hmmod.flat(), [(0.0, 0.0, 0.0), (1.0, 0.5, 0.7)]),
             ("ramp", hmmod.ramp_scene(), [(2.0, 0.0, 0.0), (3.0, 0.5, 0.3)]),
             ("box", hmmod.box_scene(), [(-1.0, 0.0, 0.0), (0.9, 0.0, 0.0)])]
    worst = 0.0
    for name, scene, poses in cases:
        env, raw = hmmod.wheel_envelope(scene, 0.35), scene
        te, g = _upload(env, "cpu")
        tr, _ = _upload(raw, "cpu")
        B = len(poses)
        pose = wp.array(np.asarray(poses, np.float32), dtype=wp.vec3, device="cpu")
        loads = wp.zeros(B, dtype=wp.vec3, device="cpu")
        clear = wp.zeros(B, dtype=float, device="cpu")
        wp.launch(_loads_probe, B, inputs=[te, tr, g, robot, sp, pose],
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


def selftest_step():
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
        d_xy = np.abs(out["controlled"][1:, :2] - ref["pose2"][:, :2]).max()
        d_yaw = np.abs(out["controlled"][1:, 2] - ref["pose2"][:, 2]).max()
        d_N = np.abs(out["loads"] - ref["loads"]).max()
        d_a = np.abs(out["turn"][:, 0] - ref["alpha"]).max()
        d_x = np.abs(out["turn"][:, 1] - ref["x_icr"]).max()
        d_c = np.abs(out["clear"] - ref["chassis_clear"]).max()
        worst = max(worst, d_xy, d_yaw, d_N * 1e-3, d_c)
        print(f"  {name:9s} dXY={d_xy:.2e} dyaw={d_yaw:.2e} dN={d_N:.2e} "
              f"dalpha={d_a:.2e} dxicr={d_x:.2e} dclear={d_c:.2e}")
    print(f"step device-vs-oracle  {'OK' if worst < 5e-3 else 'REVIEW'}")


def selftest_rollout_kernel():
    """The fused forward rollout_kernel MUST equal init_state_kernel + T*step_kernel
    bit-for-bit (it duplicates their physics for the register-carry fusion)."""
    wp.init()
    scene, mu = hmmod.box_scene(), friction.uniform(0.8)
    robot_params = RobotParams()
    robot = robot_params.build("cpu")
    sp = SolverParams(newton_iters=12, dt=0.05, k_turn=2.0).build()
    te, g = _upload(hmmod.wheel_envelope(scene, robot_params.wheel_radius), "cpu")
    tr, _ = _upload(scene, "cpu")
    tm, _ = _upload(mu, "cpu")
    B, T = 8, 30
    rng = np.random.default_rng(0)
    omega = wp.array(np.clip(2.0 + rng.normal(0, 1.0, (T, B, 3)), -4, 4).astype(np.float32),
                     dtype=wp.vec3, device="cpu")
    pose0 = wp.array(np.tile([-1.0, 0.0, 0.0], (B, 1)).astype(np.float32), dtype=wp.vec3, device="cpu")

    def buffers():
        return [wp.zeros((T + 1, B), dtype=wp.vec3, device="cpu"),
                wp.zeros((T + 1, B), dtype=wp.vec3, device="cpu"),
                wp.zeros((T, B), dtype=wp.vec3, device="cpu"), wp.zeros((T, B), dtype=wp.vec2, device="cpu"),
                wp.zeros((T, B), dtype=float, device="cpu"), wp.zeros((T, B), dtype=float, device="cpu")]

    fused = buffers()
    wp.launch(rollout_kernel, B, inputs=[T, te, tr, tm, g, robot, sp, pose0, omega],
              outputs=fused, device="cpu")
    perstep = buffers()
    wp.launch(init_state_kernel, B, inputs=[te, g, robot, sp, pose0],
              outputs=[perstep[0], perstep[1]], device="cpu")
    for t in range(T):
        wp.launch(step_kernel, B, inputs=[te, tr, tm, g, robot, sp, omega[t], perstep[0][t], perstep[1][t]],
                  outputs=[perstep[0][t + 1], perstep[1][t + 1], perstep[2][t], perstep[3][t],
                           perstep[4][t], perstep[5][t]], device="cpu")
    worst = max(np.abs(f.numpy() - s.numpy()).max() for f, s in zip(fused, perstep))
    print(f"fused rollout_kernel == per-step  worst={worst:.2e}  {'OK' if worst == 0.0 else 'REVIEW'}")


if __name__ == "__main__":
    selftest_settle()
    selftest_loads()
    selftest_step()
    selftest_rollout_kernel()
