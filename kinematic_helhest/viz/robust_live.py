"""Real-time view of robust (CVaR) MPPI planning past the wall.

The robot drives itself to a goal using the robust MppiGpu (K slip scenarios, CVaR). Every
cycle it draws the chosen plan (YELLOW) and its SLIP FAN -- the chosen control rolled out under
the K scenarios, each path CYAN if it stays feasible / RED if it high-centers. The robust
planner steers to keep the fan cyan: that cyan margin IS the robustness (a non-robust plan
would let the fan dip red into the wall -- see planning/robust.py --animate for that contrast).

Run:        python -m kinematic_helhest.viz.robust_live [--device cuda] [--K 8]
Shot test:  python -m kinematic_helhest.viz.robust_live --shot /tmp/robust_live.png
"""
import argparse

import numpy as np

from .. import heightmap as hmmod
from ..engine import GridParams
from ..engine import RobotParams
from ..engine import Simulator
from ..engine import SolverParams
from ..planning.mppi_gpu import MppiGpu
from .drive import WarpDriver
from .render import WIN_H
from .render import WIN_W
from .render import _init_gl
from .render import _render
from .render import build_robot
from .render import build_terrain

_CLIP, _CLEAR, _PLAN = (0.95, 0.05, 0.05), (0.0, 0.85, 0.95), (1.0, 0.85, 0.05)  # red/cyan/yellow vs green terrain


def _polyline(scene, xy, color, width, dz):
    from OpenGL import GL as gl
    z = np.minimum(scene.sample(xy[:, 0], xy[:, 1]), 0.7) + dz
    gl.glColor3f(*color); gl.glLineWidth(width)
    gl.glBegin(gl.GL_LINE_STRIP)
    for (x, y), zz in zip(xy, z):
        gl.glVertex3f(float(x), float(y), float(zz))
    gl.glEnd()


def _fan_omega(U, slips):
    """chosen control U [T, 2] x slips [K, 2] -> omega [T, K, 3] (k=0 = no slip = the plan)."""
    eff = U[None, :, :] * slips[:, None, :]            # [K, T, 2]
    rear = eff.mean(2, keepdims=True)
    return np.ascontiguousarray(np.concatenate([eff, rear], 2).transpose(1, 0, 2), np.float32)


def run(shot=None, device="cuda", K=8, slip_lo=0.5, beta=0.4, goal=(4.0, 1.15), T=80, B=4096):
    import glfw

    scene = hmmod.demo_terrain()
    mu = hmmod.Heightmap(np.full((scene.ny, scene.nx), 0.8, np.float32), (scene.x0, scene.y0), scene.cell)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    params = SolverParams(dt=0.1, k_turn=2.0, newton_iters=6, atol=1e-4)
    import warp as wp
    terr = wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)

    drv = WarpDriver(scene, mu, init_pose=(0.0, 0.0, 0.0), device=device)  # the driven robot
    plan_sim = Simulator(RobotParams(), params, grid, B, T, device)
    plan_sim.set_terrain(terr); plan_sim.set_friction(mu)
    fan_sim = Simulator(RobotParams(), params, grid, K, T, device)         # roll the plan under K slips
    fan_sim.set_terrain(terr); fan_sim.set_friction(mu)

    w = dict(term=3.0, run=0.3, head=2.0, invalid=1e5, eff=2e-3, smooth=2e-3)
    planner = MppiGpu(plan_sim, 0.5, 4.0, w, 0.05, 1e-2, 0, sigma_knot=1.0, n_knots=4,
                      n_scenarios=K, cvar_beta=beta, slip_lo=slip_lo)
    planner.reset_nominal(1.5)
    slips = planner.slip.numpy()
    goal = np.asarray(goal, np.float64)

    if not glfw.init():
        raise RuntimeError("glfw init failed")
    if shot:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(WIN_W, WIN_H, "Helhest — robust CVaR planning (slip fan)", None, None)
    glfw.make_context_current(win)
    _init_gl()
    from OpenGL import GL as gl
    terrain, robot = build_terrain(scene), build_robot()
    cam = [-2.1, 0.62, 8.5]
    mouse = {"down": False, "x": 0.0, "y": 0.0}

    def on_button(w_, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["down"] = action == glfw.PRESS
            mouse["x"], mouse["y"] = glfw.get_cursor_pos(w_)

    def on_cursor(w_, x, y):
        if mouse["down"]:
            cam[0] -= (x - mouse["x"]) * 0.01
            cam[1] = float(np.clip(cam[1] + (y - mouse["y"]) * 0.01, 0.05, 1.5))
            mouse["x"], mouse["y"] = x, y

    def on_scroll(w_, dx, dy):
        cam[2] = float(np.clip(cam[2] - dy * 0.5, 2.0, 30.0))

    glfw.set_mouse_button_callback(win, on_button)
    glfw.set_cursor_pos_callback(win, on_cursor)
    glfw.set_scroll_callback(win, on_scroll)

    trail, frame = [], 0
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS or glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break
        st = drv.render_state()
        state = np.array([st.x, st.y, st.yaw], np.float32)
        reached = np.hypot(st.x - goal[0], st.y - goal[1]) < 0.3
        if not reached:
            planner.replan(state, goal, 3)
            U = planner.nominal()
            controlled, _, clearance, residual = fan_sim.rollout(_fan_omega(U, slips), state)
            cmd = np.array([U[0, 0], U[0, 1], 0.5 * (U[0, 0] + U[0, 1])], np.float32)
            drv.step(cmd)
            trail.append([st.x, st.y, st.place["z"] + 0.02]); trail = trail[-3000:]

        _render(st, cam, terrain, robot, trail)
        # draw the slip fan: each scenario red if it high-centers, plan (k=0) thick green
        bad = (clearance < 0.05) | (residual > 1e-2)  # [T, K]
        gl.glDisable(gl.GL_LIGHTING)
        for k in range(1, K):
            _polyline(scene, controlled[:, k, :2], _CLIP if bad[:, k].any() else _CLEAR, 2.5, 0.05)
        _polyline(scene, controlled[:, 0, :2], _PLAN, 5.0, 0.09)  # the chosen plan
        gz = float(scene.sample(np.array([goal[0]]), np.array([goal[1]]))[0])
        gl.glColor3f(0.95, 0.1, 0.1); gl.glLineWidth(5.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(goal[0], goal[1], gz); gl.glVertex3f(goal[0], goal[1], gz + 1.2)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)

        if shot:
            frame += 1
            if frame >= 14:  # plan reaching the wall, fan spread visible
                gl.glReadBuffer(gl.GL_BACK)
                buf = gl.glReadPixels(0, 0, WIN_W, WIN_H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(buf, np.uint8).reshape(WIN_H, WIN_W, 3)[::-1]
                import matplotlib.pyplot as plt
                plt.imsave(shot, img); print(f"saved {shot}")
                break
            continue
        glfw.swap_buffers(win)
    glfw.terminate()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shot", default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--slip-lo", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.4)
    args = ap.parse_args()
    run(shot=args.shot, device=args.device, K=args.K, slip_lo=args.slip_lo, beta=args.beta)


if __name__ == "__main__":
    main()
