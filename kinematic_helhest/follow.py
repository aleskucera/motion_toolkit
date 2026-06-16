"""Live: MPPI plans a path to a goal, you drive and try to follow it.

The Warp engine runs twice: (1) a user-controlled robot you drive with I/J/K/L,
and (2) an MPPI planner that re-plans a horizon trajectory from YOUR current pose
to the goal every few frames (the green line on the terrain, arcing around the
wall). Steer to keep the robot on the green line; it goes red if you high-center
or wedge into the wall.

Keys:  I fwd  K back  J left  L right   ESC/Q quit ; mouse orbit, scroll zoom.

Run:        python -m kinematic_helhest.follow [--device cuda] [--gx 4 --gy 1.5]
Shot test:  python -m kinematic_helhest.follow --shot /tmp/follow.png
"""
import argparse
import time

import numpy as np
import warp as wp

from . import friction
from . import heightmap as hmmod
from .drive import DT
from .drive import WIN_H
from .drive import WIN_W
from .drive import _commands
from .drive import _init_gl
from .drive import _render
from .drive import build_robot
from .drive import build_terrain
from .drive import demo_terrain
from .drive_warp import WarpDriver
from .mppi import BatchRollout
from .mppi import _cost
from .mppi import _to_omega
from .warp_engine.solver import SolverParams


class Planner:
    """Re-plans an MPPI horizon trajectory from a given pose to the goal."""

    def __init__(self, scene, mu, goal, device="cpu", T=90, B=1024, n_refine=3,
                 sigma=2.5, lam=0.5, wmax=4.0, clear_margin=0.05, resid_tol=1e-2, seed=0):
        params = SolverParams(dt=DT, k_turn=2.0, newton_iters=12)
        self.br = BatchRollout(scene, mu, B, T, params, device=device)
        self.scene, self.goal = scene, np.asarray(goal[:2], np.float64)
        self.T, self.B, self.n_refine = T, B, n_refine
        self.sigma, self.lam, self.wmax = sigma, lam, wmax
        self.cm, self.rt = clear_margin, resid_tol
        self.w = dict(term=3.0, run=0.3, invalid=1e5, eff=2e-3, smooth=2e-3)
        self.U = np.full((T, 2), 1.5, np.float32)
        self.rng = np.random.default_rng(seed)

    def replan(self, state):
        """state (x,y,yaw) -> predicted path xy [T+1, 2] from the optimized nominal."""
        B, T = self.B, self.T
        for _ in range(self.n_refine):
            eps = self.rng.normal(0.0, self.sigma, (B, T, 2)).astype(np.float32)
            eps[0] = 0.0
            Ub = np.clip(self.U[None] + eps, -self.wmax, self.wmax)
            planar, clear, resid = self.br.rollout(_to_omega(Ub), state)
            J, _ = _cost(planar, clear, resid, Ub, self.goal, self.cm, self.rt, self.w)
            beta = np.exp(-(J - J.min()) / self.lam)
            beta /= beta.sum()
            self.U = np.clip(np.einsum("b,btc->tc", beta, Ub), -self.wmax, self.wmax).astype(np.float32)
        planar, _, _ = self.br.rollout(_to_omega(np.tile(self.U, (B, 1, 1))), state)
        return planar[:, 0, :2].copy()


def _draw_plan(plan_xy, scene, goal):
    """Draw the planned path (green) and goal (red pole) in world coords."""
    from OpenGL import GL as gl
    gl.glDisable(gl.GL_LIGHTING)
    z = np.minimum(scene.sample(plan_xy[:, 0], plan_xy[:, 1]), 0.55) + 0.06  # clamp so it doesn't ride up the wall
    gl.glColor3f(1.0, 0.0, 1.0); gl.glLineWidth(5.0)  # magenta (contrasts the green terrain)
    gl.glBegin(gl.GL_LINE_STRIP)
    for (x, y), zz in zip(plan_xy, z):
        gl.glVertex3f(float(x), float(y), float(zz))
    gl.glEnd()
    gz = float(scene.sample(np.array([goal[0]]), np.array([goal[1]]))[0])
    gl.glColor3f(0.95, 0.1, 0.1); gl.glLineWidth(5.0)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(float(goal[0]), float(goal[1]), gz)
    gl.glVertex3f(float(goal[0]), float(goal[1]), gz + 1.2)
    gl.glEnd()
    gl.glEnable(gl.GL_LIGHTING)


def _pursue(state, plan_xy, speed=3.5, wmax=4.0, lookahead=0.9):
    """Pure-pursuit: steer toward a point ~lookahead ahead on the plan."""
    x, y, yaw = float(state[0]), float(state[1]), float(state[2])
    d = np.hypot(plan_xy[:, 0] - x, plan_xy[:, 1] - y)
    ahead = np.where(d > lookahead)[0]
    tx, ty = plan_xy[ahead[0]] if len(ahead) else plan_xy[-1]
    err = (np.arctan2(ty - y, tx - x) - yaw + np.pi) % (2 * np.pi) - np.pi
    turn = np.clip(2.5 * err, -wmax, wmax)
    wL, wR = np.clip(speed - turn, -wmax, wmax), np.clip(speed + turn, -wmax, wmax)
    return np.array([wL, wR, (wL + wR) / 2.0], np.float32)


def run(shot=None, device="cpu", goal=(4.0, 1.5), replan_every=4, record=None):
    import glfw
    from OpenGL import GL as gl

    hm = demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    drv = WarpDriver(hm, mu, init_pose=(0.0, 0.0, 0.0), device=device)
    planner = Planner(hm, mu, goal, device=device)
    goal = np.asarray(goal, np.float64)

    if not glfw.init():
        raise RuntimeError("glfw init failed")
    if shot:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(WIN_W, WIN_H, "Helhest — follow the MPPI plan (I/J/K/L)", None, None)
    glfw.make_context_current(win)
    _init_gl()
    terrain, robot = build_terrain(hm), build_robot()
    cam = [-2.2, 0.5, 6.0]
    mouse = {"down": False, "x": 0.0, "y": 0.0}

    def on_mouse_button(w, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["down"] = action == glfw.PRESS
            mouse["x"], mouse["y"] = glfw.get_cursor_pos(w)

    def on_cursor(w, x, y):
        if mouse["down"]:
            cam[0] -= (x - mouse["x"]) * 0.01
            cam[1] = float(np.clip(cam[1] + (y - mouse["y"]) * 0.01, -1.4, 1.4))
            mouse["x"], mouse["y"] = x, y

    def on_scroll(w, dx, dy):
        cam[2] = float(np.clip(cam[2] - dy * 0.5, 1.5, 30.0))

    glfw.set_mouse_button_callback(win, on_mouse_button)
    glfw.set_cursor_pos_callback(win, on_cursor)
    glfw.set_scroll_callback(win, on_scroll)

    st = drv.render_state()
    s0 = np.array([st.x, st.y, st.yaw], np.float32)
    for _ in range(6):  # warm up the initial plan so it's converged on frame 0
        plan_xy = planner.replan(s0)
    trail, last_status, frame, rec = [], 0.0, 0, []
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS or \
           glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break
        if shot:
            cmd = np.array([1.6, 1.6, 1.6])
        elif record is not None:
            cmd = _pursue((st.x, st.y, st.yaw), plan_xy)
        else:
            cmd = _commands(lambda k: glfw.get_key(win, k))
        drv.step(cmd)
        st = drv.render_state()
        if frame % replan_every == 0:
            plan_xy = planner.replan(np.array([st.x, st.y, st.yaw], np.float32))
        trail.append([st.x, st.y, st.place["z"] + 0.02]); trail = trail[-3000:]

        _render(st, cam, terrain, robot, trail)
        _draw_plan(plan_xy, hm, goal)

        if record is not None:
            if frame % 3 == 0:
                buf = gl.glReadPixels(0, 0, WIN_W, WIN_H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                rec.append(np.frombuffer(buf, np.uint8).reshape(WIN_H, WIN_W, 3)[::-1])
            reached = np.hypot(st.x - goal[0], st.y - goal[1]) < 0.4
            frame += 1
            if reached or frame >= 320:
                from PIL import Image
                imgs = [Image.fromarray(f).resize((640, 400)) for f in rec]
                imgs[0].save(record, save_all=True, append_images=imgs[1:], duration=70, loop=0)
                print(f"saved {record}  ({len(imgs)} frames, reached={reached})")
                break
            continue

        if shot:
            frame += 1
            if frame >= 12:
                gl.glReadBuffer(gl.GL_BACK)
                buf = gl.glReadPixels(0, 0, WIN_W, WIN_H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(buf, np.uint8).reshape(WIN_H, WIN_W, 3)[::-1]
                import matplotlib.pyplot as plt
                plt.imsave(shot, img)
                print(f"saved {shot}  robot=({st.x:.2f},{st.y:.2f}) valid={st.valid}")
                break
            continue

        glfw.swap_buffers(win)
        now = time.perf_counter()
        if now - last_status > 0.4:
            dist = float(np.hypot(st.x - goal[0], st.y - goal[1]))
            te = float(np.min(np.hypot(plan_xy[:, 0] - st.x, plan_xy[:, 1] - st.y)))
            print(f"\rpos=({st.x:+5.2f},{st.y:+5.2f}) goal_dist={dist:4.2f} "
                  f"track_err={te:4.2f} valid={st.valid}   ", end="", flush=True)
            last_status = now
        frame += 1
        time.sleep(max(0.0, DT - (time.perf_counter() - now)))

    glfw.terminate()
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", default=None)
    ap.add_argument("--record", default=None, help="auto-follow the plan and save a GIF")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gx", type=float, default=4.0)
    ap.add_argument("--gy", type=float, default=1.5)
    args = ap.parse_args()
    run(shot=args.shot, record=args.record, device=args.device, goal=(args.gx, args.gy))


if __name__ == "__main__":
    main()
