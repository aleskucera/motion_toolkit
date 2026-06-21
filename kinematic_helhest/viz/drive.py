"""Interactive driver backed by the WARP engine — the runtime path.

Every frame runs one Warp `step` launch (predict + implicit settle) on the
device. Confirms the engine behaves like the numpy oracle (reference/drive.py)
in real time. Rendering/input come from viz.render.

Keys:  I forward  K back  J turn-left  L turn-right  ESC/Q quit ; mouse orbit/zoom.

Run:        python -m kinematic_helhest.viz.drive [--device cuda]
Shot test:  python -m kinematic_helhest.viz.drive --shot /tmp/drive_warp.png
"""
import argparse
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

from .. import dynamics
from .. import friction
from .. import heightmap as hmmod
from ..engine import GridParams
from ..engine import Simulator
from ..engine import SolverParams
from ..model import euler_zyx
from .render import WIN_H
from .render import WIN_W
from .render import _commands
from .render import _init_gl
from .render import _render
from .render import build_robot
from .render import build_terrain


class WarpDriver:
    """Wraps a B=1, T=1 `Simulator` and the current pose; steps one frame per call."""

    def __init__(self, hm, mu, init_pose=(0.0, 0.0, 0.0), device="cpu", dt=dynamics.DT,
                 k_turn=dynamics.K_TURN, resid_tol=1e-2, clear_margin=0.0, tilt_clamp=1.2):
        wp.init()
        self.resid_tol, self.clear_margin = resid_tol, clear_margin
        sp = SolverParams(dt=dt, k_turn=k_turn, newton_iters=12, tilt_clamp=tilt_clamp)
        self.sim = Simulator(dynamics.robot_params(), sp,
                             GridParams(hm.nx, hm.ny, hm.cell, hm.x0, hm.y0), 1, 1, device)
        self.sim.set_terrain(wp.array(np.ascontiguousarray(hm.H, np.float32),
                                      dtype=wp.float32, device=device))
        self.sim.set_friction(mu)

        # frame 0: settle at the start pose (zero control)
        controlled, derived, _, _ = self.sim.rollout(np.zeros((1, 1, 3), np.float32), init_pose)
        self.controlled = controlled[0, 0].copy()  # (x, y, yaw)
        self.derived = derived[0, 0].copy()       # (z, pitch, roll)
        self.clear, self.alpha, self.resid = 1.0, 1.0, 0.0

    def step(self, omega3):
        omega = np.asarray(omega3, np.float32).reshape(1, 1, 3)
        controlled, derived, clear, resid = self.sim.rollout(omega, self.controlled)
        self.controlled = controlled[1, 0].copy()
        self.derived = derived[1, 0].copy()
        self.clear = float(clear[0, 0])
        self.resid = float(resid[0, 0])
        self.alpha = float(self.sim.turning.numpy()[0, 0][0])

    def render_state(self):
        x, y, yaw = (float(v) for v in self.controlled)
        z, pitch, roll = (float(v) for v in self.derived)
        R = euler_zyx(yaw, pitch, roll)
        valid = self.clear >= self.clear_margin and self.resid < self.resid_tol
        return SimpleNamespace(
            x=x, y=y, yaw=yaw, alpha=self.alpha, valid=valid,
            place={"z": z, "R": R, "pitch": pitch, "roll": roll},
        )


def run(shot=None, device="cpu", resid_tol=1e-2, clear_margin=0.0, tilt_clamp=1.2):
    import glfw
    from OpenGL import GL as gl

    hm = hmmod.demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    drv = WarpDriver(hm, mu, init_pose=(0.0, 0.0, 0.0), device=device,
                     resid_tol=resid_tol, clear_margin=clear_margin, tilt_clamp=tilt_clamp)

    if not glfw.init():
        raise RuntimeError("glfw init failed")
    if shot:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(WIN_W, WIN_H, "Helhest kinematic (WARP) — I/J/K/L drive", None, None)
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

    trail, last_status, frame = [], 0.0, 0
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS or \
           glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break
        cmd = np.array([3.0, 3.0, 3.0]) if shot else _commands(lambda k: glfw.get_key(win, k))
        drv.step(cmd)
        st = drv.render_state()
        trail.append([st.x, st.y, st.place["z"] + 0.02])
        trail = trail[-3000:]
        _render(st, cam, terrain, robot, trail)

        if shot:
            frame += 1
            if frame >= 45:
                gl.glReadBuffer(gl.GL_BACK)
                buf = gl.glReadPixels(0, 0, WIN_W, WIN_H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(buf, np.uint8).reshape(WIN_H, WIN_W, 3)[::-1]
                import matplotlib.pyplot as plt
                plt.imsave(shot, img)
                print(f"saved {shot}  pose=({st.x:.2f},{st.y:.2f}) z={st.place['z']:.2f} "
                      f"pitch={np.rad2deg(st.place['pitch']):+.1f} valid={st.valid}")
                break
            continue

        glfw.swap_buffers(win)
        now = time.perf_counter()
        if now - last_status > 0.4:
            print(f"\rpos=({st.x:+5.2f},{st.y:+5.2f}) yaw={np.rad2deg(st.yaw):+6.1f}  "
                  f"z={st.place['z']:.2f} pitch={np.rad2deg(st.place['pitch']):+5.1f} "
                  f"roll={np.rad2deg(st.place['roll']):+5.1f}  a={st.alpha:.2f} "
                  f"valid={st.valid}   ", end="", flush=True)
            last_status = now
        time.sleep(max(0.0, dynamics.DT - (time.perf_counter() - now)))

    glfw.terminate()
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", default=None, help="render ~45 auto-drive frames, save PNG, exit")
    ap.add_argument("--device", default="cpu", help="warp device: cpu or cuda")
    ap.add_argument("--resid-tol", type=float, default=1e-2, help="settle residual above which the pose is invalid (lower = stricter)")
    ap.add_argument("--clear-margin", type=float, default=0.0, help="min belly-terrain gap [m] (higher = stricter)")
    ap.add_argument("--tilt-clamp", type=float, default=1.2, help="max settle tilt [rad] (lower = refuses steeper slopes)")
    args = ap.parse_args()
    run(shot=args.shot, device=args.device, resid_tol=args.resid_tol,
        clear_margin=args.clear_margin, tilt_clamp=args.tilt_clamp)


if __name__ == "__main__":
    main()
