"""Interactive driver backed by the WARP engine (vs drive.py's numpy state.step).

Same feel/controls as drive.py, but every frame runs one Warp `step` launch
(predict + implicit settle) on the device. Confirms the ported engine behaves
like the numpy oracle in real time. Rendering/input are reused from drive.py.

Keys:  I forward  K back  J turn-left  L turn-right  ESC/Q quit ; mouse orbit/zoom.

Run:        python -m kinematic_helhest.drive_warp [--device cuda]
Shot test:  python -m kinematic_helhest.drive_warp --shot /tmp/drive_warp.png
"""
import argparse
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

from . import friction
from . import heightmap as hmmod
from . import placement
from .drive import DT
from .drive import WIN_H
from .drive import WIN_W
from .drive import _commands
from .drive import _init_gl
from .drive import _render
from .drive import build_robot
from .drive import build_terrain
from .drive import demo_terrain
from .model import WHEEL_RADIUS
from .warp_engine.kinematics import init_state
from .warp_engine.kinematics import step as wstep
from .warp_engine.solver import RobotParams
from .warp_engine.solver import SolverParams
from .warp_engine.terrain import to_terrain


class WarpDriver:
    """Holds the device terrain/robot/solver and the current (B=1) state; steps it."""

    def __init__(self, hm, mu, init_pose=(0.0, 0.0, 0.0), device="cpu", dt=DT, k_turn=2.0):
        wp.init()
        self.dev = device
        self.te = to_terrain(hmmod.wheel_envelope(hm, WHEEL_RADIUS), device)
        self.tr = to_terrain(hm, device)
        self.tm = to_terrain(mu, device)
        self.robot = RobotParams().build(device)
        self.sp = SolverParams(dt=dt, k_turn=k_turn, newton_iters=12).build()

        planar = wp.zeros((2, 1), dtype=wp.vec3, device=device)
        tilt = wp.zeros((2, 1), dtype=wp.vec3, device=device)
        pose0 = wp.array(np.asarray([init_pose], np.float32), dtype=wp.vec3, device=device)
        wp.launch(init_state, 1, inputs=[self.te.H, self.te.g, self.robot, self.sp, pose0],
                  outputs=[planar, tilt], device=device)
        self.planar = planar.numpy()[0, 0].copy()  # (x, y, yaw)
        self.tilt = tilt.numpy()[0, 0].copy()       # (z, pitch, roll)
        self.clear, self.alpha, self.resid = 1.0, 1.0, 0.0

    def step(self, omega3):
        dev = self.dev
        pn = np.zeros((2, 1, 3), np.float32); pn[0, 0] = self.planar
        tn = np.zeros((2, 1, 3), np.float32); tn[0, 0] = self.tilt
        planar = wp.array(pn, dtype=wp.vec3, device=dev)
        tilt = wp.array(tn, dtype=wp.vec3, device=dev)
        omega = wp.array(np.asarray([[omega3]], np.float32), dtype=wp.vec3, device=dev)
        loads = wp.zeros((1, 1), dtype=wp.vec3, device=dev)
        turn = wp.zeros((1, 1), dtype=wp.vec2, device=dev)
        clear = wp.zeros((1, 1), dtype=float, device=dev)
        resid = wp.zeros((1, 1), dtype=float, device=dev)
        wp.launch(wstep, 1,
                  inputs=[0, self.te.H, self.tr.H, self.te.g, self.tm.H, self.tm.g,
                          self.robot, self.sp, omega],
                  outputs=[planar, tilt, loads, turn, clear, resid], device=dev)
        self.planar = planar.numpy()[1, 0].copy()
        self.tilt = tilt.numpy()[1, 0].copy()
        self.clear = float(clear.numpy()[0, 0])
        self.resid = float(resid.numpy()[0, 0])
        self.alpha = float(turn.numpy()[0, 0][0])

    def render_state(self):
        x, y, yaw = (float(v) for v in self.planar)
        z, pitch, roll = (float(v) for v in self.tilt)
        R = placement.euler_zyx(yaw, pitch, roll)
        valid = self.clear >= 0.0 and self.resid < 1e-2  # belly clears AND settle feasible
        return SimpleNamespace(
            x=x, y=y, yaw=yaw, alpha=self.alpha, valid=valid,
            place={"z": z, "R": R, "pitch": pitch, "roll": roll},
        )


def run(shot=None, device="cpu"):
    import glfw
    from OpenGL import GL as gl

    hm = demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    drv = WarpDriver(hm, mu, init_pose=(0.0, 0.0, 0.0), device=device)

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
        time.sleep(max(0.0, DT - (time.perf_counter() - now)))

    glfw.terminate()
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", default=None, help="render ~45 auto-drive frames, save PNG, exit")
    ap.add_argument("--device", default="cpu", help="warp device: cpu or cuda")
    args = ap.parse_args()
    run(shot=args.shot, device=args.device)


if __name__ == "__main__":
    main()
