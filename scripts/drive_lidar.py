"""Interactive 3D LiDAR viewer — drive an Ouster OSDome around a room.

Models an Ouster OSDome (Rev7) mounted FRONT-facing: a 180° hemisphere of
128 uniform beams pointing along the robot's heading, with the datasheet's real
min/max range (0.5–45 m) and distance-dependent range precision. Real-time
glfw + legacy OpenGL (same stack as motion_toolkit's viewer, which works on
Wayland where open3d's GL viewer fails). Each frame polls the keyboard, moves the
sensor, re-casts the Warp ray-cast LiDAR against a world of box obstacles, and
draws the point cloud in 3D — so you see the returns, occlusion shadows, and the
max-range cutoff update live as you drive. (Beam angles are a nominal uniform
hemisphere; drop in the sensor's metadata JSON for the exact calibrated table.)

Controls (polled — hold to move):
    W / S     forward / back        arrows    orbit camera
    A / D     turn left / right     - / =     zoom out / in
    Q / E     strafe                R         reset pose
    Esc       quit

Needs `glfw` + `PyOpenGL` and a display:  uv pip install glfw PyOpenGL
(the system libglfw is already present on most Linux desktops).

Run: python scripts/drive_lidar.py
"""

from __future__ import annotations

import math

import numpy as np
from terrain_toolkit.sim import GroundSpec
from terrain_toolkit.sim import make_osdome_lidar
from terrain_toolkit.sim.ouster import OSDOME_MAX_RANGE_M

WIN_W, WIN_H = 1280, 800
ROOM = 10.0  # half-extent of the room walls (m)
GROUND = 60.0  # half-extent of the (open) ground plane — past the 45 m max range
SENSOR_Z = 0.6  # sensor height (m)
MOVE_SPEED = 3.0  # m/s
TURN_SPEED = 1.6  # rad/s
DROPOUT = 0.03  # stand-in for far / low-reflectivity return dropouts

# Ouster OSDome, FRONT-facing (128-channel hemisphere along +x). Real modes are
# 1024/2048 columns; drop to 512 if the interactive loop feels heavy.
OSDOME_COLS = 1024

# viridis-ish 3-stop ramp for coloring points by height.
_STOPS = np.array([[0.27, 0.0, 0.33], [0.13, 0.55, 0.55], [0.99, 0.91, 0.14]], np.float32)


def _world() -> tuple[np.ndarray, np.ndarray]:
    """Box obstacles as (M, 3) lo/hi corners: three walls (front open) + props.

    The +x wall is intentionally omitted, so facing forward you look out the open
    side over empty ground — where returns stop at the sensor's max range.
    """
    t, h = 0.2, 2.5
    boxes = [
        ([-ROOM, -ROOM, 0.0], [ROOM, -ROOM + t, h]),  # back (y = -ROOM)
        ([-ROOM, ROOM - t, 0.0], [ROOM, ROOM, h]),  # front-left (y = +ROOM)
        ([-ROOM, -ROOM, 0.0], [-ROOM + t, ROOM, h]),  # rear (x = -ROOM)
        # (+x wall dropped — the open side)
        ([-4.0, 2.0, 0.0], [-3.4, 2.6, h]),
        ([3.0, -3.0, 0.0], [3.6, -2.4, h]),
        ([1.0, 4.5, 0.0], [1.6, 5.1, h]),
        ([-1.5, -1.5, 0.0], [-1.1, -1.1, 1.8]),  # person
        ([5.0, 1.0, 0.0], [5.4, 1.4, 1.8]),  # person
    ]
    lo = np.array([b[0] for b in boxes], dtype=np.float32)
    hi = np.array([b[1] for b in boxes], dtype=np.float32)
    return lo, hi


def _height_colors(z: np.ndarray, zmin: float = 0.0, zmax: float = 2.5) -> np.ndarray:
    t = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
    low = t < 0.5
    a = np.where(low, t / 0.5, (t - 0.5) / 0.5)[:, None]
    c0 = np.where(low[:, None], _STOPS[0], _STOPS[1])
    c1 = np.where(low[:, None], _STOPS[1], _STOPS[2])
    return (c0 * (1.0 - a) + c1 * a).astype(np.float32)


# --------------------------------------------------------------------------- #
# OpenGL drawing (legacy fixed-function)
# --------------------------------------------------------------------------- #
def _init_gl() -> None:
    from OpenGL import GL as gl

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glPointSize(3.0)
    gl.glClearColor(0.6, 0.72, 0.85, 1.0)


def _draw_grid(gl, step: float = 3.0) -> None:
    gl.glColor3f(0.45, 0.5, 0.55)
    gl.glBegin(gl.GL_LINES)
    n = int(GROUND // step)
    for i in range(-n, n + 1):
        gl.glVertex3f(-GROUND, i * step, 0.0)
        gl.glVertex3f(GROUND, i * step, 0.0)
        gl.glVertex3f(i * step, -GROUND, 0.0)
        gl.glVertex3f(i * step, GROUND, 0.0)
    gl.glEnd()


_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]


def _draw_box(gl, lo, hi, color) -> None:
    c = [
        (lo[0], lo[1], lo[2]),
        (hi[0], lo[1], lo[2]),
        (hi[0], hi[1], lo[2]),
        (lo[0], hi[1], lo[2]),
        (lo[0], lo[1], hi[2]),
        (hi[0], lo[1], hi[2]),
        (hi[0], hi[1], hi[2]),
        (lo[0], hi[1], hi[2]),
    ]
    gl.glColor3f(*color)
    gl.glBegin(gl.GL_QUADS)
    for f in _FACES:
        for idx in f:
            gl.glVertex3f(*c[idx])
    gl.glEnd()


def _draw_points(gl, pts: np.ndarray, cols: np.ndarray) -> None:
    if len(pts) == 0:
        return
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glEnableClientState(gl.GL_COLOR_ARRAY)
    gl.glVertexPointer(3, gl.GL_FLOAT, 0, np.ascontiguousarray(pts, np.float32))
    gl.glColorPointer(3, gl.GL_FLOAT, 0, np.ascontiguousarray(cols, np.float32))
    gl.glDrawArrays(gl.GL_POINTS, 0, len(pts))
    gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
    gl.glDisableClientState(gl.GL_COLOR_ARRAY)


class State:
    def __init__(self):
        self.lidar = make_osdome_lidar(
            GroundSpec(z=0.0, x_range=(-GROUND, GROUND), y_range=(-GROUND, GROUND)),
            cols=OSDOME_COLS,
            facing="front",
            dropout=DROPOUT,
        )
        self.boxes_lo, self.boxes_hi = _world()
        self._seed = 0
        self.reset()

    def reset(self) -> None:
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.cam_off = math.pi  # camera azimuth relative to heading (behind)
        self.cam_el = 0.5
        self.cam_dist = 30.0  # start zoomed out (zoom with -/= to see the 45 m cutoff)

    def scan(self) -> np.ndarray:
        self._seed += 1
        origin = np.array([self.x, self.y, SENSOR_Z])
        return self.lidar.scan(origin, self.yaw, self.boxes_lo, self.boxes_hi, seed=self._seed)


def _render(gl, glu, st: State, pts: np.ndarray, fb: tuple[int, int]) -> None:
    w, h = fb
    az = st.yaw + st.cam_off
    tgt = np.array([st.x, st.y, 1.0])
    eye = tgt + st.cam_dist * np.array(
        [
            math.cos(st.cam_el) * math.cos(az),
            math.cos(st.cam_el) * math.sin(az),
            math.sin(st.cam_el),
        ]
    )

    gl.glViewport(0, 0, w, h)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(55.0, w / max(h, 1), 0.1, 200.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    glu.gluLookAt(*eye, *tgt, 0.0, 0.0, 1.0)

    _draw_grid(gl)
    for lo, hi in zip(st.boxes_lo, st.boxes_hi):
        _draw_box(gl, lo, hi, (0.55, 0.57, 0.6))
    _draw_points(gl, pts, _height_colors(pts[:, 2]) if len(pts) else pts)

    # Sensor body + heading line.
    _draw_box(
        gl,
        (st.x - 0.15, st.y - 0.15, SENSOR_Z - 0.15),
        (st.x + 0.15, st.y + 0.15, SENSOR_Z + 0.15),
        (0.9, 0.1, 0.1),
    )
    gl.glColor3f(0.9, 0.1, 0.1)
    gl.glLineWidth(3.0)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(st.x, st.y, SENSOR_Z)
    gl.glVertex3f(st.x + math.cos(st.yaw), st.y + math.sin(st.yaw), SENSOR_Z)
    gl.glEnd()


def _handle_input(glfw, win, st: State, dt: float) -> None:
    def down(key) -> bool:
        return glfw.get_key(win, key) == glfw.PRESS

    fwd = MOVE_SPEED * dt * ((1 if down(glfw.KEY_W) else 0) - (1 if down(glfw.KEY_S) else 0))
    strafe = MOVE_SPEED * dt * ((1 if down(glfw.KEY_E) else 0) - (1 if down(glfw.KEY_Q) else 0))
    st.yaw += TURN_SPEED * dt * ((1 if down(glfw.KEY_A) else 0) - (1 if down(glfw.KEY_D) else 0))
    c, s = math.cos(st.yaw), math.sin(st.yaw)
    st.x += fwd * c - strafe * s
    st.y += fwd * s + strafe * c
    m = GROUND - 0.5  # roam the whole ground, including out the open side
    st.x = float(np.clip(st.x, -m, m))
    st.y = float(np.clip(st.y, -m, m))

    # Camera orbit + zoom.
    st.cam_off += (
        1.5 * dt * ((1 if down(glfw.KEY_LEFT) else 0) - (1 if down(glfw.KEY_RIGHT) else 0))
    )
    st.cam_el += 1.0 * dt * ((1 if down(glfw.KEY_UP) else 0) - (1 if down(glfw.KEY_DOWN) else 0))
    st.cam_el = float(np.clip(st.cam_el, 0.05, 1.5))
    st.cam_dist += (
        25.0 * dt * ((1 if down(glfw.KEY_MINUS) else 0) - (1 if down(glfw.KEY_EQUAL) else 0))
    )
    st.cam_dist = float(np.clip(st.cam_dist, 3.0, 120.0))

    if down(glfw.KEY_R):
        st.reset()


def main() -> None:
    try:
        import glfw
        from OpenGL import GL as gl
        from OpenGL import GLU as glu
    except ImportError as exc:
        raise SystemExit(f"needs glfw + PyOpenGL: `uv pip install glfw PyOpenGL` ({exc})")

    if not glfw.init():
        raise SystemExit("glfw.init() failed (no display / missing libglfw?)")
    title = f"terrain_toolkit — Ouster OSDome (front-facing, max {OSDOME_MAX_RANGE_M:.0f} m)"
    win = glfw.create_window(WIN_W, WIN_H, title, None, None)
    if not win:
        glfw.terminate()
        raise SystemExit("failed to create a window")
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    _init_gl()

    st = State()
    prev = glfw.get_time()
    while not glfw.window_should_close(win):
        glfw.poll_events()
        now = glfw.get_time()
        dt = min(now - prev, 0.05)
        prev = now
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        _handle_input(glfw, win, st, dt)
        _render(gl, glu, st, st.scan(), glfw.get_framebuffer_size(win))
        glfw.swap_buffers(win)

    glfw.terminate()


if __name__ == "__main__":
    main()
