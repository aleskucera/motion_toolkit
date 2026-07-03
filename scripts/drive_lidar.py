"""Interactive 3D LiDAR viewer — drive an Ouster OSDome and filter moving people.

Models an Ouster OSDome (Rev7) mounted FRONT-facing: a 180° hemisphere of
128 uniform beams pointing along the robot's heading, with the datasheet's real
min/max range (0.5-45 m) and distance-dependent range precision. Real-time
glfw + legacy OpenGL (same stack as motion_toolkit's viewer, which works on
Wayland where open3d's GL viewer fails).

Two people walk around an OUTDOOR scene (open sky, no ceiling). Toggle the
accumulated map on and you build a persistent cloud as you drive (poses are exact
in sim, so no ICP needed); toggle the filter and free-space RAY-CARVING deletes
the moving people and their trails — including the tops of their heads, which
have only open sky behind them (a no-return beam is itself proof of free space).
Their orange ground-truth boxes keep moving so you can see what was removed.

Controls (polled — hold to move):
    W / S     forward / back        arrows    orbit camera
    A / D     turn left / right     - / =     zoom out / in
    Q / E     strafe                R         reset pose
    M         toggle accumulated map (vs live scan)
    F         toggle dynamic-obstacle filter        C   clear the map
    Esc       quit

Needs `glfw` + `PyOpenGL` and a display:  uv pip install glfw PyOpenGL

Run: python scripts/drive_lidar.py
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp
from terrain_toolkit import DeviceMapAccumulator
from terrain_toolkit import DynamicPointFilter
from terrain_toolkit.sim import GroundSpec
from terrain_toolkit.sim import make_osdome_lidar
from terrain_toolkit.sim import osdome_sensor_config
from terrain_toolkit.sim.ouster import OSDOME_MAX_RANGE_M

WIN_W, WIN_H = 1280, 800
GROUND = 60.0  # half-extent of the (open) ground plane — past the 45 m max range
SENSOR_Z = 0.6  # sensor height (m)
MOVE_SPEED = 3.0  # m/s
TURN_SPEED = 1.6  # rad/s
DROPOUT = 0.03  # stand-in for far / low-reflectivity return dropouts
OSDOME_COLS = 1024  # Ouster OSDome azimuth columns (drop to 512 if heavy)

MAP_VOXEL = 0.15  # accumulated-map voxel size (m)
MAP_RADIUS = 25.0  # keep accumulated points within this radius of the sensor (m)

# viridis-ish 3-stop ramp for coloring points by height.
_STOPS = np.array([[0.27, 0.0, 0.33], [0.13, 0.55, 0.55], [0.99, 0.91, 0.14]], np.float32)


def _device() -> wp.context.Device:
    wp.init()
    return wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")


def _static_world() -> tuple[np.ndarray, np.ndarray]:
    """Static obstacles (M, 3) lo/hi: an OUTDOOR scene under open sky.

    Scattered walls and pillars, not enclosed and with no ceiling — the real
    setting for this robot. Ray-carving removes the moving people here even though
    there's open sky behind their heads (a no-return beam is itself free-space
    evidence), so no artificial background is needed.
    """
    h = 3.0
    boxes = [
        ([6.0, -7.0, 0.0], [6.4, 7.0, h]),  # a building face ahead
        ([-8.0, -2.0, 0.0], [-2.0, -1.6, h]),  # a fence segment
        ([-4.0, 2.0, 0.0], [-3.4, 2.6, h]),  # pillars
        ([3.0, -3.0, 0.0], [3.6, -2.4, h]),
        ([1.0, 5.0, 0.0], [1.6, 5.6, h]),
    ]
    lo = np.array([b[0] for b in boxes], dtype=np.float32)
    hi = np.array([b[1] for b in boxes], dtype=np.float32)
    return lo, hi


def _people(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Two walking people as (2, 3) lo/hi boxes, on steady circular paths.

    Constant-speed orbits (no lingering at a bearing) so the visibility filter
    always has fresh background behind them to carve against.
    """
    centers = [
        (5.0 * math.cos(0.6 * t), 5.0 * math.sin(0.6 * t)),
        (3.5 * math.cos(-0.9 * t + 1.5), 3.5 * math.sin(-0.9 * t + 1.5)),
    ]
    half, height = 0.25, 1.8
    lo = np.array([[cx - half, cy - half, 0.0] for cx, cy in centers], dtype=np.float32)
    hi = np.array([[cx + half, cy + half, height] for cx, cy in centers], dtype=np.float32)
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
    def __init__(self, device: wp.context.Device | None = None):
        self.device = device if device is not None else _device()
        # One sensor description drives both the simulated lidar and the filter's
        # range image, so their FOV / resolution / range can't drift apart.
        sensor = osdome_sensor_config(columns=OSDOME_COLS)
        ground = GroundSpec(z=0.0, x_range=(-GROUND, GROUND), y_range=(-GROUND, GROUND))
        self.lidar = make_osdome_lidar(
            ground, sensor=sensor, facing="front", dropout=DROPOUT, device=self.device
        )
        # FOV / range come from the sensor; the range-image resolution is tuned
        # for carving (deliberately coarser than the sensor's native columns —
        # too-fine bins split a map point from its frontier and under-carve).
        self.filt = DynamicPointFilter.from_sensor(
            sensor, margin_m=0.3, margin_rel=0.03, az_bins=720, el_bins=180, device=self.device
        )
        # On-device rolling map: carve + add + crop + voxel-thin without leaving
        # the GPU. Only the displayed cloud is downloaded (for OpenGL).
        self.acc = DeviceMapAccumulator(MAP_VOXEL, MAP_RADIUS, device=self.device)
        self.static_lo, self.static_hi = _static_world()
        self._seed = 0
        self._held: set[str] = set()
        self.map_wp: wp.array | None = None  # accumulated map, resident on device
        self.last_scan = np.empty((0, 3), np.float32)
        self.people_lo, self.people_hi = _people(0.0)
        self.accumulate = False
        self.filter_on = True
        self.reset()

    def reset(self) -> None:
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.cam_off = math.pi  # camera azimuth relative to heading (behind)
        self.cam_el = 0.5
        self.cam_dist = 30.0

    def update(self, t: float) -> None:
        """Advance people, cast a scan, and (optionally) accumulate + filter."""
        self.people_lo, self.people_hi = _people(t)
        boxes_lo = np.vstack([self.static_lo, self.people_lo])
        boxes_hi = np.vstack([self.static_hi, self.people_hi])
        self._seed += 1
        origin = np.array([self.x, self.y, SENSOR_Z])
        # Device-native: keep the scan on the GPU (points/valid/frontier as wp.arrays).
        pts_wp, valid_wp, free_wp = self.lidar.scan(
            origin, self.yaw, boxes_lo, boxes_hi, seed=self._seed, return_device=True
        )

        if not self.accumulate:
            # Live view: download just this scan's returns for OpenGL.
            self.last_scan = pts_wp.numpy()[valid_wp.numpy().astype(bool)]
            return

        carve = None
        if self.filter_on and self.map_wp is not None and len(self.map_wp) > 0:
            # Ray-carve on-device: a mask of map points this scan's beams passed
            # through. A no-return beam frees space to max range, so heads against
            # open sky get carved too — with no host round trip.
            carve = self.filt.carve(self.map_wp, free_wp, origin)
        self.map_wp = self.acc.step(self.map_wp, carve, pts_wp, valid_wp, (self.x, self.y))

    def view_points(self) -> np.ndarray:
        if not self.accumulate:
            return self.last_scan
        # Download the map once, only for rendering.
        return self.map_wp.numpy() if self.map_wp is not None else np.empty((0, 3), np.float32)


def _render(gl, glu, st: State, fb: tuple[int, int]) -> None:
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
    glu.gluPerspective(55.0, w / max(h, 1), 0.1, 250.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    glu.gluLookAt(*eye, *tgt, 0.0, 0.0, 1.0)

    _draw_grid(gl)
    for lo, hi in zip(st.static_lo, st.static_hi):
        _draw_box(gl, lo, hi, (0.55, 0.57, 0.6))
    # People ground truth (orange) — stay visible even when filtered from the map.
    for lo, hi in zip(st.people_lo, st.people_hi):
        _draw_box(gl, lo, hi, (0.95, 0.55, 0.1))

    pts = st.view_points()
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


def _title(st: State) -> str:
    mode = "accumulated map" if st.accumulate else "live scan"
    filt = "filter ON" if st.filter_on else "filter OFF"
    return f"OSDome (max {OSDOME_MAX_RANGE_M:.0f} m) — {mode}, {filt} [{len(st.view_points())} pts]"


def _handle_input(glfw, win, st: State, dt: float) -> bool:
    """Apply held-key motion; return True if a toggle changed the window title."""

    def down(key) -> bool:
        return glfw.get_key(win, key) == glfw.PRESS

    fwd = MOVE_SPEED * dt * ((1 if down(glfw.KEY_W) else 0) - (1 if down(glfw.KEY_S) else 0))
    strafe = MOVE_SPEED * dt * ((1 if down(glfw.KEY_E) else 0) - (1 if down(glfw.KEY_Q) else 0))
    st.yaw += TURN_SPEED * dt * ((1 if down(glfw.KEY_A) else 0) - (1 if down(glfw.KEY_D) else 0))
    c, s = math.cos(st.yaw), math.sin(st.yaw)
    st.x += fwd * c - strafe * s
    st.y += fwd * s + strafe * c
    m = GROUND - 0.5
    st.x = float(np.clip(st.x, -m, m))
    st.y = float(np.clip(st.y, -m, m))

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

    # Rising-edge toggles (M/F/C).
    toggled = False
    for key, name in ((glfw.KEY_M, "m"), (glfw.KEY_F, "f"), (glfw.KEY_C, "c")):
        if down(key):
            if name not in st._held:
                if name == "m":
                    st.accumulate = not st.accumulate
                elif name == "f":
                    st.filter_on = not st.filter_on
                else:
                    st.map_wp = None
                toggled = True
            st._held.add(name)
        else:
            st._held.discard(name)
    return toggled


def main() -> None:
    try:
        import glfw
        from OpenGL import GL as gl
        from OpenGL import GLU as glu
    except ImportError as exc:
        raise SystemExit(f"needs glfw + PyOpenGL: `uv pip install glfw PyOpenGL` ({exc})")

    if not glfw.init():
        raise SystemExit("glfw.init() failed (no display / missing libglfw?)")
    st = State()
    win = glfw.create_window(WIN_W, WIN_H, _title(st), None, None)
    if not win:
        glfw.terminate()
        raise SystemExit("failed to create a window")
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    _init_gl()

    prev = glfw.get_time()
    while not glfw.window_should_close(win):
        glfw.poll_events()
        now = glfw.get_time()
        dt = min(now - prev, 0.05)
        prev = now
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        _handle_input(glfw, win, st, dt)
        st.update(now)
        glfw.set_window_title(win, _title(st))
        _render(gl, glu, st, glfw.get_framebuffer_size(win))
        glfw.swap_buffers(win)

    glfw.terminate()


if __name__ == "__main__":
    main()
