"""Shared real-time rendering toolkit (glfw + legacy OpenGL).

Engine-agnostic: every viewer (the Warp drivers and the numpy reference driver)
builds its scene/robot geometry here and calls `_render` each frame. `_render`
duck-types the state object — it only reads `st.x`, `st.y`, `st.yaw`, `st.valid`
and `st.place` ({"z", "R", ...}) — so it does not care whether the pose came from
the Warp engine or the numpy oracle. Uses legacy OpenGL directly (no GLEW), which
works where open3d's GL viewer fails on Wayland.
"""
import numpy as np

from ..dynamics import DT  # re-exported here for back-compat (viz used to own DT)
from ..model import WHEEL_POS
from ..model import WHEEL_RADIUS

WHEEL_WIDTH = 0.10
CHASSIS_BOXES = [(-0.13, 0.0, 0.0, 0.48, 0.56, 0.20),
                 (-0.61, 0.0, 0.0, 0.48, 0.24, 0.20)]
BASE_SPEED = 3.0
TURN_SPEED = 2.0
WIN_W, WIN_H = 1280, 800


# --------------------------------------------------------------------------- #
# Geometry -> triangle soup (verts, normals, colors), float32
# --------------------------------------------------------------------------- #
def _box_tris(cx, cy, cz, sx, sy, sz, color):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    c = np.array([cx, cy, cz])
    # 8 corners
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    P = c + s * np.array([hx, hy, hz])
    # faces as (4 corner indices, normal)
    faces = [((0, 1, 3, 2), (-1, 0, 0)), ((4, 6, 7, 5), (1, 0, 0)),
             ((0, 4, 5, 1), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)),
             ((0, 2, 6, 4), (0, 0, -1)), ((1, 5, 7, 3), (0, 0, 1))]
    V, N = [], []
    for (a, b, d, e), n in faces:
        for tri in ((a, b, d), (a, d, e)):
            V += [P[tri[0]], P[tri[1]], P[tri[2]]]
            N += [n, n, n]
    V = np.array(V, np.float32)
    return V, np.array(N, np.float32), np.tile(color, (len(V), 1)).astype(np.float32)


def _cyl_tris(center, radius, half_w, color, segs=24):
    """Cylinder with axis along body Y (wheel spins about Y)."""
    cx, cy, cz = center
    ang = np.linspace(0, 2 * np.pi, segs + 1)
    ring = np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], 1)  # XZ circle
    V, N = [], []
    for i in range(segs):
        p0, p1 = ring[i], ring[i + 1]
        o0 = np.array([cx + radius * p0[0], cy - half_w, cz + radius * p0[2]])
        o1 = np.array([cx + radius * p1[0], cy - half_w, cz + radius * p1[2]])
        i0 = o0 + [0, 2 * half_w, 0]
        i1 = o1 + [0, 2 * half_w, 0]
        n0, n1 = np.array([p0[0], 0, p0[2]]), np.array([p1[0], 0, p1[2]])
        V += [o0, o1, i1, o0, i1, i0]
        N += [n0, n1, n1, n0, n1, n0]
        # caps
        for cyl_y, ny in ((cy - half_w, -1), (cy + half_w, 1)):
            ctr = np.array([cx, cyl_y, cz])
            a = np.array([cx + radius * p0[0], cyl_y, cz + radius * p0[2]])
            b = np.array([cx + radius * p1[0], cyl_y, cz + radius * p1[2]])
            V += [ctr, a, b] if ny > 0 else [ctr, b, a]
            N += [[0, ny, 0]] * 3
    V = np.array(V, np.float32)
    return V, np.array(N, np.float32), np.tile(color, (len(V), 1)).astype(np.float32)


def build_robot():
    chassis_c = (0.55, 0.57, 0.62)
    wheel_c = (0.12, 0.12, 0.14)
    parts = [_box_tris(*b, chassis_c) for b in CHASSIS_BOXES]
    parts += [_cyl_tris(wp, WHEEL_RADIUS, WHEEL_WIDTH / 2, wheel_c) for wp in WHEEL_POS]
    V = np.concatenate([p[0] for p in parts])
    N = np.concatenate([p[1] for p in parts])
    C = np.concatenate([p[2] for p in parts])
    red = np.tile([0.85, 0.12, 0.12], (len(V), 1)).astype(np.float32)
    return V, N, C, red


def build_terrain(hm):
    from matplotlib import cm
    ny, nx = hm.H.shape
    xs = hm.x0 + (np.arange(nx) + 0.5) * hm.cell  # cell centers
    ys = hm.y0 + (np.arange(ny) + 0.5) * hm.cell
    XX, YY = np.meshgrid(xs, ys)
    V = np.stack([XX, YY, hm.H], -1).reshape(-1, 3).astype(np.float32)
    gy, gx = np.gradient(hm.H, hm.cell)
    Nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1).reshape(-1, 3)
    Nrm /= np.linalg.norm(Nrm, axis=1, keepdims=True)
    hn = (hm.H - hm.H.min()) / (np.ptp(hm.H) + 1e-9)
    C = cm.terrain(0.25 + 0.7 * hn.ravel())[:, :3].astype(np.float32)
    ii, jj = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    v0 = (ii * nx + jj).ravel()
    v1, v2, v3 = v0 + 1, v0 + nx, v0 + nx + 1
    idx = np.concatenate([np.stack([v0, v2, v1], 1), np.stack([v1, v2, v3], 1)], 0)
    return V, Nrm.astype(np.float32), C, idx.astype(np.uint32).ravel()


# --------------------------------------------------------------------------- #
# OpenGL
# --------------------------------------------------------------------------- #
def _init_gl():
    from OpenGL import GL as gl
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_LIGHT0)
    gl.glEnable(gl.GL_COLOR_MATERIAL)
    gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
    gl.glEnable(gl.GL_NORMALIZE)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0.4, 0.5, 1.0, 0.0])
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.35, 0.35, 0.35, 1.0])
    gl.glClearColor(0.6, 0.72, 0.85, 1.0)


def _draw(V, N, C, idx=None):
    from OpenGL import GL as gl
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
    gl.glEnableClientState(gl.GL_COLOR_ARRAY)
    gl.glVertexPointer(3, gl.GL_FLOAT, 0, V)
    gl.glNormalPointer(gl.GL_FLOAT, 0, N)
    gl.glColorPointer(3, gl.GL_FLOAT, 0, C)
    if idx is None:
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(V))
    else:
        gl.glDrawElements(gl.GL_TRIANGLES, len(idx), gl.GL_UNSIGNED_INT, idx)
    gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
    gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
    gl.glDisableClientState(gl.GL_COLOR_ARRAY)


def _render(st, cam, terrain, robot, trail_pts):
    from OpenGL import GL as gl
    from OpenGL import GLU as glu
    az, el, dist = cam
    tgt = np.array([st.x, st.y, st.place["z"]])
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = tgt + dist * d

    gl.glViewport(0, 0, WIN_W, WIN_H)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glMatrixMode(gl.GL_PROJECTION); gl.glLoadIdentity()
    glu.gluPerspective(50.0, WIN_W / WIN_H, 0.1, 100.0)
    gl.glMatrixMode(gl.GL_MODELVIEW); gl.glLoadIdentity()
    glu.gluLookAt(*eye, *tgt, 0, 0, 1)

    _draw(*terrain[:3], terrain[3])  # ground

    V, N, C, red = robot
    R4 = np.eye(4, dtype=np.float32); R4[:3, :3] = st.place["R"]
    gl.glPushMatrix()
    gl.glTranslatef(st.x, st.y, st.place["z"])
    gl.glMultMatrixf(np.ascontiguousarray(R4.T))
    _draw(V, N, C if st.valid else red)
    gl.glPopMatrix()

    if len(trail_pts) > 1:
        gl.glDisable(gl.GL_LIGHTING)
        gl.glColor3f(1.0, 0.35, 0.0); gl.glLineWidth(2.0)
        gl.glBegin(gl.GL_LINE_STRIP)
        for p in trail_pts:
            gl.glVertex3f(*p)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)


def _commands(get_key):
    import glfw
    left = right = 0.0
    if get_key(glfw.KEY_I) == glfw.PRESS:
        left += BASE_SPEED; right += BASE_SPEED
    if get_key(glfw.KEY_K) == glfw.PRESS:
        left -= BASE_SPEED; right -= BASE_SPEED
    if get_key(glfw.KEY_J) == glfw.PRESS:
        left -= TURN_SPEED; right += TURN_SPEED
    if get_key(glfw.KEY_L) == glfw.PRESS:
        left += TURN_SPEED; right -= TURN_SPEED
    return np.array([left, right, (left + right) / 2.0], np.float64)
