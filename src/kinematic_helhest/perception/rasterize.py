"""Point cloud -> heightmap rasterizer: the real-sensor front-end that turns a 3D lidar point cloud
into the [ny, nx] elevation grid the planner consumes (the role synthetic `lidar_scan` plays today).

This is THE sim-to-real seam -- the one place a frame transpose or a handedness flip silently corrupts
everything downstream -- so it is guarded by an asymmetric round-trip test
(tests/perception/test_rasterize.py).

Convention (matches lidar_scan / MultiScanMap / GridParams): REP-103 x-forward, y-left; the grid is
H[ny, nx] = H[row, col] with row -> y and col -> x; cell (r, c)'s CENTER is world
(x0 + c*cell, y0 + r*cell). So (x0, y0) is the center of cell (0, 0), and the cell index of world
point (x, y) is (round((y - y0) / cell), round((x - x0) / cell)).
"""

import numpy as np


def rasterize(points, x0, y0, cell, ny, nx):
    """[N, 3] (x, y, z) world points -> (H[ny, nx], known[ny, nx]). H = the MAX z per cell (the top
    surface -- what clearance / obstacle height care about); cells with no point are H = 0, known = False.
    Points outside the grid are dropped."""
    p = np.asarray(points, np.float64)
    ci = np.round((p[:, 0] - x0) / cell).astype(np.int64)  # col <- x
    ri = np.round((p[:, 1] - y0) / cell).astype(np.int64)  # row <- y
    inb = (ri >= 0) & (ri < ny) & (ci >= 0) & (ci < nx)
    flat = ri[inb] * nx + ci[inb]
    H = np.full(ny * nx, -np.inf)
    np.maximum.at(H, flat, p[inb, 2])  # max-z per cell
    known = np.isfinite(H)
    H = np.where(known, H, 0.0).reshape(ny, nx).astype(np.float32)
    return H, known.reshape(ny, nx)


def heightmap_to_points(H, x0, y0, cell):
    """Inverse for testing: one point at each cell CENTER. rasterize() of these recovers H exactly --
    the round-trip that pins the frame convention end to end."""
    ny, nx = H.shape
    rr, cc = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return np.stack([(x0 + cc * cell).ravel(), (y0 + rr * cell).ravel(), H.ravel()], 1)
