"""Heightmap terrain: analytic-scene rasterization + differentiable sampling.

A heightmap is a regular grid H[ny, nx] of heights with a world origin (x0, y0)
and uniform cell size. Values sit at CELL CENTERS: H[i,j] is the height at world
(x0 + (j+0.5)*cell, y0 + (i+0.5)*cell) -- matching terrain_toolkit's raster, so a
device grid handed over from perception needs no half-cell shift. World->grid:
col = (x-x0)/cell - 0.5, row = (y-y0)/cell - 0.5.

For Phase 0 this is a plain numpy reference (bilinear value + central-difference
normal). The Warp port (Phase 2/5) mirrors this exactly so the numpy version
stays usable as a finite-difference gradient oracle.
"""

import numpy as np

from . import data


class Heightmap:
    def __init__(self, heights, origin, cell):
        self.H = np.asarray(heights, dtype=np.float64)  # [ny, nx]
        self.ny, self.nx = self.H.shape
        self.x0, self.y0 = float(origin[0]), float(origin[1])
        self.cell = float(cell)

    def _grid_coords(self, x, y):
        fx = (np.asarray(x, dtype=np.float64) - self.x0) / self.cell - 0.5
        fy = (np.asarray(y, dtype=np.float64) - self.y0) / self.cell - 0.5
        ix = np.clip(np.floor(fx).astype(int), 0, self.nx - 2)
        iy = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        tx = np.clip(fx - ix, 0.0, 1.0)
        ty = np.clip(fy - iy, 0.0, 1.0)
        return ix, iy, tx, ty

    def sample(self, x, y):
        """Bilinear height at world (x, y). Scalar or array."""
        ix, iy, tx, ty = self._grid_coords(x, y)
        H = self.H
        h00 = H[iy, ix]
        h10 = H[iy, ix + 1]
        h01 = H[iy + 1, ix]
        h11 = H[iy + 1, ix + 1]
        h = (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + (1 - tx) * ty * h01 + tx * ty * h11
        return h

    def normal(self, x, y, eps=None):
        """Unit terrain normal at (x, y) from central differences of the height."""
        e = self.cell if eps is None else eps
        dhdx = (self.sample(x + e, y) - self.sample(x - e, y)) / (2 * e)
        dhdy = (self.sample(x, y + e) - self.sample(x, y - e)) / (2 * e)
        n = np.stack([-dhdx, -dhdy, np.ones_like(dhdx)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _shift(A, di, dj):
    """A[i+di, j+dj] with edge clamping (no wrap)."""
    ny, nx = A.shape
    i = np.clip(np.arange(ny)[:, None] + di, 0, ny - 1)
    j = np.clip(np.arange(nx)[None, :] + dj, 0, nx - 1)
    return A[i, j]


def wheel_envelope(hm, R):
    """Sphere-wheel upper envelope: the height a radius-R wheel hub minus R can
    reach resting on the terrain (morphological dilation by a spherical cap).

        H_eff(p) = max_{|q-p| <= R} [ h(q) + sqrt(R^2 - |q-p|^2) ] - R

    On flat ground H_eff = h; at a sharp step the wheel rests on the edge, so the
    step is smoothed into a ~R-wide climb. Used as the placement surface: the
    wheel constraint becomes simply hub_z - H_eff(hub_xy) = R, no slope term.
    """
    H = hm.H
    rad = int(np.ceil(R / hm.cell))
    Heff = np.full_like(H, -np.inf)
    for di in range(-rad, rad + 1):
        for dj in range(-rad, rad + 1):
            d = np.hypot(di, dj) * hm.cell
            if d > R:
                continue
            cap = np.sqrt(R * R - d * d) - R
            Heff = np.maximum(Heff, _shift(H, di, dj) + cap)
    return Heightmap(Heff, (hm.x0, hm.y0), hm.cell)


def _grid(xlim, ylim, cell):
    x0, x1 = xlim
    y0, y1 = ylim
    nx = int(round((x1 - x0) / cell)) + 1
    ny = int(round((y1 - y0) / cell)) + 1
    xs = x0 + (np.arange(nx) + 0.5) * cell  # cell centers
    ys = y0 + (np.arange(ny) + 0.5) * cell
    XX, YY = np.meshgrid(xs, ys)  # [ny, nx]
    return XX, YY


def flat(xlim=(-2.0, 6.0), ylim=(-3.0, 3.0), cell=0.05):
    XX, _ = _grid(xlim, ylim, cell)
    return Heightmap(np.zeros_like(XX), (xlim[0], ylim[0]), cell)


def box_scene(xlim=(-2.0, 6.0), ylim=(-3.0, 3.0), cell=0.05):
    """Flat ground with the measured replay box raised flush on it."""
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    cx, cy, _ = data.BOX_CENTER
    hx, hy, hz = data.BOX_HALF_EXTENTS
    top = 2 * hz  # box rests on the ground, top at z = 2*hz
    inside = (np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)
    H[inside] = top
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def ramp_scene(angle_deg=11.3, length=5.0, xlim=(-2.0, 8.0), ylim=(-3.0, 3.0), cell=0.05):
    """Flat ground that rises into a constant-slope ramp starting at x=1.0."""
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    slope = np.tan(np.deg2rad(angle_deg))
    x_start = 1.0
    rise = np.clip(XX - x_start, 0.0, length) * slope
    H = rise
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def demo_terrain(cell=0.06):
    """Flat ground with a curb, a wall, a ramp+plateau, and a smooth hill.

    The interactive-viewer scene (drive into the wall and the robot turns red on
    the infeasible settle). Lives here with the other scene builders.
    """
    xlim, ylim = (-3.0, 10.0), (-4.0, 4.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    H[(np.abs(XX - 1.3) <= 0.35) & (np.abs(YY) <= 1.0)] = 0.12  # curb
    # wall: drive straight in -> robot turns red (infeasible settle)
    H[(np.abs(XX - 2.2) <= 0.15) & (np.abs(YY) <= 1.0)] = 1.0
    H += np.clip(XX - 3.0, 0.0, 3.0) / 3.0 * 0.5  # ramp+plateau
    H += 0.6 * np.exp(-((XX - 8.0) ** 2 + (YY + 2.0) ** 2) / (2 * 1.2**2))  # hill
    return Heightmap(H, (xlim[0], ylim[0]), cell)


if __name__ == "__main__":
    # Phase-0 smoke test: height under a wheel matches the analytic scene.
    hm = box_scene()
    cx, cy, _ = data.BOX_CENTER
    assert abs(hm.sample(cx, cy) - 2 * data.BOX_HALF_EXTENTS[2]) < 1e-9, "box top wrong"
    assert abs(hm.sample(-1.0, 0.0) - 0.0) < 1e-9, "ground wrong"
    n = hm.normal(-1.0, 0.0)
    assert np.allclose(n, [0, 0, 1], atol=1e-6), f"flat normal wrong: {n}"
    print(
        "heightmap smoke test OK:",
        f"box top={hm.sample(cx, cy):.3f}, ground={hm.sample(-1.0, 0.0):.3f}, normal={n}",
    )
