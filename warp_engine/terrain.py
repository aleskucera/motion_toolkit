"""Device-side heightmap: bilinear height + central-difference normal.

Mirrors the numpy `heightmap.Heightmap` exactly (same clamping, same stencil) so
the numpy version stays usable as a finite-difference oracle. The wheel-envelope
surface is precomputed on the host (`heightmap.wheel_envelope`) and handed in as
the height grid; this module just samples it.

The height grid `H` is passed to kernels as a plain `wp.array` (not a struct
member) so the tape accumulates its adjoint cleanly for the Phase-5 d/dh gradient.
The grid's non-differentiated metadata (origin, cell, size) rides in a small
`GridMeta` struct alongside it.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.struct
class GridMeta:
    """Non-differentiated grid metadata, passed with the plain `H` array."""

    x0: wp.float32
    y0: wp.float32
    cell: wp.float32
    nx: wp.int32
    ny: wp.int32


@dataclass
class Terrain:
    """Host-side bag: a differentiated grid `H` (plain array) + its GridMeta."""

    H: wp.array  # array2d(float32) [ny, nx]
    g: GridMeta


def to_terrain(hm, device="cpu", requires_grad=False) -> Terrain:
    """Build a device Terrain from a numpy `heightmap.Heightmap`."""
    H = wp.array(
        np.ascontiguousarray(hm.H, dtype=np.float32),
        dtype=wp.float32, device=device, requires_grad=requires_grad,
    )
    g = GridMeta()
    g.x0 = float(hm.x0)
    g.y0 = float(hm.y0)
    g.cell = float(hm.cell)
    g.nx = int(hm.nx)
    g.ny = int(hm.ny)
    return Terrain(H, g)


@wp.func
def sample_height(H: wp.array2d(dtype=wp.float32), g: GridMeta, x: float, y: float):
    fx = (x - g.x0) / g.cell
    fy = (y - g.y0) / g.cell
    ix = wp.clamp(int(wp.floor(fx)), 0, g.nx - 2)
    iy = wp.clamp(int(wp.floor(fy)), 0, g.ny - 2)
    tx = wp.clamp(fx - float(ix), 0.0, 1.0)
    ty = wp.clamp(fy - float(iy), 0.0, 1.0)
    h00 = H[iy, ix]
    h10 = H[iy, ix + 1]
    h01 = H[iy + 1, ix]
    h11 = H[iy + 1, ix + 1]
    return ((1.0 - tx) * (1.0 - ty) * h00 + tx * (1.0 - ty) * h10
            + (1.0 - tx) * ty * h01 + tx * ty * h11)


@wp.func
def sample_normal(H: wp.array2d(dtype=wp.float32), g: GridMeta, x: float, y: float):
    e = g.cell
    dhdx = (sample_height(H, g, x + e, y) - sample_height(H, g, x - e, y)) / (2.0 * e)
    dhdy = (sample_height(H, g, x, y + e) - sample_height(H, g, x, y - e)) / (2.0 * e)
    return wp.normalize(wp.vec3(-dhdx, -dhdy, 1.0))


@wp.kernel
def _probe(H: wp.array2d(dtype=wp.float32), g: GridMeta,
           xs: wp.array(dtype=wp.float32), ys: wp.array(dtype=wp.float32),
           out_h: wp.array(dtype=wp.float32), out_n: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out_h[i] = sample_height(H, g, xs[i], ys[i])
    out_n[i] = sample_normal(H, g, xs[i], ys[i])


def _selftest():
    """Compare device sample/normal to the numpy oracle on random points."""
    from .. import heightmap as hmmod

    wp.init()
    rng = np.random.default_rng(0)
    for scene in (hmmod.flat(), hmmod.box_scene(), hmmod.ramp_scene()):
        env = hmmod.wheel_envelope(scene, 0.35)  # the real placement surface
        xs = rng.uniform(scene.x0 + 0.2, scene.x0 + (scene.nx - 2) * scene.cell, 200)
        ys = rng.uniform(scene.y0 + 0.2, scene.y0 + (scene.ny - 2) * scene.cell, 200)
        t = to_terrain(env, "cpu")
        wx = wp.array(xs.astype(np.float32), dtype=wp.float32, device="cpu")
        wy = wp.array(ys.astype(np.float32), dtype=wp.float32, device="cpu")
        oh = wp.zeros(len(xs), dtype=wp.float32, device="cpu")
        on = wp.zeros(len(xs), dtype=wp.vec3, device="cpu")
        wp.launch(_probe, len(xs), inputs=[t.H, t.g, wx, wy], outputs=[oh, on], device="cpu")
        h_ref = env.sample(xs, ys)
        n_ref = env.normal(xs, ys)
        dh = np.abs(oh.numpy() - h_ref).max()
        dn = np.abs(on.numpy() - n_ref).max()
        print(f"  {h_ref.size} pts  max|dh|={dh:.2e}  max|dn|={dn:.2e}")
        assert dh < 1e-4 and dn < 1e-4, (dh, dn)
    print("terrain device-vs-numpy OK")


if __name__ == "__main__":
    _selftest()
