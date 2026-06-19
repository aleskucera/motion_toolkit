"""Device terrain sampling (engine) vs the numpy oracle (heightmap).

Run:  python -m tests.engine.terrain
"""
import numpy as np
import warp as wp

from kinematic_helhest import heightmap as hmmod
from kinematic_helhest.engine import Grid
from kinematic_helhest.engine import GridParams
from kinematic_helhest.engine import sample_field
from kinematic_helhest.engine import sample_normal


@wp.kernel
def _probe(elevation: wp.array2d(dtype=wp.float32), g: Grid,
           xs: wp.array(dtype=wp.float32), ys: wp.array(dtype=wp.float32),
           out_h: wp.array(dtype=wp.float32), out_n: wp.array(dtype=wp.vec3)):
    """Verification-only: launch the engine samplers from host code."""
    i = wp.tid()
    out_h[i] = sample_field(elevation, g, xs[i], ys[i])
    out_n[i] = sample_normal(elevation, g, xs[i], ys[i])


def selftest():
    """Compare device sample/normal to the numpy oracle on random points."""
    wp.init()
    rng = np.random.default_rng(0)
    for scene in (hmmod.flat(), hmmod.box_scene(), hmmod.ramp_scene()):
        env = hmmod.wheel_envelope(scene, 0.35)  # the real placement surface
        xs = rng.uniform(scene.x0 + 0.2, scene.x0 + (scene.nx - 2) * scene.cell, 200)
        ys = rng.uniform(scene.y0 + 0.2, scene.y0 + (scene.ny - 2) * scene.cell, 200)
        elev = wp.array(np.ascontiguousarray(env.H, np.float32), dtype=wp.float32, device="cpu")
        g = GridParams(env.nx, env.ny, env.cell, env.x0, env.y0).build()
        wx = wp.array(xs.astype(np.float32), dtype=wp.float32, device="cpu")
        wy = wp.array(ys.astype(np.float32), dtype=wp.float32, device="cpu")
        oh = wp.zeros(len(xs), dtype=wp.float32, device="cpu")
        on = wp.zeros(len(xs), dtype=wp.vec3, device="cpu")
        wp.launch(_probe, len(xs), inputs=[elev, g, wx, wy], outputs=[oh, on], device="cpu")
        h_ref = env.sample(xs, ys)
        n_ref = env.normal(xs, ys)
        dh = np.abs(oh.numpy() - h_ref).max()
        dn = np.abs(on.numpy() - n_ref).max()
        print(f"  {h_ref.size} pts  max|dh|={dh:.2e}  max|dn|={dn:.2e}")
        assert dh < 1e-4 and dn < 1e-4, (dh, dn)
    print("terrain device-vs-numpy OK")


if __name__ == "__main__":
    selftest()
