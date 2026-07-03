"""Phase-1 verification: device terrain ingest + on-device wheel envelope.

Three checks (CPU by default; pass --device cuda for the GPU path):
  1. cell-center alignment — a cell-center raster (helhest.perception-style) sampled
     at known world points matches the analytic plane with the grid origin set to
     the bounds min corner directly (one convention end-to-end, no half-cell shift);
  2. on-device `engine.envelope.wheel_envelope` == numpy `heightmap.wheel_envelope`
     oracle on the real scenes;
  3. a device-fed `ForwardSimulator` reproduces the host (numpy-scene) path
     rollout-for-rollout on the same cell grid.

Run:  python -m tests.engine.verify_device [--device cuda]
"""

import argparse

import numpy as np
import warp as wp
from helhest import friction
from helhest import heightmap as hmmod
from helhest.control.reference import _to_wheel_omega
from helhest.engine import ForwardSimulator
from helhest.engine import Grid
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import sample_field
from helhest.engine import SolverParams
from helhest.engine.envelope import _contact_kernel
from helhest.engine.envelope import _gather_kernel


def wheel_envelope(elevation, cell_size, wheel_radius, device="cpu"):
    """Verification-only: allocate scratch + run the two engine envelope passes
    (raw elevation -> dilated). Mirrors what `ForwardSimulator.set_terrain` does into its
    owned buffers."""
    ny, nx = elevation.shape
    env_radius = int(np.ceil(wheel_radius / cell_size))
    contact_iy = wp.zeros((ny, nx), dtype=wp.int32, device=device)
    contact_ix = wp.zeros((ny, nx), dtype=wp.int32, device=device)
    contact_cap = wp.zeros((ny, nx), dtype=wp.float32, device=device)
    envelope = wp.zeros((ny, nx), dtype=wp.float32, device=device)
    wp.launch(
        _contact_kernel,
        dim=elevation.shape,
        inputs=[elevation, float(cell_size), float(wheel_radius), env_radius],
        outputs=[contact_iy, contact_ix, contact_cap],
        device=device,
    )
    wp.launch(
        _gather_kernel,
        dim=elevation.shape,
        inputs=[elevation, contact_iy, contact_ix, contact_cap],
        outputs=[envelope],
        device=device,
    )
    return envelope


@wp.kernel
def _probe_h(
    elevation: wp.array2d(dtype=wp.float32),
    g: Grid,
    xs: wp.array(dtype=wp.float32),
    ys: wp.array(dtype=wp.float32),
    out_h: wp.array(dtype=wp.float32),
):
    """Verification-only: launch the engine height sampler from host code."""
    i = wp.tid()
    out_h[i] = sample_field(elevation, g, xs[i], ys[i])


def _sample(elevation, g, xs, ys, device):
    """Bilinear height at world (xs, ys) via the engine's device sampler."""
    wx = wp.array(xs.astype(np.float32), dtype=wp.float32, device=device)
    wy = wp.array(ys.astype(np.float32), dtype=wp.float32, device=device)
    oh = wp.zeros(len(xs), dtype=wp.float32, device=device)
    wp.launch(_probe_h, len(xs), inputs=[elevation, g, wx, wy], outputs=[oh], device=device)
    return oh.numpy()


def check_alignment(device):
    """Cell-center raster (helhest.perception-style) of a tilted plane f = a*x + b*y.
    The engine sampler uses the cell-center convention, so the grid origin is the
    bounds min corner directly -- no half-cell shift -- and bilinear (exact for a
    plane) recovers f at arbitrary world points."""
    a, b, res = 0.3, 0.2, 0.05
    bounds = (-1.0, 1.0, -1.0, 1.0)  # (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = bounds
    nx = int(round((xmax - xmin) / res))
    ny = int(round((ymax - ymin) / res))
    xc = xmin + (np.arange(nx) + 0.5) * res
    yc = ymin + (np.arange(ny) + 0.5) * res
    XX, YY = np.meshgrid(xc, yc)  # [ny, nx], values at cell centers
    H = (a * XX + b * YY).astype(np.float32)
    H_wp = wp.array(np.ascontiguousarray(H), dtype=wp.float32, device=device)

    g = GridParams(nx, ny, res, xmin, ymin).build()  # origin = min corner, no shift
    xs = np.array([-0.3, 0.1, 0.42], np.float64)
    ys = np.array([0.2, -0.15, 0.05], np.float64)
    f = a * xs + b * ys
    err = np.abs(_sample(H_wp, g, xs, ys, device) - f).max()
    print(f"  alignment: cell-center origin err={err:.2e}")
    assert err < 1e-5, err
    print("  alignment OK")


def check_envelope(device, R=0.35):
    worst = 0.0
    for name, scene in [
        ("flat", hmmod.flat()),
        ("box", hmmod.box_scene()),
        ("ramp", hmmod.ramp_scene()),
    ]:
        ref = hmmod.wheel_envelope(scene, R).H
        H_wp = wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
        got = wheel_envelope(H_wp, scene.cell, R, device).numpy()
        d = float(np.abs(got - ref).max())
        worst = max(worst, d)
        print(f"  envelope[{name}] max|dHenv|={d:.2e}")
    assert worst < 1e-4, worst
    print(f"  envelope parity OK (worst={worst:.2e})")


def check_end_to_end(device, B=16, T=25):
    """Same node grid both ways: device path must match the host path exactly."""
    scene = hmmod.box_scene()
    mu = friction.uniform(0.8)  # default extent matches box_scene
    params = SolverParams(dt=0.05, k_turn=2.0, newton_iters=12)
    start = (-1.0, 0.0, 0.0)
    wheel_omega = _to_wheel_omega(np.full((B, T, 2), 2.0, np.float32))

    host = ForwardSimulator(
        RobotParams(),
        params,
        GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0),
        B,
        T,
        device,
    )
    host.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    host.set_friction(mu)
    ph, _, ch, rh = host.rollout(wheel_omega, start)

    H_wp = wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    dev = ForwardSimulator(RobotParams(), params, grid, B, T, device)
    dev.set_terrain(H_wp)
    dev.set_uniform_friction(0.8)
    pd, _, cd, rd = dev.rollout(wheel_omega, start)

    dp = float(np.abs(ph - pd).max())
    dc = float(np.abs(ch - cd).max())
    dr = float(np.abs(rh - rd).max())
    print(f"  end-to-end device-vs-host  dplanar={dp:.2e} dclear={dc:.2e} dresid={dr:.2e}")
    assert max(dp, dc, dr) < 1e-4, (dp, dc, dr)
    print("  end-to-end OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu", help="warp device: cpu or cuda")
    args = ap.parse_args()
    wp.init()
    print(f"[1/3] alignment ({args.device})")
    check_alignment(args.device)
    print(f"[2/3] envelope parity ({args.device})")
    check_envelope(args.device)
    print(f"[3/3] end-to-end ({args.device})")
    check_end_to_end(args.device)
    print("Phase-1 device path: ALL OK")


if __name__ == "__main__":
    main()
