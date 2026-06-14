"""Differentiable wheel-envelope dilation (Warp).

Grayscale morphological dilation of the raw heightmap by the spherical wheel cap:

    Henv[i,j] = max_{|d|<=R} ( H[i+di, j+dj] + sqrt(R^2 - d^2) - R ),   d = |off|*cell

Run ONCE per heightmap (shared across the whole batch + horizon), then the step
kernels sample `Henv`. One thread per output cell; the `max` reduction's adjoint
routes to the arg-max (contact) cell, so Warp autodiff gives d(loss)/d(raw h) for
free -- no hand-written backward. Mirrors `heightmap.wheel_envelope`.
"""
import numpy as np
import warp as wp


@wp.kernel
def _argmax_kernel(H: wp.array2d(dtype=wp.float32), cell: float, R: float, rad: int,
                   src_y: wp.array2d(dtype=wp.int32), src_x: wp.array2d(dtype=wp.int32),
                   capv: wp.array2d(dtype=wp.float32)):
    """Non-diff pass: pick the contact cell (arg-max of H[neighbor] + cap)."""
    iy, ix = wp.tid()
    ny = H.shape[0]
    nx = H.shape[1]
    best = float(-1.0e9)
    by = iy
    bx = ix
    bc = float(0.0)
    for di in range(-rad, rad + 1):
        for dj in range(-rad, rad + 1):
            d = wp.sqrt(float(di * di + dj * dj)) * cell
            if d <= R:
                cap = wp.sqrt(R * R - d * d) - R
                yy = wp.clamp(iy + di, 0, ny - 1)
                xx = wp.clamp(ix + dj, 0, nx - 1)
                v = H[yy, xx] + cap
                if v > best:
                    best = v
                    by = yy
                    bx = xx
                    bc = cap
    src_y[iy, ix] = by
    src_x[iy, ix] = bx
    capv[iy, ix] = bc


@wp.kernel
def _gather_kernel(H: wp.array2d(dtype=wp.float32),
                   src_y: wp.array2d(dtype=wp.int32), src_x: wp.array2d(dtype=wp.int32),
                   capv: wp.array2d(dtype=wp.float32), Henv: wp.array2d(dtype=wp.float32)):
    """Diff pass: Henv = H[contact cell] + cap. Adjoint scatters to the contact cell."""
    iy, ix = wp.tid()
    Henv[iy, ix] = H[src_y[iy, ix], src_x[iy, ix]] + capv[iy, ix]


def wheel_envelope(H, cell, R, device="cpu"):
    """H: device wp.array2d(float32) raw heights -> device Henv (same shape/grid).

    Two passes: arg-max selection (non-diff) then a differentiable gather, so the
    tape gets d(loss)/d(raw h) routed to the contact cell. Carries H.requires_grad.
    """
    ny, nx = H.shape
    rad = int(np.ceil(R / cell))
    src_y = wp.zeros((ny, nx), dtype=wp.int32, device=device)
    src_x = wp.zeros((ny, nx), dtype=wp.int32, device=device)
    capv = wp.zeros((ny, nx), dtype=wp.float32, device=device)
    Henv = wp.zeros((ny, nx), dtype=wp.float32, device=device,
                    requires_grad=H.requires_grad)
    wp.launch(_argmax_kernel, (ny, nx), inputs=[H, float(cell), float(R), rad],
              outputs=[src_y, src_x, capv], device=device)
    wp.launch(_gather_kernel, (ny, nx), inputs=[H, src_y, src_x, capv],
              outputs=[Henv], device=device)
    return Henv


@wp.kernel
def _weighted_sum(Henv: wp.array2d(dtype=wp.float32), W: wp.array2d(dtype=wp.float32),
                  loss: wp.array(dtype=wp.float32)):
    iy, ix = wp.tid()
    wp.atomic_add(loss, 0, W[iy, ix] * Henv[iy, ix])


def _selftest_forward():
    """Device dilation vs numpy heightmap.wheel_envelope on the real scenes."""
    from .. import heightmap as hmmod

    wp.init()
    worst = 0.0
    for name, scene in [("flat", hmmod.flat()), ("box", hmmod.box_scene()),
                        ("ramp", hmmod.ramp_scene())]:
        R = 0.35
        ref = hmmod.wheel_envelope(scene, R).H
        H = wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device="cpu")
        Henv = wheel_envelope(H, scene.cell, R, "cpu").numpy()
        d = np.abs(Henv - ref).max()
        worst = max(worst, d)
        print(f"  {name:4s} grid={scene.H.shape}  max|dHenv|={d:.2e}")
    print(f"envelope forward device-vs-numpy worst={worst:.2e}  {'OK' if worst < 1e-4 else 'REVIEW'}")


def _selftest_backward():
    """Autodiff d(loss)/d(raw h) vs the numpy analytic subgradient (tie-immune),
    plus a coarse finite-difference sanity check (tie noise expected)."""
    wp.init()
    rng = np.random.default_rng(1)
    ny, nx, cell, R = 12, 12, 0.1, 0.35
    H0 = rng.uniform(0.0, 0.5, (ny, nx)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, (ny, nx)).astype(np.float32)
    Wd = wp.array(W, dtype=wp.float32, device="cpu")

    H = wp.array(H0, dtype=wp.float32, device="cpu", requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device="cpu", requires_grad=True)
    tape = wp.Tape()
    with tape:
        Henv = wheel_envelope(H, cell, R, "cpu")
        wp.launch(_weighted_sum, (ny, nx), inputs=[Henv, Wd], outputs=[loss], device="cpu")
    tape.backward(loss=loss)
    g_ad = H.grad.numpy()

    g_an = _numpy_subgrad(H0, W, cell, R)
    err = np.abs(g_ad - g_an).max()

    eps = 1e-3  # FD: expect tie-switch noise at a few cells
    g_fd = np.zeros_like(H0)
    for i in range(ny):
        for j in range(nx):
            g_fd[i, j] = (_loss_at(H0, W, cell, R, i, j, +eps)
                          - _loss_at(H0, W, cell, R, i, j, -eps)) / (2 * eps)
    fd_med = np.median(np.abs(g_ad - g_fd))

    print(f"  grid={ny}x{nx}  max|g_ad-g_analytic|={err:.2e}  "
          f"median|g_ad-g_fd|={fd_med:.2e}  ||g||={np.abs(g_an).max():.2f}")
    print(f"envelope backward autodiff-vs-analytic  {'OK' if err < 1e-3 else 'REVIEW'}")


def _numpy_subgrad(H0, W, cell, R):
    """Analytic subgradient: route each output's weight to its arg-max contact cell
    (same first-wins convention as _argmax_kernel)."""
    ny, nx = H0.shape
    rad = int(np.ceil(R / cell))
    g = np.zeros_like(H0)
    for i in range(ny):
        for j in range(nx):
            best, sy, sx = -1e9, i, j
            for di in range(-rad, rad + 1):
                for dj in range(-rad, rad + 1):
                    d = np.hypot(di, dj) * cell
                    if d <= R:
                        cap = np.sqrt(R * R - d * d) - R
                        yy = min(max(i + di, 0), ny - 1)
                        xx = min(max(j + dj, 0), nx - 1)
                        v = H0[yy, xx] + cap
                        if v > best:
                            best, sy, sx = v, yy, xx
            g[sy, sx] += W[i, j]
    return g


def _loss_at(H0, W, cell, R, i, j, delta):
    Hp = H0.copy()
    Hp[i, j] += delta
    H = wp.array(Hp, dtype=wp.float32, device="cpu")
    Henv = wheel_envelope(H, cell, R, "cpu").numpy().astype(np.float64)
    return float((W.astype(np.float64) * Henv).sum())


if __name__ == "__main__":
    _selftest_forward()
    _selftest_backward()
