"""Differentiable wheel-envelope dilation (Warp).

Grayscale morphological dilation of the raw elevation by the spherical wheel cap:

    envelope[i,j] = max_{|d|<=R} ( elevation[i+dy, j+dx] + sqrt(R^2 - d^2) - R ),  d = |off|*cell_size

Run ONCE per elevation grid (shared across the whole batch + horizon), then the
step kernels sample `envelope`. One thread per output cell; the `max` reduction's
adjoint routes to the arg-max (contact) cell, so Warp autodiff gives
d(loss)/d(raw elevation) for free -- no hand-written backward. Mirrors
`heightmap.wheel_envelope`.
"""

import warp as wp


@wp.kernel
def _contact_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    cell_size: float,
    wheel_radius: float,
    env_radius: int,
    contact_iy: wp.array2d(dtype=wp.int32),
    contact_ix: wp.array2d(dtype=wp.int32),
    contact_cap: wp.array2d(dtype=wp.float32),
):
    """Non-diff pass: pick the contact cell (arg-max of elevation[neighbor] + cap)."""
    iy, ix = wp.tid()
    ny = elevation.shape[0]
    nx = elevation.shape[1]
    best_lift = float(-1.0e9)
    best_iy = iy
    best_ix = ix
    best_cap = float(0.0)
    for dy in range(-env_radius, env_radius + 1):
        for dx in range(-env_radius, env_radius + 1):
            dist = wp.sqrt(float(dy * dy + dx * dx)) * cell_size
            if dist <= wheel_radius:
                cap = wp.sqrt(wheel_radius * wheel_radius - dist * dist) - wheel_radius
                qy = wp.clamp(iy + dy, 0, ny - 1)
                qx = wp.clamp(ix + dx, 0, nx - 1)
                lift = elevation[qy, qx] + cap
                if lift > best_lift:
                    best_lift = lift
                    best_iy = qy
                    best_ix = qx
                    best_cap = cap
    contact_iy[iy, ix] = best_iy
    contact_ix[iy, ix] = best_ix
    contact_cap[iy, ix] = best_cap


@wp.kernel
def _gather_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    contact_iy: wp.array2d(dtype=wp.int32),
    contact_ix: wp.array2d(dtype=wp.int32),
    contact_cap: wp.array2d(dtype=wp.float32),
    envelope: wp.array2d(dtype=wp.float32),
):
    """Diff pass: envelope = elevation[contact cell] + cap. Adjoint scatters to it."""
    iy, ix = wp.tid()
    envelope[iy, ix] = elevation[contact_iy[iy, ix], contact_ix[iy, ix]] + contact_cap[iy, ix]
