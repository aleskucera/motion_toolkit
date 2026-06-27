"""Per-cell friction field mu(x, y).

Reuses the Heightmap grid container (same bilinear sampling) to hold mu values.
This is the free per-cell learnable field of Phase 5/6; here we just build test
fields (uniform, and a low-friction strip along the rear-wheel track).
"""

import numpy as np

from . import heightmap


def uniform(value, xlim=(-2.0, 6.0), ylim=(-3.0, 3.0), cell=0.05):
    XX, _ = heightmap._grid(xlim, ylim, cell)
    return heightmap.Heightmap(np.full_like(XX, float(value)), (xlim[0], ylim[0]), cell)


def with_strip(base, low, ywidth=0.18):
    """Copy `base` and set a low-friction strip |y| < ywidth (the rear-wheel
    track at y=0; front wheels at |y|=0.365 stay on the base value)."""
    fm = heightmap.Heightmap(base.H.copy(), (base.x0, base.y0), base.cell)
    ys = base.y0 + (np.arange(base.ny) + 0.5) * base.cell  # cell centers
    fm.H[np.abs(ys) < ywidth, :] = float(low)
    return fm
