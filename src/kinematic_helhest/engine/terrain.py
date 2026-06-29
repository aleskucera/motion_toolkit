from dataclasses import dataclass

import warp as wp


@wp.struct
class Grid:
    cells_x: wp.int32
    cells_y: wp.int32
    cell_size: wp.float32  # meters per cell
    origin_x: wp.float32  # world x of the min corner
    origin_y: wp.float32  # world y of the min corner


@dataclass
class GridParams:
    cells_x: int
    cells_y: int
    cell_size: float
    origin_x: float
    origin_y: float

    @property
    def bounds(self) -> tuple:
        """World extent (xmin, xmax, ymin, ymax) -- the convention the geodesic solvers take."""
        return (
            self.origin_x,
            self.origin_x + self.cells_x * self.cell_size,
            self.origin_y,
            self.origin_y + self.cells_y * self.cell_size,
        )

    def build(self) -> Grid:
        grid = Grid()
        grid.cells_x, grid.cells_y = int(self.cells_x), int(self.cells_y)
        grid.cell_size = float(self.cell_size)
        grid.origin_x, grid.origin_y = float(self.origin_x), float(self.origin_y)
        return grid


@wp.func
def _locate(grid: Grid, x: wp.float32, y: wp.float32):
    """World (x, y) -> bilinear stencil, packed as vec4(x_idx, y_idx, frac_x, frac_y): the lower-left
    corner index (as a float -- cast back with int()) and the in-cell offset toward +1, in [0,1].
    The ONE place the cell-center mapping `(x - origin)/cell_size - 0.5` lives -- shared by
    sample_field, its analytic gradient, and the d/dH adjoint scatter so the convention can never
    drift. Packed in a PLAIN vec4, not a struct: an int-member struct round-trip zeroes the auto-grad
    of `frac` w.r.t. (x, y), which silently kills sample_field's POSITION gradient (e.g. friction
    sampled at a pose-dependent contact point -- a cross-step term that grows with the rollout)."""
    fx = (x - grid.origin_x) / grid.cell_size - 0.5
    fy = (y - grid.origin_y) / grid.cell_size - 0.5
    x_idx = wp.clamp(int(wp.floor(fx)), 0, grid.cells_x - 2)
    y_idx = wp.clamp(int(wp.floor(fy)), 0, grid.cells_y - 2)
    frac_x = wp.clamp(fx - float(x_idx), 0.0, 1.0)
    frac_y = wp.clamp(fy - float(y_idx), 0.0, 1.0)
    return wp.vec4(float(x_idx), float(y_idx), frac_x, frac_y)


@wp.func
def sample_field(
    field: wp.array2d(dtype=wp.float32),
    grid: Grid,
    x: wp.float32,
    y: wp.float32,
):
    """Bilinear-interpolate a 2D grid field (elevation, envelope, friction, ...) at world (x, y).
    Differentiable w.r.t. BOTH the field values and the sample position (x, y) -- see `_locate`."""
    c = _locate(grid, x, y)
    xi = int(c[0])
    yi = int(c[1])
    frac_x = c[2]
    frac_y = c[3]

    v00 = field[yi, xi]
    v10 = field[yi, xi + 1]
    v01 = field[yi + 1, xi]
    v11 = field[yi + 1, xi + 1]

    return (
        (1.0 - frac_x) * (1.0 - frac_y) * v00
        + frac_x * (1.0 - frac_y) * v10
        + (1.0 - frac_x) * frac_y * v01
        + frac_x * frac_y * v11
    )


@wp.func
def sample_height_grad(
    elevation: wp.array2d(dtype=wp.float32),
    grid: Grid,
    x: wp.float32,
    y: wp.float32,
):
    c = _locate(grid, x, y)
    xi = int(c[0])
    yi = int(c[1])
    frac_x = c[2]
    frac_y = c[3]
    h00 = elevation[yi, xi]
    h10 = elevation[yi, xi + 1]
    h01 = elevation[yi + 1, xi]
    h11 = elevation[yi + 1, xi + 1]

    h = (
        (1.0 - frac_x) * (1.0 - frac_y) * h00
        + frac_x * (1.0 - frac_y) * h10
        + (1.0 - frac_x) * frac_y * h01
        + frac_x * frac_y * h11
    )

    gx = ((1.0 - frac_y) * (h10 - h00) + frac_y * (h11 - h01)) / grid.cell_size
    gy = ((1.0 - frac_x) * (h01 - h00) + frac_x * (h11 - h10)) / grid.cell_size
    return wp.vec3(h, gx, gy)


@wp.func
def sample_normal(
    elevation: wp.array2d(dtype=wp.float32),
    grid: Grid,
    x: float,
    y: float,
):
    e = grid.cell_size
    dhdx = (sample_field(elevation, grid, x + e, y) - sample_field(elevation, grid, x - e, y)) / (
        2.0 * e
    )
    dhdy = (sample_field(elevation, grid, x, y + e) - sample_field(elevation, grid, x, y - e)) / (
        2.0 * e
    )
    return wp.normalize(wp.vec3(-dhdx, -dhdy, 1.0))
