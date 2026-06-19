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

    def build(self) -> Grid:
        grid = Grid()
        grid.cells_x, grid.cells_y = int(self.cells_x), int(self.cells_y)
        grid.cell_size = float(self.cell_size)
        grid.origin_x, grid.origin_y = float(self.origin_x), float(self.origin_y)
        return grid


@wp.struct
class _Cell:
    """Bilinear stencil for a world point: lower-left corner index + in-cell fraction."""

    x_idx: wp.int32  # x index of the lower-left corner cell (-> elevation[y_idx, x_idx])
    y_idx: wp.int32  # y index of the lower-left corner cell
    frac_x: wp.float32  # in-cell offset toward x_idx+1, in [0,1] (the bilinear weight)
    frac_y: wp.float32  # in-cell offset toward y_idx+1, in [0,1]


@wp.func
def _locate(grid: Grid, x: wp.float32, y: wp.float32):
    """World (x, y) -> bilinear stencil. The ONE place the cell-center mapping
    `(x - origin)/cell_size - 0.5` lives -- shared by sample_field, its analytic
    gradient, and the d/dH adjoint scatter so the convention can never drift."""
    fx = (x - grid.origin_x) / grid.cell_size - 0.5
    fy = (y - grid.origin_y) / grid.cell_size - 0.5
    c = _Cell()
    c.x_idx = wp.clamp(int(wp.floor(fx)), 0, grid.cells_x - 2)
    c.y_idx = wp.clamp(int(wp.floor(fy)), 0, grid.cells_y - 2)
    c.frac_x = wp.clamp(fx - float(c.x_idx), 0.0, 1.0)
    c.frac_y = wp.clamp(fy - float(c.y_idx), 0.0, 1.0)
    return c


@wp.func
def sample_field(
    field: wp.array2d(dtype=wp.float32),
    grid: Grid,
    x: wp.float32,
    y: wp.float32,
):
    """Bilinear-interpolate a 2D grid field (elevation, envelope, friction, ...) at world (x, y)."""
    c = _locate(grid, x, y)

    v00 = field[c.y_idx, c.x_idx]
    v10 = field[c.y_idx, c.x_idx + 1]
    v01 = field[c.y_idx + 1, c.x_idx]
    v11 = field[c.y_idx + 1, c.x_idx + 1]

    return (
        (1.0 - c.frac_x) * (1.0 - c.frac_y) * v00
        + c.frac_x * (1.0 - c.frac_y) * v10
        + (1.0 - c.frac_x) * c.frac_y * v01
        + c.frac_x * c.frac_y * v11
    )


@wp.func
def sample_height_grad(
    elevation: wp.array2d(dtype=wp.float32),
    grid: Grid,
    x: wp.float32,
    y: wp.float32,
):
    c = _locate(grid, x, y)
    h00 = elevation[c.y_idx, c.x_idx]
    h10 = elevation[c.y_idx, c.x_idx + 1]
    h01 = elevation[c.y_idx + 1, c.x_idx]
    h11 = elevation[c.y_idx + 1, c.x_idx + 1]

    h = (
        (1.0 - c.frac_x) * (1.0 - c.frac_y) * h00
        + c.frac_x * (1.0 - c.frac_y) * h10
        + (1.0 - c.frac_x) * c.frac_y * h01
        + c.frac_x * c.frac_y * h11
    )

    gx = ((1.0 - c.frac_y) * (h10 - h00) + c.frac_y * (h11 - h01)) / grid.cell_size
    gy = ((1.0 - c.frac_x) * (h01 - h00) + c.frac_x * (h11 - h10)) / grid.cell_size
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
