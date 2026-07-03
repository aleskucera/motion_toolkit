from __future__ import annotations


def meters_to_cells(distance_m: float, grid_resolution: float) -> int:
    """Convert a distance in meters to the nearest integer number of grid cells."""
    if grid_resolution <= 0:
        return 0
    return int(round(distance_m / grid_resolution))


def cells_to_meters(distance_cells: int, grid_resolution: float) -> float:
    """Convert a distance in grid cells to meters."""
    return float(distance_cells * grid_resolution)
