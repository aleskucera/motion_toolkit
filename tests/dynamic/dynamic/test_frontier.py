"""Frontier reconstruction from an organized scan, and its payoff for carving.

Run: python tests/dynamic/test_frontier.py   (CPU)
"""

from __future__ import annotations

import numpy as np
import warp as wp
from terrain_toolkit import DynamicFilterConfig
from terrain_toolkit import DynamicPointFilter
from terrain_toolkit import frontier_from_organized

MAX_RANGE = 45.0


def test_hit_and_miss_placement() -> None:
    # Three beams: forward, up, left. Middle one is a miss (zero point).
    beam_dirs = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0]], np.float64)
    beam_dirs /= np.linalg.norm(beam_dirs, axis=1, keepdims=True)
    points = np.array([beam_dirs[0] * 8.0, [0, 0, 0], beam_dirs[2] * 3.0], np.float32)

    frontier = frontier_from_organized(points, beam_dirs.astype(np.float32), MAX_RANGE)
    assert np.allclose(frontier[0], points[0])  # hit → keeps its point
    assert np.allclose(frontier[2], points[2])  # hit
    assert np.allclose(frontier[1], beam_dirs[1] * MAX_RANGE)  # miss → max-range point


def test_frontier_carves_sky_ghost() -> None:
    # A wall ahead (beams hit it) and up-tilted beams that miss into open sky.
    dev = wp.get_device("cpu")
    filt = DynamicPointFilter(
        DynamicFilterConfig(az_bins=180, el_bins=90, el_min_deg=-90, el_max_deg=90), device=dev
    )
    rng = np.random.default_rng(0)
    # Beam directions: a spread of forward (wall) and up (sky) beams.
    el = np.concatenate([rng.uniform(-0.2, 0.2, 400), rng.uniform(0.5, 1.0, 400)])  # rad
    az = rng.uniform(-0.3, 0.3, 800)
    beam_dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    # Forward beams hit a wall at x=8; up beams miss (zero).
    points = np.zeros((800, 3), np.float32)
    fwd = el < 0.3
    points[fwd] = beam_dirs[fwd] * (8.0 / np.cos(el[fwd]))[:, None]

    o = np.zeros(3)
    frontier = frontier_from_organized(points, beam_dirs.astype(np.float32), MAX_RANGE)

    # A stale ghost floating up in the sky (an up-beam bearing, at 3 m).
    ghost = (beam_dirs[el > 0.6][:50] * 3.0).astype(np.float32)

    def carve_keep(scan: np.ndarray) -> float:
        # carve() is device-native: upload the clouds, read back the mask.
        m = filt.carve(
            wp.array(ghost, dtype=wp.vec3, device=dev),
            wp.array(scan, dtype=wp.vec3, device=dev),
            o,
        )
        return float(m.numpy().mean())

    kept_frontier = carve_keep(frontier)
    kept_hits_only = carve_keep(points[fwd])  # only the wall returns
    assert kept_frontier < 0.1, f"frontier should carve the sky ghost; kept {kept_frontier:.2f}"
    assert kept_hits_only > 0.9, f"hits-only has no sky evidence; should keep {kept_hits_only:.2f}"


def main() -> None:
    wp.init()
    test_hit_and_miss_placement()
    test_frontier_carves_sky_ghost()
    print("PASS: frontier reconstruction + sky-ghost carving (2 tests)")


if __name__ == "__main__":
    main()
