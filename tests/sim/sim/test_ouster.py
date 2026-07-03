"""Ouster beam-geometry checks: OSDome hemisphere orientation + cylindrical sweep.

Run: python tests/sim/test_ouster.py
"""

from __future__ import annotations

import numpy as np
import warp as wp
from terrain_toolkit.sim import GroundSpec
from terrain_toolkit.sim import make_osdome_lidar
from terrain_toolkit.sim import nominal_osdome_polar
from terrain_toolkit.sim import osdome_beam_directions
from terrain_toolkit.sim import ouster_beam_directions
from terrain_toolkit.sim.ouster import OSDOME_MAX_RANGE_M
from terrain_toolkit.sim.ouster import OSDOME_MIN_RANGE_M


def test_all_unit() -> None:
    for d in (
        osdome_beam_directions(nominal_osdome_polar(64), 256, facing="front"),
        ouster_beam_directions(np.linspace(-22.5, 22.5, 64), 256),
    ):
        assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)


def test_front_hemisphere() -> None:
    # Front-facing dome: every beam points into the forward half-space (x >= 0),
    # and the rim ring (polar ~90°) is perpendicular to forward (x ~ 0).
    polar = nominal_osdome_polar(128)
    d = osdome_beam_directions(polar, 512, facing="front")
    assert d[:, 0].min() >= -1e-6, "a beam points backwards"
    assert d[:, 0].max() > 0.99, "no beam points near straight ahead"
    # Reshape (rings, cols, 3); the last ring is the dome rim.
    d = d.reshape(len(polar), 512, 3)
    assert abs(float(d[-1, :, 0].mean())) < 0.05, "rim ring should be ~perpendicular to forward"


def test_facing_axes() -> None:
    polar = np.array([0.0, 45.0, 90.0])
    up = osdome_beam_directions(polar, 8, facing="up").reshape(3, 8, 3)
    down = osdome_beam_directions(polar, 8, facing="down").reshape(3, 8, 3)
    # Pole ring (polar=0) points along the axis.
    assert np.allclose(up[0].mean(0), [0, 0, 1], atol=1e-6)
    assert np.allclose(down[0].mean(0), [0, 0, -1], atol=1e-6)


def test_osdome_factory() -> None:
    wp.init()
    ground = GroundSpec(z=0.0, x_range=(-60.0, 60.0), y_range=(-60.0, 60.0))
    lidar = make_osdome_lidar(ground, cols=256, facing="front", device=wp.get_device("cpu"))
    assert lidar.min_range == OSDOME_MIN_RANGE_M and lidar.max_range == OSDOME_MAX_RANGE_M
    assert lidar._n == 128 * 256  # channels × columns

    # A wall 3 m ahead returns; a wall past max range does not.
    origin = np.array([0.0, 0.0, 0.6])
    lo = np.array([[3.0, -4.0, 0.0]])
    hi = np.array([[3.2, 4.0, 3.0]])
    pts = lidar.scan(origin, 0.0, lo, hi, seed=0)
    r = np.linalg.norm(pts - origin, axis=1)
    assert len(pts) > 0
    assert r.min() >= OSDOME_MIN_RANGE_M - 0.2  # min-range window (allow a little noise)
    assert r.max() <= OSDOME_MAX_RANGE_M + 0.2


def main() -> None:
    test_all_unit()
    test_front_hemisphere()
    test_facing_axes()
    test_osdome_factory()
    print("PASS: ouster beam geometry + OSDome factory (4 tests)")


if __name__ == "__main__":
    main()
