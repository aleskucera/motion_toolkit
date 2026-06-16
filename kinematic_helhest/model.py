"""Helhest Junior geometry / mass constants for the kinematic simulator.

Values copied from examples/helhest_junior/common.py (HelhestJuniorConfig) so the
kinematic twin matches the dynamic model without importing Newton/Ostrich.

Body frame: X forward, Y left, Z up. Origin = chassis link origin; wheel hubs lie
in the Z=0 plane. Wheel order is [left, right, rear] == dynamic-sim DOFs 6,7,8.
"""
import numpy as np

WHEEL_RADIUS = 0.35   # R  [m]
HALF_TRACK = 0.365    # b  [m]  front wheels at |y| = b
REAR_OFFSET = 0.75    # l  [m]  rear wheel at x = -l
GRAVITY = 9.81        # [m/s^2]

# Wheel hub positions in body frame, order [left, right, rear].
WHEEL_POS = np.array(
    [
        [0.0, HALF_TRACK, 0.0],    # left
        [0.0, -HALF_TRACK, 0.0],   # right
        [-REAR_OFFSET, 0.0, 0.0],  # rear
    ],
    dtype=np.float64,
)
WHEEL_X = WHEEL_POS[:, 0].copy()  # body-frame longitudinal coord per wheel

# Point masses (chassis boxes + wheels): (cx, cy, cz, mass_kg) in body frame.
_MASSES = np.array(
    [
        [-0.13, 0.0, 0.0, 78.8375],   # front box
        [-0.61, 0.0, 0.0, 10.8625],   # rear box
        [0.0, HALF_TRACK, 0.0, 5.5],  # left wheel
        [0.0, -HALF_TRACK, 0.0, 5.5], # right wheel
        [-REAR_OFFSET, 0.0, 0.0, 5.5],# rear wheel
    ]
)
MASS = float(_MASSES[:, 3].sum())                        # 106.2 kg
COM = (_MASSES[:, :3] * _MASSES[:, 3:4]).sum(0) / MASS   # body-frame CoM (x≈-0.198)
# NOTE: geometric CoM_z = 0 here (boxes/wheels centered in the hub plane). The
# real CoM sits slightly higher; this only affects load transfer on slopes
# (Phase 2+), where a nonzero CoM_z can be set if the data demands it.


def chassis_sample_points(nx=3, ny=3):
    """Bottom-face sample grid of the two chassis boxes, body frame [Np, 3].

    Used as unilateral non-penetration candidates for high-center detection
    (Phase 3). A grid (not just corners) so obstacles under the belly center are
    caught, not only those under a corner.
    """
    # (cx, cy, cz, hx, hy, hz) — half-extents of front and rear box.
    boxes = [
        (-0.13, 0.0, 0.0, 0.24, 0.28, 0.10),
        (-0.61, 0.0, 0.0, 0.24, 0.12, 0.10),
    ]
    pts = []
    for cx, cy, cz, hx, hy, hz in boxes:
        for sx in np.linspace(-1.0, 1.0, nx):
            for sy in np.linspace(-1.0, 1.0, ny):
                pts.append([cx + sx * hx, cy + sy * hy, cz - hz])
    return np.array(pts, dtype=np.float64)
