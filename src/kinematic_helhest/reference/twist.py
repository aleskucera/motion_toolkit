"""Differential-drive (skid-steer) twist: wheel speeds -> body velocity.

No-slip skid-steer kinematics with friction-dependent turning lumped into two
ICR parameters:

    vx  = R (wL + wR) / 2
    wz  = R (wR - wL) / (2 b alpha)        # alpha >= 1 widens the effective track
    vy  = -x_ICR * wz                      # lateral drift from a longitudinal ICR offset

(alpha, x_ICR) come from the friction field via the moment-centroid map (Phase 4),
or are passed as constants. The rear wheel is kinematically redundant (consistent
when driven at the L/R average) and does not enter the twist.

Conventions: body X fwd, Y left, yaw CCW+. Arrays may carry a leading batch dim.
"""

import numpy as np

from ..model import HALF_TRACK
from ..model import WHEEL_RADIUS


def wheel_twist(omega, alpha=1.0, x_icr=0.0, R=WHEEL_RADIUS, b=HALF_TRACK):
    """Body twist (vx, vy, wz) from wheel speeds.

    omega : [..., 3] angular velocities [left, right, rear] (rad/s).
    alpha, x_icr : scalars or [...] broadcastable arrays.
    Returns vx, vy, wz each shaped like omega[..., 0].
    """
    omega = np.asarray(omega, dtype=np.float64)
    wL, wR = omega[..., 0], omega[..., 1]
    vx = R * (wL + wR) / 2.0
    wz = R * (wR - wL) / (2.0 * b * np.asarray(alpha, dtype=np.float64))
    vy = -np.asarray(x_icr, dtype=np.float64) * wz
    return vx, vy, wz
