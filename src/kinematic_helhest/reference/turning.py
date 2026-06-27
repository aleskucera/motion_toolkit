"""Friction -> turning-parameter map (Phase 4).

The moment-centroid map from per-wheel friction and normal load to the two ICR
parameters used by the twist:

    w_i   = mu_i * N_i                      # lateral grip "weight" at each wheel
    x_ICR = sum_i w_i x_i / sum_i w_i       # rotation center pulled toward grippy wheels
    alpha = 1 + k * sum_i w_i / (g * m)     # more total grip -> wider effective track

For uniform friction on flat ground this reduces to alpha = 1 + k*mu and
x_ICR = CoM_x (rotation about the load centroid). Lowering the rear friction
shrinks sum w -> alpha down -> the robot yaws MORE (the replay_real mu_rear
finding), and pulls x_ICR toward the front axle (rear kicks out).
"""

import numpy as np

from ..model import GRAVITY
from ..model import MASS
from ..model import WHEEL_X


def turning_params(mu, N, k, wheel_x=WHEEL_X, mass=MASS, g=GRAVITY):
    """mu [3], N [3] (normal load magnitudes, Newtons) -> (alpha, x_ICR)."""
    mu = np.asarray(mu, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    w = mu * N
    sw = w.sum()
    x_icr = float((w * wheel_x).sum() / sw)
    alpha = float(1.0 + k * sw / (g * mass))
    return alpha, x_icr
