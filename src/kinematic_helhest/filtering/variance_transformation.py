"""Experimental: how the helhest kinematic model transforms process-noise covariance.

Linearizes the flat-ground planar predict step around a given (state, input) point to
obtain the discrete state-transition matrix F, then evaluates F @ Q @ F^T.
"""

from __future__ import annotations

import numpy as np

from ..dynamics import DT
from ..model import HALF_TRACK
from ..model import WHEEL_RADIUS

# ---------------------------------------------------------------------------
# Nonlinear helhest kinematic model — flat-ground specialisation
# (pitch = roll = 0, so R_full = Rz(yaw) only)
#
# State:  s = [x, y, yaw]          (world-frame position + heading)
# Input:  u = [omega_L, omega_R]   (left / right wheel angular speeds, rad/s)
# Params: R  = wheel radius [m]
#         b  = half-track [m]
#         alpha  >= 1, friction-dependent effective-track widening (default 1)
#         x_icr  longitudinal ICR offset [m] (default 0)
#
# --- Body twist (skid-steer, reference/twist.py) ---
#
#   vx = R * (omega_L + omega_R) / 2
#   wz = R * (omega_R - omega_L) / (2 * b * alpha)
#   vy = -x_icr * wz
#
# --- Rotation into world frame ---
#
#   Rz(yaw) = [[cos(yaw), -sin(yaw), 0],
#              [sin(yaw),  cos(yaw), 0],
#              [0,         0,        1]]
#
#   v_world = Rz(yaw) @ [vx, vy, 0]
#           = [vx*cos(yaw) - vy*sin(yaw),
#              vx*sin(yaw) + vy*cos(yaw),
#              0]
#
# --- Forward Euler integration ---
#
#   x_{k+1}   = x_k   + v_world_x * dt
#   y_{k+1}   = y_k   + v_world_y * dt
#   yaw_{k+1} = yaw_k + wz        * dt
#
# ---------------------------------------------------------------------------


def step(
    s: np.ndarray,
    u: np.ndarray,
    dt: float = DT,
    alpha: float = 1.0,
    x_icr: float = 0.0,
    R: float = WHEEL_RADIUS,
    b: float = HALF_TRACK,
) -> np.ndarray:
    """One flat-ground forward-Euler step of the helhest kinematic model.

    s : [x, y, yaw]
    u : [omega_L, omega_R]  (rad/s)
    Returns next state [x, y, yaw].
    """
    x, y, yaw = s
    omega_L, omega_R = u

    vx = R * (omega_L + omega_R) / 2.0
    wz = R * (omega_R - omega_L) / (2.0 * b * alpha)
    vy = -x_icr * wz

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    vw_x = vx * cos_yaw - vy * sin_yaw
    vw_y = vx * sin_yaw + vy * cos_yaw

    return np.array([x + vw_x * dt, y + vw_y * dt, yaw + wz * dt])


def jacobian_F(
    s0: np.ndarray,
    u0: np.ndarray,
    dt: float = DT,
    alpha: float = 1.0,
    x_icr: float = 0.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Discrete state-transition Jacobian F = df/ds at (s0, u0), via central differences.

    # Analytical F at s0=(0,0,0), u0=[w0,w0] (straight motion, x_icr=0, alpha=1):
    #
    #   vx = R * w0,  wz = 0
    #
    #   F = [[1,  0,     0    ],
    #        [0,  1,  vx*dt  ],
    #        [0,  0,     1    ]]
    #
    # => F @ I @ F^T = [[1,            0,         0    ],
    #                   [0,  1+(vx*dt)^2,    vx*dt  ],
    #                   [0,      vx*dt,         1    ]]
    #
    # Interpretation: yaw noise (sigma_yaw^2 = 1) leaks into y with gain vx*dt
    # and inflates y variance by (vx*dt)^2.  x and yaw variances are unchanged.
    """
    n = len(s0)
    F = np.zeros((n, n))
    for i in range(n):
        s_plus = s0.copy()
        s_plus[i] += eps
        s_minus = s0.copy()
        s_minus[i] -= eps
        F[:, i] = (
            step(s_plus, u0, dt, alpha, x_icr) - step(s_minus, u0, dt, alpha, x_icr)
        ) / (2.0 * eps)
    return F


def transform_covariance(F: np.ndarray, Q: np.ndarray | None = None) -> np.ndarray:
    """Return F @ Q @ F^T.

    Q defaults to the identity matrix (unit variance on each state component).
    """
    if Q is None:
        Q = np.eye(F.shape[0])
    return F @ Q @ F.T


if __name__ == "__main__":
    omega0 = 1.0  # rad/s — nominal forward wheel speed (~0.35 m/s)
    s0 = np.array([0.0, 0.0, 0.0])
    u0 = np.array([omega0, omega0])

    F = jacobian_F(s0, u0)
    P = transform_covariance(F)

    np.set_printoptions(precision=6, suppress=True)
    print("Linearisation point:  s0 =", s0, "  u0 =", u0)
    print()
    print("F =\n", F)
    print()
    print("Q = I_3  (default)")
    print()
    print("F @ Q @ F^T =\n", P)
