"""Extended Kalman Filter for the Helhest planar state [x, y, ψ].

State dimension: 3
    x   — world-frame easting  [m]
    y   — world-frame northing [m]
    ψ   — heading, Z-Y-X convention, [rad]; nose-up pitch is NEGATIVE (not used here)

Two measurement update paths are provided:
    update_icp      — LiDAR ICP pose  [x, y, ψ]
    update_odom_imu — odom [x, y] + IMU [ψ] fused into a single [x, y, ψ] observation

The caller is responsible for:
    • running the nonlinear process model  f(xₜ, u)  to get x_pred
    • computing the linearised state-transition matrix  A = ∂f/∂x  at (xₜ, u)
Both are passed directly to predict(); the EKF does not call f internally.
"""

from __future__ import annotations

import numpy as np


class EKF:
    """Minimal EKF for the 3-DOF Helhest planar pose."""

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R_icp: np.ndarray,
        R_odom_imu: np.ndarray,
    ) -> None:
        """
        x0          : [3]   initial state  [x, y, ψ]
        P0          : [3,3] initial state covariance
        Q           : [3,3] process noise covariance
        R_icp       : [3,3] ICP measurement noise covariance
        R_odom_imu  : [3,3] odom+IMU measurement noise covariance
        """
        self.x: np.ndarray = x0.copy()
        self.P: np.ndarray = P0.copy()
        self.Q: np.ndarray = Q.copy()
        self.R_icp: np.ndarray = R_icp.copy()
        self.R_odom_imu: np.ndarray = R_odom_imu.copy()

    def predict(self, A: np.ndarray, x_pred: np.ndarray) -> None:
        """Propagate the filter by one timestep.

        A      : [3,3] linearised state-transition matrix  ∂f/∂x  evaluated at
                 the current (state, input) pair — supplied by the caller
        x_pred : [3]   nonlinear model prediction  f(xₜ, u) — also from caller
        """
        self.x = x_pred.copy()
        self.P = A @ self.P @ A.T + self.Q

    def update_icp(self, z: np.ndarray) -> None:
        """Measurement update from LiDAR ICP.

        z : [3]  observed pose  [x, y, ψ]  in world frame [m, m, rad]
        """
        self._update(z, self.R_icp)

    def update_odom_imu(self, z: np.ndarray) -> None:
        """Measurement update from wheel odometry (x, y) + IMU (ψ).

        z : [3]  stacked observation  [x, y, ψ]  [m, m, rad]
            z[0:2] comes from odometry dead-reckoning
            z[2]   comes from IMU heading integration
        """
        self._update(z, self.R_odom_imu)

    def _update(self, z: np.ndarray, R: np.ndarray) -> None:
        """Shared EKF update for any sensor whose observation model is H = I₃.

        S  = P⁻ + R
        K  = P⁻ S⁻¹
        y  = z − x⁻          (innovation; ψ component wrapped to (−π, π])
        x  = x⁻ + K y
        P  = (I − K) P⁻
        """
        S = self.P + R
        # K = P S⁻¹; use solve(S, P)ᵀ to avoid explicit matrix inversion
        # valid because S is symmetric (P and R are both symmetric)
        K = np.linalg.solve(S, self.P).T

        y = z - self.x
        # wrap ψ innovation to (−π, π] so a 359° error is not treated as 359° [rad]
        y[2] = (y[2] + np.pi) % (2.0 * np.pi) - np.pi

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K) @ self.P
