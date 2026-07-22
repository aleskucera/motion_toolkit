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

    def update_icp(self, z: np.ndarray, R: np.ndarray | None = None) -> None:
        """Measurement update from LiDAR ICP.

        z : [3]   observed pose  [x, y, ψ]  in world frame [m, m, rad]
        R : [3,3] measurement noise override; None uses stored R_icp
        """
        self._update(z, self.R_icp if R is None else R)

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


class EKF6D:
    """EKF for the 6-DOF Helhest state [x, y, ψ, ẋᵂ, ẏᵂ, ψ̇].

    State dimension: 6
        x, y  — world-frame position [m]
        ψ     — heading [rad]
        ẋ, ẏ  — world-frame linear velocity [m/s]
        ψ̇     — yaw rate [rad/s]

    The prediction is driven by the Helhest kinematic forward model, whose nonlinear
    output f(q, u) and linearised transition F = ∂f/∂q are supplied by the caller (see
    helhest.filtering.jacobian.predict_q6d / jacobian_F_6d) — the EKF never calls the
    model itself. Two measurement sources are supported, both observing [x, y, ψ] through
    H = [I₃ | 0₃] (position/heading measured, velocity states not):
        update_icp      — LiDAR ICP pose (stored R_icp)
        update_odom_imu — wheel odometry (x, y) + IMU heading (ψ); uses stored R_odom
    """

    # Position states are observed, velocity states are not: z = H q with H = [I₃ | 0₃].
    H: np.ndarray = np.hstack([np.eye(3), np.zeros((3, 3))])

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R_icp: np.ndarray,
        R_odom: np.ndarray,
    ) -> None:
        """
        x0     : [6]    initial state  [x, y, ψ, ẋ, ẏ, ψ̇]
        P0     : [6,6]  initial state covariance
        Q      : [6,6]  process noise covariance
        R_icp  : [3,3]  ICP measurement noise covariance (on [x, y, ψ])
        R_odom : [3,3]  odom+IMU measurement noise covariance (on [x, y, ψ])
        """
        self.x: np.ndarray = x0.copy()
        self.P: np.ndarray = P0.copy()
        self.Q: np.ndarray = Q.copy()
        self.R_icp: np.ndarray = R_icp.copy()
        self.R_odom: np.ndarray = R_odom.copy()

    def predict(self, F: np.ndarray, x_pred: np.ndarray, q_scale: float = 1.0) -> None:
        """Propagate the filter by one timestep.

        F       : [6,6] linearised state-transition matrix ∂f/∂q at (q, u) — from caller
        x_pred  : [6]   nonlinear model prediction f(q, u) — also from caller
        q_scale : process-noise time scale — pass (actual_dt / model_DT) so P grows
                  linearly with real elapsed time (random-walk Q model). Default 1.0
                  keeps the original behaviour when dt is not measured.
        """
        self.x = x_pred.copy()
        self.P = F @ self.P @ F.T + q_scale * self.Q

    def update_icp(self, z: np.ndarray, R: np.ndarray | None = None) -> None:
        """Measurement update from a LiDAR ICP pose.

        z : [3]   observed pose  [x, y, ψ]  in world frame [m, m, rad]
        R : [3,3] measurement noise override; None uses stored R_icp
        """
        self._update(z, self.R_icp if R is None else R)

    def update_odom_imu(self, z: np.ndarray) -> None:
        """Measurement update from wheel odometry (x, y) + IMU heading (ψ).

        z : [3]  observed pose  [x, y, ψ]  from dead-reckoning [m, m, rad]
            z[0:2] comes from integrated wheel-encoder translation
            z[2]   comes from gyro / IMU heading integration
        """
        self._update(z, self.R_odom)

    def _update(self, z: np.ndarray, R: np.ndarray) -> None:
        """Shared EKF update for both [x, y, ψ] sensors (H = [I₃ | 0₃]).

        S = H P⁻ Hᵀ + R
        K = P⁻ Hᵀ S⁻¹
        y = z − H x⁻          (innovation; ψ component wrapped to (−π, π])
        x = x⁻ + K y
        P = (I₆ − K H) P⁻
        """
        H = self.H
        S = H @ self.P @ H.T + R
        # K = P Hᵀ S⁻¹; solve(S, (P Hᵀ)ᵀ)ᵀ avoids explicit inversion (S symmetric).
        K = np.linalg.solve(S, (self.P @ H.T).T).T

        y = z - H @ self.x
        # wrap ψ innovation to (−π, π] so a 359° error is not treated as 359° [rad]
        y[2] = (y[2] + np.pi) % (2.0 * np.pi) - np.pi

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
