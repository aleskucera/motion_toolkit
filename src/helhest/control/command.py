"""Map a planner wheel command to the real robot's /cmd_joints, with motor-safety conditioning.

The planner works in a self-consistent "both wheels >= 0 = forward" convention (the MPPI box has
wmin=0). The REAL robot's LEFT (and rear) drive-wheel joint sign is INVERTED -- the wheels are
mirror-mounted, so forward is (left < 0, right > 0). Calibrated 2026-07-10 against ICP truth over
manual-drive bags (see the wheel_sign_convention_calibration memory). Sending the planner's raw
command would make the robot SPIN in place instead of driving forward, so the sign flip is
MANDATORY.

This is the single place all actuator-safety logic lives, so it is auditable and unit-tested:
  1. sign flip + rear-as-follower   (left/rear negated, right as-is; rear = mean(L, R))
  2. slew-rate limit vs the last published command  (a jumpy MPPI step can't shock the drivetrain)
  3. a hard per-joint magnitude clamp               (final backstop below the motor's safe max)
"""
from __future__ import annotations

import numpy as np

# /cmd_joints JointState.name order -- matches /joint_setpoint on the robot. The [left, rear, right]
# arrays returned by condition_command are in THIS order.
JOINT_NAMES = ("left_wheel_j", "rear_wheel_j", "right_wheel_j")


def condition_command(
    wl: float,
    wr: float,
    prev: np.ndarray,
    *,
    max_omega: float,
    max_slew: float,
    dt: float,
) -> np.ndarray:
    """Planner (wl, wr) -> conditioned [left, rear, right] wheel-velocity command for /cmd_joints.

    wl, wr: planner wheel speeds (model convention, >= 0; both positive = forward).
    prev: the previously PUBLISHED [left, rear, right] command (real-robot convention). Pass zeros
        on the first call / after an e-stop so the slew limiter ramps up from rest.
    max_omega: hard cap on |wheel velocity| [rad/s] -- set to the motor's safe max.
    max_slew: hard cap on |d(command)/dt| per joint [rad/s^2] -- limits the change over one dt.
    Returns [left, rear, right] velocities to publish (real-robot convention). To STOP, call with
    wl = wr = 0 -- the slew limiter ramps the command down to rest.
    """
    rear = 0.5 * (wl + wr)  # rear rolls as a follower at the body forward speed (mean of L/R)
    # Real-robot convention: the left + rear joints are mirror-mounted (their + spins the robot
    # backward), so negate them; the right joint is as-is. This makes the robot execute the
    # planner's intended forward/turn. Verified: forward (wl=wr=v) -> [-v, -v, +v], the exact
    # command shape the forward0 bag drove. See wheel_sign_convention_calibration.
    target = np.array([-wl, -rear, wr], dtype=np.float32)
    prev = np.asarray(prev, dtype=np.float32)
    d = float(max_slew) * float(dt)  # max change per joint this step
    cmd = np.clip(target, prev - d, prev + d)  # slew-rate limit
    cmd = np.clip(cmd, -float(max_omega), float(max_omega))  # hard magnitude backstop
    return cmd.astype(np.float32)
