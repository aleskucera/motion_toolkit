"""Map a planner wheel command to the real robot's /cmd_joints, with motor-safety conditioning.

The planner works in a self-consistent "both wheels >= 0 = forward" convention (the MPPI box has
wmin=0). The robot's /cmd_joints INPUT convention is the same: forward = all wheels POSITIVE
([+wl, +rear, +wr]), so the command passes straight through -- no sign flip. Verified live on the
robot 2026-07-10 (an all-positive /cmd_joints drove straight forward).

  NOTE the LLC's /joint_setpoint OUTPUT echoes forward as [-, -, +] (it negates left/rear
  internally). That output convention is what the manual-drive bags recorded, but it does NOT
  apply to commands -- the LLC does its own internal sign mapping on the /cmd_joints input.

This is the single place all actuator-safety logic lives, so it is auditable and unit-tested:
  1. rear-as-follower                               (rear = mean(L, R); left/right pass through)
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
    prev: the previously PUBLISHED [left, rear, right] command. Pass zeros on the first call /
        after an e-stop so the slew limiter ramps up from rest.
    max_omega: hard cap on |wheel velocity| [rad/s] -- set to the motor's safe max.
    max_slew: hard cap on |d(command)/dt| per joint [rad/s^2] -- limits the change over one dt.
    Returns [left, rear, right] velocities to publish. To STOP, call with wl = wr = 0 -- the slew
    limiter ramps the command down to rest.
    """
    rear = 0.5 * (wl + wr)  # rear rolls as a follower at the body forward speed (mean of L/R)
    # /cmd_joints input convention: forward = all positive, so left/right pass straight through
    # (rear = mean). No sign flip -- the LLC applies its own internal signs. Verified: forward
    # (wl=wr=v) -> [+v, +v, +v] drove the robot straight forward. See wheel_sign_convention_calibration.
    target = np.array([wl, rear, wr], dtype=np.float32)
    prev = np.asarray(prev, dtype=np.float32)
    d = float(max_slew) * float(dt)  # max change per joint this step
    cmd = np.clip(target, prev - d, prev + d)  # slew-rate limit
    cmd = np.clip(cmd, -float(max_omega), float(max_omega))  # hard magnitude backstop
    return cmd.astype(np.float32)
