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
  2. asymmetric accel/decel rate limit  (a jumpy MPPI step can't shock the drivetrain)
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
    max_decel: float | None = None,
    turn_boost: float = 1.0,
    goal_dist: float | None = None,
    brake_dist: float = 0.0,
) -> np.ndarray:
    """Planner (wl, wr) -> conditioned [left, rear, right] wheel-velocity command for /cmd_joints.

    wl, wr: planner wheel speeds (model convention, >= 0; both positive = forward).
    prev: the previously PUBLISHED [left, rear, right] command. Pass zeros on the first call /
        after an e-stop so the slew limiter ramps up from rest.
    max_omega: hard cap on |wheel velocity| [rad/s] -- set to the motor's safe max.
    max_slew: cap on |d(command)/dt| for a joint SPEEDING UP [rad/s^2] (acceleration).
    max_decel: cap on |d(command)/dt| for a joint SLOWING toward rest [rad/s^2] (deceleration,
        incl. the stop ramp). None = use max_slew (symmetric limit, the old behaviour).
    goal_dist: current robot->goal distance [m]. With brake_dist > 0, scales the FORWARD speed down
        on the final approach (the goal brake below). None/0 disables the brake.
    brake_dist: [m] start braking within this range of the goal. 0 = no brake.
    Returns [left, rear, right] velocities to publish. To STOP, call with wl = wr = 0 -- the slew
    limiter ramps the command down to rest.
    """
    # Split into forward (mean) + turn (differential). The drivetrain realizes the forward command
    # 1:1 but only ~half the TURN differential (the two motors equalize under load; measured over
    # outdoor bags). turn_boost amplifies the commanded differential to compensate so the wheels
    # actually deliver the yaw MPPI intended -- forward speed (mean) is untouched. 1.0 = no boost;
    # ~2.0 recovers the measured ~0.5 realization. Tune in the field.
    # *** HOTFIX / stopgap for a drivetrain defect -- NOT a real fix. Read docs/turn_differential_hotfix.md
    #     before changing/removing this: what it papers over, and what to actually fix. ***
    mean = 0.5 * (wl + wr)  # forward speed; also the rear follower target (rear = mean of L/R)
    # GOAL BRAKE: the robot is forward-only (wmin=0) -- it cannot pivot in place to re-aim, so if it
    # arrives fast and slightly off it flies PAST the goal and orbits (a hard stop-radius misses an
    # offset flyby entirely). Scaling forward speed linearly to 0 over the last brake_dist metres
    # makes it nose in slow -> settles AT the goal. The turn differential is NOT scaled (tighter arc
    # at low speed helps the final aim) and the far-field cruise is untouched (no slow-down until
    # inside brake_dist). Verified in sim vs a term_v MPPI cost + a sqrt profile: this linear output
    # brake settled cleanest (~0.2 m, zero overshoot) across straight/offset/sharp goals.
    if brake_dist > 0.0 and goal_dist is not None:
        mean *= min(1.0, float(goal_dist) / float(brake_dist))
    diff = (wr - wl) * float(turn_boost)  # turn differential, amplified
    # /cmd_joints input convention: forward = all positive, no sign flip -- the LLC applies its own
    # internal signs. Verified: forward (wl=wr=v) -> [+v, +v, +v] drove the robot straight forward.
    target = np.array([mean - 0.5 * diff, mean, mean + 0.5 * diff], dtype=np.float32)
    prev = np.asarray(prev, dtype=np.float32)
    # Asymmetric rate limit: a joint speeding UP (|cmd| growing) is capped by max_slew (accel); a
    # joint slowing DOWN toward rest (|cmd| shrinking, incl. the stop ramp) by max_decel. Per-joint
    # because in a turn one wheel accelerates while the other decelerates. None = symmetric.
    d_acc = float(max_slew) * float(dt)
    d_dec = float(max_slew if max_decel is None else max_decel) * float(dt)
    lim = np.where(np.abs(target) >= np.abs(prev), d_acc, d_dec)  # per joint: accel vs decel cap
    cmd = prev + np.clip(target - prev, -lim, lim)  # rate limit
    cmd = np.clip(cmd, -float(max_omega), float(max_omega))  # hard magnitude backstop
    return cmd.astype(np.float32)
