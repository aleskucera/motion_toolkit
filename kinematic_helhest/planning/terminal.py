"""Terminal 'dock' controller -- the final-approach stage handed off from MPPI routing.

MPPI + cost-to-go routes the robot to within a radius R of the goal; from there a horizon-limited,
forward-only MPPI tends to OVERSHOOT (it never commands deceleration) and then CIRCLE (it can't
reverse), so it orbits the goal instead of stopping. The dock controller fixes both explicitly --
no sampling, no horizon:

  * decelerate: forward speed scales with distance, so the robot slows to a stop AT the goal,
  * align-then-drive: forward speed also scales with cos(heading error), so when the goal is off to
    the side the robot turns toward it (differential) before driving, instead of looping past it.

This is the separate terminal stage (vs patching the MPPI cost): routing and docking are different
control problems, so they get different controllers.
"""
import numpy as np


def dock_control(state, goal, dock_speed=2.0, slow_radius=1.5, wmax=4.0, turn_gain=3.0, turn_width=0.5):
    """state (x, y, yaw), goal (x, y) -> wheel command (wL, wR, rear) for one step.

    dock_speed: top forward speed for the final approach -- well below the routing wmax, since this
    is only the last metre or two and should be a gentle glide, not a charge. slow_radius: distance
    over which forward speed ramps from dock_speed down to a stop (>= the handoff radius so it
    decelerates the WHOLE approach). turn_gain/turn_width: how hard to steer toward the goal (the
    turn can still use the full wmax, so alignment stays crisp while the approach is slow)."""
    x, y, yaw = float(state[0]), float(state[1]), float(state[2])
    dx, dy = float(goal[0]) - x, float(goal[1]) - y
    dist = np.hypot(dx, dy)
    bearing = np.arctan2(dy, dx) - yaw
    bearing = (bearing + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]
    v = dock_speed * min(1.0, dist / slow_radius)        # gentle, decelerating-to-stop approach
    v *= max(0.0, np.cos(bearing))                       # only drive forward when ~facing the goal
    turn = turn_gain * bearing
    wl = float(np.clip(v - turn * turn_width, 0.0, wmax))
    wr = float(np.clip(v + turn * turn_width, 0.0, wmax))
    return np.array([wl, wr, 0.5 * (wl + wr)], np.float32)
