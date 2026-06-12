"""Drive the kinematic twin over a heightmap and collect the trajectory.

`rollout_terrain` settles an initial pose into a valid State (state.py) and
applies the physics step for each recorded wheel-speed command.
"""
import numpy as np

from .model import HALF_TRACK
from .model import WHEEL_RADIUS


def rollout_terrain(setpoints, dt, hm, alpha=1.0, x_icr=0.0, init_pose=(0.0, 0.0, 0.0),
                    mu_field=None, k=2.0, R=WHEEL_RADIUS, b=HALF_TRACK):
    """Roll out on a heightmap by repeated `state.step` (predict->project).

    Settles the initial pose into a valid State, then applies the physics step T
    times. pose7[t] is the valid (settled, non-penetrative) pose AFTER command t
    — the end-of-step state. Per-step turning params (alpha, x_ICR) come from the
    friction sampled at the contacts + the normal loads of the state each step
    starts FROM (Phase 4 map, coefficient `k`); without `mu_field` the scalar
    args are used.

    Chassis non-penetration is a post-check only (the settle stays wheels-only):
    `valid` is False if the belly high-centers at ANY step, so a planner can
    simply reject the whole trajectory. `first_high_center` is the first such
    step (-1 if none).

    Returns dict of arrays over T: pose7 [T,7], pose2 [T,3] (x,y,yaw),
    loads [T,3] (N_i), fz [T], chassis_clear [T], high_center [T],
    alpha [T], x_icr [T], pitch/roll/residual [T]; plus scalars
    valid (bool) and first_high_center (int). Single rollout (no batch).
    """
    from . import heightmap as _hm
    from . import state as _state
    setpoints = np.asarray(setpoints, dtype=np.float64)
    T = setpoints.shape[0]

    surf = _hm.wheel_envelope(hm, R)  # sphere-wheel placement surface
    x0, y0, yaw0 = (float(v) for v in init_pose)
    st = _state.make_state(x0, y0, yaw0, surf, hm)  # valid initial state

    pose7 = np.empty((T, 7), dtype=np.float32)
    pose2 = np.empty((T, 3), dtype=np.float64)
    loads = np.empty((T, 3), dtype=np.float64)
    fz = np.empty(T); chassis_clear = np.empty(T)
    alpha_log = np.empty(T); xicr_log = np.empty(T)
    pitch = np.empty(T); roll = np.empty(T); resid = np.empty(T)

    for t in range(T):
        st = _state.step(st, setpoints[t], surf, hm, dt,
                         mu_field=mu_field, k=k, alpha=alpha, x_icr=x_icr, R=R, b=b)
        pose7[t] = st.pose7
        pose2[t] = st.pose2
        loads[t] = st.loads
        fz[t] = st.fz
        chassis_clear[t] = st.chassis_clear
        alpha_log[t] = st.alpha; xicr_log[t] = st.x_icr
        pitch[t] = st.place["pitch"]; roll[t] = st.place["roll"]; resid[t] = st.place["residual"]

    high_center = chassis_clear < 0.0
    return {"pose7": pose7, "pose2": pose2, "loads": loads, "fz": fz,
            "chassis_clear": chassis_clear, "high_center": high_center,
            "valid": not bool(high_center.any()),
            "first_high_center": int(np.argmax(high_center)) if high_center.any() else -1,
            "alpha": alpha_log, "x_icr": xicr_log,
            "pitch": pitch, "roll": roll, "residual": resid}


def cruise_decomposition(pose2, setpoints, dt, x_max=0.9, t_min=0.3, R=WHEEL_RADIUS):
    """Flat-ground cruise check (pre-box window). Mirrors replay_real's metric.

    Returns dict with commanded wheel speed, no-slip ground speed (= w*R), and
    the realized ground speed from the integrated path (== w*R by construction,
    since the kinematic model has zero slip)."""
    x = pose2[:, 0]
    t = np.arange(len(x)) * dt
    win = (t > t_min) & (x < x_max)
    if win.sum() < 5:
        return None
    cmd = np.abs(setpoints[win]).mean()
    speed = np.linalg.norm(np.diff(pose2[:, :2], axis=0), axis=1)[win[:-1]] / dt
    return {
        "commanded_wheel_speed": float(cmd),
        "noslip_ground_speed": float(cmd * R),
        "ground_speed": float(np.median(speed)),
        "window_steps": int(win.sum()),
    }
