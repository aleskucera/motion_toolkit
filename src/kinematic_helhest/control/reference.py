"""Numpy reference for the MPPI cost + control packing (the GPU planner lives in mppi).

`_cost` is the readable, independent ORACLE that the GPU `_cost` kernel is differential-tested
against (tests/control/test_mppi.py): it defines, in plain numpy, what the per-rollout cost
MEANS, so a kernel that diverges from intent is caught even when the robot still roughly reaches
the goal. `_to_wheel_omega` packs the [B, T, 2] wheel-speed controls into the engine's [T, B, 3] layout.
"""

import numpy as np


def _to_wheel_omega(Ub):
    """Controls [B, T, 2] (wL, wR) -> wheel_omega [T, B, 3] (rear = mean)."""
    bt = np.transpose(Ub, (1, 0, 2))  # [T, B, 2]
    rear = bt.mean(2, keepdims=True)
    return np.concatenate([bt, rear], axis=2).astype(np.float32)


def _cost(controlled, derived, clear, resid, Ub, lattice_const, clear_margin, resid_tol, w):
    """Per-rollout cost [B]. `derived` is [T+1, B, 3] = (z, pitch, roll).

    The goal cost is the orientation-aware cost-to-go V(x,y,theta)^2. The parity test uses a CONSTANT
    field (value `lattice_const`), so V is that constant everywhere and the trilinear sample need not
    be replicated here -- the routing path is verified end to end. The stability-envelope limits
    (max_roll/pitch) come via `w`, matching the Robot struct the GPU kernel reads.
    """
    vc2 = lattice_const * lattice_const  # V^2, constant across the (constant) cost-to-go field
    # graded validity (option C): how far past margin/tol, weighted by how early (T,B -> B)
    T = clear.shape[0]
    early = ((T - np.arange(T)) / T)[:, None]  # [T, 1]
    clear_viol = np.maximum(clear_margin - clear, 0.0)  # [T, B]
    resid_viol = np.maximum(resid - resid_tol, 0.0)  # [T, B]
    # robot stability envelope: tipping is invalid. climbing is nose-up = NEGATIVE pitch, so the climb
    # limit is on -pitch.
    pitch, roll = derived[:T, :, 1], derived[:T, :, 2]  # [T, B]
    roll_viol = np.maximum(np.abs(roll) - w["max_roll"], 0.0)
    climb_viol = np.maximum(-pitch - w["max_pitch_up"], 0.0)
    descend_viol = np.maximum(pitch - w["max_pitch_down"], 0.0)
    inv = (early * (clear_viol + resid_viol + roll_viol + climb_viol + descend_viol)).sum(0)  # [B]
    eff = (Ub**2).sum((1, 2))
    smooth = (np.diff(Ub, axis=1) ** 2).sum((1, 2))
    J = (
        (w["goal_terminal"] + w["goal_running"]) * vc2
        + w["effort"] * eff
        + w["smoothness"] * smooth
    )
    return J + inv * w["infeasible"], inv > 0
