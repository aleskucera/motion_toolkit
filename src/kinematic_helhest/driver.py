"""The 'real robot': a single-vehicle stepper on the Warp engine, one settle per control step.

This is the execution model the planner is steering -- distinct from the B-batch planning ForwardSimulator.
It lives in core (not viz) so headless code (the eval harness, tests) can drive a robot without
pulling in OpenGL. The interactive viewer and the eval loop both step this same object, so what you
watch on screen and what we measure are the same vehicle.
"""

from types import SimpleNamespace

import numpy as np
import warp as wp

from . import dynamics
from .engine import ForwardSimulator
from .engine import GridParams
from .engine import SolverParams
from .model import euler_zyx


class WarpDriver:
    """Wraps a B=1, T=1 `ForwardSimulator` and the current pose; steps one frame per call."""

    def __init__(
        self,
        hm,
        mu,
        init_pose=(0.0, 0.0, 0.0),
        device="cpu",
        dt=dynamics.DT,
        k_turn=dynamics.K_TURN,
        resid_tol=1e-2,
        clear_margin=0.0,
        tilt_clamp=1.2,
    ):
        wp.init()
        self.resid_tol, self.clear_margin = resid_tol, clear_margin
        sp = SolverParams(dt=dt, k_turn=k_turn, newton_iters=12, tilt_clamp=tilt_clamp)
        self.sim = ForwardSimulator(
            dynamics.robot_params(),
            sp,
            GridParams(hm.nx, hm.ny, hm.cell, hm.x0, hm.y0),
            1,
            1,
            device,
        )
        self.sim.set_terrain(
            wp.array(np.ascontiguousarray(hm.H, np.float32), dtype=wp.float32, device=device)
        )
        self.sim.set_friction(mu)

        # frame 0: settle at the start pose (zero control)
        controlled, derived, _, _ = self.sim.rollout(np.zeros((1, 1, 3), np.float32), init_pose)
        self.controlled = controlled[0, 0].copy()  # (x, y, yaw)
        self.derived = derived[0, 0].copy()  # (z, pitch, roll)
        self.clear, self.alpha, self.resid = 1.0, 1.0, 0.0

    def step(self, wheel_omega):
        wheel_omega = np.asarray(wheel_omega, np.float32).reshape(1, 1, 3)
        controlled, derived, clear, resid = self.sim.rollout(wheel_omega, self.controlled)
        self.controlled = controlled[1, 0].copy()
        self.derived = derived[1, 0].copy()
        self.clear = float(clear[0, 0])
        self.resid = float(resid[0, 0])
        self.alpha = float(self.sim.turning.numpy()[0, 0][0])

    def render_state(self):
        x, y, yaw = (float(v) for v in self.controlled)
        z, pitch, roll = (float(v) for v in self.derived)
        R = euler_zyx(yaw, pitch, roll)
        valid = self.clear >= self.clear_margin and self.resid < self.resid_tol
        return SimpleNamespace(
            x=x,
            y=y,
            yaw=yaw,
            alpha=self.alpha,
            valid=valid,
            place={"z": z, "R": R, "pitch": pitch, "roll": roll},
        )
