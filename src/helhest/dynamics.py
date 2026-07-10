"""Canonical robot + solver dynamics: the SINGLE source of truth for the vehicle model.

The planner, the interactive driver, and the settle-feasibility sampler must all simulate the
*same* vehicle -- otherwise the plan describes a different robot than the one being driven (the
plan->real lag/clip bug came from exactly this: plan dt=0.1 vs drive dt=0.05). Everything that
builds a ForwardSimulator pulls its RobotParams / SolverParams / timestep from here instead of writing
its own, so the configs can't drift.

Two solver fidelities share the same dt and turn gain:
  planning_solver   -- the B-batch MPPI rollouts: fewer Newton iters (speed across thousands)
  execution_solver  -- the single driven / settled robot: more Newton iters (accuracy)
"""

from .engine import RobotParams
from .engine import SolverParams

DT = 0.1  # control timestep -- the plan horizon step AND the driver frame step (must match)
# skid-steer turn gain: alpha = 1 + K_TURN*mu sets the turn resistance (yaw rate = ideal / alpha).
# Calibrated 2026-07-10 against ICP-truth over real manual-drive bags (rotate_in_place0 + arc_diff0
# agree at alpha ~= 1.33 vs the old 2.6): the robot turns ~2x more than the old gain assumed. 0.4 gives
# 1 + 0.4*0.8 = 1.32 at the planner's mu=0.8. (Forward gain measured ~0.97 -> wheel_radius unchanged.
# The /cmd_joints command convention is all-positive-forward, same sign as this model -- no flip;
# see control/command.py and the wheel_sign_convention_calibration memory.)
K_TURN = 0.4


def robot_params():
    """The canonical robot geometry/mass model."""
    return RobotParams()


def planning_solver(dt=DT):
    """Solver for the MPPI rollouts (B in the thousands): shallow + loose settle, for speed."""
    return SolverParams(dt=dt, k_turn=K_TURN, newton_iters=6, atol=1e-4)


def execution_solver(dt=DT):
    """Solver for the single driven / settled robot: deeper settle for fidelity."""
    return SolverParams(dt=dt, k_turn=K_TURN, newton_iters=12, tilt_clamp=1.2)
