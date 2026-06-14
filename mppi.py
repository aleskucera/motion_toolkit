"""Minimal MPPI planner on the kinematic Warp engine (Phase 8, sampling-based).

Receding-horizon MPPI: each control cycle samples B noisy wheel-speed sequences,
rolls them all out in one batched launch, costs them (goal distance + a hard
validity penalty that does the obstacle/high-center avoidance), reweights with a
softmax, updates the nominal sequence, executes its first control, shifts, repeats.

No gradients -- pure forward sampling. The validity flags (high-center /
infeasible settle, with the tunable resid_tol / clear_margin / tilt_clamp) are
what steer the robot around the wall.

Demo:  python -m kinematic_helhest.mppi [--device cuda] [--out plan.png]
"""
import argparse

import numpy as np
import warp as wp

from . import friction
from . import heightmap as hmmod
from .drive import demo_terrain
from .warp_engine.kinematics import init_state
from .warp_engine.kinematics import step as wstep
from .warp_engine.solver import RobotParams
from .warp_engine.solver import SolverParams
from .warp_engine.terrain import to_terrain


class BatchRollout:
    """Persistent device buffers for B rollouts of horizon T on a fixed scene."""

    def __init__(self, scene, mu, B, T, params, device="cuda", robot_params=None):
        wp.init()
        rp = robot_params or RobotParams()
        self.robot = rp.build(device)
        self.sp = params.build()
        Rw = rp.wheel_radius
        self.te = to_terrain(hmmod.wheel_envelope(scene, Rw), device)
        self.tr = to_terrain(scene, device)
        self.tm = to_terrain(mu, device)
        self.B, self.T, self.dev = B, T, device
        self.planar = wp.zeros((T + 1, B), dtype=wp.vec3, device=device)
        self.tilt = wp.zeros((T + 1, B), dtype=wp.vec3, device=device)
        self.loads = wp.zeros((T, B), dtype=wp.vec3, device=device)
        self.turn = wp.zeros((T, B), dtype=wp.vec2, device=device)
        self.clear = wp.zeros((T, B), dtype=float, device=device)
        self.resid = wp.zeros((T, B), dtype=float, device=device)

    def rollout(self, omega_np, init_pose):
        """omega_np [T, B, 3], init_pose (x,y,yaw) shared by all rollouts.
        Returns planar [T+1, B, 3], clear [T, B], resid [T, B] (numpy)."""
        B, T, dev = self.B, self.T, self.dev
        omega = wp.array(np.ascontiguousarray(omega_np, np.float32), dtype=wp.vec3, device=dev)
        pose0 = wp.array(np.tile(np.asarray(init_pose, np.float32), (B, 1)), dtype=wp.vec3, device=dev)
        wp.launch(init_state, B, inputs=[self.te.H, self.te.g, self.robot, self.sp, pose0],
                  outputs=[self.planar, self.tilt], device=dev)
        for t in range(T):
            wp.launch(wstep, B,
                      inputs=[t, self.te.H, self.tr.H, self.te.g, self.tm.H, self.tm.g,
                              self.robot, self.sp, omega],
                      outputs=[self.planar, self.tilt, self.loads, self.turn, self.clear, self.resid],
                      device=dev)
        return self.planar.numpy(), self.clear.numpy(), self.resid.numpy()


def _to_omega(Ub):
    """Controls [B, T, 2] (wL, wR) -> omega [T, B, 3] (rear = mean)."""
    bt = np.transpose(Ub, (1, 0, 2))  # [T, B, 2]
    rear = bt.mean(2, keepdims=True)
    return np.concatenate([bt, rear], axis=2).astype(np.float32)


def _cost(planar, clear, resid, Ub, goal, clear_margin, resid_tol, w):
    """Per-rollout cost [B]. goal [2]."""
    xy = planar[:, :, :2]                                  # [T+1, B, 2]
    d = np.linalg.norm(xy - goal[None, None, :], axis=2)   # [T+1, B]
    invalid = (clear < clear_margin).any(0) | (resid > resid_tol).any(0)  # [B]
    eff = (Ub ** 2).sum((1, 2))
    smooth = (np.diff(Ub, axis=1) ** 2).sum((1, 2))
    J = w["term"] * d[-1] ** 2 + w["run"] * (d ** 2).mean(0) + w["eff"] * eff + w["smooth"] * smooth
    return J + invalid.astype(np.float64) * w["invalid"], invalid


def plan(scene, mu, start, goal, T=70, B=2048, n_refine=3, max_steps=260, dt=0.05,
         sigma=2.5, lam=0.5, wmax=4.0, goal_tol=0.3, resid_tol=1e-2, clear_margin=0.05,
         device="cuda", seed=0, weights=None):
    params = SolverParams(dt=dt, k_turn=2.0, newton_iters=12)
    br = BatchRollout(scene, mu, B, T, params, device=device)
    w = weights or dict(term=3.0, run=0.3, invalid=1e5, eff=2e-3, smooth=2e-3)
    goal = np.asarray(goal[:2], np.float64)
    rng = np.random.default_rng(seed)

    U = np.full((T, 2), 1.5, np.float32)        # nominal wheel speeds, gentle forward
    state = np.asarray(start, np.float32)        # (x, y, yaw)
    path = [state.copy()]
    reached = False
    for k in range(max_steps):
        if np.linalg.norm(state[:2] - goal) < goal_tol:
            reached = True
            break
        for _ in range(n_refine):
            eps = rng.normal(0.0, sigma, (B, T, 2)).astype(np.float32)
            eps[0] = 0.0                          # keep the nominal as a sample
            Ub = np.clip(U[None] + eps, -wmax, wmax)
            planar, clear, resid = br.rollout(_to_omega(Ub), state)
            J, _ = _cost(planar, clear, resid, Ub, goal, clear_margin, resid_tol, w)
            beta = np.exp(-(J - J.min()) / lam)
            beta /= beta.sum()
            U = np.clip(np.einsum("b,btc->tc", beta, Ub), -wmax, wmax).astype(np.float32)
        # execute first control: roll the nominal out and take the step-1 pose
        planar, clear, resid = br.rollout(_to_omega(np.tile(U, (B, 1, 1))), state)
        state = planar[1, 0].astype(np.float32).copy()
        path.append(state.copy())
        U = np.roll(U, -1, axis=0)
        U[-1] = U[-2]
    return np.array(path), reached


def _plot(scene, path, start, goal, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nx, ny = scene.nx, scene.ny
    ext = [scene.x0, scene.x0 + (nx - 1) * scene.cell, scene.y0, scene.y0 + (ny - 1) * scene.cell]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(scene.H, origin="lower", extent=ext, cmap="terrain", alpha=0.9)
    ax.plot(path[:, 0], path[:, 1], "-", color="orange", lw=2.5, label="MPPI path")
    ax.plot(*start[:2], "o", color="white", mec="k", ms=10, label="start")
    ax.plot(*goal[:2], "*", color="red", ms=18, label="goal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(loc="upper left")
    ax.set_title("MPPI on kinematic engine — detour around the wall")
    ax.axis("equal"); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/mppi.png")
    ap.add_argument("--gx", type=float, default=4.0)
    ap.add_argument("--gy", type=float, default=1.5)
    ap.add_argument("--B", type=int, default=2048)
    args = ap.parse_args()

    scene = demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    start = np.array([0.0, 0.0, 0.0], np.float32)
    goal = np.array([args.gx, args.gy], np.float64)
    path, reached = plan(scene, mu, start, goal, B=args.B, device=args.device)
    d = np.linalg.norm(path[-1, :2] - goal)
    print(f"reached={reached}  final=({path[-1,0]:+.2f},{path[-1,1]:+.2f})  "
          f"dist_to_goal={d:.2f}m  steps={len(path)-1}")
    _plot(scene, path, start, goal, args.out)


if __name__ == "__main__":
    main()
