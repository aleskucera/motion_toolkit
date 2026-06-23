"""Minimal MPPI planner on the kinematic Warp engine (Phase 8, sampling-based).

Receding-horizon MPPI: each control cycle samples B noisy wheel-speed sequences,
rolls them all out in one batched launch, costs them (goal distance + a hard
validity penalty that does the obstacle/high-center avoidance), reweights with a
softmax, updates the nominal sequence, executes its first control, shifts, repeats.

No gradients -- pure forward sampling. The validity flags (high-center /
infeasible settle, with the tunable resid_tol / clear_margin / tilt_clamp) are
what steer the robot around the wall.

Demo:  python -m kinematic_helhest.planning.mppi [--device cuda] [--out plan.png]
"""
import argparse

import numpy as np
import warp as wp

from .. import dynamics
from .. import friction
from .. import heightmap as hmmod
from ..engine import GridParams
from ..engine import Simulator
from .mppi_gpu import MppiGpu


def _to_omega(Ub):
    """Controls [B, T, 2] (wL, wR) -> omega [T, B, 3] (rear = mean)."""
    bt = np.transpose(Ub, (1, 0, 2))  # [T, B, 2]
    rear = bt.mean(2, keepdims=True)
    return np.concatenate([bt, rear], axis=2).astype(np.float32)


def _cost(controlled, derived, clear, resid, Ub, goal, clear_margin, resid_tol, w):
    """Per-rollout cost [B]. goal [2]. `derived` is [T+1, B, 3] = (z, pitch, roll).

    `w["tilt"]` (optional) penalizes the robot's total tilt from vertical along the
    rollout — the traversability term. The settle gives the true body pitch/roll for
    each candidate trajectory, so this is trajectory-aware (a diagonal slope crossing
    rolls differently than a straight climb), not a static per-cell terrain slope.
    """
    xy = controlled[:, :, :2]                                  # [T+1, B, 2]
    d = np.linalg.norm(xy - goal[None, None, :], axis=2)   # [T+1, B]
    # graded validity (option C): how far past margin/tol, weighted by how early (T,B -> B)
    T = clear.shape[0]
    early = ((T - np.arange(T)) / T)[:, None]              # [T, 1]
    clear_viol = np.maximum(clear_margin - clear, 0.0)     # [T, B]
    resid_viol = np.maximum(resid - resid_tol, 0.0)        # [T, B]
    # robot stability envelope (same limits as the cost-to-go): tipping is invalid. climbing is
    # nose-up = NEGATIVE pitch, so the climb limit is on -pitch. Large default = inactive.
    pitch, roll = derived[:T, :, 1], derived[:T, :, 2]     # [T, B]
    roll_viol = np.maximum(np.abs(roll) - w.get("max_roll", 1e3), 0.0)
    climb_viol = np.maximum(-pitch - w.get("max_pitch_up", 1e3), 0.0)
    descend_viol = np.maximum(pitch - w.get("max_pitch_down", 1e3), 0.0)
    inv = (early * (clear_viol + resid_viol + roll_viol + climb_viol + descend_viol)).sum(0)  # [B]
    eff = (Ub ** 2).sum((1, 2))
    smooth = (np.diff(Ub, axis=1) ** 2).sum((1, 2))
    J = w["term"] * d[-1] ** 2 + w["run"] * (d ** 2).mean(0) + w["eff"] * eff + w["smooth"] * smooth
    if w.get("tilt", 0.0) > 0.0:
        # total tilt from vertical: angle of body-z off world-z = arccos(cos p cos r)
        cpr = np.cos(derived[:, :, 1]) * np.cos(derived[:, :, 2])
        ang = np.arccos(np.clip(cpr, -1.0, 1.0))          # [T+1, B] radians
        # deadzone: tilt below `tilt_free` is free (drivable ramps), so the robot
        # still climbs gentle slopes to reach a goal; only steep tilt is penalized.
        over = np.maximum(ang - w.get("tilt_free", 0.0), 0.0)
        J = J + w["tilt"] * (over ** 2).mean(0)
    if w.get("head", 0.0) > 0.0:
        # heading: penalize facing away from the goal (1 - cos angle); drives the U-turn
        dx = controlled[:, :, 0] - goal[0]; dy = controlled[:, :, 1] - goal[1]  # [T+1, B]
        dist = np.hypot(dx, dy)
        cos_align = -(np.cos(controlled[:, :, 2]) * dx + np.sin(controlled[:, :, 2]) * dy) / np.maximum(dist, 1e-3)
        head = np.where(dist > 1e-3, 1.0 - cos_align, 0.0).mean(0)  # [B]
        J = J + w["head"] * head
    return J + inv * w["invalid"], inv > 0


def plan(scene, mu, start, goal, T=60, B=8192, n_refine=3, max_steps=260, dt=0.1,
         sigma=0.5, sigma_knot=1.0, n_knots=4, wmax=4.0, wmin=0.0, elite_frac=0.02, goal_tol=0.3,
         device="cuda", seed=0, weights=None, record=False, n_show=60, costtogo=False, lattice=False,
         n_scenarios=1, cvar_beta=0.5, slip_lo=0.6, n_theta=16, lat_robot_radius=0.3,
         trav_config=None, obstacle_threshold=0.8, tilt=0.0, tilt_free=0.0, lat_trav_weight=0.0,
         lat_feasibility="traversability", dock_radius=None, lat_coarsen=1):
    sim = Simulator(
        dynamics.robot_params(), dynamics.planning_solver(dt=dt),
        GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0),
        B, T, device,
    )
    sim.set_terrain(wp.array(np.ascontiguousarray(scene.H, np.float32),
                             dtype=wp.float32, device=device))
    sim.set_friction(mu)
    w = weights or dict(term=3.0, run=0.3, head=2.0, invalid=1e5, eff=2e-3, smooth=2e-3)
    if lattice:  # orientation-aware cost-to-go V(x,y,theta): routing + feasibility only --
        w = {**w, "lattice": 1.0}  # the terminal dock handles reach+stop, so no endgame cost-patches
        if weights is None:
            w["head"] = 0.0    # V(x,y,theta) already encodes the desired heading
            w["oob"] = 50.0    # soft wall at the grid edge (routing safety, not endgame)
            rp = dynamics.robot_params()  # rollouts share the robot's tip-over envelope with the cost-to-go
            w["max_roll"] = rp.max_roll
            w["max_pitch_up"] = rp.max_pitch_up
            w["max_pitch_down"] = rp.max_pitch_down
    elif costtogo:  # option E: score by obstacle-aware cost-to-go instead of straight-line distance
        w = {**w, "ctg": 1.0}
        if weights is None:
            w["head"] = 4.0   # the -grad V heading is a softer signal than Euclidean -> commit harder
            w["oob"] = 50.0   # soft wall at the grid edge: V is clamped off-grid, so the goal term
            w["term_v"] = 1.0  # alone lets it drive off the map; and end the plan stopped at the goal
    if tilt > 0.0:  # per-rollout tilt cost: penalize body tilt past tilt_free [rad] along the trajectory
        w = {**w, "tilt": float(tilt), "tilt_free": float(tilt_free)}
    goal = np.asarray(goal[:2], np.float64)
    rp = dynamics.robot_params()  # feasibility thresholds live on the robot, not here
    clear_margin, resid_tol = rp.clear_margin, rp.resid_tol
    drv = MppiGpu(sim, sigma, wmax, w, clear_margin, resid_tol, seed,
                  sigma_knot=sigma_knot, n_knots=n_knots, wmin=wmin, elite_frac=elite_frac,
                  n_scenarios=n_scenarios, cvar_beta=cvar_beta, slip_lo=slip_lo, n_theta=n_theta)
    drv.reset_nominal(1.5)  # nominal wheel speeds, gentle forward
    if lattice:  # solve the orientation-aware V(x,y,theta) once (fixed goal), before any replan
        # the routing field can be solved COARSE (k>1): the rollouts do fine obstacle avoidance, so the
        # global router needn't be sim-resolution. ~k^3 cheaper -> fast enough to re-solve a moving goal.
        k = max(1, int(lat_coarsen))
        if k > 1:
            cny, cnx, ccell = scene.ny // k, scene.nx // k, scene.cell * k
            Hc = scene.H[:cny * k, :cnx * k].reshape(cny, k, cnx, k).max(axis=(1, 3))  # max-pool keeps thin walls
        else:
            cny, cnx, ccell, Hc = scene.ny, scene.nx, scene.cell, scene.H
        Hc = np.ascontiguousarray(Hc, np.float32)
        cgrid = GridParams(cnx, cny, ccell, scene.x0, scene.y0)
        if lat_feasibility == "settle":  # feasibility from the robot's own settle, not a traversability threshold
            from .costtogo_settle import CostToGoLatticeSettle
            clat = CostToGoLatticeSettle(cgrid, dynamics.robot_params(), dynamics.planning_solver(dt=dt), sim.device,
                                         n_theta=n_theta, flatness_weight=lat_trav_weight)
        else:
            from .costtogo import CostToGoLattice
            clat = CostToGoLattice(cgrid, sim.device,
                                   n_theta=n_theta, turn_radius=dynamics.robot_params().min_turn_radius,
                                   robot_radius=lat_robot_radius,
                                   obstacle_threshold=obstacle_threshold, trav_weight=lat_trav_weight,
                                   config=trav_config)
        drv.set_lattice(clat.compute(Hc, goal), cgrid.build())
    elif costtogo:  # goal is fixed for the whole drive -> solve V(x,y) once, before any replan
        from .costtogo import CostToGo
        ctg = CostToGo(GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0), sim.device,
                       obstacle_threshold=obstacle_threshold, config=trav_config)
        drv.set_costtogo(ctg.compute(np.ascontiguousarray(scene.H, np.float32), goal))

    # terminal dock replaces the endgame cost-patches; on by default for lattice (pass 0 to disable)
    if dock_radius is None:
        dock_radius = 1.5 if lattice else 0.0
    dock_sim = None
    if dock_radius > 0.0:  # terminal stage: a B=1 sim to execute the dock control near the goal
        from .terminal import dock_control
        dock_sim = Simulator(dynamics.robot_params(), dynamics.planning_solver(dt=dt),
                             GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0), 1, 1, device)
        dock_sim.set_terrain(wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device))
        dock_sim.set_friction(mu)

    state = np.asarray(start, np.float32)        # (x, y, yaw)
    path = [state.copy()]
    idx = np.linspace(0, B - 1, n_show).astype(int)  # sampled rollouts to display
    frames = []
    reached = False
    for k in range(max_steps):
        if np.linalg.norm(state[:2] - goal) < goal_tol:
            reached = True
            break
        if dock_sim is not None and np.linalg.norm(state[:2] - goal) < dock_radius:
            # hand off from MPPI routing to the terminal dock: decelerate + align to a precise stop
            omega = dock_control(state, goal, wmax=wmax)
            cc, _, _, _ = dock_sim.rollout(omega.reshape(1, 1, 3), state)
            state = cc[1, 0].astype(np.float32).copy()
            path.append(state.copy())
            continue
        drv.replan(state, goal, n_refine)   # the whole MPPI refine, on GPU
        U = drv.nominal()
        # the nominal's trajectory is already column b=0 of the last refine's rollout (which
        # started at `state`); read just that column -- no re-rollout, no full-B readback.
        nominal = sim.controlled[:, 0].numpy()  # [T+1, 3]
        if record:  # read back the last refine's fan for the animation (slow path)
            samp = sim.controlled.numpy()[:, idx, :2].copy()      # [T+1, n_show, 2]
            badstep = ((sim.clearance.numpy()[:, idx] < clear_margin)
                       | (sim.residual.numpy()[:, idx] > resid_tol)).copy()  # [T, n_show]
            frames.append({"state": state.copy(), "samples": samp,
                           "stepbad": badstep, "chosen": nominal[:, :2].copy()})
        state = nominal[1].astype(np.float32).copy()  # step-1 pose of the nominal
        path.append(state.copy())
        U = np.roll(U, -1, axis=0)
        U[-1] = U[-2]
        drv.set_nominal(U)
    return (np.array(path), reached, frames) if record else (np.array(path), reached)


def _plot(scene, path, start, goal, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nx, ny = scene.nx, scene.ny
    ext = [scene.x0, scene.x0 + nx * scene.cell, scene.y0, scene.y0 + ny * scene.cell]  # cell edges
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


def _animate(scene, frames, path, start, goal, out, stride=3, fps=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    nx, ny = scene.nx, scene.ny
    ext = [scene.x0, scene.x0 + nx * scene.cell, scene.y0, scene.y0 + ny * scene.cell]  # cell edges
    sel = list(range(0, len(frames), stride))
    fig, ax = plt.subplots(figsize=(9, 6))

    def draw(fi):
        ax.clear()
        ax.imshow(scene.H, origin="lower", extent=ext, cmap="terrain", alpha=0.9)
        f = frames[fi]
        samp = f["samples"].transpose(1, 0, 2)   # [n_show, T+1, 2]
        bad = f["stepbad"].transpose(1, 0)       # [n_show, T]  per-step violation
        for s, b in zip(samp, bad):
            if b.any():
                j = int(b.argmax())              # first step that goes invalid
                ax.plot(s[:j + 1, 0], s[:j + 1, 1], "-", lw=0.4, alpha=0.3, color="deepskyblue")
                ax.plot(s[j:, 0], s[j:, 1], "-", lw=0.5, alpha=0.4, color="crimson")
            else:
                ax.plot(s[:, 0], s[:, 1], "-", lw=0.4, alpha=0.3, color="deepskyblue")
        ax.plot(f["chosen"][:, 0], f["chosen"][:, 1], "-", color="yellow", lw=2.0)  # plan
        tr = path[:fi + 1]
        ax.plot(tr[:, 0], tr[:, 1], "-", color="orange", lw=2.5)                    # driven
        ax.plot(f["state"][0], f["state"][1], "o", color="white", mec="k", ms=9)    # robot
        ax.plot(*start[:2], "o", color="0.4", ms=7)
        ax.plot(*goal[:2], "*", color="red", ms=18)
        ax.set_xlim(-1.0, 6.0); ax.set_ylim(-2.5, 3.0)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(f"MPPI live — blue=valid / red=invalid samples, yellow=plan  (step {fi})")

    anim = FuncAnimation(fig, draw, frames=sel, interval=1000 / fps)
    anim.save(out, writer=PillowWriter(fps=fps))
    print(f"saved {out}  ({len(sel)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/mppi.png")
    ap.add_argument("--gx", type=float, default=4.0)
    ap.add_argument("--gy", type=float, default=1.5)
    ap.add_argument("--B", type=int, default=2048)
    ap.add_argument("--animate", action="store_true", help="save a GIF of the planning")
    args = ap.parse_args()

    scene = hmmod.demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    start = np.array([0.0, 0.0, 0.0], np.float32)
    goal = np.array([args.gx, args.gy], np.float64)
    out = plan(scene, mu, start, goal, B=args.B, device=args.device, record=args.animate)
    path, reached = out[0], out[1]
    d = np.linalg.norm(path[-1, :2] - goal)
    print(f"reached={reached}  final=({path[-1,0]:+.2f},{path[-1,1]:+.2f})  "
          f"dist_to_goal={d:.2f}m  steps={len(path)-1}")
    if args.animate:
        _animate(scene, out[2], path, start, goal, args.out)
    else:
        _plot(scene, path, start, goal, args.out)


if __name__ == "__main__":
    main()
