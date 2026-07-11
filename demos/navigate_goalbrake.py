"""Full closed-loop pipeline with the goal-brake endgame -> a scrubable GIF.

Runs cost-to-go routing + MPPI + WarpDriver (the same stack as eval.py / the ROS node's planner)
with the three shipped endgame fixes, so you can WATCH the robot drive in fast and settle instead of
orbiting the goal:

  1. SHORT horizon (plan_horizon=25): the cost-to-go lattice routes globally, so a short MPPI just
     follows it decisively (a long horizon defers braking into its never-executed tail and orbits).
  2. OUTPUT BRAKE (control/command.condition_command, brake_dist): scales forward speed ~ distance on
     the final approach so the forward-only robot noses in and stops instead of flying past.
  3. PLAN CONSISTENCY: EMA the nominal toward last frame's plan -> no frame-to-frame jitter.

  python demos/navigate_goalbrake.py --world pocket --out /tmp/goalbrake.gif
  python demos/navigate_goalbrake.py --world gap --legacy --out /tmp/legacy.gif   # T=70, no brake -> orbits

The GIF: grey = terrain height; the MPPI rollout fan (green=cheap -> red=costly) with the chosen plan
(cyan); the robot (red dot+heading); the goal (star) + reach radius; a live speed read-out.
"""

import argparse

import numpy as np
import warp as wp

from helhest import dynamics
from helhest import worlds as W
from helhest.control.command import condition_command
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.control.mppi import SamplingConfig
from helhest.driver import WarpDriver
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo

DT = dynamics.DT


def simulate(world, *, T, brake_dist, consistency, wmax, effort, max_frames, device, fan_n=60,
             max_slew=50.0, raw=False, reach_radius=0.3, straight_frac=0.0, n_refine=3, sigma=0.5):
    """Run the closed loop; return per-frame snapshots for rendering."""
    builder, start, goal = W.WORLDS[world]
    scene = builder()
    mu = W.matching_friction(scene)
    goal = np.asarray(goal, np.float64)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    plan_sim = ForwardSimulator(dynamics.robot_params(), dynamics.planning_solver(), grid, 4096, int(T), device)
    plan_sim.set_terrain(wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device))
    plan_sim.set_friction(mu)
    planner = MppiGpu(plan_sim, CostParams(effort=effort),
                      sampling=SamplingConfig(wmax=wmax, straight_frac=straight_frac, sigma=sigma), n_theta=24)
    planner.reset_nominal(1.5)
    ctg = CostToGo(grid, dynamics.robot_params(), dynamics.planning_solver(), n_theta=24, device=device)
    V = ctg.compute(wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device), goal)
    planner.cw.lattice_cap = ctg._vcap
    planner.set_lattice(V, grid.build())
    drv = WarpDriver(scene, mu, init_pose=tuple(start), device=device)

    frames = []
    prev_cmd = np.zeros(3, np.float32)
    prev_U = None
    reached = False
    last_fan, last_cost, last_plan = None, None, None
    for f in range(max_frames):
        st = drv.render_state()
        d = float(np.hypot(st.x - goal[0], st.y - goal[1]))
        v_now = float(np.hypot(st.x - frames[-1]["x"], st.y - frames[-1]["y"]) / DT) if frames else 0.0
        if d < reach_radius:
            reached = True  # latched: mirror the node -> stop + idle (do NOT keep driving MPPI past
            # the goal, or a forward-only planner sails through and orbits).
        if reached:
            # goal reached -> command a ramped STOP each frame (the node's reach-idle behavior).
            cmd = condition_command(0.0, 0.0, prev_cmd, max_omega=wmax, max_slew=max_slew, dt=DT)
            prev_cmd = cmd
            drv.step([float(cmd[0]), float(cmd[2]), float(cmd[1])])
            fan, cost, plan = last_fan, last_cost, last_plan  # freeze the last viz snapshot
        else:
            planner.replan(np.array([st.x, st.y, st.yaw], np.float32), goal, n_refine)
            U = planner.nominal()
            if consistency > 0.0 and prev_U is not None:
                sh = np.roll(prev_U, -1, axis=0)
                sh[-1] = prev_U[-1]
                U = (1.0 - consistency) * U + consistency * sh
                planner.set_nominal(U)
            prev_U = U.copy()
            ctrl = planner.sim.controlled.numpy()  # [T+1, B, 3]
            J = planner.J.numpy()
            ci = np.linspace(0, planner.n_cand - 1, min(fan_n, planner.n_cand)).astype(int)
            fan, cost, plan = ctrl[:, ci, :2].astype(np.float32), J[ci].astype(np.float32), ctrl[:, 0, :2].astype(np.float32)
            last_fan, last_cost, last_plan = fan, cost, plan
            # actuate through the SAME conditioning path as the robot (output brake lives here)
            wl, wr = float(U[0, 0]), float(U[0, 1])
            if raw:  # bypass condition_command: forward-clamped brake only
                mean, diff = 0.5 * (wl + wr), (wr - wl)
                if brake_dist > 0:
                    mean *= min(1.0, d / brake_dist)
                cwl, cwr = max(0.0, mean - 0.5 * diff), max(0.0, mean + 0.5 * diff)
                drv.step([cwl, cwr, 0.5 * (cwl + cwr)])
            else:
                cmd = condition_command(wl, wr, prev_cmd, max_omega=wmax, max_slew=max_slew, dt=DT,
                                        goal_dist=d, brake_dist=brake_dist)
                prev_cmd = cmd
                drv.step([float(cmd[0]), float(cmd[2]), float(cmd[1])])  # [L, R, rear] for the driver
        frames.append(dict(x=st.x, y=st.y, yaw=st.yaw, d=d, v=v_now, reached=reached,
                           fan=fan, cost=cost, plan=plan))
        if reached and v_now < 0.03:
            break
    return scene, goal, frames


def render_gif(scene, goal, frames, out, stride, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import PillowWriter

    ext = [scene.x0, scene.x0 + scene.nx * scene.cell, scene.y0, scene.y0 + scene.ny * scene.cell]
    idx = list(range(0, len(frames), stride))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(scene.H, origin="lower", extent=ext, cmap="Greys", alpha=0.9)
    ax.plot(*goal, "*", color="gold", ms=22, mec="k", zorder=6)
    ax.add_patch(plt.Circle(goal, 0.3, fill=False, ls="--", ec="k", lw=1))
    trail = ax.plot([], [], "-", color="deepskyblue", lw=1.5, alpha=0.7)[0]
    fan_lines = [ax.plot([], [], "-", lw=0.5, alpha=0.5)[0] for _ in range(len(frames[0]["fan"][0]))]
    plan_line = ax.plot([], [], "-", color="cyan", lw=2.2, zorder=5)[0]
    robot = ax.plot([], [], "o", color="red", ms=9, zorder=7)[0]
    head = ax.plot([], [], "-", color="red", lw=2, zorder=7)[0]
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=11,
                  bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_title("goal-brake pipeline: cost-to-go routing + MPPI + WarpDriver")
    cmap = colormaps["RdYlGn_r"]
    xs = [fr["x"] for fr in frames]; ys = [fr["y"] for fr in frames]

    def upd(k):
        fr = frames[k]
        trail.set_data(xs[: k + 1], ys[: k + 1])
        c = fr["cost"]; c = (c - c.min()) / (np.ptp(c) + 1e-9)
        for j, ln in enumerate(fan_lines):
            ln.set_data(fr["fan"][:, j, 0], fr["fan"][:, j, 1]); ln.set_color(cmap(c[j]))
        plan_line.set_data(fr["plan"][:, 0], fr["plan"][:, 1])
        robot.set_data([fr["x"]], [fr["y"]])
        head.set_data([fr["x"], fr["x"] + 0.6 * np.cos(fr["yaw"])],
                      [fr["y"], fr["y"] + 0.6 * np.sin(fr["yaw"])])
        state = "REACHED — settling" if fr["reached"] else "driving"
        txt.set_text(f"frame {idx[k] if k < len(idx) else k}\n"
                     f"dist to goal: {fr['d']:.2f} m\nspeed: {fr['v']:.2f} m/s\n{state}")
        return fan_lines + [trail, plan_line, robot, head, txt]

    an = FuncAnimation(fig, upd, frames=idx, blit=False)
    an.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="pocket", choices=list(W.WORLDS))
    ap.add_argument("--out", default="/tmp/goalbrake.gif")
    ap.add_argument("--legacy", action="store_true", help="T=70, no brake, no consistency (the old orbit)")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--brake-dist", type=float, default=3.0)
    ap.add_argument("--consistency", type=float, default=0.3)
    ap.add_argument("--wmax", type=float, default=8.0)
    ap.add_argument("--effort", type=float, default=5e-4)
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-slew", type=float, default=50.0, help="slew limit [rad/s^2]; big = ~off")
    ap.add_argument("--raw", action="store_true", help="forward-clamp brake, bypass condition_command")
    ap.add_argument("--straight-frac", type=float, default=0.2, help="fraction of straight-prior samples")
    ap.add_argument("--n-refine", type=int, default=3, help="MPPI refine iterations per frame")
    ap.add_argument("--sigma", type=float, default=0.5, help="per-step sampling noise")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    wp.init()
    T, brake, cons = (70, 0.0, 0.0) if args.legacy else (args.horizon, args.brake_dist, args.consistency)
    scene, goal, frames = simulate(args.world, T=T, brake_dist=brake, consistency=cons, wmax=args.wmax,
                                   effort=args.effort, max_frames=args.max_frames, device=args.device,
                                   max_slew=args.max_slew, raw=args.raw, straight_frac=args.straight_frac,
                                   n_refine=args.n_refine, sigma=args.sigma)
    closest = min(fr["d"] for fr in frames)
    print(f"world={args.world} {'LEGACY(T70,no-brake)' if args.legacy else f'FIXED(T{T},brake{brake},cons{cons})'}"
          f" -> frames={len(frames)} closest={closest:.2f} reached={frames[-1]['reached']}")
    render_gif(scene, goal, frames, args.out, args.stride, args.fps)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
