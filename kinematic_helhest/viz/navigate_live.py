"""Watch the planner navigate a stress-test world -- or drive it yourself.

Loads a world from worlds.WORLDS and runs MppiGpu toward the goal (optionally with cost-to-go
and/or robust CVaR). Draws the terrain, robot, the plan (yellow) and the driven trail so you
can watch it reach or STALL -- e.g. the pocket cul-de-sac, or the slalom weave. `--drive` lets
you steer with I/J/K/L instead of the planner.

Run:  python -m kinematic_helhest.viz.navigate_live --world pocket [--costtogo] [--K 8] [--drive]
Shot: python -m kinematic_helhest.viz.navigate_live --world pocket --shot /tmp/nav.png
"""
import argparse

import numpy as np
import warp as wp

from .. import dynamics
from .. import worlds as W
from ..engine import GridParams
from ..engine import Simulator
from ..driver import WarpDriver
from ..planning.mppi_gpu import MppiGpu
from ..planning.terminal import dock_control
from .render import WIN_H
from .render import WIN_W
from .render import _commands
from .render import _init_gl
from .render import _render
from .render import build_robot
from .render import build_terrain

_PLAN, _GOAL = (1.0, 0.85, 0.05), (0.95, 0.1, 0.1)  # yellow plan, red goal pole


def _polyline(scene, xy, color, width, dz):
    from OpenGL import GL as gl
    z = np.minimum(scene.sample(xy[:, 0], xy[:, 1]), 0.7) + dz
    gl.glColor3f(*color); gl.glLineWidth(width)
    gl.glBegin(gl.GL_LINE_STRIP)
    for (x, y), zz in zip(xy, z):
        gl.glVertex3f(float(x), float(y), float(zz))
    gl.glEnd()


def run(world="pocket", costtogo=False, K=1, drive=False, shot=None, device="cuda", T=70, B=4096,
        tilt=0.0, tilt_free_deg=0.0, lattice=False, trav_weight=0.0, feasibility="traversability",
        dock_radius=1.5):
    import glfw

    builder, start, goal = W.WORLDS[world]
    scene = builder()
    mu = W.matching_friction(scene)
    goal = np.asarray(goal, np.float64)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    terr = wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)

    # planner and driver pull the SAME vehicle from dynamics (one timestep DT) -- a mismatch makes
    # the plan lag/clip. The residual plan->real gap is absorbed by CVaR (--K).
    drv = WarpDriver(scene, mu, init_pose=tuple(start), device=device)
    plan_sim = Simulator(dynamics.robot_params(), dynamics.planning_solver(), grid, B, T, device)
    plan_sim.set_terrain(terr); plan_sim.set_friction(mu)

    n_theta = 24
    w = dict(term=3.0, run=0.3, head=2.0, invalid=1e5, eff=2e-3, smooth=2e-3)
    if lattice:  # orientation-aware cost-to-go V(x,y,theta); trav_weight makes routing prefer flat
        w = {**w, "lattice": 1.0, "head": 0.0, "oob": 50.0, "term_v": 1.0,
             "endgame": 12.0, "endgame_r2": 2.25}
    elif costtogo:
        w = {**w, "ctg": 1.0, "head": 4.0}  # cost-to-go heading (-grad V) wants more weight
    if tilt > 0.0:  # penalize body tilt past tilt_free_deg along each rollout (steer onto flat ground)
        w = {**w, "tilt": float(tilt), "tilt_free": float(np.radians(tilt_free_deg))}
    planner = MppiGpu(plan_sim, 0.5, 4.0, w, 0.05, 1e-2, 0, sigma_knot=1.0, n_knots=4,
                      n_scenarios=K, n_theta=n_theta)
    planner.reset_nominal(1.5)
    if lattice:
        if feasibility == "settle":
            from ..planning.costtogo import CostToGoLatticeSettle
            clat = CostToGoLatticeSettle(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0, drv.sim.device,
                                         n_theta=n_theta, turn_radius=0.5, tilt_weight=trav_weight)
            planner.set_lattice(clat.compute(np.ascontiguousarray(scene.H, np.float32), mu, goal))
        else:
            from ..planning.costtogo import CostToGoLattice
            clat = CostToGoLattice(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0, drv.sim.device,
                                   n_theta=n_theta, turn_radius=0.5, trav_weight=trav_weight)
            planner.set_lattice(clat.compute(np.ascontiguousarray(scene.H, np.float32), goal))
    elif costtogo:
        from ..planning.costtogo import CostToGo
        ctg = CostToGo(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0, drv.sim.device)
        planner.set_costtogo(ctg.compute(np.ascontiguousarray(scene.H, np.float32), goal))

    if not glfw.init():
        raise RuntimeError("glfw init failed")
    if shot:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    mode = ", lattice" if lattice else (", cost-to-go" if costtogo else "")
    if lattice and trav_weight > 0.0:
        mode += f" (trav_w={trav_weight:g})"
    title = f"Helhest — {world} ({'drive' if drive else 'auto'}{mode})"
    win = glfw.create_window(WIN_W, WIN_H, title, None, None)
    glfw.make_context_current(win)
    _init_gl()
    from OpenGL import GL as gl
    terrain, robot = build_terrain(scene), build_robot()
    cam = [-2.1, 0.7, 10.0]
    mouse = {"down": False, "x": 0.0, "y": 0.0}

    def on_button(w_, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["down"] = action == glfw.PRESS
            mouse["x"], mouse["y"] = glfw.get_cursor_pos(w_)

    def on_cursor(w_, x, y):
        if mouse["down"]:
            cam[0] -= (x - mouse["x"]) * 0.01
            cam[1] = float(np.clip(cam[1] + (y - mouse["y"]) * 0.01, 0.05, 1.5))
            mouse["x"], mouse["y"] = x, y

    def on_scroll(w_, dx, dy):
        cam[2] = float(np.clip(cam[2] - dy * 0.5, 2.0, 40.0))

    glfw.set_mouse_button_callback(win, on_button)
    glfw.set_cursor_pos_callback(win, on_cursor)
    glfw.set_scroll_callback(win, on_scroll)

    trail, frame, plan_xy = [], 0, None
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS or glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break
        st = drv.render_state()
        state = np.array([st.x, st.y, st.yaw], np.float32)
        reached = np.hypot(st.x - goal[0], st.y - goal[1]) < 0.3
        planner.replan(state, goal, 3)
        U = planner.nominal()
        plan_xy = plan_sim.controlled[:, 0].numpy()[:, :2].copy()  # the nominal's path
        dist = float(np.hypot(st.x - goal[0], st.y - goal[1]))
        if drive:
            cmd = np.array([1.6, 1.6, 1.6]) if shot else _commands(lambda k: glfw.get_key(win, k))
        elif reached:
            cmd = np.zeros(3, np.float32)
        elif dock_radius > 0.0 and dist < dock_radius:
            cmd = dock_control(state, goal)  # terminal stage: decelerate + align to a precise stop
        else:
            cmd = np.array([U[0, 0], U[0, 1], 0.5 * (U[0, 0] + U[0, 1])], np.float32)
        drv.step(cmd)
        trail.append([st.x, st.y, st.place["z"] + 0.03]); trail = trail[-4000:]

        _render(st, cam, terrain, robot, trail)
        gl.glDisable(gl.GL_LIGHTING)
        if plan_xy is not None:
            _polyline(scene, plan_xy, _PLAN, 4.0, 0.08)
        gz = float(scene.sample(np.array([goal[0]]), np.array([goal[1]]))[0])
        gl.glColor3f(*_GOAL); gl.glLineWidth(5.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(goal[0], goal[1], gz); gl.glVertex3f(goal[0], goal[1], gz + 1.2)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)

        frame += 1
        if shot:
            if frame >= 20:
                gl.glReadBuffer(gl.GL_BACK)
                buf = gl.glReadPixels(0, 0, WIN_W, WIN_H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(buf, np.uint8).reshape(WIN_H, WIN_W, 3)[::-1]
                import matplotlib.pyplot as plt
                plt.imsave(shot, img); print(f"saved {shot}")
                break
            continue
        glfw.swap_buffers(win)
    glfw.terminate()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="pocket", choices=list(W.WORLDS))
    ap.add_argument("--costtogo", action="store_true", help="2D cost-to-go routing (avoids hard obstacles)")
    ap.add_argument("--lattice", action="store_true",
                    help="orientation-aware cost-to-go V(x,y,theta); pair with --trav-weight for flat-preferring routing")
    ap.add_argument("--trav-weight", type=float, default=0.0,
                    help="lattice arc-cost weight on terrain traversability (0 = pure distance; try 3) -> routes around bumps")
    ap.add_argument("--feasibility", default="traversability", choices=["traversability", "settle"],
                    help="lattice feasibility source: traversability map, or the robot's own settle (residual/tilt)")
    ap.add_argument("--K", type=int, default=8,
                    help="robust CVaR slip scenarios -- margin that absorbs the plan->real gap (1 = off)")
    ap.add_argument("--dock-radius", type=float, default=1.5,
                    help="terminal-stage handoff radius: within this the dock controller takes over (0 = off)")
    ap.add_argument("--drive", action="store_true", help="steer yourself instead of the planner")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shot", default=None)
    ap.add_argument("--tilt", type=float, default=0.0, help="tilt-penalty weight (0 = off; try 300)")
    ap.add_argument("--tilt-free-deg", type=float, default=8.0, help="tilt below this (deg) is free")
    args = ap.parse_args()
    run(world=args.world, costtogo=args.costtogo, K=args.K, drive=args.drive,
        shot=args.shot, device=args.device, tilt=args.tilt, tilt_free_deg=args.tilt_free_deg,
        lattice=args.lattice, trav_weight=args.trav_weight, feasibility=args.feasibility,
        dock_radius=args.dock_radius)


if __name__ == "__main__":
    main()
