"""Two SEPARATE windows showing the same drive from the two planners' points of view, each with its
OWN orbit camera (so you can frame them independently and place them side by side / on two monitors).

  WINDOW 1 (GLOBAL / routing): the accumulated map with the GROUND COLORED by the cost-to-go V
        (min over heading; viridis = cheap->far, dark = unreachable / outside the routing window),
        plus the optimal forward-only lattice route (orange) the router would follow.
  WINDOW 2 (LOCAL / MPPI): the same scene, but with the MPPI's actual sampled ROLLOUT FAN (thin lines,
        green = low cost -> red = high), the chosen nominal plan (bold cyan), and the live-scan walls
        the local planner sees (yellow). Cyan box = fine planning window; orange box = routing window.

Both windows step in lockstep (same robot/sim); only the cameras are independent. With --drift the
global map (window 1) smears while the local rollouts (window 2) stay on true geometry. In each window:
mouse-drag orbits, scroll zooms, ESC/Q quits.

Perception is helhest.perception's real OSDome 3D lidar; the local planning map (window 2) is dense
via helhest.perception inpaint + confidence masks (occlusion & support).

  python demos/navigate_partial_view.py --world pocket
  python demos/navigate_partial_view.py --world pocket --drift 0.04
  python demos/navigate_partial_view.py --world pocket --distrust-policy height-split
  python demos/navigate_partial_view.py --world pocket --shot /tmp/dual.png --shot-frame 150
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
from helhest.heightmap import Heightmap
from helhest.perception.lidar import crop_window
from helhest.planning.costtogo import CostToGo
from helhest.planning.lattice_solver import trace_optimal
from helhest.viz.render import _commands
from helhest.viz.render import _draw
from helhest.viz.render import _init_gl
from helhest.viz.render import build_robot
from helhest.viz.render import build_terrain

PW, PH = 900, 780  # each window's size


def _se2_points(pts, rx, ry, dx, dy, dyaw):
    """Apply an accumulated SE(2) drift (rotate dyaw about the robot, then translate) to a point
    cloud -- the point-cloud analogue of drift_scan, for smearing the accumulated global map."""
    c, s = np.cos(dyaw), np.sin(dyaw)
    x, y = pts[:, 0] - rx, pts[:, 1] - ry
    qx = rx + c * x - s * y + dx
    qy = ry + s * x + c * y + dy
    return np.column_stack([qx, qy, pts[:, 2]]).astype(np.float32)


def _view(cam, st, w, h):
    """Aim the current context's perspective camera at the robot, filling a w x h window."""
    from OpenGL import GL as gl
    from OpenGL import GLU as glu

    az, el, dist = cam
    tgt = np.array([st.x, st.y, st.place["z"]])
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = tgt + dist * d
    gl.glViewport(0, 0, w, h)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(50.0, w / max(h, 1), 0.1, 100.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    glu.gluLookAt(*eye, *tgt, 0, 0, 1)


def _draw_robot(robot, st):
    from OpenGL import GL as gl

    V, N, C, red = robot
    R4 = np.eye(4, dtype=np.float32)
    R4[:3, :3] = st.place["R"]
    gl.glPushMatrix()
    gl.glTranslatef(st.x, st.y, st.place["z"])
    gl.glMultMatrixf(np.ascontiguousarray(R4.T))
    _draw(V, N, C if st.valid else red)
    gl.glPopMatrix()


def _box(rx, ry, side, z):
    from OpenGL import GL as gl

    gl.glBegin(gl.GL_LINE_LOOP)
    for dx, dy in [
        (-side / 2, -side / 2),
        (side / 2, -side / 2),
        (side / 2, side / 2),
        (-side / 2, side / 2),
    ]:
        gl.glVertex3f(rx + dx, ry + dy, z)
    gl.glEnd()


def _goal_pole(goal):
    from OpenGL import GL as gl

    gl.glColor3f(0.95, 0.1, 0.1)
    gl.glLineWidth(5.0)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(goal[0], goal[1], 0.0)
    gl.glVertex3f(goal[0], goal[1], 1.6)
    gl.glEnd()


def _cost_colors(Vmin, vcap, scene, rwx0, rwy0, rccell, rcny, rcnx):
    """Per scene-cell RGB from the routing field V (min over heading), aligned to the global mesh's
    row*nx+col vertex order. Cells outside the routing window or unreachable -> dark."""
    from matplotlib import cm

    ny, nx = scene.ny, scene.nx
    xs = scene.x0 + (np.arange(nx) + 0.5) * scene.cell
    ys = scene.y0 + (np.arange(ny) + 0.5) * scene.cell
    XX, YY = np.meshgrid(xs, ys)
    cc = np.round((XX - rwx0) / rccell).astype(int)
    rr = np.round((YY - rwy0) / rccell).astype(int)
    inb = (rr >= 0) & (rr < rcny) & (cc >= 0) & (cc < rcnx)
    val = Vmin[np.clip(rr, 0, rcny - 1), np.clip(cc, 0, rcnx - 1)]
    reach = inb & (val < vcap * 0.9)
    vmax = float(np.percentile(val[reach], 95)) if reach.any() else 1.0
    C = cm.viridis(np.clip(val / max(vmax, 1e-6), 0.0, 1.0))[:, :, :3].astype(np.float32)
    C[~reach] = (0.12, 0.12, 0.16)
    return C.reshape(-1, 3)


def _draw_fan(fan, stepB):
    """Window-2 rollouts, each draped over the ground at its SETTLED height. A faint background is
    colored by FEASIBILITY (green = valid, red = hits a wall / exceeds the tilt envelope); the ELITE
    set MPPI actually averages is drawn bright green; the chosen nominal plan is bold cyan. ctr is
    [T+1, B] vec3 in planning-LOCAL coords; z is the settled body height [T+1, B]."""
    from OpenGL import GL as gl

    ctr, z, feas, elite, wx0, wy0 = (fan[k] for k in ("ctr", "z", "feas", "elite", "wx0", "wy0"))
    T1, B = z.shape

    def strip(b, dz):
        gl.glBegin(gl.GL_LINE_STRIP)
        for t in range(T1):
            gl.glVertex3f(wx0 + ctr[t, b, 0], wy0 + ctr[t, b, 1], z[t, b] + dz)
        gl.glEnd()

    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glLineWidth(1.4)
    for b in range(0, B, stepB):  # cloud: green = feasible, red = rejected (hits wall / tips)
        gl.glColor4f(0.3, 0.95, 0.4, 0.4) if feas[b] else gl.glColor4f(1.0, 0.25, 0.2, 0.4)
        strip(b, 0.02)
    gl.glDisable(gl.GL_BLEND)
    gl.glLineWidth(2.5)
    gl.glColor3f(0.5, 1.0, 0.6)  # the elites MPPI averages
    for b in elite:
        strip(int(b), 0.04)
    gl.glColor3f(0.1, 0.95, 0.98)
    gl.glLineWidth(4.0)  # chosen plan
    strip(0, 0.06)


def _grab(win):
    """Read the current GL back buffer (call after drawing, before swap) -> HxWx3 uint8, top-up."""
    import glfw
    from OpenGL import GL as gl

    w, h = glfw.get_framebuffer_size(win)
    gl.glReadBuffer(gl.GL_BACK)
    buf = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    return np.frombuffer(buf, np.uint8).reshape(h, w, 3)[::-1]


def _cbs(cam, ms):
    """Build (button, cursor, scroll) callbacks that orbit/zoom `cam` via mouse-state dict `ms`."""
    import glfw

    def on_button(w_, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            ms["down"] = action == glfw.PRESS
            ms["x"], ms["y"] = glfw.get_cursor_pos(w_)

    def on_cursor(w_, x, y):
        if ms["down"]:
            cam[0] -= (x - ms["x"]) * 0.01
            cam[1] = float(np.clip(cam[1] + (y - ms["y"]) * 0.01, 0.05, 1.5))
            ms["x"], ms["y"] = x, y

    def on_scroll(w_, dx, dy):
        cam[2] = float(np.clip(cam[2] - dy * 0.8, 3.0, 60.0))

    return on_button, on_cursor, on_scroll


def run(
    world="pocket",
    dock_radius=1.5,
    lat_coarsen=4,
    win_m=9.0,
    route_m=16.0,
    drift=0.0,
    drive=False,
    fan_n=160,
    distrust_policy="flat",
    support_ratio=0.35,
    support_radius_m=0.5,
    device="cuda",
    shot=None,
    shot_frame=None,
    max_frames=2000,
):
    import glfw
    from OpenGL import GL as gl

    wp.init()
    builder, start, goal = W.WORLDS[world]
    scene = builder()
    mu = W.matching_friction(scene)
    # perception front-end: helhest.perception's real OSDome 3D ray-cast, rasterized for the global
    # routing map, and inpaint + confidence masks (occlusion & support) for the local planning map.
    from helhest.perception.terrain_lidar import TerrainAccumMap
    from helhest.perception.terrain_lidar import TerrainInpaintMap
    from helhest.perception.terrain_lidar import TerrainLidar

    tlidar = TerrainLidar(scene, device=device)
    inpaint_map = TerrainInpaintMap(
        scene,
        distrust_policy=distrust_policy,
        support_ratio=support_ratio,
        support_radius_m=support_radius_m,
        device=device,
    )
    goal = np.asarray(goal, np.float64)
    cell = scene.cell
    ww = wh = int(round(win_m / cell))

    drv = WarpDriver(scene, mu, init_pose=tuple(start), device=device)  # reality
    win_grid = GridParams(ww, wh, cell, 0.0, 0.0)
    # match the ROS node's planner config (elevation_node params): short horizon, wmax 8, and the
    # straightness stack (turn penalty + peaky elite + straight prior).
    plan_sim = ForwardSimulator(
        dynamics.robot_params(), dynamics.planning_solver(), win_grid, 4096, 25, device
    )
    plan_sim.set_uniform_friction(0.8)
    planner = MppiGpu(
        plan_sim,
        CostParams(goal_running=0.3, effort=2e-3, turn=0.03),
        sampling=SamplingConfig(wmax=8.0, straight_frac=0.2, elite_frac=0.01),
        n_theta=24,
    )
    planner.reset_nominal(1.5)
    rww = rwh = int(round(max(route_m, win_m) / cell))
    kr = max(1, int(lat_coarsen))
    rcny, rcnx, rccell = rwh // kr, rww // kr, cell * kr
    route_grid = GridParams(rcnx, rcny, rccell, 0.0, 0.0)
    ctg = CostToGo(
        route_grid, dynamics.robot_params(), dynamics.planning_solver(), n_theta=24, device=device
    )
    # arm the saturation fallback (explore toward an out-of-window goal)
    planner.cw.lattice_cap = ctg._vcap
    sgrid = GridParams(
        rcnx, rcny, rccell, (ww // 2 - rww // 2) * cell, (wh // 2 - rwh // 2) * cell
    ).build()
    # GLOBAL routing map: helhest.perception rolling accumulator (25 m radius) -> inpaint + occlusion
    mm = TerrainAccumMap(scene, radius_m=25.0, device=device)
    local = inpaint_map  # LOCAL map the MPPI plans on (rebuilt per frame from the current scan)
    rng = np.random.default_rng(0)
    drift_x = drift_y = drift_yaw = 0.0  # accumulated SE(2) global-map drift (m, m, rad)
    prev_cmd = np.zeros(3, np.float32)  # last published [L, rear, R] for the slew limiter (node parity)
    prev_plan_U = None  # last nominal plan for the consistency EMA (node plan_consistency)
    stepB = max(1, plan_sim.batch_size // max(1, fan_n))
    # feasibility limits for coloring the rollouts (same as the MPPI cost invalid term)
    CM, RT, T = 0.05, 1e-2, plan_sim.n_steps
    _rp = dynamics.robot_params()
    MAXR, MAXPU, MAXPD = _rp.max_roll, _rp.max_pitch_up, _rp.max_pitch_down
    sXX, sYY = np.meshgrid(
        scene.x0 + (np.arange(scene.nx) + 0.5) * cell,  # cell centers (for cropping)
        scene.y0 + (np.arange(scene.ny) + 0.5) * cell,
    )

    if not glfw.init():
        raise RuntimeError("glfw init failed")
    if shot:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win_l = glfw.create_window(PW, PH, f"Helhest {world} - GLOBAL routing (cost-to-go)", None, None)
    # window 2 SHARES window 1's GL objects context (None here -> independent is fine; client arrays)
    win_r = glfw.create_window(PW, PH, f"Helhest {world} - LOCAL MPPI (rollouts)", None, None)
    if not shot:
        glfw.set_window_pos(win_l, 60, 90)
        glfw.set_window_pos(win_r, 80 + PW, 90)
    for w_ in (win_l, win_r):
        glfw.make_context_current(w_)
        _init_gl()
    cam_l = [-2.1, 0.85, 16.0]
    cam_r = [-2.1, 0.85, 16.0]
    for w_, cam in ((win_l, cam_l), (win_r, cam_r)):
        ms = {"down": False, "x": 0.0, "y": 0.0}
        bcb, ccb, scb = _cbs(cam, ms)
        glfw.set_mouse_button_callback(w_, bcb)
        glfw.set_cursor_pos_callback(w_, ccb)
        glfw.set_scroll_callback(w_, scb)

    robot = build_robot()
    _twr, _twc = np.nonzero(scene.H > 0.5)
    # true wall centers
    true_wall = np.column_stack([scene.x0 + _twc * cell, scene.y0 + _twr * cell])
    trail = []
    route = None  # latest optimal lattice route (world coords) for window 1
    fan = None  # latest rollouts (dict) for window 2
    crop = None  # (wx0, wy0) of the fine planning window, to clip window-2's terrain to it
    cost_C = None
    mode = {"manual": drive, "m_prev": False}  # press M (either window) to toggle AUTO <-> MANUAL

    def _press(k):  # key down in EITHER window -> driving works whichever window has focus
        return (
            glfw.PRESS
            if any(glfw.get_key(w_, k) == glfw.PRESS for w_ in (win_l, win_r))
            else glfw.RELEASE
        )

    f = 0
    while not (glfw.window_should_close(win_l) or glfw.window_should_close(win_r)):
        glfw.poll_events()
        if any(
            glfw.get_key(w_, k) == glfw.PRESS
            for w_ in (win_l, win_r)
            for k in (glfw.KEY_ESCAPE, glfw.KEY_Q)
        ):
            break
        m_now = _press(glfw.KEY_M) == glfw.PRESS  # edge-detected mode toggle
        if m_now and not mode["m_prev"]:
            mode["manual"] = not mode["manual"]
            print("MANUAL drive: I=fwd K=back J/L=turn" if mode["manual"] else "AUTO (MPPI)")
        mode["m_prev"] = m_now
        st = drv.render_state()
        rx, ry, yaw = st.x, st.y, st.yaw
        trail.append([rx, ry, st.place["z"] + 0.05])
        trail = trail[-6000:]
        d = float(np.hypot(rx - goal[0], ry - goal[1]))

        if d >= 0.3:
            pts = tlidar.scan_points((rx, ry, yaw))  # raw hit cloud -> feeds both maps
            if drift > 0.0:
                drift_x += float(rng.normal(0.0, drift))
                drift_y += float(rng.normal(0.0, drift))
                drift_yaw += float(rng.normal(0.0, 0.1 * drift))  # coupled rotational drift
                gpts = _se2_points(pts, rx, ry, drift_x, drift_y, drift_yaw)
                gcenter = (rx + drift_x, ry + drift_y)
            else:
                gpts, gcenter = pts, (rx, ry)
            # GLOBAL routing map: roll the accumulator in the (possibly drifted) belief frame
            mm.integrate(gpts, gcenter, tlidar.last_sensor_z)
            # LOCAL MPPI map: current single scan at the TRUE pose (drift-free); confidence-gated so
            # it holds only trusted cells, occlusion viewpoint is exactly this scan's pose
            inpaint_map.update(pts, (rx, ry), tlidar.last_sensor_z)
            local = inpaint_map
            elev, kn, wx0, wy0 = crop_window(local, scene, rx, ry, ww, wh, cell)
            elev = np.where(kn, elev, 0.0).astype(np.float32)
            goal_l = (goal[0] - wx0, goal[1] - wy0)
            state_l = np.array([rx - wx0, ry - wy0, yaw], np.float32)
            crop = (wx0, wy0)
            plan_sim.set_terrain(
                wp.array(np.ascontiguousarray(elev), dtype=wp.float32, device=device)
            )
            relev, rkn, rwx0, rwy0 = crop_window(mm, scene, rx, ry, rww, rwh, cell)
            relev = np.where(rkn, relev, 0.0).astype(np.float32)
            goal_r = (goal[0] - rwx0, goal[1] - rwy0)
            Hc = (
                relev[: rcny * kr, : rcnx * kr].reshape(rcny, kr, rcnx, kr).max(axis=(1, 3))
                if kr > 1
                else relev
            )
            V = ctg.compute(
                wp.array(np.ascontiguousarray(Hc), dtype=wp.float32, device=device), goal_r
            )
            planner.set_lattice(V, sgrid)
            # ENDGAME = the ROS node's (no dock): always replan (so window 2 shows the fan), EMA the
            # plan for consistency, then condition the command (output goal-brake + slew + clamp).
            planner.replan(state_l, goal_l, 3)
            U = planner.nominal()
            if prev_plan_U is not None and prev_plan_U.shape == U.shape:  # plan_consistency 0.3
                sh = np.roll(prev_plan_U, -1, axis=0)
                sh[-1] = prev_plan_U[-1]
                U = 0.7 * U + 0.3 * sh
                planner.set_nominal(U)
            prev_plan_U = U.copy()
            der = planner.sim.derived.numpy()
            clr, rsd = planner.sim.clearance.numpy(), planner.sim.residual.numpy()
            pit, rol = der[:T, :, 1], der[:T, :, 2]
            feas = ~(
                (clr < CM) | (rsd > RT) | (np.abs(rol) > MAXR) | (-pit > MAXPU) | (pit > MAXPD)
            ).any(
                0
            )  # per-rollout validity
            elite = np.argsort(planner.J.numpy())[: max(8, int(0.01 * planner.n_cand))]
            fan = dict(
                ctr=planner.sim.controlled.numpy(),
                z=der[..., 0],
                feas=feas,
                elite=elite,
                wx0=wx0,
                wy0=wy0,
            )
            # MANUAL: you drive (keyboard); AUTO: the node's conditioned command. Inside reach_radius
            # (0.3 m) command a ramped stop; else the MPPI step 0 with the distance-scaled goal brake.
            if mode["manual"]:
                cmd = _commands(_press).astype(np.float32)
            else:
                wl_c, wr_c = (0.0, 0.0) if d < 0.3 else (float(U[0, 0]), float(U[0, 1]))
                c = condition_command(wl_c, wr_c, prev_cmd, max_omega=8.0, max_slew=50.0,
                                      dt=dynamics.DT, goal_dist=d, brake_dist=3.0)
                prev_cmd = c
                cmd = np.array([c[0], c[2], c[1]], np.float32)  # [L, rear, R] -> driver [wl, wr, rear]
            drv.step(cmd)
            Vmin = V.numpy().min(axis=2)
            cost_C = _cost_colors(Vmin, ctg._vcap, scene, rwx0, rwy0, rccell, rcny, rcnx)
            rpts = trace_optimal(ctg, (rx - rwx0, ry - rwy0, yaw), 24, rcnx, rcny, 0.0, 0.0, rccell)
            route = (rpts + np.array([rwx0, rwy0])) if len(rpts) > 1 else None

        # window-1 mesh: the GLOBAL belief; window-2 mesh: the LOCAL map the MPPI actually plans on
        gbelief = np.where(mm.known, mm.elev, 0.0).astype(np.float32)
        Vg, Ng, Cg, idxg = build_terrain(Heightmap(gbelief, (scene.x0, scene.y0), cell))
        Cg[(~mm.known).ravel()] = (0.09, 0.09, 0.16)
        left_C = cost_C if cost_C is not None else Cg
        lbelief = np.where(local.known, local.elev, 0.0).astype(np.float32)
        Vl, Nl, Cl, idxl = build_terrain(Heightmap(lbelief, (scene.x0, scene.y0), cell))
        # clip the local terrain to the fine planning WINDOW (the bounded costmap the MPPI actually
        # crops to): inside & seen -> real terrain; inside & unseen -> flat (optimism); outside -> dark.
        if crop is not None:
            in_win = (
                (sXX >= crop[0])
                & (sXX < crop[0] + ww * cell)
                & (sYY >= crop[1])
                & (sYY < crop[1] + wh * cell)
            )
            Cl[(in_win & ~local.known).ravel()] = (0.28, 0.30, 0.34)
            Cl[(~in_win).ravel()] = (0.13, 0.14, 0.17)
        else:
            Cl[(~local.known).ravel()] = (0.28, 0.30, 0.34)

        # ===== WINDOW 1: global routing (ground = cost-to-go) =====
        glfw.make_context_current(win_l)
        _view(cam_l, st, *glfw.get_framebuffer_size(win_l))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        _draw(Vg, Ng, left_C, idxg)
        _draw_robot(robot, st)
        gl.glDisable(gl.GL_LIGHTING)
        if route is not None:
            gl.glColor3f(1.0, 0.45, 0.0)
            gl.glLineWidth(3.5)
            gl.glBegin(gl.GL_LINE_STRIP)
            for px, py in route:
                gl.glVertex3f(float(px), float(py), 0.18)
            gl.glEnd()
        gl.glColor3f(1.0, 0.6, 0.1)
        gl.glLineWidth(2.0)
        _box(rx, ry, max(route_m, win_m), 0.05)
        if drift > 0.0 and len(
            true_wall
        ):  # RED ghost of the TRUE walls -> the belief rotates off it
            gl.glColor3f(0.95, 0.15, 0.15)
            gl.glPointSize(3.0)
            gl.glBegin(gl.GL_POINTS)
            for px, py in true_wall:
                gl.glVertex3f(float(px), float(py), 0.2)
            gl.glEnd()
        _goal_pole(goal)
        gl.glEnable(gl.GL_LIGHTING)
        img_l = (
            _grab(win_l)
            if (
                shot
                and (
                    shot_frame is None and d < 0.3 or shot_frame is not None and f + 1 >= shot_frame
                )
            )
            else None
        )
        glfw.swap_buffers(win_l)

        # ===== WINDOW 2: local MPPI (rollout fan over the LOCAL terrain) =====
        glfw.make_context_current(win_r)
        _view(cam_r, st, *glfw.get_framebuffer_size(win_r))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        _draw(Vl, Nl, Cl, idxl)
        _draw_robot(robot, st)
        gl.glDisable(gl.GL_LIGHTING)
        if fan is not None:
            _draw_fan(fan, stepB)
        gl.glColor3f(0.1, 0.9, 0.95)
        gl.glLineWidth(2.0)
        _box(rx, ry, win_m, 0.06)
        _goal_pole(goal)
        gl.glEnable(gl.GL_LIGHTING)
        img_r = _grab(win_r) if img_l is not None else None
        glfw.swap_buffers(win_r)

        f += 1
        if f >= max_frames or img_l is not None:
            if img_l is not None:
                import matplotlib.pyplot as plt

                h = min(img_l.shape[0], img_r.shape[0])
                plt.imsave(shot, np.concatenate([img_l[:h], img_r[:h]], axis=1))
                print(f"saved {shot}")
            break
    glfw.terminate()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="pocket", choices=list(W.WORLDS))
    ap.add_argument("--dock-radius", type=float, default=1.5)
    ap.add_argument("--lat-coarsen", type=int, default=4)
    ap.add_argument("--win-m", type=float, default=9.0)
    ap.add_argument("--route-m", type=float, default=16.0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument(
        "--drive",
        action="store_true",
        help="start in MANUAL drive mode (I/J/K/L, either window); press M to toggle live",
    )
    ap.add_argument("--fan-n", type=int, default=160, help="how many rollouts to draw in window 2")
    ap.add_argument(
        "--distrust-policy",
        default="flat",
        choices=["flat", "height-split"],
        help="what the local inpaint map does with a confidence-distrusted cell: 'flat' (optimism) "
        "or 'height-split' (keep untrusted-but-tall visible cells as obstacles)",
    )
    ap.add_argument(
        "--support-ratio",
        type=float,
        default=0.35,
        help="trust an inpainted cell only if >= this fraction of its neighborhood was measured "
        "(lower = trust more)",
    )
    ap.add_argument(
        "--support-radius-m", type=float, default=0.5, help="support-mask neighborhood radius"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shot", default=None, help="save a side-by-side still of both windows")
    ap.add_argument(
        "--shot-frame",
        type=int,
        default=None,
        help="capture --shot at this frame (default: at goal reach)",
    )
    args = ap.parse_args()
    run(
        world=args.world,
        dock_radius=args.dock_radius,
        lat_coarsen=args.lat_coarsen,
        win_m=args.win_m,
        route_m=args.route_m,
        drift=args.drift,
        drive=args.drive,
        fan_n=args.fan_n,
        distrust_policy=args.distrust_policy,
        support_ratio=args.support_ratio,
        support_radius_m=args.support_radius_m,
        device=args.device,
        shot=args.shot,
        shot_frame=args.shot_frame,
    )


if __name__ == "__main__":
    main()
