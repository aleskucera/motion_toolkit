"""Full-pipeline sim where localization is a real EKF instead of "take the gated ICP pose".

pipeline_sim's closed loop localizes by refining an odom prediction with ICP and adopting
the gated ICP pose directly (no statistical fusion). This demo replaces that estimate stage
with an Extended Kalman Filter over the 6-DOF Helhest state q = [x, y, ψ, ẋ, ẏ, ψ̇]:

  PREDICT   x_pred = f(q, u)            — the ForwardSimulator kinematic model (predict_q6d)
            F = ∂f/∂q                    — the numerical 6×6 Jacobian (jacobian_F_6d)
            P⁻ = F P Fᵀ + Q
  MEASURE   z = ICP pose [x, y, ψ]       — LiDAR is the only sensor (no odom, no IMU)
            H = [I₃ | 0₃]
            standard EKF update, ψ-wrapped innovation

The control input u fed to PREDICT is the wheel-speed command the planner applied on the
previous frame. ICP is seeded by the EKF's own predicted pose (replacing the odom prior),
and a rejected/sparse ICP outcome is simply a missing measurement → predict-only step.

Everything downstream of localization (mapping, cost-to-go, MPPI, drive) is unchanged and
runs on the EKF-fused estimate. Reused verbatim from pipeline_sim: the worlds, the synthetic
OSDome lidar scan, the SE(2) helpers, and the Localizer (used here purely as the scan-to-map
ICP measurement front-end).

  python demos/pipeline_ekf.py --out /tmp/pipeline_ekf.gif [--world lane|narrow|slalom] [--dynamic]
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pipeline_sim  # same demos/ dir when run as a script
import warp as wp
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from PIL import Image

from helhest.filtering.ekf import EKF6D
from helhest.filtering.jacobian import jacobian_F_6d
from helhest.filtering.jacobian import predict_q6d

SENSOR_Z = pipeline_sim.SENSOR_Z
GROUND = pipeline_sim.GROUND
se2_to_mat = pipeline_sim.se2_to_mat
mat_to_se2 = pipeline_sim.mat_to_se2

ZMIN, ZMAX = -0.2, 2.0  # shared height colour-scale (ground → pillar top)
HCMAP = plt.cm.viridis
_NORM = plt.Normalize(ZMIN, ZMAX)
_GROUND = HCMAP(_NORM(0.0))  # colour of flat ground — the continuous-world backdrop
_RED = ListedColormap(["#ff2020"])

# ---------------------------------------------------------------------------
# EKF tuning — the single place to read / tweak the noise model.
#
# P0  — initial covariance; small because the filter bootstraps from the true pose.
# Q   — process noise;  velocity rows are intentionally generous because F[:,3:6]=0
#        (the sim re-derives velocity from u each step, so those states are driven by
#        the model, not filtered — their Q terms just set a numerical floor).
# R   — ICP measurement noise on [x, y, ψ].

# sigma (std-dev) values used to build the diagonal matrices
_SIG_P0 = np.array([0.10, 0.10, np.deg2rad(2.0), 0.30, 0.30, 0.20])   # [m, m, rad, m/s, m/s, rad/s]
_SIG_Q  = np.array([0.02, 0.02, np.deg2rad(0.5), 0.15, 0.15, 0.10])   # [m, m, rad, m/s, m/s, rad/s]
_SIG_R  = np.array([0.05, 0.05, np.deg2rad(1.0)])                       # [m, m, rad]

P0 = np.diag(_SIG_P0 ** 2)   # [6×6]  initial state covariance
Q  = np.diag(_SIG_Q  ** 2)   # [6×6]  process-noise covariance
R  = np.diag(_SIG_R  ** 2)   # [3×3]  ICP measurement-noise covariance
# ---------------------------------------------------------------------------

def _wrap(a: float) -> float:
    """Wrap an angle to (−π, π]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _build_flat_sims(scene, device: str):
    """Two flat-ground ForwardSimulators for the EKF process model, sized to the scene.

    The worlds are flat z=0 ground with pillar obstacles the robot never drives onto, so a
    locally-flat kinematic model is the right predictor at the robot's pose (friction 0.8 to
    match the planner's model). sim_pred (batch 1) evaluates f; sim_jac (batch 6) evaluates F.
    """
    from helhest import dynamics
    from helhest.engine import ForwardSimulator
    from helhest.engine import GridParams

    cell = 0.1
    margin = 3.0  # keep the robot pose comfortably inside the grid for the whole drive
    gx0, gy0 = scene.x0 - margin, scene.y0 - margin
    gnx = int(np.ceil((scene.nx * scene.cell + 2.0 * margin) / cell))
    gny = int(np.ceil((scene.ny * scene.cell + 2.0 * margin) / cell))
    grid = GridParams(gnx, gny, cell, gx0, gy0)
    elev0 = wp.zeros((gny, gnx), dtype=wp.float32, device=device)

    robot, solver = dynamics.robot_params(), dynamics.planning_solver()
    sims = []
    for batch in (1, 6):
        sim = ForwardSimulator(robot, solver, grid, batch, 1, device)
        sim.set_terrain(elev0)
        sim.set_uniform_friction(0.8)
        sims.append(sim)
    return sims[0], sims[1]


def run_closed_loop_ekf(
    device: str = "cuda",
    world: str = "lane",
    max_frames: int = 400,
    dt: float = 0.1,
    columns: int = 512,
    dropout: float = 0.03,
    exec_slip: float = 0.0,
    win_m: float = 8.0,
    route_m: float = 16.0,
    lat_coarsen: int = 4,
    local_support: int = 2,
    local_max_gap_m: float = 0.4,
    n_theta: int = 24,
    B: int = 4096,
    T: int = 70,
    dock_radius: float = 1.2,
    seed: int = 0,
    frame_hook=None,
    dynamic: bool = False,
) -> dict:
    """Closed loop driven on an EKF-fused pose estimate (LiDAR ICP the only sensor).

    Mirrors pipeline_sim.run_closed_loop stage-for-stage; only the localization estimate
    is different: EKF predict (model + numerical Jacobian on the previous wheel command)
    followed by an ICP measurement update.
    """
    from helhest import dynamics
    from helhest import worlds as W
    from helhest.control.mppi import CostParams
    from helhest.control.mppi import MppiGpu
    from helhest.control.terminal import dock_control
    from helhest.driver import WarpDriver
    from helhest.engine import ForwardSimulator
    from helhest.engine import GridParams
    from helhest.localization import Localizer
    from helhest.localization import LocalizerConfig
    from helhest.localization.pose_math import invert_pose
    from helhest.perception import DeviceMapAccumulator
    from helhest.perception import HeightMapBuilder
    from helhest.perception import IcpAligner
    from helhest.perception import IcpConfig
    from helhest.perception import multigrid_inpaint
    from helhest.perception.cloud_ops import transform_points
    from helhest.perception.sim import GroundSpec
    from helhest.perception.sim import make_osdome_lidar
    from helhest.perception.sim import osdome_sensor_config
    from helhest.planning.costtogo import CostToGo

    scene, box_lo, box_hi, start, goal = pipeline_sim._WORLDS[world]()
    cell = scene.cell
    mu = W.matching_friction(scene)
    drv = WarpDriver(scene, mu, init_pose=tuple(start), device=device)  # REALITY
    slip_rng = np.random.default_rng(seed + 7777)

    sensor = osdome_sensor_config(columns=columns)
    ground = GroundSpec(z=0.0, x_range=(-GROUND, GROUND), y_range=(-GROUND, GROUND))
    lidar = make_osdome_lidar(ground, sensor=sensor, facing="front", dropout=dropout, device=device)
    acc = DeviceMapAccumulator(pipeline_sim.MAP_VOXEL, pipeline_sim.MAP_RADIUS, device=device)
    aligner = IcpAligner(IcpConfig(max_iters=30, max_correspondence_dist_m=0.5), device=device)
    localizer = Localizer(aligner, LocalizerConfig())
    if dynamic:
        from helhest.perception import DynamicPointFilter

        filt = DynamicPointFilter.from_sensor(sensor, margin_m=0.3, margin_rel=0.03, az_bins=720, el_bins=180, device=device)
        acc_raw = DeviceMapAccumulator(pipeline_sim.MAP_VOXEL, pipeline_sim.MAP_RADIUS, device=device)
        map_raw = None

    # EKF process-model sims (flat ground) + filter noise model (module-level defaults).
    sim_pred, sim_jac = _build_flat_sims(scene, device)
    R_icp = R
    ekf: EKF6D | None = None

    ww = wh = int(round(win_m / cell))
    win_grid = GridParams(ww, wh, cell, 0.0, 0.0)
    plan_sim = ForwardSimulator(dynamics.robot_params(), dynamics.planning_solver(), win_grid, B, T, device)
    plan_sim.set_uniform_friction(0.8)
    planner = MppiGpu(plan_sim, CostParams(), n_theta=n_theta)
    planner.reset_nominal(1.5)
    kr = max(1, int(lat_coarsen))
    rww = rwh = int(round(route_m / cell))
    rcny, rcnx, rccell = rwh // kr, rww // kr, cell * kr
    ctg = CostToGo(
        GridParams(rcnx, rcny, rccell, 0.0, 0.0),
        dynamics.robot_params(), dynamics.planning_solver(), n_theta=n_theta, device=device,
    )
    planner.cw.lattice_cap = ctg._vcap
    sgrid = GridParams(rcnx, rcny, rccell, (ww // 2 - rww // 2) * cell, (wh // 2 - rwh // 2) * cell).build()

    map_wp: wp.array | None = None
    # commanded wheel speeds [ω_L, ω_R, ω_rear] applied last frame — the EKF predict input.
    prev_cmd = np.zeros(3, np.float32)
    true_tr, est_tr, icp_tr = [], [], []
    err_fused, err_icp = [], []
    # per-frame diagonals of P: after predict (P⁻) and after update (P⁺); shape [6] each
    p_pred_diag: list[np.ndarray] = []
    p_upd_diag: list[np.ndarray] = []
    innov_x, innov_y, innov_yaw, status_hist = [], [], [], []
    contacts, reached, f = 0, False, 0

    for f in range(max_frames):
        st = drv.render_state()
        true_tr.append((st.x, st.y))
        if float(np.hypot(st.x - goal[0], st.y - goal[1])) < 0.3:
            reached = True
            break
        T_true = se2_to_mat(st.x, st.y, st.yaw)

        if dynamic:
            wlo, whi, walker = pipeline_sim._walker_box(f, dt)
            blo, bhi = np.vstack([box_lo, wlo]), np.vstack([box_hi, whi])
        else:
            blo, bhi, walker = box_lo, box_hi, None
        scan_base, free_wp = pipeline_sim._scan_base(lidar, st.x, st.y, st.yaw, blo, bhi, f + 1, device)

        # --- EKF localization (predict via model+Jacobian, update via ICP) ---
        z_icp = None
        innovation = None
        if ekf is None:
            localizer.bootstrap(T_true, T_true)
            ekf = EKF6D(np.array([st.x, st.y, st.yaw, 0.0, 0.0, 0.0]), P0, Q, R_icp)
            x_pred = ekf.x.copy()
            T_wb = T_true
            icp_status = "bootstrap"
            icp_pose = T_true
            # bootstrap: no predict step; P⁻ and P⁺ are both the initial covariance
            p_minus_diag = ekf.P.diagonal().copy()
            p_plus_diag = ekf.P.diagonal().copy()
        else:
            u = prev_cmd.astype(np.float64)
            x_pred = predict_q6d(ekf.x, u, sim_pred)
            F = jacobian_F_6d(ekf.x, u, sim_jac)
            ekf.predict(F, x_pred)
            p_minus_diag = ekf.P.diagonal().copy()  # P⁻: after predict, before update
            T_pred = se2_to_mat(ekf.x[0], ekf.x[1], ekf.x[2])
            # Localizer used purely as the scan-to-map ICP front-end, seeded by the EKF
            # prediction (the odom_T args just seed/store its unused internal state).
            outcome = localizer.update(scan_base, T_pred, map_wp, T_pred)
            icp_status = outcome.status
            icp_pose = outcome.pose
            if outcome.status == "ok":
                zx, zy, zpsi = mat_to_se2(outcome.pose)
                z_icp = np.array([zx, zy, zpsi])
                innovation = np.array([zx - ekf.x[0], zy - ekf.x[1], _wrap(zpsi - ekf.x[2])])
                ekf.update_icp(z_icp)
            p_plus_diag = ekf.P.diagonal().copy()   # P⁺: after update (== P⁻ if no measurement)
            T_wb = se2_to_mat(ekf.x[0], ekf.x[1], ekf.x[2])
        ex, ey, eyaw = mat_to_se2(T_wb)

        # --- bookkeeping for the dashboard ---
        est_tr.append((ex, ey))
        icx, icy, _ = mat_to_se2(icp_pose)
        icp_tr.append((icx, icy))
        err_fused.append(float(np.hypot(ex - st.x, ey - st.y)))
        err_icp.append(float(np.hypot(icx - st.x, icy - st.y)))
        p_pred_diag.append(p_minus_diag)
        p_upd_diag.append(p_plus_diag)
        if innovation is not None:
            innov_x.append(innovation[0]); innov_y.append(innovation[1]); innov_yaw.append(innovation[2])
        else:
            innov_x.append(np.nan); innov_y.append(np.nan); innov_yaw.append(np.nan)
        status_hist.append(icp_status)

        # --- map accumulation (on the fused estimate) ---
        world_corrected = transform_points(scan_base, len(scan_base), T_wb)
        valid = wp.full(len(scan_base), 1, dtype=wp.int32, device=device)
        carve = None
        if dynamic and map_wp is not None:
            free_est = transform_points(free_wp, len(free_wp), T_wb @ invert_pose(T_true))
            carve = filt.carve(map_wp, free_est, np.array([ex, ey, SENSOR_Z], np.float32))
        map_wp = acc.step(map_wp, carve, world_corrected, valid, (ex, ey))
        if dynamic:
            map_raw = acc_raw.step(map_raw, None, world_corrected, valid, (ex, ey))

        # --- local + global heightmaps → MPPI terrain (identical to pipeline_sim) ---
        half = win_m / 2.0
        xmin, ymin = ex - half, ey - half
        bounds = (xmin, ex + half, ymin, ey + half)
        ll = HeightMapBuilder(cell, bounds, device=device).build(world_corrected)
        conf = (ll.count.numpy() >= local_support)[:wh, :ww]
        hm = np.where(conf, ll.max.numpy()[:wh, :ww], np.nan).astype(np.float32)
        filled = np.nan_to_num(np.asarray(multigrid_inpaint(hm)), nan=0.0).astype(np.float32)
        known_local = pipeline_sim._dilate_bool(conf, int(round(local_max_gap_m / cell)))
        rhalf = route_m / 2.0
        rxmin, rymin = ex - rhalf, ey - rhalf
        rgl = HeightMapBuilder(cell, (rxmin, ex + rhalf, rymin, ey + rhalf), device=device).build(map_wp)
        relev = np.where(rgl.count.numpy() > 0, rgl.max.numpy(), 0.0).astype(np.float32)[:rwh, :rww]
        oy, ox = (rwh - wh) // 2, (rww - ww) // 2
        mem = relev[oy : oy + wh, ox : ox + ww]
        elev_local = np.where(known_local, filled, mem).astype(np.float32)

        # --- cost-to-go + MPPI (on the fused estimate) ---
        state_l = np.array([ex - xmin, ey - ymin, eyaw], np.float32)
        goal_l = (goal[0] - xmin, goal[1] - ymin)
        plan_sim.set_terrain(wp.array(np.ascontiguousarray(elev_local), dtype=wp.float32, device=device))
        Hc = relev[: rcny * kr, : rcnx * kr].reshape(rcny, kr, rcnx, kr).max(axis=(1, 3)) if kr > 1 else relev
        goal_r = (goal[0] - rxmin, goal[1] - rymin)
        V = ctg.compute(wp.array(np.ascontiguousarray(Hc), dtype=wp.float32, device=device), goal_r)
        planner.set_lattice(V, sgrid)

        if dock_radius > 0.0 and float(np.hypot(ex - goal[0], ey - goal[1])) < dock_radius:
            cmd = dock_control(state_l, goal_l)
        else:
            planner.replan(state_l, goal_l, 3)
            u_plan = planner.nominal()
            cmd = np.array([u_plan[0, 0], u_plan[0, 1], 0.5 * (u_plan[0, 0] + u_plan[0, 1])], np.float32)
        if exec_slip > 0.0:
            sL, sR = slip_rng.uniform(exec_slip, 1.0, 2)
            cmd = np.array([cmd[0] * sL, cmd[1] * sR, 0.5 * (cmd[0] * sL + cmd[1] * sR)], np.float32)
        drv.step(cmd)
        prev_cmd = cmd  # the command that moves reality from this frame to the next
        if drv.clear < 0.05:
            contacts += 1

        if frame_hook is not None:
            frame_hook(dict(
                f=f, true=(st.x, st.y, st.yaw), est=(ex, ey, eyaw), map_wp=map_wp,
                elev_local=elev_local, known_local=known_local,
                xmin=xmin, ymin=ymin, cell=cell, ww=ww, wh=wh, goal=goal,
                box_lo=box_lo, box_hi=box_hi, scene=scene, scan_world=world_corrected,
                ekf_x=ekf.x.copy(), ekf_P=ekf.P.copy(), x_pred=x_pred.copy(),
                z_icp=z_icp, innovation=innovation, icp_status=icp_status,
                true_tr=list(true_tr), est_tr=list(est_tr), icp_tr=list(icp_tr),
                err_fused=list(err_fused), err_icp=list(err_icp),
                p_pred_diag=list(p_pred_diag), p_upd_diag=list(p_upd_diag),
                innov_x=list(innov_x), innov_y=list(innov_y), innov_yaw=list(innov_yaw),
                status_hist=list(status_hist), contacts=contacts,
                map_raw=(map_raw if dynamic else None), walker=walker,
            ))

    return dict(
        true=np.asarray(true_tr), est=np.asarray(est_tr), icp=np.asarray(icp_tr),
        err_fused=np.asarray(err_fused), err_icp=np.asarray(err_icp),
        box_lo=box_lo, box_hi=box_hi, goal=goal, reached=reached, frames=f + 1,
        contacts=contacts,
    )


def _robot(ax, x, y, yaw, scale):
    """Robot = a dot + a heading arrow."""
    ax.plot(x, y, "o", color="magenta", ms=6, mec="k", zorder=6)
    ax.arrow(x, y, scale * math.cos(yaw), scale * math.sin(yaw), color="magenta",
             width=scale * 0.08, head_width=scale * 0.42, length_includes_head=True, zorder=6)


def _cov_ellipse(ax, mean_xy, P_xy, n_sigma=2.0, **kw):
    """Draw the n_sigma covariance ellipse of a 2×2 xy block at mean_xy."""
    vals, vecs = np.linalg.eigh(P_xy)
    vals = np.clip(vals, 1e-9, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_sigma * np.sqrt(vals)
    ax.add_patch(Ellipse(mean_xy, width, height, angle=angle, **kw))


class Dashboard:
    """Per-frame EKF inspector → scrubbable GIF."""

    def __init__(self, stride, view_m):
        self.stride, self.V = stride, view_m
        self.frames: list[Image.Image] = []
        self.fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        (self.ax_world, self.ax_scan, self.ax_track) = axes[0]
        (self.ax_global, self.ax_cov, self.ax_err) = axes[1]
        from helhest.perception import HeightMapBuilder

        self._HMB = HeightMapBuilder

    def __call__(self, s):
        if s["f"] % self.stride:
            return
        V = self.V
        ex, ey, eyaw = s["est"]
        tx, ty, tyaw = s["true"]
        cell, ww, wh = s["cell"], s["ww"], s["wh"]
        gx, gy = s["goal"]
        walker = s.get("walker")
        for a in (self.ax_world, self.ax_scan, self.ax_track, self.ax_global, self.ax_cov, self.ax_err):
            a.clear()

        def big(ax):
            ax.set_xlim(ex - V, ex + V)
            ax.set_ylim(ey - V, ey + V)
            ax.set_aspect("equal")

        # --- real world (ground truth): continuous heightmap backdrop
        sc = s["scene"]
        aw = self.ax_world
        aw.set_facecolor(_GROUND)
        aw.imshow(sc.H, origin="lower", extent=[sc.x0, sc.x0 + sc.nx * sc.cell, sc.y0, sc.y0 + sc.ny * sc.cell],
                  cmap=HCMAP, vmin=ZMIN, vmax=ZMAX)
        if walker is not None:
            aw.add_patch(Rectangle((walker[0] - 0.35, walker[1] - 0.35), 0.7, 0.7, color=HCMAP(_NORM(1.8))))
        _robot(aw, tx, ty, tyaw, V * 0.12)
        big(aw)
        aw.set_title("Real world (ground truth)")

        # --- live lidar scan, coloured by height
        asc = self.ax_scan
        asc.set_facecolor("#101014")
        sw = s["scan_world"].numpy()
        asc.scatter(sw[:, 0], sw[:, 1], c=sw[:, 2], s=2, cmap=HCMAP, vmin=ZMIN, vmax=ZMAX)
        _robot(asc, ex, ey, eyaw, V * 0.12)
        big(asc)
        asc.set_title("Live lidar scan (what it sees now)")

        # --- tracks: truth vs raw-ICP measurement vs EKF fused, + covariance ellipse
        at = self.ax_track
        at.set_facecolor("#f4f4f8")
        tr = np.asarray(s["true_tr"])
        icp = np.asarray(s["icp_tr"])
        est = np.asarray(s["est_tr"])
        at.plot(tr[:, 0], tr[:, 1], "-", color="#2ca02c", lw=2.4, label="ground truth")
        at.plot(icp[:, 0], icp[:, 1], ".", color="#1f77b4", ms=3, alpha=0.6, label="raw ICP measurement")
        at.plot(est[:, 0], est[:, 1], "-", color="#d62728", lw=1.6, label="EKF fused")
        # 2σ position-covariance ellipse of the current estimate
        _cov_ellipse(at, (ex, ey), s["ekf_P"][:2, :2], n_sigma=2.0,
                     fill=False, ec="#d62728", lw=1.8, ls="--", zorder=5)
        # predicted vs corrected pose this frame (shows what the measurement pulled)
        xp = s["x_pred"]
        at.plot(xp[0], xp[1], "x", color="k", ms=8, mew=2, label="EKF predict")
        _robot(at, ex, ey, eyaw, V * 0.12)
        big(at)
        at.legend(loc="upper left", fontsize=8)
        at.set_title(f"Tracks + 2σ cov ellipse  (ICP: {s['icp_status']})")

        # --- global accumulated map → routing
        ag = self.ax_global
        ag.set_facecolor(_GROUND)
        gext = [ex - V, ex + V, ey - V, ey + V]
        if s["map_wp"] is not None and len(s["map_wp"]):
            dev = s["map_wp"].device
            gl = self._HMB(0.15, tuple(gext), device=dev).build(s["map_wp"])
            gcount = gl.count.numpy()
            ag.imshow(np.where(gcount > 0, gl.max.numpy(), np.nan), origin="lower", extent=gext,
                      cmap=HCMAP, vmin=ZMIN, vmax=ZMAX)
        if walker is not None:
            ag.add_patch(Rectangle((walker[0] - 0.35, walker[1] - 0.35), 0.7, 0.7, fill=False, ec="orange", lw=2))
        _robot(ag, ex, ey, eyaw, V * 0.12)
        big(ag)
        ag.set_title("Global map (built on the fused estimate) → routing")

        # --- diagonal state covariance: σᵢ = √P_ii, post-predict (dashed) and post-update (solid)
        ac = self.ax_cov
        fr = np.arange(len(s["p_pred_diag"]))
        # stack into [N, 6] arrays of std-devs
        sig_pred = np.sqrt(np.maximum(np.asarray(s["p_pred_diag"]), 0.0))  # [N, 6]
        sig_upd  = np.sqrt(np.maximum(np.asarray(s["p_upd_diag"]),  0.0))  # [N, 6]
        labels = ["x [m]", "y [m]", "ψ [rad]", "ẋ [m/s]", "ẏ [m/s]", "ψ̇ [rad/s]"]
        colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
        for i, (lbl, col) in enumerate(zip(labels, colors)):
            ac.plot(fr, sig_pred[:, i], "--", color=col, lw=1.0, alpha=0.7)
            ac.plot(fr, sig_upd[:, i],  "-",  color=col, lw=1.6, label=lbl)
        # shade predict-only frames so the gap between dashed/solid being zero is visible
        rej = np.array([st != "ok" for st in s["status_hist"]])
        for idx in fr[rej]:
            ac.axvspan(idx - 0.5, idx + 0.5, color="#ff7f0e", alpha=0.15)
        ac.set_xlabel("frame")
        ac.set_ylabel("σ (std-dev)")
        ac.legend(loc="upper right", fontsize=7, ncol=2)
        ac.set_title("P diagonal: σᵢ = √P_ii  (dashed = post-predict, solid = post-update)")

        # --- localization error over time
        ae = self.ax_err
        ef = np.asarray(s["err_fused"])
        ei = np.asarray(s["err_icp"])
        ae.plot(fr, ei, "-", color="#1f77b4", lw=1.2, alpha=0.8, label="raw ICP error")
        ae.plot(fr, ef, "-", color="#d62728", lw=1.6, label="EKF fused error")
        for idx in fr[rej]:
            ae.axvspan(idx - 0.5, idx + 0.5, color="#ff7f0e", alpha=0.15)
        ae.set_xlabel("frame")
        ae.set_ylabel("translation error (m)")
        ae.legend(loc="upper left", fontsize=8)
        ae.set_title("Localization error: EKF fused vs raw ICP  (orange = predict-only)")

        self.fig.suptitle(
            f"frame {s['f']}   fused-err {ef[-1]:.2f} m   icp-err {ei[-1]:.2f} m   "
            f"contacts {s['contacts']}", fontsize=15,
        )
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.frames.append(Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())).convert("RGB"))

    def save(self, out, fps):
        if not self.frames:
            print("no frames")
            return
        self.frames[0].save(out, save_all=True, append_images=self.frames[1:], duration=int(1000 / fps), loop=0)
        print(f"saved {out}  ({len(self.frames)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--world", choices=list(pipeline_sim._WORLDS), default="lane")
    ap.add_argument("--max-frames", type=int, default=340)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--view-m", type=float, default=12.0, help="half-extent of the big robot-centered panels (m)")
    ap.add_argument("--dynamic", action="store_true", help="add a moving obstacle")
    ap.add_argument("--exec-slip", type=float, default=0.0, help="reality wheel slip: driver keeps uniform[x,1] per wheel")
    ap.add_argument("--out", default="/tmp/pipeline_ekf.gif")
    args = ap.parse_args()
    wp.init()

    dash = Dashboard(args.stride, args.view_m)
    res = run_closed_loop_ekf(
        device=args.device, world=args.world, max_frames=args.max_frames, frame_hook=dash,
        dynamic=args.dynamic, exec_slip=args.exec_slip,
    )
    print(
        f"EKF CLOSED LOOP  reached={res['reached']} frames={res['frames']} contacts={res['contacts']}  "
        f"mean fused-err={res['err_fused'].mean():.3f} m  mean icp-err={res['err_icp'].mean():.3f} m  "
        f"(max fused {res['err_fused'].max():.3f} m)"
    )
    dash.save(args.out, args.fps)


if __name__ == "__main__":
    main()
