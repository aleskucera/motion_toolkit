"""EKF noise-model tuning sweep — which (Q, R) localizes the Helhest best?

This reuses pipeline_ekf's closed loop verbatim (LiDAR-ICP-only EKF over the 6-DOF state
q = [x, y, ψ, ẋ, ẏ, ψ̇]) but strips the per-frame dashboard. Instead of one run + a GIF it
runs the loop many times on the same world (default --world slalom) with a grid of process-
and measurement-noise matrices, records the localization error against ground truth, and
plots which (Q, R) wins.

For each run we record, per frame:
    translation error  ‖(x̂, ŷ) − (x, y)‖           [m]
    heading error      |wrap(ψ̂ − ψ)|                [rad]

The sweep scales the baseline diagonals Q and R by scalar multipliers on a grid
(q_scale × r_scale). Bigger q_scale → trust the model less / the ICP measurement more;
bigger r_scale → trust the ICP measurement less / the model more. The output is a 2×2
figure: RMS-error heatmaps over the grid (translation, heading) plus overlaid error-vs-frame
curves per config, and a printed "best (Q_scale, R_scale)" for each metric.

  python demos/ekf_tuning.py --out /tmp/ekf_tuning.png [--world slalom]
      [--q-scales 0.1 1 10] [--r-scales 0.1 1 10] [--max-frames 340]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pipeline_sim  # same demos/ dir when run as a script
import warp as wp

from helhest.filtering.ekf import EKF6D
from helhest.filtering.jacobian import jacobian_F_6d
from helhest.filtering.jacobian import predict_q6d

SENSOR_Z = pipeline_sim.SENSOR_Z
GROUND = pipeline_sim.GROUND
se2_to_mat = pipeline_sim.se2_to_mat
mat_to_se2 = pipeline_sim.mat_to_se2

# ---------------------------------------------------------------------------
# Baseline EKF noise model (identical to pipeline_ekf). The sweep scales the
# Q and R diagonals below by scalar multipliers; P0 is held fixed.
#
# P0  — initial covariance; small because the filter bootstraps from the true pose.
# Q   — process noise;  velocity rows are intentionally generous because F[:,3:6]=0
#        (the sim re-derives velocity from u each step).
# R   — ICP measurement noise on [x, y, ψ].

_SIG_P0 = np.array([0.10, 0.10, np.deg2rad(2.0), 0.30, 0.30, 0.20])   # [m, m, rad, m/s, m/s, rad/s]
_SIG_Q  = np.array([0.02, 0.02, np.deg2rad(0.5), 0.15, 0.15, 0.10])   # [m, m, rad, m/s, m/s, rad/s]
_SIG_R  = np.array([0.05, 0.05, np.deg2rad(1.0)])                       # [m, m, rad]

P0 = np.diag(_SIG_P0 ** 2)   # [6×6]  initial state covariance
Q  = np.diag(_SIG_Q  ** 2)   # [6×6]  process-noise covariance   (baseline)
R  = np.diag(_SIG_R  ** 2)   # [3×3]  ICP measurement-noise covariance (baseline)
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
    world: str = "slalom",
    Q_mat: np.ndarray | None = None,
    R_mat: np.ndarray | None = None,
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
    use_ekf: bool = True,
) -> dict:
    """Closed loop driven on a pose estimate from LiDAR ICP, optionally fused via EKF.

    Mirrors pipeline_ekf.run_closed_loop_ekf stage-for-stage; the only differences are that
    the noise matrices (Q_mat, R_mat) are passed in per-run for tuning, and the dashboard
    frame_hook / cov / innovation bookkeeping is stripped. Per frame we record the localization
    error against ground truth: translation ‖Δxy‖ [m] and heading |Δψ| [rad].

    use_ekf=False: raw ICP result is taken as the pose estimate directly (no state model).
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

    Q_mat = Q if Q_mat is None else Q_mat
    R_mat = R if R_mat is None else R_mat

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

    # EKF process-model sims (flat ground) + the per-run noise model.
    if use_ekf:
        sim_pred, sim_jac = _build_flat_sims(scene, device)
    ekf: EKF6D | None = None
    T_wb: np.ndarray | None = None  # SE(2) matrix; None = first frame sentinel

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
    err_trans: list[float] = []   # ‖(x̂, ŷ) − (x, y)‖ per frame [m]
    err_psi: list[float] = []     # |wrap(ψ̂ − ψ)| per frame [rad]
    contacts, reached, f = 0, False, 0

    for f in range(max_frames):
        st = drv.render_state()
        if float(np.hypot(st.x - goal[0], st.y - goal[1])) < 0.3:
            reached = True
            break
        T_true = se2_to_mat(st.x, st.y, st.yaw)

        blo, bhi, walker = box_lo, box_hi, None
        scan_base, free_wp = pipeline_sim._scan_base(lidar, st.x, st.y, st.yaw, blo, bhi, f + 1, device)

        # --- localization: EKF-fused or raw ICP ---
        if T_wb is None:
            # First frame: bootstrap from ground truth.
            localizer.bootstrap(T_true, T_true)
            if use_ekf:
                ekf = EKF6D(np.array([st.x, st.y, st.yaw, 0.0, 0.0, 0.0]), P0, Q_mat, R_mat)
            T_wb = T_true
        elif use_ekf:
            u = prev_cmd.astype(np.float64)
            x_pred = predict_q6d(ekf.x, u, sim_pred)
            F = jacobian_F_6d(ekf.x, u, sim_jac)
            ekf.predict(F, x_pred)
            T_pred = se2_to_mat(ekf.x[0], ekf.x[1], ekf.x[2])
            # Localizer used purely as the scan-to-map ICP front-end, seeded by the EKF
            # prediction (the odom_T args just seed/store its unused internal state).
            outcome = localizer.update(scan_base, T_pred, map_wp, T_pred)
            if outcome.status == "ok":
                zx, zy, zpsi = mat_to_se2(outcome.pose)
                ekf.update_icp(np.array([zx, zy, zpsi]))
            T_wb = se2_to_mat(ekf.x[0], ekf.x[1], ekf.x[2])
        else:
            # Baseline: seed ICP from the last accepted ICP pose; take the result directly.
            outcome = localizer.update(scan_base, T_wb, map_wp, T_wb)
            if outcome.status == "ok":
                T_wb = outcome.pose
        ex, ey, eyaw = mat_to_se2(T_wb)

        # --- localization error vs ground truth ---
        err_trans.append(float(np.hypot(ex - st.x, ey - st.y)))
        err_psi.append(abs(_wrap(eyaw - st.yaw)))

        # --- map accumulation (on the fused estimate) ---
        world_corrected = transform_points(scan_base, len(scan_base), T_wb)
        valid = wp.full(len(scan_base), 1, dtype=wp.int32, device=device)
        map_wp = acc.step(map_wp, None, world_corrected, valid, (ex, ey))

        # --- local + global heightmaps → MPPI terrain (identical to pipeline_sim) ---
        half = win_m / 2.0
        xmin, ymin = ex - half, ey - half
        ll = HeightMapBuilder(cell, (xmin, ex + half, ymin, ey + half), device=device).build(world_corrected)
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

    return dict(
        err_trans=np.asarray(err_trans), err_psi=np.asarray(err_psi),
        reached=reached, frames=f + 1, contacts=contacts,
    )


def sweep(
    q_scales: list[float],
    r_scales: list[float],
    world: str = "slalom",
    device: str = "cuda",
    max_frames: int = 340,
    seed: int = 0,
) -> dict:
    """Run the closed loop for every (q_scale, r_scale) on the grid, same world/seed.

    Q_mat = Q * q_scale, R_mat = R * r_scale. Fixed seed so error differences are due to the
    noise model, not sim randomness. Returns RMS translation / heading error grids
    [n_q, n_r] plus every run's per-frame series (for the overlaid curve plots).
    """
    n_q, n_r = len(q_scales), len(r_scales)
    rms_trans = np.full((n_q, n_r), np.nan)   # [m]
    rms_psi = np.full((n_q, n_r), np.nan)      # [rad]
    runs: dict[tuple[float, float], dict] = {}

    print("  [baseline] raw ICP (no EKF)")
    baseline = run_closed_loop_ekf(
        device=device, world=world, max_frames=max_frames, seed=seed, use_ekf=False,
    )
    print(
        f"  [baseline]  "
        f"RMS trans={np.sqrt(np.mean(baseline['err_trans'] ** 2)):.3f} m  "
        f"RMS ψ={np.rad2deg(np.sqrt(np.mean(baseline['err_psi'] ** 2))):.2f}°  "
        f"reached={baseline['reached']} frames={baseline['frames']} contacts={baseline['contacts']}"
    )

    for i, qs in enumerate(q_scales):
        for j, rs in enumerate(r_scales):
            res = run_closed_loop_ekf(
                device=device, world=world, Q_mat=Q * qs, R_mat=R * rs,
                max_frames=max_frames, seed=seed,
            )
            et, ep = res["err_trans"], res["err_psi"]
            rms_trans[i, j] = float(np.sqrt(np.mean(et ** 2)))
            rms_psi[i, j] = float(np.sqrt(np.mean(ep ** 2)))
            runs[(qs, rs)] = res
            print(
                f"  q_scale={qs:<6g} r_scale={rs:<6g}  "
                f"RMS trans={rms_trans[i, j]:.3f} m  RMS ψ={np.rad2deg(rms_psi[i, j]):.2f}°  "
                f"reached={res['reached']} frames={res['frames']} contacts={res['contacts']}"
            )

    return dict(
        q_scales=list(q_scales), r_scales=list(r_scales),
        rms_trans=rms_trans, rms_psi=rms_psi, runs=runs, baseline=baseline,
    )


def _heatmap(ax, grid, q_scales, r_scales, title, cbar_label, fmt):
    """Draw a Q-scale (rows) × R-scale (cols) heatmap, annotate cells, mark the min."""
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.figure.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(r_scales)), [f"{s:g}" for s in r_scales])
    ax.set_yticks(range(len(q_scales)), [f"{s:g}" for s in q_scales])
    ax.set_xlabel("R scale (ICP measurement noise)")
    ax.set_ylabel("Q scale (process noise)")
    bi, bj = np.unravel_index(np.nanargmin(grid), grid.shape)
    for i in range(len(q_scales)):
        for j in range(len(r_scales)):
            best = i == bi and j == bj
            ax.text(j, i, format(grid[i, j], fmt), ha="center", va="center",
                    color="white" if best else "0.85",
                    fontweight="bold" if best else "normal", fontsize=9)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="red", lw=2.5))
    ax.set_title(title)


def visualize(sw: dict, out: str, world: str) -> None:
    """2×2 figure: RMS heatmaps (translation, heading) + overlaid error-vs-frame curves."""
    q_scales, r_scales = sw["q_scales"], sw["r_scales"]
    rms_trans, rms_psi, runs = sw["rms_trans"], sw["rms_psi"], sw["runs"]
    baseline = sw.get("baseline")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    (ax_ht, ax_hp), (ax_ct, ax_cp) = axes

    _heatmap(ax_ht, rms_trans, q_scales, r_scales,
             "RMS translation error (red = best)", "RMS ‖Δxy‖ [m]", ".3f")
    _heatmap(ax_hp, np.rad2deg(rms_psi), q_scales, r_scales,
             "RMS heading error (red = best)", "RMS |Δψ| [deg]", ".2f")

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(runs)))
    for (key, res), col in zip(runs.items(), colors):
        qs, rs = key
        fr = np.arange(len(res["err_trans"]))
        lbl = f"Q×{qs:g}, R×{rs:g}"
        ax_ct.plot(fr, res["err_trans"], color=col, lw=1.3, label=lbl)
        ax_cp.plot(fr, np.rad2deg(res["err_psi"]), color=col, lw=1.3, label=lbl)

    if baseline is not None:
        fr_b = np.arange(len(baseline["err_trans"]))
        ax_ct.plot(fr_b, baseline["err_trans"], color="black", lw=2.0, ls="--",
                   label="baseline (raw ICP)", zorder=5)
        ax_cp.plot(fr_b, np.rad2deg(baseline["err_psi"]), color="black", lw=2.0, ls="--",
                   label="baseline (raw ICP)", zorder=5)

    for ax, ylabel, title in (
        (ax_ct, "translation error [m]", "Translation error vs frame"),
        (ax_cp, "heading error [deg]", "Heading error vs frame"),
    ):
        ax.set_xlabel("frame")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(alpha=0.3)

    bt = np.unravel_index(np.nanargmin(rms_trans), rms_trans.shape)
    bp = np.unravel_index(np.nanargmin(rms_psi), rms_psi.shape)
    fig.suptitle(
        f"EKF (Q, R) tuning sweep — world: {world}\n"
        f"best translation: Q×{q_scales[bt[0]]:g}, R×{r_scales[bt[1]]:g} "
        f"({rms_trans[bt]:.3f} m)   |   "
        f"best heading: Q×{q_scales[bp[0]]:g}, R×{r_scales[bp[1]]:g} "
        f"({np.rad2deg(rms_psi[bp]):.2f}°)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--world", choices=list(pipeline_sim._WORLDS), default="slalom")
    ap.add_argument("--max-frames", type=int, default=340)
    ap.add_argument("--q-scales", type=float, nargs="+", default=[0.1, 1.0, 10.0],
                    help="scalar multipliers applied to the baseline Q diagonal")
    ap.add_argument("--r-scales", type=float, nargs="+", default=[0.1, 1.0, 10.0],
                    help="scalar multipliers applied to the baseline R diagonal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/ekf_tuning.png")
    args = ap.parse_args()
    wp.init()

    print(
        f"EKF tuning sweep  world={args.world}  "
        f"{len(args.q_scales)}×{len(args.r_scales)} grid  max_frames={args.max_frames}"
    )
    sw = sweep(
        args.q_scales, args.r_scales, world=args.world, device=args.device,
        max_frames=args.max_frames, seed=args.seed,
    )
    bt = np.unravel_index(np.nanargmin(sw["rms_trans"]), sw["rms_trans"].shape)
    bp = np.unravel_index(np.nanargmin(sw["rms_psi"]), sw["rms_psi"].shape)
    print(
        f"BEST translation: Q×{args.q_scales[bt[0]]:g}, R×{args.r_scales[bt[1]]:g} "
        f"→ {sw['rms_trans'][bt]:.3f} m\n"
        f"BEST heading:     Q×{args.q_scales[bp[0]]:g}, R×{args.r_scales[bp[1]]:g} "
        f"→ {np.rad2deg(sw['rms_psi'][bp]):.2f}°"
    )
    visualize(sw, args.out, args.world)


if __name__ == "__main__":
    main()
