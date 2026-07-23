#!/usr/bin/env python3
"""EKF predict vs update logger — trajectory and/or covariance report.

Subscribes to:
  /ekf/pose_pred    — planar x/y/ψ after the EKF predict step (PoseStamped)      [traj]
  /ekf/odom         — fused pose after the ICP measurement update (Odometry)      [traj]
  /ekf/diagnostics  — per-frame scalar summary incl. covariance kv pairs          [cov]

On Ctrl-C: writes CSV(s) and opens matplotlib figure(s) for each active report.

Usage inside the ekf-demo tmuxinator session (ROS sourced):

    python3 ros/ekf_traj_logger.py                   # --report all (default)
    python3 ros/ekf_traj_logger.py --report traj
    python3 ros/ekf_traj_logger.py --report cov
"""

from __future__ import annotations

import argparse
import csv
import datetime
import pathlib
import sys
from typing import Any

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation about world z) from a unit quaternion."""
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _stamp_key(stamp: Any) -> int:
    """Nanosecond integer key for exact stamp pairing."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _longest_lap(arr: np.ndarray) -> np.ndarray:
    """Return the longest contiguous segment with monotonically advancing timestamps.

    The bag is played with --loop, which resets ROS time, producing a large negative
    dt jump between laps.  Matplotlib draws a cross-plot connector between the last
    sample of lap N and the first of lap N+1 unless we clip to one clean segment.
    """
    t = arr[:, 0]
    dt = np.diff(t)
    rewind_idx = np.flatnonzero(dt < 0)
    boundaries = np.concatenate([[0], rewind_idx + 1, [len(t)]])
    best_start, best_end = 0, len(t)
    best_len = 0
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e - s > best_len:
            best_len = e - s
            best_start, best_end = s, e
    return arr[best_start:best_end]


def _plot_series(
    ax: Any,
    t: np.ndarray,
    pred: np.ndarray,
    upd: np.ndarray,
    ylabel: str,
    color: str,
    pred_label: str,
    upd_label: str,
    xlabel: str = "",
) -> None:
    """Plot one predict (dashed) + update (solid) time series on ax."""
    ax.plot(t, pred, "--", color=color, alpha=0.7, linewidth=1.0, label=pred_label)
    ax.plot(t, upd, "-", color=color, linewidth=1.3, label=upd_label)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)


# ---------------------------------------------------------------------------
# Trajectory report
# ---------------------------------------------------------------------------


def _plot_traj(
    rows: list[tuple[float, float, float, float, float, float, float]],
    out: pathlib.Path,
) -> None:
    """4-panel figure: x(t), y(t), ψ(t) predict vs update, plus xy track.

    Handles bag --loop time rewinds (keeps longest lap) and yaw ±π wraps.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ekf_traj_logger] matplotlib not available — skipping traj plot.", file=sys.stderr)
        return

    if not rows:
        print("[ekf_traj_logger] no traj data recorded — nothing to plot.", file=sys.stderr)
        return

    arr = _longest_lap(np.asarray(rows, dtype=np.float64))
    t = arr[:, 0] - arr[0, 0]
    x_p, y_p = arr[:, 1], arr[:, 2]
    x_u, y_u = arr[:, 4], arr[:, 5]

    # Unwrap yaw before converting so ±π boundaries don't produce vertical spikes.
    psi_p = np.rad2deg(np.unwrap(arr[:, 3]))
    psi_u = np.rad2deg(np.unwrap(arr[:, 6]))

    # Corrections: frames where ICP moved the pose more than 5 cm from predict.
    dxy = np.hypot(x_u - x_p, y_u - y_p)
    corr_mask = dxy > 0.05

    fig, axes = plt.subplots(
        4, 1, figsize=(11, 12), gridspec_kw={"height_ratios": [1, 1, 1, 1.6]}
    )
    fig.suptitle(f"EKF predict vs update — {out.name}", fontsize=11)

    _plot_series(axes[0], t, x_p, x_u, "x [m]", "tab:blue", "x predict", "x update (TF)")
    _plot_series(axes[1], t, y_p, y_u, "y [m]", "tab:green", "y predict", "y update (TF)")
    _plot_series(
        axes[2], t, psi_p, psi_u, "ψ [deg]", "tab:orange",
        "ψ predict", "ψ update (TF)", xlabel="time [s]",
    )

    ax = axes[3]
    ax.plot(x_p, y_p, "--", color="0.55", linewidth=0.8, label="predict path")
    ax.plot(x_u, y_u, "-", color="tab:purple", linewidth=1.0, label="update path (TF)")
    if np.any(corr_mask):
        for i in np.flatnonzero(corr_mask):
            ax.plot([x_p[i], x_u[i]], [y_p[i], y_u[i]], "-", color="tab:red",
                    linewidth=0.7, alpha=0.6)
        ax.plot(
            x_u[corr_mask], y_u[corr_mask], ".",
            color="tab:red", markersize=3,
            label=f"|update−predict| > 5 cm  (n={int(corr_mask.sum())})",
        )
    ax.plot(x_u[0], y_u[0], "o", color="tab:blue", markersize=7, label="start")
    ax.plot(x_u[-1], y_u[-1], "s", color="tab:red", markersize=7, label="end")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linewidth=0.4)

    fig.tight_layout()
    plt.show()


class EkfTrajLogger(Node):
    def __init__(self, out: pathlib.Path) -> None:
        super().__init__("ekf_traj_logger")
        self._out = out
        # stamp_ns → (t, x, y, psi) from pose_pred, waiting for matching odom.
        self._pending_pred: dict[int, tuple[float, float, float, float]] = {}
        self._rows: list[tuple[float, float, float, float, float, float, float]] = []

        self.create_subscription(PoseStamped, "ekf/pose_pred", self._on_pred, 50)
        self.create_subscription(Odometry, "ekf/odom", self._on_odom, 50)
        self.get_logger().info(f"traj: /ekf/pose_pred + /ekf/odom  →  {out}")

    def _on_pred(self, msg: PoseStamped) -> None:
        key = _stamp_key(msg.header.stamp)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.position
        q = msg.pose.orientation
        psi = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._pending_pred[key] = (t, float(p.x), float(p.y), psi)

    def _on_odom(self, msg: Odometry) -> None:
        key = _stamp_key(msg.header.stamp)
        pred = self._pending_pred.pop(key, None)
        if pred is None:
            return  # odom without matching predict (e.g. bootstrap-only frames)
        t, x_p, y_p, psi_p = pred
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        psi_u = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._rows.append((t, x_p, y_p, psi_p, float(p.x), float(p.y), psi_u))
        # Drop stale unmatched predicts so a long run cannot grow without bound.
        if len(self._pending_pred) > 30:
            for old in sorted(self._pending_pred)[:-10]:
                del self._pending_pred[old]

    def flush(self) -> None:
        """Write accumulated paired rows to CSV."""
        if not self._rows:
            self.get_logger().warn("No paired predict/update frames — traj CSV not written.")
            return
        self._out.parent.mkdir(parents=True, exist_ok=True)
        with self._out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["t_sec", "x_pred_m", "y_pred_m", "psi_pred_rad",
                 "x_upd_m", "y_upd_m", "psi_upd_rad"]
            )
            writer.writerows(self._rows)
        self.get_logger().info(f"Wrote {len(self._rows)} traj frames to {self._out}")

    @property
    def rows(self) -> list[tuple[float, float, float, float, float, float, float]]:
        return self._rows


# ---------------------------------------------------------------------------
# Covariance report
# ---------------------------------------------------------------------------

# State names matching the EKF state vector [x, y, ψ, vx, vy, ψ̇].
_COV_STATE_NAMES = ["x", "y", "psi", "vx", "vy", "psidot"]

# Row layout: (t, pred_x, pred_y, pred_psi, pred_vx, pred_vy, pred_psidot,
#              upd_x,  upd_y,  upd_psi,  upd_vx,  upd_vy,  upd_psidot,
#              pred_trace, pred_logdet, upd_trace, upd_logdet, updated,
#              nis, innov_x_m, innov_y_m, innov_yaw_deg)
# Indices 0: t, 1-6: pred diag, 7-12: upd diag, 13-16: scalars, 17: updated,
#           18: nis, 19-21: innovations (NaN when no ICP update this frame)
_COV_NCOLS = 22


def _plot_cov(rows: list[tuple], out: pathlib.Path) -> None:
    """7-panel covariance figure: variance diagonals, PSD size, innovations, NIS.

    Predict is drawn dashed, posterior (after ICP update) solid.  Only the longest
    monotonic time segment is plotted to avoid bag-loop artefacts.

    Panels:
      1 — position variances σ²_x, σ²_y  [m²]
      2 — heading variance σ²_ψ  [deg²]
      3 — velocity variances σ²_vx, σ²_vy, σ²_ψ̇
      4 — PSD size: trace(P) and log|det(P)|, predict vs posterior
      5 — innovations innov_x [m] and innov_y [m]  (ICP update frames only)
      6 — heading innovation innov_ψ [deg]  (ICP update frames only)
      7 — NIS scalar with χ²(3) reference bounds  (ICP update frames only)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ekf_traj_logger] matplotlib not available — skipping cov plot.", file=sys.stderr)
        return

    if not rows:
        print("[ekf_traj_logger] no cov data recorded — nothing to plot.", file=sys.stderr)
        return

    arr = _longest_lap(np.asarray(rows, dtype=np.float64))
    t = arr[:, 0] - arr[0, 0]

    # Diagonal variances — indices 1..12
    pred_diag = arr[:, 1:7]   # columns: x, y, psi, vx, vy, psidot
    upd_diag  = arr[:, 7:13]

    pred_trace  = arr[:, 13]
    pred_logdet = arr[:, 14]
    upd_trace   = arr[:, 15]
    upd_logdet  = arr[:, 16]

    # Innovations and NIS — NaN on predict-only frames.
    nis        = arr[:, 18]
    innov_x    = arr[:, 19]
    innov_y    = arr[:, 20]
    innov_yaw  = arr[:, 21]

    fig, axes = plt.subplots(7, 1, figsize=(11, 20), sharex=True)
    fig.suptitle(f"EKF covariance — P⁻ (predict) vs P⁺ (posterior) — {out.name}", fontsize=11)

    # --- Panel 1: position variances σ²_x, σ²_y ---
    ax = axes[0]
    ax.plot(t, pred_diag[:, 0], "--", color="tab:blue",  linewidth=1.0, alpha=0.7, label="σ²_x pred")
    ax.plot(t, upd_diag[:, 0],  "-",  color="tab:blue",  linewidth=1.3,            label="σ²_x upd")
    ax.plot(t, pred_diag[:, 1], "--", color="tab:green", linewidth=1.0, alpha=0.7, label="σ²_y pred")
    ax.plot(t, upd_diag[:, 1],  "-",  color="tab:green", linewidth=1.3,            label="σ²_y upd")
    ax.set_ylabel("position var [m²]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    # --- Panel 2: heading variance σ²_ψ (converted rad² → deg²) ---
    # EKF stores ψ variance in rad²; multiply by (180/π)² to get deg².
    _r2d2 = np.rad2deg(1.0) ** 2
    ax = axes[1]
    ax.plot(t, pred_diag[:, 2] * _r2d2, "--", color="tab:orange",
            linewidth=1.0, alpha=0.7, label="σ²_ψ pred")
    ax.plot(t, upd_diag[:, 2]  * _r2d2, "-",  color="tab:orange",
            linewidth=1.3,            label="σ²_ψ upd")
    ax.set_ylabel("heading var [deg²]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    # --- Panel 3: velocity variances σ²_vx, σ²_vy, σ²_ψ̇ ---
    ax = axes[2]
    for col, clr, lbl in zip(
        [3, 4, 5],
        ["tab:red", "tab:purple", "tab:brown"],
        ["vx", "vy", "ψ̇"],
    ):
        ax.plot(t, pred_diag[:, col], "--", color=clr, linewidth=1.0, alpha=0.7,
                label=f"σ²_{lbl} pred")
        ax.plot(t, upd_diag[:, col],  "-",  color=clr, linewidth=1.3,
                label=f"σ²_{lbl} upd")
    ax.set_ylabel("velocity var [m²/s²\nor rad²/s²]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    # --- Panel 4: PSD size — trace and log-det ---
    ax = axes[3]
    ax.plot(t, pred_trace,  "--", color="tab:blue",   linewidth=1.0, alpha=0.7, label="trace(P⁻)")
    ax.plot(t, upd_trace,   "-",  color="tab:blue",   linewidth=1.3,            label="trace(P⁺)")
    ax.set_ylabel("trace(P)")
    ax.grid(True, linewidth=0.4)
    ax2 = ax.twinx()
    ax2.plot(t, pred_logdet, "--", color="tab:red",  linewidth=1.0, alpha=0.7, label="log|det(P⁻)|")
    ax2.plot(t, upd_logdet,  "-",  color="tab:red",  linewidth=1.3,            label="log|det(P⁺)|")
    ax2.set_ylabel("log|det(P)|")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # --- Panel 5: position innovations innov_x, innov_y [m] (ICP update frames only) ---
    # NaN on frames with no ICP update → matplotlib skips them, leaving gaps that
    # correctly reflect predict-only frames (no measurement → no innovation).
    ax = axes[4]
    ax.plot(t, innov_x, ".", color="tab:blue",  markersize=3, label="innov_x [m]")
    ax.plot(t, innov_y, ".", color="tab:green", markersize=3, label="innov_y [m]")
    ax.axhline(0, color="0.6", linewidth=0.6, linestyle="--")
    ax.set_ylabel("innovation [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    # --- Panel 6: heading innovation [deg] (ICP update frames only) ---
    ax = axes[5]
    ax.plot(t, innov_yaw, ".", color="tab:orange", markersize=3, label="innov_ψ [deg]")
    ax.axhline(0, color="0.6", linewidth=0.6, linestyle="--")
    ax.set_ylabel("innov_ψ [deg]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    # --- Panel 7: NIS scalar with χ²(3) reference bounds (ICP update frames only) ---
    # NIS ~ χ²(3) when the filter is consistent.  Mean = 3, 95th = 7.81, 99th = 11.34.
    ax = axes[6]
    ax.plot(t, nis, ".", color="tab:purple", markersize=3, label="NIS")
    ax.axhline(3.0,  color="tab:green", linewidth=0.8, linestyle="--", label="mean χ²(3) = 3")
    ax.axhline(7.81, color="tab:orange", linewidth=0.8, linestyle="--", label="95th pct = 7.81")
    ax.axhline(11.34, color="tab:red",  linewidth=0.8, linestyle="--", label="99th pct = 11.34")
    ax.set_ylabel("NIS")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    fig.tight_layout()
    plt.show()


class EkfCovLogger(Node):
    """Subscribes to ekf/diagnostics and buffers per-frame covariance diagonal + PSD scalars."""

    def __init__(self, out: pathlib.Path) -> None:
        super().__init__("ekf_cov_logger")
        self._out = out
        self._rows: list[tuple] = []
        self.create_subscription(DiagnosticArray, "ekf/diagnostics", self._on_diag, 50)
        self.get_logger().info(f"cov: /ekf/diagnostics  →  {out}")

    def _on_diag(self, msg: DiagnosticArray) -> None:
        if not msg.status:
            return
        kv_map = {kv.key: kv.value for kv in msg.status[0].values}
        # Skip frames that predate the covariance fields (old bags or publish_ekf_debug=false).
        if "cov_pred_x" not in kv_map:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            pred = [float(kv_map[f"cov_pred_{n}"]) for n in _COV_STATE_NAMES]
            upd  = [float(kv_map[f"cov_upd_{n}"])  for n in _COV_STATE_NAMES]
            pred_trace  = float(kv_map["cov_pred_trace"])
            pred_logdet = float(kv_map["cov_pred_logdet"])
            upd_trace   = float(kv_map["cov_upd_trace"])
            upd_logdet  = float(kv_map["cov_upd_logdet"])
            updated     = float(kv_map["cov_updated"])
        except (KeyError, ValueError):
            return
        # Innovations and NIS are "n/a" on predict-only frames (no ICP update).
        # Store NaN so matplotlib naturally leaves gaps at those time steps.
        _nan = float("nan")

        def _fv(key: str) -> float:
            v = kv_map.get(key, "n/a")
            return float(v) if v != "n/a" else _nan

        nis       = _fv("nis")
        innov_x   = _fv("innov_x_m")
        innov_y   = _fv("innov_y_m")
        innov_yaw = _fv("innov_yaw_deg")
        self._rows.append(
            (t, *pred, *upd, pred_trace, pred_logdet, upd_trace, upd_logdet,
             updated, nis, innov_x, innov_y, innov_yaw)
        )

    def flush(self) -> None:
        if not self._rows:
            self.get_logger().warn(
                "No covariance frames — cov CSV not written.  "
                "Check that publish_ekf_debug is true on the node."
            )
            return
        self._out.parent.mkdir(parents=True, exist_ok=True)
        header = (
            ["t_sec"]
            + [f"pred_{n}" for n in _COV_STATE_NAMES]
            + [f"upd_{n}"  for n in _COV_STATE_NAMES]
            + ["pred_trace", "pred_logdet", "upd_trace", "upd_logdet", "updated",
               "nis", "innov_x_m", "innov_y_m", "innov_yaw_deg"]
        )
        with self._out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self._rows)
        self.get_logger().info(f"Wrote {len(self._rows)} cov frames to {self._out}")

    @property
    def rows(self) -> list[tuple]:
        return self._rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Log EKF state and/or covariance and plot on exit.  "
            "--report traj|cov|all (default: all)"
        )
    )
    parser.add_argument("--out", default=None, help="Base path for CSV output (no extension).")
    parser.add_argument(
        "--report",
        choices=["traj", "cov", "all"],
        default="all",
        help="Which reports to produce: traj (x/y/ψ timeline), cov (covariance), all (both).",
    )
    args, ros_args = parser.parse_known_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = pathlib.Path.home() / "bags"

    do_traj = args.report in ("traj", "all")
    do_cov  = args.report in ("cov",  "all")

    if args.out:
        base = pathlib.Path(args.out).expanduser()
        traj_out = base.with_suffix("") if do_traj else None
        cov_out  = base.with_suffix("") if do_cov  else None
        if do_traj and do_cov:
            traj_out = pathlib.Path(str(base) + "_traj.csv")
            cov_out  = pathlib.Path(str(base) + "_cov.csv")
        elif do_traj:
            traj_out = base.with_suffix(".csv")
        else:
            cov_out = base.with_suffix(".csv")
    else:
        traj_out = base_dir / f"ekf_traj_{ts}.csv" if do_traj else None
        cov_out  = base_dir / f"ekf_cov_{ts}.csv"  if do_cov  else None

    rclpy.init(args=ros_args or None)

    traj_node: EkfTrajLogger | None = None
    cov_node:  EkfCovLogger  | None = None

    if do_traj:
        traj_node = EkfTrajLogger(out=traj_out)
    if do_cov:
        cov_node = EkfCovLogger(out=cov_out)

    executor = rclpy.executors.SingleThreadedExecutor()
    if traj_node:
        executor.add_node(traj_node)
    if cov_node:
        executor.add_node(cov_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if traj_node:
            traj_node.flush()
        if cov_node:
            cov_node.flush()
        traj_rows = traj_node.rows if traj_node else []
        cov_rows  = cov_node.rows  if cov_node  else []
        executor.shutdown()
        rclpy.try_shutdown()

    if do_traj:
        _plot_traj(traj_rows, traj_out)
    if do_cov:
        _plot_cov(cov_rows, cov_out)


if __name__ == "__main__":
    main()
