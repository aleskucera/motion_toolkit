#!/usr/bin/env python3
"""Trajectory logger — reads map→base_link from /tf and writes x, y, ψ to a CSV.

Run inside the elevation-demo-local or ekf-demo tmuxinator session (ROS sourced).
Polls at 10 Hz (matching the node's cloud rate). On Ctrl-C: flushes the CSV and
opens a 3-panel matplotlib figure showing x(t), y(t), ψ(t), and the x-y track.

Usage:
    python3 ros/traj_logger.py [--map-frame map] [--base-frame base_link]
                               [--rate 10] [--out ~/bags/traj_TIMESTAMP.csv]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import pathlib
import sys

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
from rclpy.node import Node


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation about world z) from a unit quaternion."""
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _plot(rows: list[tuple[float, float, float, float]], out: pathlib.Path) -> None:
    """Show a 3-panel figure: x/y vs time, ψ vs time, x-y track."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[traj_logger] matplotlib not available — skipping plot.", file=sys.stderr)
        return

    if not rows:
        print("[traj_logger] no data recorded — nothing to plot.", file=sys.stderr)
        return

    arr = np.array(rows)  # (N, 4): t, x, y, psi
    t, x, y, psi = arr[:, 0], arr[:, 1], arr[:, 2], np.rad2deg(arr[:, 3])
    t -= t[0]  # time relative to first sample

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(f"Trajectory — {out.name}", fontsize=11)

    # Row 1: x and y vs time
    axes[0].plot(t, x, label="x [m]")
    axes[0].plot(t, y, label="y [m]")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("position [m]")
    axes[0].legend()
    axes[0].grid(True)

    # Row 2: ψ vs time
    axes[1].plot(t, psi, color="tab:orange", label="ψ [deg]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("yaw [deg]")
    axes[1].legend()
    axes[1].grid(True)

    # Row 3: x-y trajectory
    axes[2].plot(x, y, color="tab:green", linewidth=0.8)
    axes[2].plot(x[0], y[0], "o", color="tab:blue",  markersize=8, label="start")
    axes[2].plot(x[-1], y[-1], "s", color="tab:red",  markersize=8, label="end")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("y [m]")
    axes[2].set_aspect("equal")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    plt.show()


class TrajLogger(Node):
    def __init__(
        self,
        map_frame: str,
        base_frame: str,
        rate_hz: float,
        out: pathlib.Path,
    ) -> None:
        super().__init__("traj_logger")
        self._map_frame = map_frame
        self._base_frame = base_frame
        self._out = out
        self._rows: list[tuple[float, float, float, float]] = []

        self._buf = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._buf, self)

        period = 1.0 / rate_hz
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f"traj_logger: {map_frame}→{base_frame} @ {rate_hz} Hz  →  {out}"
        )

    def _tick(self) -> None:
        try:
            tf = self._buf.lookup_transform(
                self._map_frame, self._base_frame, rclpy.time.Time()
            )
        except tf2_ros.LookupException:
            return  # frame not yet available — wait silently
        except Exception as exc:
            self.get_logger().warn(f"TF lookup failed: {exc}", throttle_duration_sec=5.0)
            return

        stamp = tf.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        psi = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._rows.append((t, tx, ty, psi))

    def flush(self) -> None:
        """Write accumulated rows to CSV."""
        if not self._rows:
            self.get_logger().warn("No TF data received — CSV not written.")
            return
        self._out.parent.mkdir(parents=True, exist_ok=True)
        with self._out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_sec", "x_m", "y_m", "psi_rad"])
            writer.writerows(self._rows)
        self.get_logger().info(
            f"Wrote {len(self._rows)} rows to {self._out}"
        )

    @property
    def rows(self) -> list[tuple[float, float, float, float]]:
        return self._rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log map→base_link trajectory from /tf to CSV and plot on exit."
    )
    parser.add_argument("--map-frame",  default="map",       help="World/map frame (default: map)")
    parser.add_argument("--base-frame", default="base_link", help="Robot frame (default: base_link)")
    parser.add_argument("--rate",       type=float, default=10.0, help="Poll rate in Hz (default: 10)")
    parser.add_argument(
        "--out", default=None,
        help="CSV output path (default: ~/bags/traj_YYYYMMDD_HHMMSS.csv)",
    )
    # parse_known_args so any leftover ROS args (--ros-args ...) pass through to rclpy
    args, ros_args = parser.parse_known_args()

    out = pathlib.Path(
        args.out
        or pathlib.Path.home() / "bags" / f"traj_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    ).expanduser()

    rclpy.init(args=ros_args or None)
    node = TrajLogger(
        map_frame=args.map_frame,
        base_frame=args.base_frame,
        rate_hz=args.rate,
        out=out,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.flush()
        rows = node.rows
        node.destroy_node()
        rclpy.try_shutdown()

    _plot(rows, out)


if __name__ == "__main__":
    main()
