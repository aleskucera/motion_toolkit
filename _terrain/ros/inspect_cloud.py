"""Snapshot the next /terrain_map message and report NaN stats per field.

Run inside the apptainer + sourced dev-shell:

    python3 ~/projects/terrain_toolkit/ros/inspect_cloud.py
"""

import struct
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


def main() -> None:
    rclpy.init()
    node = Node("inspect_cloud")
    got: list[PointCloud2] = []

    node.create_subscription(
        PointCloud2,
        "/terrain_map",
        lambda m: got.append(m) if not got else None,
        10,
    )

    deadline = time.time() + 5.0
    while not got and time.time() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)

    if not got:
        print("no message received in 5 s — is the node publishing?")
        return

    msg = got[0]
    n_pts = msg.width
    field_names = [f.name for f in msg.fields]
    print(f"width = {n_pts}")
    print(f"fields = {field_names}")

    for f in msg.fields:
        if f.name in ("x", "y", "z"):
            continue
        nans = 0
        for i in range(n_pts):
            (v,) = struct.unpack_from("f", msg.data, i * msg.point_step + f.offset)
            if v != v:  # NaN
                nans += 1
        print(f"  {f.name:<20} nan = {nans:>5} / {n_pts}")

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
