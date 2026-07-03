"""Animated demo: a ray-cast LiDAR watching a person walk across a scene.

Two panels, animated as the person walks:
  * left  — the ground-truth scene (ground + back wall + person, densely sampled);
  * right — the simulated LiDAR point cloud: one return per beam, nearest surface
    only, so the person casts a real occlusion shadow on the wall/ground behind.

The point of this demo is to eyeball the sensor model itself (beam structure,
occlusion) before it feeds anything else. Top-down view (x right, y up), colored
by height, so the shadow wedge behind the person is obvious.

Headless (Agg backend) → GIF, matching motion_toolkit's matplotlib + pillow viz.

Run: python scripts/demo_lidar.py [out.gif]
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from helhest.perception.sim import GroundSpec  # noqa: E402
from helhest.perception.sim import PrimitiveLidar  # noqa: E402

SENSOR = np.array([0.0, 0.0, 0.5])  # stationary sensor origin
GROUND_X = (1.0, 10.0)  # ground extent in x (m)
Y_RANGE = (-6.0, 6.0)  # scene extent in y (m)
WALL_X = 10.0  # back wall plane (m)
WALL_Z = (0.0, 2.0)  # wall height band (m)
PERSON_X = 5.0  # person distance from sensor (m)
PERSON_HALF = 0.25  # person half-width (m)
PERSON_H = 1.8  # person height (m)
N_FRAMES = 40

# LiDAR beam grid (a forward arc of a spinning sensor).
N_AZ = 260
N_EL = 72
AZ_RANGE = np.deg2rad((-80.0, 80.0))
EL_RANGE = np.deg2rad((-30.0, 20.0))

# Sensor noise model.
RANGE_NOISE_STD = 0.02  # Gaussian range noise along the beam (m)
DROPOUT_PROB = 0.08  # fraction of beams that return nothing (low reflectivity, etc.)


def _beam_dirs() -> np.ndarray:
    """Unit direction of every beam (B, 3), sensor looking down +x."""
    az = np.linspace(*AZ_RANGE, N_AZ)
    el = np.linspace(*EL_RANGE, N_EL)
    az, el = np.meshgrid(az, el)
    az, el = az.ravel(), el.ravel()
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)


def _boxes(person_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Obstacle AABBs: a thin back wall + the person, as (M, 3) lo/hi corners."""
    lo = np.array(
        [
            [WALL_X, Y_RANGE[0], WALL_Z[0]],  # wall
            [PERSON_X - PERSON_HALF, person_y - PERSON_HALF, 0.0],  # person
        ]
    )
    hi = np.array(
        [
            [WALL_X + 0.1, Y_RANGE[1], WALL_Z[1]],
            [PERSON_X + PERSON_HALF, person_y + PERSON_HALF, PERSON_H],
        ]
    )
    return lo, hi


def _scene_points(person_y: float) -> np.ndarray:
    """Ground-truth geometry, densely sampled (for the left panel)."""
    rng = np.random.default_rng(0)
    n = 9000
    gx = rng.uniform(*GROUND_X, n)
    gy = rng.uniform(*Y_RANGE, n)
    ground = np.stack([gx, gy, np.zeros(n)], axis=1)

    m = 5000
    wy = rng.uniform(*Y_RANGE, m)
    wz = rng.uniform(*WALL_Z, m)
    wall = np.stack([np.full(m, WALL_X), wy, wz], axis=1)

    k = 1600
    px = rng.uniform(PERSON_X - PERSON_HALF, PERSON_X + PERSON_HALF, k)
    py = rng.uniform(person_y - PERSON_HALF, person_y + PERSON_HALF, k)
    pz = rng.uniform(0.0, PERSON_H, k)
    person = np.stack([px, py, pz], axis=1)
    return np.vstack([ground, wall, person])


def _animate(out_path: str) -> None:
    lidar = PrimitiveLidar(
        _beam_dirs(),
        ground=GroundSpec(z=0.0, x_range=GROUND_X, y_range=Y_RANGE),
        noise_std=RANGE_NOISE_STD,
        dropout=DROPOUT_PROB,
    )
    ys = np.linspace(-3.0, 3.0, N_FRAMES)

    fig, (ax_scene, ax_scan) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Ray-cast LiDAR — ground-truth scene vs simulated point cloud (top-down)")

    def draw(ax, pts, title):
        ax.clear()
        ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=3, cmap="viridis", vmin=0.0, vmax=2.0)
        ax.scatter(*SENSOR[:2], c="red", marker="^", s=90)
        ax.set_xlim(-1, 12)
        ax.set_ylim(*Y_RANGE)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)

    def update(i):
        y = ys[i]
        scene = _scene_points(y)
        lo, hi = _boxes(y)
        # Seed per frame so noise/dropout vary frame-to-frame but stay reproducible.
        scan = lidar.scan(SENSOR, 0.0, lo, hi, seed=1000 + i)
        draw(ax_scene, scene, f"scene (ground truth) — frame {i}")
        draw(ax_scan, scan, f"simulated LiDAR — {len(scan)} returns")
        # Person's true center.
        for ax in (ax_scene, ax_scan):
            ax.scatter(PERSON_X, y, c="crimson", marker="x", s=110)
        print(f"frame {i:2d}: person y={y:+.2f}  returns={len(scan)}")
        return []

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=120)
    anim.save(out_path, writer=PillowWriter(fps=8))
    plt.close(fig)


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "lidar_demo.gif"
    _animate(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
