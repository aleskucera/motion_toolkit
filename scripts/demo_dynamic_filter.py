"""Animated demo: filtering a moving person out of the accumulated map.

Scenario — a STATIONARY sensor at the origin watches a person walk laterally
across flat ground in front of a back wall. Each frame we simulate one occluded
LiDAR scan (nearest surface per bearing), then accumulate it two ways:

  * filter OFF — every scan is fused as-is, so the person smears a solid trail
    across the map (and shows up as a ridge of obstacle in the heightmap);
  * filter ON  — `DynamicPointFilter` drops the person (they sit in front of the
    known static wall/ground) and carves the ghost they leave behind, so the map
    stays clean.

Renders a 2x2 animation (accumulated cloud + elevation heightmap, OFF vs ON) to a
GIF. Headless (Agg backend), matching motion_toolkit's viz stack (matplotlib +
pillow).

Run: python scripts/demo_dynamic_filter.py [out.gif]
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from terrain_toolkit import DynamicFilterConfig  # noqa: E402
from terrain_toolkit import DynamicPointFilter  # noqa: E402
from terrain_toolkit import TerrainPipeline  # noqa: E402
from terrain_toolkit import VoxelGrid  # noqa: E402

# Demo-side numpy boundary around the device-native VoxelGrid.
_VG: dict[tuple, VoxelGrid] = {}


def voxel_downsample(points: np.ndarray, voxel_size: float, *, device) -> np.ndarray:
    if len(points) == 0:
        return points
    key = (round(voxel_size, 4), str(device))
    vg = _VG.get(key)
    if vg is None or vg.max_points < len(points):
        vg = VoxelGrid(voxel_size, max_points=max(len(points), 200_000), device=device)
        _VG[key] = vg
    pw = wp.array(np.ascontiguousarray(points, np.float32), dtype=wp.vec3, device=device)
    ds, n = vg.downsample(pw, len(points))
    return ds.numpy()[:n].astype(points.dtype, copy=False)

SENSOR = np.array([0.0, 0.0, 0.5])  # stationary sensor origin
WALL_X = 10.0  # back wall distance (m)
PERSON_X = 5.0  # person's distance from the sensor (m)
N_FRAMES = 40
BOUNDS = (0.0, 12.0, -6.0, 6.0)  # heightmap window (xmin, xmax, ymin, ymax)
RESOLUTION = 0.15
VOXEL = 0.1
EL_MIN, EL_MAX = -0.9, 0.9  # sensor vertical band (rad) ≈ ±51°


def _ground(n: int = 9000) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, WALL_X, n)
    y = rng.uniform(BOUNDS[2], BOUNDS[3], n)
    return np.stack([x, y, np.zeros(n)], axis=1)


def _wall(n: int = 5000) -> np.ndarray:
    rng = np.random.default_rng(1)
    y = rng.uniform(BOUNDS[2], BOUNDS[3], n)
    z = rng.uniform(0.0, 2.0, n)
    return np.stack([np.full(n, WALL_X), y, z], axis=1)


def _person(y_center: float, n: int = 1600) -> np.ndarray:
    rng = np.random.default_rng(2)
    x = rng.uniform(PERSON_X - 0.22, PERSON_X + 0.22, n)
    y = rng.uniform(y_center - 0.22, y_center + 0.22, n)
    z = rng.uniform(0.0, 1.8, n)  # a ~1.8 m tall column
    return np.stack([x, y, z], axis=1)


def _occlude(points: np.ndarray, n_az: int = 720, n_el: int = 200) -> np.ndarray:
    """Simulate one LiDAR return per bearing: keep the nearest point per (az, el) bin."""
    d = points - SENSOR
    r = np.linalg.norm(d, axis=1)
    az = np.arctan2(d[:, 1], d[:, 0])
    el = np.arcsin(np.clip(d[:, 2] / np.maximum(r, 1e-6), -1.0, 1.0))
    m = (el >= EL_MIN) & (el <= EL_MAX) & (r > 1e-3)
    r, az, el, pts = r[m], az[m], el[m], points[m]

    col = np.clip(((az + np.pi) / (2 * np.pi) * n_az).astype(int), 0, n_az - 1)
    row = np.clip(((el - EL_MIN) / (EL_MAX - EL_MIN) * n_el).astype(int), 0, n_el - 1)
    bins = row * n_az + col
    # Sort by range so np.unique keeps the nearest hit as each bin's first occurrence.
    order = np.argsort(r)
    _, first = np.unique(bins[order], return_index=True)
    return pts[order[first]]


def _simulate() -> list[dict]:
    """Run the accumulation loop both ways; return per-frame snapshots."""
    device = wp.get_device("cpu")
    filt = DynamicPointFilter(
        DynamicFilterConfig(
            az_bins=360,
            el_bins=140,
            el_min_deg=np.degrees(EL_MIN),
            el_max_deg=np.degrees(EL_MAX),
            margin_m=0.2,
            margin_rel=0.02,
            min_range_m=0.3,
        ),
        device=device,
    )
    pipe = TerrainPipeline(
        resolution=RESOLUTION,
        bounds=BOUNDS,
        z_max=3.0,
        primary="max",
        inpaint=True,
        smooth_sigma=0.0,
        device="cpu",
    )

    static = np.vstack([_ground(), _wall()])
    ys = np.linspace(-3.0, 3.0, N_FRAMES)

    map_off = np.empty((0, 3))
    map_on = np.empty((0, 3))
    frames: list[dict] = []

    for k, y in enumerate(ys):
        scan = _occlude(np.vstack([static, _person(y)]))

        # filter OFF: fuse the raw scan.
        map_off = voxel_downsample(np.vstack([map_off, scan]), VOXEL, device=device)

        # filter ON: drop dynamic scan points, carve stale map ghosts, then fuse.
        dropped = carved = 0
        if len(map_on) > 0:
            scan_keep, map_keep = filt.filter(map_on, scan, SENSOR)
            dropped = int((~scan_keep).sum())
            carved = int((~map_keep).sum())
            map_on = map_on[map_keep]
            scan = scan[scan_keep]
        map_on = voxel_downsample(np.vstack([map_on, scan]), VOXEL, device=device)

        elev_off = pipe.process(map_off).elevation
        elev_on = pipe.process(map_on).elevation
        frames.append(
            {
                "y": float(y),
                "off": map_off.copy(),
                "on": map_on.copy(),
                "elev_off": elev_off,
                "elev_on": elev_on,
            }
        )
        print(
            f"frame {k:2d}: person y={y:+.2f}  scan_dropped={dropped:4d}  map_carved={carved:4d}  "
            f"|map_off|={len(map_off):6d}  |map_on|={len(map_on):6d}"
        )

    return frames


def _animate(frames: list[dict], out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Dynamic obstacle filtering — a person walking past a stationary sensor")
    (ax_c_off, ax_c_on), (ax_h_off, ax_h_on) = axes

    def _scatter(ax, pts, title):
        ax.clear()
        s = pts[:: max(1, len(pts) // 6000)]
        ax.scatter(s[:, 0], s[:, 1], c=s[:, 2], s=2, cmap="viridis", vmin=0.0, vmax=2.0)
        ax.scatter(*SENSOR[:2], c="red", marker="^", s=80, label="sensor")
        ax.set_xlim(0, 12)
        ax.set_ylim(-6, 6)
        ax.set_aspect("equal")
        ax.set_title(title)

    def _heat(ax, elev, title):
        ax.clear()
        ax.imshow(
            elev,
            origin="lower",
            extent=BOUNDS,
            cmap="terrain",
            vmin=0.0,
            vmax=2.0,
            aspect="equal",
        )
        ax.set_title(title)

    def update(i):
        f = frames[i]
        _scatter(ax_c_off, f["off"], f"accumulated cloud — filter OFF (frame {i})")
        _scatter(ax_c_on, f["on"], "accumulated cloud — filter ON")
        # Mark the person's true current position.
        for ax in (ax_c_off, ax_c_on):
            ax.scatter(PERSON_X, f["y"], c="crimson", marker="x", s=90)
        _heat(ax_h_off, f["elev_off"], "elevation heightmap — filter OFF")
        _heat(ax_h_on, f["elev_on"], "elevation heightmap — filter ON")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=120)
    anim.save(out_path, writer=PillowWriter(fps=8))
    plt.close(fig)


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "dynamic_filter_demo.gif"
    wp.init()
    frames = _simulate()
    _animate(frames, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
