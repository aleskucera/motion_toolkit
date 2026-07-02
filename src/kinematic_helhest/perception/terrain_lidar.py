"""terrain_toolkit real-sensor front-end -- a drop-in replacement for lidar_scan.

Swaps the synthetic 2.5D horizon sweep for terrain_toolkit's 3D ray-cast: an Ouster OSDome
(the same sensor as terrain_toolkit's scripts/drive_lidar.py) casts against a per-cell AABB
decomposition of the world (faithful for ANY heightmap, walls or bumpy terrain), and the
returned point cloud is rasterized to the SAME (obs[ny,nx], known[ny,nx]) grid the perception
pipeline consumes. Frame convention is the shared min-corner / cell-center one (see
perception/rasterize.py, engine/terrain.py), so the output drops straight into MultiScanMap /
crop_window / drift_scan with no shift.

terrain_toolkit is an OPTIONAL dependency (the `perception` extra) -- only import this module
when you actually want the real front-end.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import warp as wp
from terrain_toolkit.sim import GroundSpec
from terrain_toolkit.sim import make_osdome_lidar
from terrain_toolkit.sim import osdome_sensor_config

from ..heightmap import Heightmap
from .rasterize import rasterize


class TerrainLidar:
    """Real-sensor replacement for lidar_scan: OSDome ray-cast -> rasterize -> (obs, known).

    Same call shape as lidar_scan (one scan from a world pose), so a caller swaps the two with a
    single branch. The sensor is an Ouster OSDome (Rev7), front-facing -- a 180deg hemisphere of
    `channels` uniform beams along the heading, with the datasheet range window (0.5-45 m) and
    distance-dependent range precision -- matching terrain_toolkit's drive_lidar.py. Pass
    `max_range_m` to clamp the datasheet 45 m to something tighter for small worlds."""

    def __init__(
        self,
        scene: Heightmap,
        *,
        mount_height: float = 0.6,  # matches drive_lidar.py SENSOR_Z
        channels: int = 128,
        cols: int = 1024,
        dropout: float = 0.03,
        max_range_m: float | None = None,
        device: str = "cuda",
    ) -> None:
        self.scene = scene
        self.mount_height = float(mount_height)
        ny, nx = scene.H.shape
        self.ny, self.nx = ny, nx
        self.x0, self.y0, self.cell = scene.x0, scene.y0, scene.cell
        bounds = (self.x0, self.x0 + nx * self.cell, self.y0, self.y0 + ny * self.cell)

        # world -> per-cell AABBs (each occupied cell a box rising z=0..height), built once.
        ri, ci = np.nonzero(scene.H > 1e-3)
        xc = self.x0 + (ci + 0.5) * self.cell  # engine cell-center (min-corner convention)
        yc = self.y0 + (ri + 0.5) * self.cell
        z = scene.H[ri, ci].astype(np.float32)
        half = self.cell / 2.0
        self._boxes_lo = np.stack([xc - half, yc - half, np.zeros_like(z)], 1).astype(np.float32)
        self._boxes_hi = np.stack([xc + half, yc + half, z], 1).astype(np.float32)

        sensor = osdome_sensor_config(columns=cols, channels=channels)
        if max_range_m is not None:
            sensor = dataclasses.replace(sensor, max_range_m=float(max_range_m))
        self._lidar = make_osdome_lidar(
            GroundSpec(z=0.0, x_range=(bounds[0], bounds[1]), y_range=(bounds[2], bounds[3])),
            sensor=sensor,
            dropout=dropout,
            device=wp.get_device(device),
        )
        self._seed = 0

    def scan(self, pose: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        """One scan from world pose (x, y, yaw) -> (obs[ny,nx] float32, known[ny,nx] bool),
        matching lidar_scan's output and frame. Sensor sits mount_height above local ground.
        The raw hit cloud + sensor height are stashed on `last_points` / `last_sensor_z` for a
        point-cloud consumer (e.g. TerrainInpaintMap)."""
        px, py, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        gz = float(self.scene.sample(px, py))  # local ground -> sensor height
        self.last_sensor_z = gz + self.mount_height
        origin = np.array([px, py, self.last_sensor_z])
        hits = self._lidar.scan(origin, yaw, self._boxes_lo, self._boxes_hi, seed=self._seed)
        self._seed += 1
        self.last_points = hits
        if len(hits) == 0:
            return np.zeros((self.ny, self.nx), np.float32), np.zeros((self.ny, self.nx), bool)
        obs, known = rasterize(hits, self.x0, self.y0, self.cell, self.ny, self.nx)
        return obs.astype(np.float32), known


class TerrainInpaintMap:
    """Dense (elev, known) map from recent scan points via terrain_toolkit's inpaint + the two
    `confidence/` masks. A drop-in for MultiScanMap (exposes `.elev`/`.known`, so crop_window and
    the viz read it unchanged).

    inpaint fills a wall's unseen INTERIOR (the cells between sparse face hits) so the wall reads
    solid -- closing the phantom-gap that the raw sparse map leaves for the optimistic planner.
    Then `known` is gated by BOTH confidence masks -- a cell is trusted only if it is not in a
    line-of-sight SHADOW (OcclusionMask) AND its neighborhood holds enough real returns
    (SupportRatioMask), so thinly-supported guesses and occluded fills stay unknown (-> the
    caller's optimism fill, not hallucinated terrain). `elev` is the dense inpainted heightmap.

    distrust_policy -- what happens to an UNtrusted cell (in `.elev`, since the caller reads
    `.known`/`.elev`):
      'flat'         : leave it to the caller's unknown handling (optimism -> flat). Safe in the
                       near field where walls are densely hit (well-supported, so kept solid);
                       untrusted cells are mostly far-field extrapolation.
      'height-split' : untrusted-but-TALL cells keep their inpainted height (a possible obstacle we
                       won't optimistically flatten) and are marked known; untrusted-low -> flat.

    The occlusion viewpoint is set per `update()`, so feed it the last-N scans (near the current
    pose) to keep the single-viewpoint model valid."""

    def __init__(
        self,
        scene: Heightmap,
        *,
        support_radius_m: float = 0.5,
        support_ratio: float = 0.35,  # keep a cell if >=35% of its neighborhood was measured
        distrust_policy: str = "flat",
        obstacle_height_m: float = 0.25,  # 'height-split' threshold: keep untrusted cells above this
        device: str = "cuda",
    ) -> None:
        from terrain_toolkit.confidence import OcclusionConfig
        from terrain_toolkit.confidence import SupportConfig
        from terrain_toolkit.confidence import SupportRatioMask
        from terrain_toolkit.pipeline import TerrainPipeline
        from terrain_toolkit.traversability import TraversabilityConfig

        ny, nx = scene.H.shape
        self.ny, self.nx = ny, nx
        self.distrust_policy = distrust_policy
        self.obstacle_height_m = float(obstacle_height_m)
        cell = scene.cell
        bounds = (scene.x0, scene.x0 + nx * cell, scene.y0, scene.y0 + ny * cell)
        self._pipe = TerrainPipeline(
            resolution=cell,
            bounds=bounds,
            inpaint=True,  # fill unseen cells (walls read solid) ...
            primary="max",
            traversability=TraversabilityConfig(),
            occlusion=OcclusionConfig(),  # ... but keep true line-of-sight shadows unknown
            layers=("elevation", "traversability", "count"),
            device=device,
        )
        # Applied directly (not via the pipeline's filter=... inflation path) -- distrusts inpaint
        # whose neighborhood lacks real returns.
        self._support = SupportRatioMask(
            cell,
            ny,
            nx,
            SupportConfig(support_radius_m=support_radius_m, support_ratio=support_ratio),
            device=wp.get_device(device),
        )
        self._zeros = np.zeros((ny, nx), np.float32)  # clean cost for the support-only mask
        self.elev = np.zeros((ny, nx), np.float32)
        self.known = np.zeros((ny, nx), bool)

    def update(self, points: np.ndarray, sensor_xy: tuple[float, float], sensor_z: float) -> None:
        """Rebuild the dense map from `points` (N,3), with occlusion cast from (sensor_xy, sensor_z)."""
        if len(points) == 0:
            return
        self._pipe.occlusion_mask.config.sensor_xy = (float(sensor_xy[0]), float(sensor_xy[1]))
        self._pipe.occlusion_mask.config.sensor_z = float(sensor_z)
        tm = self._pipe.process(np.ascontiguousarray(points, np.float32))
        elev = np.nan_to_num(tm.elevation).astype(np.float32)
        occ_ok = np.isfinite(tm.traversability)  # visible (not in a line-of-sight shadow)
        # raw (pre-inpaint) heightmap: measured cells keep their height, the rest NaN -- what the
        # support mask counts. Apply it on a clean zero cost so ONLY the support NaNs come through.
        raw = np.where(tm.count > 0, elev, np.nan).astype(np.float32)
        support_ok = np.isfinite(self._support.apply(raw, self._zeros).numpy())
        if self.distrust_policy == "height-split":
            # Occluded shadow ALWAYS stays unknown (a genuine no-see). Among VISIBLE cells, distrust
            # thin support -- UNLESS the cell is tall enough to be a possible obstacle we must not
            # optimistically flatten.
            keep = occ_ok & (support_ok | (elev > self.obstacle_height_m))
            self.elev = np.where(keep, elev, 0.0).astype(np.float32)
            self.known = keep
        else:  # 'flat': untrusted cells fall through to the caller's optimism fill
            self.known = occ_ok & support_ok
            self.elev = elev
