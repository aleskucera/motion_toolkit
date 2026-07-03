from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

from .confidence import OcclusionConfig
from .confidence import OcclusionMask
from .confidence import SupportConfig
from .confidence import SupportRatioMask
from .gridmap import GridMap
from .heightmap import FlatGroundFootprint
from .heightmap import FootprintConfig
from .heightmap import gaussian_smooth
from .heightmap import HeightMapBuilder
from .heightmap import multigrid_inpaint
from .outlier import OutlierFilterConfig
from .outlier import RadiusOutlierFilter
from .outlier import RadiusOutlierFilterConfig
from .outlier import StatisticalOutlierFilter
from .traversability import FilterConfig
from .traversability import GeometricTraversabilityAnalyzer
from .traversability import ObstacleInflator
from .traversability import TemporalGate
from .traversability import TraversabilityConfig

PrimaryLayer = Literal["max", "mean", "min"]

LayerName = Literal[
    "max",
    "mean",
    "min",
    "count",
    "elevation",
    "slope_cost",
    "step_cost",
    "roughness_cost",
    "traversability",
]

_ALL_LAYERS: tuple[str, ...] = (
    "max",
    "mean",
    "min",
    "count",
    "elevation",
    "slope_cost",
    "step_cost",
    "roughness_cost",
    "traversability",
)
_COST_LAYERS: frozenset[str] = frozenset(
    ("slope_cost", "step_cost", "roughness_cost", "traversability"),
)


@dataclass
class TerrainMap:
    """Output of `TerrainPipeline.process`.

    Fields are populated only when their layer was selected for download (see
    `TerrainPipeline(layers=...)`) *and* the corresponding stage was enabled.
    Everything else is `None`.
    """

    resolution: float
    bounds: tuple[float, float, float, float]

    # raw reductions
    max: np.ndarray | None = None
    mean: np.ndarray | None = None
    min: np.ndarray | None = None
    count: np.ndarray | None = None

    # primary reduction after inpaint + smooth
    elevation: np.ndarray | None = None

    # geometric cost layers (populated only when traversability is enabled)
    slope_cost: np.ndarray | None = None
    step_cost: np.ndarray | None = None
    roughness_cost: np.ndarray | None = None
    traversability: np.ndarray | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return all non-None layers as a flat name → array dict."""
        d: dict[str, np.ndarray] = {}
        for name in _ALL_LAYERS:
            arr = getattr(self, name)
            if arr is not None:
                d[name] = arr
        return d

    def as_gridmap(self) -> GridMap:
        """The elevation layer as the shared `GridMap` contract: origin = (xmin, ymin)
        from `bounds`, cell = `resolution`."""
        xmin, _, ymin, _ = self.bounds
        return GridMap(elevation=self.elevation, origin=(xmin, ymin), cell=self.resolution)


@dataclass
class TerrainMapGPU:
    """Device-resident output of `TerrainPipeline.process(return_device=True)`.

    Each field is a live `wp.array` handle into the pipeline's internal buffers —
    NOT a copy. The handles are BORROWED: they stay valid only until the next
    `process()` call on the same pipeline (buffers are reused frame-to-frame). A
    consumer that needs to retain a layer across frames must `wp.clone()` it.

    Lets a downstream Warp library consume the elevation grid on-device with no
    GPU→CPU→GPU round-trip. Grid layout matches `TerrainMap`: shape
    (height, width) = (ny, nx), row=y, col=x; cell-center world coords
    x = xmin + (j+0.5)*resolution, y = ymin + (i+0.5)*resolution.
    """

    resolution: float
    bounds: tuple[float, float, float, float]
    elevation: wp.array | None = None
    traversability: wp.array | None = None
    slope_cost: wp.array | None = None
    step_cost: wp.array | None = None
    roughness_cost: wp.array | None = None

    def as_gridmap(self) -> GridMap:
        """The (device-resident) elevation layer as the shared `GridMap` contract: origin =
        (xmin, ymin) from `bounds`, cell = `resolution`. The handle stays BORROWED -- valid
        only until the next `process()`; `wp.clone()` it to retain across frames."""
        xmin, _, ymin, _ = self.bounds
        return GridMap(elevation=self.elevation, origin=(xmin, ymin), cell=self.resolution)


class TerrainPipeline:
    """Points → (max, mean, min, count) → inpaint → smooth → cost → filter.

    Single entry point: `process(points)` returns a fully-populated
    `TerrainMap`. Stateful: reuses GPU buffers and filter hysteresis across
    calls, so the same instance should be reused frame-to-frame.

    Internally keeps data on the GPU: point cloud is uploaded once, every
    stage consumes and produces `wp.array`, and a single download happens
    at the end to build the numpy-backed `TerrainMap` for the caller.
    """

    def __init__(
        self,
        resolution: float,
        bounds: tuple[float, float, float, float],
        *,
        primary: PrimaryLayer = "max",
        inpaint: bool = True,
        smooth_sigma: float = 0.0,
        inpaint_iters_per_level: int = 50,
        inpaint_coarse_iters: int = 200,
        z_max: float | None = None,
        outlier: OutlierFilterConfig | RadiusOutlierFilterConfig | None = None,
        traversability: TraversabilityConfig | None = None,
        filter: FilterConfig | None = None,
        occlusion: OcclusionConfig | None = None,
        footprint: FootprintConfig | None = None,
        layers: tuple[str, ...] | None = None,
        device: str | wp.context.Device | None = None,
    ):
        if primary not in ("max", "mean", "min"):
            raise ValueError(f"primary must be 'max', 'mean', or 'min'; got {primary!r}")
        if traversability is not None and not inpaint:
            raise ValueError(
                "traversability requires inpaint=True — the cost kernels assume a filled grid"
            )
        if filter is not None and traversability is None:
            raise ValueError("filter is only meaningful when traversability is enabled")
        if occlusion is not None and traversability is None:
            raise ValueError("occlusion masks the cost map and requires traversability=...")

        # Resolve the Warp device once. Accepts "cpu", "cuda:0", a wp.Device, or
        # None (use Warp's current default). Outlier filtering uses wp.HashGrid
        # which is CUDA-only, so reject that combination up front.
        if device is None:
            self.device = wp.get_device()
        elif isinstance(device, str):
            self.device = wp.get_device(device)
        else:
            self.device = device
        if outlier is not None and not self.device.is_cuda:
            raise ValueError(
                "outlier filtering requires a CUDA device (wp.HashGrid is GPU-only); "
                f"got device={self.device}"
            )

        # Resolve which layers to download. `None` = everything the configured
        # stages produce. Skipping unused layers saves a D2H copy each — about
        # 0.05–0.1 ms per layer on an 80×150 grid.
        if layers is None:
            selected = set(_ALL_LAYERS)
        else:
            unknown = set(layers) - set(_ALL_LAYERS)
            if unknown:
                raise ValueError(f"unknown layer names {sorted(unknown)}; valid: {_ALL_LAYERS}")
            selected = set(layers)
        if traversability is None:
            selected -= _COST_LAYERS
        self._layers: set[str] = selected

        self.resolution = resolution
        self.bounds = bounds
        self.primary = primary
        self.z_max = z_max
        self.inpaint = inpaint
        self.smooth_sigma = smooth_sigma
        self.inpaint_iters_per_level = inpaint_iters_per_level
        self.inpaint_coarse_iters = inpaint_coarse_iters

        self.builder = HeightMapBuilder(
            resolution=resolution,
            bounds=bounds,
            device=self.device,
        )
        self.height = self.builder.height
        self.width = self.builder.width

        self.outlier_filter: StatisticalOutlierFilter | RadiusOutlierFilter | None = None
        if isinstance(outlier, RadiusOutlierFilterConfig):
            self.outlier_filter = RadiusOutlierFilter(config=outlier, device=self.device)
        elif isinstance(outlier, OutlierFilterConfig):
            self.outlier_filter = StatisticalOutlierFilter(config=outlier, device=self.device)

        self.footprint_stamp: FlatGroundFootprint | None = None
        if footprint is not None:
            self.footprint_stamp = FlatGroundFootprint(
                resolution=resolution,
                bounds=bounds,
                height=self.height,
                width=self.width,
                config=footprint,
                device=self.device,
            )

        self.analyzer: GeometricTraversabilityAnalyzer | None = None
        if traversability is not None:
            self.analyzer = GeometricTraversabilityAnalyzer(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=traversability,
                device=self.device,
            )

        self.inflator: ObstacleInflator | None = None
        self.temporal_gate: TemporalGate | None = None
        self.support_mask: SupportRatioMask | None = None
        if filter is not None:
            self.inflator = ObstacleInflator(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=filter,
                device=self.device,
            )
            self.temporal_gate = TemporalGate(config=filter, device=self.device)
            self.support_mask = SupportRatioMask(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=SupportConfig(
                    support_radius_m=filter.support_radius_m,
                    support_ratio=filter.support_ratio,
                ),
                device=self.device,
            )

        self.occlusion_mask: OcclusionMask | None = None
        if occlusion is not None:
            self.occlusion_mask = OcclusionMask(
                resolution=resolution,
                bounds=bounds,
                height=self.height,
                width=self.width,
                config=occlusion,
                device=self.device,
            )

    def process(
        self,
        points: np.ndarray | wp.array,
        *,
        footprint_plane: tuple[float, float, float] | None = None,
        return_device: bool = False,
    ) -> TerrainMap | TerrainMapGPU:
        """Run the pipeline on a point cloud.

        `points` may be a host (N, 3) numpy array or a device `wp.array(vec3)` for a
        zero-copy handoff from another Warp stage. Default: download the selected
        layers to a numpy-backed `TerrainMap`. `return_device=True`: skip the host
        copy and return a `TerrainMapGPU` of live device `wp.array` handles (valid
        only until the next `process()` — `wp.clone()` to retain).

        The `z_max` cutoff is applied here only for numpy input; a device cloud is
        taken as-is, so crop z on the caller side (e.g. `BoxCrop(z_max=...)`).
        """
        if self.z_max is not None and not isinstance(points, wp.array):
            points = points[points[:, 2] <= self.z_max]

        # Scope every allocation and kernel launch in this frame to the chosen
        # device — without this, helpers that call wp.array(...) or wp.zeros(...)
        # without an explicit device= argument would land on Warp's global
        # default, which can differ from self.device.
        with wp.ScopedDevice(self.device):
            dev = self._compute(points, footprint_plane)
            wp.synchronize()
            return self._wrap_device(dev) if return_device else self._download(dev)

    def _compute(
        self,
        points: np.ndarray | wp.array,
        footprint_plane: tuple[float, float, float] | None = None,
    ) -> dict[str, object]:
        """Run all GPU stages and return the device buffers (no download)."""
        # Device input is consumed directly (zero-copy); numpy uploads once here.
        if isinstance(points, wp.array):
            pts_wp = points
        else:
            pts_wp = wp.array(np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3)
        if self.outlier_filter is not None:
            pts_wp = self.outlier_filter.apply(pts_wp)

        layers = self.builder.build(pts_wp)
        primary_layer = layers[self.primary]  # wp.array

        # Force a flat ground patch under the robot, in-place on the primary
        # reduction. Done BEFORE the raw_primary snapshot so the footprint reads
        # as "measured" to the support mask, and before inpaint so the patch acts
        # as a known boundary.
        if self.footprint_stamp is not None:
            self.footprint_stamp.apply(primary_layer, footprint_plane)

        # multigrid_inpaint mutates its input buffer. The support mask needs the
        # ORIGINAL NaN-bearing primary to know which cells were really measured,
        # so snapshot it before inpaint runs.
        raw_primary = wp.clone(primary_layer) if self.inpaint else primary_layer

        elevation = primary_layer
        if self.inpaint:
            elevation = multigrid_inpaint(
                elevation,
                iters_per_level=self.inpaint_iters_per_level,
                coarse_iters=self.inpaint_coarse_iters,
            )
        if self.smooth_sigma > 0.0:
            elevation = gaussian_smooth(elevation, sigma=self.smooth_sigma)

        traversability_wp: wp.array | None = None
        costs_wp: dict[str, wp.array] | None = None
        if self.analyzer is not None:
            costs = self.analyzer.compute(elevation)
            total = costs.total
            if self.inflator is not None:
                # Inflate obstacles, then temporally gate, then mask by support ratio.
                # Inpainted cells with enough real neighbors keep their cost; the rest
                # become NaN, and frames that spike in obstacle count are rejected.
                inflated = self.inflator.apply(total)
                if self.temporal_gate.is_stable(inflated):
                    total = self.support_mask.apply(raw_primary, inflated)
                else:
                    total = self.support_mask.rejected_frame()
            if self.occlusion_mask is not None:
                # Drop inpainted cost in the line-of-sight shadow of obstacles, so
                # the planner can't route through occluded space behind walls.
                total = self.occlusion_mask.apply(raw_primary, elevation, total)
            traversability_wp = total
            costs_wp = {"slope": costs.slope, "step": costs.step, "roughness": costs.roughness}

        return {
            "layers": layers,
            "elevation": elevation,
            "traversability": traversability_wp,
            "costs": costs_wp,
        }

    def _download(self, dev: dict[str, object]) -> TerrainMap:
        """Copy the selected device layers into a numpy-backed `TerrainMap`."""
        sel = self._layers
        layers = dev["layers"]
        elevation = dev["elevation"]
        traversability_wp = dev["traversability"]
        costs_wp = dev["costs"]
        tm = TerrainMap(resolution=self.resolution, bounds=self.bounds)
        if "max" in sel:
            tm.max = layers.max.numpy().copy()
        if "mean" in sel:
            tm.mean = layers.mean.numpy().copy()
        if "min" in sel:
            tm.min = layers.min.numpy().copy()
        if "count" in sel:
            tm.count = layers.count.numpy().copy()
        if "elevation" in sel:
            tm.elevation = elevation.numpy().copy()
        if costs_wp is not None:
            if "slope_cost" in sel:
                tm.slope_cost = costs_wp["slope"].numpy().copy()
            if "step_cost" in sel:
                tm.step_cost = costs_wp["step"].numpy().copy()
            if "roughness_cost" in sel:
                tm.roughness_cost = costs_wp["roughness"].numpy().copy()
        if traversability_wp is not None and "traversability" in sel:
            tm.traversability = traversability_wp.numpy().copy()
        return tm

    def _wrap_device(self, dev: dict[str, object]) -> TerrainMapGPU:
        """Wrap the selected device layers as borrowed handles (no copy)."""
        sel = self._layers
        costs = dev["costs"]
        trav = dev["traversability"]
        return TerrainMapGPU(
            resolution=self.resolution,
            bounds=self.bounds,
            elevation=dev["elevation"] if "elevation" in sel else None,
            traversability=trav if (trav is not None and "traversability" in sel) else None,
            slope_cost=costs["slope"] if (costs is not None and "slope_cost" in sel) else None,
            step_cost=costs["step"] if (costs is not None and "step_cost" in sel) else None,
            roughness_cost=(
                costs["roughness"] if (costs is not None and "roughness_cost" in sel) else None
            ),
        )
