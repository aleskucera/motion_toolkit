from __future__ import annotations

import contextlib
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import warp as wp

from .kernels import accumulate_system_kernel
from .kernels import estimate_normals_kernel
from .kernels import keep_going_kernel
from .kernels import se3_update_kernel
from .kernels import solve6x6_kernel
from .kernels import transform_points_kernel
from .kernels import voxel_accumulate_kernel
from .kernels import voxel_compact_kernel


@dataclass
class IcpConfig:
    """Configuration for `IcpAligner`."""

    max_iters: int = 30
    max_correspondence_dist_m: float = 0.5
    huber_delta: float = 0.1
    normal_radius_m: float = 0.2
    normal_min_neighbors: int = 5
    normal_power_iters: int = 12
    convergence_rotation_rad: float = 1.0e-4
    convergence_translation_m: float = 1.0e-4
    damping: float = 1.0e-6

    # Upper bound on source/target points after downsampling; the aligner
    # preallocates its device buffers to this size and subsamples anything larger
    # (fixed buffers are what make the GN loop CUDA-graph-capturable).
    max_points: int = 120_000

    # Voxel downsampling applied to source (and target if `voxel_target`) before ICP.
    # Set to None or 0 to disable. Using the centroid per voxel gives a more
    # uniform spatial distribution than random subsampling.
    voxel_size_m: float | None = None
    voxel_target: bool = False
    # Optional fixed world bounds for the voxel grid: (xmin, xmax, ymin, ymax, zmin, zmax).
    # When set, skips the per-call CPU min/max scan. Points outside are dropped.
    voxel_bounds_m: tuple[float, float, float, float, float, float] | None = None


_MAX_VOXEL_GRID_CELLS = 20_000_000


class _EventTimer:
    """Per-stage GPU timing via CUDA events; a no-op unless enabled and on CUDA.

    Each `stage()` records a start/stop event pair on the stream — recording is
    async and adds no sync, so profiling does not serialize the pipeline the way
    a per-stage `wp.synchronize()` would. `read()` folds the elapsed intervals
    into per-stage milliseconds at the end (that read syncs on the stop events,
    so treat the per-stage means as the cost, not the profiled wall-clock).
    """

    def __init__(self, device: wp.context.Device, enabled: bool):
        self.device = device
        self.enabled = bool(enabled) and device.is_cuda
        self.ms: dict[str, float] = {}
        self._intervals: list[tuple[str, wp.Event, wp.Event]] = []

    @contextlib.contextmanager
    def stage(self, key: str):
        if not self.enabled:
            yield
            return
        start = wp.Event(device=self.device, enable_timing=True)
        stop = wp.Event(device=self.device, enable_timing=True)
        wp.record_event(start)
        try:
            yield
        finally:
            wp.record_event(stop)
            self._intervals.append((key, start, stop))

    def read(self) -> None:
        """Fold every recorded interval into `ms` (sums repeats, e.g. per-iter)."""
        for key, start, stop in self._intervals:
            self.ms[key] = self.ms.get(key, 0.0) + wp.get_event_elapsed_time(start, stop)
        self._intervals.clear()


def _voxel_grid_dims(mins: np.ndarray, maxs: np.ndarray, voxel_size: float) -> tuple[np.ndarray, int]:
    """Per-axis cell counts and total cells for a voxel grid over [mins, maxs]."""
    dims = np.ceil((maxs - mins) / voxel_size).astype(np.int64) + 1
    n_vx = int(dims.prod())
    if n_vx > _MAX_VOXEL_GRID_CELLS:
        raise ValueError(
            f"voxel grid has {n_vx} cells (>{_MAX_VOXEL_GRID_CELLS}); use a larger voxel_size"
        )
    return dims, n_vx


def _voxel_bin(
    points_wp: wp.array,
    n: int,
    min_corner: wp.vec3,
    inv_voxel: float,
    dims: np.ndarray,
    sums: wp.array,
    counts: wp.array,
    occupied: wp.array,
    occ_counter: wp.array,
    out: wp.array,
) -> int:
    """Bin `points_wp` to one centroid per occupied voxel in caller-owned buffers.

    `sums`/`counts` (len ≥ grid cells) must start zeroed and are left zeroed —
    the compact pass resets each cell it reads, so it (and the clear) cost
    O(occupied), not O(grid). `occupied`/`out` (len ≥ n) are scratch. Returns the
    number of centroids written to `out`.
    """
    occ_counter.zero_()
    wp.launch(
        voxel_accumulate_kernel,
        dim=n,
        inputs=[points_wp, min_corner, inv_voxel, int(dims[0]), int(dims[1]), int(dims[2])],
        outputs=[sums, counts, occupied, occ_counter],
    )
    wp.synchronize()
    n_out = int(occ_counter.numpy()[0])
    if n_out:
        wp.launch(
            voxel_compact_kernel,
            dim=n_out,
            inputs=[occupied],
            outputs=[sums, counts, out],
        )
    return n_out


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    *,
    device: wp.context.Device | None = None,
) -> np.ndarray:
    """GPU voxel downsample: return one centroid per occupied voxel."""
    if voxel_size <= 0.0 or len(points) == 0:
        return points

    pts = np.ascontiguousarray(points, dtype=np.float32)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    dims, n_vx = _voxel_grid_dims(mins, maxs, voxel_size)

    device = wp.get_device(device)
    with wp.ScopedDevice(device):
        pts_wp = wp.array(pts, dtype=wp.vec3)
        sums = wp.zeros(n_vx, dtype=wp.vec3)
        counts = wp.zeros(n_vx, dtype=wp.int32)
        occupied = wp.empty(len(pts), dtype=wp.int32)
        occ_counter = wp.zeros(1, dtype=wp.int32)
        out = wp.empty(len(pts), dtype=wp.vec3)
        min_corner = wp.vec3(float(mins[0]), float(mins[1]), float(mins[2]))
        n_out = _voxel_bin(
            pts_wp,
            len(pts),
            min_corner,
            1.0 / voxel_size,
            dims,
            sums,
            counts,
            occupied,
            occ_counter,
            out,
        )
        result = out.numpy()[:n_out]

    return result.astype(points.dtype, copy=False)


@dataclass
class IcpResult:
    pose: np.ndarray  # (4, 4) float64 — target_T_source
    iterations: int
    final_cost: float
    num_inliers: int
    converged: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


def _hashgrid_dims(points: np.ndarray, radius: float) -> tuple[int, int, int]:
    """Pick reasonable hash grid dimensions for the target cloud."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = np.maximum(maxs - mins, radius)
    cells = np.ceil(extent / max(radius, 1.0e-6)).astype(int)
    # Clamp to reasonable values.
    cells = np.clip(cells, 8, 256)
    return int(cells[0]), int(cells[1]), int(cells[2])


class IcpAligner:
    """Point-to-plane ICP on the GPU via Warp.

    Frame-to-frame usage: pass source and target point clouds plus an initial
    pose guess (e.g. from odometry); returns the refined `target_T_source`
    transform. Target normals are re-estimated every call.
    """

    def __init__(
        self,
        config: IcpConfig | None = None,
        *,
        device: wp.context.Device | None = None,
        verbose: bool = False,
    ):
        self.config = config or IcpConfig()
        self.device = wp.get_device(device)
        self.verbose = verbose
        self._grid: wp.HashGrid | None = None
        self._graph = None  # captured GN-loop graph (built lazily on the first align)
        self._capped_warned = False

        # Voxel-downsample scratch buffers, grown on demand. sums/counts are kept
        # zeroed between calls by the compact pass, so they are never re-zeroed.
        self._vx_cells: int = 0
        self._vx_out_capacity: int = 0
        self._vx_sums: wp.array | None = None
        self._vx_counts: wp.array | None = None
        self._vx_counter: wp.array | None = None
        self._vx_occupied: wp.array | None = None
        self._vx_out: wp.array | None = None

        # Device-resident GN-loop state: fixed-size buffers (max_points) plus the
        # per-iteration scalars, so the loop touches only these and is graph-ready.
        mp = self.config.max_points
        with wp.ScopedDevice(self.device):
            self._src = wp.zeros(mp, dtype=wp.vec3)
            self._tgt = wp.zeros(mp, dtype=wp.vec3)
            self._transformed = wp.zeros(mp, dtype=wp.vec3)
            self._normals = wp.zeros(mp, dtype=wp.vec3)
            self._valid = wp.zeros(mp, dtype=wp.int32)
            self._n_src = wp.zeros(1, dtype=wp.int32)
            self._JtJ = wp.zeros((6, 6), dtype=wp.float32)
            self._Jtr = wp.zeros(6, dtype=wp.float32)
            self._cost = wp.zeros(1, dtype=wp.float32)
            self._inliers = wp.zeros(1, dtype=wp.int32)
            self._pose = wp.zeros(1, dtype=wp.mat44)
            self._delta = wp.zeros(6, dtype=wp.float32)
            self._dr = wp.zeros(1, dtype=wp.float32)
            self._dt = wp.zeros(1, dtype=wp.float32)
            self._iter = wp.zeros(1, dtype=wp.int32)
            self._keep_running = wp.zeros(1, dtype=wp.int32)
            self._converged = wp.zeros(1, dtype=wp.int32)

    def _ensure_grid(self, radius: float, points: np.ndarray) -> wp.HashGrid:
        dims = _hashgrid_dims(points, radius)
        if self._grid is None or self._grid.device != self.device:
            self._grid = wp.HashGrid(*dims, device=self.device)
        return self._grid

    def _voxel_bin_into_scratch(self, points: np.ndarray, voxel_size: float) -> int:
        """Voxel-downsample into `self._vx_out` (device); return the point count.

        Unlike the module-level `voxel_downsample`, the result is left on device
        (no cloud readback) so `align` can copy it straight into its GN buffers.
        """
        pts = np.ascontiguousarray(points, dtype=np.float32)
        bounds = self.config.voxel_bounds_m
        if bounds is not None:
            mins = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
            maxs = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
        else:
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
        dims, n_vx = _voxel_grid_dims(mins, maxs, voxel_size)

        # Grow the shared scratch buffers on demand; sums/counts stay zeroed
        # between calls (the compact pass clears each cell it reads).
        if self._vx_cells < n_vx:
            self._vx_sums = wp.zeros(n_vx, dtype=wp.vec3)
            self._vx_counts = wp.zeros(n_vx, dtype=wp.int32)
            self._vx_cells = n_vx
        if self._vx_out_capacity < len(pts):
            self._vx_out = wp.empty(len(pts), dtype=wp.vec3)
            self._vx_occupied = wp.empty(len(pts), dtype=wp.int32)
            self._vx_out_capacity = len(pts)
        if self._vx_counter is None:
            self._vx_counter = wp.zeros(1, dtype=wp.int32)

        min_corner = wp.vec3(float(mins[0]), float(mins[1]), float(mins[2]))
        return _voxel_bin(
            wp.array(pts, dtype=wp.vec3),
            len(pts),
            min_corner,
            1.0 / voxel_size,
            dims,
            self._vx_sums,
            self._vx_counts,
            self._vx_occupied,
            self._vx_counter,
            self._vx_out,
        )

    def _load_cloud(self, points: np.ndarray, out: wp.array, apply_voxel: bool) -> int:
        """Fill `out` (device) with the optionally-downsampled, capped cloud; return count.

        When voxelizing, the downsampled cloud stays on device — a device→device
        copy into `out`, avoiding a downsample→host→re-upload round trip. Only the
        (tiny) point count crosses back to the host.
        """
        if apply_voxel and len(points) > 0:
            n = self._voxel_bin_into_scratch(points, self.config.voxel_size_m)
            if n <= self.config.max_points:
                wp.copy(out, self._vx_out, 0, 0, n)
                return n
            points = self._vx_out.numpy()[:n]  # rare: still over the cap → host cap below
        capped = self._cap(np.ascontiguousarray(points, dtype=np.float32))
        n = len(capped)
        if n:
            wp.copy(out, wp.array(capped, dtype=wp.vec3), 0, 0, n)
        return n

    def _cap(self, points: np.ndarray) -> np.ndarray:
        """Subsample to at most `max_points` (the fixed buffers require a bound)."""
        n = len(points)
        if n <= self.config.max_points:
            return points
        if not self._capped_warned:
            print(f"[icp] cloud of {n} pts exceeds max_points; subsampling to {self.config.max_points}")
            self._capped_warned = True
        idx = np.linspace(0, n - 1, self.config.max_points).astype(np.int64)
        return points[idx]

    def _seed_pose(self, init_pose: np.ndarray | None) -> None:
        """Load the initial pose and reset the loop state into the device buffers."""
        pose = np.eye(4, dtype=np.float32) if init_pose is None else np.asarray(init_pose, np.float32)
        self._pose.assign(pose.reshape(1, 4, 4))
        self._iter.zero_()
        self._converged.zero_()
        self._keep_running.fill_(1)

    def _gn_body(self, grid: wp.HashGrid) -> None:
        """One Gauss-Newton iteration, entirely on device (this is the graph body).

        Launches over the fixed `max_points` extent; the transform/accumulate
        kernels early-return past the live count `_n_src`.
        """
        cfg = self.config
        mp = cfg.max_points
        self._JtJ.zero_()
        self._Jtr.zero_()
        self._cost.zero_()
        self._inliers.zero_()
        wp.launch(
            transform_points_kernel,
            dim=mp,
            inputs=[self._src, self._pose, self._n_src],
            outputs=[self._transformed],
        )
        wp.launch(
            accumulate_system_kernel,
            dim=mp,
            inputs=[
                grid.id,
                self._tgt,
                self._normals,
                self._valid,
                self._transformed,
                self._n_src,
                float(cfg.max_correspondence_dist_m),
                float(cfg.huber_delta),
            ],
            outputs=[self._JtJ, self._Jtr, self._cost, self._inliers],
        )
        wp.launch(
            solve6x6_kernel,
            dim=1,
            inputs=[self._JtJ, self._Jtr, float(cfg.damping)],
            outputs=[self._delta],
        )
        wp.launch(
            se3_update_kernel,
            dim=1,
            inputs=[self._delta],
            outputs=[self._pose, self._dr, self._dt],
        )
        wp.launch(
            keep_going_kernel,
            dim=1,
            inputs=[
                self._dr,
                self._dt,
                self._inliers,
                self._iter,
                int(cfg.max_iters),
                float(cfg.convergence_rotation_rad),
                float(cfg.convergence_translation_m),
            ],
            outputs=[self._keep_running, self._converged],
        )

    def _run_gn(self, grid: wp.HashGrid) -> None:
        """Run the GN loop: a captured CUDA-graph replay on CUDA, eager otherwise.

        The body touches only preallocated buffers with the grid rebuilt in place,
        so one capture is replayed across every align — the convergence loop runs
        on device via `capture_while`, with no per-iteration host sync. `capture=`
        on CPU (no graph support) falls back to a host-driven loop.
        """
        if self.device.is_cuda:
            if self._graph is None:
                with wp.ScopedCapture(device=self.device) as cap:
                    wp.capture_while(self._keep_running, lambda: self._gn_body(grid))
                self._graph = cap.graph
            wp.capture_launch(self._graph)
        else:
            for _ in range(self.config.max_iters):
                self._gn_body(grid)
                wp.synchronize()
                if int(self._keep_running.numpy()[0]) == 0:
                    break

    def align(
        self,
        source: np.ndarray,
        target: np.ndarray,
        init_pose: np.ndarray | None = None,
        *,
        profile: bool = False,
    ) -> IcpResult:
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError(f"source must be (N, 3); got {source.shape}")
        if target.ndim != 2 or target.shape[1] != 3:
            raise ValueError(f"target must be (N, 3); got {target.shape}")

        cfg = self.config
        grid_radius = max(cfg.max_correspondence_dist_m, cfg.normal_radius_m)
        prof = _EventTimer(self.device, profile)

        voxel_on = cfg.voxel_size_m is not None and cfg.voxel_size_m > 0.0

        with wp.ScopedDevice(self.device):
            # Downsample (if enabled), cap, and load into the fixed GN buffers —
            # device-resident, so only the point counts return to the host. The
            # rest of each buffer is never read (kernels guard on the live count).
            with prof.stage("voxel_downsample"):
                n_src = self._load_cloud(source, self._src, voxel_on)
                n_tgt = self._load_cloud(target, self._tgt, voxel_on and cfg.voxel_target)
                self._n_src.fill_(n_src)

            with prof.stage("grid_build"):
                grid = self._ensure_grid(grid_radius, target)
                grid.build(points=self._tgt[:n_tgt], radius=float(grid_radius))

            with prof.stage("normals"):
                wp.launch(
                    estimate_normals_kernel,
                    dim=n_tgt,
                    inputs=[
                        grid.id,
                        self._tgt,
                        float(cfg.normal_radius_m),
                        int(cfg.normal_min_neighbors),
                        int(cfg.normal_power_iters),
                    ],
                    outputs=[self._normals, self._valid],
                )

            self._seed_pose(init_pose)
            with prof.stage("iterations"):
                self._run_gn(grid)

            prof.read()
            wp.synchronize()
            pose = self._pose.numpy().reshape(4, 4).astype(np.float64)
            iters_run = int(self._iter.numpy()[0])
            final_inliers = int(self._inliers.numpy()[0])
            final_cost = float(self._cost.numpy()[0])
            converged = bool(self._converged.numpy()[0])

        return IcpResult(
            pose=pose,
            iterations=iters_run,
            final_cost=final_cost,
            num_inliers=final_inliers,
            converged=converged,
            timings_ms=prof.ms if profile else {},
        )
