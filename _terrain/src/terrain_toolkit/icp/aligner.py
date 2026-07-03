from __future__ import annotations

import contextlib
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import warp as wp

from ..voxel import VoxelGrid
from .kernels import accumulate_system_kernel
from .kernels import estimate_normals_kernel
from .kernels import keep_going_kernel
from .kernels import se3_update_kernel
from .kernels import solve6x6_kernel
from .kernels import transform_points_kernel


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

    # Upper bound on input points per cloud. The aligner preallocates its device
    # buffers (and the voxel hash table) to this size; the fixed buffers are what
    # make the GN loop CUDA-graph-capturable.
    max_points: int = 120_000

    # Voxel downsampling applied to source (and target if `voxel_target`) before
    # ICP. Set to None or 0 to disable. One centroid per occupied voxel gives a
    # more uniform spatial distribution than random subsampling.
    voxel_size_m: float | None = None
    voxel_target: bool = False


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


@dataclass
class IcpResult:
    pose: np.ndarray  # (4, 4) float64 — target_T_source
    iterations: int
    final_cost: float
    num_inliers: int
    converged: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


# Hash-grid buckets per axis. Only affects query performance (the neighbor
# search checks true distances, so it's exact for any dims); a fixed generous
# grid avoids scanning the target extent (~4 ms of min/max) every align.
_HASHGRID_DIM = 128


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

        cfg = self.config
        self._voxel: VoxelGrid | None = None
        if cfg.voxel_size_m is not None and cfg.voxel_size_m > 0.0:
            self._voxel = VoxelGrid(
                cfg.voxel_size_m, max_points=cfg.max_points, device=self.device
            )

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

    def _ensure_grid(self) -> wp.HashGrid:
        if self._grid is None or self._grid.device != self.device:
            self._grid = wp.HashGrid(
                _HASHGRID_DIM, _HASHGRID_DIM, _HASHGRID_DIM, device=self.device
            )
        return self._grid

    def _prepare_cloud(self, points: wp.array, out: wp.array, apply_voxel: bool) -> int:
        """Fill `out` (device) with the optionally-downsampled cloud; return count.

        Fully device-resident: `points` is a `wp.array` and any downsampled result
        is copied device→device into `out`. A cloud larger than `max_points` uses
        its first `max_points` (rare — downsampling normally keeps it well under).
        """
        mp = self.config.max_points
        n_in = min(len(points), mp)
        if len(points) > mp and not self._capped_warned:
            print(f"[icp] cloud of {len(points)} pts exceeds max_points={mp}; using first {n_in}")
            self._capped_warned = True
        if apply_voxel:
            downsampled, n_out = self._voxel.downsample(points, n_in)
            wp.copy(out, downsampled, 0, 0, n_out)
            return n_out
        wp.copy(out, points, 0, 0, n_in)
        return n_in

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
        source: wp.array,
        target: wp.array,
        init_pose: np.ndarray | None = None,
        *,
        profile: bool = False,
    ) -> IcpResult:
        """Align `source` to `target`, both device `wp.array(vec3)`; return the pose.

        Fully device-resident — the clouds stay on the GPU end to end (upload them
        once at the sensor boundary). `init_pose` is a host 4x4 (the odom guess);
        only it and the final pose cross the host boundary.
        """
        if not isinstance(source, wp.array) or not isinstance(target, wp.array):
            raise TypeError("source and target must be wp.array(vec3) on the device")

        cfg = self.config
        grid_radius = max(cfg.max_correspondence_dist_m, cfg.normal_radius_m)
        prof = _EventTimer(self.device, profile)

        voxel_on = self._voxel is not None

        with wp.ScopedDevice(self.device):
            # Downsample (if enabled) and load into the fixed GN buffers, all on
            # device; only the point counts return to the host. The rest of each
            # buffer is never read (kernels guard on the live count).
            with prof.stage("voxel_downsample"):
                n_src = self._prepare_cloud(source, self._src, voxel_on)
                n_tgt = self._prepare_cloud(target, self._tgt, voxel_on and cfg.voxel_target)
                self._n_src.fill_(n_src)

            with prof.stage("grid_build"):
                grid = self._ensure_grid()
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
