from __future__ import annotations

import contextlib
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import warp as wp

from ..voxel import VoxelGrid
from .kernels import all_converged_batch_kernel
from .kernels import accumulate_gravity_prior_batch_kernel
from .kernels import accumulate_gravity_prior_kernel
from .kernels import accumulate_system_batch_kernel
from .kernels import accumulate_system_kernel
from .kernels import estimate_normals_kernel
from .kernels import se3_update_batch_kernel
from .kernels import solve6x6_batch_kernel
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
    # Trimmed loss: reject correspondences whose point-to-plane residual exceeds this (a hard
    # cut on gross outliers, more decisive than Huber's soft down-weight). 0 disables.
    trim_residual_m: float = 0.0
    normal_radius_m: float = 0.2
    normal_min_neighbors: int = 5
    normal_power_iters: int = 12
    convergence_rotation_rad: float = 1.0e-4
    convergence_translation_m: float = 1.0e-4
    damping: float = 1.0e-6

    # Soft gravity prior: when > 0 and an up-vector is passed to align(), anchor the
    # pose's roll/pitch to gravity so geometry-only tilt cannot drift off level. The
    # weight trades off against the per-point geometry residual (roughly "equivalent
    # point-to-plane rows"); 0 disables it entirely.
    gravity_weight: float = 0.0

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
        self._prepared_grid: wp.HashGrid | None = None  # set by prepare(), used by align_from()
        self._n_src_host = 0  # live source count after the last _prepare_target (batch launch dim)
        self._batch_h = 0  # H the batched buffers below are sized for (0 = not allocated)
        self._capped_warned = False

        cfg = self.config
        self._voxel: VoxelGrid | None = None
        if cfg.voxel_size_m is not None and cfg.voxel_size_m > 0.0:
            self._voxel = VoxelGrid(cfg.voxel_size_m, max_points=cfg.max_points, device=self.device)

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
            # Gravity-prior inputs, read inside the (graph-captured) GN body. Set per
            # align(); grav_w = 0 makes the prior kernel a no-op, so it can stay in the
            # captured graph and be toggled by value without a re-capture.
            self._grav_up = wp.zeros(1, dtype=wp.vec3)
            self._grav_w = wp.zeros(1, dtype=wp.float32)
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
        pose = (
            np.eye(4, dtype=np.float32) if init_pose is None else np.asarray(init_pose, np.float32)
        )
        self._pose.assign(pose.reshape(1, 4, 4))
        self._iter.zero_()
        self._converged.zero_()
        self._keep_running.fill_(1)

    def _set_gravity_prior(self, gravity_up: np.ndarray | None) -> None:
        """Load the per-align gravity up-vector + weight into the device buffers the
        (graph-captured) prior kernel reads. weight 0 => the kernel is a no-op."""
        if gravity_up is None or self.config.gravity_weight <= 0.0:
            self._grav_w.zero_()
            return
        u = np.asarray(gravity_up, np.float32).reshape(3)
        norm = float(np.linalg.norm(u))
        if norm < 1.0e-9:
            self._grav_w.zero_()
            return
        self._grav_up.assign((u / norm).reshape(1, 3))
        self._grav_w.fill_(float(self.config.gravity_weight))

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
                float(cfg.trim_residual_m),
            ],
            outputs=[self._JtJ, self._Jtr, self._cost, self._inliers],
        )
        # Gravity soft-prior on roll/pitch (no-op when _grav_w == 0). Added after the
        # geometry rows, before the solve, so both fold into the same 6x6 system.
        wp.launch(
            accumulate_gravity_prior_kernel,
            dim=1,
            inputs=[self._pose, self._grav_up, self._grav_w],
            outputs=[self._JtJ, self._Jtr],
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
        gravity_up: np.ndarray | None = None,
        profile: bool = False,
    ) -> IcpResult:
        """Align `source` to `target`, both device `wp.array(vec3)`; return the pose.

        Fully device-resident — the clouds stay on the GPU end to end (upload them
        once at the sensor boundary). `init_pose` is a host 4x4 (the odom guess);
        only it and the final pose cross the host boundary.

        `gravity_up` (3,) is the measured up-direction in the SOURCE frame (e.g. the
        IMU gravity vector, in base). With `config.gravity_weight > 0` it anchors the
        pose's roll/pitch to gravity each iteration; None (or weight 0) disables it.
        """
        if not isinstance(source, wp.array) or not isinstance(target, wp.array):
            raise TypeError("source and target must be wp.array(vec3) on the device")
        prof = _EventTimer(self.device, profile)
        with wp.ScopedDevice(self.device):
            grid = self._prepare_target(source, target, prof)
            return self._solve_from(grid, init_pose, gravity_up, prof)

    def prepare(self, source: wp.array, target: wp.array) -> None:
        """Prepare `target` (downsample + grid + normals) and load `source`, once, for
        repeated `align_from()` calls with different init poses on the SAME clouds.

        This is the init-independent, expensive part; `align_from()` is then just the GN
        loop. Multi-start (e.g. a yaw sweep to escape rotational local minima) uses this +
        `align_from()` so the target's grid and normals are built once, not per hypothesis.
        """
        if not isinstance(source, wp.array) or not isinstance(target, wp.array):
            raise TypeError("source and target must be wp.array(vec3) on the device")
        with wp.ScopedDevice(self.device):
            self._prepare_target(source, target, _EventTimer(self.device, False))

    def align_from(
        self, init_pose: np.ndarray, *, gravity_up: np.ndarray | None = None
    ) -> IcpResult:
        """Align from `init_pose` on the target set by the last `prepare()` — only the GN
        loop, so it is cheap to call many times over. Call `prepare()` first."""
        if self._prepared_grid is None:
            raise RuntimeError("align_from() called before prepare()")
        with wp.ScopedDevice(self.device):
            return self._solve_from(
                self._prepared_grid, init_pose, gravity_up, _EventTimer(self.device, False)
            )

    def _prepare_target(
        self, source: wp.array, target: wp.array, prof: _EventTimer
    ) -> wp.HashGrid:
        """Downsample source+target into the GN buffers, build the target hash grid, and
        estimate target normals. Everything here depends only on the clouds, not the init."""
        cfg = self.config
        grid_radius = max(cfg.max_correspondence_dist_m, cfg.normal_radius_m)
        voxel_on = self._voxel is not None
        # Downsample (if enabled) into the fixed GN buffers, all on device; only the point
        # counts return to the host. The rest of each buffer is never read (kernels guard
        # on the live count).
        with prof.stage("voxel_downsample"):
            n_src = self._prepare_cloud(source, self._src, voxel_on)
            n_tgt = self._prepare_cloud(target, self._tgt, voxel_on and cfg.voxel_target)
            self._n_src.fill_(n_src)
            self._n_src_host = n_src
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
        self._prepared_grid = grid
        return grid

    def _solve_from(
        self,
        grid: wp.HashGrid,
        init_pose: np.ndarray | None,
        gravity_up: np.ndarray | None,
        prof: _EventTimer,
    ) -> IcpResult:
        """Run the GN loop from `init_pose` on the already-prepared target; read the result."""
        self._seed_pose(init_pose)
        self._set_gravity_prior(gravity_up)
        with prof.stage("iterations"):
            self._run_gn(grid)
        prof.read()
        wp.synchronize()
        return IcpResult(
            pose=self._pose.numpy().reshape(4, 4).astype(np.float64),
            iterations=int(self._iter.numpy()[0]),
            final_cost=float(self._cost.numpy()[0]),
            num_inliers=int(self._inliers.numpy()[0]),
            converged=bool(self._converged.numpy()[0]),
            timings_ms=prof.ms if prof.enabled else {},
        )

    def _ensure_batch_buffers(self, h: int) -> None:
        """Allocate the per-hypothesis GN accumulators for `h` hypotheses (reused if unchanged)."""
        if self._batch_h == h:
            return
        with wp.ScopedDevice(self.device):
            self._pose_h = wp.zeros(h, dtype=wp.mat44)
            self._JtJ_h = wp.zeros((h, 6, 6), dtype=wp.float32)
            self._Jtr_h = wp.zeros((h, 6), dtype=wp.float32)
            self._cost_h = wp.zeros(h, dtype=wp.float32)
            self._inliers_h = wp.zeros(h, dtype=wp.int32)
            self._delta_h = wp.zeros((h, 6), dtype=wp.float32)
            self._dr_h = wp.zeros(h, dtype=wp.float32)
            self._dt_h = wp.zeros(h, dtype=wp.float32)
            self._alldone = wp.zeros(1, dtype=wp.int32)
        self._batch_h = h

    def align_batch(
        self,
        source: wp.array,
        target: wp.array,
        inits: np.ndarray,
        *,
        gravity_up: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align `inits` (H host 4x4 guesses) to the same target in ONE batched GN loop.

        All H hypotheses share the target grid + normals and advance together — each
        iteration's correspondence search for all H runs in a single 2D launch, so the cost
        is ~one ICP rather than H. Runs a fixed `max_iters` (no per-hypothesis early stop).
        Returns (poses (H, 4, 4), final_costs (H,), num_inliers (H,)); the caller picks the
        best (e.g. lowest RMS = sqrt(cost / inliers) clearing an inlier gate).
        """
        if not isinstance(source, wp.array) or not isinstance(target, wp.array):
            raise TypeError("source and target must be wp.array(vec3) on the device")
        inits = np.ascontiguousarray(inits, np.float32).reshape(-1, 4, 4)
        n_hyp = int(inits.shape[0])
        cfg = self.config
        with wp.ScopedDevice(self.device):
            grid = self._prepare_target(source, target, _EventTimer(self.device, False))
            self._ensure_batch_buffers(n_hyp)
            self._pose_h.assign(inits)
            self._set_gravity_prior(gravity_up)
            n = max(int(self._n_src_host), 1)
            for _ in range(cfg.max_iters):
                self._JtJ_h.zero_()
                self._Jtr_h.zero_()
                self._cost_h.zero_()
                self._inliers_h.zero_()
                wp.launch(
                    accumulate_system_batch_kernel,
                    dim=(n_hyp, n),
                    inputs=[
                        grid.id,
                        self._tgt,
                        self._normals,
                        self._valid,
                        self._src,
                        self._pose_h,
                        self._n_src,
                        float(cfg.max_correspondence_dist_m),
                        float(cfg.huber_delta),
                        float(cfg.trim_residual_m),
                    ],
                    outputs=[self._JtJ_h, self._Jtr_h, self._cost_h, self._inliers_h],
                )
                wp.launch(
                    accumulate_gravity_prior_batch_kernel,
                    dim=n_hyp,
                    inputs=[self._pose_h, self._grav_up, self._grav_w],
                    outputs=[self._JtJ_h, self._Jtr_h],
                )
                wp.launch(
                    solve6x6_batch_kernel,
                    dim=n_hyp,
                    inputs=[self._JtJ_h, self._Jtr_h, float(cfg.damping)],
                    outputs=[self._delta_h],
                )
                wp.launch(
                    se3_update_batch_kernel,
                    dim=n_hyp,
                    inputs=[self._delta_h],
                    outputs=[self._pose_h, self._dr_h, self._dt_h],
                )
                # Early stop once every hypothesis has settled (a loose tolerance — the tight
                # single-ICP one rarely trips on real data). One tiny reducer + int readback.
                wp.launch(
                    all_converged_batch_kernel,
                    dim=1,
                    inputs=[self._dr_h, self._dt_h, n_hyp, 1.0e-3, 1.0e-3],
                    outputs=[self._alldone],
                )
                if int(self._alldone.numpy()[0]) == 1:
                    break
            wp.synchronize()
            poses = self._pose_h.numpy().reshape(n_hyp, 4, 4).astype(np.float64)
            costs = self._cost_h.numpy().copy()
            inliers = self._inliers_h.numpy().copy()
        return poses, costs, inliers
