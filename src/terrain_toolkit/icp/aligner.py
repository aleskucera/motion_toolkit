from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import warp as wp

from .kernels import accumulate_system_kernel
from .kernels import estimate_normals_kernel
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


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)


def _exp_se3(xi: np.ndarray) -> np.ndarray:
    """Lie-algebra exp: xi = [omega; v] (6,) → SE(3) 4x4."""
    omega = xi[:3]
    v = xi[3:]
    theta = float(np.linalg.norm(omega))
    W = _skew(omega)
    if theta < 1.0e-8:
        R = np.eye(3) + W
        V = np.eye(3) + 0.5 * W
    else:
        W2 = W @ W
        R = (
            np.eye(3)
            + (math.sin(theta) / theta) * W
            + ((1.0 - math.cos(theta)) / (theta * theta)) * W2
        )
        V = (
            np.eye(3)
            + ((1.0 - math.cos(theta)) / (theta * theta)) * W
            + ((theta - math.sin(theta)) / (theta**3)) * W2
        )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


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

        # Voxel-downsample scratch buffers, grown on demand. sums/counts are kept
        # zeroed between calls by the compact pass, so they are never re-zeroed.
        self._vx_cells: int = 0
        self._vx_out_capacity: int = 0
        self._vx_sums: wp.array | None = None
        self._vx_counts: wp.array | None = None
        self._vx_counter: wp.array | None = None
        self._vx_occupied: wp.array | None = None
        self._vx_out: wp.array | None = None

    def _ensure_grid(self, radius: float, points: np.ndarray) -> wp.HashGrid:
        dims = _hashgrid_dims(points, radius)
        if self._grid is None or self._grid.device != self.device:
            self._grid = wp.HashGrid(*dims, device=self.device)
        return self._grid

    def _voxel_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Voxel downsample using reusable pre-allocated buffers."""
        if voxel_size <= 0.0 or len(points) == 0:
            return points

        pts = np.ascontiguousarray(points, dtype=np.float32)
        bounds = self.config.voxel_bounds_m
        if bounds is not None:
            mins = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
            maxs = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
        else:
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
        dims, n_vx = _voxel_grid_dims(mins, maxs, voxel_size)

        with wp.ScopedDevice(self.device):
            # Grow the shared scratch buffers on demand. sums/counts are left
            # zeroed by the previous call's compact pass, so no re-zeroing here.
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

            pts_wp = wp.array(pts, dtype=wp.vec3)
            min_corner = wp.vec3(float(mins[0]), float(mins[1]), float(mins[2]))
            n_out = _voxel_bin(
                pts_wp,
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
            out = self._vx_out.numpy()[:n_out]

        return out.astype(points.dtype, copy=False)

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

        with wp.ScopedDevice(self.device):
            if cfg.voxel_size_m is not None and cfg.voxel_size_m > 0.0:
                with prof.stage("voxel_downsample"):
                    source = self._voxel_downsample(source, cfg.voxel_size_m)
                    if cfg.voxel_target:
                        target = self._voxel_downsample(target, cfg.voxel_size_m)

            with prof.stage("upload"):
                src_wp = wp.array(np.ascontiguousarray(source, dtype=np.float32), dtype=wp.vec3)
                tgt_wp = wp.array(np.ascontiguousarray(target, dtype=np.float32), dtype=wp.vec3)

            with prof.stage("grid_build"):
                grid = self._ensure_grid(grid_radius, target)
                grid.build(points=tgt_wp, radius=float(grid_radius))

            with prof.stage("normals"):
                normals_wp = wp.empty(len(target), dtype=wp.vec3)
                valid_wp = wp.empty(len(target), dtype=wp.int32)
                wp.launch(
                    estimate_normals_kernel,
                    dim=len(target),
                    inputs=[
                        grid.id,
                        tgt_wp,
                        float(cfg.normal_radius_m),
                        int(cfg.normal_min_neighbors),
                        int(cfg.normal_power_iters),
                    ],
                    outputs=[normals_wp, valid_wp],
                )

            transformed_wp = wp.empty(len(source), dtype=wp.vec3)
            JtJ_wp = wp.zeros((6, 6), dtype=wp.float32)
            Jtr_wp = wp.zeros(6, dtype=wp.float32)
            cost_wp = wp.zeros(1, dtype=wp.float32)
            inliers_wp = wp.zeros(1, dtype=wp.int32)

            T = (
                np.eye(4, dtype=np.float64)
                if init_pose is None
                else np.asarray(init_pose, dtype=np.float64).copy()
            )

            converged = False
            final_cost = float("inf")
            final_inliers = 0
            iters_run = 0

            for it in range(cfg.max_iters):
                iters_run = it + 1
                R = T[:3, :3].astype(np.float32)
                t = T[:3, 3].astype(np.float32)

                with prof.stage("iterations"):
                    JtJ_wp.zero_()
                    Jtr_wp.zero_()
                    cost_wp.zero_()
                    inliers_wp.zero_()
                    wp.launch(
                        transform_points_kernel,
                        dim=len(source),
                        inputs=[src_wp, wp.mat33(R.flatten().tolist()), wp.vec3(*t.tolist())],
                        outputs=[transformed_wp],
                    )
                    wp.launch(
                        accumulate_system_kernel,
                        dim=len(source),
                        inputs=[
                            grid.id,
                            tgt_wp,
                            normals_wp,
                            valid_wp,
                            transformed_wp,
                            float(cfg.max_correspondence_dist_m),
                            float(cfg.huber_delta),
                        ],
                        outputs=[JtJ_wp, Jtr_wp, cost_wp, inliers_wp],
                    )
                wp.synchronize()  # pull the normal equations back to solve on the host

                H_upper = JtJ_wp.numpy().astype(np.float64)
                H = np.triu(H_upper) + np.triu(H_upper, 1).T
                g = Jtr_wp.numpy().astype(np.float64)
                n_in = int(inliers_wp.numpy()[0])
                c = float(cost_wp.numpy()[0])

                if n_in == 0:
                    if self.verbose:
                        print(f"[icp] iter {it}: no inliers, aborting")
                    break

                H += cfg.damping * np.eye(6)
                try:
                    delta = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    if self.verbose:
                        print(f"[icp] iter {it}: singular system, aborting")
                    break

                T = _exp_se3(delta) @ T
                final_cost = c
                final_inliers = n_in

                dr = float(np.linalg.norm(delta[:3]))
                dt = float(np.linalg.norm(delta[3:]))
                if self.verbose:
                    mean_cost = c / max(n_in, 1)
                    print(
                        f"[icp] iter {it:2d}  inliers={n_in}  cost/pt={mean_cost:.5f}  "
                        f"|dω|={dr:.5f}  |dv|={dt:.5f}"
                    )
                if dr < cfg.convergence_rotation_rad and dt < cfg.convergence_translation_m:
                    converged = True
                    break

            prof.read()

        return IcpResult(
            pose=T,
            iterations=iters_run,
            final_cost=final_cost,
            num_inliers=final_inliers,
            converged=converged,
            timings_ms=prof.ms if profile else {},
        )
