"""On-device rolling point-cloud accumulator (Warp).

The temporal-fusion stage of the pipeline (scans → **accumulate + carve** →
heightmap → traversability). Robot-agnostic: it takes point clouds plus an
optional carve mask (e.g. from `DynamicPointFilter.carve`) — the same class
backs both the offline sim demo and on-robot accumulation.

Keeps the accumulated map resident on the GPU across frames so the per-frame
pipeline (carve → add new returns → crop to a radius → voxel-thin) never rounds
through host memory. Each `step()` accumulates the carved map and the new scan
into a fixed voxel grid with two masked launches (skipping carved / out-of-radius
/ invalid points), then emits one centroid per occupied voxel.

The grid is dense (millions of cells) but a scan only fills a few thousand, so
the zero and compact passes track the *occupied* cells: a cell records its index
the frame it first fills, and only those cells are cleared next frame and scanned
at compact — making both O(occupied points) instead of O(grid cells).
"""

from __future__ import annotations

import warp as wp

wp.init()

_MAX_CELLS = 20_000_000


@wp.kernel
def _voxel_accumulate_masked_kernel(
    points: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
    cx: wp.float32,
    cy: wp.float32,
    r2: wp.float32,
    min_corner: wp.vec3,
    inv_voxel: wp.float32,
    dx: wp.int32,
    dy: wp.int32,
    dz: wp.int32,
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    occupied: wp.array(dtype=wp.int32),
    occ_counter: wp.array(dtype=wp.int32),
):
    """Sum each kept point into its voxel cell, recording newly-filled cells.

    Skips invalid points and points outside the xy crop radius, so the carve /
    crop / bin steps fuse into this one launch (no separate mask pass). The first
    thread to fill a cell (its count goes 0→1) appends the cell index to
    `occupied`, so the clear and compact passes can touch only filled cells.
    """
    i = wp.tid()
    if valid[i] == 0:
        return
    p = points[i]
    if (p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy) > r2:
        return
    ix = int((p[0] - min_corner[0]) * inv_voxel)
    iy = int((p[1] - min_corner[1]) * inv_voxel)
    iz = int((p[2] - min_corner[2]) * inv_voxel)
    if ix < 0 or ix >= dx or iy < 0 or iy >= dy or iz < 0 or iz >= dz:
        return
    idx = (ix * dy + iy) * dz + iz
    # atomic_add returns the previous count; exactly one thread sees 0 for a cell,
    # so each occupied cell is appended to the list exactly once.
    if wp.atomic_add(counts, idx, 1) == 0:
        occupied[wp.atomic_add(occ_counter, 0, 1)] = idx
    wp.atomic_add(sums, idx, p)


@wp.kernel
def _voxel_clear_kernel(
    occupied: wp.array(dtype=wp.int32),
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
):
    """Reset only the cells filled last frame back to empty (invariant: every
    non-zero cell is in `occupied`, so this leaves the whole grid zeroed)."""
    idx = occupied[wp.tid()]
    sums[idx] = wp.vec3(0.0, 0.0, 0.0)
    counts[idx] = 0


@wp.kernel
def _voxel_compact_kernel(
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    occupied: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
):
    """Write one centroid per occupied voxel (indexed straight off `occupied`)."""
    idx = occupied[wp.tid()]
    out_points[wp.tid()] = sums[idx] / float(counts[idx])


class DeviceMapAccumulator:
    """Rolling map kept on-device: carve + add + crop + voxel-thin per frame.

    Fixed robot-centric voxel grid (`radius` half-extent in xy, `z_bounds` in z),
    so bounds come from the robot pose with no host readback. `step()` returns the
    new map as a `wp.array(vec3)` living on `device`.
    """

    def __init__(
        self,
        voxel_size: float,
        radius: float,
        *,
        z_bounds: tuple[float, float] = (-2.0, 6.0),
        device: wp.context.Device | None = None,
    ):
        self.device = wp.get_device(device)
        self.voxel = float(voxel_size)
        self.radius = float(radius)
        self.z0, self.z1 = float(z_bounds[0]), float(z_bounds[1])
        self.dx = int(2.0 * self.radius / self.voxel) + 1
        self.dy = self.dx
        self.dz = int((self.z1 - self.z0) / self.voxel) + 1
        n_vx = self.dx * self.dy * self.dz
        if n_vx > _MAX_CELLS:
            raise ValueError(
                f"voxel grid has {n_vx} cells (>{_MAX_CELLS}); coarsen voxel_size or shrink radius"
            )
        # Count of cells filled by the previous step(), so the next one clears
        # exactly those (grid starts fully zeroed, so nothing to clear yet).
        self._prev_occ = 0
        with wp.ScopedDevice(self.device):
            self._sums = wp.zeros(n_vx, dtype=wp.vec3)
            self._counts = wp.zeros(n_vx, dtype=wp.int32)
            # At most n_vx distinct cells can be occupied in a frame.
            self._occupied = wp.empty(n_vx, dtype=wp.int32)
            self._occ_counter = wp.zeros(1, dtype=wp.int32)

    def _accumulate(
        self,
        points: wp.array,
        mask: wp.array,
        cx: float,
        cy: float,
        min_corner: wp.vec3,
    ) -> None:
        """Bin `points` (kept where `mask != 0`) into the shared grid."""
        wp.launch(
            _voxel_accumulate_masked_kernel,
            dim=len(points),
            inputs=[
                points,
                mask,
                cx,
                cy,
                self.radius * self.radius,
                min_corner,
                1.0 / self.voxel,
                self.dx,
                self.dy,
                self.dz,
            ],
            outputs=[self._sums, self._counts, self._occupied, self._occ_counter],
        )

    def step(
        self,
        map_wp: wp.array | None,
        carve_mask: wp.array | None,
        points_wp: wp.array,
        valid_wp: wp.array,
        center: tuple[float, float],
    ) -> wp.array:
        """Return the new map: (carved map ∪ valid new points), cropped + voxel-thinned.

        `map_wp` is the previous map (None on the first frame). `carve_mask` (int32,
        len == map) marks map points to keep (None → keep all). `points_wp` are the
        new scan's per-beam points with `valid_wp` (int32) selecting real returns.
        """
        cx, cy = float(center[0]), float(center[1])
        n_map = 0 if map_wp is None else len(map_wp)
        n_pts = len(points_wp)
        with wp.ScopedDevice(self.device):
            if n_map + n_pts == 0:
                self._prev_occ = 0
                return wp.zeros(0, dtype=wp.vec3)

            # Reset only last frame's occupied cells; the rest are already zero.
            if self._prev_occ:
                wp.launch(
                    _voxel_clear_kernel,
                    dim=self._prev_occ,
                    inputs=[self._occupied],
                    outputs=[self._sums, self._counts],
                )
            self._occ_counter.zero_()

            min_corner = wp.vec3(cx - self.radius, cy - self.radius, self.z0)
            # Two masked passes into the same grid — carved map, then new scan —
            # so no host-side concatenation of the two clouds is needed.
            if n_map:
                keep = carve_mask if carve_mask is not None else wp.full(n_map, 1, dtype=wp.int32)
                self._accumulate(map_wp, keep, cx, cy, min_corner)
            if n_pts:
                self._accumulate(points_wp, valid_wp, cx, cy, min_corner)

            wp.synchronize()
            n_out = int(self._occ_counter.numpy()[0])
            self._prev_occ = n_out

            new_map = wp.empty(n_out, dtype=wp.vec3)
            if n_out:
                wp.launch(
                    _voxel_compact_kernel,
                    dim=n_out,
                    inputs=[self._sums, self._counts, self._occupied],
                    outputs=[new_map],
                )
            return new_map
