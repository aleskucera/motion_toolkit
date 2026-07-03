"""On-device rolling point-cloud accumulator (Warp).

The temporal-fusion stage of the pipeline (scans → **accumulate + carve** →
heightmap → traversability). Robot-agnostic: it takes point clouds plus an
optional carve mask (e.g. from `DynamicPointFilter.carve`) — the same class
backs both the offline sim demo and on-robot accumulation.

Keeps the accumulated map resident on the GPU across frames so the per-frame
pipeline (carve → add new returns → crop to a radius → voxel-thin) never rounds
through host memory. Each `step()` bins the carved map and the new scan into a
sparse voxel hash (open addressing, keyed on the robot-centric cell coords),
then emits one centroid per occupied cell. Storage is proportional to occupancy,
not to the grid volume — so a large-radius / fine-voxel map has no cell cap and
costs a few MB instead of gigabytes.
"""

from __future__ import annotations

import warp as wp

from ..voxel import _cell_key
from ..voxel import _EMPTY
from ..voxel import _slot_of

wp.init()


@wp.kernel
def _accumulate_masked_kernel(
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
    cap_mask: wp.int64,
    keys: wp.array(dtype=wp.int64),
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    slots: wp.array(dtype=wp.int32),
    slot_counter: wp.array(dtype=wp.int32),
):
    """Sum each kept point into its voxel cell (sparse hash, open addressing).

    Skips invalid points, points outside the xy crop radius, and points outside
    the [robot-centric] grid box, so the carve / crop / bin steps fuse into this
    one launch. The first thread to fill a cell appends its slot to `slots`.
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
    key = _cell_key(ix, iy, iz)
    slot = _slot_of(key, cap_mask)
    cap = int(cap_mask) + 1
    for _ in range(cap):
        cur = wp.atomic_cas(keys, slot, _EMPTY, key)
        if cur == _EMPTY:
            slots[wp.atomic_add(slot_counter, 0, 1)] = slot
            wp.atomic_add(sums, slot, p)
            wp.atomic_add(counts, slot, 1)
            return
        if cur == key:
            wp.atomic_add(sums, slot, p)
            wp.atomic_add(counts, slot, 1)
            return
        slot = wp.int32((wp.int64(slot) + wp.int64(1)) & cap_mask)


@wp.kernel
def _compact_kernel(
    slots: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
):
    """Emit one centroid per occupied cell and reset that slot, leaving the table
    empty for the next step (O(occupied), no full-table scan)."""
    s = slots[wp.tid()]
    out_points[wp.tid()] = sums[s] / float(counts[s])
    keys[s] = _EMPTY
    sums[s] = wp.vec3(0.0, 0.0, 0.0)
    counts[s] = 0


class DeviceMapAccumulator:
    """Rolling map kept on-device: carve + add + crop + voxel-thin per frame.

    Robot-centric crop (`radius` half-extent in xy, `z_bounds` in z) with the box
    origin taken from the robot pose — no host readback. Storage is a sparse voxel
    hash sized for up to `max_points` input points (map + scan) per `step()`, so
    memory scales with occupancy, not with the grid volume. `step()` returns the
    new map as a `wp.array(vec3)` on `device`.
    """

    def __init__(
        self,
        voxel_size: float,
        radius: float,
        *,
        z_bounds: tuple[float, float] = (-2.0, 6.0),
        max_points: int = 2_000_000,
        load_factor: float = 0.5,
        device: wp.context.Device | None = None,
    ):
        self.device = wp.get_device(device)
        self.voxel = float(voxel_size)
        self.radius = float(radius)
        self.z0, self.z1 = float(z_bounds[0]), float(z_bounds[1])
        # Cell-grid extent (for the crop / cell index; storage is sparse below).
        self.dx = int(2.0 * self.radius / self.voxel) + 1
        self.dy = self.dx
        self.dz = int((self.z1 - self.z0) / self.voxel) + 1
        self.max_points = int(max_points)
        capacity = 1
        while capacity < int(self.max_points / load_factor):
            capacity <<= 1  # power of two so the hash masks instead of taking modulo
        self.capacity = capacity
        with wp.ScopedDevice(self.device):
            self._keys = wp.full(capacity, wp.int64(-1), dtype=wp.int64)
            self._sums = wp.zeros(capacity, dtype=wp.vec3)
            self._counts = wp.zeros(capacity, dtype=wp.int32)
            self._slots = wp.empty(self.max_points, dtype=wp.int32)
            self._slot_counter = wp.zeros(1, dtype=wp.int32)

    def _accumulate(self, points: wp.array, mask: wp.array, cx: float, cy: float) -> None:
        """Bin `points` (kept where `mask != 0`) into the shared hash table."""
        min_corner = wp.vec3(cx - self.radius, cy - self.radius, self.z0)
        wp.launch(
            _accumulate_masked_kernel,
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
                wp.int64(self.capacity - 1),
            ],
            outputs=[self._keys, self._sums, self._counts, self._slots, self._slot_counter],
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
        `map` + `points` together must not exceed `max_points`.
        """
        cx, cy = float(center[0]), float(center[1])
        n_map = 0 if map_wp is None else len(map_wp)
        n_pts = len(points_wp)
        with wp.ScopedDevice(self.device):
            if n_map + n_pts == 0:
                return wp.zeros(0, dtype=wp.vec3)

            self._slot_counter.zero_()
            # Two masked passes into the same hash — carved map, then new scan —
            # so no host-side concatenation of the two clouds is needed.
            if n_map:
                keep = carve_mask if carve_mask is not None else wp.full(n_map, 1, dtype=wp.int32)
                self._accumulate(map_wp, keep, cx, cy)
            if n_pts:
                self._accumulate(points_wp, valid_wp, cx, cy)

            wp.synchronize()
            n_out = int(self._slot_counter.numpy()[0])

            new_map = wp.empty(n_out, dtype=wp.vec3)
            if n_out:
                wp.launch(
                    _compact_kernel,
                    dim=n_out,
                    inputs=[self._slots, self._keys, self._sums, self._counts],
                    outputs=[new_map],
                )
            return new_map
