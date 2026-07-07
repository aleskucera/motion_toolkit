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

# Empty-slot `last_seen` before any observation stamps it. Well below any real frame
# index so the first atomic-max always wins.
_AGE_SENTINEL = -2_000_000_000


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


@wp.kernel
def _accumulate_stamped_kernel(
    points: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
    stamps: wp.array(dtype=wp.int32),  # per-point frame index it was observed
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
    last_seen: wp.array(dtype=wp.int32),  # per-cell max observed frame (atomic-max)
):
    """Like `_accumulate_masked_kernel`, but also carries a per-point frame stamp into
    a per-cell `last_seen = max(stamp)` — the recency signal the eviction pass reads."""
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
    s = stamps[i]
    for _ in range(cap):
        cur = wp.atomic_cas(keys, slot, _EMPTY, key)
        if cur == _EMPTY:
            slots[wp.atomic_add(slot_counter, 0, 1)] = slot
            wp.atomic_add(sums, slot, p)
            wp.atomic_add(counts, slot, 1)
            wp.atomic_max(last_seen, slot, s)
            return
        if cur == key:
            wp.atomic_add(sums, slot, p)
            wp.atomic_add(counts, slot, 1)
            wp.atomic_max(last_seen, slot, s)
            return
        slot = wp.int32((wp.int64(slot) + wp.int64(1)) & cap_mask)


@wp.kernel
def _compact_evict_kernel(
    slots: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    last_seen: wp.array(dtype=wp.int32),
    cx: wp.float32,
    cy: wp.float32,
    view_r2: wp.float32,  # (view range)^2; a cell within it is "should have been seen"
    frame: wp.int32,
    max_unseen: wp.int32,
    age_sentinel: wp.int32,
    out_counter: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
    out_ages: wp.array(dtype=wp.int32),
):
    """Emit each occupied cell's centroid + last_seen, EXCEPT cells that are in view yet
    stale (frame - last_seen > max_unseen) — those are dynamic residue and get dropped.
    Cells out of view are always kept (occluded / off-camera, not contradicted). Resets
    the slot for the next step whether kept or evicted."""
    s = slots[wp.tid()]
    c = sums[s] / float(counts[s])
    ls = last_seen[s]
    keys[s] = _EMPTY
    sums[s] = wp.vec3(0.0, 0.0, 0.0)
    counts[s] = 0
    last_seen[s] = age_sentinel
    in_view = ((c[0] - cx) * (c[0] - cx) + (c[1] - cy) * (c[1] - cy)) < view_r2
    if in_view and (frame - ls) > max_unseen:
        return
    idx = wp.atomic_add(out_counter, 0, 1)
    out_points[idx] = c
    out_ages[idx] = ls


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
            # Recency pruning (opt-in via step(max_unseen=...)): per-cell last-seen frame
            # and a compaction counter for the eviction pass. Untouched on the legacy path.
            self._last_seen = wp.full(capacity, _AGE_SENTINEL, dtype=wp.int32)
            self._out_counter = wp.zeros(1, dtype=wp.int32)

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

    def _accumulate_stamped(
        self, points: wp.array, mask: wp.array, stamps: wp.array, cx: float, cy: float
    ) -> None:
        """Bin `points` (kept where `mask != 0`) into the hash, carrying per-point frame
        `stamps` into the per-cell `last_seen` (max)."""
        min_corner = wp.vec3(cx - self.radius, cy - self.radius, self.z0)
        wp.launch(
            _accumulate_stamped_kernel,
            dim=len(points),
            inputs=[
                points,
                mask,
                stamps,
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
            outputs=[
                self._keys,
                self._sums,
                self._counts,
                self._slots,
                self._slot_counter,
                self._last_seen,
            ],
        )

    def step(
        self,
        map_wp: wp.array | None,
        carve_mask: wp.array | None,
        points_wp: wp.array,
        valid_wp: wp.array,
        center: tuple[float, float],
        *,
        map_ages: wp.array | None = None,
        frame: int | None = None,
        max_unseen: int | None = None,
        view_range: float | None = None,
    ) -> wp.array | tuple[wp.array, wp.array]:
        """Return the new map: (carved map ∪ valid new points), cropped + voxel-thinned.

        `map_wp` is the previous map (None on the first frame). `carve_mask` (int32,
        len == map) marks map points to keep (None → keep all). `points_wp` are the
        new scan's per-beam points with `valid_wp` (int32) selecting real returns.
        `map` + `points` together must not exceed `max_points`.

        Recency pruning (opt-in): pass `max_unseen` (+ `frame`, `view_range`) to also
        evict cells that are within `view_range` of `center` yet have not been observed
        for more than `max_unseen` frames — dynamic residue the geometric carve missed.
        Cells out of view are preserved. In this mode the return is `(new_map, new_ages)`;
        thread `new_ages` back in as `map_ages` next step (None → treat map as fresh).
        The legacy path (no `max_unseen`) returns just `new_map` and is byte-identical.
        """
        cx, cy = float(center[0]), float(center[1])
        n_map = 0 if map_wp is None else len(map_wp)
        n_pts = len(points_wp)
        recency = max_unseen is not None
        with wp.ScopedDevice(self.device):
            if n_map + n_pts == 0:
                empty = wp.zeros(0, dtype=wp.vec3)
                return (empty, wp.zeros(0, dtype=wp.int32)) if recency else empty

            self._slot_counter.zero_()
            if not recency:
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

            # Recency path: stamp cells, then compact + evict stale in-view cells.
            fr = int(frame)
            if n_map:
                keep = carve_mask if carve_mask is not None else wp.full(n_map, 1, dtype=wp.int32)
                mstamp = map_ages if map_ages is not None else wp.full(n_map, fr, dtype=wp.int32)
                self._accumulate_stamped(map_wp, keep, mstamp, cx, cy)
            if n_pts:
                sstamp = wp.full(n_pts, fr, dtype=wp.int32)
                self._accumulate_stamped(points_wp, valid_wp, sstamp, cx, cy)
            wp.synchronize()
            n_used = int(self._slot_counter.numpy()[0])
            view_r2 = float(view_range) ** 2 if view_range is not None else 1.0e18
            full_pts = wp.empty(n_used, dtype=wp.vec3)
            full_ages = wp.empty(n_used, dtype=wp.int32)
            self._out_counter.zero_()
            if n_used:
                wp.launch(
                    _compact_evict_kernel,
                    dim=n_used,
                    inputs=[
                        self._slots,
                        self._keys,
                        self._sums,
                        self._counts,
                        self._last_seen,
                        cx,
                        cy,
                        view_r2,
                        fr,
                        int(max_unseen),
                        _AGE_SENTINEL,
                    ],
                    outputs=[self._out_counter, full_pts, full_ages],
                )
            wp.synchronize()
            n_out = int(self._out_counter.numpy()[0])
            new_map = wp.empty(n_out, dtype=wp.vec3)
            new_ages = wp.empty(n_out, dtype=wp.int32)
            if n_out:
                wp.copy(new_map, full_pts, 0, 0, n_out)
                wp.copy(new_ages, full_ages, 0, 0, n_out)
            return new_map, new_ages
