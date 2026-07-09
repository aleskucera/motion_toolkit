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
# Empty-slot seen-through streak before any point min-reduces into it — well ABOVE any real
# streak so the first atomic-min always wins.
_STREAK_SENTINEL = 2_000_000_000


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
def _compact_stamped_kernel(
    slots: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    last_seen: wp.array(dtype=wp.int32),
    age_sentinel: wp.int32,
    out_points: wp.array(dtype=wp.vec3),
    out_ages: wp.array(dtype=wp.int32),
):
    """Emit each occupied cell's centroid + its last_seen frame, and reset the slot for the
    next step. No eviction here — dropping stale/dynamic cells is done upstream via the carve
    mask (see DynamicPointFilter.carve_recency); this only carries the recency stamp forward."""
    s = slots[wp.tid()]
    out_points[wp.tid()] = sums[s] / float(counts[s])
    out_ages[wp.tid()] = last_seen[s]
    keys[s] = _EMPTY
    sums[s] = wp.vec3(0.0, 0.0, 0.0)
    counts[s] = 0
    last_seen[s] = age_sentinel


@wp.kernel
def _accumulate_streak_kernel(
    points: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
    streaks: wp.array(dtype=wp.int32),
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
    streak: wp.array(dtype=wp.int32),
):
    """Like the stamped accumulate but MIN-reduces a per-point seen-through streak into the cell,
    so any point with streak 0 (a fresh scan return = re-confirmed) resets the cell's streak."""
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
    s = streaks[i]
    for _ in range(cap):
        cur = wp.atomic_cas(keys, slot, _EMPTY, key)
        if cur == _EMPTY:
            slots[wp.atomic_add(slot_counter, 0, 1)] = slot
            wp.atomic_add(sums, slot, p)
            wp.atomic_add(counts, slot, 1)
            wp.atomic_min(streak, slot, s)
            return
        if cur == key:
            wp.atomic_add(sums, slot, p)
            wp.atomic_add(counts, slot, 1)
            wp.atomic_min(streak, slot, s)
            return
        slot = wp.int32((wp.int64(slot) + wp.int64(1)) & cap_mask)


@wp.kernel
def _compact_streak_kernel(
    slots: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    streak: wp.array(dtype=wp.int32),
    streak_sentinel: wp.int32,
    out_points: wp.array(dtype=wp.vec3),
    out_streak: wp.array(dtype=wp.int32),
):
    """Emit each occupied cell's centroid + its min-reduced seen-through streak, reset the slot."""
    s = slots[wp.tid()]
    out_points[wp.tid()] = sums[s] / float(counts[s])
    out_streak[wp.tid()] = streak[s]
    keys[s] = _EMPTY
    sums[s] = wp.vec3(0.0, 0.0, 0.0)
    counts[s] = 0
    streak[s] = streak_sentinel


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
        # Cell-grid extent (for the crop / cell index; storage is sparse below). +2 (not +1):
        # the grid origin is snapped DOWN to the voxel lattice (see _min_corner), which can push
        # it up to one voxel below cx-radius, so one extra cell is needed to still reach cx+radius.
        self.dx = int(2.0 * self.radius / self.voxel) + 2
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
            # Recency tracking (opt-in via step(frame=...)): per-cell last-seen frame.
            # Untouched on the legacy path.
            self._last_seen = wp.full(capacity, _AGE_SENTINEL, dtype=wp.int32)
            # Consecutive-free carve (opt-in via step(map_streak=...)): per-cell seen-through
            # streak. Untouched on the legacy / recency paths.
            self._streak = wp.full(capacity, _STREAK_SENTINEL, dtype=wp.int32)

    def _min_corner(self, cx: float, cy: float) -> wp.vec3:
        """Grid origin SNAPPED to the global voxel lattice, so a world point falls in the same
        cell regardless of the robot pose. Anchoring it to the raw pose (cx - radius) shifts the
        grid's sub-voxel phase whenever the robot translates; the whole map is re-binned every
        step, so a shifted phase re-partitions it and merges/drops points that cross the moved
        cell boundaries — eroding regions the robot has driven away from (they stop being
        refilled by new scans). Snapping makes the per-step re-voxelization idempotent under
        translation. The crop still follows the robot via the radius test in the kernel."""
        mx = ((cx - self.radius) // self.voxel) * self.voxel
        my = ((cy - self.radius) // self.voxel) * self.voxel
        return wp.vec3(mx, my, self.z0)

    def _accumulate(self, points: wp.array, mask: wp.array, cx: float, cy: float) -> None:
        """Bin `points` (kept where `mask != 0`) into the shared hash table."""
        min_corner = self._min_corner(cx, cy)
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
        min_corner = self._min_corner(cx, cy)
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

    def _accumulate_streak(
        self, points: wp.array, mask: wp.array, streaks: wp.array, cx: float, cy: float
    ) -> None:
        """Bin `points` (kept where `mask != 0`), MIN-reducing per-point `streaks` into the cell."""
        min_corner = self._min_corner(cx, cy)
        wp.launch(
            _accumulate_streak_kernel,
            dim=len(points),
            inputs=[
                points,
                mask,
                streaks,
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
                self._streak,
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
        map_streak: wp.array | None = None,
    ) -> wp.array | tuple[wp.array, wp.array]:
        """Return the new map: (carved map ∪ valid new points), cropped + voxel-thinned.

        `map_wp` is the previous map (None on the first frame). `carve_mask` (int32,
        len == map) marks map points to keep (None → keep all). `points_wp` are the
        new scan's per-beam points with `valid_wp` (int32) selecting real returns.
        `map` + `points` together must not exceed `max_points`.

        Recency tracking (opt-in): pass `frame` to maintain a per-cell last-seen stamp and
        return `(new_map, new_ages)`; thread `new_ages` back in as `map_ages` next step
        (None → treat the map as seen this frame). Eviction of stale/dynamic cells is NOT
        done here — encode it in `carve_mask` (see `DynamicPointFilter.carve_recency`), which
        decides what to forget from actual sensor visibility. The legacy path (no `frame`)
        returns just `new_map` and is byte-identical.
        """
        cx, cy = float(center[0]), float(center[1])
        n_map = 0 if map_wp is None else len(map_wp)
        n_pts = len(points_wp)
        recency = frame is not None
        streak_mode = map_streak is not None  # consecutive-free carve: thread per-cell streak
        with wp.ScopedDevice(self.device):
            if n_map + n_pts == 0:
                empty = wp.zeros(0, dtype=wp.vec3)
                return (empty, wp.zeros(0, dtype=wp.int32)) if (recency or streak_mode) else empty

            self._slot_counter.zero_()
            if streak_mode:
                # Consecutive-free carve: carry the seen-through streak per cell. Map points bring
                # their (classify-updated) streak; scan returns stamp 0, so a re-observed cell
                # min-reduces to 0 (re-confirmed). carve_mask already dropped points that hit the
                # persist threshold this frame.
                if n_map:
                    keep = carve_mask if carve_mask is not None else wp.full(n_map, 1, dtype=wp.int32)
                    self._accumulate_streak(map_wp, keep, map_streak, cx, cy)
                if n_pts:
                    zeros = wp.zeros(n_pts, dtype=wp.int32)
                    self._accumulate_streak(points_wp, valid_wp, zeros, cx, cy)
                wp.synchronize()
                n_out = int(self._slot_counter.numpy()[0])
                new_map = wp.empty(n_out, dtype=wp.vec3)
                new_streak = wp.empty(n_out, dtype=wp.int32)
                if n_out:
                    wp.launch(
                        _compact_streak_kernel,
                        dim=n_out,
                        inputs=[self._slots, self._keys, self._sums, self._counts,
                                self._streak, _STREAK_SENTINEL],
                        outputs=[new_map, new_streak],
                    )
                return new_map, new_streak

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

            # Recency path: stamp cells (last_seen), then compact carrying the stamp forward.
            # `carve_mask` already dropped the stale/dynamic map points upstream.
            fr = int(frame)
            if n_map:
                keep = carve_mask if carve_mask is not None else wp.full(n_map, 1, dtype=wp.int32)
                mstamp = map_ages if map_ages is not None else wp.full(n_map, fr, dtype=wp.int32)
                self._accumulate_stamped(map_wp, keep, mstamp, cx, cy)
            if n_pts:
                sstamp = wp.full(n_pts, fr, dtype=wp.int32)
                self._accumulate_stamped(points_wp, valid_wp, sstamp, cx, cy)
            wp.synchronize()
            n_out = int(self._slot_counter.numpy()[0])
            new_map = wp.empty(n_out, dtype=wp.vec3)
            new_ages = wp.empty(n_out, dtype=wp.int32)
            if n_out:
                wp.launch(
                    _compact_stamped_kernel,
                    dim=n_out,
                    inputs=[
                        self._slots,
                        self._keys,
                        self._sums,
                        self._counts,
                        self._last_seen,
                        _AGE_SENTINEL,
                    ],
                    outputs=[new_map, new_ages],
                )
            return new_map, new_ages
