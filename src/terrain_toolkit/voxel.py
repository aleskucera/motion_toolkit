"""Device-resident voxel downsampler (sparse hash, Warp).

One centroid per occupied voxel, entirely on device: a preallocated open-
addressing hash table (keyed on the voxel's integer cell coords) accumulates
sum+count per cell, then a compact pass emits centroids and clears the touched
slots. No host round trip, no per-call allocation, and — unlike a dense grid —
no bounds, extent, or `min/max` scan: cells are `floor(p / voxel)` in world
space, so memory is proportional to *occupancy*, not to a fixed volume.

`downsample(points, n)` takes and returns `wp.array`s on the device; only the
occupied-cell count crosses back to the host.
"""

from __future__ import annotations

import warp as wp

wp.init()

_EMPTY = wp.constant(wp.int64(-1))
# Knuth/Fibonacci multiplicative-hash constant (0x9E3779B97F4A7C15 as signed i64).
_HASH_MULT = wp.constant(wp.int64(-7046029254386353131))


@wp.func
def _cell_key(ix: int, iy: int, iz: int) -> wp.int64:
    """Pack signed cell coords into one int64 (21 bits each, ±1M cell range)."""
    kx = wp.int64(ix + (1 << 20))
    ky = wp.int64(iy + (1 << 20))
    kz = wp.int64(iz + (1 << 20))
    return (kx << wp.int64(42)) | (ky << wp.int64(21)) | kz


@wp.func
def _slot_of(key: wp.int64, mask: wp.int64) -> wp.int32:
    h = key * _HASH_MULT
    return wp.int32((h ^ (h >> wp.int64(29))) & mask)


@wp.kernel
def _accumulate_kernel(
    points: wp.array(dtype=wp.vec3),
    n: wp.int32,
    inv_voxel: wp.float32,
    cap_mask: wp.int64,  # capacity - 1 (capacity is a power of two)
    keys: wp.array(dtype=wp.int64),
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    slots: wp.array(dtype=wp.int32),
    slot_counter: wp.array(dtype=wp.int32),
):
    """Insert each point into its voxel cell (open addressing, linear probe).

    Exactly one thread wins the CAS that claims an empty slot for a new cell and
    appends it to `slots`; everyone else with the same cell falls into the
    `cur == key` branch and just accumulates.
    """
    i = wp.tid()
    if i >= n:
        return
    p = points[i]
    ix = int(wp.floor(p[0] * inv_voxel))
    iy = int(wp.floor(p[1] * inv_voxel))
    iz = int(wp.floor(p[2] * inv_voxel))
    key = _cell_key(ix, iy, iz)
    slot = _slot_of(key, cap_mask)
    cap = int(cap_mask) + 1
    for _ in range(cap):
        cur = wp.atomic_cas(keys, slot, _EMPTY, key)
        if cur == _EMPTY:
            s = wp.atomic_add(slot_counter, 0, 1)
            slots[s] = slot
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
    out: wp.array(dtype=wp.vec3),
):
    """Emit one centroid per occupied slot and reset that slot to empty, so the
    table is left clean for the next call (O(occupied), not O(capacity))."""
    s = slots[wp.tid()]
    out[wp.tid()] = sums[s] / float(counts[s])
    keys[s] = _EMPTY
    sums[s] = wp.vec3(0.0, 0.0, 0.0)
    counts[s] = 0


class VoxelGrid:
    """Sparse-hash voxel downsampler with fixed preallocated buffers.

    Sized for up to `max_points` input points (and hence occupied cells). The
    hash table is left empty between calls by the compact pass, so buffers are
    never re-zeroed in bulk. Not thread-safe: one `VoxelGrid` per caller.
    """

    def __init__(
        self,
        voxel_size: float,
        *,
        max_points: int,
        load_factor: float = 0.5,
        device: wp.context.Device | None = None,
    ):
        self.device = wp.get_device(device)
        self.voxel = float(voxel_size)
        self.max_points = int(max_points)
        capacity = 1
        while capacity < int(self.max_points / load_factor):
            capacity <<= 1  # power of two so the hash can mask instead of modulo
        self.capacity = capacity
        with wp.ScopedDevice(self.device):
            self._keys = wp.full(capacity, wp.int64(-1), dtype=wp.int64)
            self._sums = wp.zeros(capacity, dtype=wp.vec3)
            self._counts = wp.zeros(capacity, dtype=wp.int32)
            self._slots = wp.empty(self.max_points, dtype=wp.int32)
            self._slot_counter = wp.zeros(1, dtype=wp.int32)
            self._out = wp.empty(self.max_points, dtype=wp.vec3)

    def downsample(self, points: wp.array, n: int) -> tuple[wp.array, int]:
        """One centroid per occupied voxel; returns `(centroids_device, count)`.

        `points` is a device `wp.array(vec3)`; only its first `n` entries are read
        (`n <= max_points`). The returned array is an owned buffer, valid until
        the next `downsample()` call.
        """
        if n > self.max_points:
            raise ValueError(f"n={n} exceeds max_points={self.max_points}")
        if n == 0:
            return self._out, 0
        with wp.ScopedDevice(self.device):
            self._slot_counter.zero_()
            wp.launch(
                _accumulate_kernel,
                dim=n,
                inputs=[points, n, 1.0 / self.voxel, wp.int64(self.capacity - 1)],
                outputs=[self._keys, self._sums, self._counts, self._slots, self._slot_counter],
            )
            wp.synchronize()
            n_out = int(self._slot_counter.numpy()[0])
            if n_out:
                wp.launch(
                    _compact_kernel,
                    dim=n_out,
                    inputs=[self._slots, self._keys, self._sums, self._counts],
                    outputs=[self._out],
                )
            return self._out, n_out
