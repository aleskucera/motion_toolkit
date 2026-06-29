"""Differentiable wheel-envelope dilation (Warp).

Grayscale morphological dilation of the raw elevation by the spherical wheel cap:

    envelope[i,j] = max_{|d|<=R} ( elevation[i+dy, j+dx] + sqrt(R^2 - d^2) - R ),  d = |off|*cell_size

The dilation = arg-max of (elevation[neighbor] + cap) over the disk. The gradient is analytical: the
adjoint scatters to the arg-max (contact) cell. Implementations:
  - 2D `_contact_kernel` + `_gather_kernel` (one thread per output cell): the forward planner's
    shared [ny, nx] terrain, and the FD oracle. CPU or CUDA.
  - batched [B, ny, nx] for DifferentiableSimulator -- split into `contact` (off-tape: pick the
    arg-max offset `best_k`) + `gather_bt` (on-tape: envelope = elevation[contact] + cap, whose
    cheap scatter adjoint IS the analytical gradient). Contact has two forms producing the same
    `best_k`: `contact_offset_table` (CPU+CUDA) and `make_tiled_contact` (shared-memory tiled
    arg-max, ~2x faster, CUDA-only, needs an edge-padded input via `pad_edge`).
"""

import math

import numpy as np
import warp as wp


@wp.kernel
def _contact_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    cell_size: float,
    wheel_radius: float,
    env_radius: int,
    contact_iy: wp.array2d(dtype=wp.int32),
    contact_ix: wp.array2d(dtype=wp.int32),
    contact_cap: wp.array2d(dtype=wp.float32),
):
    """Non-diff pass: pick the contact cell (arg-max of elevation[neighbor] + cap)."""
    iy, ix = wp.tid()
    ny = elevation.shape[0]
    nx = elevation.shape[1]
    best_lift = float(-1.0e9)
    best_iy = iy
    best_ix = ix
    best_cap = float(0.0)
    for dy in range(-env_radius, env_radius + 1):
        for dx in range(-env_radius, env_radius + 1):
            dist = wp.sqrt(float(dy * dy + dx * dx)) * cell_size
            if dist <= wheel_radius:
                cap = wp.sqrt(wheel_radius * wheel_radius - dist * dist) - wheel_radius
                qy = wp.clamp(iy + dy, 0, ny - 1)
                qx = wp.clamp(ix + dx, 0, nx - 1)
                lift = elevation[qy, qx] + cap
                if lift > best_lift:
                    best_lift = lift
                    best_iy = qy
                    best_ix = qx
                    best_cap = cap
    contact_iy[iy, ix] = best_iy
    contact_ix[iy, ix] = best_ix
    contact_cap[iy, ix] = best_cap


@wp.kernel
def _gather_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    contact_iy: wp.array2d(dtype=wp.int32),
    contact_ix: wp.array2d(dtype=wp.int32),
    contact_cap: wp.array2d(dtype=wp.float32),
    envelope: wp.array2d(dtype=wp.float32),
):
    """Diff pass: envelope = elevation[contact cell] + cap. Adjoint scatters to it."""
    iy, ix = wp.tid()
    envelope[iy, ix] = elevation[contact_iy[iy, ix], contact_ix[iy, ix]] + contact_cap[iy, ix]


# --- batched dilation for DifferentiableSimulator: contact (off-tape, picks the arg-max offset k)
# + gather (on-tape, the analytical gradient). Splitting them keeps the expensive arg-max off the
# tape and makes the backward a cheap scatter (vs. autodiffing the whole convolution). `contact`
# has two implementations -- an offset-table scan (A, CPU+CUDA) and a shared-memory tiled arg-max
# (B, CUDA, ~2x faster) -- that produce the SAME `best_k`, so the single `gather` serves both.
def wheel_offset_table(
    env_radius: int, cell_size: float, wheel_radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-circle spherical-cap offsets (dy, dx, cap), precomputed once on the host. Shared by the
    offset-table contact and the gather (both index into it via `best_k`)."""
    dy_l, dx_l, cap_l = [], [], []
    for dy in range(-env_radius, env_radius + 1):
        for dx in range(-env_radius, env_radius + 1):
            dist = math.sqrt(float(dy * dy + dx * dx)) * cell_size
            if dist <= wheel_radius:
                dy_l.append(dy)
                dx_l.append(dx)
                cap_l.append(math.sqrt(wheel_radius * wheel_radius - dist * dist) - wheel_radius)
    return np.array(dy_l, np.int32), np.array(dx_l, np.int32), np.array(cap_l, np.float32)


@wp.kernel
def contact_offset_table(
    elevation: wp.array3d(dtype=wp.float32),  # [B, ny, nx]
    off_dy: wp.array(dtype=wp.int32),
    off_dx: wp.array(dtype=wp.int32),
    off_cap: wp.array(dtype=wp.float32),
    best_k: wp.array3d(dtype=wp.float32),  # arg-max offset index per cell (float; gather casts int)
):
    """Contact pass A (CPU+CUDA): per-cell arg-max over the offset table; write the winning index."""
    b, iy, ix = wp.tid()
    ny = elevation.shape[1]
    nx = elevation.shape[2]
    best_lift = float(-1.0e9)
    bk = int(0)
    for k in range(off_dy.shape[0]):
        qy = wp.clamp(iy + off_dy[k], 0, ny - 1)
        qx = wp.clamp(ix + off_dx[k], 0, nx - 1)
        lift = elevation[b, qy, qx] + off_cap[k]
        if lift > best_lift:
            best_lift = lift
            bk = k
    best_k[b, iy, ix] = float(bk)


@wp.kernel
def gather_bt(
    elevation: wp.array3d(dtype=wp.float32),
    best_k: wp.array3d(dtype=wp.float32),
    off_dy: wp.array(dtype=wp.int32),
    off_dx: wp.array(dtype=wp.int32),
    off_cap: wp.array(dtype=wp.float32),
    envelope: wp.array3d(dtype=wp.float32),
):
    """Gather (on-tape, the analytical gradient): envelope = elevation[contact] + cap, where the
    contact offset is `best_k` (fixed = the subgradient). The adjoint is a cheap scatter of
    envelope.grad to the contact cell -- no autodiff through the arg-max."""
    b, iy, ix = wp.tid()
    ny = elevation.shape[1]
    nx = elevation.shape[2]
    k = int(best_k[b, iy, ix])
    qy = wp.clamp(iy + off_dy[k], 0, ny - 1)
    qx = wp.clamp(ix + off_dx[k], 0, nx - 1)
    envelope[b, iy, ix] = elevation[b, qy, qx] + off_cap[k]


# --- shared-memory tiled arg-max contact (B, CUDA): ~2x faster contact, off-tape (non-diff) ---
@wp.func
def _addf(v: wp.float32, c: wp.float32):
    return v + c


@wp.func
def _maxf(a: wp.float32, b: wp.float32):
    return wp.max(a, b)


@wp.func
def _sel_k(lift: wp.float32, acc: wp.float32, bk: wp.float32, kf: wp.float32):
    """Index update: if this offset's lifted value beats the running max, take its index k."""
    if lift > acc:
        return kf
    return bk


@wp.kernel
def pad_edge(raw: wp.array3d(dtype=wp.float32), pad_r: int, padded: wp.array3d(dtype=wp.float32)):
    """Edge-replicate pad each [ny, nx] slice into a tile-aligned [ny+slack, nx+slack] so the tiled
    halo loads never go out of bounds (used by the tiled contact)."""
    b, py, px = wp.tid()
    ny = raw.shape[1]
    nx = raw.shape[2]
    padded[b, py, px] = raw[b, wp.clamp(py - pad_r, 0, ny - 1), wp.clamp(px - pad_r, 0, nx - 1)]


def make_tiled_contact(env_radius: int, tile: int = 16):
    """Build a batched tiled arg-max contact kernel specialized to `env_radius` (only the tile/halo
    dims are baked; the offset table is passed at launch, so the disk is a RUNTIME loop -- not
    unrolled. Unrolling the disk OR carrying the index in a vec2 both blow up Warp's compile;
    runtime loop + two float tiles (running max + running index) keeps it fast to compile). Maps
    over (B, ny_tiles, nx_tiles) via `launch_tiled`: each block loads its (tile+2R)^2 halo into a
    shared tile once, then arg-max-accumulates the cap-shifted views. Writes `best_k` (the contact
    offset per cell). Input must be edge-padded (`pad_edge`). Off-tape (the `gather` supplies grad).
    """
    R = env_radius
    T = tile
    HALO = T + 2 * R

    @wp.kernel
    def contact_tiled(
        elev_pad: wp.array3d(dtype=wp.float32),  # [B, ny+slack, nx+slack] edge-padded
        off_dy: wp.array(dtype=wp.int32),
        off_dx: wp.array(dtype=wp.int32),
        off_cap: wp.array(dtype=wp.float32),
        best_k: wp.array3d(dtype=wp.float32),  # [B, ny, nx] arg-max offset index
    ):
        b, ti, tj = wp.tid()
        halo = wp.tile_load(
            elev_pad[b],
            shape=(HALO, HALO),
            offset=(ti * T, tj * T),
            storage="shared",
            bounds_check=True,
        )
        acc = wp.tile_full((T, T), -1.0e9, dtype=wp.float32, storage="register")
        bk = wp.tile_full((T, T), 0.0, dtype=wp.float32, storage="register")
        for k in range(off_dy.shape[0]):
            win = wp.tile_view(halo, offset=(R + off_dy[k], R + off_dx[k]), shape=(T, T))
            capt = wp.tile_full((T, T), off_cap[k], dtype=wp.float32, storage="register")
            lifted = wp.tile_map(_addf, win, capt)
            kt = wp.tile_full((T, T), float(k), dtype=wp.float32, storage="register")
            bk = wp.tile_map(_sel_k, lifted, acc, bk, kt)  # update index BEFORE acc (uses old max)
            acc = wp.tile_map(_maxf, acc, lifted)
        wp.tile_store(best_k[b], bk, offset=(ti * T, tj * T), bounds_check=True)

    return contact_tiled
