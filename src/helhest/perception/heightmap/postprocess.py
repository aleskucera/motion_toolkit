from __future__ import annotations

import math

import numpy as np
import warp as wp

from .kernels import blur_axis_kernel
from .kernels import diffuse_step_kernel
from .kernels import downsample_kernel
from .kernels import upsample_inject_kernel


def _as_wp_float32(heightmap: np.ndarray | wp.array) -> tuple[wp.array, bool]:
    """Return (wp.array view/copy, input_was_numpy)."""
    if isinstance(heightmap, wp.array):
        return heightmap, False
    if heightmap.ndim != 2:
        raise ValueError("heightmap must be 2D.")
    data = np.ascontiguousarray(heightmap, dtype=np.float32)
    return wp.array(data, dtype=wp.float32), True


def _run_diffusion(a: wp.array, b: wp.array, fixed: wp.array, iters: int) -> wp.array:
    """Run `iters` diffusion steps. Uses a captured CUDA graph on GPU; on CPU
    (or any non-CUDA device) falls back to plain `wp.launch` calls."""
    if iters <= 0:
        return a
    dim = (a.shape[0], a.shape[1])
    if a.device.is_cuda:
        wp.capture_begin()
        try:
            wp.launch(diffuse_step_kernel, dim=dim, inputs=[a, fixed, b])
            wp.launch(diffuse_step_kernel, dim=dim, inputs=[b, fixed, a])
        finally:
            graph = wp.capture_end()
        for _ in range(iters // 2):
            wp.capture_launch(graph)
    else:
        for _ in range(iters // 2):
            wp.launch(diffuse_step_kernel, dim=dim, inputs=[a, fixed, b])
            wp.launch(diffuse_step_kernel, dim=dim, inputs=[b, fixed, a])
    if iters % 2:
        wp.launch(diffuse_step_kernel, dim=dim, inputs=[a, fixed, b])
        a, b = b, a
    return a


def _fixed_mask_from(heightmap_wp: wp.array) -> wp.array:
    """Build a (int32) mask of finite cells from a float32 wp array."""
    # Readback-free path: download tiny + re-upload is cheaper than a custom kernel
    # for the common case. Keep it simple; replace with a kernel later if needed.
    finite = np.isfinite(heightmap_wp.numpy()).astype(np.int32)
    return wp.array(finite, dtype=wp.int32)


def multigrid_inpaint(
    heightmap: np.ndarray | wp.array,
    iters_per_level: int = 50,
    coarse_iters: int = 200,
    min_size: int = 8,
) -> np.ndarray | wp.array:
    """Multigrid diffusion inpainting: solve coarse, upsample, refine at each level.

    Accepts numpy or `wp.array`; returns the matching type. The GPU path allocates
    a fresh pyramid each call (shape-dependent).
    """
    hm_wp, from_numpy = _as_wp_float32(heightmap)
    fixed = _fixed_mask_from(hm_wp)

    # Build pyramid (finest → coarsest).
    levels = [(hm_wp, fixed)]
    while levels[-1][0].shape[0] > min_size and levels[-1][0].shape[1] > min_size:
        prev, prev_f = levels[-1]
        ch = (prev.shape[0] + 1) // 2
        cw = (prev.shape[1] + 1) // 2
        coarse = wp.zeros((ch, cw), dtype=wp.float32)
        coarse_f = wp.zeros((ch, cw), dtype=wp.int32)
        wp.launch(downsample_kernel, dim=(ch, cw), inputs=[prev, prev_f, coarse, coarse_f])
        levels.append((coarse, coarse_f))

    # Solve coarsest level.
    a, fixed = levels[-1]
    b = wp.zeros_like(a)
    a = _run_diffusion(a, b, fixed, coarse_iters)
    levels[-1] = (a, fixed)

    # Upsample and refine from coarse to fine.
    for lvl in range(len(levels) - 2, -1, -1):
        fine, fine_fixed = levels[lvl]
        coarse_solved = levels[lvl + 1][0]
        wp.launch(
            upsample_inject_kernel,
            dim=(fine.shape[0], fine.shape[1]),
            inputs=[coarse_solved, fine, fine_fixed],
        )
        b = wp.zeros_like(fine)
        fine = _run_diffusion(fine, b, fine_fixed, iters_per_level)
        levels[lvl] = (fine, fine_fixed)

    out = levels[0][0]
    if from_numpy:
        wp.synchronize()
        return out.numpy().copy()
    return out


def diffuse_inpaint(
    heightmap: np.ndarray | wp.array,
    max_iters: int = 500,
) -> np.ndarray | wp.array:
    """Fill NaN cells by iterative diffusion from known cells (Laplace inpainting).

    Uses CUDA graph capture on GPU to replay the iteration loop without per-launch
    Python overhead; on CPU falls back to plain `wp.launch` calls.
    """
    a, from_numpy = _as_wp_float32(heightmap)
    fixed = _fixed_mask_from(a)
    b = wp.zeros(a.shape, dtype=wp.float32)

    full_rounds = max_iters // 2
    remainder = max_iters % 2
    if a.device.is_cuda:
        wp.capture_begin()
        try:
            wp.launch(diffuse_step_kernel, dim=a.shape, inputs=[a, fixed, b])
            wp.launch(diffuse_step_kernel, dim=a.shape, inputs=[b, fixed, a])
        finally:
            graph = wp.capture_end()
        for _ in range(full_rounds):
            wp.capture_launch(graph)
    else:
        for _ in range(full_rounds):
            wp.launch(diffuse_step_kernel, dim=a.shape, inputs=[a, fixed, b])
            wp.launch(diffuse_step_kernel, dim=a.shape, inputs=[b, fixed, a])
    if remainder:
        wp.launch(diffuse_step_kernel, dim=a.shape, inputs=[a, fixed, b])
        a, b = b, a
    if from_numpy:
        wp.synchronize()
        return a.numpy().copy()
    return a


def gaussian_smooth(
    heightmap: np.ndarray | wp.array,
    sigma: float = 1.0,
    truncate: float = 3.0,
) -> np.ndarray | wp.array:
    """NaN-aware separable Gaussian blur on a 2D heightmap (sigma in cells)."""
    if sigma <= 0.0:
        if isinstance(heightmap, wp.array):
            # Zero-sigma fast path: copy so the caller can't alias the output.
            out = wp.empty_like(heightmap)
            wp.copy(out, heightmap)
            return out
        return heightmap.copy()

    src, from_numpy = _as_wp_float32(heightmap)
    if len(src.shape) != 2:
        raise ValueError("heightmap must be 2D.")

    radius = max(1, int(math.ceil(truncate * sigma)))
    k = np.arange(-radius, radius + 1, dtype=np.float32)
    weights = np.exp(-(k * k) / (2.0 * sigma * sigma)).astype(np.float32)

    tmp = wp.zeros(src.shape, dtype=wp.float32)
    dst = wp.zeros(src.shape, dtype=wp.float32)
    w_wp = wp.array(weights, dtype=wp.float32)

    wp.launch(blur_axis_kernel, dim=src.shape, inputs=[src, w_wp, radius, 1, tmp])
    wp.launch(blur_axis_kernel, dim=src.shape, inputs=[tmp, w_wp, radius, 0, dst])

    if from_numpy:
        wp.synchronize()
        return dst.numpy().copy()
    return dst
