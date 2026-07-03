from __future__ import annotations

import numpy as np
import warp as wp


def as_gpu(arr: np.ndarray | wp.array, device: wp.context.Device) -> wp.array:
    """Return `arr` as a float32 `wp.array` on `device` (pass through if already one)."""
    if isinstance(arr, wp.array):
        return arr
    return wp.array(
        np.ascontiguousarray(arr, dtype=np.float32),
        dtype=wp.float32,
        device=device,
    )
