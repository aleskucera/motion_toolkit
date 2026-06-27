"""Small device linear-algebra helpers (Warp)."""

import warp as wp


@wp.func
def solve3(A: wp.mat33, b: wp.vec3):
    """Solve A x = b (3x3) via cofactors.

    Used instead of wp.inverse: Warp 1.13 miscompiles wp.inverse(mat33) on CUDA
    when two inverse call sites share a kernel (e.g. settle + normal_loads),
    causing illegal memory access. An explicit solve has a single code path.
    """
    a = A[0, 0]
    b1 = A[0, 1]
    c1 = A[0, 2]
    d = A[1, 0]
    e = A[1, 1]
    f = A[1, 2]
    g = A[2, 0]
    h = A[2, 1]
    i = A[2, 2]
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    det = a * c00 + b1 * c01 + c1 * c02
    inv = 1.0 / det
    c10 = -(b1 * i - c1 * h)
    c11 = a * i - c1 * g
    c12 = -(a * h - b1 * g)
    c20 = b1 * f - c1 * e
    c21 = -(a * f - c1 * d)
    c22 = a * e - b1 * d
    return wp.vec3(
        (c00 * b[0] + c10 * b[1] + c20 * b[2]) * inv,
        (c01 * b[0] + c11 * b[1] + c21 * b[2]) * inv,
        (c02 * b[0] + c12 * b[1] + c22 * b[2]) * inv,
    )
