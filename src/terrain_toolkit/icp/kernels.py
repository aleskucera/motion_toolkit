from __future__ import annotations

import warp as wp

wp.init()

# Fixed-size types for the device-side 6x6 Gauss-Newton solve.
mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def _accumulate_row(
    JtJ: wp.array(dtype=wp.float32, ndim=2),
    Jtr: wp.array(dtype=wp.float32),
    j0: wp.float32,
    j1: wp.float32,
    j2: wp.float32,
    j3: wp.float32,
    j4: wp.float32,
    j5: wp.float32,
    r: wp.float32,
    w: wp.float32,
):
    """Atomically add w·J^T J (6x6) and w·J^T r (6x1) into accumulators."""
    wp.atomic_add(Jtr, 0, w * j0 * r)
    wp.atomic_add(Jtr, 1, w * j1 * r)
    wp.atomic_add(Jtr, 2, w * j2 * r)
    wp.atomic_add(Jtr, 3, w * j3 * r)
    wp.atomic_add(Jtr, 4, w * j4 * r)
    wp.atomic_add(Jtr, 5, w * j5 * r)

    wp.atomic_add(JtJ, 0, 0, w * j0 * j0)
    wp.atomic_add(JtJ, 0, 1, w * j0 * j1)
    wp.atomic_add(JtJ, 0, 2, w * j0 * j2)
    wp.atomic_add(JtJ, 0, 3, w * j0 * j3)
    wp.atomic_add(JtJ, 0, 4, w * j0 * j4)
    wp.atomic_add(JtJ, 0, 5, w * j0 * j5)
    wp.atomic_add(JtJ, 1, 1, w * j1 * j1)
    wp.atomic_add(JtJ, 1, 2, w * j1 * j2)
    wp.atomic_add(JtJ, 1, 3, w * j1 * j3)
    wp.atomic_add(JtJ, 1, 4, w * j1 * j4)
    wp.atomic_add(JtJ, 1, 5, w * j1 * j5)
    wp.atomic_add(JtJ, 2, 2, w * j2 * j2)
    wp.atomic_add(JtJ, 2, 3, w * j2 * j3)
    wp.atomic_add(JtJ, 2, 4, w * j2 * j4)
    wp.atomic_add(JtJ, 2, 5, w * j2 * j5)
    wp.atomic_add(JtJ, 3, 3, w * j3 * j3)
    wp.atomic_add(JtJ, 3, 4, w * j3 * j4)
    wp.atomic_add(JtJ, 3, 5, w * j3 * j5)
    wp.atomic_add(JtJ, 4, 4, w * j4 * j4)
    wp.atomic_add(JtJ, 4, 5, w * j4 * j5)
    wp.atomic_add(JtJ, 5, 5, w * j5 * j5)


@wp.func
def _power_iterate(C: wp.mat33, v0: wp.vec3, iters: int) -> wp.vec3:
    """Largest-eigenvector power iteration on a 3x3 symmetric matrix."""
    v = wp.normalize(v0)
    for _ in range(iters):
        v = C @ v
        n = wp.length(v)
        if n > 1.0e-20:
            v = v / n
    return v


@wp.kernel
def estimate_normals_kernel(
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    radius: wp.float32,
    min_neighbors: wp.int32,
    power_iters: wp.int32,
    normals: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
):
    """Per-point PCA normal via power iteration on covariance."""
    i = wp.tid()
    p = points[i]

    # Gather neighbor statistics.
    mean = wp.vec3(0.0, 0.0, 0.0)
    count = int(0)
    neighbors = wp.hash_grid_query(grid, p, radius)
    for index in neighbors:
        q = points[index]
        d = wp.length(q - p)
        if d <= radius:
            mean = mean + q
            count += 1

    if count < min_neighbors:
        normals[i] = wp.vec3(0.0, 0.0, 0.0)
        valid[i] = 0
        return

    mean = mean / float(count)

    # Covariance.
    c00 = float(0.0)
    c01 = float(0.0)
    c02 = float(0.0)
    c11 = float(0.0)
    c12 = float(0.0)
    c22 = float(0.0)
    neighbors2 = wp.hash_grid_query(grid, p, radius)
    for index in neighbors2:
        q = points[index]
        d = wp.length(q - p)
        if d <= radius:
            dq = q - mean
            c00 += dq[0] * dq[0]
            c01 += dq[0] * dq[1]
            c02 += dq[0] * dq[2]
            c11 += dq[1] * dq[1]
            c12 += dq[1] * dq[2]
            c22 += dq[2] * dq[2]

    inv_n = 1.0 / float(count)
    C = wp.mat33(
        c00 * inv_n,
        c01 * inv_n,
        c02 * inv_n,
        c01 * inv_n,
        c11 * inv_n,
        c12 * inv_n,
        c02 * inv_n,
        c12 * inv_n,
        c22 * inv_n,
    )

    # Largest eigenvector of C.
    v1 = _power_iterate(C, wp.vec3(1.0, 0.0, 0.0), power_iters)
    l1 = wp.dot(v1, C @ v1)

    # Deflate and find second-largest.
    v1v1 = wp.outer(v1, v1)
    D = C - l1 * v1v1
    # Seed orthogonal to v1 for the second iteration.
    seed = wp.vec3(0.0, 1.0, 0.0)
    if wp.abs(v1[1]) > 0.9:
        seed = wp.vec3(1.0, 0.0, 0.0)
    v2 = _power_iterate(D, seed, power_iters)

    # Smallest eigenvector is orthogonal to the top two.
    n = wp.normalize(wp.cross(v1, v2))

    normals[i] = n
    valid[i] = 1


@wp.kernel
def transform_points_kernel(
    src: wp.array(dtype=wp.vec3),
    pose: wp.array(dtype=wp.mat44),  # device-resident current pose (target_T_source)
    n_src: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= n_src[0]:
        return
    m = pose[0]
    p = src[i]
    out[i] = wp.vec3(
        m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2] * p[2] + m[0, 3],
        m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2] * p[2] + m[1, 3],
        m[2, 0] * p[0] + m[2, 1] * p[1] + m[2, 2] * p[2] + m[2, 3],
    )


@wp.kernel
def accumulate_system_kernel(
    grid: wp.uint64,
    target: wp.array(dtype=wp.vec3),
    target_normals: wp.array(dtype=wp.vec3),
    target_valid: wp.array(dtype=wp.int32),
    transformed_src: wp.array(dtype=wp.vec3),
    n_src: wp.array(dtype=wp.int32),
    max_dist: wp.float32,
    huber_delta: wp.float32,
    JtJ: wp.array(dtype=wp.float32, ndim=2),
    Jtr: wp.array(dtype=wp.float32),
    cost: wp.array(dtype=wp.float32),
    num_inliers: wp.array(dtype=wp.int32),
):
    """For each source point: find nearest target, build point-to-plane Jacobian row."""
    i = wp.tid()
    if i >= n_src[0]:
        return
    p = transformed_src[i]

    best = max_dist * max_dist
    best_idx = int(-1)
    neighbors = wp.hash_grid_query(grid, p, max_dist)
    for index in neighbors:
        if target_valid[index] == 0:
            continue
        diff = target[index] - p
        d2 = wp.dot(diff, diff)
        if d2 < best:
            best = d2
            best_idx = index

    if best_idx < 0:
        return

    n = target_normals[best_idx]
    q = target[best_idx]
    r = wp.dot(p - q, n)

    # Huber weight.
    ar = wp.abs(r)
    w = float(1.0)
    if ar > huber_delta:
        w = huber_delta / ar

    # Jacobian row J = [p × n, n] (6-vector: rotation then translation).
    j0 = p[1] * n[2] - p[2] * n[1]
    j1 = p[2] * n[0] - p[0] * n[2]
    j2 = p[0] * n[1] - p[1] * n[0]
    j3 = n[0]
    j4 = n[1]
    j5 = n[2]

    _accumulate_row(JtJ, Jtr, j0, j1, j2, j3, j4, j5, r, w)
    wp.atomic_add(cost, 0, w * r * r)
    wp.atomic_add(num_inliers, 0, 1)


@wp.kernel
def solve6x6_kernel(
    JtJ: wp.array(dtype=wp.float32, ndim=2),  # upper triangle populated
    Jtr: wp.array(dtype=wp.float32),
    damping: wp.float32,
    delta: wp.array(dtype=wp.float32),
):
    """delta = -(H + damping·I)^-1 · Jtr, H symmetrized from JtJ, via float32 Cholesky (dim=1)."""
    h = mat66()
    for i in range(6):
        for j in range(6):
            if i <= j:
                h[i, j] = JtJ[i, j]
            else:
                h[i, j] = JtJ[j, i]
    for d in range(6):
        h[d, d] = h[d, d] + damping

    # Cholesky H = L Lᵀ.
    ll = mat66()
    for j in range(6):
        s = h[j, j]
        for k in range(j):
            s = s - ll[j, k] * ll[j, k]
        ljj = wp.sqrt(wp.max(s, 1.0e-12))  # damping keeps H SPD; clamp guards fp noise
        ll[j, j] = ljj
        for i in range(j + 1, 6):
            s2 = h[i, j]
            for k in range(j):
                s2 = s2 - ll[i, k] * ll[j, k]
            ll[i, j] = s2 / ljj

    # Forward solve L y = -Jtr, then back solve Lᵀ x = y.
    y = vec6()
    for i in range(6):
        s = -Jtr[i]
        for k in range(i):
            s = s - ll[i, k] * y[k]
        y[i] = s / ll[i, i]
    x = vec6()
    for ii in range(6):
        i = 5 - ii
        s = y[i]
        for k in range(i + 1, 6):
            s = s - ll[k, i] * x[k]
        x[i] = s / ll[i, i]

    for i in range(6):
        delta[i] = x[i]


@wp.func
def _skew(w: wp.vec3) -> wp.mat33:
    return wp.mat33(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0)


@wp.kernel
def se3_update_kernel(
    delta: wp.array(dtype=wp.float32),
    pose: wp.array(dtype=wp.mat44),  # pose <- exp(delta) @ pose (in place)
    dr: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
):
    """Left-multiply the pose by the SE(3) exponential of the GN step (dim=1)."""
    omega = wp.vec3(delta[0], delta[1], delta[2])
    v = wp.vec3(delta[3], delta[4], delta[5])
    theta = wp.length(omega)
    ident = wp.identity(n=3, dtype=wp.float32)
    w_mat = _skew(omega)
    if theta < 1.0e-8:
        rot = ident + w_mat
        v_mat = ident + 0.5 * w_mat
    else:
        w2 = w_mat @ w_mat
        a = wp.sin(theta) / theta
        b = (1.0 - wp.cos(theta)) / (theta * theta)
        c = (theta - wp.sin(theta)) / (theta * theta * theta)
        rot = ident + a * w_mat + b * w2
        v_mat = ident + b * w_mat + c * w2
    tvec = v_mat @ v
    step = wp.mat44(
        rot[0, 0], rot[0, 1], rot[0, 2], tvec[0],
        rot[1, 0], rot[1, 1], rot[1, 2], tvec[1],
        rot[2, 0], rot[2, 1], rot[2, 2], tvec[2],
        0.0, 0.0, 0.0, 1.0,
    )
    pose[0] = step @ pose[0]
    dr[0] = theta
    dt[0] = wp.length(v)


@wp.kernel
def keep_going_kernel(
    dr: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    inliers: wp.array(dtype=wp.int32),
    iter_count: wp.array(dtype=wp.int32),
    cap: wp.int32,
    conv_rot: wp.float32,
    conv_trans: wp.float32,
    keep_running: wp.array(dtype=wp.int32),
    converged: wp.array(dtype=wp.int32),
):
    """Device loop condition (dim=1): stop on convergence, no inliers, or the cap."""
    it = iter_count[0] + 1
    iter_count[0] = it
    stop = int(0)
    if dr[0] < conv_rot and dt[0] < conv_trans:
        converged[0] = 1
        stop = 1
    if inliers[0] == 0:
        stop = 1
    if it >= cap:
        stop = 1
    if stop == 1:
        keep_running[0] = 0
    else:
        keep_running[0] = 1
