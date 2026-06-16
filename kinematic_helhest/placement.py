"""Quasi-static placement on a heightmap (Phase 2): wheels bilateral.

Given the planar pose (x, y, yaw), solve the derived DOF (z, pitch, roll) so all
three wheel hubs rest on the terrain, then recover the contact normal loads N_i
from gravity equilibrium.

This is the numpy reference. The FB-NCP / Warp version (Phase 3/5) generalizes
the wheel equalities to complementarity for the chassis and supplies implicit
gradients; this forward solve stays as the finite-difference oracle.

Orientation convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (Z-Y-X intrinsic).
Pitch nose-up is negative (rotation about +Y tilts +X toward -Z).
"""
import numpy as np

from .model import chassis_sample_points
from .model import COM
from .model import GRAVITY
from .model import MASS
from .model import WHEEL_POS
from .model import WHEEL_RADIUS

CHASSIS_PTS = chassis_sample_points()  # [Np, 3] body-frame bottom-face grid


def euler_zyx(yaw, pitch, roll):
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx


def R_to_quat(R):
    """3x3 rotation -> quaternion [qx, qy, qz, qw]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def _wheel_clearances(x, y, yaw, z, pitch, roll, hm, R_wheel=WHEEL_RADIUS):
    """Signed clearance c_i = hub_z - H_eff(hub_xy) - R for each wheel.

    `hm` is the wheel-envelope heightmap (heightmap.wheel_envelope), so the
    spherical-cap geometry is already baked in and no slope term is needed.
    """
    R = euler_zyx(yaw, pitch, roll)
    p = np.array([x, y, z])
    hubs = p[None, :] + WHEEL_POS @ R.T  # [3,3] world hub positions
    h = hm.sample(hubs[:, 0], hubs[:, 1])
    n = hm.normal(hubs[:, 0], hubs[:, 1])  # [3,3]  (envelope normal, for loads)
    c = hubs[:, 2] - h - R_wheel
    return c, hubs, n


def settle(x, y, yaw, hm, init=None, R_wheel=WHEEL_RADIUS, max_iter=20, tol=1e-7):
    """Solve (z, pitch, roll) so all three wheels rest on the terrain.

    Returns dict: z, pitch, roll, R (3x3), hubs [3,3], normals [3,3],
    contacts [3,3], converged (bool), residual.
    """
    if init is None:
        z0 = hm.sample(x, y) + R_wheel
        u = np.array([z0, 0.0, 0.0])  # (z, pitch, roll)
    else:
        u = np.array(init, dtype=np.float64)

    eps = 1e-6
    max_step = np.array([0.1, 0.2, 0.2])  # cap (z, pitch, roll) move per iter
    c = None
    for _ in range(max_iter):
        c, hubs, n = _wheel_clearances(x, y, yaw, u[0], u[1], u[2], hm, R_wheel)
        if np.max(np.abs(c)) < tol:
            break
        # Numerical 3x3 Jacobian dc/d(z,pitch,roll).
        J = np.empty((3, 3))
        for k in range(3):
            du = u.copy()
            du[k] += eps
            ck, _, _ = _wheel_clearances(x, y, yaw, du[0], du[1], du[2], hm, R_wheel)
            J[:, k] = (ck - c) / eps
        step = np.linalg.solve(J, c)
        step = np.clip(step, -max_step, max_step)  # damp to keep the solve stable
        u = u - step
        u[1] = np.clip(u[1], -1.2, 1.2)  # |pitch|, |roll| < ~69 deg
        u[2] = np.clip(u[2], -1.2, 1.2)

    z, pitch, roll = u
    R = euler_zyx(yaw, pitch, roll)
    c, hubs, n = _wheel_clearances(x, y, yaw, z, pitch, roll, hm, R_wheel)
    contacts = hubs - R_wheel * n  # contact points on the terrain
    return {
        "z": float(z), "pitch": float(pitch), "roll": float(roll),
        "R": R, "hubs": hubs, "normals": n, "contacts": contacts,
        "converged": bool(np.max(np.abs(c)) < 1e-4), "residual": float(np.max(np.abs(c))),
    }


def normal_loads(place, x, y):
    """Quasi-static contact normal loads N_i (along terrain normals) from gravity.

    Solves vertical-force + horizontal-torque balance about the CoM (the 3
    determined equations; tangential friction carries the rest). Returns N [3].
    """
    R = place["R"]
    contacts = place["contacts"]   # [3,3]
    n = place["normals"]           # [3,3]
    com_world = np.array([x, y, place["z"]]) + R @ COM
    r = contacts - com_world[None, :]  # lever arms [3,3]

    # A[0,i] = n_z ;  A[1,i] = (r x n)_x ;  A[2,i] = (r x n)_y
    rxn = np.cross(r, n)  # [3,3]
    A = np.stack([n[:, 2], rxn[:, 0], rxn[:, 1]], axis=0)  # [3,3]
    b = np.array([MASS * GRAVITY, 0.0, 0.0])
    return np.linalg.solve(A, b)


def chassis_clearance(R, x, y, z, hm):
    """Signed clearance of each chassis bottom-face point above the raw terrain.

    Uses the RAW heightmap (the box belly contacts the actual ground, not the
    wheel-inflated envelope). Returns (clearances [Np], world_points [Np, 3]).
    Negative clearance == high-centered (belly penetrates terrain).
    """
    p = np.array([x, y, z])
    world = p[None, :] + CHASSIS_PTS @ R.T
    h = hm.sample(world[:, 0], world[:, 1])
    return world[:, 2] - h, world


def place_pose7(place, x, y):
    """Full SE(3) pose7 (px,py,pz, qx,qy,qz,qw) from a settle result."""
    q = R_to_quat(place["R"])
    return np.array([x, y, place["z"], q[0], q[1], q[2], q[3]], dtype=np.float32)
