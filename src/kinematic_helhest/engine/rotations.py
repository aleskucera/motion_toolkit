"""Device rotation-matrix builders (Warp).

Elementary axis rotations and the body orientation R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
All `@wp.func`, so they inline into the kinematics kernels at no runtime cost. Mirrors
the numpy `model.euler_zyx`, which stays the finite-difference oracle.
"""

import warp as wp


@wp.func
def rot_z(a: wp.float32):
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.mat33(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


@wp.func
def rot_y(a: wp.float32):
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.mat33(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)


@wp.func
def rot_x(a: wp.float32):
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.mat33(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)


# --- elementwise derivatives d(rot_axis)/da, for the analytic settle Jacobian ---
@wp.func
def drot_z(a: wp.float32):
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.mat33(-s, -c, 0.0, c, -s, 0.0, 0.0, 0.0, 0.0)


@wp.func
def drot_y(a: wp.float32):
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.mat33(-s, 0.0, c, 0.0, 0.0, 0.0, -c, 0.0, -s)


@wp.func
def drot_x(a: wp.float32):
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.mat33(0.0, 0.0, 0.0, 0.0, -s, -c, 0.0, c, -s)


@wp.func
def euler_zyx(yaw: wp.float32, pitch: wp.float32, roll: wp.float32):
    """Body orientation R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return rot_z(yaw) * rot_y(pitch) * rot_x(roll)
