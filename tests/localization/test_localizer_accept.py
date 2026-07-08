"""Localizer acceptance gate: a good-but-not-`converged` registration must be adopted.

Regression for the reject→map-reset sawtooth: ICP routinely exhausts max_iters without
tripping the tight convergence tolerance, so gating acceptance on `converged` threw away
good alignments. Acceptance now rests on inliers + correction bounds + (optional) RMS fit.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.localization import Localizer
from helhest.localization import LocalizerConfig
from helhest.perception.icp.aligner import IcpResult

wp.init()


class _StubAligner:
    """Stands in for IcpAligner: returns a canned IcpResult, ignoring the clouds."""

    def __init__(self, result: IcpResult) -> None:
        self._result = result

    def align(self, source, target, init_pose=None, *, gravity_up=None, profile=False) -> IcpResult:
        return self._result


def _run(result: IcpResult, cfg: LocalizerConfig) -> str:
    """Register one frame with a stub aligner; return the outcome status."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    with wp.ScopedDevice(device):
        # A reference cloud dense enough to clear min_submap_points, centered on the robot.
        ref_np = (np.random.default_rng(0).random((200, 3)).astype(np.float32) - 0.5) * 2.0
        reference = wp.array(ref_np, dtype=wp.vec3)
        scan = wp.array(ref_np[:20], dtype=wp.vec3)
        loc = Localizer(_StubAligner(result), cfg)
        loc.bootstrap(np.eye(4), np.eye(4))
        pred, _ = loc.predict(np.eye(4))
        return loc.update(scan, pred, reference, np.eye(4)).status


def _result(*, converged: bool, num_inliers: int, final_cost: float, pose=None) -> IcpResult:
    return IcpResult(
        pose=np.eye(4) if pose is None else pose,
        iterations=30,
        final_cost=final_cost,
        num_inliers=num_inliers,
        converged=converged,
    )


def test_unconverged_good_fit_is_accepted() -> None:
    # Hit the iteration cap (converged=False) but plenty of inliers, zero correction, tight fit.
    cfg = LocalizerConfig(min_submap_points=10, min_inliers=100)
    status = _run(_result(converged=False, num_inliers=500, final_cost=0.5), cfg)
    assert status == "ok"  # previously "rejected" solely because converged was False


def test_too_few_inliers_is_rejected() -> None:
    cfg = LocalizerConfig(min_submap_points=10, min_inliers=100)
    status = _run(_result(converged=True, num_inliers=50, final_cost=0.1), cfg)
    assert status == "rejected"


def test_rms_residual_gate_rejects_poor_fit_when_enabled() -> None:
    # num_inliers=500, final_cost=5.0 -> rms = sqrt(5/500) = 0.1 m, above the 0.05 m gate.
    cfg = LocalizerConfig(min_submap_points=10, min_inliers=100, max_rms_residual_m=0.05)
    status = _run(_result(converged=False, num_inliers=500, final_cost=5.0), cfg)
    assert status == "rejected"


def test_rms_gate_disabled_by_default_admits_the_same_fit() -> None:
    # Same poor fit, but with the RMS gate off (default 0.0) it is admitted.
    cfg = LocalizerConfig(min_submap_points=10, min_inliers=100)
    status = _run(_result(converged=False, num_inliers=500, final_cost=5.0), cfg)
    assert status == "ok"


def _yaw(deg: float, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[0, 3], T[1, 3] = tx, ty
    return T


def test_motion_prior_takes_imu_rotation_and_odom_translation() -> None:
    # Skid case: odom says 5° yaw + 1 m forward, but the IMU says the body rotated 30°.
    loc = Localizer(None, LocalizerConfig())
    loc.bootstrap(np.eye(4), np.eye(4), _yaw(0.0)[:3, :3])
    _, sweep = loc.predict(_yaw(5.0, tx=1.0), _yaw(30.0)[:3, :3])
    assert np.allclose(sweep[:3, :3], _yaw(30.0)[:3, :3], atol=1e-9)  # rotation from IMU
    assert np.allclose(sweep[:3, 3], [1.0, 0.0, 0.0], atol=1e-9)  # translation from odom


def test_motion_prior_falls_back_to_odom_without_imu() -> None:
    loc = Localizer(None, LocalizerConfig())
    loc.bootstrap(np.eye(4), np.eye(4), _yaw(0.0)[:3, :3])
    _, sweep = loc.predict(_yaw(5.0, tx=1.0), None)  # no IMU rotation this frame
    assert np.allclose(sweep, _yaw(5.0, tx=1.0), atol=1e-9)  # pure odom delta
