"""Scan-to-map localization: odom-predicted pose, ICP-refined, drift-gated.

Wraps the ICP aligner in the trajectory state machine the accumulating mapper
needs: predict the next pose from a frame-to-frame motion delta, register the
scan against a reference cloud, and gate the correction so a diverging alignment
falls back to dead reckoning. The prediction's rotation can come from the IMU
(slip-immune) with only its translation taken from wheel odom — see predict().

The localizer owns the pose state (previous corrected pose + previous odom) but
NOT the reference cloud — that is passed in per frame. So the same localizer
works whether it aligns against a map it shares with the terrain pipeline or a
dedicated localization map; swapping one for the other is a caller-side change.

Fully device-resident: the scan and reference cloud are device `wp.array(vec3)`
(upload the scan once at the sensor boundary); the submap crop runs on device and
ICP consumes them directly. Only the 4x4 poses and point counts touch the host.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from helhest.perception.cloud_ops import BoxCrop
from helhest.perception.icp import IcpAligner
from helhest.perception.icp import IcpResult
from .pose_math import odom_delta
from .pose_math import pose_correction_magnitude


@dataclass
class LocalizerConfig:
    """Submap extent + divergence-gate thresholds for scan-to-map registration."""

    enable: bool = True
    # Half-extent of the reference-cloud crop used as the ICP target.
    submap_radius_m: float = 15.0
    # Skip ICP (use the odom prediction) if the submap is sparser than this.
    min_submap_points: int = 2000
    # Reject the alignment below this inlier count.
    min_inliers: int = 500
    # Reject if ICP moves / rotates the prediction farther than these.
    max_correction_trans_m: float = 1.0
    max_correction_rot_rad: float = float(np.deg2rad(15.0))
    # Reject if the RMS point-to-plane residual exceeds this — the actual fitness of the
    # alignment. <= 0 disables it. Preferred over gating on the ICP `converged` flag: that
    # flag only means a GN step reached the (tight) iteration tolerance within max_iters, and
    # a perfectly good alignment routinely plateaus just above it and burns all iterations
    # without ever tripping it, so gating on it rejects good registrations.
    max_rms_residual_m: float = 0.0
    # Bypass the translation correction cap when the RMS is strictly below this value.
    # An aliased ICP result cannot produce sub-threshold RMS with many inliers, so the
    # cap only harms genuinely clean fits that happen to correct a large odom drift.
    # 0.0 disables the bypass (default — no change in behaviour vs. the plain trans cap).
    # Typical value: ~half of max_rms_residual_m (e.g. 0.04 m when the cap is 0.08 m).
    min_rms_to_bypass_trans_m: float = 0.0
    # Yaw multi-start: run this many ICPs from initial headings spread across
    # +-yaw_search_deg/2 about the predicted yaw, and keep the best-fitting (lowest RMS with
    # enough inliers). Escapes the wrong rotational basin a single init falls into under fast
    # skid-steer yaw. 1 (or 0) = single ICP from the prediction (no sweep).
    yaw_restarts: int = 1
    yaw_search_deg: float = 30.0


@dataclass
class RegistrationOutcome:
    """One registration's result: the adopted pose plus why it was (not) taken."""

    # world_T_base actually adopted: the refined pose, or the odom prediction on fallback.
    pose: np.ndarray
    status: str  # "ok" | "rejected" | "sparse" | "disabled"
    num_inliers: int = 0
    converged: bool = False
    correction_rot_rad: float = 0.0
    correction_trans_m: float = 0.0
    rms_residual_m: float = 0.0
    submap_points: int = 0


class Localizer:
    """Odom-predicted, ICP-refined, drift-gated pose tracker (scan-to-map)."""

    def __init__(self, aligner: IcpAligner | None, config: LocalizerConfig) -> None:
        # aligner may be None only when config.enable is False (pure dead reckoning).
        self.aligner = aligner
        self.config = config
        self._world_T_base_prev: np.ndarray | None = None
        self._odom_T_base_prev: np.ndarray | None = None
        self._imu_R_base_prev: np.ndarray | None = None  # world_R_base at the last frame (IMU)
        self._crop: BoxCrop | None = None  # lazy; sizes to the reference cloud on first use

    @property
    def initialized(self) -> bool:
        return self._world_T_base_prev is not None

    def bootstrap(
        self,
        odom_T_base: np.ndarray,
        world_T_base: np.ndarray,
        imu_R_base: np.ndarray | None = None,
    ) -> None:
        """Seed the trajectory: adopt world_T_base as the first corrected pose."""
        self._odom_T_base_prev = odom_T_base
        self._world_T_base_prev = world_T_base
        self._imu_R_base_prev = imu_R_base

    def predict(
        self,
        odom_T_base_curr: np.ndarray,
        imu_R_base_curr: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the new pose from the motion delta; return (world_T_base_pred, sweep_delta).

        The delta's TRANSLATION always comes from odom. Its ROTATION comes from the IMU
        orientation delta (`imu_R_base_curr`, world_R_base this frame) when available —
        wheel odom yaw is unreliable under skid (in-place rotation), while the IMU can't
        give position; each sensor supplies the DOF it is trustworthy on. With no IMU
        rotation the delta is the pure odom delta.

        The returned delta (base_prev→base_curr) doubles as the constant-velocity sweep
        motion for deskew, so the caller can motion-compensate the sweep before update().
        """
        if self._world_T_base_prev is None or self._odom_T_base_prev is None:
            raise RuntimeError("Localizer.predict called before bootstrap")
        sweep_delta = odom_delta(self._odom_T_base_prev, odom_T_base_curr)
        if imu_R_base_curr is not None and self._imu_R_base_prev is not None:
            sweep_delta = sweep_delta.copy()  # keep odom translation, swap in the IMU rotation
            sweep_delta[:3, :3] = self._imu_R_base_prev.T @ imu_R_base_curr
        return self._world_T_base_prev @ sweep_delta, sweep_delta

    def update(
        self,
        scan_base: wp.array,
        world_T_base_pred: np.ndarray,
        reference_cloud: wp.array,
        odom_T_base_curr: np.ndarray,
        *,
        imu_R_base_curr: np.ndarray | None = None,
        gravity_up: np.ndarray | None = None,
    ) -> RegistrationOutcome:
        """Register the device scan against the device reference_cloud, commit, return.

        `scan_base` and `reference_cloud` are device `wp.array(vec3)`. The adopted
        pose (refined or odom fallback) becomes the previous corrected pose that
        seeds the next predict() — corrections compound forward. `odom_T_base_curr`
        and `imu_R_base_curr` (world_R_base from the IMU, or None) are stored as the
        previous frame's references the next predict() differences against.

        `gravity_up` (3,) is the measured up-direction in the base frame at this scan
        (e.g. the IMU gravity vector); with the aligner's `gravity_weight > 0` it
        anchors the ICP roll/pitch to gravity. None disables it.
        """
        outcome = self._register(scan_base, world_T_base_pred, reference_cloud, gravity_up)
        self._world_T_base_prev = outcome.pose
        self._odom_T_base_prev = odom_T_base_curr
        self._imu_R_base_prev = imu_R_base_curr
        return outcome

    def set_corrected_pose(self, world_T_base: np.ndarray) -> None:
        """Override the corrected pose that seeds the next predict().
        Call after an external filter (e.g. EKF) has refined the pose so the
        Kalman-fused x/y/yaw propagates into the next frame's ICP seed rather
        than the raw ICP result. Only _world_T_base_prev is touched; the stored
        odom and IMU references are unchanged (they drive sweep_delta, not the seed).
        """
        self._world_T_base_prev = world_T_base

    def _register(
        self,
        scan_base: wp.array,
        world_T_base_pred: np.ndarray,
        reference_cloud: wp.array,
        gravity_up: np.ndarray | None = None,
    ) -> RegistrationOutcome:
        cfg = self.config
        if not cfg.enable:
            return RegistrationOutcome(world_T_base_pred, "disabled")

        # Crop the ICP target submap out of the reference cloud, on device (square xy
        # box of half-extent submap_radius_m around the predicted robot xy).
        n_ref = len(reference_cloud)
        if self._crop is None or self._crop.max_points < n_ref:
            self._crop = BoxCrop(max(n_ref, 400_000), device=reference_cloud.device)
        cx, cy = float(world_T_base_pred[0, 3]), float(world_T_base_pred[1, 3])
        submap, n_submap = self._crop.crop(reference_cloud, n_ref, (cx, cy), cfg.submap_radius_m)
        if n_submap < cfg.min_submap_points:
            return RegistrationOutcome(world_T_base_pred, "sparse", submap_points=n_submap)

        if cfg.yaw_restarts > 1:
            result = self._align_yaw_sweep(scan_base, submap[:n_submap], world_T_base_pred, gravity_up)
        else:
            result = self.aligner.align(
                scan_base,
                submap[:n_submap],
                init_pose=world_T_base_pred,
                gravity_up=gravity_up,
            )
        rot, trans = pose_correction_magnitude(world_T_base_pred, result.pose)
        # RMS point-to-plane residual over the inliers: sqrt(mean weighted r²). This is the
        # fitness gate — see max_rms_residual_m for why we do NOT gate on result.converged.
        rms = (
            float(np.sqrt(result.final_cost / result.num_inliers))
            if result.num_inliers > 0
            else float("inf")
        )
        # Translation cap: normal rail OR bypassed when the fit is exceptionally clean.
        # An aliased basin cannot produce rms < min_rms_to_bypass_trans_m with many inliers,
        # so a sub-threshold RMS is evidence the correction is genuine, not a hallucination.
        trans_ok = trans <= cfg.max_correction_trans_m or (
            cfg.min_rms_to_bypass_trans_m > 0.0 and rms < cfg.min_rms_to_bypass_trans_m
        )
        accepted = (
            result.num_inliers >= cfg.min_inliers
            and trans_ok
            and rot <= cfg.max_correction_rot_rad
            and (cfg.max_rms_residual_m <= 0.0 or rms <= cfg.max_rms_residual_m)
        )
        return RegistrationOutcome(
            pose=result.pose if accepted else world_T_base_pred,
            status="ok" if accepted else "rejected",
            num_inliers=int(result.num_inliers),
            converged=bool(result.converged),
            correction_rot_rad=rot,
            correction_trans_m=trans,
            rms_residual_m=rms,
            submap_points=n_submap,
        )

    def _align_yaw_sweep(self, scan, submap, pred, gravity_up):
        """Run `yaw_restarts` ICPs from headings spanning ±yaw_search_deg/2 about the predicted
        yaw; return the best-fitting result (lowest RMS with enough inliers).

        Fast skid-steer yaw drops a single ICP into the wrong rotational basin; seeding several
        headings and keeping the best fit escapes it. The sweep includes the prediction itself
        (offset 0), so this never does worse than the single-start alignment.
        """
        cfg = self.config
        half = float(np.deg2rad(cfg.yaw_search_deg)) * 0.5
        # Build the H yaw-perturbed init poses and align them all in ONE batched GN pass.
        inits = np.empty((cfg.yaw_restarts, 4, 4), dtype=np.float64)
        for k, dyaw in enumerate(np.linspace(-half, half, cfg.yaw_restarts)):
            c, s = float(np.cos(dyaw)), float(np.sin(dyaw))
            rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            inits[k] = pred
            inits[k, :3, :3] = rz @ pred[:3, :3]  # perturb heading about world-up, keep position
        poses, costs, inliers = self.aligner.align_batch(scan, submap, inits, gravity_up=gravity_up)
        # Pick the best: lowest RMS = sqrt(cost / inliers) among those clearing the inlier gate.
        rms = np.where(inliers > 0, np.sqrt(costs / np.maximum(inliers, 1)), np.inf)
        rms[inliers < cfg.min_inliers] = np.inf
        b = int(np.argmin(rms)) if np.isfinite(rms).any() else int(np.argmax(inliers))
        return IcpResult(
            pose=poses[b],
            iterations=self.aligner.config.max_iters,
            final_cost=float(costs[b]),
            num_inliers=int(inliers[b]),
            converged=True,
        )
