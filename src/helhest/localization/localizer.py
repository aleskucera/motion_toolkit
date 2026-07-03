"""Scan-to-map localization: odom-predicted pose, ICP-refined, drift-gated.

Wraps the ICP aligner in the trajectory state machine the accumulating mapper
needs: predict the next pose from the odometry frame-to-frame delta, register
the scan against a reference cloud, and gate the correction so a diverging
alignment falls back to dead-reckoned odometry.

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
    submap_points: int = 0


class Localizer:
    """Odom-predicted, ICP-refined, drift-gated pose tracker (scan-to-map)."""

    def __init__(self, aligner: IcpAligner | None, config: LocalizerConfig) -> None:
        # aligner may be None only when config.enable is False (pure dead reckoning).
        self.aligner = aligner
        self.config = config
        self._world_T_base_prev: np.ndarray | None = None
        self._odom_T_base_prev: np.ndarray | None = None
        self._crop: BoxCrop | None = None  # lazy; sizes to the reference cloud on first use

    @property
    def initialized(self) -> bool:
        return self._world_T_base_prev is not None

    def bootstrap(self, odom_T_base: np.ndarray, world_T_base: np.ndarray) -> None:
        """Seed the trajectory: adopt world_T_base as the first corrected pose."""
        self._odom_T_base_prev = odom_T_base
        self._world_T_base_prev = world_T_base

    def predict(self, odom_T_base_curr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the new pose from the odom delta; return (world_T_base_pred, sweep_delta).

        The returned delta (base_prev→base_curr) doubles as the constant-velocity
        sweep motion for deskew, so the caller can motion-compensate the sweep
        before handing it to update().
        """
        if self._world_T_base_prev is None or self._odom_T_base_prev is None:
            raise RuntimeError("Localizer.predict called before bootstrap")
        sweep_delta = odom_delta(self._odom_T_base_prev, odom_T_base_curr)
        return self._world_T_base_prev @ sweep_delta, sweep_delta

    def update(
        self,
        scan_base: wp.array,
        world_T_base_pred: np.ndarray,
        reference_cloud: wp.array,
        odom_T_base_curr: np.ndarray,
    ) -> RegistrationOutcome:
        """Register the device scan against the device reference_cloud, commit, return.

        `scan_base` and `reference_cloud` are device `wp.array(vec3)`. The adopted
        pose (refined or odom fallback) becomes the previous corrected pose that
        seeds the next predict() — corrections compound forward.
        """
        outcome = self._register(scan_base, world_T_base_pred, reference_cloud)
        self._world_T_base_prev = outcome.pose
        self._odom_T_base_prev = odom_T_base_curr
        return outcome

    def _register(
        self,
        scan_base: wp.array,
        world_T_base_pred: np.ndarray,
        reference_cloud: wp.array,
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

        result = self.aligner.align(
            scan_base,
            submap[:n_submap],
            init_pose=world_T_base_pred,
        )
        rot, trans = pose_correction_magnitude(world_T_base_pred, result.pose)
        accepted = (
            result.converged
            and result.num_inliers >= cfg.min_inliers
            and trans <= cfg.max_correction_trans_m
            and rot <= cfg.max_correction_rot_rad
        )
        return RegistrationOutcome(
            pose=result.pose if accepted else world_T_base_pred,
            status="ok" if accepted else "rejected",
            num_inliers=int(result.num_inliers),
            converged=bool(result.converged),
            correction_rot_rad=rot,
            correction_trans_m=trans,
            submap_points=n_submap,
        )
