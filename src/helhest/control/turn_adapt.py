"""Optional online turn-boost from yaw feedback: self-tune the commanded turn differential so the
REALIZED yaw matches what MPPI planned -- across terrains and the drivetrain differential defect.

The vehicle turn model is  wz = R*(wR - wL) / (2b*alpha),  alpha = 1 + k_turn*mu  (engine/step.py).
MPPI plans assuming that model; the real robot may realize a different yaw for the same command
(grippier terrain -> understeer -> higher alpha; the two drive motors equalize a commanded
differential -> under-turn). Rather than re-identify the model online, close a slow loop: from the
ACTUALLY-COMMANDED differential and the MEASURED yaw rate, estimate the boost that would make the
realized yaw equal the model's prediction, and low-pass it:

    tb_target = model_yaw(diff_cmd) / yaw_meas = [R*diff_cmd / (2b*alpha_model)] / yaw_meas

Note diff_cmd cancels in the ideal (both numerator and yaw_meas scale with it), so this is a direct
estimate of model_gain / real_gain -- exactly the multiplier condition_command needs. It is a smarter,
self-tuning generalization of the fixed `plan_turn_boost` (docs/turn_differential_hotfix.md): it
adapts to whatever terrain the robot is on instead of a hand-picked indoor/outdoor constant.

Guardrails: only updates while genuinely turning (the gain is unobservable on straights, and a small
yaw_meas would blow up the ratio); ignores steps where command and measurement disagree in sign
(noise / transients); clamps to a safe band; EMA time constant of seconds so it cannot fight the MPPI
replanning loop.
"""
from __future__ import annotations


class AdaptiveTurnBoost:
    """Slow yaw-feedback estimate of the turn_boost that makes realized yaw match the plan."""

    def __init__(
        self,
        *,
        alpha_model: float,  # planner's turn resistance 1 + k_turn*mu (the model the plan assumes)
        wheel_radius: float,
        half_track: float,
        dt: float,
        tau_s: float = 3.0,  # EMA time constant [s] -- slow, so it can't fight the replanning loop
        init: float = 1.0,  # starting boost (1.0 = no compensation)
        clamp: tuple[float, float] = (1.0, 3.0),  # safe band; a bad gyro moment can't run it away
        min_diff: float = 1.0,  # only adapt when |wR - wL| exceeds this [rad/s] (observable turning)
        min_yaw: float = 0.05,  # ...and |yaw_meas| exceeds this [rad/s] (avoid divide-by-noise)
    ):
        self._c = wheel_radius / (2.0 * half_track * alpha_model)  # model yaw per unit differential
        self._beta = min(1.0, dt / max(tau_s, 1e-6))  # EMA blend per update
        self._lo, self._hi = clamp
        self._min_diff = min_diff
        self._min_yaw = min_yaw
        self.turn_boost = float(min(max(init, self._lo), self._hi))

    def update(self, diff_cmd: float, yaw_meas: float) -> float:
        """diff_cmd: the differential actually commanded (wR - wL) [rad/s]; yaw_meas: measured yaw
        rate (gyro) [rad/s]. Returns the current turn_boost; only moves it when there is turning
        signal, so on straights it holds the last learned value."""
        model_yaw = self._c * diff_cmd
        if abs(diff_cmd) < self._min_diff or abs(yaw_meas) < self._min_yaw:
            return self.turn_boost
        if model_yaw * yaw_meas <= 0.0:  # command vs measurement disagree in sign -> skip (noise)
            return self.turn_boost
        tb_target = model_yaw / yaw_meas  # >1 when under-realizing, <1 when over-realizing
        tb_target = min(max(tb_target, self._lo), self._hi)
        self.turn_boost += self._beta * (tb_target - self.turn_boost)
        return self.turn_boost
